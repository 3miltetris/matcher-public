"""
Funding Strategy sessions (Stage 12)
------------------------------------
The consultants' roadmap workflow, moved inside the matcher.

Until now a consultant took the Aspect Match / Grant Search CSV out of this
app, opened a Claude.ai Project holding one of two playbooks, pasted or
uploaded the client's documents, and worked through a long approval-gated
conversation: opportunity-by-opportunity approve/revise/reject, then a
portfolio, then the "why now" claims, then a deck. The artefacts that fall out
of that live in project-context/ as markdown.

This module is the Streamlit-free half of doing the same thing here: the
playbook store, the context assembly, the Claude call, and the session record.
views/strategy_chat.py owns the widgets.

Three things in here are easy to get wrong and are deliberate:

  * **Playbooks live in GCS, not the repo.** The NIH knowledge base is
    refreshed on a cadence (see its own 00_README_and_Refresh_Log.md), so
    baking it into the image would mean a redeploy per refresh. It is also
    BW&CO's own material, and `Dockerfile.app` ends in `COPY . .` — anything
    committed here is published into Artifact Registry. Same reasoning that
    keeps the SAM.gov key out of the image.

  * **The cache layout is load-bearing.** The system block (playbook + skills
    + knowledge base) runs to ~50-80k tokens and is byte-identical on every
    turn of a session; the context block (match results + capability profile)
    is identical too. Both carry a `cache_control` breakpoint, and a third
    rolling breakpoint sits on the newest message. Without them every turn
    re-bills the whole prefix at full rate — the same arithmetic that took
    deep-research-job from $0.97 to $0.59 a site. `turn_cost()` bills cache
    reads at 0.10x and writes at 1.25x, so the saving is visible rather than
    hidden.

  * **A document attached mid-session becomes a user message, never a change
    to the system or context blocks.** Editing either would invalidate the
    cached prefix for the rest of the session. Mid-conversation `role:'system'`
    messages would be the tidier channel but are not supported on the Sonnet
    models this view defaults to, so a plain user turn it is.

`thinking` is never passed. Omitting it gives each model its own default (off
on Sonnet 4.6, adaptive on Sonnet 5, Opus 5 and Opus 5.5), which is what we
want, and it sidesteps two rules at once: `thinking: disabled` is a 400 above
`high` effort on Opus 5, and a 400 at EVERY effort level on Opus 5.5. Depth is
controlled through `output_config.effort` instead — see `effort_config()`.
`temperature`/`top_p`/`top_k` are never passed either — see the SDK
version-pin note in CLAUDE.md.

Streamlit-free so the same assembly can move into a Cloud Run job later if
these sessions ever need to run unattended; today only the view imports it.
"""

import json
import re
from datetime import datetime

import pandas as pd

# ── Stores ─────────────────────────────────────────────────────────────────

BUCKET           = 'cc-matcher-bucket-jeg-v1'
PLAYBOOK_PREFIX  = 'strategy-playbooks/'
MANIFEST_BLOB    = f'{PLAYBOOK_PREFIX}manifest.json'
SESSION_PREFIX   = 'strategy-sessions/'

# ── Models ─────────────────────────────────────────────────────────────────

# Opus 5.5 leads the list because the view's picker defaults to `index=0`. It
# is both stronger and CHEAPER than Opus 5 ($4/$20 per MTok against $5/$25), so
# there is no cost argument for keeping a Sonnet first any more.
#
# One caveat that is specific to this app's material: Opus 5.5 widened its
# safety classifiers — `bio` and `reasoning_extraction` now join `cyber`. The
# refusals documented below are BIO material (dried/powdered biologics), so on
# an Inaedis-class client this model is if anything likelier to decline than
# the Sonnets were. That is survivable precisely because FALLBACK_MODEL exists
# and Haiku is measured to answer; a consultant working a bio client can also
# pick Haiku outright from the mid-session switcher.
MODELS = ['claude-opus-5-5', 'claude-sonnet-4-6', 'claude-sonnet-5',
          'claude-opus-5', 'claude-haiku-4-5-20251001']

MODEL_LABELS = {
    'claude-opus-5-5':   'Opus 5.5 — strongest, and cheaper than Opus 5',
    'claude-sonnet-4-6': 'Sonnet 4.6 — former house default',
    'claude-sonnet-5':   'Sonnet 5 — stronger reasoning',
    'claude-opus-5':     'Opus 5 — superseded by 5.5, kept for comparison',
    # Haiku is selectable, not just the automatic fallback, because on some
    # real client material it is the ONLY model that completes the turn.
    # Measured on the live Inaedis session (NIH playbook, 85,614 prompt
    # tokens): Sonnet 5 refused mid-stream after 807 chars, Opus 5 after 401,
    # while Haiku finished normally with 18,803 chars. A consultant whose
    # client trips the classifier needs to be able to pick it deliberately
    # rather than discovering the fallback by accident.
    'claude-haiku-4-5-20251001':
        'Haiku 4.5 — weakest, but completes material the others decline',
}

# Models that accept the dynamic-filtering web-search tool. Deliberately NOT
# `MODELS`: Haiku is in the picker but rejects `_20260209`.
_DYNAMIC_SEARCH_MODELS = {
    'claude-opus-5-5', 'claude-sonnet-4-6', 'claude-sonnet-5', 'claude-opus-5',
}

# A refusal fallback, for the same reason client_profile_job carries one.
#
# Measured 2026-09-28 while building this view: a results table whose
# grant_summary read "Thermostable biologic drying." or "Spray-dried vaccine
# powder for cold-chain-free delivery." returns stop_reason='refusal' with
# ZERO content blocks on claude-sonnet-4-6, claude-sonnet-5 AND claude-opus-5
# alike — with a bare "You are terse." system prompt, so it is the client
# material, not the playbook. The same rows on an unrelated technology
# ("Drone path planning in GPS-denied terrain") answer normally, as does
# "Lipid nanoparticle mRNA delivery", so the trigger is narrow: dried or
# powdered biologics, which reads as aerosolisable.
#
# That is a live BW&CO client profile, not an edge case — CLAUDE.md already
# records Inaedis (aerosolised thermostable vaccine powders, DoD CBD SBIR)
# being unbuildable for two runs for exactly this reason, and Haiku building
# it normally. Haiku answers both strings here too.
#
# Because a refusal is deterministic for the same model and material, retrying
# the same model can only burn a second call — so the retry goes to a
# different one. Only a PRE-OUTPUT refusal is retried; see stream_turn for
# why a mid-stream refusal must not be. Switching model invalidates the prompt cache (caches are
# model-scoped), so the fallback turn pays a full cache write; that is the
# right trade against handing the consultant an empty answer. The session
# model is NOT changed, so the next turn tries the good model again and the
# cache stays warm for everything that does not trip the classifier.
FALLBACK_MODEL = 'claude-haiku-4-5-20251001'

# $ per 1M tokens (input, output). Cache reads bill at 0.10x input, writes at
# 1.25x — billing them at the full input rate would hide the entire saving the
# cache breakpoints exist to produce.
_PRICES = {
    'claude-opus-5-5':   (4.00, 20.00),
    'claude-sonnet-4-6': (3.00, 15.00),
    'claude-sonnet-5':   (3.00, 15.00),
    'claude-opus-5':     (5.00, 25.00),
    'claude-haiku-4-5-20251001': (1.00, 5.00),   # the refusal fallback
}

# Cache reads are 0.10x input on every model here EXCEPT Opus 5.5, whose reads
# are $0.20/MTok against $4.00 input — 0.05x. A flat 0.10x would overstate the
# dominant token category on the default model by 2x, which defeats the point
# of pricing reads separately at all.
_CACHE_READ_MULT = {'claude-opus-5-5': 0.05}
_DEFAULT_CACHE_READ_MULT = 0.10
_WEB_SEARCH_USD = 10.00 / 1000      # per search request

# Two web-search tool versions, and picking the wrong one is a 400.
#
# `_20260209` adds dynamic filtering (Claude writes code to filter results
# before they reach the context window), which every model in MODELS supports.
# FALLBACK_MODEL does NOT: claude-haiku-4-5 returns
#   400 invalid_request_error: 'claude-haiku-4-5-20251001' does not support
#   programmatic tool calling
# and needs the basic `_20250305` variant (verified against the live API
# 2026-09-28). Since the fallback exists to rescue a refused turn, sending it a
# tool it cannot accept would make the rescue fail exactly when it is needed —
# so the tool is chosen per attempt from the model actually being called, not
# fixed once in build_request.
_WEB_SEARCH_DYNAMIC = {'type': 'web_search_20260209', 'name': 'web_search'}
_WEB_SEARCH_BASIC   = {'type': 'web_search_20250305', 'name': 'web_search'}


def web_search_tool(model: str) -> dict:
    return (_WEB_SEARCH_DYNAMIC if model in _DYNAMIC_SEARCH_MODELS
            else _WEB_SEARCH_BASIC)


# Reasoning effort, and it has to be set explicitly now.
#
# Every other model in MODELS defaults to effort `high`. Opus 5.5 defaults to
# `medium` — so promoting it to the house default WITHOUT this would quietly
# run the playbook a rung BELOW what Opus 5 gave, while the view's own caption
# promises it "reasons harder on the portfolio". A silent downgrade, not an
# error. Sending `high` on the others is a no-op restatement of their default,
# which is the point: one rule, no per-model special case to forget.
#
# FALLBACK_MODEL is excluded, and this is the same trap as the web-search tool
# version: `output_config.effort` ERRORS on claude-haiku-4-5. Fixing effort
# once in build_request would break the rescue request exactly when the rescue
# is needed, so — like the tool — it is chosen per attempt from the model
# actually being called.
_EFFORT = 'high'
_NO_EFFORT_MODELS = {'claude-haiku-4-5-20251001'}


def effort_config(model: str) -> dict | None:
    """`output_config` for this model, or None where effort is unsupported."""
    return None if model in _NO_EFFORT_MODELS else {'effort': _EFFORT}

MAX_TOKENS       = 32000            # streamed, so the SDK timeout is not a factor
MAX_PAUSE_RESUMES = 6               # server-tool turns can stop with pause_turn


# ── Playbook store ─────────────────────────────────────────────────────────

def load_manifest(gcs) -> dict:
    """The playbook registry. An empty registry is a legitimate state (nothing
    uploaded yet) and the view says so rather than erroring."""
    blob = gcs.bucket(BUCKET).blob(MANIFEST_BLOB)
    if not blob.exists():
        return {'playbooks': []}
    return json.loads(blob.download_as_text())


def save_manifest(gcs, manifest: dict) -> None:
    gcs.bucket(BUCKET).blob(MANIFEST_BLOB).upload_from_string(
        json.dumps(manifest, indent=2), content_type='application/json'
    )


def playbook(manifest: dict, playbook_id: str) -> dict | None:
    for p in manifest.get('playbooks', []):
        if p.get('id') == playbook_id:
            return p
    return None


def read_text_blob(gcs, path: str) -> str:
    blob = gcs.bucket(BUCKET).blob(path)
    if not blob.exists():
        raise FileNotFoundError(f'Playbook asset missing from GCS: {path}')
    return blob.download_as_text()


def put_text_blob(gcs, path: str, text: str) -> None:
    gcs.bucket(BUCKET).blob(path).upload_from_string(
        text, content_type='text/markdown; charset=utf-8'
    )


def assemble_system(gcs, pb: dict, optional_keys: list[str]) -> tuple[str, list[str]]:
    """Playbook + its always-on assets + the optional assets the user ticked,
    concatenated into one system string. Returns (text, names_included).

    Order is fixed — playbook, then always[], then optional[] in manifest
    order — because the string is the cached prefix: re-ordering it on a later
    turn would silently cost a full re-bill.
    """
    parts, names = [], []
    parts.append(read_text_blob(gcs, pb['prompt']))
    names.append(pb.get('label', pb['id']))

    for asset in pb.get('always', []):
        parts.append(
            f"\n\n===== {asset['label']} =====\n\n"
            + read_text_blob(gcs, asset['blob'])
        )
        names.append(asset['label'])

    for asset in pb.get('optional', []):
        if asset['key'] in optional_keys:
            parts.append(
                f"\n\n===== {asset['label']} =====\n\n"
                + read_text_blob(gcs, asset['blob'])
            )
            names.append(asset['label'])

    return ''.join(parts), names


# ── Context assembly ───────────────────────────────────────────────────────

# The column set the general playbook names verbatim ("The exports may
# include: client, client_website, market, ..."). Emitting exactly these, in
# this order, means the playbook's stage 2 reads what it was written against.
RESULT_COLUMNS = [
    'pool', 'client', 'client_website', 'market', 'market_kind', 'market_tier',
    'market_subtitle', 'market_rationale', 'aspect_label', 'aspect_kind',
    'aspect_score', 'aspect_scores', 'aspects_hit', 'aspects_total',
    'llm_score', 'llm_rationale', 'record_kind', 'topic_number', 'title',
    'agency', 'broad_agency', 'open_date', 'due_date', 'close_date',
    'is_rolling', 'last_verified_active', 'verify_status',
    'funding_amount', 'grant_summary', 'source',
]

MAX_RESULT_ROWS = 400
DOC_CHAR_CAP    = 40000     # per attached document
NOTES_CHAR_CAP  = 20000


def cap_middle(text: str, limit: int) -> str:
    """Trim by dropping the MIDDLE, not the tail.

    Same reasoning as fathom_client.transcript_text and browser_agent: a
    document front-loads what the company is and back-loads deadlines,
    signatures and next steps. Cutting the tail throws away the half a funding
    strategist most needs.
    """
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return (
        text[:head]
        + f'\n\n[... {len(text) - limit:,} characters omitted from the middle ...]\n\n'
        + text[-tail:]
    )


def results_block(df: pd.DataFrame, *, label: str = 'Matched opportunities') -> dict:
    """The match results as a tab-delimited table the playbook can consolidate."""
    if df is None or df.empty:
        return {'kind': 'results', 'label': label, 'text': '(no rows)', 'rows': 0}

    cols = [c for c in RESULT_COLUMNS if c in df.columns]
    cols += [c for c in df.columns if c not in cols and not str(c).startswith('_')]
    out = df[cols].head(MAX_RESULT_ROWS).copy()

    # Two cleanups, both load-bearing:
    #
    #  * Missing values become EMPTY, not the strings 'nan'/'NaT'/'None'.
    #    `astype(str)` on a column with a missing due_date yields the literal
    #    "NaT", and the playbook reads a deadline column to decide whether an
    #    opportunity is even reachable — "no deadline published" and a deadline
    #    of "nan" are not the same claim, and only one of them is true.
    #  * Tabs and newlines inside a cell would break the row alignment, which
    #    is the same rule the playbook states for its own spreadsheet export.
    for c in out.columns:
        out[c] = (
            out[c].where(out[c].notna(), '')
            .astype(str)
            .str.replace(r'[\t\r\n]+', ' ', regex=True)
            .str.strip()
            .replace({'nan': '', 'NaT': '', 'None': '', '<NA>': ''})
        )

    note = ''
    if len(df) > MAX_RESULT_ROWS:
        note = (
            f'\n\n[Showing the first {MAX_RESULT_ROWS:,} of {len(df):,} rows. '
            'Report this truncation when you account for input rows.]'
        )
    return {
        'kind':  'results',
        'label': f'{label} ({min(len(df), MAX_RESULT_ROWS):,} rows)',
        'text':  out.to_csv(sep='\t', index=False) + note,
        'rows':  int(len(df)),
    }


def profile_block(prof) -> dict:
    """A capability profile rendered for reading, not for parsing.

    Aspects and markets are stored as JSON strings on the profile row; handing
    Claude the raw JSON wastes tokens on punctuation and reads worse than prose.
    """
    def _j(key):
        raw = prof.get(key) or '[]'
        try:
            return json.loads(raw) if isinstance(raw, str) else (raw or [])
        except (ValueError, TypeError):
            return []

    name = str(prof.get('company_name') or '').strip()
    site = str(prof.get('companyWebsite') or '').strip()

    lines = [f'# Capability profile — {name}', f'Website: {site}', '']
    if prof.get('profile_summary'):
        lines += ['## Summary', str(prof['profile_summary']), '']

    aspects = _j('aspects')
    if aspects:
        lines.append('## Capability aspects')
        for i, a in enumerate(aspects, 1):
            kws = ', '.join(a.get('keywords') or [])
            mkts = ', '.join(a.get('markets') or [])
            lines.append(f"{i}. {a.get('label', '')} ({a.get('kind', '')})")
            lines.append(f"   {a.get('text', '')}")
            if kws:
                lines.append(f'   Keywords: {kws}')
            if mkts:
                lines.append(f'   Serves markets: {mkts}')
        lines.append('')

    markets = _j('markets')
    if markets:
        lines.append('## Markets served today')
        for m in markets:
            lines.append(f"- {m.get('market', '')} (tier {m.get('tier', '')})"
                         f" — {m.get('subtitle', '')}")
            if m.get('narrative'):
                lines.append(f"  {m['narrative']}")
        lines.append('')

    unexplored = _j('unexplored_markets')
    if unexplored:
        lines.append('## Unexplored markets (HYPOTHESES — the client does NOT serve these)')
        for m in unexplored:
            lines.append(f"- {m.get('market', '')} — {m.get('subtitle', '')}")
            if m.get('narrative'):
                lines.append(f"  {m['narrative']}")
            if m.get('rationale'):
                lines.append(f"  Gap remaining: {m['rationale']}")
        lines.append('')

    if prof.get('dod_assessment'):
        lines += ['## Defense assessment', str(prof['dod_assessment']), '']
    if prof.get('sources_used'):
        lines.append(f"Profile built from: {prof['sources_used']} "
                     f"(model {prof.get('model', '')}, {prof.get('built_at', '')})")

    return {
        'kind':  'profile',
        'label': f'Capability profile — {name}' if name else 'Capability profile',
        'text':  '\n'.join(lines),
    }


def document_block(name: str, text: str, *, source: str = 'Google Drive') -> dict:
    return {
        'kind':  'document',
        'label': name,
        'text':  f'# {name}\n(source: {source})\n\n{cap_middle(text, DOC_CHAR_CAP)}',
    }


def notes_block(text: str, *, label: str = 'Consultant notes') -> dict:
    return {
        'kind':  'notes',
        'label': label,
        'text':  f'# {label}\n\n{cap_middle(text, NOTES_CHAR_CAP)}',
    }


def render_context(blocks: list[dict]) -> str:
    """The context blocks as one string — the second cached prefix."""
    if not blocks:
        return ''
    out = [
        'The following material comes from The Matcher and is the input for '
        'this engagement. Treat matching scores and rationales as leads to '
        'investigate, not as verified facts.\n'
    ]
    for b in blocks:
        out.append(f"\n\n========== {b['label']} ==========\n\n{b['text']}")
    return ''.join(out)


# ── The Claude call ────────────────────────────────────────────────────────

def build_request(*, system_text: str, context_text: str, messages: list[dict],
                  model: str, web_search: bool) -> dict:
    """Assemble the Messages API kwargs, with the three cache breakpoints.

    Breakpoints, in prefix order (the API allows at most 4 per request):
      1. the system block  — playbook + skills + knowledge base, fixed per session
      2. the context block — match results + profile + attachments at open
      3. the newest message — rolls forward one turn at a time

    Nothing volatile (no timestamp, no run id) goes above breakpoint 3, or the
    prefix would differ every turn and none of this would cache.
    """
    system = [{
        'type': 'text',
        'text': system_text,
        'cache_control': {'type': 'ephemeral'},
    }]

    msgs: list[dict] = []
    if context_text:
        msgs.append({'role': 'user', 'content': [{
            'type': 'text',
            'text': context_text,
            'cache_control': {'type': 'ephemeral'},
        }]})
        msgs.append({'role': 'assistant', 'content': [{
            'type': 'text',
            'text': 'Received. Tell me when to begin.',
        }]})

    # Project each stored message down to the two fields the API accepts.
    # The session record deliberately carries per-turn metadata on assistant
    # messages (`model`, `fell_back`, `usage` — read by the sidebar, the
    # fell-back caption and transcript_markdown), and the API rejects unknown
    # keys inside messages[]:
    #   400 invalid_request_error: messages.1.model: Extra inputs are not permitted
    # Verified against the live API 2026-09-28. Without this projection every
    # session 400s on its SECOND turn — the first one sends only a user
    # message, so it passes and the bug hides.
    #
    # A message with empty content is dropped rather than sent: a non-final
    # assistant message must carry content, and a turn refused by every model
    # comes back with zero blocks.
    msgs.extend(
        {'role': m['role'], 'content': m['content']}
        for m in messages if m.get('content')
    )
    msgs = _roll_cache_breakpoint(msgs)

    kwargs = {
        'model':      model,
        'max_tokens': MAX_TOKENS,
        'system':     system,
        'messages':   msgs,
    }
    if web_search:
        # Replaced per attempt in stream_turn with the variant that model
        # supports; set here only to signal that search is enabled.
        kwargs['tools'] = [web_search_tool(model)]

    # Likewise replaced (or dropped) per attempt — Haiku rejects it outright.
    oc = effort_config(model)
    if oc:
        kwargs['output_config'] = oc
    return kwargs


def _roll_cache_breakpoint(messages: list[dict]) -> list[dict]:
    """Put a single ephemeral marker on the newest message's last text block.

    One rolling marker rather than one per turn, because the API allows at most
    4 per request and two are already spent on system + context. Copies rather
    than mutating, so the stored session record never accumulates markers.
    """
    if not messages:
        return messages
    out = [dict(m) for m in messages]
    for msg in reversed(out):
        content = msg.get('content')
        if not isinstance(content, list):
            continue
        blocks = [dict(b) for b in content]
        for b in reversed(blocks):
            if b.get('type') == 'text':
                b['cache_control'] = {'type': 'ephemeral'}
                msg['content'] = blocks
                return out
        break
    return out


def stream_turn(client, *, request: dict, on_text=None, on_activity=None,
                fallback_model: str | None = FALLBACK_MODEL) -> dict:
    """Run one assistant turn, streaming text out through `on_text`.

    Returns {'content', 'usage', 'stop_reason', 'model_used', 'fell_back'}.

    A `stop_reason` of 'refusal' on the requested model is retried once on
    `fallback_model` — see the FALLBACK_MODEL note above for the measurement
    that motivates it. Usage is accumulated across both attempts so the
    reported cost stays honest even when the first one produced nothing.
    """
    models = [request['model']]
    if fallback_model and fallback_model != request['model']:
        models.append(fallback_model)

    usage = {'input_tokens': 0, 'output_tokens': 0,
             'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0,
             'web_search_requests': 0}
    # Each attempt is priced at ITS OWN model's rate and the dollars summed.
    # Merging the token counts first and pricing the total at the model that
    # happened to answer would bill the refused model's prefix — tens of
    # thousands of cache tokens on a long session — at Haiku's rate, and the
    # running cost readout is the only spend signal the consultant has.
    cost = 0.0

    for i, model in enumerate(models):
        attempt = dict(request, model=model)
        if request.get('tools'):
            attempt['tools'] = [web_search_tool(model)]
        # Both of these are per-model: the fallback rejects the dynamic search
        # tool AND `output_config` entirely, so carrying either one over from
        # the requested model would 400 the rescue attempt.
        oc = effort_config(model)
        if oc:
            attempt['output_config'] = oc
        else:
            attempt.pop('output_config', None)

        result = _stream_once(client, attempt, on_text, on_activity)
        for k in usage:
            usage[k] += result['usage'][k]
        cost += turn_cost(model, result['usage'])

        last = i == len(models) - 1
        refused = result['stop_reason'] == 'refusal'

        # Only a PRE-OUTPUT refusal is retried. A refusal that fires after
        # Claude has already written part of the answer ("mid-stream") is a
        # different situation and re-running is the wrong response:
        #
        #   * The partial is real work the consultant has already watched
        #     appear, and on the first live session it was "what I was looking
        #     for". Re-running throws it away.
        #   * The retry repeats the whole turn on the fallback model with a
        #     cold cache — on the measured NIH session that is 85,614 prompt
        #     tokens and minutes of web research, during which the page looks
        #     frozen.
        #   * Both attempts stream through the same on_text callback, so the
        #     second answer is appended to the first on screen, while only the
        #     second is stored. Displayed and saved text then disagree.
        #
        # So: mid-stream refusal returns what was produced, flagged, and the
        # view shows it as a truncated answer with Retry available.
        if not refused or last or result['content']:
            return {**result, 'usage': usage, 'cost_usd': cost,
                    'model_used': model, 'fell_back': i > 0,
                    'partial_refusal': refused and bool(result['content'])}

        if on_activity:
            on_activity(f'⚠️ {model} declined — retrying on {models[i + 1]}')

    return {'content': [], 'usage': usage, 'cost_usd': cost,
            'stop_reason': 'refusal', 'model_used': models[-1],
            'fell_back': True, 'partial_refusal': False}


def _stream_once(client, request: dict, on_text, on_activity) -> dict:
    """One model's attempt at the turn, resuming through any `pause_turn`.

    `pause_turn` is not an error: the server-side search loop has an iteration
    ceiling, and hitting it means "re-send and I will carry on". Without this
    loop a long verification turn would come back truncated with no error
    anywhere, which is the silent-failure class this codebase keeps getting
    bitten by. The resume cap stops a pathological turn from looping forever.
    """
    content: list[dict] = []
    usage   = {'input_tokens': 0, 'output_tokens': 0,
               'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0,
               'web_search_requests': 0}
    messages = list(request['messages'])
    stop_reason = None

    for _ in range(MAX_PAUSE_RESUMES + 1):
        kwargs = dict(request, messages=messages)
        with client.messages.stream(**kwargs) as stream:
            for event in stream:
                if event.type == 'content_block_delta' and \
                        getattr(event.delta, 'type', '') == 'text_delta':
                    if on_text:
                        on_text(event.delta.text)
                elif event.type == 'content_block_start':
                    kind = getattr(event.content_block, 'type', '')
                    if kind != 'text' and on_activity:
                        on_activity(_activity_label(event.content_block))
            msg = stream.get_final_message()

        blocks = [_dump(b) for b in msg.content]
        content.extend(blocks)
        _accumulate(usage, msg.usage)
        stop_reason = msg.stop_reason

        if msg.stop_reason != 'pause_turn':
            break
        # Re-send with the paused turn appended; the server resumes from there.
        messages = messages + [{'role': 'assistant', 'content': blocks}]

    return {'content': content, 'usage': usage, 'stop_reason': stop_reason}


def _dump(block) -> dict:
    for attr in ('model_dump', 'dict'):
        fn = getattr(block, attr, None)
        if callable(fn):
            try:
                return fn(exclude_none=True)
            except TypeError:
                return fn()
    return dict(block)


def _accumulate(usage: dict, u) -> None:
    for k in ('input_tokens', 'output_tokens',
              'cache_creation_input_tokens', 'cache_read_input_tokens'):
        usage[k] += int(getattr(u, k, 0) or 0)
    stu = getattr(u, 'server_tool_use', None)
    if stu is not None:
        usage['web_search_requests'] += int(
            getattr(stu, 'web_search_requests', 0) or 0
        )


def _activity_label(block) -> str:
    kind = getattr(block, 'type', '')
    if kind == 'server_tool_use':
        name  = getattr(block, 'name', 'tool')
        query = (getattr(block, 'input', None) or {}).get('query')
        return f'🔎 {name}: {query}' if query else f'🔎 {name}'
    if kind == 'web_search_tool_result':
        return '📄 read search results'
    if kind == 'web_fetch_tool_result':
        return '📄 fetched a page'
    if kind in ('code_execution_tool_result', 'bash_code_execution_tool_result'):
        return '⚙️ filtered results'
    if kind in ('thinking', 'redacted_thinking'):
        return '💭 thinking'
    return f'· {kind}'


def turn_cost(model: str, usage: dict) -> float:
    """Actual dollars for one turn, from the response's own usage."""
    rate_in, rate_out = _PRICES.get(model, _PRICES['claude-sonnet-4-6'])
    m = 1_000_000
    return (
        usage.get('input_tokens', 0) / m * rate_in
        + usage.get('cache_read_input_tokens', 0) / m * rate_in
          * _CACHE_READ_MULT.get(model, _DEFAULT_CACHE_READ_MULT)
        + usage.get('cache_creation_input_tokens', 0) / m * rate_in * 1.25
        + usage.get('output_tokens', 0) / m * rate_out
        + usage.get('web_search_requests', 0) * _WEB_SEARCH_USD
    )


# ── Reading a stored turn back ─────────────────────────────────────────────

def message_text(content) -> str:
    """Every text block of a message, joined.

    Never `content[0]['text']` — a turn that searched the web leads with
    server_tool_use blocks, and one that declines has no blocks at all. Same
    failure this repo already documents in anthropic_utils.
    """
    if isinstance(content, str):
        return content
    return ''.join(
        b.get('text', '') or ''
        for b in (content or [])
        if isinstance(b, dict) and b.get('type') == 'text'
    ).strip()


def activity_lines(content) -> list[str]:
    """Non-text blocks of a stored message, as display captions."""
    if not isinstance(content, list):
        return []
    out = []
    for b in content:
        if not isinstance(b, dict) or b.get('type') == 'text':
            continue
        kind = b.get('type', '')
        if kind == 'server_tool_use':
            q = (b.get('input') or {}).get('query')
            out.append(f'🔎 {b.get("name", "tool")}: {q}' if q
                       else f'🔎 {b.get("name", "tool")}')
        elif kind == 'web_search_tool_result':
            n = len(b.get('content') or []) if isinstance(b.get('content'), list) else 0
            out.append(f'📄 {n} search result(s)' if n else '📄 search results')
        elif kind.endswith('code_execution_tool_result'):
            n = len(output_file_ids([b]))
            out.append(f'📦 exported {n} file(s)' if n else '⚙️ ran code')
        elif kind in ('thinking', 'redacted_thinking'):
            continue
        else:
            out.append(f'· {kind}')
    return out


# ── Files Claude built in the sandbox ──────────────────────────────────────
#
# The general playbook ends in a deck, and with web search on Claude also has a
# code-execution sandbox (dynamic filtering runs there). It uses it: on the
# Cell X session it wrote a python-pptx script, built a 14-slide deck and
# copied it to $OUTPUT_DIR, which the API turns into a Files API upload and
# reports only as a `file_id` inside the tool result. Nothing rendered it, so
# the consultant was told to "download it from the attachment area above" when
# no such area existed — and asking again paid for a full rebuild, because the
# sandbox container is not carried across turns.
#
# The fix is to treat those file_ids as deliverables: download each once,
# copy it into GCS beside the session (the Files API is not a durable store
# and is scoped to the API key's workspace), and record it on the message.

FILES_SUBDIR = 'files'


def output_file_ids(content) -> list[str]:
    """file_ids of every file a code-execution result in `content` exported."""
    if not isinstance(content, list):
        return []
    ids: list[str] = []
    for b in content:
        if not isinstance(b, dict) or not str(b.get('type', '')).endswith(
                'code_execution_tool_result'):
            continue
        inner = b.get('content')
        outputs = inner.get('content') if isinstance(inner, dict) else None
        for o in outputs or []:
            fid = o.get('file_id') if isinstance(o, dict) else None
            if fid and fid not in ids:
                ids.append(fid)
    return ids


def file_blob(session_id: str, file_id: str, filename: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', filename or 'file').strip('_') or 'file'
    return f'{SESSION_PREFIX}{session_id}/{FILES_SUBDIR}/{file_id}_{safe}'


def archive_output_file(client, gcs, session_id: str, file_id: str) -> dict:
    """Download one sandbox file from the Files API and copy it into GCS.

    Returns the entry stored on the message: {file_id, filename, mime, size,
    blob}, or {file_id, error} when it could not be fetched — a failure is
    recorded rather than raised so one unreachable file never loses the turn.
    """
    try:
        meta = client.beta.files.retrieve_metadata(file_id)
        data = client.beta.files.download(file_id).read()
        name = getattr(meta, 'filename', '') or file_id
        mime = getattr(meta, 'mime_type', '') or 'application/octet-stream'
        blob = file_blob(session_id, file_id, name)
        gcs.bucket(BUCKET).blob(blob).upload_from_string(data, content_type=mime)
        return {'file_id': file_id, 'filename': name, 'mime': mime,
                'size': len(data), 'blob': blob}
    except Exception as e:
        return {'file_id': file_id, 'error': f'{type(e).__name__}: {e}'}


def archive_message_files(client, gcs, session_id: str, msg: dict) -> bool:
    """Make sure every exported file of an assistant message is in GCS.

    Fills `msg['files']`. Entries that failed are retried on the next call;
    returns True when the message changed and the record should be saved.
    """
    ids = output_file_ids(msg.get('content'))
    if not ids:
        return False
    have = {f['file_id']: f for f in msg.get('files') or [] if f.get('blob')}
    missing = [fid for fid in ids if fid not in have]
    if not missing:
        return False
    for fid in missing:
        have[fid] = archive_output_file(client, gcs, session_id, fid)
    msg['files'] = [have[fid] for fid in ids]
    return True


def read_file_blob(gcs, blob: str) -> bytes:
    return gcs.bucket(BUCKET).blob(blob).download_as_bytes()


# ── Session records ────────────────────────────────────────────────────────

def new_session_id(company_name: str = '') -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', (company_name or 'session').lower()).strip('-')
    return f'strategy_{datetime.now():%Y-%m-%d_%H-%M-%S}_{slug[:32] or "session"}'


def session_blob(session_id: str) -> str:
    return f'{SESSION_PREFIX}{session_id}/session.json'


def save_session(gcs, record: dict) -> None:
    record['updated_at'] = datetime.now().isoformat(timespec='seconds')
    gcs.bucket(BUCKET).blob(session_blob(record['session_id'])).upload_from_string(
        json.dumps(record), content_type='application/json'
    )


def load_session(gcs, session_id: str) -> dict | None:
    blob = gcs.bucket(BUCKET).blob(session_blob(session_id))
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def list_sessions(gcs, limit: int = 60) -> list[dict]:
    """Recent sessions, newest first. Summary fields only — the message list of
    a long session is megabytes and the picker never needs it."""
    rows = []
    for blob in gcs.list_blobs(BUCKET, prefix=SESSION_PREFIX):
        if not blob.name.endswith('/session.json'):
            continue
        try:
            rec = json.loads(blob.download_as_text())
        except Exception:
            continue
        rows.append({
            'session_id': rec.get('session_id', ''),
            'title':      rec.get('title', ''),
            'company':    (rec.get('company') or {}).get('name', ''),
            'playbook':   rec.get('playbook', ''),
            'model':      rec.get('model', ''),
            'turns':      sum(1 for m in rec.get('messages', [])
                              if m.get('role') == 'user'),
            'cost_usd':   float(rec.get('cost_usd', 0.0)),
            'created_by': rec.get('created_by', ''),
            'updated_at': rec.get('updated_at', ''),
        })
    rows.sort(key=lambda r: r['updated_at'], reverse=True)
    return rows[:limit]


def transcript_markdown(record: dict) -> str:
    """The session as a markdown document — the same shape as the artefacts
    already in project-context/, so a consultant marks it up the same way."""
    company = (record.get('company') or {}).get('name', '')
    out = [
        f"# Funding strategy session — {company or record.get('title', '')}",
        '',
        f"Session: `{record.get('session_id', '')}`  ",
        f"Playbook: {record.get('playbook_label', record.get('playbook', ''))}  ",
        f"Model: {record.get('model', '')}  ",
        f"Started: {record.get('created_at', '')} by {record.get('created_by', 'unknown')}  ",
        f"Cost so far: ${float(record.get('cost_usd', 0.0)):.2f}",
        '',
        '---',
        '',
    ]
    ctx  = record.get('context_blocks') or []
    atts = record.get('attachments') or []
    if ctx or atts:
        out += ['## Attached context', '']
        out += [f"- {b.get('label', '')}" for b in ctx]
        out += [f'- {label} (attached mid-session)' for label in atts]
        out += ['', '---', '']

    for msg in record.get('messages', []):
        text = message_text(msg.get('content'))
        if not text:
            continue
        who = 'Consultant' if msg.get('role') == 'user' else 'Claude'
        if msg.get('fell_back'):
            who += f" ({msg.get('model', '')} — after a refusal on the session model)"
        elif msg.get('continued_on'):
            who += f" ({msg.get('model', '')} — continuing a partially refused answer)"
        if msg.get('partial_refusal'):
            who += ' — stopped part-way by a safety classifier'
        out += [f'## {who}', '', text, '']
        for line in activity_lines(msg.get('content')):
            out.append(f'> {line}')
        for f in msg.get('files') or []:
            if f.get('blob'):
                out.append(f"> 📦 {f['filename']} — gs://{BUCKET}/{f['blob']}")
        out.append('')
    return '\n'.join(out)
