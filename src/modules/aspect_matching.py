"""
Aspect matching core
--------------------
Streamlit-free scoring and LLM re-ranking shared by the Bulk Aspect Match view
(the whole client directory, one unit per client or per market) and the Grant
Search view (a single company — an existing client profile, or one built ad hoc
from pasted notes).

This module was extracted from views/aspect_match.py so there is exactly one
implementation of the parts that are load-bearing and easy to get subtly wrong:

  * a topic qualifies for a unit on `min_hits` of its ASPECT vectors, with the
    market narrative scored but counted only when the unit has no aspects;
  * confirmed and unexplored market units are re-ranked by DIFFERENT prompts
    (the confirmed prompt asks "could they propose today", which scores every
    hypothesis 1-2 and silently empties an unexplored run);
  * re-rank calls are deduped by (client, agency, topic, aspect, market kind),
    never across kinds, because the two prompts answer different questions;
  * each re-rank pair is sampled RERANK_SAMPLES times (median kept) and cached
    in GCS by a hash of model + prompts, so a repeated search returns the same
    scores instead of fresh draws at the model's default temperature.

Nothing here imports streamlit: progress is reported through plain callables so
a caller can wire it to st.progress, a log line, or nothing at all.
"""

import asyncio
import hashlib
import json
import random
import re
import statistics
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from anthropic import AsyncAnthropic

import src.modules.anthropic_utils as au
import src.modules.aspect_profile as ap

# ── Constants ──────────────────────────────────────────────────────────────

RERANK_MODELS = ['claude-haiku-4-5-20251001', 'claude-sonnet-4-6']
CONCURRENCY   = 15
MAX_RETRIES   = 5

# Re-rank stability. The SDK no longer accepts `temperature`, so every call
# samples at the model default and a pair on a 3/4 boundary lands on either
# side run to run. Three independent samples per pair, median kept, narrows
# the first score; the GCS cache makes every later run of the same pair return
# that exact score. The cache key covers model + system prompt + user message
# + sample count, so any change to the profile, the topic text, the rubric or
# the model is a miss and re-scores on its own — nothing needs invalidating.
RERANK_SAMPLES      = 3
RERANK_CACHE_PREFIX = 'aspect-match-cache/rerank/'
_CACHE_IO_WORKERS   = 32

# Stored on every result row, and what the re-rank prompt selection keys off.
ROW_CONFIRMED  = 'confirmed'
ROW_UNEXPLORED = 'unexplored'

# The "no category filter" sentinel for plan_units. It lives here rather than in
# the view because plan_units compares against it; the view aliases this name.
CATEGORY_ALL = 'All categories'

# One scoring subject: a whole company (market None, kind ''), or one market of
# one company. `mi` indexes that market's narrative vector inside the profile's
# flat market_embeddings / unexplored_embeddings block.
Unit = namedtuple('Unit', 'prof market mi kind')

# Topic columns carried into the results, when present.
TOPIC_COLS = [
    'topic_number', 'title', 'agency', 'broad_agency', 'due_date', 'close_date',
    'open_date', 'funding_amount', 'grant_summary', 'source',
]

# Result columns worth seeing first, in this order.
DISPLAY_FIRST = [
    'client', 'pool', 'market', 'market_kind', 'market_tier', 'aspect_label', 'aspect_score',
    'aspects_hit', 'aspects_total',
    'llm_score', 'llm_samples', 'llm_rationale', 'topic_number', 'title', 'agency', 'broad_agency',
]


def display_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Result columns reordered for display, with the internal _-prefixed
    prompt payload columns dropped."""
    internal = [c for c in df.columns if c.startswith('_')]
    first    = [c for c in DISPLAY_FIRST if c in df.columns]
    rest     = [c for c in df.columns if c not in first and c not in internal]
    return df[first + rest]


# ── Scoring ────────────────────────────────────────────────────────────────

def stack_topic_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """(T, dim) float32 matrix + the topic rows it corresponds to. Rows with a
    missing or wrong-length vector are dropped."""
    keep, vecs = [], []
    for idx, emb in df['embeddings'].items():
        if isinstance(emb, (list, np.ndarray)) and len(emb) == ap.EMBED_DIM:
            keep.append(idx)
            vecs.append(np.asarray(emb, dtype=np.float32))
    if not vecs:
        return np.zeros((0, ap.EMBED_DIM), dtype=np.float32), df.iloc[0:0]
    meta = df.loc[keep].drop(columns=['embeddings'], errors='ignore').reset_index(drop=True)
    return np.vstack(vecs), meta


def unit_markets(prof, kinds: list[str]) -> list[tuple[str, dict, int]]:
    """(kind, market, index-into-its-vector-block) for one profile, restricted
    to the requested kinds. The index is positional within the kind's own flat
    embedding block, which is why the kind has to travel with it."""
    out: list[tuple[str, dict, int]] = []
    if ROW_CONFIRMED in kinds:
        out += [(ROW_CONFIRMED, m, i)
                for i, m in enumerate(ap.profile_markets(prof))]
    if ROW_UNEXPLORED in kinds:
        out += [(ROW_UNEXPLORED, m, i)
                for i, m in enumerate(ap.profile_unexplored(prof))]
    return out


def market_counts(selected: pd.DataFrame, kinds: list[str]) -> dict[str, int]:
    """Canonical market name → how many of the selected clients have it, across
    the requested kinds."""
    counts: dict[str, int] = {}
    for _, prof in selected.iterrows():
        names = {str(m.get('market') or '') for _k, m, _i in unit_markets(prof, kinds)}
        for name in names:
            if name:
                counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def tier_counts(selected: pd.DataFrame, kinds: list[str]) -> dict[int, int]:
    """Tier rank → how many markets of the selected clients sit at it. Drives
    the tier picker, so it only ever offers ranks that exist."""
    counts: dict[int, int] = {}
    for _, prof in selected.iterrows():
        for _k, market, _i in unit_markets(prof, kinds):
            rank = ap.market_tier_rank(market)
            counts[rank] = counts.get(rank, 0) + 1
    return dict(sorted(counts.items()))


def plan_units(
    selected: pd.DataFrame,
    by_market: bool,
    kinds: list[str],
    tiers: set[int],
    category: str,
) -> tuple[list[Unit], list[str]]:
    """One Unit per scoring subject.

    Whole-company mode gives one unit per client (market None). Market mode
    gives one unit per (client, kind, market) surviving the tier and category
    filters — each is scored, capped and re-ranked as if it were its own
    company. Cheap enough to call on every rerun for the run-size estimate.

    An empty `tiers` means every tier, matching how CATEGORY_ALL behaves — a
    deselect-everything state would otherwise silently plan zero units."""
    plan: list[Unit] = []
    skipped: list[str] = []

    for _, prof in selected.iterrows():
        if not by_market:
            plan.append(Unit(prof, None, -1, ''))
            continue

        if ROW_CONFIRMED in kinds and not ap.profile_markets(prof):
            skipped.append(
                f'{prof["company_name"]}: profile has no markets — rebuild it in '
                'Client Profiles to match it by market'
            )
        if ROW_UNEXPLORED in kinds and not ap.profile_unexplored(prof):
            skipped.append(
                f'{prof["company_name"]}: profile has no unexplored markets — '
                'rebuild it in Client Profiles with "Assess unexplored markets" on'
            )

        for kind, market, i in unit_markets(prof, kinds):
            name = str(market.get('market') or '')
            if tiers and ap.market_tier_rank(market) not in tiers:
                continue
            if category != CATEGORY_ALL and name != category:
                continue
            plan.append(Unit(prof, market, i, kind))

    return plan, skipped


def match_units(
    plan: list[Unit],
    topic_matrix: np.ndarray,
    topic_meta: pd.DataFrame,
    threshold: float,
    min_hits: int,
    top_k: int,
    progress=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Candidate (unit, topic) rows: a topic qualifies for a unit when at least
    `min_hits` of the unit's *aspect* vectors clear `threshold`; the top `top_k`
    per unit by best score are kept. A confirmed market unit carries its
    earmarked aspect vectors plus the market narrative vector, which for Defense
    is the only place the DoD framing exists.

    An unexplored market unit carries its narrative vector and nothing else.
    Scoring the confirmed aspect vectors under it would mostly return the topics
    its confirmed markets already returned, and rerank_groups would then share
    one score between two different framings of the same (client, topic, aspect)
    pair — written from whichever framing happened to come first.

    `progress`, when given, is called as progress(fraction, text) per unit."""
    meta_cols = [c for c in TOPIC_COLS if c in topic_meta.columns]
    # Materialised once — .iloc[ti] per candidate builds a Series per row and
    # dominates the loop on large runs.
    meta_records = topic_meta[meta_cols].to_dict('records')
    rows: list[dict] = []
    skipped: list[str] = []
    total = len(plan)

    # A client appears once per market, so unpack its stored vectors once.
    aspect_cache: dict[str, tuple] = {}
    market_cache: dict[str, np.ndarray] = {}
    unexp_cache:  dict[str, np.ndarray] = {}

    for n, (prof, market, mi, kind) in enumerate(plan, 1):
        client = str(prof['company_name'])
        label  = f'{client} · {market["market"]}' if market else client
        if kind == ROW_UNEXPLORED:
            label += ' (unexplored)'
        if progress is not None:
            progress(n / total, f'Scoring {label} ({n}/{total})')

        key = str(prof['company_key'])
        if key not in aspect_cache:
            aspect_cache[key] = (ap.profile_aspects(prof), ap.unpack_embeddings(prof))
        aspects, matrix = aspect_cache[key]

        if matrix.shape[0] == 0 or len(aspects) != matrix.shape[0]:
            skipped.append(f'{client}: profile has no usable aspect vectors — rebuild it')
            continue

        if market is None:
            unit_aspects, unit_matrix = aspects, matrix
            n_aspect_vecs = unit_matrix.shape[0]
        elif kind == ROW_UNEXPLORED:
            # Narrative-only by design (see the docstring). The narrative is the
            # whole hypothesis, so it is also the only thing worth scoring.
            if key not in unexp_cache:
                unexp_cache[key] = ap.unpack_unexplored_embeddings(prof)
            unexp_vectors = unexp_cache[key]
            if not 0 <= mi < unexp_vectors.shape[0]:
                skipped.append(
                    f'{label}: unexplored market has no narrative vector — '
                    'rebuild the profile'
                )
                continue
            unit_aspects = [{
                'label': f'Market: {market.get("market")}',
                'kind':  'market',
                'text':  str(market.get('narrative') or ''),
            }]
            unit_matrix   = unexp_vectors[mi][None, :]
            n_aspect_vecs = 0
        else:
            idx          = ap.market_aspect_indices(aspects, str(market.get('market')))
            unit_aspects = [aspects[i] for i in idx]
            unit_matrix  = matrix[idx] if idx else matrix[:0]

            if key not in market_cache:
                market_cache[key] = ap.unpack_market_embeddings(prof)
            market_vectors = market_cache[key]
            n_aspect_vecs = unit_matrix.shape[0]
            if 0 <= mi < market_vectors.shape[0]:
                unit_aspects = unit_aspects + [{
                    'label': f'Market: {market.get("market")}',
                    'kind':  'market',
                    'text':  str(market.get('narrative') or ''),
                }]
                unit_matrix = np.vstack([unit_matrix, market_vectors[mi][None, :]])

            if unit_matrix.shape[0] == 0:
                skipped.append(
                    f'{label}: market has no aspect or narrative vectors — rebuild the profile'
                )
                continue

        scores  = unit_matrix @ topic_matrix.T     # (n_vectors, T)
        best_i  = scores.argmax(axis=0)
        best    = scores.max(axis=0)
        cleared = scores >= threshold
        # min_hits counts capabilities. The market narrative is one more way of
        # describing the same market, not an extra capability, so it stays out
        # of the count - unless it is all the unit has, which is the case for a
        # market no aspect was earmarked to.
        hits    = (cleared[:n_aspect_vecs] if n_aspect_vecs else cleared).sum(axis=0)
        # A unit with no aspect vectors at all — an unexplored market, or a
        # confirmed market no aspect was earmarked to — is scored on its
        # narrative alone, so there is at most one hit to count. Applying a
        # min_hits above 1 would drop every such unit without saying why.
        effective_hits = min_hits if n_aspect_vecs else 1

        qualified = np.where((best >= threshold) & (hits >= effective_hits))[0]
        if qualified.size == 0:
            continue
        order = qualified[np.argsort(best[qualified])[::-1][:top_k]]

        # The confirmed capabilities an unexplored market says it would draw on.
        # Not scored — this is the evidence the re-ranker judges the extension
        # against, so a hypothesis can't be scored on its own optimism.
        linked_text = ''
        if kind == ROW_UNEXPLORED:
            linked_text = '\n\n'.join(
                f"{aspects[i].get('label', '')}: {aspects[i].get('text', '')}"
                for i in ap.unexplored_aspect_indices(aspects, market)
            )

        for ti in order:
            ai     = int(best_i[ti])
            aspect = unit_aspects[ai]
            row = {
                'client':           client,
                # Which pool the profile came from (aspect_profile.load_profiles
                # stamps it). A run over both pools must never let a prospect
                # row read as a client one; '' for pre-pool callers.
                'pool':             str(prof.get('pool') or ''),
                'client_website':   prof['companyWebsite'],
                'market':           str(market.get('market')) if market else '',
                'market_kind':      kind or ROW_CONFIRMED,
                'market_tier':      ap.tier_ordinal(ap.market_tier_rank(market)) if market else '',
                'market_subtitle':  str(market.get('subtitle') or '') if market else '',
                'market_rationale': str(market.get('rationale') or '') if market else '',
                'aspect_label':     aspect.get('label', ''),
                'aspect_kind':      aspect.get('kind', ''),
                'aspect_score':     round(float(best[ti]), 4),
                'aspects_hit':      int(hits[ti]),
                'aspects_total':    n_aspect_vecs or len(unit_aspects),
                'aspect_scores':    json.dumps({
                    unit_aspects[j].get('label', f'aspect_{j + 1}'): round(float(scores[j, ti]), 4)
                    for j in range(len(unit_aspects))
                }),
                '_company_key':     prof['company_key'],
                '_aspect_text':     aspect.get('text', ''),
                '_linked_aspects':  linked_text,
                '_profile_summary': prof['profile_summary'],
            }
            row.update(meta_records[int(ti)])
            rows.append(row)

    return pd.DataFrame(rows), skipped


# ── LLM re-rank ────────────────────────────────────────────────────────────

RERANK_SYSTEM = (
    'You are screening a federal grant topic against one specific capability of a '
    'company that a proposal-writing firm represents.\n'
    'First identify the topic\'s CORE technical need — the thing a proposal must '
    'deliver — then check whether the capability text describes doing it.\n'
    'Score the fit from 1 to 5:\n'
    '5 = the core need is squarely what the capability describes; the company could '
    'write the technical approach today with no new capability\n'
    '4 = the core need is covered, but one secondary requirement (a sub-system, '
    'application domain, test environment or deliverable) is not evidenced\n'
    '3 = the capability addresses part of the core need; real adaptation or a '
    'partner would be needed\n'
    '2 = shared domain or keywords only; the core need is something the capability '
    'does not do\n'
    '1 = no meaningful connection\n\n'
    'Judge only the capability as described — never assume capabilities that are not stated.\n'
    'When torn between two scores, choose the lower.\n'
    'Return ONLY a JSON object, no markdown or headings — rationale first, then score:\n'
    '{"rationale": "<at most 50 words naming the core need and whether it is '
    'covered>", "score": <integer 1-5>}'
)


# The confirmed prompt above asks "could they propose to this TODAY", and
# rightly answers 1 or 2 for anything speculative. Pointing it at an unexplored
# market — a hypothesis by construction — would score the entire kind below any
# selectable minimum, and an unscored/low-scored run shows an empty table with
# no error at all. So unexplored units get their own question, and their own
# scale, judged against the capabilities the client demonstrably has.
RERANK_UNEXPLORED_SYSTEM = (
    'You are screening a federal grant topic against a market a company does NOT '
    'currently serve, but which a capability analysis suggests it could extend into.\n'
    'You are given the market hypothesis, the gap that analysis says remains, and the '
    'capabilities the company demonstrably has today.\n'
    'Score from 1 to 5 how plausibly this company could pursue this topic by extending '
    'those existing capabilities:\n'
    '5 = a clear extension of a stated capability — adaptation only, no new science\n'
    '4 = a strong extension with one identifiable gap to close\n'
    '3 = plausible, but a real capability, qualification or certification gap stands in the way\n'
    '2 = would require capabilities the company has not demonstrated\n'
    '1 = unrelated to anything the company can do\n\n'
    'Judge the extension ONLY against the listed capabilities — never assume capabilities '
    'that are not stated, and never credit the hypothesis for being ambitious.\n'
    'When torn between two scores, choose the lower.\n'
    'Return ONLY a JSON object, no markdown or headings — rationale first, then score:\n'
    '{"rationale": "<at most 50 words naming which listed capability carries over and '
    'what gap remains>", "score": <integer 1-5>}'
)


def rerank_user_message(row: dict) -> str:
    if row.get('market_kind') == ROW_UNEXPLORED:
        subtitle = str(row.get('market_subtitle') or '')
        return (
            f"Company: {row.get('client', '')}\n"
            f"Company profile: {str(row.get('_profile_summary') or '')[:1500]}\n\n"
            f"Market the company does NOT currently serve — {row.get('market', '')}"
            f"{(': ' + subtitle) if subtitle else ''}\n"
            f"{str(row.get('_aspect_text') or '')[:2000]}\n\n"
            f"Gap the analysis says remains: "
            f"{str(row.get('market_rationale') or 'not stated')[:800]}\n\n"
            f"Capabilities the company demonstrably has today:\n"
            f"{str(row.get('_linked_aspects') or '(none recorded)')[:3000]}\n\n"
            f"Grant topic: {row.get('title', '')}\n"
            f"Agency: {row.get('agency', '') or row.get('broad_agency', '')}\n"
            f"Topic description:\n{str(row.get('grant_summary') or '')[:6000]}"
        )
    return (
        f"Company: {row.get('client', '')}\n"
        f"Company profile: {str(row.get('_profile_summary') or '')[:1500]}\n\n"
        f"Matched capability — {row.get('aspect_label', '')}:\n"
        f"{str(row.get('_aspect_text') or '')[:2000]}\n\n"
        f"Grant topic: {row.get('title', '')}\n"
        f"Agency: {row.get('agency', '') or row.get('broad_agency', '')}\n"
        f"Topic description:\n{str(row.get('grant_summary') or '')[:6000]}"
    )


def parse_rerank(text: str) -> tuple[int | None, str]:
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', (text or '').strip())
    start, end = cleaned.find('{'), cleaned.rfind('}')
    if start != -1 and end > start:
        try:
            obj = json.loads(cleaned[start:end + 1])
            score = int(float(obj.get('score')))
            return max(1, min(5, score)), str(obj.get('rationale') or '')[:400]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # Tolerate a stray sentence around the JSON, or a markdown answer
    # ("**Rationale:** ... **Score:** 4"), rather than losing the score. The
    # LAST score mention wins: a rationale-first answer can quote the rubric.
    hits = re.findall(r'score\W{0,6}?[:=]\W{0,6}?([1-5])\b', cleaned, re.IGNORECASE)
    if hits:
        r = (
            re.search(r'"rationale"\s*:\s*"([^"]*)"', cleaned, re.IGNORECASE)
            or re.search(r'rationale\W*[:=]\W*(.+?)\s*(?:\*\*)?score', cleaned,
                         re.IGNORECASE | re.DOTALL)
        )
        return int(hits[-1]), (r.group(1).strip()[:400] if r else '')
    return None, '(unparseable response)'


def rerank_system(row: dict) -> str:
    return RERANK_UNEXPLORED_SYSTEM if row.get('market_kind') == ROW_UNEXPLORED else RERANK_SYSTEM


def combine_samples(samples: list[tuple[int | None, str]]) -> tuple[int | None, str]:
    """Median of the samples that produced a score, with the rationale of a
    sample that gave exactly that score. median_low, so an even count (one
    sample failed) settles on the lower score — the rubric's own tie-break."""
    scored = [(s, r) for s, r in samples if s is not None]
    if not scored:
        # Every sample failed: surface the first failure reason
        return None, (samples[0][1] if samples else '(scoring failed: no samples)')
    med = statistics.median_low(s for s, _ in scored)
    return med, next(r for s, r in scored if s == med)


async def rerank_async(
    rows: list[tuple[int, dict]], api_key: str, model: str, on_done,
    samples: int = RERANK_SAMPLES, persist=None,
) -> list[tuple[int, int | None, str, list[int | None]]]:
    """Score each row `samples` times and keep the median. Returns
    (idx, score, rationale, per-sample scores). `persist(idx, score, rationale,
    sample_scores)`, when given, runs on a worker thread as each pair finishes,
    so a page that dies mid-run keeps everything already scored."""
    sem = asyncio.Semaphore(CONCURRENCY)

    async with AsyncAnthropic(api_key=api_key) as client:
        async def sample(row: dict) -> tuple[int | None, str]:
            async with sem:
                for attempt in range(MAX_RETRIES):
                    try:
                        resp = await client.messages.create(
                            model=model,
                            max_tokens=300,
                            # No temperature: the anthropic 1.x SDK removed the
                            # parameter. Calls therefore sample at the model
                            # default, which is why each pair is scored
                            # `samples` times and the median kept.
                            system=rerank_system(row),
                            messages=[{'role': 'user', 'content': rerank_user_message(row)}],
                        )
                        return parse_rerank(au.response_text(resp))
                    except Exception as e:
                        err = str(e)
                        retryable = any(
                            x in err for x in
                            ('429', '529', 'overloaded', 'rate_limit', 'rate limit', 'timeout')
                        )
                        if retryable and attempt < MAX_RETRIES - 1:
                            await asyncio.sleep((2 ** attempt) + random.random())
                            continue
                        return None, f'(scoring failed: {type(e).__name__})'
                return None, '(scoring failed: retries exhausted)'

        async def one(idx: int, row: dict):
            got = await asyncio.gather(*(sample(row) for _ in range(max(1, samples))))
            score, rationale = combine_samples(got)
            sample_scores = [s for s, _ in got]
            if persist is not None:
                try:
                    await asyncio.to_thread(persist, idx, score, rationale, sample_scores)
                except Exception:
                    pass  # a cache write must never cost the score
            return idx, score, rationale, sample_scores

        tasks   = [asyncio.create_task(one(i, r)) for i, r in rows]
        results = []
        for fut in asyncio.as_completed(tasks):
            results.append(await fut)
            on_done(len(results))
        return results


def rerank_groups(candidates: pd.DataFrame) -> list[list[int]]:
    """Row indices grouped by (client, topic, matched aspect, market kind).

    In market mode the same aspect can win the same topic under two markets.
    The re-rank prompt carries no market context, so those rows would get
    identical answers — score the pair once and share it.

    The kind IS part of the key: confirmed and unexplored rows are scored by
    different prompts answering different questions, so sharing one score
    across them would stamp a hypothesis score onto a confirmed row."""
    groups: dict[tuple, list[int]] = {}
    for idx, row in candidates.iterrows():
        key = (
            str(row.get('_company_key') or ''),
            # Agency included: one topic_number can belong to two agencies, and
            # blank-numbered rows fall back to titles that repeat across them.
            # Without it two different topics would share one score, written
            # from only the first row's grant_summary.
            str(row.get('broad_agency') or ''),
            str(row.get('agency') or ''),
            str(row.get('topic_number') or row.get('title') or ''),
            str(row.get('aspect_label') or ''),
            str(row.get('market_kind') or ''),
        )
        groups.setdefault(key, []).append(int(idx))
    return list(groups.values())


# ── Re-rank score cache ────────────────────────────────────────────────────
#
# One JSON blob per scored pair, keyed by a hash of exactly what the model saw.
# Per-pair blobs rather than one shared file: concurrent runs by two consultants
# can't clobber each other, and nothing ever has to be read whole.

def rerank_cache_key(row: dict, model: str, samples: int = RERANK_SAMPLES) -> str:
    payload = json.dumps(
        [model, rerank_system(row), rerank_user_message(row), int(samples)],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _cache_blob(storage_client, key: str):
    return storage_client.bucket(ap.BUCKET).blob(f'{RERANK_CACHE_PREFIX}{key}.json')


def _cache_read(storage_client, key: str) -> dict | None:
    try:
        rec = json.loads(_cache_blob(storage_client, key).download_as_bytes())
        return rec if isinstance(rec.get('score'), int) else None
    except Exception:
        return None  # missing, unreadable or malformed: re-score


def _cache_write(storage_client, key: str, rec: dict) -> None:
    _cache_blob(storage_client, key).upload_from_string(
        json.dumps(rec), content_type='application/json'
    )


def _fmt_samples(scores) -> str:
    return ' · '.join('–' if s is None else str(s) for s in scores)


def run_rerank(
    candidates: pd.DataFrame, api_key: str, model: str, progress=None, *,
    storage_client=None, rescore: bool = False, samples: int = RERANK_SAMPLES,
) -> pd.DataFrame:
    """Score every deduped (client, agency, topic, aspect, kind) pair 1-5 and
    write the result onto every row of its group. `progress`, when given, is
    called as progress(done, total) — cache hits count as done up front.

    With a `storage_client`, scores are read from and written to the GCS
    re-rank cache, so the same pair scores identically on every later run.
    `rescore=True` ignores what is cached and overwrites it. A pair is cached
    only when every sample succeeded; a partial or failed pair is used for this
    run but re-scored next time.

    Adds llm_score, llm_rationale, llm_samples (the per-sample scores, e.g.
    '3 · 4 · 4') and llm_cached."""
    groups = rerank_groups(candidates)
    rows   = [(n, candidates.loc[g[0]].to_dict()) for n, g in enumerate(groups)]
    total  = len(rows)
    keys   = {n: rerank_cache_key(r, model, samples) for n, r in rows}

    cached: dict[int, dict] = {}
    if storage_client is not None and not rescore and rows:
        with ThreadPoolExecutor(max_workers=_CACHE_IO_WORKERS) as ex:
            for n, rec in zip(keys, ex.map(lambda k: _cache_read(storage_client, k),
                                           keys.values())):
                if rec is not None:
                    cached[n] = rec

    todo = [(n, r) for n, r in rows if n not in cached]
    hits = len(cached)
    if progress is not None and total:
        progress(hits, total)

    def on_done(done: int) -> None:
        if progress is not None:
            progress(hits + done, total)

    persist = None
    if storage_client is not None:
        def persist(n, score, rationale, sample_scores):
            if score is None or any(s is None for s in sample_scores):
                return
            _cache_write(storage_client, keys[n], {
                'score': int(score), 'rationale': rationale,
                'samples': [int(s) for s in sample_scores], 'model': model,
                'scored_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            })

    results = (
        asyncio.run(rerank_async(todo, api_key, model, on_done, samples, persist))
        if todo else []
    )

    out = candidates.copy()
    out['llm_score']     = 0
    out['llm_rationale'] = ''
    out['llm_samples']   = ''
    out['llm_cached']    = False

    def stamp(n, score, rationale, sample_scores, from_cache):
        for idx in groups[n]:
            # 0 keeps unscored pairs visible but below any usable minimum
            out.at[idx, 'llm_score']     = int(score) if score is not None else 0
            out.at[idx, 'llm_rationale'] = rationale
            out.at[idx, 'llm_samples']   = _fmt_samples(sample_scores)
            out.at[idx, 'llm_cached']    = from_cache

    for n, rec in cached.items():
        stamp(n, rec['score'], str(rec.get('rationale') or ''), rec.get('samples') or [], True)
    for n, score, rationale, sample_scores in results:
        stamp(n, score, rationale, sample_scores, False)
    return out