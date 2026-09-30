"""
Newsletter screening
--------------------
Tags every grant topic that enters the store with whether it belongs in the
vertical newsletter, which of the newsletter's verticals it belongs to, and the
official title of the solicitation it came from.

This is a DIFFERENT question from matching, and the prompt is built around the
difference. Matching asks "is this a great fit for company X?". The newsletter
goes to a mass audience grouped by vertical, and asks "would someone working in
this field be glad they read about this?" — interesting, consequential, worth
knowing about, even for a reader who will not apply. The bar is high on
purpose: an empty vertical is better than a weak item, so the default answer is
no and every vertical has to be earned.

The solicitation title matters because a mass email cannot lean on the
personalised framing the matcher's outreach uses. A reader needs the name the
agency itself published ("Army xTech Search 9", "DoD SBIR 26.1 BAA",
"NSF SBIR/STTR Phase I") to recognise and look up the opportunity, so the
model is told to copy it verbatim from the supplied text and to leave it blank
rather than invent one. Where an ingest path already knows it (the Topic
Importer's extraction, the Funding Source agent reading it off the page), that
value is passed in as a hint and wins over an empty answer.

Called from every ingest path — Topic Importer, SAM.gov CSV upload, Grants.gov
fetch, sam-gov-job and deep-research-job — and from the Newsletter view's
backfill. Streamlit-free and synchronous (a thread pool, not asyncio), so it
runs unchanged in a view, in a job, and on the deep-research job's worker
thread, none of which can share one event loop.

A failed classification never blocks an import: the row is saved with
`newsletter_checked_at` blank, which is what the Newsletter view reads as
"unchecked" and offers to re-run.
"""

import html
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
from anthropic import Anthropic

import src.modules.anthropic_utils as au

# ── Verticals ──────────────────────────────────────────────────────────────
# The names are the newsletter's own section headings and are stored verbatim,
# so they must not be reworded here. The glosses exist only to stop the model
# guessing at the edges (Healthtech vs Medtech, Defensetech vs everything with
# a DoD customer).

VERTICALS = [
    'Advanced Materials & Manufacturing',
    'Aerospace & Space Tech',
    'AI/ML',
    'Biotech',
    'Cleantech & Energy',
    'Cybersecurity',
    'Defensetech & Dual-Use',
    'eXtended Reality',
    'Healthtech',
    'Medtech',
    'Other Tech',
    'Quantum & Photonics',
    'Robotics & Autonomous Systems',
]

_VERTICAL_GLOSS = {
    'Advanced Materials & Manufacturing': 'novel materials, composites, additive/advanced manufacturing, semiconductors fabrication, process innovation',
    'Aerospace & Space Tech':             'aircraft, propulsion, hypersonics, satellites, launch, in-space systems, space domain awareness',
    'AI/ML':                              'where AI/ML is the substance of the work, not a buzzword attached to something else',
    'Biotech':                            'therapeutics, vaccines, synthetic biology, biomanufacturing, genomics, ag-bio',
    'Cleantech & Energy':                 'generation, storage, grid, nuclear, hydrogen, efficiency, carbon management, climate tech',
    'Cybersecurity':                      'defensive/offensive cyber, zero trust, cryptography, secure systems and supply chains',
    'Defensetech & Dual-Use':             'technology built for defense/national-security missions, or commercial tech the DoD is actively seeking to adopt',
    'eXtended Reality':                   'AR, VR, MR, immersive training, spatial computing, digital twins experienced immersively',
    'Healthtech':                         'digital health, health IT, care delivery, public-health data and software',
    'Medtech':                            'medical devices, diagnostics, imaging, surgical and clinical hardware',
    'Other Tech':                         'ONLY a notable tech opportunity that genuinely fits none of the other verticals',
    'Quantum & Photonics':                'quantum computing/sensing/networking, lasers, optics, photonic integrated circuits',
    'Robotics & Autonomous Systems':      'robots, drones/UAS/UUV/UGV, autonomy software, human-machine teaming',
}

MAX_VERTICALS = 3

# ── Models ─────────────────────────────────────────────────────────────────
# Opus 5.5 because the call is an editorial judgment, not a keyword match, and
# its volume (tens to low hundreds of topics a day) makes quality cheap enough.
# Effort is set explicitly: Opus 5.5 defaults to `medium`, which is what this
# wants — stating it keeps a later model swap from silently changing depth.
#
# FALLBACK_MODEL exists for the same reason it exists in strategy_session and
# client-profile-job: biotech topics (dried/powdered biologics especially) trip
# false-positive refusals, and Opus 5.5 widened its bio classifier. A refusal is
# deterministic for the same model and text, so the retry goes to Haiku, which
# answers that material. `output_config.effort` ERRORS on Haiku, so the rescue
# request carries the format but not the effort.

MODEL          = 'claude-opus-5-5'
FALLBACK_MODEL = 'claude-haiku-4-5-20251001'
_EFFORT        = 'medium'
_NO_EFFORT_MODELS = {FALLBACK_MODEL}

_MAX_TOKENS  = 4000     # thinking is always on for Opus 5.5 and counts against this
_WORKERS     = 8
_DESC_CAP    = 7000     # chars of topic text sent; eligibility often sits at the end
_MAX_RETRIES = 4        # SDK-level retries on 429 / 5xx / 529

# ── Stored columns ─────────────────────────────────────────────────────────

COL_GOOD      = 'newsletter_good'        # bool — at least one vertical passed the bar
COL_VERTICALS = 'newsletter_verticals'   # ' | '-joined, most relevant first
COL_TITLE     = 'solicitation_title'     # official title of the parent solicitation
COL_HOOK      = 'newsletter_hook'        # one sentence a reader would get in the email
COL_REASON    = 'newsletter_reason'      # the editorial call, esp. why it was skipped
COL_MODEL     = 'newsletter_model'       # model that answered (Haiku after a refusal)
COL_CHECKED   = 'newsletter_checked_at'  # ISO date; BLANK = never successfully checked

COLUMNS = (COL_GOOD, COL_VERTICALS, COL_TITLE, COL_HOOK, COL_REASON, COL_MODEL, COL_CHECKED)
VERTICAL_SEP = ' | '


# ── Prompt ─────────────────────────────────────────────────────────────────

def _vertical_block() -> str:
    return '\n'.join(f'- {v}: {_VERTICAL_GLOSS[v]}' for v in VERTICALS)


_SYSTEM = f"""\
You are the editor of a vertical newsletter about government and institutional \
funding opportunities — SBIR/STTR, BAAs, CSOs, OTAs, prize challenges, grants, \
consortium project calls. It goes out as a mass email to founders, engineers, \
researchers and investors, grouped by vertical. You decide whether one \
opportunity earns a place in it.

THE TEST
Would a reader who works in this vertical finish the item and think "I'm glad I \
read that"? Not "this is a fit for my company" — most readers will never apply. \
The question is whether it is interesting and consequential to people in the \
field: it signals where money and attention are going, it is unusually large or \
unusually open, it is a new program or a new agency priority, it targets a hard \
or novel technical problem, or it is simply something a well-informed person in \
that field would want to have heard about.

The bar is high. Skipping a vertical is always better than filling it with \
something weak — an empty section costs nothing, a dull one costs readers. \
Your default answer is no; every vertical has to be earned.

Strong signals:
- a new program, a new round of a flagship program, or a first-of-its-kind call
- substantial funding, or a large number of awards
- broad eligibility (small businesses, startups, universities, open to all)
- a clearly stated, technically interesting problem
- a notable agency or mission priority (a new focus area, a strategic initiative)
- a prize challenge or unusual mechanism with real stakes

Disqualifiers — any one of these means no vertical at all:
- routine procurement: supplies, spare parts, commodity equipment, maintenance, \
repair, sustainment, facilities, construction, janitorial, staffing, admin \
services, training delivery, IT help desk, software licences
- sources-sought/RFIs that only survey the market for an existing product, \
sole-source or brand-name justifications, notices of intent to award
- scope so narrow that only an incumbent or a handful of existing vendors could \
care
- eligibility limited to one state, city, institution or pre-selected group, \
unless the program is itself large or notable
- the deadline has already passed, or there is no substantive technical content \
to write about
- anything that is an award already made, news, or an event rather than an \
opportunity

VERTICALS — use these exact names, and only these:
{_vertical_block()}

Choose a vertical only where the opportunity passes the test for readers of that \
vertical specifically. At most {MAX_VERTICALS}, most relevant first. One strong \
vertical is the normal case; more than one only when the opportunity is \
genuinely central to each. A DoD customer alone does not make something \
Defensetech & Dual-Use; the technology has to be the story. Never use Other Tech \
as a place to put something that was weak in its real vertical.

SOLICITATION TITLE
Give the official title of the solicitation or funding mechanism this \
opportunity belongs to, exactly as the issuing organisation publishes it — the \
name a reader would search for. Copy it verbatim from the supplied text (for \
example a BAA, CSO, program announcement, SBIR/STTR release, challenge or \
funding call name, with its number if one is shown). When the opportunity is one \
topic inside a larger solicitation, give the larger solicitation's name, not the \
topic's. When the supplied title already is the solicitation's own published \
title, repeat it unchanged. A bare number is not a title: when the text shows \
only a number such as "HR001126S0010", give the published name with the number \
after it ("DICE — HR001126S0010"), or the supplied title if there is no other \
name. If a "stated solicitation title" is supplied, keep it unless the text \
plainly shows it is wrong. If the text never names the solicitation, return an \
empty string — never invent or paraphrase a title.

OUTPUT
- verticals: the verticals it earns (empty list when it earns none)
- hook: when it earns a vertical, a newsletter blurb of one or two plain \
sentences (at most 45 words) saying what is being funded and why a reader in \
the field should care. Lead with the substance. The deadline and funding amount \
are printed on their own line beside the hook, so mention them only when the \
size itself is part of the story, and never say that they are unknown or tell \
the reader to check the listing. No hype words, no "exciting opportunity". \
Empty string when it earns none.
- reason: one sentence explaining your decision — for a skip, the specific \
reason it fell short.
- solicitation_title: as described above.\
"""

_SYSTEM_BLOCKS = [{
    'type': 'text',
    'text': _SYSTEM,
    # Identical for every topic in a batch, so every call after the first reads
    # it from cache.
    'cache_control': {'type': 'ephemeral'},
}]

_SCHEMA = {
    'type': 'object',
    'properties': {
        'verticals': {
            'type': 'array',
            'items': {'type': 'string', 'enum': VERTICALS},
        },
        'hook':               {'type': 'string'},
        'reason':             {'type': 'string'},
        'solicitation_title': {'type': 'string'},
    },
    'required': ['verticals', 'hook', 'reason', 'solicitation_title'],
    'additionalProperties': False,
}


def _clean(v) -> str:
    if v is None:
        return ''
    s = str(v).strip()
    return '' if s.lower() in ('nan', 'none', 'nat', 'null') else s


_INVISIBLE = dict.fromkeys(map(ord, '​‌‍⁠﻿'), None)


def clean_title(v) -> str:
    """A title fit to print in an email: HTML entities decoded (Grants.gov titles
    arrive as '&#8203;Mitigating…'), zero-width characters dropped, whitespace
    collapsed."""
    s = html.unescape(_clean(v)).translate(_INVISIBLE)
    # U+FFFD in a stored title is a dash lost to a Windows-1252 round-trip
    # upstream ("Drone Dominance Program � Phase III") — never printable as-is.
    s = s.replace('�', '–')
    return ' '.join(s.split())


def _user_message(topic: dict, today: str) -> str:
    """One topic as the model sees it. Fields are read defensively because the
    five ingest paths name them slightly differently."""
    desc = _clean(topic.get('description')) or _clean(topic.get('grant_summary'))
    summary = _clean(topic.get('grant_summary'))
    lines = [
        f"Today's date: {today}",
        f"Title: {_clean(topic.get('title'))}",
        f"Agency / issuer: {_clean(topic.get('agency'))}",
        f"Topic / solicitation number: {_clean(topic.get('topic_number'))}",
        f"Deadline: {_clean(topic.get('due_date')) or _clean(topic.get('close_date'))}",
        f"Funding: {_clean(topic.get('funding_amount')) or _clean(topic.get('award_ceiling'))}",
        f"Source: {_clean(topic.get('source'))}",
    ]
    hint = _clean(topic.get('solicitation_title_hint'))
    if hint:
        lines.append(f'Stated solicitation title: {hint}')
    if len(desc) > _DESC_CAP:
        # Keep both ends: the scope leads, eligibility and awards often trail.
        half = _DESC_CAP // 2
        desc = desc[:half] + '\n[…]\n' + desc[-half:]
    lines.append(f'\nFull text:\n{desc}')
    if summary and summary != desc:
        lines.append(f'\nTechnical summary:\n{summary[:2000]}')
    return '\n'.join(lines)


# ── Classification ─────────────────────────────────────────────────────────

def _request(client: Anthropic, model: str, message: str):
    output_config = {'format': {'type': 'json_schema', 'schema': _SCHEMA}}
    if model not in _NO_EFFORT_MODELS:
        output_config['effort'] = _EFFORT
    return client.messages.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        system=_SYSTEM_BLOCKS,
        messages=[{'role': 'user', 'content': message}],
        output_config=output_config,
    )


def _normalise(parsed: dict, hint: str, model: str, today: str) -> dict:
    verticals, seen = [], set()
    for v in parsed.get('verticals') or []:
        # The schema enum already guarantees the spelling; this only dedups and
        # caps, since the enum cannot express either.
        if v in _VERTICAL_GLOSS and v not in seen:
            verticals.append(v)
            seen.add(v)
    verticals = verticals[:MAX_VERTICALS]
    title = clean_title(parsed.get('solicitation_title')) or clean_title(hint)
    return {
        COL_GOOD:      bool(verticals),
        COL_VERTICALS: VERTICAL_SEP.join(verticals),
        COL_TITLE:     title,
        COL_HOOK:      _clean(parsed.get('hook')) if verticals else '',
        COL_REASON:    _clean(parsed.get('reason')),
        COL_MODEL:     model,
        COL_CHECKED:   today,
    }


def unchecked(topic: dict, error: str = '') -> dict:
    """The columns for a topic that could not be classified. `newsletter_checked_at`
    stays blank, so the Newsletter view lists it as unchecked and can retry it."""
    return {
        COL_GOOD:      False,
        COL_VERTICALS: '',
        COL_TITLE:     clean_title(topic.get('solicitation_title_hint')),
        COL_HOOK:      '',
        COL_REASON:    f'Not checked: {error}'[:500] if error else '',
        COL_MODEL:     '',
        COL_CHECKED:   '',
    }


def classify_one(client: Anthropic, topic: dict, today: str | None = None) -> dict:
    """Classify one topic. Never raises — a failure comes back as `unchecked()`."""
    today = today or datetime.today().strftime('%Y-%m-%d')
    hint = _clean(topic.get('solicitation_title_hint'))
    message = _user_message(topic, today)
    try:
        model = MODEL
        resp = _request(client, model, message)
        if resp.stop_reason == 'refusal':
            model = FALLBACK_MODEL
            resp = _request(client, model, message)
        if resp.stop_reason == 'max_tokens':
            return unchecked(topic, f'{model} hit max_tokens')
        parsed = json.loads(au.response_text(resp))
        return _normalise(parsed, hint, model, today)
    except Exception as e:                                    # noqa: BLE001
        return unchecked(topic, f'{type(e).__name__}: {e}')


def classify_topics(
    topics: list[dict],
    api_key: str,
    progress=None,
    workers: int = _WORKERS,
) -> list[dict]:
    """Classify many topics concurrently, preserving order.

    `topics` are plain dicts carrying any of title / agency / topic_number /
    description / grant_summary / due_date / close_date / funding_amount /
    source, plus an optional `solicitation_title_hint`. Returns one dict of the
    COLUMNS per topic. `progress(done, total)` is called as each finishes so the
    caller owns the widget or the log line.
    """
    if not topics:
        return []
    client = Anthropic(api_key=api_key, max_retries=_MAX_RETRIES)
    today = datetime.today().strftime('%Y-%m-%d')
    out: list[dict | None] = [None] * len(topics)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(classify_one, client, t, today): i for i, t in enumerate(topics)}
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result()
            done += 1
            if progress:
                try:
                    progress(done, len(topics))
                except Exception:                              # noqa: BLE001
                    pass
    return out


def apply_to_frame(df, results: list[dict]):
    """Write classification results onto a DataFrame, row-aligned. Returns df."""
    for col in COLUMNS:
        df[col] = [r.get(col, '' if col != COL_GOOD else False) for r in results]
    return df


def tag_frame(df, api_key: str, progress=None, title_hints=None):
    """Classify every row of a topics DataFrame and add the newsletter columns.

    `title_hints` is an optional per-row list of solicitation titles the ingest
    path already knows. A failure of the whole call (bad key, no network) is
    swallowed into unchecked rows: newsletter tagging must never cost an import.
    """
    records = df.to_dict('records')
    if title_hints is not None:
        for rec, hint in zip(records, title_hints):
            rec['solicitation_title_hint'] = hint
    elif COL_TITLE in df.columns:
        for rec in records:
            rec['solicitation_title_hint'] = rec.get(COL_TITLE)
    try:
        results = classify_topics(records, api_key, progress=progress)
    except Exception as e:                                    # noqa: BLE001
        results = [unchecked(r, f'{type(e).__name__}: {e}') for r in records]
    return apply_to_frame(df, results)


def parse_verticals(text) -> tuple[list[str], list[str]]:
    """A hand-edited ' | '-joined vertical string -> (canonical names, unknown
    names). Matching is case- and whitespace-insensitive; commas also split, so
    'AI/ML, Biotech' works."""
    canon = {v.lower(): v for v in VERTICALS}
    good, bad = [], []
    for part in str(text or '').replace(',', '|').split('|'):
        p = ' '.join(part.split())
        if not p:
            continue
        v = canon.get(p.lower())
        if v is None:
            bad.append(p)
        elif v not in good:
            good.append(v)
    return good, bad


# ── Consultant review + write-back ─────────────────────────────────────────

COL_REVIEWED_BY = 'newsletter_reviewed_by'
COL_REVIEWED_AT = 'newsletter_reviewed_at'
REVIEW_COLUMNS  = (COL_REVIEWED_BY, COL_REVIEWED_AT)


def write_back(gcs, bucket: str, updates: dict) -> tuple[int, list[str]]:
    """Rewrite newsletter columns onto stored topic rows, in place.

    `updates` maps blob path -> list of (row_position, expected_title, {col: value}).
    Each parquet is re-read from GCS immediately before the write, and every
    row is re-located by its title: a file rewritten since it was loaded (a
    SAM.gov revision, another consultant's save) must not have values landed on
    the wrong row. A row whose title is not at its old position is looked up by
    title; if that is not unique it is skipped and reported, never guessed.

    Returns (rows written, error strings). Never raises for one bad blob.
    """
    written, errors = 0, []
    for path, items in updates.items():
        try:
            blob = gcs.bucket(bucket).blob(path)
            df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
            titles = df['title'].astype(str) if 'title' in df.columns else pd.Series([''] * len(df))
            n_here = 0
            for pos, expected, values in items:
                if not (0 <= pos < len(df) and titles.iloc[pos] == expected):
                    hits = titles.index[titles == expected].tolist()
                    if len(hits) != 1:
                        errors.append(f'{path}: could not locate “{expected[:80]}” — skipped')
                        continue
                    pos = df.index.get_loc(hits[0])
                for col, val in values.items():
                    if col not in df.columns:
                        df[col] = False if col == COL_GOOD else ''
                    df.iat[pos, df.columns.get_loc(col)] = val
                n_here += 1
            if n_here:
                buf = io.BytesIO()
                df.to_parquet(buf, index=False)
                buf.seek(0)
                blob.upload_from_file(buf, content_type='application/octet-stream')
                written += n_here
        except Exception as e:                                # noqa: BLE001
            errors.append(f'{path}: {type(e).__name__}: {e}')
    return written, errors


def summarize(df) -> dict:
    """Counts off a tagged frame, for a status payload / log line / banner."""
    if df is None or len(df) == 0 or COL_CHECKED not in df.columns:
        return {'newsletter_checked': 0, 'newsletter_good': 0, 'newsletter_failed': 0}
    checked = df[COL_CHECKED].fillna('').astype(str).str.strip() != ''
    good = checked & df[COL_GOOD].fillna(False).astype(bool)
    return {
        'newsletter_checked': int(checked.sum()),
        'newsletter_good':    int(good.sum()),
        'newsletter_failed':  int((~checked).sum()),
    }
