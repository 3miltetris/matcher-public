"""
grant_utils.py — shared helpers for normalizing grant topic DataFrames.

The canonical column for grant text is `grant_summary`. Different import paths
write it under different names (`description`, `summary`). Call
`normalize_grant_columns` whenever a topics DataFrame is loaded so all
downstream code can unconditionally reference `grant_summary`.

This module also owns the rolling-deadline tracking columns (Stage: rolling
deadlines). Continuously-open solicitations (BAAs, CSOs, the *werx/OTA/consortium
challenges, some Grants.gov forecasts) have no usable deadline, so a separate set
of columns records whether a topic is rolling, when a source last confirmed it
still open, and whether a re-check has since found it gone. Keeping these in one
place (added on every read via `ensure_rolling_columns`, filtered on via
`drop_dead_topics`) stops the defaults and the auto-expire rule from drifting
across the many readers and writers that touch grant topics.
"""

import re
from datetime import date, datetime

import pandas as pd

# Rolling-deadline tracking columns and their defaults. `is_rolling` flags a
# continuously-open solicitation; `last_verified_active` is the ISO date a source
# last confirmed it still open; `verify_status` is active/inactive/unverified.
ROLLING_DEFAULTS: dict = {
    'is_rolling':           False,
    'last_verified_active': '',
    'verify_status':        'unverified',
}

# Stored `due_date`/`close_date` are a genuine mix of ISO and mm/dd/yyyy — the
# same formats the Bulk Matching / Grant Search / matching-job parsers accept.
_DATE_FORMATS = ('%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%m-%d-%Y', '%b %d, %Y')

# Tokens that mean "no real date" when they land in a date column. `astype(str)`
# on a missing value yields the literal 'nan'/'NaT'/'<NA>'; the others are what
# sources write for a continuously-open solicitation.
_BLANK_DATE_TOKENS = {'', 'nan', 'nat', 'none', 'null', '<na>'}
_ROLLING_TOKENS = {'rolling', 'continuous', 'open until filled', 'n/a', 'tbd',
                   'see solicitation'}


def clean_date_display(value) -> str:
    """Return '' for a blank/junk date token ('nan', 'NaT', 'none', …), else the
    stripped original. Keeps the literal string 'nan' from `astype(str)` out of
    anything a reader renders as a published deadline."""
    s = str(value).strip() if value is not None else ''
    return '' if s.lower() in _BLANK_DATE_TOKENS else s


def parse_deadline(value):
    """Best-effort `datetime.date` from a stored deadline string; `None` for a
    blank, a rolling token ('open until filled', 'TBD', …), or anything that
    doesn't match a known date format. Supersedes the SAM-only `_iso_date`."""
    s = clean_date_display(value)
    if not s or s.lower() in _ROLLING_TOKENS:
        return None
    s = s[:10] if (len(s) > 10 and s[4] == '-') else s  # ISO datetime → date part
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _row_get(row, key, default=''):
    """Read a key from a dict or a pandas row, treating NaN as absent."""
    try:
        v = row.get(key, default) if hasattr(row, 'get') else row[key]
    except (KeyError, IndexError, TypeError):
        return default
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    return v


def topic_open_status(row, today=None) -> str:
    """Classify one topic as 'rolling' / 'expired' / 'open' / 'unknown'.

    - 'rolling'  — `is_rolling` is truthy, or the deadline is a rolling token.
    - 'expired'  — a parseable deadline strictly before `today`, and not rolling.
    - 'open'     — a parseable deadline on/after `today`.
    - 'unknown'  — no parseable deadline (kept, never dropped: `is_rolling` is an
                   unreliable *negative* signal, so an undated topic is not
                   assumed expired).

    A completed award (`record_kind == 'award'`) is never 'expired'. Accepts a
    dict or a pandas row.
    """
    today = today or date.today()
    if str(_row_get(row, 'record_kind')).strip().lower() == 'award':
        return 'rolling' if _is_truthy(_row_get(row, 'is_rolling', False)) else 'unknown'
    if _is_truthy(_row_get(row, 'is_rolling', False)):
        return 'rolling'
    raw = _row_get(row, 'due_date') or _row_get(row, 'close_date')
    if str(raw).strip().lower() in _ROLLING_TOKENS:
        return 'rolling'
    d = parse_deadline(raw)
    if d is None:
        return 'unknown'
    return 'expired' if d < today else 'open'


def _is_truthy(v) -> bool:
    """Truthiness for an `is_rolling` cell that may be a bool, a numpy bool, or a
    string like 'True'/'False'/'' read back from parquet or a CSV."""
    if isinstance(v, str):
        return v.strip().lower() in ('true', '1', 'yes')
    return bool(v)


# ── Record kind: fundable R&D solicitation vs procurement vs completed award ──
#
# A coworker review found the matcher's top hits dominated by SAM.gov *supply
# buys* (buying a product, not funding R&D), *sole-source / notice-of-intent-to-
# award* notices, and already-awarded contracts — all topically similar to the
# client but none of them a fundable opportunity. `record_kind` splits a topic
# three ways so every reader can filter and so the output names what each row is:
#
#   'solicitation' — a fundable R&D opportunity (the default, and the only kind
#                    matched by default)
#   'procurement'  — a commodity/supply buy, sources-sought market survey,
#                    sole-source or intent-to-award notice; filtered from matching
#                    unless explicitly opted in
#   'award'        — a contract already awarded (intelligence only, never bid on)
#
# The classifier is deliberately *conservative* — it fires only on strong,
# specific signals, because dropping a real solicitation is worse than leaving a
# stray supply buy for the re-rank safety net to score down. The persisted
# `notice_type`/`set_aside` fields (written going forward) and the screen-time
# disqualifiers are the primary defense; this is the layer that works on rows
# already in the store, which no longer carry those fields.
RECORD_KINDS = ('solicitation', 'procurement', 'award')

# Strong procurement signals in a title/summary. Sole-source and intent-to-award
# notices are not competable; a sources-sought is a market survey, not a funded
# call; BPAs/RFQs are purchasing vehicles. Mirrors the disqualifier language in
# `src/modules/newsletter.py`.
_PROCUREMENT_RE = re.compile(
    r'\b(sole[\s-]?source'
    r'|notice of intent to (?:award|sole[\s-]?source)'
    r'|intent to (?:award|sole[\s-]?source)'
    r'|brand[\s-]?name'
    r'|sources sought'
    r'|blanket purchase agreement'
    r'|request for quote|\brfq\b)\b',
    re.IGNORECASE,
)
# SAM.gov product/supply notices title as a 4-digit PSC code then '--'
# (e.g. '6505--Radiopharmaceuticals' — 65xx are medical/dental supplies). R&D
# service codes are alphanumeric, so a leading 4-digit numeric code is a reliable
# supply-buy signal. The VA supply buys the review flagged all match this. The
# `(?!\d{4}\b)` guard excludes a year-range title ('2024--2025 Advanced
# Manufacturing …'), which is a date, not a PSC code.
_PSC_SUPPLY_RE = re.compile(r'^\s*\d{4}\s*--\s*(?!\d{4}\b)')

# A procurement phrase that is explicitly negated — "this is NOT a sources-sought
# notice", "unlike a typical RFQ…" — is not a procurement signal. Python `re` has
# no variable-width lookbehind, so check the short run of text before each match
# for a negation token instead.
_NEGATION_RE = re.compile(
    r"\b(?:not|isn['’]?t|aren['’]?t|unlike|rather than|as opposed to"
    r"|no longer|other than)\b",
    re.IGNORECASE,
)


def _procurement_hit(text: str) -> bool:
    """True if `text` contains a procurement phrase that is not immediately
    negated. Guards the one false-positive class that matters here: an LLM-written
    `grant_summary` that names a procurement term only to say the notice is *not*
    one (checked only on prose; the structured `notice_type` check stays direct)."""
    for m in _PROCUREMENT_RE.finditer(text):
        prefix = text[max(0, m.start() - 40):m.start()]
        if _NEGATION_RE.search(prefix):
            continue
        return True
    return False


def _nonempty(v) -> bool:
    """A field is present if it is not None/NaN and not a blank/junk string."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    s = str(v).strip().lower()
    return s not in _BLANK_DATE_TOKENS and s not in ('0', '0.0')


def classify_record_kind(row, default: str = 'solicitation') -> str:
    """Classify one topic as 'solicitation' / 'procurement' / 'award' from its
    stored fields (accepts a dict or a pandas row). Conservative by design — see
    the module comment above. An existing `record_kind == 'award'` is preserved.
    """
    if str(_row_get(row, 'record_kind')).strip().lower() == 'award':
        return 'award'
    # ── award: a contract already made ──
    if (_nonempty(_row_get(row, 'awardee_name'))
            or _nonempty(_row_get(row, 'award_amount'))
            or _nonempty(_row_get(row, 'award_number'))
            or str(_row_get(row, 'sam_status')).strip().lower() == 'awarded'):
        return 'award'
    nt = (f"{_row_get(row, 'notice_type')} {_row_get(row, 'base_type')}").lower()
    if 'award' in nt:
        return 'award'
    # ── procurement: a buy, a market survey, a sole-source notice ──
    title = str(_row_get(row, 'title'))
    if _PSC_SUPPLY_RE.match(title):
        return 'procurement'
    text = ' '.join(str(_row_get(row, k)) for k in
                    ('title', 'grant_summary', 'sam_reason'))
    if _procurement_hit(text) or 'sources sought' in nt:
        return 'procurement'
    return default


def add_record_kind(df: pd.DataFrame, default: str = 'solicitation') -> pd.DataFrame:
    """Set/refine the `record_kind` column in place via `classify_record_kind`,
    row by row. `default` is the kind for a topic with no procurement/award
    signal — 'solicitation' for the processed store, 'award' for the awards
    store (the awards loader passes default='award' so every awards row stays an
    award even without an awardee field). No-op on an empty frame.
    """
    if df.empty:
        return df
    df['record_kind'] = [classify_record_kind(r, default=default)
                         for _, r in df.iterrows()]
    return df


def ensure_rolling_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing rolling-deadline column with its default, in place.

    Mirrors the backfill pattern already used for the revision-tracking columns
    in `jobs/sam_gov_job.py`, so legacy parquets written before these columns
    existed load cleanly. No-op on an empty frame.
    """
    if df.empty:
        return df
    for col, default in ROLLING_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    return df


def _expired_mask(df: pd.DataFrame, today) -> pd.Series:
    """Vectorized `topic_open_status(...) == 'expired'` over a whole frame: a
    parseable deadline strictly before `today`, on a topic that is neither a
    completed award nor rolling. Mirrors `topic_open_status`'s expired branch
    exactly (award check first, then rolling, then the raw `due_date or
    close_date` coalesce, then the rolling-token guard, then the date compare)
    so the single-row and bulk paths cannot drift. The `strptime` loop runs once
    per *unique* deadline string, not once per row, which is the whole point —
    every reader calls this on the full processed store.
    """
    idx = df.index

    # Awards and rolling topics are never 'expired'.
    is_award = (df['record_kind'].astype(str).str.strip().str.lower() == 'award'
                if 'record_kind' in df.columns else pd.Series(False, index=idx))
    # NaN `is_rolling` → False (mirrors `_row_get`, which maps NaN to its default);
    # a bare `_is_truthy(nan)` would be True, since `bool(float('nan'))` is True.
    def _rolling_cell(v) -> bool:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        return _is_truthy(v)
    rolling = (df['is_rolling'].map(_rolling_cell).astype(bool)
               if 'is_rolling' in df.columns else pd.Series(False, index=idx))

    # raw = `due_date or close_date` with NaN→'' (mirrors the two `_row_get`s):
    # due_date where it is a non-empty value, else close_date.
    def _col(name):
        if name not in df.columns:
            return pd.Series([''] * len(idx), index=idx)
        return df[name].where(df[name].notna(), '')
    due = _col('due_date')
    raw = due.where(due.map(bool), _col('close_date'))

    # Parse each distinct deadline once; a rolling token or an unparseable value
    # is not expired (→ False), matching topic_open_status.
    def _deadline_expired(value) -> bool:
        if str(value).strip().lower() in _ROLLING_TOKENS:
            return False
        d = parse_deadline(value)
        return d is not None and d < today
    per_value = {v: _deadline_expired(v) for v in raw.unique()}
    deadline_expired = raw.map(per_value).astype(bool)

    return deadline_expired & ~is_award & ~rolling


def drop_dead_topics(df: pd.DataFrame, today=None,
                     drop_expired: bool = True) -> pd.DataFrame:
    """Drop topics that are no longer live, across every reader (single chokepoint):

    - archived SAM.gov notices (`sam_status == 'archived'`),
    - topics a re-check marked `verify_status == 'inactive'`,
    - **provably-expired** topics — a parseable deadline strictly before today
      that is not rolling (`topic_open_status == 'expired'`) — only when
      `drop_expired` is True.

    The expiry drop is deliberately anchored on a parseable past deadline, never
    on `is_rolling == False` (an unreliable negative). An undated/unparseable or
    rolling topic is kept, as are completed awards. Does not reset the index —
    callers add `.reset_index(drop=True)` where they need it.

    `drop_expired` defaults True: the match readers (Grant Search, Aspect Match,
    Bulk Matching, matching-job, the MCP search) want only live opportunities.
    The Newsletter passes `drop_expired=False` so a just-scraped item with an
    already-past deadline still reaches the human review rather than vanishing —
    its prior behaviour, before the expiry drop was folded into this helper.
    """
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    if 'sam_status' in df.columns:
        mask &= df['sam_status'].fillna('').astype(str) != 'archived'
    if 'verify_status' in df.columns:
        mask &= df['verify_status'].fillna('').astype(str) != 'inactive'
    if drop_expired and ('due_date' in df.columns or 'close_date' in df.columns):
        today = today or date.today()
        mask &= ~_expired_mask(df, today)
    return df[mask]


def normalize_grant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure `grant_summary` is always present and populated, and that the
    rolling-deadline columns exist.

    - If only `description` exists: rename it to `grant_summary`.
    - If both exist: fill any empty `grant_summary` values from `description`.
    - Always backfill the `ROLLING_DEFAULTS` columns if missing.
    """
    if df.empty:
        return df

    if 'grant_summary' not in df.columns:
        if 'description' in df.columns:
            df = df.rename(columns={'description': 'grant_summary'})
    elif 'description' in df.columns:
        mask = df['grant_summary'].isna() | (df['grant_summary'].astype(str).str.strip() == '')
        df.loc[mask, 'grant_summary'] = df.loc[mask, 'description']

    # Scrub the literal 'nan'/'NaT' that `astype(str)` on a missing deadline
    # writes, so no reader renders it as a published date. Blanks it in place;
    # the rolling columns still carry the open/rolling signal.
    for _col in ('due_date', 'close_date'):
        if _col in df.columns:
            df[_col] = df[_col].map(clean_date_display)

    # Guarantee a `source` column so the solicitation link survives a multi-parquet
    # concat instead of arriving as NaN for importers that never wrote it (a
    # coworker found it blank for every SAM.gov supply-buy row). Blank the same
    # junk tokens, so a missing source reads as '' rather than 'nan'/'None'.
    if 'source' not in df.columns:
        df['source'] = ''
    else:
        df['source'] = df['source'].map(
            lambda v: '' if str(v).strip().lower() in _BLANK_DATE_TOKENS
            else str(v).strip())

    df = ensure_rolling_columns(df)
    return df
