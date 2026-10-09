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

import pandas as pd

# Rolling-deadline tracking columns and their defaults. `is_rolling` flags a
# continuously-open solicitation; `last_verified_active` is the ISO date a source
# last confirmed it still open; `verify_status` is active/inactive/unverified.
ROLLING_DEFAULTS: dict = {
    'is_rolling':           False,
    'last_verified_active': '',
    'verify_status':        'unverified',
}


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


def drop_dead_topics(df: pd.DataFrame) -> pd.DataFrame:
    """Drop topics that are no longer live — archived SAM.gov notices and any
    topic a re-check has marked `verify_status == 'inactive'` (an expired rolling
    deadline). Unifies the auto-expire filter across every source. Does not reset
    the index — callers add `.reset_index(drop=True)` where they need it, matching
    their prior behaviour.
    """
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    if 'sam_status' in df.columns:
        mask &= df['sam_status'].fillna('').astype(str) != 'archived'
    if 'verify_status' in df.columns:
        mask &= df['verify_status'].fillna('').astype(str) != 'inactive'
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

    df = ensure_rolling_columns(df)
    return df
