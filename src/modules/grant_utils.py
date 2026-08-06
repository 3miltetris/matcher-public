"""
grant_utils.py — shared helpers for normalizing grant topic DataFrames.

The canonical column for grant text is `grant_summary`. Different import paths
write it under different names (`description`, `summary`). Call
`normalize_grant_columns` whenever a topics DataFrame is loaded so all
downstream code can unconditionally reference `grant_summary`.
"""

import pandas as pd


def normalize_grant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure `grant_summary` is always present and populated.

    - If only `description` exists: rename it to `grant_summary`.
    - If both exist: fill any empty `grant_summary` values from `description`.
    """
    if df.empty:
        return df

    if 'grant_summary' not in df.columns:
        if 'description' in df.columns:
            df = df.rename(columns={'description': 'grant_summary'})
    elif 'description' in df.columns:
        mask = df['grant_summary'].isna() | (df['grant_summary'].astype(str).str.strip() == '')
        df.loc[mask, 'grant_summary'] = df.loc[mask, 'description']

    return df
