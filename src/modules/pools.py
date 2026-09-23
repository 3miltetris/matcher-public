"""
Company pools
-------------
The pipeline holds two directories of companies that get the same treatment —
Deep Research, Fathom meetings, capability profiles, aspect matching, HubSpot
export — and differ only in where they live and what may feed them:

    clients    data/all-contacts/clients/    + data/client-profiles/profiles.parquet
    prospects  data/all-contacts/prospects/  + data/client-profiles/prospect_profiles.parquet

Clients are companies we work for; prospects are companies we are targeting.
A prospect has no Google Drive folder (Drive Sync is client-only), and is
promoted into the client pool when it signs — see src/modules/pool_transfer.py.

**Separate prefixes and separate profile blobs, not one store with a `pool`
column.** Every consumer enumerates the store it loads, so a shared store makes
filtering something each reader has to remember, and the reader that forgets
silently mixes prospects into client work. That is the trap documented for
data/all-topics/awards/. Here the pool is a keyword argument that defaults to
'clients', so it cannot be forgotten at a call site and every pre-pool caller
keeps its exact previous behaviour.

**Column convention.** Files under clients/ use `company_name` / `summary`,
while the Contact Importer's lead parquets use `companyName` /
`company_summary`. Prospect parquets are written in the clients convention (see
jobs/contact_import_job.py), and load_frames() normalizes anything that is not,
so every pool-aware view and job reads one spelling.

Streamlit-free — client-profile-job, contact-import-job and fathom-sync-job all
import it. The Streamlit-side selector is ui_common.pool_selector().
"""

import io

import pandas as pd

import src.modules.aspect_profile as ap

# ── Constants ───────────────────────────────────────────────────────────────

BUCKET = ap.BUCKET

CLIENTS   = ap.POOL_CLIENTS
PROSPECTS = ap.POOL_PROSPECTS
DEFAULT   = ap.DEFAULT_POOL

CLIENTS_PREFIX   = ap.CLIENTS_PREFIX
PROSPECTS_PREFIX = 'data/all-contacts/prospects/'

POOLS: dict[str, dict] = {
    CLIENTS: {
        'key':             CLIENTS,
        'label':           'Clients',
        'icon':            '🏢',
        'noun':            'client',
        'contacts_prefix': CLIENTS_PREFIX,
        'profiles_blob':   ap.profiles_blob(CLIENTS),
        'supports_drive':  True,
        'blurb':           'Companies we write proposals for.',
    },
    PROSPECTS: {
        'key':             PROSPECTS,
        'label':           'Prospects',
        'icon':            '🎯',
        'noun':            'prospect',
        'contacts_prefix': PROSPECTS_PREFIX,
        'profiles_blob':   ap.profiles_blob(PROSPECTS),
        # No Drive folder until a prospect signs — Drive Sync stays client-only.
        'supports_drive':  False,
        'blurb':           'Targeted companies we are pursuing, fed by research, '
                           'scraping and meetings.',
    },
}

POOL_KEYS = list(POOLS)


# ── Registry lookups ────────────────────────────────────────────────────────

def pool(key: str | None = None) -> dict:
    """The pool record. An unrecognised key falls back to the default pool
    rather than raising: these values come from session state and job configs,
    and a stale one must not take a page or a run down."""
    return POOLS.get(str(key or ''), POOLS[DEFAULT])


def is_pool(key: str | None) -> bool:
    return str(key or '') in POOLS


def label(key: str | None = None) -> str:
    return pool(key)['label']


def noun(key: str | None = None) -> str:
    return pool(key)['noun']


def display(key: str | None = None) -> str:
    """'🎯 Prospects' — the selector/label spelling used across the views."""
    p = pool(key)
    return f"{p['icon']} {p['label']}"


def contacts_prefix(key: str | None = None) -> str:
    return pool(key)['contacts_prefix']


def profiles_blob(key: str | None = None) -> str:
    return pool(key)['profiles_blob']


def supports_drive(key: str | None = None) -> bool:
    return pool(key)['supports_drive']


def pool_of_prefix(prefix: str) -> str | None:
    """Which pool a contacts prefix belongs to, or None for a plain lead
    source folder. Used when a write has to land back in the file it came
    from and the pool is not otherwise known."""
    for key, p in POOLS.items():
        if p['contacts_prefix'] == prefix:
            return key
    return None


# ── Column convention ───────────────────────────────────────────────────────

# canonical column -> the lead-import spelling it may arrive under
COLUMN_ALIASES = {
    'company_name': 'companyName',
    'summary':      'company_summary',
}


def normalize_company_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Give a contact frame the clients-convention columns.

    The alias is *renamed*, not copied, so a frame never carries two spellings
    of the same value — an edit written back to one of them would otherwise
    leave the other stale, and jobs/matching_job.py reads whichever it finds.
    Frames already in the clients convention come back untouched.
    """
    if df is None or df.empty:
        return df
    out = df
    for canonical, alias in COLUMN_ALIASES.items():
        if alias not in out.columns:
            continue
        if out is df:
            out = df.copy()
        if canonical not in out.columns:
            out = out.rename(columns={alias: canonical})
            continue
        blank = out[canonical].isna() | (
            out[canonical].astype(str).str.strip().isin(['', 'None', 'nan'])
        )
        out.loc[blank, canonical] = out.loc[blank, alias]
        out = out.drop(columns=[alias])
    return out


def company_key(row) -> str:
    """name||website — the identity every pool-aware view joins on."""
    return ap.company_key(row)


def key_mask(df: pd.DataFrame, key: str) -> pd.Series:
    """Rows of a contact frame belonging to one company key."""
    name, _, website = str(key).partition('||')
    names = (df.get('company_name', pd.Series('', index=df.index))
               .fillna('').astype(str).str.strip())
    if 'company_name' not in df.columns and 'companyName' in df.columns:
        names = df['companyName'].fillna('').astype(str).str.strip()
    sites = (df.get('companyWebsite', pd.Series('', index=df.index))
               .fillna('').astype(str).str.strip())
    return (names == name.strip()) & (sites == website.strip())


# ── Frame loading ───────────────────────────────────────────────────────────

def load_frames(
    gcs_client, key: str = DEFAULT, bucket: str = BUCKET
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """({blob_name: frame}, errors) for one pool.

    Frames are kept per blob, not concatenated, because every write path in
    this codebase rewrites the exact file a row came from."""
    frames: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for blob in gcs_client.list_blobs(bucket, prefix=contacts_prefix(key)):
        if not blob.name.endswith('.parquet'):
            continue
        try:
            frames[blob.name] = normalize_company_columns(
                pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
            )
        except Exception as e:
            errors.append(f'{blob.name}: {e}')
    return frames, errors


def load_all_frames(
    gcs_client, keys=None, bucket: str = BUCKET
) -> tuple[dict[str, dict[str, pd.DataFrame]], list[str]]:
    """{pool_key: {blob_name: frame}} for several pools in one pass — for the
    jobs that must resolve a company key without being told which pool it is
    in (fathom-sync-job attributes a meeting domain across both)."""
    out: dict[str, dict[str, pd.DataFrame]] = {}
    errors: list[str] = []
    for key in (keys or POOL_KEYS):
        frames, errs = load_frames(gcs_client, key, bucket=bucket)
        out[key] = frames
        errors.extend(errs)
    return out, errors


def company_names(frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    """company_key -> company_name for every company in the loaded frames."""
    out: dict[str, str] = {}
    for df in frames.values():
        if df is None or df.empty or 'company_name' not in df.columns:
            continue
        names = df['company_name'].fillna('').astype(str).str.strip()
        sites = (df.get('companyWebsite', pd.Series('', index=df.index))
                 .fillna('').astype(str).str.strip())
        for name, site in zip(names, sites):
            if name:
                out[f'{name}||{site}'] = name
    return out


def combined_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every row of a pool in one frame, with `_blob` naming the file it came
    from and `_key` the company key. Empty frame when the pool has no files."""
    parts = []
    for blob_name, df in frames.items():
        if df is None or df.empty:
            continue
        part = df.copy()
        part['_blob'] = blob_name
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    combined['_key'] = combined.apply(company_key, axis=1)
    return combined
