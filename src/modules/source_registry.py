"""
source_registry.py — the master list of funding-source websites (Stage 10).

Streamlit-free, shared by ``views/funding_sources.py`` and
``jobs/deep_research_job.py`` — the same split ``aspect_profile`` uses for the
client-profile store.

The registry replaces the hand-maintained "Active Approved Programs — Sources of
Funding" Google Sheet. One row per site, one blob:

    deep-research-configs/sources.parquet

A companion per-source "seen index" lives at

    deep-research-configs/seen/{source_id}.json

and records every opportunity already extracted from that site. It does two
jobs: it stops a still-open solicitation from being re-imported every week, and
its titles are fed back into the agent's prompt so the agent skips known items
instead of paying to re-extract them.

Nothing here imports streamlit, and no function builds its own GCS client —
every one takes an already-built ``storage.Client`` so the view can pass a
service-account client and the job can pass an ADC one.
"""

import io
import json
import re
import secrets
from datetime import datetime, timedelta

import pandas as pd

BUCKET        = 'cc-matcher-bucket-jeg-v1'
SOURCES_BLOB  = 'deep-research-configs/sources.parquet'
SEEN_PREFIX   = 'deep-research-configs/seen/'
CONFIG_PREFIX = 'deep-research-configs/'
STATUS_PREFIX = 'deep-research-jobs/'

# Cadence -> how stale `last_checked` may get before the site is due again.
# `daily` is 0 so a daily site is due on every run; `paused` is never due.
CADENCE_DAYS = {'daily': 0, 'weekly': 6, 'monthly': 27}
CADENCES     = ['daily', 'weekly', 'monthly', 'paused']

# A site that errors this many times in a row is paused rather than retried
# forever. Login-walled sites (the LinkedIn rows, gocolosseum) hit this fast,
# which is the point — an unattended daily run must not burn budget on them.
MAX_CONSECUTIVE_FAILURES = 3

DEFAULT_MAX_PAGES    = 12
DEFAULT_BROAD_AGENCY = 'CONSORTIUM'

# Column -> default. The order here is the stored column order.
COLUMN_DEFAULTS: dict = {
    'source_id':            '',
    'url':                  '',
    'name':                 '',
    'cadence':              'weekly',
    'instructions':         '',
    'broad_agency':         DEFAULT_BROAD_AGENCY,
    'sub_agency':           '',
    'enabled':              True,
    'max_pages':            DEFAULT_MAX_PAGES,
    'last_checked':         '',
    'last_status':          '',
    'last_error':           '',
    'last_run_id':          '',
    'consecutive_failures': 0,
    'has_api':              False,
    'api_note':             '',
    'requires_login':       False,
    'notes':                '',
    'added_at':             '',
    'added_by':             '',
}
COLUMNS = list(COLUMN_DEFAULTS)

_BOOL_COLS = ('enabled', 'has_api', 'requires_login')
_INT_COLS  = ('max_pages', 'consecutive_failures')

# Outcome codes written to `last_status` by the job.
STATUS_OK            = 'ok'
STATUS_NOTHING_FOUND = 'no_opportunities'
STATUS_LOGIN         = 'needs_login'
STATUS_ERROR         = 'error'
STATUS_DEFERRED      = 'deferred'


# -- URL handling -----------------------------------------------------------

_TRACKING_PARAMS = re.compile(
    r'(?:^|&)(?:utm_[a-z_]+|fbclid|gclid|mc_cid|mc_eid)=[^&]*', re.I
)


def normalize_url(url: str) -> str:
    """Storage form: scheme-prefixed, whitespace-stripped, tracking params gone.

    The sheet carries rows like `https://www.sofwerx.org/?utm_source=chatgpt.com`
    — the tracking param is noise that would also defeat URL-keyed dedup.
    """
    # Callers pass raw spreadsheet cells, so NaN floats arrive here routinely.
    if url is None or (isinstance(url, float) and pd.isna(url)):
        return ''
    url = str(url).strip()
    if not url or url.lower() in ('nan', 'none'):
        return ''
    if not url.lower().startswith(('http://', 'https://')):
        url = 'https://' + url.lstrip('/')
    if '?' in url:
        base, _, query = url.partition('?')
        query = _TRACKING_PARAMS.sub('', query).lstrip('&')
        url = base + ('?' + query if query else '')
    return url


def url_key(url: str) -> str:
    """Comparison form — lowercased, scheme / `www.` / trailing-slash agnostic."""
    u = normalize_url(url).lower()
    u = re.sub(r'^https?://', '', u)
    u = re.sub(r'^www\.', '', u)
    return u.rstrip('/')


def host_of(url: str) -> str:
    u = re.sub(r'^https?://', '', normalize_url(url).lower())
    return u.split('/')[0].split('?')[0].replace('www.', '')


# -- Agency routing ---------------------------------------------------------

# Host substring -> broad_agency folder. Checked in order, first match wins.
# This is a *suggestion* surfaced in the import preview and the editor, never a
# silent decision — the stored `broad_agency` column is what the job routes on.
_AGENCY_RULES: list = [
    # The store already separates DARPA and DEVCOM from the broad DOD bucket —
    # check the live folder list before adding a rule, or one source's
    # opportunities end up split across two folders.
    ('darpa.mil', 'DARPA'), ('darpaconnect.us', 'DARPA'),
    ('devcom.army.mil', 'DEVCOM'),
    ('dodsbirsttr.mil', 'DOD'), ('diu.mil', 'DOD'),
    ('spacecom.mil', 'DOD'), ('socom.mil', 'DOD'), ('dla.mil', 'DOD'),
    ('afwerx.com', 'DOD'), ('spacewerx.us', 'DOD'), ('navysbir.com', 'DOD'),
    ('arpa-h.gov', 'ARPA-H'),
    ('cirm.ca.gov', 'CIRM'),
    ('medicalcountermeasures.gov', 'BARDA'), ('barda', 'BARDA'),
    ('doi.gov', 'DOI'),
    ('grants.nih.gov', 'HHS'), ('nih.gov', 'HHS'), ('hhs.gov', 'HHS'),
    ('cdc.gov', 'HHS'), ('fda.gov', 'HHS'), ('samhsa.gov', 'HHS'),
    ('arpa-e', 'DOE'), ('energy.gov', 'DOE'), ('ornl.gov', 'DOE'),
    ('sandia.gov', 'DOE'), ('pnnl.gov', 'DOE'), ('inl.gov', 'DOE'),
    ('nsf.gov', 'NSF'),
    ('nasa.gov', 'NASA'),
    ('dhs.gov', 'DHS'),
    ('noaa.gov', 'NOAA'),
    ('nist.gov', 'DOC'), ('commerce.gov', 'DOC'),
    ('transportation.gov', 'DOT'), ('dot.gov', 'DOT'),
    ('va.gov', 'VA'),
    ('nifa.usda.gov', 'USDA'), ('usda.gov', 'USDA'),
    ('ed.gov', 'ED'),
    ('sbir.gov', 'SBA'), ('sba.gov', 'SBA'),
    ('grants.gov', 'GRANTS-GOV'),
    ('cprit.texas.gov', 'TEXAS'), ('cpritgrants.org', 'TEXAS'),
]

# State economic-development programs are a distinct kind of source — routing
# ~40 of them into CONSORTIUM would make that folder mean nothing. `va.gov` is
# deliberately handled above as Veterans Affairs, not Virginia.
_STATE_NAMES = {
    'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
    'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
    'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine',
    'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi',
    'missouri', 'montana', 'nebraska', 'nevada', 'newhampshire', 'newjersey',
    'newmexico', 'newyork', 'northcarolina', 'northdakota', 'ohio', 'oklahoma',
    'oregon', 'pennsylvania', 'rhodeisland', 'southcarolina', 'southdakota',
    'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington',
    'westvirginia', 'wisconsin', 'wyoming',
}
_STATE_ABBREV = {
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id',
    'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms',
    'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny', 'nc', 'nd', 'oh', 'ok',
    'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'wa', 'wv', 'wi',
    'wy',
}


def infer_broad_agency(url: str) -> str:
    """Suggest a destination folder from the host. Never authoritative."""
    host = host_of(url)
    if not host:
        return DEFAULT_BROAD_AGENCY
    for needle, key in _AGENCY_RULES:
        if needle in host:
            return key
    if host.endswith('.mil'):
        return 'DOD'
    if host.endswith('.gov'):
        parts = host.split('.')
        if len(parts) >= 2:
            label = parts[-2]
            if label in _STATE_ABBREV or label.replace('-', '') in _STATE_NAMES:
                return 'STATE'
        stem = host.replace('.gov', '').replace('.', '')
        if any(s in stem for s in _STATE_NAMES):
            return 'STATE'
    # Only the `<agency>.<state>.us` form is a state site. A bare `.us` is not:
    # americamakes.us and nextflex.us are Manufacturing USA institutes, i.e.
    # consortia, and an over-broad rule routed them to STATE.
    if re.search(r'\.([a-z]{2})\.us$', host):
        if re.search(r'\.([a-z]{2})\.us$', host).group(1) in _STATE_ABBREV:
            return 'STATE'
    return DEFAULT_BROAD_AGENCY


# -- Frame helpers ----------------------------------------------------------

def new_source_id() -> str:
    return secrets.token_hex(3)


def normalize_cadence(value) -> str:
    v = str(value or '').strip().lower()
    if v in CADENCES:
        return v
    if v.startswith('dai'):
        return 'daily'
    if v.startswith('week') or v.startswith('bi-week'):
        return 'weekly'
    if v.startswith('month') or v.startswith('quarter'):
        return 'monthly'
    return 'weekly'


def _parse_date(value) -> str:
    """Best-effort ISO date. The sheet mixes `9/5/2026` with `30-Mar`."""
    txt = str(value or '').strip()
    if not txt or txt.lower() in ('nan', 'none', 'nat'):
        return ''
    try:
        ts = pd.to_datetime(txt, errors='coerce')
    except Exception:
        return ''
    if pd.isna(ts):
        return ''
    return ts.strftime('%Y-%m-%d')


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add every missing column with its default and coerce dtypes.

    Called on read and before write, so a parquet written by an older version
    of this module keeps loading.
    """
    out = df.copy() if df is not None else pd.DataFrame()
    for col, default in COLUMN_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
    for col in _BOOL_COLS:
        out[col] = out[col].fillna(COLUMN_DEFAULTS[col]).astype(bool)
    for col in _INT_COLS:
        out[col] = (
            pd.to_numeric(out[col], errors='coerce')
            .fillna(COLUMN_DEFAULTS[col])
            .astype(int)
        )
    for col in COLUMNS:
        if col not in _BOOL_COLS and col not in _INT_COLS:
            out[col] = out[col].fillna('').astype(str)
    out['cadence'] = out['cadence'].map(normalize_cadence)
    extra = [c for c in out.columns if c not in COLUMNS]
    return out[COLUMNS + extra].reset_index(drop=True)


def blank_row(url: str = '', added_by: str = '') -> dict:
    row = dict(COLUMN_DEFAULTS)
    row['source_id']    = new_source_id()
    row['url']          = normalize_url(url)
    row['name']         = host_of(url)
    row['broad_agency'] = infer_broad_agency(url)
    row['added_at']     = datetime.today().strftime('%Y-%m-%d')
    row['added_by']     = added_by
    return row


# -- Store I/O --------------------------------------------------------------

def load_sources(client) -> pd.DataFrame:
    """Read the registry. A missing blob is an empty registry, not an error."""
    blob = client.bucket(BUCKET).blob(SOURCES_BLOB)
    if not blob.exists():
        return ensure_columns(pd.DataFrame(columns=COLUMNS))
    df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
    return ensure_columns(df)


def save_sources(client, df: pd.DataFrame) -> None:
    out = ensure_columns(df)
    buf = io.BytesIO()
    out.to_parquet(buf, index=False)
    buf.seek(0)
    client.bucket(BUCKET).blob(SOURCES_BLOB).upload_from_file(
        buf, content_type='application/octet-stream'
    )


def upsert_sources(client, rows: list) -> pd.DataFrame:
    """Re-read from GCS, merge `rows` by source_id, write back.

    Re-reading rather than trusting a caller's in-memory copy is the
    client_profile_job precedent: a long job must not clobber an edit someone
    made in the view while it was running.
    """
    current = load_sources(client)
    if not rows:
        return current

    by_id = {r['source_id']: r for r in rows if r.get('source_id')}
    if by_id:
        mask = current['source_id'].isin(by_id)
        for idx in current.index[mask]:
            sid = current.at[idx, 'source_id']
            for key, val in by_id[sid].items():
                if key in current.columns and key != 'source_id':
                    current.at[idx, key] = val
        known = set(current['source_id'])
        fresh = [r for sid, r in by_id.items() if sid not in known]
        if fresh:
            current = pd.concat([current, pd.DataFrame(fresh)], ignore_index=True)

    merged = ensure_columns(current)
    save_sources(client, merged)
    return merged


def delete_sources(client, source_ids: list) -> pd.DataFrame:
    current = load_sources(client)
    kept    = current[~current['source_id'].isin(set(source_ids))]
    merged  = ensure_columns(kept)
    save_sources(client, merged)
    return merged


# -- Cadence ----------------------------------------------------------------

def is_due(row, today: str = '') -> bool:
    if not bool(row.get('enabled', True)):
        return False
    cadence = normalize_cadence(row.get('cadence'))
    if cadence == 'paused':
        return False
    last = str(row.get('last_checked') or '').strip()
    if not last:
        return True
    try:
        last_dt = datetime.strptime(last, '%Y-%m-%d')
    except ValueError:
        return True
    ref = datetime.strptime(today, '%Y-%m-%d') if today else datetime.today()
    return (ref - last_dt) >= timedelta(days=CADENCE_DAYS.get(cadence, 6))


def due_sources(df: pd.DataFrame, today: str = '') -> pd.DataFrame:
    if df is None or df.empty:
        return df
    mask = df.apply(lambda r: is_due(r, today), axis=1)
    return df[mask].reset_index(drop=True)


def days_since_checked(row):
    last = str(row.get('last_checked') or '').strip()
    if not last:
        return None
    try:
        return (datetime.today() - datetime.strptime(last, '%Y-%m-%d')).days
    except ValueError:
        return None


# -- One-time CSV import ----------------------------------------------------

def import_rows(raw: pd.DataFrame, col_map: dict, added_by: str = '') -> pd.DataFrame:
    """Turn the exported Google Sheet into candidate registry rows.

    `col_map` maps registry fields to source column names, e.g.
    {'url': 'List here', 'cadence': 'Check: ', 'last_checked': 'Last Checked',
     'instructions': 'Unnamed: 3'}

    Rows without a usable URL are dropped — the sheet has at least one
    free-text entry ("AEDC Velocity Alliance") that is a name, not a link.
    """
    url_col = col_map.get('url')
    if not url_col or url_col not in raw.columns:
        raise ValueError('import_rows: a url column must be mapped')

    rows  = []
    seen  = set()
    for _, src in raw.iterrows():
        url = normalize_url(src.get(url_col, ''))
        if not url or not re.match(r'^https?://[^/\s]+\.[^/\s]', url):
            continue
        key = url_key(url)
        if key in seen:
            continue
        seen.add(key)

        row = blank_row(url, added_by=added_by)
        for field in ('cadence', 'last_checked', 'instructions', 'name',
                      'broad_agency', 'sub_agency', 'notes'):
            col = col_map.get(field)
            if not col or col not in raw.columns:
                continue
            val = src.get(col)
            if pd.isna(val) or str(val).strip() == '':
                continue
            if field == 'cadence':
                row['cadence'] = normalize_cadence(val)
            elif field == 'last_checked':
                row['last_checked'] = _parse_date(val)
            else:
                row[field] = str(val).strip()
        rows.append(row)

    return ensure_columns(pd.DataFrame(rows))


def merge_import(existing: pd.DataFrame, incoming: pd.DataFrame):
    """Append only the incoming rows whose URL is not already registered."""
    existing = ensure_columns(existing)
    incoming = ensure_columns(incoming)
    known    = {url_key(u) for u in existing['url']}
    fresh    = incoming[~incoming['url'].map(lambda u: url_key(u) in known)]
    merged   = ensure_columns(pd.concat([existing, fresh], ignore_index=True))
    return merged, len(fresh), len(incoming) - len(fresh)


# -- Seen index -------------------------------------------------------------

def _norm_title(title: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', str(title or '').lower()).strip()


def seen_key(source_id: str, url: str = '', title: str = '') -> str:
    """Identity of one extracted opportunity.

    The opportunity's own URL when there is one (stable across re-listings),
    otherwise the source plus its normalized title.
    """
    if url and str(url).strip():
        return 'url:' + url_key(url)
    return f'{source_id}:{_norm_title(title)}'


def load_seen(client, source_id: str) -> dict:
    blob = client.bucket(BUCKET).blob(f'{SEEN_PREFIX}{source_id}.json')
    if not blob.exists():
        return {}
    try:
        data = json.loads(blob.download_as_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_seen(client, source_id: str, index: dict) -> None:
    client.bucket(BUCKET).blob(f'{SEEN_PREFIX}{source_id}.json').upload_from_string(
        json.dumps(index), content_type='application/json'
    )


def known_titles(index: dict, limit: int = 60) -> list:
    """Titles handed to the agent so it can skip what is already stored.

    Newest first and capped — on a site with hundreds of historical entries the
    full list would cost more in prompt tokens than the re-extraction it saves.
    """
    items = sorted(
        (v for v in index.values() if isinstance(v, dict) and v.get('title')),
        key=lambda v: str(v.get('first_seen') or ''),
        reverse=True,
    )
    return [str(v['title'])[:160] for v in items[:limit]]
