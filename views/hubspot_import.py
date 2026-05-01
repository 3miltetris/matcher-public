"""
HubSpot Import
--------------
Builds a deduplicated contacts-only CSV from a completed matching run and
submits it to HubSpot via the CRM Imports API (crm.import scope).
Polls for completion and surfaces the imported/updated/error counts.
"""

import io
import json
import time
import traceback

import pandas as pd
import requests
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account

_BUCKET         = 'cc-matcher-bucket-jeg-v1'
_RESULTS_PREFIX = 'matching-results/'
_HS_BASE        = 'https://api.hubapi.com'
_POLL_INTERVAL  = 3   # seconds between status polls

_OBJECT_TYPE = 'COMPANY'

# Standard HubSpot company properties — no creation needed
# (csv_col, hs_property, idColumnType or None)
_STANDARD_COLS = [
    ('companyWebsite', 'domain',      'HUBSPOT_ALTERNATE_ID'),
    ('companyName',    'name',        None),
    ('company_summary','description', None),
]

# Custom properties created automatically if missing
# csv_col: (hs_internal_name, display_label, type, fieldType)
_CUSTOM_PROPS: dict[str, tuple[str, str, str, str]] = {
    'source_y':      ('matcher_source',        'Matcher Source',        'string', 'text'),
    'topic_number':  ('matcher_topic_number',  'Matcher Topic Number',  'string', 'text'),
    'title':         ('matcher_grant_title',   'Matcher Grant Title',   'string', 'text'),
    'agency':        ('matcher_agency',        'Matcher Agency',        'string', 'text'),
    'broad_agency':  ('matcher_broad_agency',  'Matcher Broad Agency',  'string', 'text'),
    'due_date':      ('matcher_due_date',        'Matcher Due Date',      'string', 'text'),
    'funding_amount':('matcher_funding_amount', 'Matcher Funding Amount', 'string', 'text'),
    'grant_summary': ('matcher_grant_summary', 'Matcher Grant Summary', 'string', 'textarea'),
    'good_match':    ('matcher_good_match',    'Matcher Good Match',    'string', 'text'),
    'subject_line':  ('matcher_subject_line',  'Matcher Subject Line',  'string', 'text'),
    'ai_message':    ('matcher_ai_message',    'Matcher AI Message',    'string', 'textarea'),
}

for _k in ['hs_import_id', 'hs_import_rows', 'hs_df', 'hs_run_id']:
    if _k not in st.session_state:
        st.session_state[_k] = None


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _gcs_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)


def _hs_headers() -> dict:
    return {'Authorization': f'Bearer {st.secrets["hubspot_api_key"]}'}


# ── GCS helpers ───────────────────────────────────────────────────────────────

def _list_completed_runs(client: storage.Client) -> list[str]:
    blobs = client.list_blobs(_BUCKET, prefix=_RESULTS_PREFIX, delimiter='/')
    list(blobs)
    run_ids = sorted(
        (p.replace(_RESULTS_PREFIX, '').strip('/') for p in blobs.prefixes),
        reverse=True,
    )
    completed = []
    for run_id in run_ids:
        blob = client.bucket(_BUCKET).blob(f'{_RESULTS_PREFIX}{run_id}/status.json')
        if blob.exists():
            status = json.loads(blob.download_as_text())
            if not status.get('error'):
                completed.append(run_id)
    return completed


def _load_run(client: storage.Client, run_id: str) -> pd.DataFrame:
    prefix = f'{_RESULTS_PREFIX}{run_id}/'
    blobs  = sorted(
        [b for b in client.list_blobs(_BUCKET, prefix=prefix) if b.name.endswith('.csv')],
        key=lambda b: b.name,
    )
    frames = [pd.read_csv(io.BytesIO(b.download_as_bytes())) for b in blobs]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── HubSpot helpers ───────────────────────────────────────────────────────────

def _submit_import(df: pd.DataFrame, run_id: str) -> tuple[str, int]:
    """
    Build a deduplicated contacts CSV and submit to HubSpot imports API.
    Returns (import_id, row_count).
    """
    # Build full column map: standard props + any custom props present in the DataFrame
    col_map = [(src, hs, id_type) for src, hs, id_type in _STANDARD_COLS if src in df.columns]
    for csv_col, (hs_name, _label, _type, _field) in _CUSTOM_PROPS.items():
        if csv_col in df.columns:
            col_map.append((csv_col, hs_name, None))

    present = col_map
    if not present:
        raise ValueError('Run CSV has none of the expected company columns (companyName, companyWebsite).')

    export = df[[c[0] for c in present]].copy()

    # Require a website — domain is the HubSpot dedup key, rows without one can't be imported
    if 'companyWebsite' in export.columns:
        valid_site = (
            export['companyWebsite'].notna() &
            (export['companyWebsite'].astype(str).str.strip() != '') &
            (export['companyWebsite'].astype(str).str.strip().str.lower() != 'nan')
        )
        export = export[valid_site].drop_duplicates(subset=['companyWebsite']).reset_index(drop=True)

    if export.empty:
        raise ValueError('No rows with a valid companyWebsite to import.')

    column_mappings = []
    for src, hs_prop, id_type in present:
        mapping = {'columnObjectType': _OBJECT_TYPE, 'columnName': src, 'propertyName': hs_prop}
        if id_type:
            mapping['idColumnType'] = id_type
        column_mappings.append(mapping)

    import_request = {
        'name': f'Matcher: {run_id}',
        'files': [{
            'fileName': 'contacts.csv',
            'fileFormat': 'CSV',
            'dateFormat': 'YEAR_MONTH_DAY',
            'fileImportPage': {
                'hasHeader': True,
                'columnMappings': column_mappings,
            },
        }],
    }

    resp = requests.post(
        f'{_HS_BASE}/crm/v3/imports',
        headers=_hs_headers(),
        files={
            'importRequest': (None, json.dumps(import_request), 'application/json'),
            'files': ('contacts.csv', export.to_csv(index=False).encode('utf-8'), 'text/csv'),
        },
        timeout=60,
    )

    if resp.status_code not in (200, 201):
        raise RuntimeError(f'HTTP {resp.status_code}: {resp.text[:400]}')

    return str(resp.json()['id']), len(export)


def _ensure_properties() -> list[str]:
    """
    Create any missing custom company properties in HubSpot.
    Returns a list of property names that were created.
    Requires crm.schemas.companies.write scope on the Private App token.
    """
    resp = requests.get(
        f'{_HS_BASE}/crm/v3/properties/companies',
        headers=_hs_headers(),
        timeout=15,
    )
    if resp.status_code != 200:
        raise RuntimeError(f'Could not read HubSpot company properties: HTTP {resp.status_code}: {resp.text[:200]}')

    existing = {p['name'] for p in resp.json().get('results', [])}
    created  = []

    for _csv_col, (hs_name, label, prop_type, field_type) in _CUSTOM_PROPS.items():
        if hs_name in existing:
            continue
        create_resp = requests.post(
            f'{_HS_BASE}/crm/v3/properties/companies',
            headers={**_hs_headers(), 'Content-Type': 'application/json'},
            json={
                'name':      hs_name,
                'label':     label,
                'type':      prop_type,
                'fieldType': field_type,
                'groupName': 'companyinformation',
            },
            timeout=15,
        )
        if create_resp.status_code not in (200, 201):
            raise RuntimeError(
                f'Could not create property `{hs_name}`: '
                f'HTTP {create_resp.status_code}: {create_resp.text[:200]}\n'
                'Make sure your Private App token has the '
                '`crm.schemas.companies.write` scope.'
            )
        created.append(hs_name)

    return created


def _poll_import(import_id: str) -> dict:
    resp = requests.get(
        f'{_HS_BASE}/crm/v3/imports/{import_id}',
        headers=_hs_headers(),
        timeout=15,
    )
    if resp.status_code != 200:
        raise RuntimeError(f'HTTP {resp.status_code}: {resp.text[:200]}')
    return resp.json()


# ── Page ──────────────────────────────────────────────────────────────────────

st.title('🔗 HubSpot Import')
st.caption('Load a completed matching run and import companies to HubSpot via the Imports API.')

if 'hubspot_api_key' not in st.secrets:
    st.info(
        'Add `hubspot_api_key = "pat-..."` **above** the `[gcp_service_account]` section in '
        'your Streamlit secrets. Generate a Private App token in HubSpot → Settings → '
        'Integrations → Private Apps with the **crm.import** scope.'
    )
    st.stop()

# ── Polling UI ────────────────────────────────────────────────────────────────

if st.session_state.hs_import_id:
    import_id = st.session_state.hs_import_id
    st.subheader('Import in progress')
    st.caption(f'Import ID: `{import_id}`')

    try:
        status = _poll_import(import_id)
        state  = status.get('state', '')
        stats  = status.get('statistics', {})

        if state == 'DONE':
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Created',  stats.get('objectsCreated', 0))
            c2.metric('Updated',  stats.get('objectsUpdated', 0))
            c3.metric('Ignored',  stats.get('objectsIgnored', 0))
            c4.metric('Errors',   stats.get('errorsCount',    0))
            st.success('Import complete.')
            st.session_state.hs_import_id = None

        elif state in ('FAILED', 'CANCELED'):
            st.error(f'Import {state.lower()}.')
            st.json(status)
            st.session_state.hs_import_id = None

        else:
            rows_done  = stats.get('rowsProcessed', 0)
            rows_total = stats.get('totalRows', 0) or st.session_state.hs_import_rows or '?'
            st.info(f'State: **{state}** — {rows_done} / {rows_total} rows. Rechecking in {_POLL_INTERVAL}s…')
            if st.button('Stop monitoring (import keeps running in HubSpot)'):
                st.session_state.hs_import_id = None
                st.rerun()
            time.sleep(_POLL_INTERVAL)
            st.rerun()

    except Exception as e:
        st.error(f'Error polling import: {e}')
        if st.button('Stop monitoring'):
            st.session_state.hs_import_id = None
            st.rerun()

    st.stop()

# ── Run selector ──────────────────────────────────────────────────────────────

gcs = _gcs_client()

with st.spinner('Fetching completed runs from GCS…'):
    runs = _list_completed_runs(gcs)

if not runs:
    st.warning('No completed matching runs found in GCS.')
    st.stop()

run_id = st.selectbox('Select a completed run', runs)

if st.session_state.hs_run_id != run_id:
    st.session_state.hs_df     = None
    st.session_state.hs_run_id = run_id

if st.button('Load run data', type='primary'):
    with st.spinner('Concatenating segment CSVs…'):
        df = _load_run(gcs, run_id)
    st.session_state.hs_df = df if not df.empty else None
    if df.empty:
        st.warning('No CSV segments found for this run.')

df: pd.DataFrame | None = st.session_state.hs_df
if df is None:
    st.stop()

# ── Preview ───────────────────────────────────────────────────────────────────

st.divider()

unique_sites = (
    df['companyWebsite'].dropna()
    .astype(str).str.strip()
    .pipe(lambda s: s[s != ''])
    .pipe(lambda s: s[s.str.lower() != 'nan'])
    .nunique()
) if 'companyWebsite' in df.columns else 0

c1, c2, c3 = st.columns(3)
c1.metric('Total rows',              f'{len(df):,}')
c2.metric('Unique websites',         f'{unique_sites:,}')
c3.metric('Companies to import',     f'{unique_sites:,}')

all_mapped_cols = (
    [src for src, _, _ in _STANDARD_COLS] +
    list(_CUSTOM_PROPS.keys())
)
import_cols = [c for c in all_mapped_cols if c in df.columns]
st.caption(f'Columns that will be imported: `{"`, `".join(import_cols)}`')
st.dataframe(df[import_cols].head(50), use_container_width=True, hide_index=True)

# ── Import ────────────────────────────────────────────────────────────────────

st.divider()
if not st.button('▶ Import to HubSpot', type='primary'):
    st.stop()

try:
    with st.spinner('Ensuring HubSpot custom properties exist…'):
        created_props = _ensure_properties()
    if created_props:
        st.info(f'Created {len(created_props)} new HubSpot property/ies: `{"`, `".join(created_props)}`')

    with st.spinner('Building CSV and submitting to HubSpot…'):
        import_id, row_count = _submit_import(df, run_id)

    st.session_state.hs_import_id   = import_id
    st.session_state.hs_import_rows = row_count
    st.success(f'Submitted — **{row_count:,}** contacts · Import ID: `{import_id}`')
    time.sleep(1)
    st.rerun()

except Exception as e:
    st.error(f'Failed to submit import: {e}')
    st.code(traceback.format_exc())
