"""
HubSpot Import
--------------
Imports companies to HubSpot via the CRM Imports API (crm.import scope)
from any of three sources:

- **Matching run** — concatenates segment CSVs from matching-results/{run_id}/
  and imports the standard matcher_* property set (original flow).
- **Financial research run** — loads a completed Client Financials run from
  finance-research-runs/{run_id}/state.json and imports the Deep Research
  fields. Users map each financial field to an existing HubSpot company
  property or a new auto-created matcher_fin_* property.
- **Client profiles** — loads the multi-aspect capability profiles from
  data/client-profiles/profiles.parquet (Stage 8), flattens each company's
  aspects into importable columns (summary, labels, keywords, a readable
  aspects block, and optional per-aspect columns), and maps them the same way.

All modes dedupe by companyWebsite → domain and poll the import until done.
"""

import io
import json
import re
import time
import traceback
from datetime import date

import pandas as pd
import requests
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account

import src.modules.aspect_profile as ap
import src.modules.finance_research as fr

_BUCKET          = 'cc-matcher-bucket-jeg-v1'
_RESULTS_PREFIX  = 'matching-results/'
_FIN_RUNS_PREFIX = 'finance-research-runs/'
_HS_BASE         = 'https://api.hubapi.com'
_POLL_INTERVAL   = 3   # seconds between status polls

_OBJECT_TYPE = 'COMPANY'

# Standard HubSpot company properties — no creation needed
# (csv_col, hs_property, idColumnType or None)
_STANDARD_COLS = [
    ('companyWebsite', 'domain',      'HUBSPOT_ALTERNATE_ID'),
    ('companyName',    'name',        None),
    ('company_summary','description', None),
]

# Custom properties created automatically if missing (matching-run mode)
# csv_col: (hs_internal_name, display_label, type, fieldType)
_CUSTOM_PROPS: dict[str, tuple[str, str, str, str]] = {
    'source':        ('matcher_contact_source', 'Matcher Contact Source', 'string', 'text'),
    'grant_source':  ('matcher_grant_source',  'Matcher Grant Source',  'string', 'text'),
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

_KNOWN_COLS    = frozenset([src for src, _, _ in _STANDARD_COLS] + list(_CUSTOM_PROPS.keys()))
_IMPORT_EXCLUDE = frozenset({'embeddings', 'uuid'})

# ── Financial-mode constants ──────────────────────────────────────────────────

# All mappable financial fields: the digest first, then the 54 research fields
_FIN_FIELDS = ['financial_summary'] + fr.ALL_FIELDS

# Sentinel dropdown option — resolves per-row to the field's auto-created name
_CREATE_OPT = '➕ create new property'
_SKIP_OPT   = '— skip —'

# Long-form fields get textarea properties when auto-created
_FIN_TEXTAREA = frozenset({
    'financial_summary', 'award_detail_json', 'sources_used',
    'outreach_angle', 'outreach_triggers', 'risks_red_flags',
    'confidence_notes',
})

# Fields included by default in the mapping table (headline findings)
_FIN_DEFAULT_ON = frozenset({
    'financial_summary', 'revenue_estimate', 'revenue_year',
    'total_venture_funding', 'total_grant_funding',
    'federal_awards_count_3yr', 'federal_awards_total_3yr',
    'employee_count_current', 'headcount_trend',
    'score_total', 'recommendation', 'confidence_score',
})


# ── Client-profile-mode constants ─────────────────────────────────────────────

# Company-level profile fields, in mapping-table order:
# (df column, auto-created property name, on by default, textarea)
_PROFILE_BASE_FIELDS: list[tuple[str, str, bool, bool]] = [
    ('profile_summary',  'matcher_profile_summary',   True,  True),
    ('aspect_labels',    'matcher_aspect_labels',     True,  True),
    ('aspects_full',     'matcher_aspects_full',      True,  True),
    ('aspect_keywords',  'matcher_aspect_keywords',   True,  True),
    ('aspect_kinds',     'matcher_aspect_kinds',      False, False),
    ('n_aspects',        'matcher_aspect_count',      True,  False),
    ('sources_used',     'matcher_profile_sources',   False, False),
    ('profile_model',    'matcher_profile_model',     False, False),
    ('profile_built_at', 'matcher_profile_built_at',  True,  False),
]


def _profile_aspect_fields(max_aspects: int) -> list[tuple[str, str, bool, bool]]:
    """Per-aspect columns for the loaded profiles — off by default, since most
    portals only want the rolled-up fields."""
    fields = []
    for i in range(1, max_aspects + 1):
        fields.append((f'aspect_{i}_label', f'matcher_aspect_{i}_label', False, False))
        fields.append((f'aspect_{i}_text',  f'matcher_aspect_{i}_text',  False, True))
    return fields


def _col_to_label(col: str) -> str:
    return col.replace('_', ' ').title()


for _k in ['hs_import_id', 'hs_import_rows', 'hs_df', 'hs_run_id', 'hs_mode_last']:
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

def _list_completed_runs(client: storage.Client) -> list[tuple[str, str]]:
    """
    Every run with at least one segment CSV, newest first, as (run_id, label).
    Runs whose status.json is missing (job killed/timed out) or carries an
    error are still listed — their saved segments are importable — but get a
    partial-run label so the user knows the results are incomplete.
    """
    has_csv, status_blobs = set(), {}
    for blob in client.list_blobs(_BUCKET, prefix=_RESULTS_PREFIX):
        rel = blob.name[len(_RESULTS_PREFIX):]
        if '/' not in rel:
            continue
        run_id, filename = rel.split('/', 1)
        if filename.endswith('.csv'):
            has_csv.add(run_id)
        elif filename == 'status.json':
            status_blobs[run_id] = blob
    runs = []
    for run_id in sorted(has_csv, reverse=True):
        blob = status_blobs.get(run_id)
        if blob is None:
            runs.append((run_id, f'{run_id} — ⚠️ partial (no status file — job died or still running)'))
            continue
        try:
            status = json.loads(blob.download_as_text())
        except Exception:
            status = {}
        if status.get('error'):
            runs.append((run_id, f'{run_id} — ⚠️ partial (job errored)'))
        else:
            runs.append((run_id, run_id))
    return runs


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


def _list_fin_runs(client: storage.Client) -> list[str]:
    """Financial research runs whose state.json has ≥1 completed company."""
    blobs = client.list_blobs(_BUCKET, prefix=_FIN_RUNS_PREFIX, delimiter='/')
    list(blobs)
    run_ids = sorted(
        (p.replace(_FIN_RUNS_PREFIX, '').strip('/') for p in blobs.prefixes),
        reverse=True,
    )
    completed = []
    for run_id in run_ids:
        blob = client.bucket(_BUCKET).blob(f'{_FIN_RUNS_PREFIX}{run_id}/state.json')
        if not blob.exists():
            continue
        try:
            state = json.loads(blob.download_as_text())
        except Exception:
            continue
        if any(c.get('output') for c in state.get('companies', [])):
            completed.append(run_id)
    return completed


def _load_fin_run(client: storage.Client, run_id: str) -> pd.DataFrame:
    """One row per researched company: identity + digest + all research fields."""
    blob = client.bucket(_BUCKET).blob(f'{_FIN_RUNS_PREFIX}{run_id}/state.json')
    state = json.loads(blob.download_as_text())
    rows = []
    for c in state.get('companies', []):
        output = c.get('output')
        if not output:
            continue
        rows.append({
            'companyName':       c.get('company_name') or '',
            'companyWebsite':    c.get('website') or '',
            'financial_summary': fr.build_financial_digest(output),
            **{f: output.get(f, '') for f in fr.ALL_FIELDS},
        })
    return pd.DataFrame(rows)


def _load_profile_rows(client: storage.Client) -> pd.DataFrame:
    """data/client-profiles/profiles.parquet → one flat import row per company.

    The aspects live in the profile row as a JSON array; HubSpot needs flat
    columns, so each company gets rolled-up fields (labels, keywords, a
    readable aspects block) plus one label/text pair per aspect.
    """
    profiles = ap.load_profiles(client)
    if profiles.empty:
        return pd.DataFrame()

    rows = []
    for _, p in profiles.iterrows():
        aspects = ap.profile_aspects(p)
        labels, kinds, keywords, blocks = [], [], [], []
        row = {
            'company_key':      str(p.get('company_key') or ''),
            'companyName':      str(p.get('company_name') or ''),
            'companyWebsite':   str(p.get('companyWebsite') or ''),
            'profile_summary':  str(p.get('profile_summary') or ''),
            'sources_used':     str(p.get('sources_used') or ''),
            'profile_model':    str(p.get('model') or ''),
            'profile_built_at': str(p.get('built_at') or ''),
        }
        for i, a in enumerate(aspects, start=1):
            label = str(a.get('label') or '')
            kind  = str(a.get('kind') or '')
            text  = str(a.get('text') or '')
            kw    = str(a.get('keywords') or '')
            labels.append(label)
            if kind:
                kinds.append(kind)
            block = f'{i}. {label}' + (f' ({kind})' if kind else '')
            if text:
                block += f'\n{text}'
            if kw:
                block += f'\nKeywords: {kw}'
                keywords.extend(k.strip() for k in kw.split(',') if k.strip())
            blocks.append(block)
            row[f'aspect_{i}_label'] = label
            row[f'aspect_{i}_text']  = text + (f'\nKeywords: {kw}' if kw else '')

        # Dedup keywords case-insensitively, first spelling wins
        seen_kw, uniq_kw = set(), []
        for k in keywords:
            if k.lower() not in seen_kw:
                seen_kw.add(k.lower())
                uniq_kw.append(k)

        row['aspect_labels']   = ' | '.join(labels)
        row['aspects_full']    = '\n\n'.join(blocks)
        row['aspect_keywords'] = ', '.join(uniq_kw)
        row['aspect_kinds']    = ', '.join(dict.fromkeys(kinds))
        row['n_aspects']       = str(len(aspects))
        rows.append(row)

    # Companies differ in aspect count — missing per-aspect columns become ''
    return pd.DataFrame(rows).fillna('')


# ── HubSpot helpers ───────────────────────────────────────────────────────────

def _fetch_company_properties() -> list[dict]:
    """Writable HubSpot company properties, for the mapping dropdown."""
    resp = requests.get(
        f'{_HS_BASE}/crm/v3/properties/companies',
        headers=_hs_headers(),
        timeout=15,
    )
    if resp.status_code != 200:
        raise RuntimeError(f'Could not read HubSpot company properties: HTTP {resp.status_code}: {resp.text[:200]}')
    props = []
    for p in resp.json().get('results', []):
        if p.get('modificationMetadata', {}).get('readOnlyValue'):
            continue
        props.append({'name': p['name'], 'label': p.get('label', p['name'])})
    return sorted(props, key=lambda p: p['name'])


def _submit_import(
    df: pd.DataFrame,
    run_id: str,
    extra_col_map: dict[str, str] | None = None,
) -> tuple[str, int]:
    """
    Build a deduplicated companies CSV and submit to HubSpot imports API.
    Returns (import_id, row_count).
    """
    # Build full column map: standard props + any custom props present in the DataFrame
    col_map = [(src, hs, id_type) for src, hs, id_type in _STANDARD_COLS if src in df.columns]
    for csv_col, (hs_name, _label, _type, _field) in _CUSTOM_PROPS.items():
        if csv_col in df.columns:
            col_map.append((csv_col, hs_name, None))
    for csv_col, hs_name in (extra_col_map or {}).items():
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


def _ensure_properties(
    extra_props: dict[str, tuple] | None = None,
    include_defaults: bool = True,
) -> list[str]:
    """
    Create any missing custom company properties in HubSpot.
    Returns a list of property names that were created.
    include_defaults=False skips the matching-run _CUSTOM_PROPS set
    (financial mode creates only what the mapping table asks for).
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

    all_props = dict(_CUSTOM_PROPS) if include_defaults else {}
    if extra_props:
        all_props.update(extra_props)

    for _csv_col, (hs_name, label, prop_type, field_type) in all_props.items():
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


def _mapping_editor(
    specs: list[tuple[str, str, bool, bool]],
    prop_options: list[str],
    key: str,
    *,
    label_col: str = 'Field',
    height: int = 560,
    reserved: frozenset[str] = frozenset({'domain', 'name'}),
) -> tuple[dict[str, str], dict[str, tuple]]:
    """Field → HubSpot property mapping table, shared by the financial and
    client-profile modes.

    specs items are (df column, auto-created property name, on by default,
    long-form). Returns (df_col → hs_property, create_props for
    _ensure_properties). Stops the page when the selection is unusable.
    """
    table = pd.DataFrame({
        'Import':           [d for _f, _n, d, _t in specs],
        label_col:          [f for f, _n, _d, _t in specs],
        'HubSpot property': [_CREATE_OPT] * len(specs),
    })
    edited = st.data_editor(
        table,
        hide_index=True,
        use_container_width=True,
        height=height,
        disabled=[label_col],
        column_config={
            'Import': st.column_config.CheckboxColumn(
                'Import', help='Include this field in the HubSpot import'),
            label_col: st.column_config.TextColumn(label_col),
            'HubSpot property': st.column_config.SelectboxColumn(
                'HubSpot property',
                options=prop_options,
                required=True,
                help='Target company property. The create option makes the '
                     'matcher_* property automatically.',
            ),
        },
        key=key,
    )

    meta = {f: (n, t) for f, n, _d, t in specs}
    col_map: dict[str, str] = {}
    create_props: dict[str, tuple] = {}
    for _, row in edited.iterrows():
        if not row['Import'] or row['HubSpot property'] == _SKIP_OPT:
            continue
        field = row[label_col]
        create_name, textarea = meta[field]
        if row['HubSpot property'] == _CREATE_OPT:
            hs_name = create_name
            create_props[field] = (
                hs_name,
                _col_to_label(create_name),
                'string',
                'textarea' if textarea else 'text',
            )
        else:
            hs_name = row['HubSpot property']
        col_map[field] = hs_name

    if not col_map:
        st.warning('No fields selected for import.')
        st.stop()

    targets = pd.Series(list(col_map.values()))
    dupes   = sorted(targets[targets.duplicated()].unique())
    if dupes:
        st.error(
            'Two or more fields are mapped to the same HubSpot property: '
            f'`{"`, `".join(dupes)}`. Each field needs its own target.'
        )
        st.stop()

    # `name` and `domain` are already mapped from companyName / companyWebsite —
    # pointing a field at them would submit two mappings for one property
    clashes = sorted(set(col_map.values()) & reserved)
    if clashes:
        st.error(
            f'`{"`, `".join(clashes)}` '
            f'{"are" if len(clashes) > 1 else "is"} already mapped from the '
            'company name / website columns. Pick a different target for '
            + ', '.join(f'`{f}`' for f, t in col_map.items() if t in clashes)
            + '.'
        )
        st.stop()

    return col_map, create_props


def _hs_property_options() -> list[str]:
    """Dropdown options for the mapping table, with a refresh button."""
    col_props, col_refresh = st.columns([5, 1])
    with col_refresh:
        if st.button('↺ Refresh', help='Re-fetch HubSpot company properties'):
            st.session_state.pop('hs_company_props', None)
            st.rerun()
    if 'hs_company_props' not in st.session_state:
        with st.spinner('Fetching HubSpot company properties…'):
            try:
                st.session_state.hs_company_props = _fetch_company_properties()
            except Exception as e:
                st.error(str(e))
                st.stop()
    hs_props: list[dict] = st.session_state.hs_company_props
    with col_props:
        st.caption(f'{len(hs_props)} writable company properties available '
                   'in your HubSpot portal.')
    return [_CREATE_OPT, _SKIP_OPT] + [p['name'] for p in hs_props]


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
st.caption('Import companies to HubSpot from a matching run or a financial research run.')

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

# ── Source & run selector ─────────────────────────────────────────────────────

mode = st.radio(
    'Import source',
    ['Matching run', 'Financial research run', 'Client profiles'],
    horizontal=True,
    help='Matching runs import grant-match results; financial research runs '
         'import Deep Research findings from the Client Research view; client '
         'profiles import the multi-aspect capability profiles built in the '
         'Client Profiles view.',
)

if st.session_state.hs_mode_last != mode:
    st.session_state.hs_df        = None
    st.session_state.hs_run_id    = None
    st.session_state.hs_mode_last = mode

gcs = _gcs_client()

if mode == 'Matching run':
    with st.spinner('Fetching matching runs from GCS…'):
        matching_runs = _list_completed_runs(gcs)
    if not matching_runs:
        st.warning('No matching runs with saved results found in GCS.')
        st.stop()
    run_labels = dict(matching_runs)
    run_id = st.selectbox(
        'Select a run',
        [r for r, _ in matching_runs],
        format_func=lambda r: run_labels[r],
    )
    if run_labels[run_id] != run_id:
        st.warning(
            'This run did not finish cleanly — only the segments saved before it '
            'stopped will be imported.'
        )
elif mode == 'Financial research run':
    with st.spinner('Fetching financial research runs from GCS…'):
        runs = _list_fin_runs(gcs)
    if not runs:
        st.warning('No completed financial research runs found in GCS. '
                   'Run one from the Client Financials view first.')
        st.stop()
    run_id = st.selectbox('Select a completed run', runs)
else:
    run_id = f'client_profiles_{date.today().isoformat()}'
    st.caption(
        'Imports the multi-aspect capability profiles stored in '
        f'`{ap.PROFILES_BLOB}` — one HubSpot company per profiled client, '
        'built in the **Client Profiles** view.'
    )

if st.session_state.hs_run_id != run_id:
    st.session_state.hs_df     = None
    st.session_state.hs_run_id = run_id

_load_label = ('Load client profiles' if mode == 'Client profiles'
               else 'Load run data')
if st.button(_load_label, type='primary'):
    with st.spinner('Loading data…'):
        if mode == 'Matching run':
            df = _load_run(gcs, run_id)
        elif mode == 'Financial research run':
            df = _load_fin_run(gcs, run_id)
        else:
            df = _load_profile_rows(gcs)
    st.session_state.hs_df = df if not df.empty else None
    if df.empty:
        st.warning('No importable data found.'
                   + (' Build profiles in the Client Profiles view first.'
                      if mode == 'Client profiles' else ''))

df: pd.DataFrame | None = st.session_state.hs_df
if df is None:
    st.stop()

# ── Client selection (profiles are a store, not a run) ────────────────────────

if mode == 'Client profiles':
    st.divider()
    prof_labels = {
        str(r['company_key']): (f"{r['companyName'] or '—'}  ·  "
                                f"{r['companyWebsite'] or 'no website'}  ·  "
                                f"{r['n_aspects']} aspects  ·  {r['profile_built_at']}")
        for _, r in df.iterrows()
    }
    all_keys = list(prof_labels)

    sel1, sel2, _sel3 = st.columns([1, 1, 6])
    _pick = None
    if sel1.button('All', key='hs_prof_all'):
        _pick = all_keys
    if sel2.button('None', key='hs_prof_none'):
        _pick = []
    if _pick is not None:
        st.session_state.hs_prof_pick = _pick
        st.rerun()

    st.session_state.hs_prof_pick = [
        k for k in st.session_state.get('hs_prof_pick', all_keys) if k in prof_labels
    ]
    picked = st.multiselect(
        'Clients to import',
        options=all_keys,
        format_func=lambda k: prof_labels[k],
        key='hs_prof_pick',
        help='Every profiled client is selected by default.',
    )
    df = df[df['company_key'].isin(set(picked))].reset_index(drop=True)
    if df.empty:
        st.warning('No clients selected.')
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
c1.metric('Total rows',          f'{len(df):,}')
c2.metric('Unique websites',     f'{unique_sites:,}')
c3.metric('Companies to import', f'{unique_sites:,}')

if unique_sites < len(df):
    st.caption(
        'Rows without a website are skipped — `domain` is the HubSpot dedup key.'
    )

# ═══ MODE: Financial research run ═════════════════════════════════════════════

if mode == 'Financial research run':

    preview_cols = ['companyName', 'companyWebsite', 'financial_summary',
                    'revenue_estimate', 'score_total', 'recommendation']
    st.dataframe(
        df[[c for c in preview_cols if c in df.columns]].head(50),
        use_container_width=True, hide_index=True,
    )

    # ── Field mapping ─────────────────────────────────────────────────────────

    st.divider()
    st.subheader('Map financial fields to HubSpot properties')
    st.caption(
        'Company name → `name` and website → `domain` are always mapped. '
        'For each financial field, choose an existing HubSpot company property '
        f'or keep "{_CREATE_OPT}" to auto-create `matcher_fin_<field>`. '
        'Uncheck **Import** to leave a field out.'
    )

    prop_options = _hs_property_options()

    fin_col_map, create_props = _mapping_editor(
        [(f, f'matcher_fin_{f}', f in _FIN_DEFAULT_ON, f in _FIN_TEXTAREA)
         for f in _FIN_FIELDS],
        prop_options,
        key=f'hs_fin_map_{run_id}',
        label_col='Financial field',
    )

    n_create = len(create_props)
    st.caption(
        f'**{len(fin_col_map)}** field{"s" if len(fin_col_map) != 1 else ""} will be imported'
        + (f' — {n_create} new `matcher_fin_*` propert{"ies" if n_create != 1 else "y"} will be created.'
           if n_create else '.')
    )

    # ── Import ────────────────────────────────────────────────────────────────

    st.divider()
    if not st.button('▶ Import to HubSpot', type='primary'):
        st.stop()

    try:
        if create_props:
            with st.spinner('Creating HubSpot custom properties…'):
                created_props = _ensure_properties(
                    extra_props=create_props, include_defaults=False)
            if created_props:
                st.info(f'Created {len(created_props)} new HubSpot property/ies: '
                        f'`{"`, `".join(created_props)}`')

        with st.spinner('Building CSV and submitting to HubSpot…'):
            import_id, row_count = _submit_import(df, run_id, extra_col_map=fin_col_map)

        st.session_state.hs_import_id   = import_id
        st.session_state.hs_import_rows = row_count
        st.success(f'Submitted — **{row_count:,}** companies · Import ID: `{import_id}`')
        time.sleep(1)
        st.rerun()

    except Exception as e:
        st.error(f'Failed to submit import: {e}')
        st.code(traceback.format_exc())

    st.stop()

# ═══ MODE: Client profiles ════════════════════════════════════════════════════

if mode == 'Client profiles':

    preview_cols = ['companyName', 'companyWebsite', 'n_aspects',
                    'aspect_labels', 'profile_built_at', 'sources_used']
    st.dataframe(
        df[[c for c in preview_cols if c in df.columns]].head(50),
        use_container_width=True, hide_index=True,
    )

    with st.expander('👁 Preview one profile as it will be imported'):
        row = df.iloc[0]
        st.markdown(f"**{row['companyName']}** · {row['companyWebsite']}")
        st.text(str(row.get('profile_summary') or '')[:2000])
        st.caption('Aspects block (`aspects_full`):')
        st.text(str(row.get('aspects_full') or '')[:4000])

    # ── Field mapping ─────────────────────────────────────────────────────────

    st.divider()
    st.subheader('Map profile fields to HubSpot properties')
    st.caption(
        'Company name → `name` and website → `domain` are always mapped. '
        'For each profile field, choose an existing HubSpot company property '
        f'or keep "{_CREATE_OPT}" to auto-create the `matcher_*` property named '
        'in the dropdown help. Per-aspect columns are listed after the rolled-up '
        'fields and are off by default. Uncheck **Import** to leave a field out.'
    )

    max_aspects = max(
        (int(n) for n in pd.to_numeric(df['n_aspects'], errors='coerce')
         .fillna(0).tolist()),
        default=0,
    )
    profile_specs = [
        spec for spec in (_PROFILE_BASE_FIELDS
                          + _profile_aspect_fields(min(max_aspects, ap.MAX_ASPECTS)))
        if spec[0] in df.columns
    ]

    prop_options = _hs_property_options()

    prof_col_map, create_props = _mapping_editor(
        profile_specs,
        prop_options,
        key=f'hs_prof_map_{run_id}',
        label_col='Profile field',
        height=460,
    )

    n_create = len(create_props)
    st.caption(
        f'**{len(prof_col_map)}** field{"s" if len(prof_col_map) != 1 else ""} '
        f'will be imported for **{len(df):,}** client'
        f'{"s" if len(df) != 1 else ""}'
        + (f' — {n_create} new propert{"ies" if n_create != 1 else "y"} '
           f'will be created.' if n_create else '.')
    )

    # ── Import ────────────────────────────────────────────────────────────────

    st.divider()
    if not st.button('▶ Import to HubSpot', type='primary'):
        st.stop()

    try:
        if create_props:
            with st.spinner('Creating HubSpot custom properties…'):
                created_props = _ensure_properties(
                    extra_props=create_props, include_defaults=False)
            if created_props:
                st.info(f'Created {len(created_props)} new HubSpot property/ies: '
                        f'`{"`, `".join(created_props)}`')

        with st.spinner('Building CSV and submitting to HubSpot…'):
            import_id, row_count = _submit_import(
                df, run_id, extra_col_map=prof_col_map)

        st.session_state.hs_import_id   = import_id
        st.session_state.hs_import_rows = row_count
        st.success(f'Submitted — **{row_count:,}** companies · '
                   f'Import ID: `{import_id}`')
        time.sleep(1)
        st.rerun()

    except Exception as e:
        st.error(f'Failed to submit import: {e}')
        st.code(traceback.format_exc())

    st.stop()

# ═══ MODE: Matching run (original flow) ═══════════════════════════════════════

all_mapped_cols = (
    [src for src, _, _ in _STANDARD_COLS] +
    list(_CUSTOM_PROPS.keys())
)
import_cols = [c for c in all_mapped_cols if c in df.columns]
st.caption(f'Columns that will be imported: `{"`, `".join(import_cols)}`')
st.dataframe(df[import_cols].head(50), use_container_width=True, hide_index=True)

# ── Optional custom columns ───────────────────────────────────────────────────

extra_df_cols = [
    c for c in df.columns
    if c not in _KNOWN_COLS
    and c not in _IMPORT_EXCLUDE
    and not c.startswith('_')
]

selected_extras: dict[str, tuple] = {}
if extra_df_cols:
    st.divider()
    st.subheader('Optional custom columns')
    st.caption(
        'These columns were added during topic import and are not part of the standard HubSpot property set. '
        'Check the ones you want to include — HubSpot properties will be created automatically.'
    )
    seen_hs_names = set()
    for col in extra_df_cols:
        # HubSpot property internal names must be lowercase (letters/digits/_)
        hs_name = 'matcher_' + re.sub(r'[^a-z0-9_]', '_', col.lower())
        if hs_name in seen_hs_names:
            st.warning(f'Skipping `{col}` — it maps to `{hs_name}`, already used by another column.')
            continue
        seen_hs_names.add(hs_name)
        if st.checkbox(
            f'`{col}` → `{hs_name}`',
            value=True,
            key=f'hs_extra_{col}',
            help=f'Creates HubSpot company property `{hs_name}` (label: "{_col_to_label(col)}") if it does not exist.',
        ):
            selected_extras[col] = (hs_name, _col_to_label(col), 'string', 'text')

# ── Import ────────────────────────────────────────────────────────────────────

st.divider()
if not st.button('▶ Import to HubSpot', type='primary'):
    st.stop()

extra_col_map_flat = {csv_col: tup[0] for csv_col, tup in selected_extras.items()}

try:
    with st.spinner('Ensuring HubSpot custom properties exist…'):
        created_props = _ensure_properties(extra_props=selected_extras or None)
    if created_props:
        st.info(f'Created {len(created_props)} new HubSpot property/ies: `{"`, `".join(created_props)}`')

    with st.spinner('Building CSV and submitting to HubSpot…'):
        import_id, row_count = _submit_import(df, run_id, extra_col_map=extra_col_map_flat or None)

    st.session_state.hs_import_id   = import_id
    st.session_state.hs_import_rows = row_count
    st.success(f'Submitted — **{row_count:,}** contacts · Import ID: `{import_id}`')
    time.sleep(1)
    st.rerun()

except Exception as e:
    st.error(f'Failed to submit import: {e}')
    st.code(traceback.format_exc())
