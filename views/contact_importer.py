"""
Contact Importer
-----------------
Upload a lead spreadsheet from any source, map columns to standard fields,
deduplicate against existing GCS records, then trigger a Cloud Run job
that scrapes company websites, summarises with GPT-3.5-turbo, embeds with
text-embedding-ada-002, and saves to GCS under data/all-contacts/{source}/.

The Cloud Run job (contact-import-job) writes a status.json on completion
so this view polls without holding a long Streamlit connection.
"""

import io
import json
import re
import time
import traceback
from datetime import datetime

import pandas as pd
import requests
import tldextract
import streamlit as st
from google.cloud import run_v2, storage
from google.oauth2 import service_account

import src.modules.finance_research as fr   # Deep Research models + cost constants

# ── Constants ──────────────────────────────────────────────────────────────────

_BUCKET          = 'cc-matcher-bucket-jeg-v1'
_CONTACTS_ROOT   = 'data/all-contacts/'
_UPLOAD_PREFIX   = 'contact-import-uploads/'
_CONFIG_PREFIX   = 'contact-import-configs/'
_STATUS_PREFIX   = 'contact-import-jobs/'
_JOB_NAME        = 'projects/cc-matcher-v1/locations/us-central1/jobs/contact-import-job'
_POLL_INTERVAL   = 10   # seconds between status checks

_SOURCE_OPTIONS = ['apollo', 'sba', 'free_alert', 'hubspot', 'custom…']

_HS_BASE = 'https://api.hubapi.com'

_FIELD_CANDIDATES: dict[str, list[str]] = {
    'companyWebsite': ['website', 'website url', 'url', 'company website', 'companywebsite', 'web', 'homepage', 'domain'],
    'companyName':    ['company', 'company name', 'companyname', 'organization', 'org name', 'account name'],
    'state':          ['state', 'state/province', 'state/territory', 'region', 'province'],
    'firstName':      ['first name', 'firstname', 'first_name', 'given name'],
    'lastName':       ['last name', 'lastname', 'last_name', 'surname', 'family name'],
    'email':          ['email', 'email address', 'work email', 'e-mail'],
    'phone':          ['phone', 'phone number', 'mobile', 'telephone', 'cell'],
    'segment':        ['industry', 'segment', 'vertical', 'sector'],
}


def _detect_col(columns: list[str], field: str) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    for candidate in _FIELD_CANDIDATES[field]:
        if candidate in lower:
            return lower[candidate]
    return None


# ── Text cleanup ──────────────────────────────────────────────────────────────

_HYPERLINK_RE = re.compile(r'=HYPERLINK\s*\(\s*"[^"]*"\s*,\s*"([^"]*)"\s*\)', re.IGNORECASE)

def _strip_hyperlink(val) -> str:
    """Extract display text from an Excel HYPERLINK formula; return text unchanged otherwise."""
    text = str(val) if not isinstance(val, str) else val
    if not text or text.lower() in ('nan', 'none', ''):
        return ''
    m = _HYPERLINK_RE.match(text.strip())
    return m.group(1) if m else text


# ── URL helpers ────────────────────────────────────────────────────────────────

def _normalize_url(url) -> str:
    url = str(url or '').strip()
    if not url or url.lower() in ('nan', 'none', ''):
        return ''
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url


def _bare_domain(url: str) -> str:
    ext = tldextract.extract(str(url))
    return f'{ext.domain}.{ext.suffix}'.lower() if ext.domain else ''


# ── HubSpot helpers ────────────────────────────────────────────────────────────
# Requires Private App scopes: crm.lists.read + crm.objects.companies.read

def _hs_headers() -> dict:
    return {
        'Authorization': f"Bearer {st.secrets['hubspot_api_key']}",
        'Content-Type':  'application/json',
    }


def _hs_fetch_company_lists() -> list[dict]:
    """All company lists in the portal (objectTypeId 0-2), paged."""
    lists: list[dict] = []
    offset = 0
    while True:
        resp = requests.post(
            f'{_HS_BASE}/crm/v3/lists/search',
            headers=_hs_headers(),
            json={'query': '', 'objectTypeId': '0-2', 'count': 250, 'offset': offset},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        lists.extend(
            l for l in data.get('lists', [])
            if str(l.get('objectTypeId', '0-2')) == '0-2'
        )
        if not data.get('hasMore'):
            break
        offset = data.get('offset', 0)
    return lists


def _hs_fetch_list_company_ids(list_id: str) -> list[str]:
    """All record IDs in a list, paged."""
    ids: list[str] = []
    after = None
    while True:
        params: dict = {'limit': 250}
        if after:
            params['after'] = after
        resp = requests.get(
            f'{_HS_BASE}/crm/v3/lists/{list_id}/memberships',
            headers=_hs_headers(), params=params, timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        ids.extend(str(r['recordId']) for r in data.get('results', []))
        after = (data.get('paging') or {}).get('next', {}).get('after')
        if not after:
            break
    return ids


def _hs_fetch_companies(company_ids: list[str]) -> pd.DataFrame:
    """Batch-read companies → DataFrame with the standard contact columns
    (contact-level fields stay empty — company lists carry companies only)."""
    rows = []
    for i in range(0, len(company_ids), 100):
        chunk = company_ids[i:i + 100]
        resp = requests.post(
            f'{_HS_BASE}/crm/v3/objects/companies/batch/read',
            headers=_hs_headers(),
            json={
                'inputs':     [{'id': cid} for cid in chunk],
                'properties': ['name', 'domain', 'website', 'state', 'industry', 'phone'],
            },
            timeout=60,
        )
        resp.raise_for_status()
        for r in resp.json().get('results', []):
            p = r.get('properties') or {}
            rows.append({
                'companyWebsite': str(p.get('domain') or p.get('website') or '').strip(),
                'companyName':    str(p.get('name') or '').strip(),
                'state':          str(p.get('state') or '').strip(),
                'segment':        str(p.get('industry') or '').strip(),
                'firstName':      '',
                'lastName':       '',
                'email':          '',
                'phone':          str(p.get('phone') or '').strip(),
            })
    return pd.DataFrame(rows)


# ── GCS / Cloud Run helpers ────────────────────────────────────────────────────

def _get_credentials():
    return service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )


def _get_storage_client() -> storage.Client:
    return storage.Client(credentials=_get_credentials())


def _load_existing_domains(
    client: storage.Client, source: str, all_sources: bool = False
) -> set[str]:
    prefix  = _CONTACTS_ROOT if all_sources else f'{_CONTACTS_ROOT}{source}/'
    blobs   = client.list_blobs(_BUCKET, prefix=prefix)
    domains: set[str] = set()
    for blob in blobs:
        if not blob.name.endswith('.parquet'):
            continue
        try:
            df = pd.read_parquet(
                io.BytesIO(blob.download_as_bytes()),
                columns=['companyWebsite'],
            )
            for url in df['companyWebsite'].dropna().astype(str):
                d = _bare_domain(url)
                if d:
                    domains.add(d)
        except Exception:
            pass
    return domains


def _upload_raw_file(client: storage.Client, blob_path: str, data: bytes) -> None:
    client.bucket(_BUCKET).blob(blob_path).upload_from_string(data)


def _write_config(client: storage.Client, config: dict) -> str:
    blob_path = f"{_CONFIG_PREFIX}{config['run_id']}.json"
    client.bucket(_BUCKET).blob(blob_path).upload_from_string(
        json.dumps(config), content_type='application/json'
    )
    return blob_path


def _trigger_job(credentials, config_blob_path: str) -> None:
    job_client = run_v2.JobsClient(credentials=credentials)
    job_client.run_job(
        request=run_v2.RunJobRequest(
            name=_JOB_NAME,
            overrides=run_v2.RunJobRequest.Overrides(
                container_overrides=[
                    run_v2.RunJobRequest.Overrides.ContainerOverride(
                        args=[config_blob_path]
                    )
                ]
            ),
        )
    )


def _poll_status(client: storage.Client, run_id: str) -> dict | None:
    blob = client.bucket(_BUCKET).blob(f'{_STATUS_PREFIX}{run_id}/status.json')
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


# ── Session state init ─────────────────────────────────────────────────────────

for _k in ('ci_raw_df', 'ci_file_bytes', 'ci_file_ext', 'ci_deduped_df',
           'ci_dedup_source', 'ci_dedup_url_col', 'ci_dedup_allsrc',
           'ci_active_run', 'ci_hs_lists'):
    if _k not in st.session_state:
        st.session_state[_k] = None


# ── Page ───────────────────────────────────────────────────────────────────────

st.title('📋 Contact Importer')
st.caption(
    'Upload a lead spreadsheet (Apollo, SBA, or any source), map columns, '
    'scrape company websites, and add contacts to the matching database.'
)

# ── Active run polling (shown above everything else when a job is running) ─────

if st.session_state.ci_active_run:
    run_id = st.session_state.ci_active_run

    st.subheader('🔄 Import job in progress')
    st.caption(f'Run ID: `{run_id}`')

    try:
        gcs    = _get_storage_client()
        status = _poll_status(gcs, run_id)

        if status is None:
            st.info(f'Job is running… checking again in {_POLL_INTERVAL}s.')
            if st.button('Cancel monitoring (job keeps running)'):
                st.session_state.ci_active_run = None
                st.rerun()
            time.sleep(_POLL_INTERVAL)
            st.rerun()

        elif status.get('error'):
            st.error('Import job failed.')
            st.code(status['error'], language='text')
            st.session_state.ci_active_run = None

        else:
            st.success('Import job completed.')
            is_research = status.get('profile_method') == 'deep_research'
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Rows fetched',   f"{status.get('rows_fetched', 0):,}")
            c2.metric('After dedup',    f"{status.get('rows_after_dedup', 0):,}")
            c3.metric('Researched OK' if is_research else 'Scraped OK',
                      f"{status.get('rows_scraped_ok', 0):,}")
            c4.metric('Contacts saved', f"{status.get('rows_saved', 0):,}")
            if is_research:
                st.caption(
                    f"Deep Research ({status.get('research_model', '?')}): "
                    f"{status.get('companies_research_ok', 0)}/"
                    f"{status.get('companies_researched', 0)} companies researched — "
                    f"cost ${status.get('research_cost_usd', 0):,.2f}"
                )
            if status.get('gcs_path'):
                st.caption(f"Saved → `{status['gcs_path']}`")
            st.session_state.ci_active_run = None

    except Exception as e:
        st.error(f'Error polling status: {e}')
        st.code(traceback.format_exc())
        if st.button('Stop monitoring'):
            st.session_state.ci_active_run = None
            st.rerun()

    st.stop()

# Resume a previous run ────────────────────────────────────────────────────────

with st.expander('Resume monitoring a previous import job'):
    resume_id = st.text_input(
        'Run ID',
        key='ci_resume_run_id',
        placeholder='contact_import_2026-06-25_10-30-00_apollo',
    )
    if st.button('Check status') and resume_id.strip():
        st.session_state.ci_active_run = resume_id.strip()
        st.rerun()

# ── 1 · Input source ───────────────────────────────────────────────────────────

st.subheader('1 · Input source')


def _reset_input_state():
    for k in ('ci_raw_df', 'ci_file_bytes', 'ci_file_ext', 'ci_deduped_df',
              'ci_dedup_source', 'ci_dedup_url_col', 'ci_dedup_allsrc'):
        st.session_state[k] = None


input_mode = st.radio(
    'Where are the leads coming from?',
    options=['upload', 'hubspot'],
    format_func=lambda m: {
        'upload':  '📄 Upload spreadsheet',
        'hubspot': '🟠 HubSpot company list',
    }[m],
    horizontal=True,
    key='ci_input_mode',
    on_change=_reset_input_state,
)

if input_mode == 'upload':
    uploaded = st.file_uploader(
        'Upload CSV or Excel',
        type=['csv', 'xlsx', 'xls'],
        label_visibility='collapsed',
    )

    if uploaded:
        try:
            if uploaded.name.endswith('.xlsx'):
                raw      = pd.read_excel(uploaded, dtype=str)
                file_ext = '.xlsx'
            elif uploaded.name.endswith('.xls'):
                raw      = pd.read_excel(uploaded, dtype=str)
                file_ext = '.xls'
            else:
                try:
                    raw = pd.read_csv(uploaded, dtype=str, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded.seek(0)
                    raw = pd.read_csv(uploaded, dtype=str, encoding='latin-1')
                file_ext = '.csv'
            raw = raw.dropna(how='all')

            if (
                st.session_state.ci_raw_df is None
                or len(raw) != len(st.session_state.ci_raw_df)
            ):
                st.session_state.ci_raw_df        = raw
                st.session_state.ci_file_bytes    = uploaded.getvalue()
                st.session_state.ci_file_ext      = file_ext
                st.session_state.ci_deduped_df    = None
                st.session_state.ci_dedup_source  = None
                st.session_state.ci_dedup_url_col = None
        except Exception as e:
            st.error(f'Could not read file: {e}')

else:
    col_load, col_sel = st.columns([1, 3])
    with col_load:
        if st.button('🔄 Load lists', help='Fetch company lists from HubSpot'):
            try:
                with st.spinner('Loading HubSpot company lists…'):
                    st.session_state.ci_hs_lists = _hs_fetch_company_lists()
            except Exception as e:
                st.error(f'Failed to load HubSpot lists: {e}')

    hs_lists = st.session_state.ci_hs_lists or []
    if not hs_lists:
        st.caption(
            'Click **Load lists** to fetch company lists from HubSpot. '
            'The `hubspot_api_key` Private App needs the `crm.lists.read` '
            'and `crm.objects.companies.read` scopes.'
        )
    else:
        hs_by_id = {str(l.get('listId')): l for l in hs_lists}
        with col_sel:
            hs_selected = st.selectbox(
                'Company list',
                options=list(hs_by_id.keys()),
                format_func=lambda k: (
                    f"{hs_by_id[k].get('name') or '—'}"
                    f"  ·  {hs_by_id[k].get('processingType', '')}  ·  id {k}"
                ),
                key='ci_hs_list_sel',
            )
        if st.button('⬇️ Fetch companies from list', key='ci_hs_fetch'):
            try:
                with st.spinner('Fetching list members from HubSpot…'):
                    ids   = _hs_fetch_list_company_ids(hs_selected)
                    df_hs = _hs_fetch_companies(ids)
                if df_hs.empty:
                    st.warning('This list contains no companies.')
                else:
                    _reset_input_state()
                    st.session_state.ci_raw_df     = df_hs
                    st.session_state.ci_file_bytes = df_hs.to_csv(index=False).encode('utf-8')
                    st.session_state.ci_file_ext   = '.csv'
                    st.rerun()
            except Exception as e:
                st.error(f'Failed to fetch companies: {e}')
                st.code(traceback.format_exc())

if st.session_state.ci_raw_df is None:
    st.stop()

df_raw = st.session_state.ci_raw_df
st.caption(f'**{len(df_raw):,}** rows loaded.')
st.dataframe(df_raw.head(5), hide_index=True, use_container_width=True)

# ── 2 · Source & column mapping ────────────────────────────────────────────────

st.divider()
st.subheader('2 · Source & column mapping')

top_l, top_r = st.columns([1, 3])

with top_l:
    src_choice = st.selectbox('Lead source', _SOURCE_OPTIONS, key='ci_src_choice')
    if src_choice == 'custom…':
        source = st.text_input(
            'Custom name',
            placeholder='e.g. linkedin, event_leads',
            key='ci_src_custom',
        ).strip().lower().replace(' ', '_')
    else:
        source = src_choice

cols     = df_raw.columns.tolist()
none_opt = '— none —'
col_opts = [none_opt] + cols


def _sel(field: str, label: str, required: bool = False) -> str | None:
    detected = _detect_col(cols, field)
    idx      = col_opts.index(detected) if detected in col_opts else 0
    val      = st.selectbox(
        label + (' *' if required else ''),
        col_opts,
        index=idx,
        key=f'ci_map_{field}',
    )
    return val if val != none_opt else None


with top_r:
    mc1, mc2 = st.columns(2)
    with mc1:
        m_url     = _sel('companyWebsite', 'Website URL', required=True)
        m_name    = _sel('companyName',    'Company name')
        m_state   = _sel('state',          'State')
        m_segment = _sel('segment',        'Industry')
    with mc2:
        m_first = _sel('firstName', 'First name')
        m_last  = _sel('lastName',  'Last name')
        m_email = _sel('email',     'Email')
        m_phone = _sel('phone',     'Phone')

if not source:
    st.warning('Enter a source name to continue.')
    st.stop()
if not m_url:
    st.warning('Website URL column is required.')
    st.stop()

col_map = {
    'companyWebsite': m_url,
    'companyName':    m_name,
    'state':          m_state,
    'segment':        m_segment,
    'firstName':      m_first,
    'lastName':       m_last,
    'email':          m_email,
    'phone':          m_phone,
}

# ── 3 · Deduplicate ────────────────────────────────────────────────────────────

st.divider()
st.subheader('3 · Deduplicate')

dedup_all = st.checkbox(
    'Check against **all** sources (entire contact database, not just this '
    "source's folder)",
    value=(input_mode == 'hubspot'),
    key='ci_dedup_all',
    help='A HubSpot list can contain companies already imported under any '
         'source, so all-source checking is the default for HubSpot pulls. '
         'Note: the job re-dedups at runtime using this same scope.',
)

# Invalidate dedup if source, URL column, or scope changed since last check
if st.session_state.ci_deduped_df is not None and (
    st.session_state.ci_dedup_source  != source
    or st.session_state.ci_dedup_url_col != m_url
    or st.session_state.ci_dedup_allsrc  != dedup_all
):
    st.session_state.ci_deduped_df = None


def _build_mapped_df() -> pd.DataFrame:
    out = pd.DataFrame()
    for std_col, src_col in col_map.items():
        if src_col and src_col in df_raw.columns:
            out[std_col] = df_raw[src_col].astype(str).apply(_strip_hyperlink)
        else:
            out[std_col] = ''
    out['companyWebsite'] = out['companyWebsite'].apply(_normalize_url)
    return out[out['companyWebsite'] != ''].reset_index(drop=True)


mapped_df = _build_mapped_df()
no_url    = len(df_raw) - len(mapped_df)

if no_url:
    st.caption(f'{no_url:,} row(s) skipped — no URL value.')

if mapped_df.empty:
    st.error('No rows with a valid URL — cannot continue.')
    st.stop()

if st.button('🔍 Check for duplicates', key='ci_dedup_btn'):
    scope_label = _CONTACTS_ROOT if dedup_all else f'{_CONTACTS_ROOT}{source}/'
    with st.spinner(f'Checking existing records in {scope_label}…'):
        try:
            existing = _load_existing_domains(
                _get_storage_client(), source, all_sources=dedup_all
            )
            mask = mapped_df['companyWebsite'].apply(
                lambda u: _bare_domain(u) not in existing
            )
            st.session_state.ci_deduped_df    = mapped_df[mask].reset_index(drop=True)
            st.session_state.ci_dedup_source  = source
            st.session_state.ci_dedup_url_col = m_url
            st.session_state.ci_dedup_allsrc  = dedup_all
            st.rerun()
        except Exception as e:
            st.error(f'Dedup check failed: {e}')

if st.session_state.ci_deduped_df is None:
    st.caption(f'**{len(mapped_df):,}** rows with valid URLs ready to check.')
    st.stop()

deduped_df = st.session_state.ci_deduped_df
n_exist    = len(mapped_df) - len(deduped_df)

dm1, dm2, dm3 = st.columns(3)
dm1.metric('Valid URLs',     f'{len(mapped_df):,}')
dm2.metric('Already stored', f'{n_exist:,}')
dm3.metric('New to import',  f'{len(deduped_df):,}')

if deduped_df.empty:
    st.success('All URLs are already in the contact store — nothing new to import.')
    st.stop()

# ── 4 · Trigger import job ────────────────────────────────────────────────────

st.divider()
st.subheader('4 · Run import job')

profile_method = st.radio(
    'Company profiling method',
    options=['scrape', 'deep_research'],
    format_func=lambda m: {
        'scrape':        '🌐 Website scrape + GPT summary (standard)',
        'deep_research': '🔬 Deep Research — technology focus',
    }[m],
    help='Standard: scrape each website and summarise with GPT-3.5-turbo — '
         'fast and near-free. Deep Research: one OpenAI web-research task per '
         'unique company producing a full technology/R&D profile '
         '(stored in technology_data/technology_summary columns); the matching '
         'summary and embedding are built from the researched technology. '
         'Slower (2–10 min per company, run in parallel) and costs real money.',
    key='ci_profile_method',
)

research_model = None
confirmed      = True

if profile_method == 'deep_research':
    n_companies = int(deduped_df['companyWebsite'].apply(_bare_domain).nunique())
    research_model = st.radio(
        'Deep Research model',
        options=fr.DEEP_RESEARCH_MODELS,
        index=1,   # Terra — recommended balance of cost and depth
        format_func=lambda m: f'{m}  ({fr.EST_COST_LABEL[m]})',
        key='ci_research_model',
    )
    est_total = fr.EST_COST_PER_COMPANY[research_model] * n_companies
    st.metric(
        'Estimated research cost',
        f'~${est_total:,.0f}',
        help=f'{n_companies:,} unique companies × {fr.EST_COST_LABEL[research_model]}. '
             'Actual cost is computed from token usage and reported in the job status.',
    )
    if est_total > fr.COST_CONFIRM_THRESHOLD_USD:
        confirmed = st.checkbox(
            f'I understand this import may cost roughly ${est_total:,.0f} '
            f'({n_companies:,} companies × {fr.EST_COST_LABEL[research_model]}).',
            key='ci_cost_confirm',
        )
    st.caption(
        f'**{len(deduped_df):,}** new contacts across **{n_companies:,}** unique companies '
        f'will be profiled by Deep Research (technology focus) and embedded '
        f'(text-embedding-ada-002) by a Cloud Run job. Companies whose research '
        f'fails or exceeds the ~100-minute budget are skipped (re-importable later). '
        f'The job re-deduplicates at runtime as a safety check.'
    )
else:
    st.caption(
        f'**{len(deduped_df):,}** new contacts will be scraped, summarised (GPT-3.5-turbo), '
        f'and embedded (text-embedding-ada-002) by a Cloud Run job. '
        f'The job re-deduplicates at runtime as a safety check, so the final count may differ slightly.'
    )

if st.button('🚀 Start import job', type='primary', key='ci_trigger_btn',
             disabled=not confirmed):
    file_bytes = st.session_state.ci_file_bytes
    file_ext   = st.session_state.ci_file_ext or '.csv'
    timestamp  = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_id     = f'contact_import_{timestamp}_{source}'

    try:
        with st.spinner('Staging file to GCS…'):
            gcs            = _get_storage_client()
            file_blob_path = f'{_UPLOAD_PREFIX}{run_id}{file_ext}'
            _upload_raw_file(gcs, file_blob_path, file_bytes)

        with st.spinner('Writing job config…'):
            config = {
                'run_id':            run_id,
                'source':            source,
                'file_ext':          file_ext,
                'csv_blob_path':     file_blob_path,
                'col_map':           col_map,
                'profile_method':    profile_method,
                'dedup_all_sources': bool(dedup_all),
            }
            if profile_method == 'deep_research':
                config['research_model'] = research_model
            config_blob_path = _write_config(gcs, config)

        with st.spinner('Triggering Cloud Run job…'):
            _trigger_job(_get_credentials(), config_blob_path)

        st.session_state.ci_active_run = run_id
        st.rerun()

    except Exception as e:
        st.error(f'Failed to start job: {e}')
        st.code(traceback.format_exc())
