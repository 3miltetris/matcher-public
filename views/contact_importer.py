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
import time
import traceback
from datetime import datetime

import pandas as pd
import tldextract
import streamlit as st
from google.cloud import run_v2, storage
from google.oauth2 import service_account

# ── Constants ──────────────────────────────────────────────────────────────────

_BUCKET          = 'cc-matcher-bucket-jeg-v1'
_CONTACTS_ROOT   = 'data/all-contacts/'
_UPLOAD_PREFIX   = 'contact-import-uploads/'
_CONFIG_PREFIX   = 'contact-import-configs/'
_STATUS_PREFIX   = 'contact-import-jobs/'
_JOB_NAME        = 'projects/cc-matcher-v1/locations/us-central1/jobs/contact-import-job'
_POLL_INTERVAL   = 10   # seconds between status checks

_SOURCE_OPTIONS = ['apollo', 'sba', 'free_alert', 'custom…']

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


# ── GCS / Cloud Run helpers ────────────────────────────────────────────────────

def _get_credentials():
    return service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )


def _get_storage_client() -> storage.Client:
    return storage.Client(credentials=_get_credentials())


def _load_existing_domains(client: storage.Client, source: str) -> set[str]:
    prefix  = f'{_CONTACTS_ROOT}{source}/'
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
           'ci_dedup_source', 'ci_dedup_url_col', 'ci_active_run'):
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
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Rows fetched',   f"{status.get('rows_fetched', 0):,}")
            c2.metric('After dedup',    f"{status.get('rows_after_dedup', 0):,}")
            c3.metric('Scraped OK',     f"{status.get('rows_scraped_ok', 0):,}")
            c4.metric('Contacts saved', f"{status.get('rows_saved', 0):,}")
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

# ── 1 · Upload ─────────────────────────────────────────────────────────────────

st.subheader('1 · Upload spreadsheet')

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

# Invalidate dedup if source or URL column changed since last check
if st.session_state.ci_deduped_df is not None and (
    st.session_state.ci_dedup_source  != source
    or st.session_state.ci_dedup_url_col != m_url
):
    st.session_state.ci_deduped_df = None


# ── 3 · Deduplicate ────────────────────────────────────────────────────────────

st.divider()
st.subheader('3 · Deduplicate')


def _build_mapped_df() -> pd.DataFrame:
    out = pd.DataFrame()
    for std_col, src_col in col_map.items():
        if src_col and src_col in df_raw.columns:
            out[std_col] = df_raw[src_col].astype(str)
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
    with st.spinner(f'Checking existing records in {_CONTACTS_ROOT}{source}/…'):
        try:
            existing = _load_existing_domains(_get_storage_client(), source)
            mask = mapped_df['companyWebsite'].apply(
                lambda u: _bare_domain(u) not in existing
            )
            st.session_state.ci_deduped_df    = mapped_df[mask].reset_index(drop=True)
            st.session_state.ci_dedup_source  = source
            st.session_state.ci_dedup_url_col = m_url
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

st.caption(
    f'**{len(deduped_df):,}** new contacts will be scraped, summarised (GPT-3.5-turbo), '
    f'and embedded (text-embedding-ada-002) by a Cloud Run job. '
    f'The job re-deduplicates at runtime as a safety check, so the final count may differ slightly.'
)

if st.button('🚀 Start import job', type='primary', key='ci_trigger_btn'):
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
                'run_id':        run_id,
                'source':        source,
                'file_ext':      file_ext,
                'csv_blob_path': file_blob_path,
                'col_map':       col_map,
            }
            config_blob_path = _write_config(gcs, config)

        with st.spinner('Triggering Cloud Run job…'):
            _trigger_job(_get_credentials(), config_blob_path)

        st.session_state.ci_active_run = run_id
        st.rerun()

    except Exception as e:
        st.error(f'Failed to start job: {e}')
        st.code(traceback.format_exc())
