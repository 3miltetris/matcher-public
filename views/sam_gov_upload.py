"""
SAM.gov Upload
--------------
Configure a SAM.gov ingestion job (CSV upload or API fetch), trigger the
sam-gov-job Cloud Run job, and monitor progress here.

All heavy lifting (screening, dedup, summarization, embedding) runs in the
Cloud Run job so Streamlit can handle large batches without timeouts.
"""

import io
import json
import time
import traceback
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from google.cloud import run_v2, storage
from google.oauth2 import service_account

# ── Constants ──────────────────────────────────────────────────────────────────

_BUCKET             = 'cc-matcher-bucket-jeg-v1'
_STATUS_PREFIX      = 'sam-gov-jobs/'
_UPLOAD_PREFIX      = 'sam-gov-uploads/'
_CONFIG_PREFIX      = 'sam-gov-configs/'
_JOB_NAME           = 'projects/cc-matcher-v1/locations/us-central1/jobs/sam-gov-job'
_POLL_INTERVAL      = 10  # seconds between status checks

_NOTICE_TYPE_OPTIONS = {
    'Presolicitation':                'p',
    'Solicitation':                   'o',
    'Combined Synopsis/Solicitation': 'k',
    'Sources Sought':                 'r',
    'Special Notice':                 's',
}

_SAM_RESERVED_COLS = frozenset({
    'topic_number', 'agency', 'title', 'description', 'open_date', 'due_date',
    'scraped_at', 'sam_confidence', 'sam_reason', 'grant_summary', 'embeddings',
})

_CANDIDATES: dict[str, list[str]] = {
    'title':        ['title', 'opportunity title', 'solicitation title'],
    'description':  ['description', 'synopsis', 'description/synopsis'],
    'naics_desc':   ['naics desc', 'naics description', 'naics_desc', 'naics_description'],
    'notice_id':    ['notice id', 'notice_id', 'solicitation #', 'sol #', 'solicitation number'],
    'agency':       ['department/ind.agency', 'department', 'agency', 'department name'],
    'posted_date':  ['posted date', 'post date', 'posted_date'],
    'deadline':     ['response deadline', 'response_deadline', 'deadline', 'close date'],
}


# ── Auth / GCS helpers ─────────────────────────────────────────────────────────

def _get_credentials():
    return service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )


def _get_storage_client() -> storage.Client:
    return storage.Client(credentials=_get_credentials())


def _detect_col(columns: list[str], field: str) -> str | None:
    lower = {c.lower(): c for c in columns}
    for candidate in _CANDIDATES[field]:
        if candidate in lower:
            return lower[candidate]
    return None


# ── Job helpers ────────────────────────────────────────────────────────────────

def _upload_csv_to_gcs(client: storage.Client, df: pd.DataFrame, run_id: str) -> str:
    blob_path = f'{_UPLOAD_PREFIX}{run_id}.csv'
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    client.bucket(_BUCKET).blob(blob_path).upload_from_string(
        csv_bytes, content_type='text/csv'
    )
    return blob_path


def _write_job_config(client: storage.Client, config: dict) -> str:
    blob_path = f'{_CONFIG_PREFIX}{config["run_id"]}.json'
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


# ── Session state ──────────────────────────────────────────────────────────────

for _k in ['sam_raw_df', 'sam_col_map', 'sam_from_api', 'sam_api_params',
           'sam_active_run', 'sam_run_summary']:
    if _k not in st.session_state:
        st.session_state[_k] = None

if 'sam_custom_cols' not in st.session_state:
    st.session_state.sam_custom_cols = []


# ── Page ───────────────────────────────────────────────────────────────────────

st.title('🏛️ SAM.gov Upload')
st.caption(
    'Configure a SAM.gov ingestion run and trigger the Cloud Run job. '
    'Screening, deduplication, summarization, and embedding all run in the cloud — '
    'no timeouts, no size limits.'
)

# ── Active run: polling UI ─────────────────────────────────────────────────────

if st.session_state.sam_active_run:
    active = st.session_state.sam_active_run
    run_id = active['run_id']

    st.subheader('🔄 Job in progress')
    st.caption(f'Run ID: `{run_id}`')

    try:
        gcs    = _get_storage_client()
        status = _poll_status(gcs, run_id)

        if status is None:
            st.info(f'Job is running… checking again in {_POLL_INTERVAL}s.')
            if st.button('Cancel monitoring (job keeps running)'):
                st.session_state.sam_active_run = None
                st.rerun()
            time.sleep(_POLL_INTERVAL)
            st.rerun()

        elif status.get('error'):
            st.error('Job failed.')
            st.code(status['error'])
            st.session_state.sam_active_run = None

        else:
            st.success(
                f'Run complete — **{status.get("rows_saved", 0):,}** topics saved to the store.'
            )
            st.session_state.sam_run_summary = status
            st.session_state.sam_active_run  = None

    except Exception as e:
        st.error(f'Error polling status: {e}')
        st.code(traceback.format_exc())
        if st.button('Stop monitoring'):
            st.session_state.sam_active_run = None
            st.rerun()

    st.stop()


# ── Last run summary ───────────────────────────────────────────────────────────

if st.session_state.sam_run_summary:
    s = st.session_state.sam_run_summary
    st.success(f'Last run complete — run ID: `{s.get("run_id", "")}`')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Fetched',           f'{s.get("rows_fetched", 0):,}')
    c2.metric('Passed screening',  f'{s.get("rows_passed_screening", 0):,}')
    c3.metric('After dedup',       f'{s.get("rows_after_dedup", 0):,}')
    c4.metric('Saved',             f'{s.get("rows_saved", 0):,}')
    if s.get('gcs_path'):
        st.caption(f'Saved to `{s["gcs_path"]}`')
    st.divider()


# ── Section 1 · Load ───────────────────────────────────────────────────────────

st.subheader('1 · Load opportunities')

tab_csv, tab_api = st.tabs(['📄 Upload CSV', '🔌 Fetch from API'])

# ── Tab: Upload CSV ────────────────────────────────────────────────────────────

with tab_csv:
    files = st.file_uploader(
        'SAM.gov export CSV(s)',
        type='csv',
        accept_multiple_files=True,
        label_visibility='collapsed',
    )
    if files:
        frames = []
        for f in files:
            try:
                try:
                    frames.append(pd.read_csv(f, dtype=str, encoding='utf-8'))
                except UnicodeDecodeError:
                    f.seek(0)
                    frames.append(pd.read_csv(f, dtype=str, encoding='latin-1'))
            except Exception as e:
                st.error(f'Could not read **{f.name}**: {e}')
        if frames:
            combined = pd.concat(frames, ignore_index=True).dropna(how='all')
            if (
                st.session_state.sam_raw_df is None
                or st.session_state.sam_from_api
                or len(combined) != len(st.session_state.sam_raw_df)
            ):
                st.session_state.sam_raw_df    = combined
                st.session_state.sam_col_map   = None
                st.session_state.sam_from_api  = False
                st.session_state.sam_api_params = None
                st.session_state.sam_custom_cols = []

# ── Tab: Fetch from API ────────────────────────────────────────────────────────

with tab_api:
    sam_key = st.secrets.get('sam_gov_api_key')
    if not sam_key:
        st.warning(
            'Add `sam_gov_api_key` to `.streamlit/secrets.toml` to use API fetch. '
            'Register for a free key at **beta.sam.gov → Account Settings → API Keys**.'
        )
    else:
        today_date   = datetime.today().date()
        default_from = today_date - timedelta(days=30)

        col_l, col_r = st.columns(2)
        with col_l:
            api_date_from = st.date_input('Posted from', value=default_from, key='sam_api_date_from')
        with col_r:
            api_date_to = st.date_input('Posted to', value=today_date, key='sam_api_date_to')

        selected_type_labels = st.multiselect(
            'Notice types',
            list(_NOTICE_TYPE_OPTIONS.keys()),
            default=['Solicitation', 'Presolicitation', 'Sources Sought'],
            key='sam_api_notice_types',
        )

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            api_keyword = st.text_input(
                'Keyword filter (optional)',
                placeholder='e.g. AI, biotech, cybersecurity',
                key='sam_api_keyword',
            )
            api_max = st.number_input(
                'Max results (0 = no cap)',
                min_value=0,
                value=500,
                step=100,
                key='sam_api_max',
            )
        with col_r2:
            api_fetch_desc = st.checkbox(
                'Fetch full descriptions',
                value=True,
                key='sam_api_fetch_desc',
                help=(
                    'Makes one additional API call per opportunity to retrieve the full '
                    'synopsis text. Slower but significantly improves screening accuracy.'
                ),
            )

        if api_date_from > api_date_to:
            st.error('"Posted from" must be on or before "Posted to".')
        else:
            selected_type_codes = [_NOTICE_TYPE_OPTIONS[lbl] for lbl in selected_type_labels]
            st.session_state.sam_api_params = {
                'date_from':       api_date_from.strftime('%m/%d/%Y'),
                'date_to':         api_date_to.strftime('%m/%d/%Y'),
                'notice_types':    selected_type_codes,
                'keyword':         api_keyword.strip(),
                'max_results':     int(api_max),
                'fetch_desc':      bool(api_fetch_desc),
                'sam_gov_api_key': sam_key,
            }
            st.info(
                f'Configured: **{api_date_from}** → **{api_date_to}** · '
                f'types: {", ".join(selected_type_labels) or "all"} · '
                f'max: {int(api_max) or "no cap"}'
            )
            if st.session_state.sam_from_api is not True:
                st.session_state.sam_raw_df   = None
                st.session_state.sam_col_map  = None
                st.session_state.sam_from_api = True


# ── Determine mode and validate ────────────────────────────────────────────────

from_api  = bool(st.session_state.sam_from_api)
has_csv   = st.session_state.sam_raw_df is not None
has_api   = from_api and st.session_state.sam_api_params is not None

if not has_csv and not has_api:
    st.stop()

if has_csv:
    df_raw = st.session_state.sam_raw_df
    st.caption(f'**{len(df_raw):,}** rows loaded from CSV.')
    st.dataframe(df_raw.head(5), hide_index=True, use_container_width=True)


# ── Section 2 · Column mapping (CSV only) ────────────────────────────────────

col_map: dict = {}

if has_csv:
    st.divider()
    st.subheader('2 · Column mapping')
    st.caption('Confirm which columns map to each field — auto-detected where possible.')

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
            key=f'sam_map_{field}',
        )
        return val if val != none_opt else None

    left_col, right_col = st.columns(2)
    with left_col:
        m_title  = _sel('title',       'Title',              required=True)
        m_desc   = _sel('description', 'Description',        required=True)
        m_naics  = _sel('naics_desc',  'NAICS Descriptor')
        m_notice = _sel('notice_id',   'Notice ID')
    with right_col:
        m_agency = _sel('agency',      'Agency / Department')
        m_posted = _sel('posted_date', 'Posted Date')
        m_dl     = _sel('deadline',    'Response Deadline')

    col_map = {
        'title':       m_title,
        'description': m_desc,
        'naics_desc':  m_naics,
        'notice_id':   m_notice,
        'agency':      m_agency,
        'posted_date': m_posted,
        'deadline':    m_dl,
    }
    st.session_state.sam_col_map = col_map

    if not m_title or not m_desc:
        st.warning('Title and Description columns are required before proceeding.')
        st.stop()


# ── Section 3 · Custom columns ────────────────────────────────────────────────

st.divider()
st.subheader('3 · Custom columns (optional)')

with st.expander('Add extra metadata columns', expanded=bool(st.session_state.sam_custom_cols)):
    st.caption(
        'Tag every saved topic with campaign-specific metadata. '
        'Saved to the parquet and available as optional fields in HubSpot import.'
    )

    for entry in list(st.session_state.sam_custom_cols):
        sc1, sc2, sc3 = st.columns([2, 4, 1])
        with sc1:
            st.text(entry['name'])
        with sc2:
            entry['value'] = st.text_input(
                'Value', value=entry['value'],
                key=f'scfill_{entry["name"]}', label_visibility='collapsed',
            )
        with sc3:
            if st.button('✕', key=f'scrm_{entry["name"]}', use_container_width=True):
                st.session_state.sam_custom_cols = [
                    e for e in st.session_state.sam_custom_cols if e['name'] != entry['name']
                ]
                st.rerun()

    if st.session_state.sam_custom_cols:
        st.divider()

    sna1, sna2, sna3 = st.columns([2, 4, 1])
    with sna1:
        new_sc_name = st.text_input(
            'Column name', placeholder='e.g. campaign_name',
            key='new_sc_name', label_visibility='collapsed',
        )
    with sna2:
        new_sc_val = st.text_input(
            'Value', placeholder='e.g. Spring 2026',
            key='new_sc_val', label_visibility='collapsed',
        )
    with sna3:
        if st.button('Add', key='add_sc_btn', use_container_width=True):
            clean = new_sc_name.strip().replace(' ', '_').lower()
            existing_names = {e['name'] for e in st.session_state.sam_custom_cols}
            if (
                clean
                and clean not in _SAM_RESERVED_COLS
                and clean not in existing_names
            ):
                st.session_state.sam_custom_cols.append({'name': clean, 'value': new_sc_val.strip()})
                st.rerun()


# ── Section 4 · Run Job ────────────────────────────────────────────────────────

st.divider()
st.subheader('4 · Run job')

if has_csv:
    n_rows   = len(df_raw)
    est_mins = max(1, n_rows // 80)
    st.caption(
        f'**{n_rows:,}** rows from CSV will be screened and embedded by the Cloud Run job. '
        f'Estimated time: ~{est_mins} min.'
    )
else:
    params = st.session_state.sam_api_params
    max_r  = params.get('max_results', 0)
    n_rows = max_r if max_r else None
    if n_rows:
        est_mins = max(1, n_rows // 80)
        st.caption(
            f'The job will fetch up to **{n_rows:,}** opportunities from SAM.gov, '
            f'then screen and embed. Estimated time: ~{est_mins} min.'
        )
    else:
        st.caption(
            'The job will fetch all matching opportunities from SAM.gov, '
            'then screen and embed (uncapped — may take a while for large date ranges).'
        )

st.caption(
    'The job runs entirely in Google Cloud. You can close this tab — '
    'results will be available in the topic store when it finishes.'
)

if st.button('▶ Run SAM.gov Job', type='primary'):
    run_id = f"sam_gov_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    custom_cols_dict = {e['name']: e['value'] for e in st.session_state.sam_custom_cols}

    try:
        gcs   = _get_storage_client()
        creds = _get_credentials()

        if has_csv:
            with st.spinner('Uploading CSV to GCS…'):
                csv_blob = _upload_csv_to_gcs(gcs, df_raw, run_id)
            config = {
                'run_id':        run_id,
                'input_mode':    'csv',
                'csv_blob_path': csv_blob,
                'col_map':       st.session_state.sam_col_map,
                'custom_cols':   custom_cols_dict,
            }
        else:
            config = {
                'run_id':      run_id,
                'input_mode':  'api',
                'api_params':  st.session_state.sam_api_params,
                'custom_cols': custom_cols_dict,
            }

        with st.spinner('Writing job config to GCS…'):
            config_blob = _write_job_config(gcs, config)

        with st.spinner('Triggering Cloud Run job…'):
            _trigger_job(creds, config_blob)

        st.session_state.sam_active_run  = {'run_id': run_id, 'config_blob': config_blob}
        st.session_state.sam_run_summary = None
        st.rerun()

    except Exception as e:
        st.error(f'Failed to trigger job: {e}')
        st.code(traceback.format_exc())
