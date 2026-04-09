"""
Bulk Matching
-------------
Writes a job config to GCS, triggers the Cloud Run matching job, then polls
GCS for status.json until the job completes.

The heavy lifting (scoring, AI validation, email pre-write) happens entirely
inside the Cloud Run job — this page handles config, topic preview/filtering,
and monitoring only.
"""

import io
import json
import time
import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import run_v2, storage
from google.oauth2 import service_account

# ── Constants ────────────────────────────────────────────────────────────────

_BUCKET             = 'cc-matcher-bucket-jeg-v1'
_CONTACTS_PREFIX    = 'data/all-contacts/'
_TOPICS_PREFIX      = 'data/all-topics/processed/'
_RESULTS_PREFIX     = 'matching-results/'
_JOB_CONFIGS_PREFIX = 'job-configs/'
_MIN_THRESHOLD      = 0.82
_JOB_NAME           = 'projects/cc-matcher-v1/locations/us-central1/jobs/matching-job'
_POLL_INTERVAL      = 8  # seconds between status checks


# ── GCS / auth helpers ────────────────────────────────────────────────────────

def _get_credentials():
    return service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )


def _get_storage_client() -> storage.Client:
    return storage.Client(credentials=_get_credentials())


def _list_prefixes(client: storage.Client, prefix: str) -> list[str]:
    try:
        blobs = client.list_blobs(_BUCKET, prefix=prefix, delimiter='/')
        list(blobs)
        return sorted(p.replace(prefix, '').strip('/') for p in blobs.prefixes)
    except Exception as e:
        st.error(f'Failed to list GCS prefixes under `{prefix}`: {e}')
        return []


def _load_topics(client: storage.Client, agencies: list[str]) -> pd.DataFrame:
    frames = []
    for agency in agencies:
        prefix = f'{_TOPICS_PREFIX}{agency}/'
        for blob in client.list_blobs(_BUCKET, prefix=prefix):
            if blob.name.endswith('.parquet'):
                df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
                df['broad_agency'] = agency
                frames.append(df)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return pd.concat(frames, ignore_index=True)


# ── Filter helper ─────────────────────────────────────────────────────────────

def _apply_filters(df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    active = [f for f in filters if f['keyword'].strip() and f['column']]
    if not active:
        return df
    mask = df[active[0]['column']].astype(str).str.lower().str.contains(
        active[0]['keyword'].lower(), na=False
    )
    for f in active[1:]:
        m = df[f['column']].astype(str).str.lower().str.contains(
            f['keyword'].lower(), na=False
        )
        mask = (mask & m) if f['operator'] == 'AND' else (mask | m)
    return df[mask]


# ── Job helpers ───────────────────────────────────────────────────────────────

def _write_job_config(client: storage.Client, config: dict) -> str:
    blob_path = f'{_JOB_CONFIGS_PREFIX}{config["run_id"]}.json'
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
    blob = client.bucket(_BUCKET).blob(f'{_RESULTS_PREFIX}{run_id}/status.json')
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


# ── Session state ─────────────────────────────────────────────────────────────

for _k in ['bm_active_run', 'bm_run_summary', 'bm_topics_df']:
    if _k not in st.session_state:
        st.session_state[_k] = None
if 'bm_filters' not in st.session_state:
    st.session_state.bm_filters = [{'column': None, 'keyword': '', 'operator': 'AND'}]


# ── Page ──────────────────────────────────────────────────────────────────────

st.title('⚙️ Bulk Matching')
st.caption('Configure a matching run, trigger the Cloud Run job, and monitor progress here.')

# ── Active run: polling UI (blocks the rest of the page) ─────────────────────

if st.session_state.bm_active_run:
    active = st.session_state.bm_active_run
    run_id = active['run_id']

    st.subheader('🔄 Run in progress')
    st.caption(f'Run ID: `{run_id}`')

    try:
        gcs    = _get_storage_client()
        status = _poll_status(gcs, run_id)

        if status is None:
            st.info(f'Job is running… checking again in {_POLL_INTERVAL}s.')
            if st.button('Cancel monitoring (job keeps running)'):
                st.session_state.bm_active_run = None
                st.rerun()
            time.sleep(_POLL_INTERVAL)
            st.rerun()

        elif status.get('error'):
            st.error('Job failed.')
            st.code(status['error'])
            st.session_state.bm_active_run = None

        else:
            st.success(
                f'Run complete — **{status["total_saved"]:,}** rows saved across '
                f'**{status["segments"]}** segment(s) '
                f'(from {status["total_candidates"]:,} candidates)'
            )
            st.session_state.bm_run_summary = status
            st.session_state.bm_active_run  = None

    except Exception as e:
        st.error(f'Error polling status: {e}')
        st.code(traceback.format_exc())
        if st.button('Stop monitoring'):
            st.session_state.bm_active_run = None
            st.rerun()

    st.stop()


# ── Section 1 · Contact sources ───────────────────────────────────────────────

st.subheader('1 · Select contact sources')

gcs_client = _get_storage_client()

contact_sources = _list_prefixes(gcs_client, _CONTACTS_PREFIX)
if not contact_sources:
    st.warning('No contact sources found in GCS.')
    st.stop()

src_cols = st.columns(min(len(contact_sources), 6))
selected_sources = [
    src
    for i, src in enumerate(contact_sources)
    if src_cols[i % len(src_cols)].checkbox(src, value=True, key=f'bm_src_{src}')
]

# ── Section 2 · Grant topics ──────────────────────────────────────────────────

st.divider()
st.subheader('2 · Select grant topics')

grant_agencies = _list_prefixes(gcs_client, _TOPICS_PREFIX)
if not grant_agencies:
    st.warning('No grant agencies found in GCS.')
    st.stop()

ag_cols = st.columns(min(len(grant_agencies), 6))
selected_agencies = [
    ag
    for i, ag in enumerate(grant_agencies)
    if ag_cols[i % len(ag_cols)].checkbox(ag, value=True, key=f'bm_agency_{ag}')
]

if st.button('Load Topics', type='primary', disabled=not selected_agencies):
    with st.spinner('Loading grant topics…'):
        _df = _load_topics(gcs_client, selected_agencies)
    if _df.empty:
        st.warning('No topics found for the selected agencies.')
    else:
        if 'grant_summary' not in _df.columns and 'description' in _df.columns:
            _df = _df.rename(columns={'description': 'grant_summary'})
        _df = _df.drop(columns=['embeddings'], errors='ignore')
        st.session_state.bm_topics_df  = _df
        st.session_state.bm_filters    = [{'column': None, 'keyword': '', 'operator': 'AND'}]
        st.session_state.bm_run_summary = None
        st.success(f'Loaded **{len(_df):,}** grant topics.')

if st.session_state.bm_topics_df is not None:
    _topics_df      = st.session_state.bm_topics_df
    filterable_cols = list(_topics_df.columns)

    for f in st.session_state.bm_filters:
        if f['column'] not in filterable_cols:
            f['column'] = filterable_cols[0] if filterable_cols else None

    for i, f in enumerate(st.session_state.bm_filters):
        if i == 0:
            col_sel, kw_input, _, remove_col = st.columns([2, 3, 1, 0.5])
        else:
            op_col, col_sel, kw_input, remove_col = st.columns([1, 2, 3, 0.5])
            f['operator'] = op_col.radio(
                'op', ['AND', 'OR'],
                index=0 if f['operator'] == 'AND' else 1,
                key=f'bm_op_{i}', horizontal=True, label_visibility='collapsed'
            )
        f['column'] = col_sel.selectbox(
            'Column', filterable_cols,
            index=filterable_cols.index(f['column']) if f['column'] in filterable_cols else 0,
            key=f'bm_col_{i}', label_visibility='collapsed'
        )
        f['keyword'] = kw_input.text_input(
            'Keyword', value=f['keyword'],
            placeholder=f'Filter by {f["column"]}…',
            key=f'bm_kw_{i}', label_visibility='collapsed'
        )
        if remove_col.button('✕', key=f'bm_rm_{i}', disabled=len(st.session_state.bm_filters) == 1):
            st.session_state.bm_filters.pop(i)
            st.rerun()

    if st.button('+ Add filter', key='bm_add_filter'):
        st.session_state.bm_filters.append(
            {'column': filterable_cols[0], 'keyword': '', 'operator': 'AND'}
        )
        st.rerun()

    _filtered_preview = _apply_filters(_topics_df, st.session_state.bm_filters)
    st.caption(f'**{len(_filtered_preview):,}** topics match current filters — showing first 50')
    st.dataframe(_filtered_preview.head(50), use_container_width=True, hide_index=True)

# ── Section 3 · Run options ───────────────────────────────────────────────────

st.divider()
st.subheader('3 · Run options')

opt_left, opt_mid, _ = st.columns([1, 1, 2])
with opt_left:
    threshold = st.slider(
        'Similarity threshold',
        min_value=_MIN_THRESHOLD, max_value=1.0,
        value=_MIN_THRESHOLD, step=0.01,
    )
    top_k = st.number_input(
        'Top-K topics per contact',
        min_value=1, max_value=50,
        value=5, step=1,
    )
with opt_mid:
    ai_validation  = st.checkbox('AI validation',  value=True)
    prewrite_email = st.checkbox('Pre-write email', value=False)

can_run = bool(selected_sources) and bool(selected_agencies)

if st.button('▶ Run Matching', type='primary', disabled=not can_run):
    # Serialize active filters (only if topics have been loaded)
    active_filters = []
    if st.session_state.bm_topics_df is not None:
        active_filters = [
            f for f in st.session_state.bm_filters
            if f.get('keyword', '').strip() and f.get('column')
        ]

    ag_tag  = '-'.join(selected_agencies) if selected_agencies else 'none'
    src_tag = '-'.join(selected_sources)  if selected_sources  else 'none'
    run_id  = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_ag-{ag_tag}_src-{src_tag}"
    config = {
        'run_id':         run_id,
        'threshold':      float(threshold),
        'top_k':          int(top_k),
        'sources':        selected_sources,
        'agencies':       selected_agencies,
        'topic_filters':  active_filters,
        'ai_validation':  ai_validation,
        'prewrite_email': prewrite_email,
    }
    try:
        with st.spinner('Uploading job config to GCS…'):
            config_blob = _write_job_config(gcs_client, config)

        with st.spinner('Triggering Cloud Run job…'):
            _trigger_job(_get_credentials(), config_blob)

        st.session_state.bm_active_run  = {'run_id': run_id, 'config_blob': config_blob}
        st.session_state.bm_run_summary = None
        st.rerun()

    except Exception as e:
        st.error(f'Failed to trigger job: {e}')
        st.code(traceback.format_exc())

# ── Section 4 · Last run results ─────────────────────────────────────────────

if st.session_state.bm_run_summary:
    summary = st.session_state.bm_run_summary
    st.divider()
    st.subheader('4 · Last run results')
    st.caption(
        f'Run ID: `{summary["run_id"]}` · '
        f'GCS prefix: `{_RESULTS_PREFIX}{summary["run_id"]}/`'
    )
    c1, c2, c3 = st.columns(3)
    c1.metric('Rows saved',       f'{summary["total_saved"]:,}')
    c2.metric('Total candidates', f'{summary["total_candidates"]:,}')
    c3.metric('Segments',         summary['segments'])
