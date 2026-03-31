"""
Bulk Matching
-------------
Writes a job config to GCS, triggers the Cloud Run matching job, then polls
GCS for status.json until the job completes.

The heavy lifting (scoring, AI validation, email pre-write) happens entirely
inside the Cloud Run job — this page is config + monitoring only.
"""

import json
import time
import traceback
from datetime import datetime

import streamlit as st
from google.cloud import run_v2, storage
from google.oauth2 import service_account

# ── Constants ────────────────────────────────────────────────────────────────

_BUCKET            = 'cc-matcher-bucket-jeg-v1'
_CONTACTS_PREFIX   = 'data/all-contacts/'
_TOPICS_PREFIX     = 'data/all-topics/processed/'
_RESULTS_PREFIX    = 'matching-results/'
_JOB_CONFIGS_PREFIX = 'job-configs/'
_MIN_THRESHOLD     = 0.82
_JOB_NAME          = 'projects/cc-matcher-v1/locations/us-central1/jobs/matching-job'
_POLL_INTERVAL     = 8  # seconds between status checks


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


# ── Job helpers ───────────────────────────────────────────────────────────────

def _write_job_config(client: storage.Client, config: dict) -> str:
    """Upload config JSON to GCS, return blob path."""
    blob_path = f'{_JOB_CONFIGS_PREFIX}{config["run_id"]}.json'
    client.bucket(_BUCKET).blob(blob_path).upload_from_string(
        json.dumps(config), content_type='application/json'
    )
    return blob_path


def _trigger_job(credentials, config_blob_path: str) -> None:
    """Fire-and-forget: start the Cloud Run job with the config blob path as arg."""
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
    """Return status dict if status.json exists, else None."""
    blob = client.bucket(_BUCKET).blob(f'{_RESULTS_PREFIX}{run_id}/status.json')
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


# ── Session state ─────────────────────────────────────────────────────────────

for _k in ['bm_active_run', 'bm_run_summary']:
    if _k not in st.session_state:
        st.session_state[_k] = None


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

# ── Section 2 · Grant agencies ────────────────────────────────────────────────

st.divider()
st.subheader('2 · Select grant agencies')

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
    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config = {
        'run_id':         run_id,
        'threshold':      float(threshold),
        'top_k':          int(top_k),
        'sources':        selected_sources,
        'agencies':       selected_agencies,
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
    c1.metric('Rows saved',        f'{summary["total_saved"]:,}')
    c2.metric('Total candidates',  f'{summary["total_candidates"]:,}')
    c3.metric('Segments',          summary['segments'])
