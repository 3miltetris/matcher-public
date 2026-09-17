"""
Shared Streamlit-side plumbing
------------------------------
The single home for the GCS / Cloud Run boilerplate that was copy-pasted
across views/: 17 spellings of the storage-client builder, the bucket name
written out in 14 files, and byte-identical write-config / trigger-job /
poll-status helpers in 7.

Streamlit-only, like access_control.py — it reads st.secrets and is never
imported by anything under jobs/. Note the deliberate non-coupling with
src/modules/aspect_profile.py: that module is Streamlit-free because the
Cloud Run jobs import it, so it keeps its own BUCKET/CLIENTS_PREFIX rather
than importing them from here. Two definitions of the literal is the
intended end state, down from fourteen.
"""

import io
import json

import pandas as pd
import streamlit as st
from google.cloud import run_v2, storage
from google.oauth2 import service_account

# ── Constants ──────────────────────────────────────────────────────────────

BUCKET          = 'cc-matcher-bucket-jeg-v1'

TOPICS_PREFIX   = 'data/all-topics/processed/'
AWARDS_PREFIX   = 'data/all-topics/awards/'
CONTACTS_PREFIX = 'data/all-contacts/'
CLIENTS_PREFIX  = 'data/all-contacts/clients/'
RESUMES_PREFIX  = 'data/resumes/'

_JOB_PARENT = 'projects/cc-matcher-v1/locations/us-central1/jobs/'


def job_name(job: str) -> str:
    """'sam-gov-job' -> the fully-qualified Cloud Run Jobs resource name.

    An already-qualified name is returned unchanged, so callers that still
    hold a full 'projects/.../jobs/x' constant can pass it straight through.
    """
    return job if job.startswith('projects/') else f'{_JOB_PARENT}{job}'


# ── Auth ───────────────────────────────────────────────────────────────────

def get_credentials():
    """Service-account credentials from st.secrets.

    Kept separate from get_storage_client() because run_v2.JobsClient needs
    the raw credentials object, not a storage client.
    """
    return service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )


def get_storage_client() -> storage.Client:
    return storage.Client(credentials=get_credentials())


# ── GCS reads ──────────────────────────────────────────────────────────────

def list_prefixes(client: storage.Client, prefix: str) -> list[str]:
    """The immediate sub-'folders' of a prefix, e.g. the agency short-codes
    under data/all-topics/processed/. The blobs iterator must be consumed
    before .prefixes is populated."""
    try:
        blobs = client.list_blobs(BUCKET, prefix=prefix, delimiter='/')
        list(blobs)
        return sorted(p.replace(prefix, '').strip('/') for p in blobs.prefixes)
    except Exception as e:
        st.error(f'Failed to list GCS prefixes under `{prefix}`: {e}')
        return []


def load_parquets_from_prefix(
    client: storage.Client, prefix: str
) -> pd.DataFrame:
    """Concatenate every .parquet under a prefix. Empty frame if none."""
    frames = [
        pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        for blob in client.list_blobs(BUCKET, prefix=prefix)
        if blob.name.endswith('.parquet')
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Cloud Run job plumbing ─────────────────────────────────────────────────

def write_job_config(
    client: storage.Client, cfg_prefix: str, config: dict
) -> str:
    """Stage a job config blob and return its path (the job's only argv)."""
    blob_path = f"{cfg_prefix}{config['run_id']}.json"
    client.bucket(BUCKET).blob(blob_path).upload_from_string(
        json.dumps(config), content_type='application/json'
    )
    return blob_path


def trigger_job(credentials, job: str, config_blob_path: str) -> None:
    """Execute a Cloud Run Job with the config blob path as its argument.

    `job` is the short name ('drive-sync-job'). This uses runWithOverrides,
    which requires roles/run.admin on the job for matcher-app@ — plain
    roles/run.invoker is not enough.
    """
    run_v2.JobsClient(credentials=credentials).run_job(
        request=run_v2.RunJobRequest(
            name=job_name(job),
            overrides=run_v2.RunJobRequest.Overrides(
                container_overrides=[
                    run_v2.RunJobRequest.Overrides.ContainerOverride(
                        args=[config_blob_path]
                    )
                ]
            ),
        )
    )


def poll_status(
    client: storage.Client, status_prefix: str, run_id: str
) -> dict | None:
    """The job's status.json, or None if it has not been written yet."""
    blob = client.bucket(BUCKET).blob(f'{status_prefix}{run_id}/status.json')
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())
