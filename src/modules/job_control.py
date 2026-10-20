"""
Streamlit-free Cloud Run job control
------------------------------------
Config write, job trigger, status read — extracted from ui_common.py (which is
Streamlit-only) so Cloud Run *jobs* and the future MCP server can trigger jobs
without importing Streamlit. ui_common.py re-exports this trio, so every
existing view keeps its exact behaviour.

The three functions already take their storage client / credentials / prefixes
as arguments — the only couplings severed from ui_common are the module BUCKET
constant (now homed here, the Streamlit-free side) and the st.secrets-backed
credential builder (which stays in ui_common and is passed in). The injected-
client shape mirrors src/modules/intake/profile_trigger.py, whose optional
jobs_client lets a job trigger another job over ambient ADC.
"""

import json

from google.cloud import run_v2, storage

BUCKET = 'cc-matcher-bucket-jeg-v1'

_JOB_PARENT = 'projects/cc-matcher-v1/locations/us-central1/jobs/'


def job_name(job: str) -> str:
    """'sam-gov-job' -> the fully-qualified Cloud Run Jobs resource name.

    An already-qualified name is returned unchanged, so callers that still
    hold a full 'projects/.../jobs/x' constant can pass it straight through.
    """
    return job if job.startswith('projects/') else f'{_JOB_PARENT}{job}'


def write_job_config(
    client: storage.Client, cfg_prefix: str, config: dict, *, bucket: str = BUCKET
) -> str:
    """Stage a job config blob and return its path (the job's only argv)."""
    blob_path = f"{cfg_prefix}{config['run_id']}.json"
    client.bucket(bucket).blob(blob_path).upload_from_string(
        json.dumps(config), content_type='application/json'
    )
    return blob_path


def trigger_job(
    credentials, job: str, config_blob_path: str,
    *, jobs_client: 'run_v2.JobsClient | None' = None,
) -> None:
    """Execute a Cloud Run Job with the config blob path as its argument.

    `job` is the short name ('drive-sync-job'). This uses runWithOverrides,
    which requires roles/run.admin on the job — plain roles/run.invoker is not
    enough. Pass `credentials` (the Streamlit path) or an already-built
    `jobs_client` (a job triggering another job can pass
    run_v2.JobsClient() for ambient ADC).
    """
    (jobs_client or run_v2.JobsClient(credentials=credentials)).run_job(
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
    client: storage.Client, status_prefix: str, run_id: str, *, bucket: str = BUCKET
) -> dict | None:
    """The job's status.json, or None if it has not been written yet."""
    blob = client.bucket(bucket).blob(f'{status_prefix}{run_id}/status.json')
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())
