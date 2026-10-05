"""
profile_trigger.py — start a client-profile-job build for one intake company.

The Streamlit-free equivalent of ui_common.write_job_config + trigger_job.
runWithOverrides needs roles/run.admin on client-profile-job for the
intake-web@ service account (run.invoker is not enough).
"""

import json

from google.cloud import run_v2

import src.modules.aspect_profile as ap

PROJECT    = 'cc-matcher-v1'
REGION     = 'us-central1'
JOB        = 'client-profile-job'
CFG_PREFIX = 'client-profile-configs/'

# Every row-borne source except financials (money, not capability) — the job
# drops sources a company has no material for.
SOURCES = [k for k in ap.SOURCE_KEYS if k != 'financials']


def build_config(session_id: str, pool: str, company_key: str) -> dict:
    return {
        'run_id':            f'client_profile_intake_{session_id[:12]}',
        'pool':              pool,
        'company_keys':      [company_key],
        'sources':           SOURCES,
        'target_aspects':    4,
        'max_markets':       ap.MAX_MARKETS,
        'assess_defense':    True,
        'assess_unexplored': True,
        'max_unexplored':    ap.MAX_UNEXPLORED,
        'model':             ap.DEFAULT_MODEL,
        'concurrency':       1,
        'dry_run':           False,
    }


def trigger(storage_client, session_id: str, pool: str, company_key: str,
            jobs_client=None) -> str:
    """Stage the config and execute the job. Returns the run_id."""
    config    = build_config(session_id, pool, company_key)
    blob_path = f"{CFG_PREFIX}{config['run_id']}.json"
    storage_client.bucket(ap.BUCKET).blob(blob_path).upload_from_string(
        json.dumps(config), content_type='application/json')
    (jobs_client or run_v2.JobsClient()).run_job(
        request=run_v2.RunJobRequest(
            name=f'projects/{PROJECT}/locations/{REGION}/jobs/{JOB}',
            overrides=run_v2.RunJobRequest.Overrides(
                container_overrides=[
                    run_v2.RunJobRequest.Overrides.ContainerOverride(args=[blob_path])
                ]
            ),
        )
    )
    return config['run_id']
