"""
Drive onboarding — proposal schema, validation, and store (Phase 3.5)
---------------------------------------------------------------------
Streamlit-free. Owns the `submit_proposal` input schema (which the agent sees as
a terminal tool), the orchestrator-side validation that runs before anything is
saved, and the `drive-onboard/` store (proposals, per-folder state, agent config).
Shared by jobs/onboard_job.py, the review UI, and the shadow eval.

Nothing the agent returns is trusted: validation re-checks every claim against
what the tools actually surfaced this session, so a hallucinated file id, an
invented website, or a company_key that isn't in the named pool is rejected.
"""

import json
from datetime import datetime, timezone

BUCKET = 'cc-matcher-bucket-jeg-v1'
PROPOSALS_PREFIX = 'drive-onboard/proposals/'
STATE_BLOB = 'drive-onboard/state.json'
AGENT_BLOB = 'drive-onboard/agent.json'
SHADOW_PREFIX = 'drive-onboard/shadow/'

DECISIONS = ('existing_client', 'existing_prospect', 'new_client',
             'not_a_client', 'unsure')
CONFIDENCES = ('high', 'medium', 'low')
DOC_CATEGORIES = ('technical', 'proposal', 'pitch_deck', 'award', 'financial',
                  'legal_admin', 'correspondence', 'other')
OUTCOMES = ('proposed', 'budget_exhausted', 'refused', 'error')

# The input_schema the agent's submit_proposal tool advertises (3.5).
PROPOSAL_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'decision': {'type': 'string', 'enum': list(DECISIONS)},
        'company_key': {'type': 'string'},
        'new_company': {
            'type': 'object', 'additionalProperties': False,
            'properties': {
                'company_name': {'type': 'string'},
                'website': {'type': 'string'},
                'website_source': {'type': 'string',
                                   'enum': ['domain_signal', 'document', 'none']},
            },
        },
        'confidence': {'type': 'string', 'enum': list(CONFIDENCES)},
        'evidence': {
            'type': 'array',
            'items': {'type': 'object', 'additionalProperties': False,
                      'properties': {'claim': {'type': 'string'},
                                     'source': {'type': 'string'}},
                      'required': ['claim', 'source']},
        },
        'conflicts': {'type': 'array', 'items': {'type': 'string'}},
        'documents': {
            'type': 'array',
            'items': {'type': 'object', 'additionalProperties': False,
                      'properties': {
                          'file_id': {'type': 'string'},
                          'name': {'type': 'string'},
                          'category': {'type': 'string', 'enum': list(DOC_CATEGORIES)},
                          'include': {'type': 'boolean'},
                          'reason': {'type': 'string'}},
                      'required': ['file_id', 'include']},
        },
    },
    'required': ['decision', 'confidence', 'evidence'],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


# ── Validation (orchestrator-side, before save) ──────────────────────────────

def _evidence_file_ids(proposal: dict) -> list[str]:
    ids = []
    for e in proposal.get('evidence') or []:
        src = str(e.get('source', ''))
        if src.startswith('file:'):
            ids.append(src.split(':', 1)[1])
    return ids


def validate_proposal(proposal: dict, *, surfaced_file_ids: set,
                      surfaced_domains: set, folder_file_ids: set,
                      known_keys_by_pool: dict) -> list[str]:
    """Return a list of validation errors ([] = valid). Checks every claim
    against what the tools actually surfaced this session."""
    errs: list[str] = []
    decision = proposal.get('decision')
    if decision not in DECISIONS:
        errs.append(f'decision {decision!r} not in {DECISIONS}')

    # existing_* must name a real company_key in the matching pool
    if decision in ('existing_client', 'existing_prospect'):
        key = proposal.get('company_key') or ''
        pool = 'clients' if decision == 'existing_client' else 'prospects'
        if key not in known_keys_by_pool.get(pool, set()):
            errs.append(f'company_key {key!r} not found in {pool} pool')

    # evidence file sources must have been returned by a tool this session
    for fid in _evidence_file_ids(proposal):
        if fid not in surfaced_file_ids:
            errs.append(f'evidence cites file {fid!r} never returned by a tool')

    # a new-client website must be a domain the tools surfaced (never invented)
    if decision == 'new_client':
        nc = proposal.get('new_company') or {}
        site = str(nc.get('website', '')).strip()
        if site:
            from src.modules.tools.drive_read import bare_domain
            if bare_domain(site) not in surfaced_domains:
                errs.append(f'new_company.website {site!r} not among surfaced domains')

    # every proposed document must be in the folder listing
    for d in proposal.get('documents') or []:
        fid = d.get('file_id')
        if fid not in folder_file_ids:
            errs.append(f'documents cite file {fid!r} not in the folder listing')
    return errs


# ── Store ────────────────────────────────────────────────────────────────────

def proposal_blob(folder_id: str) -> str:
    return f'{PROPOSALS_PREFIX}{folder_id}.json'


def save_proposal(gcs_client, proposal: dict, *, bucket: str = BUCKET) -> None:
    gcs_client.bucket(bucket).blob(proposal_blob(proposal['folder_id'])) \
        .upload_from_string(json.dumps(proposal, default=str),
                            content_type='application/json')


def load_proposal(gcs_client, folder_id: str, *, bucket: str = BUCKET) -> dict | None:
    blob = gcs_client.bucket(bucket).blob(proposal_blob(folder_id))
    if not blob.exists():
        return None
    return json.loads(blob.download_as_bytes())


def list_proposals(gcs_client, *, bucket: str = BUCKET, status: str | None = None
                   ) -> list[dict]:
    out = []
    for blob in gcs_client.list_blobs(bucket, prefix=PROPOSALS_PREFIX):
        if not blob.name.endswith('.json'):
            continue
        try:
            p = json.loads(blob.download_as_bytes())
        except Exception:
            continue
        if status is None or p.get('review_status', 'open') == status:
            out.append(p)
    return out


def load_state(gcs_client, *, bucket: str = BUCKET) -> dict:
    blob = gcs_client.bucket(bucket).blob(STATE_BLOB)
    if not blob.exists():
        return {}
    try:
        return json.loads(blob.download_as_bytes())
    except Exception:
        return {}


def save_state(gcs_client, state: dict, *, bucket: str = BUCKET) -> None:
    gcs_client.bucket(bucket).blob(STATE_BLOB).upload_from_string(
        json.dumps(state, default=str), content_type='application/json')


def load_agent_config(gcs_client, *, bucket: str = BUCKET) -> dict:
    blob = gcs_client.bucket(bucket).blob(AGENT_BLOB)
    if not blob.exists():
        return {}
    try:
        return json.loads(blob.download_as_bytes())
    except Exception:
        return {}


def save_agent_config(gcs_client, config: dict, *, bucket: str = BUCKET) -> None:
    gcs_client.bucket(bucket).blob(AGENT_BLOB).upload_from_string(
        json.dumps(config, default=str), content_type='application/json')
