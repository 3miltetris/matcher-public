"""
Onboarding-proposal tools (Phase 5.1, Tier 1 read-only)
-------------------------------------------------------
list_onboarding_proposals / get_onboarding_proposal wrap the Phase 3 proposal
store (src/modules/onboarding.py). Read-only: reviewing a proposal here, applying
one (assignment / promotion / new client) is a Tier-3 staged write through
onboarding_apply.py behind a confirm token — not this module.
"""

from __future__ import annotations

from src.modules import onboarding as ob
from src.modules.tools import _common as cm


def _summary(p: dict) -> dict:
    """The compact card for a proposal list."""
    return {
        'folder_id': p.get('folder_id', ''),
        'folder_name': p.get('folder_name', ''),
        'decision': p.get('decision', ''),
        'confidence': p.get('confidence', ''),
        'company_key': p.get('company_key', ''),
        'review_status': p.get('review_status', 'open'),
        'outcome': p.get('outcome', ''),
        'n_documents': len(p.get('documents') or []),
        'n_evidence': len(p.get('evidence') or []),
    }


def list_onboarding_proposals(gcs_client, *, status: str | None = 'open',
                              limit: int = 50, bucket: str = ob.BUCKET) -> dict:
    """The onboarding agent's proposals, compact. `status` filters on
    review_status ('open' by default; None = all)."""
    flt = None if status in (None, 'all') else status
    props = ob.list_proposals(gcs_client, status=flt, bucket=bucket)
    props.sort(key=lambda p: p.get('folder_name', '').lower())
    return {'total': len(props), 'returned': min(len(props), max(1, int(limit))),
            'proposals': [_summary(p) for p in props[:max(1, int(limit))]]}


def get_onboarding_proposal(gcs_client, folder_id: str, *,
                            bucket: str = ob.BUCKET) -> dict:
    """One proposal in full: decision, confidence, the evidence list (claim +
    source), conflicts, and the per-document include/exclude table — every
    free-text field capped for the tool-output budget. `{found: false}` if the
    folder has no proposal."""
    p = ob.load_proposal(gcs_client, folder_id, bucket=bucket)
    if p is None:
        return {'folder_id': folder_id, 'found': False}

    evidence = [{'claim': cm.cap_text(e.get('claim'), 400),
                 'source': e.get('source', '')}
                for e in (p.get('evidence') or [])]
    documents = [{'file_id': d.get('file_id', ''),
                  'name': cm.cap_text(d.get('name'), 200),
                  'category': d.get('category', ''),
                  'include': bool(d.get('include', False)),
                  'reason': cm.cap_text(d.get('reason'), 300)}
                 for d in (p.get('documents') or [])]
    return {
        'found': True,
        'folder_id': p.get('folder_id', ''),
        'folder_name': p.get('folder_name', ''),
        'run_id': p.get('run_id', ''),
        'prompt_version': p.get('prompt_version', ''),
        'decision': p.get('decision', ''),
        'company_key': p.get('company_key', ''),
        'new_company': p.get('new_company') or {},
        'confidence': p.get('confidence', ''),
        'review_status': p.get('review_status', 'open'),
        'outcome': p.get('outcome', ''),
        'conflicts': [cm.cap_text(c, 300) for c in (p.get('conflicts') or [])],
        'evidence': evidence,
        'documents': documents,
    }
