"""
Approve an onboarding proposal (Phase 3.8)
------------------------------------------
Streamlit-free preview/apply, called by the Drive Sync review UI and (later) the
MCP server's Tier 3. Reuses the existing paths — it writes no new storage logic
of its own beyond the assignment entry:

- existing_client  -> assignment (match_type 'agent_confirmed') + excluded_file_ids
- existing_prospect -> pool_transfer promote (prospects -> clients), then assign
- new_client       -> a minimal client row (name/website; profile built by the
                      next sync) + assign
- not_a_client     -> assignment moved to `skipped`
- unsure           -> no-op (stays open; the user assigns manually)

Triggering the per-client drive-sync and resolving the attention item are left to
the caller (the view already owns a job trigger); apply returns what it changed.
"""

import io
import json
import secrets as _secrets
from datetime import date, datetime, timezone

import pandas as pd

from src.modules import attention
from src.modules import onboarding as ob
from src.modules import pool_transfer

_BUCKET = 'cc-matcher-bucket-jeg-v1'
_ASSIGN_BLOB = 'drive-sync-configs/assignments.json'
_CLIENTS_PREFIX = 'data/all-contacts/clients/'


def _load_assignments(gcs, bucket):
    blob = gcs.bucket(bucket).blob(_ASSIGN_BLOB)
    if not blob.exists():
        return {'assignments': {}, 'unassigned': {}, 'skipped': {}}
    doc = json.loads(blob.download_as_bytes())
    doc.setdefault('assignments', {})
    doc.setdefault('unassigned', {})
    doc.setdefault('skipped', {})
    return doc


def _save_assignments(gcs, doc, bucket):
    doc['updated_at'] = datetime.now(timezone.utc).isoformat()
    gcs.bucket(bucket).blob(_ASSIGN_BLOB).upload_from_string(
        json.dumps(doc), content_type='application/json')


def _effective(proposal: dict, overrides: dict) -> dict:
    """Merge reviewer overrides onto the proposal (decision / company_key /
    new_company / excluded_file_ids all overridable in the UI)."""
    o = overrides or {}
    decision = o.get('decision') or proposal.get('decision')
    company_key = o.get('company_key') or proposal.get('company_key') or ''
    new_company = o.get('new_company') or proposal.get('new_company') or {}
    if 'excluded_file_ids' in o:
        excluded = list(o['excluded_file_ids'])
    else:
        excluded = [d.get('file_id') for d in (proposal.get('documents') or [])
                    if d.get('file_id') and not d.get('include', True)]
    return {'decision': decision, 'company_key': company_key,
            'new_company': new_company, 'excluded_file_ids': excluded}


def preview(gcs, folder_id: str, overrides: dict | None = None, *,
            bucket: str = _BUCKET) -> dict:
    """The exact change apply() would make — no side effects."""
    proposal = ob.load_proposal(gcs, folder_id, bucket=bucket)
    if not proposal:
        return {'error': f'no proposal for folder {folder_id}'}
    eff = _effective(proposal, overrides or {})
    d = eff['decision']
    plan = {'folder_id': folder_id, 'folder_name': proposal.get('folder_name', ''),
            'decision': d, 'excluded_file_ids': eff['excluded_file_ids']}
    if d == 'existing_client':
        plan['action'] = f'assign folder to client {eff["company_key"]}'
        plan['client_key'] = eff['company_key']
    elif d == 'existing_prospect':
        plan['action'] = (f'promote prospect {eff["company_key"]} to clients, '
                          f'then assign folder')
        plan['client_key'] = eff['company_key']
    elif d == 'new_client':
        nc = eff['new_company']
        plan['action'] = (f'create client {nc.get("company_name","")!r} '
                          f'({nc.get("website","")}) and assign folder')
        plan['client_key'] = f'{nc.get("company_name","")}||{nc.get("website","")}'
    elif d == 'not_a_client':
        plan['action'] = 'mark folder skipped'
    else:
        plan['action'] = 'no change (unsure — assign manually)'
    return plan


def apply(gcs, folder_id: str, overrides: dict | None = None, *, actor: str = '',
          bucket: str = _BUCKET) -> dict:
    """Apply the approved proposal. Returns {ok, action, client_key?, errors}.
    Resolving the attention item and triggering the per-client sync are the
    caller's job (it gets client_key back)."""
    proposal = ob.load_proposal(gcs, folder_id, bucket=bucket)
    if not proposal:
        return {'ok': False, 'errors': [f'no proposal for folder {folder_id}']}
    eff = _effective(proposal, overrides or {})
    d = eff['decision']
    doc = _load_assignments(gcs, bucket)
    result = {'ok': True, 'folder_id': folder_id, 'decision': d, 'errors': []}

    def _assign(client_key: str):
        doc['assignments'][folder_id] = {
            'folder_name': proposal.get('folder_name', ''),
            'client_key': client_key, 'match_type': 'agent_confirmed',
            'assigned_at': date.today().isoformat(),
            'excluded_file_ids': eff['excluded_file_ids'],
            'approved_by': actor,
        }
        doc['unassigned'].pop(folder_id, None)
        doc['skipped'].pop(folder_id, None)

    if d == 'existing_client':
        _assign(eff['company_key'])
        result['client_key'] = eff['company_key']
    elif d == 'existing_prospect':
        rep = pool_transfer.move_companies(gcs, [eff['company_key']],
                                           source_pool='prospects', dest_pool='clients')
        if rep.get('rows_moved') or eff['company_key'] in (rep.get('conflicts') or []):
            _assign(eff['company_key'])        # already a client on a conflict
            result['client_key'] = eff['company_key']
            result['promote_report'] = rep
        else:
            result['ok'] = False
            result['errors'].append(f'promotion failed: {rep}')
    elif d == 'new_client':
        nc = eff['new_company']
        name, site = nc.get('company_name', '').strip(), nc.get('website', '').strip()
        if not name or not site:
            return {'ok': False, 'errors': ['new_client needs company_name and website']}
        _write_new_client(gcs, name, site, bucket)
        _assign(f'{name}||{site}')
        result['client_key'] = f'{name}||{site}'
    elif d == 'not_a_client':
        doc['skipped'][folder_id] = {
            'folder_name': proposal.get('folder_name', ''),
            'skipped_by': actor, 'skipped_at': date.today().isoformat()}
        doc['assignments'].pop(folder_id, None)
        doc['unassigned'].pop(folder_id, None)
        result['action'] = 'skipped'
    elif d == 'unsure':
        return {'ok': True, 'folder_id': folder_id, 'decision': d,
                'action': 'no change', 'errors': []}
    else:
        return {'ok': False, 'errors': [f'unknown decision {d!r}']}

    if result['ok']:
        _save_assignments(gcs, doc, bucket)
        proposal['review_status'] = 'applied'
        proposal['applied_by'] = actor
        proposal['applied_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        ob.save_proposal(gcs, proposal, bucket=bucket)
        try:
            df = attention.load(gcs, kind='drive_onboarding_proposal', status='open')
            match = df[df['subject_key'] == folder_id] if not df.empty else df
            for iid in match['id'].tolist():
                attention.resolve(gcs, iid, resolved_by=actor)
        except Exception as e:
            result['errors'].append(f'attention resolve: {e}')
    return result


def _write_new_client(gcs, name: str, website: str, bucket: str) -> str:
    """One minimal client row; the next sync builds the summary + embedding."""
    row = {'company_name': name, 'companyWebsite': website, 'summary': '',
           'embeddings': [], 'scraped_at': date.today().isoformat(),
           'uuid': _secrets.token_hex(8)}
    buf = io.BytesIO()
    pd.DataFrame([row]).to_parquet(buf, index=False)
    blob = f'{_CLIENTS_PREFIX}onboard_{date.today().isoformat()}_{_secrets.token_hex(3)}.parquet'
    gcs.bucket(bucket).blob(blob).upload_from_string(
        buf.getvalue(), content_type='application/octet-stream')
    return blob
