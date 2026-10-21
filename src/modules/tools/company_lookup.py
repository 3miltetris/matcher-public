"""
Company lookup tools for the onboarding agent (Phase 3.4)
---------------------------------------------------------
search_companies / get_company. Read-only, JSON-serialisable, Streamlit-free.
Reuse pools (both stores) + folder_match for name scoring. `has_drive_assignment`
comes from drive-sync-configs/assignments.json. The shadow eval (3.9) hides the
assignment of the folder's true client via `hide_assignment_for`, so the agent
can't cheat off ground truth.
"""

import io
import json
from difflib import SequenceMatcher

import pandas as pd

from src.modules import folder_match
from src.modules import pools as pl

_ASSIGN_BLOB = 'drive-sync-configs/assignments.json'
_SUMMARY_CAP = 2000


def _load_assignments(gcs_client, bucket: str) -> dict:
    try:
        blob = gcs_client.bucket(bucket).blob(_ASSIGN_BLOB)
        return json.loads(blob.download_as_bytes()).get('assignments', {})
    except Exception:
        return {}


def _companies(gcs_client, pools: list[str], bucket: str) -> dict[str, dict]:
    """{company_key: {name, website, pool, summary}} across the named pools.
    First non-empty value wins per field (a partially-updated file leaves blanks)."""
    frames_by_pool, _ = pl.load_all_frames(gcs_client, keys=pools, bucket=bucket)
    out: dict[str, dict] = {}
    for pool_key, frames in frames_by_pool.items():
        for df in frames.values():
            if 'company_name' not in df.columns or 'companyWebsite' not in df.columns:
                continue
            for _, row in df.iterrows():
                key = pl.company_key(row)
                if key == '||':
                    continue
                rec = out.setdefault(key, {
                    'company_key': key, 'pool': pool_key,
                    'name': '', 'website': '', 'summary': ''})
                rec['name'] = rec['name'] or str(row.get('company_name') or '').strip()
                rec['website'] = rec['website'] or str(row.get('companyWebsite') or '').strip()
                if not rec['summary']:
                    rec['summary'] = str(row.get('summary') or '').strip()
    return out


def search_companies(gcs_client, query: str, *, pools=('clients', 'prospects'),
                     limit: int = 10, bucket: str = pl.BUCKET,
                     hide_assignment_for=None) -> dict:
    """Top companies whose name best matches `query` (normalised ratio, with a
    containment boost). Each carries pool, website, a 0-1 score, and whether it
    already has a Drive folder assigned."""
    pools = list(pools)
    hide = set(hide_assignment_for or ())
    comps = _companies(gcs_client, pools, bucket)
    assigned_keys = {a.get('client_key') for a in
                     _load_assignments(gcs_client, bucket).values() if a.get('client_key')}

    qn = folder_match.normalize(query)
    scored = []
    for rec in comps.values():
        nn = folder_match.normalize(rec['name'])
        if not nn:
            continue
        ratio = SequenceMatcher(None, qn, nn).ratio() if qn else 0.0
        if qn and (qn in nn or nn in qn):
            ratio = max(ratio, 0.95)
        scored.append((ratio, rec))
    scored.sort(key=lambda t: -t[0])

    results = []
    for ratio, rec in scored[:limit]:
        key = rec['company_key']
        results.append({
            'company_key': key, 'name': rec['name'], 'website': rec['website'],
            'pool': rec['pool'], 'score': round(ratio, 3),
            'has_drive_assignment': (key in assigned_keys) and (key not in hide),
        })
    return {'query': query, 'results': results}


def get_company(gcs_client, company_key: str, *, pools=('clients', 'prospects'),
                bucket: str = pl.BUCKET) -> dict:
    """One company's summary (truncated), website, pool, Drive folder
    assignments, and intake_data if present. {} with `found: false` if unknown."""
    comps = _companies(gcs_client, list(pools), bucket)
    rec = comps.get(company_key)
    if not rec:
        return {'company_key': company_key, 'found': False}

    assignments = _load_assignments(gcs_client, bucket)
    folders = [{'folder_id': fid, 'folder_name': a.get('folder_name', ''),
                'match_type': a.get('match_type', '')}
               for fid, a in assignments.items() if a.get('client_key') == company_key]

    intake = ''
    frames_by_pool, _ = pl.load_all_frames(gcs_client, keys=list(pools), bucket=bucket)
    for frames in frames_by_pool.values():
        for df in frames.values():
            if 'intake_data' not in df.columns:
                continue
            mask = pl.key_mask(df, company_key)
            if mask.any():
                intake = str(df.loc[mask, 'intake_data'].iloc[0] or '')
                break
        if intake:
            break

    summary = rec['summary'][:_SUMMARY_CAP]
    return {
        'company_key': company_key, 'found': True, 'name': rec['name'],
        'website': rec['website'], 'pool': rec['pool'], 'summary': summary,
        'folder_assignments': folders,
        'has_intake': bool(intake), 'intake_data': intake[:_SUMMARY_CAP],
    }
