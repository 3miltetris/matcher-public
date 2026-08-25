"""
Drive Sync
----------
Scan a Google shared drive of client data-collection folders, auto-assign
folders to clients in data/all-contacts/clients/ (fuzzy name match, review
table for the rest), and trigger the drive-sync-job Cloud Run Job which
extracts changed documents, merges the new info into each client's matching
summary with Claude, re-embeds, and stores a docs digest. Folders with no
matching client come back as proposals reviewed and approved here.

Folder assignments persist in drive-sync-configs/assignments.json, so
after the first scan a full update is one click.

The shared drive must have both service accounts added as Viewer members:
matcher-app@cc-matcher-v1.iam.gserviceaccount.com (this view) and
matching-job@cc-matcher-v1.iam.gserviceaccount.com (the job).
"""

import io
import json
import re
import secrets as _secrets
import string
import time
import traceback
import uuid
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import run_v2, storage
from google.oauth2 import service_account

from src.modules import drive_client
from src.modules.Embedding.text_embedder import TextProcessor
from src.modules.GoogleBucketManager.bucket_manager import BucketManager

# ── Constants ──────────────────────────────────────────────────────────────

_BUCKET          = 'cc-matcher-bucket-jeg-v1'
_CLIENTS_PREFIX  = 'data/all-contacts/clients/'
_CFG_PREFIX      = 'drive-sync-configs/'
_STATUS_PREFIX   = 'drive-sync-jobs/'
_ASSIGN_BLOB     = 'drive-sync-configs/assignments.json'
_SYNC_STATE_BLOB = 'drive-sync-configs/sync_state.json'
_JOB_NAME        = 'projects/cc-matcher-v1/locations/us-central1/jobs/drive-sync-job'
_POLL_INTERVAL   = 10
_SYNC_MODEL      = 'claude-sonnet-4-6'

# Cloud Run task timeout options (the job is deployed with the 24 h maximum and
# stops itself ~10 min before whichever budget is chosen here).
_TIME_BUDGETS = {
    '1 hour':   3_600,
    '2 hours':  7_200,
    '4 hours':  14_400,
    '8 hours':  28_800,
    '12 hours': 43_200,
    '24 hours': 86_400,
}
_DEFAULT_BUDGET_LABEL = '4 hours'
_STALE_DAYS = 30

_DEFAULT_EXCLUDED_SECTIONS = ['internal projects']

_NEW_CLIENT = '— new client —'
_SKIP       = '— skip —'

_LEGAL_SUFFIXES = {'inc', 'llc', 'corp', 'co', 'ltd', 'pllc', 'incorporated',
                   'corporation', 'company'}

_FUZZY_THRESHOLD = 0.87
_FUZZY_MARGIN    = 0.05


# ── GCS / credentials ──────────────────────────────────────────────────────

def _get_credentials():
    return service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )


def _get_storage_client() -> storage.Client:
    return storage.Client(credentials=_get_credentials())


def _get_drive_service():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account'], scopes=drive_client.DRIVE_SCOPES
    )
    return drive_client.build_drive_service(creds)


def _load_client_frames() -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Returns ({blob_name: df}, errors) — frames kept per-blob so edits can
    be written back to the exact file they came from."""
    client = _get_storage_client()
    frames: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for blob in client.list_blobs(_BUCKET, prefix=_CLIENTS_PREFIX):
        if not blob.name.endswith('.parquet'):
            continue
        try:
            frames[blob.name] = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        except Exception as e:
            errors.append(f'{blob.name}: {e}')
    return frames, errors


def _company_key(row: pd.Series) -> str:
    name    = str(row.get('company_name') or '').strip()
    website = str(row.get('companyWebsite') or '').strip()
    return f'{name}||{website}'


def _load_assignments(client: storage.Client) -> dict:
    blob = client.bucket(_BUCKET).blob(_ASSIGN_BLOB)
    if not blob.exists():
        return {
            'drive_id':          '',
            'excluded_sections': list(_DEFAULT_EXCLUDED_SECTIONS),
            'updated_at':        '',
            'assignments':       {},
            'unassigned':        {},
            'skipped':           {},
        }
    doc = json.loads(blob.download_as_text())
    doc.setdefault('assignments', {})
    doc.setdefault('unassigned', {})
    doc.setdefault('skipped', {})
    doc.setdefault('excluded_sections', list(_DEFAULT_EXCLUDED_SECTIONS))
    return doc


def _save_assignments(client: storage.Client, doc: dict) -> None:
    doc['updated_at'] = datetime.now(timezone.utc).isoformat()
    client.bucket(_BUCKET).blob(_ASSIGN_BLOB).upload_from_string(
        json.dumps(doc), content_type='application/json'
    )


def _load_sync_state(client: storage.Client) -> dict:
    """File-history state written by the job: {'files': {file_id: {...}},
    'proposed': {folder_id: last-proposed date}}. Read-only here — used to show
    what has actually been synced so the selection can be made deliberately."""
    blob = client.bucket(_BUCKET).blob(_SYNC_STATE_BLOB)
    if not blob.exists():
        return {'files': {}, 'proposed': {}}
    try:
        doc = json.loads(blob.download_as_text())
    except Exception:
        return {'files': {}, 'proposed': {}}
    doc.setdefault('files', {})
    doc.setdefault('proposed', {})
    return doc


def _folder_sync_index(sync_state: dict) -> tuple[dict[str, str], dict[str, int]]:
    """(folder_id → latest synced_at, folder_id → files tracked)."""
    last: dict[str, str] = {}
    counts: dict[str, int] = {}
    for meta in (sync_state.get('files') or {}).values():
        fid = meta.get('folder_id')
        if not fid:
            continue
        counts[fid] = counts.get(fid, 0) + 1
        synced = str(meta.get('synced_at') or '')
        if synced > last.get(fid, ''):
            last[fid] = synced
    return last, counts


def _write_config(client: storage.Client, config: dict) -> str:
    blob_path = f"{_CFG_PREFIX}{config['run_id']}.json"
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


# ── Fuzzy matching ─────────────────────────────────────────────────────────

_INTERNAL_RE = re.compile(r'[\s_-]*internal\s*$', re.IGNORECASE)
_PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def _normalize(name: str) -> str:
    """Folder or company name → comparable form: strip trailing _INTERNAL,
    lowercase, drop punctuation and legal suffixes, collapse whitespace."""
    text  = _INTERNAL_RE.sub('', str(name or '')).lower()
    text  = text.translate(_PUNCT_TABLE)
    words = [w for w in text.split() if w not in _LEGAL_SUFFIXES]
    return ' '.join(words)


def _match_folder(norm_folder: str, client_norms: dict[str, str]
                  ) -> tuple[str | None, str, float]:
    """Match a normalized folder name against {client_key: normalized_name}.
    Returns (client_key | None, tier, score) — tier in exact|contains|fuzzy|none."""
    if not norm_folder:
        return None, 'none', 0.0
    # (a) exact
    for key, norm in client_norms.items():
        if norm and norm == norm_folder:
            return key, 'exact', 1.0
    # (b) containment either direction, shorter side >= 5 chars
    for key, norm in client_norms.items():
        if not norm:
            continue
        shorter = min(norm, norm_folder, key=len)
        if len(shorter) >= 5 and (norm in norm_folder or norm_folder in norm):
            return key, 'contains', 0.99
    # (c) best ratio >= threshold and clear of runner-up
    scored = sorted(
        ((SequenceMatcher(None, norm_folder, norm).ratio(), key)
         for key, norm in client_norms.items() if norm),
        reverse=True,
    )
    if scored and scored[0][0] >= _FUZZY_THRESHOLD:
        if len(scored) == 1 or scored[0][0] - scored[1][0] >= _FUZZY_MARGIN:
            return scored[0][1], 'fuzzy', scored[0][0]
    return None, 'none', scored[0][0] if scored else 0.0


# ── Session state ──────────────────────────────────────────────────────────

for _k in ('ds_frames', 'ds_sections', 'ds_active_run', 'ds_last_status'):
    if _k not in st.session_state:
        st.session_state[_k] = None


# ── Page ───────────────────────────────────────────────────────────────────

st.title('🗂️ Drive Sync')
st.caption(
    'Scan the client shared drive, assign folders to clients, and sync new '
    'document information into client profiles (summary + re-embed + docs digest).'
)

gcs = _get_storage_client()

# ── Active run polling ─────────────────────────────────────────────────────

if st.session_state.ds_active_run:
    run_id = st.session_state.ds_active_run
    st.subheader('🔄 Sync job in progress')
    st.caption(f'Run ID: `{run_id}`')
    try:
        status = _poll_status(gcs, run_id)
        if status is None or status.get('state') == 'running':
            done    = (status or {}).get('clients_done', 0)
            total   = (status or {}).get('clients_total', 0)
            p_done  = (status or {}).get('proposals_done', 0)
            p_total = (status or {}).get('proposals_total', 0)
            if p_total:
                st.progress(p_done / p_total,
                            text=f'New-client proposals: {p_done}/{p_total}')
            elif total:
                st.progress(done / total, text=f'{done}/{total} clients processed')
            else:
                st.info(f'Job is starting… checking again in {_POLL_INTERVAL}s.')
            if st.button('Cancel monitoring (job keeps running)'):
                st.session_state.ds_active_run = None
                st.rerun()
            time.sleep(_POLL_INTERVAL)
            st.rerun()
        elif status.get('error'):
            st.error('Sync job failed.')
            st.code(status['error'], language='text')
            st.session_state.ds_active_run = None
        else:
            st.session_state.ds_last_status = status
            st.session_state.ds_active_run  = None
            st.session_state.pop('ds_frames', None)   # profiles changed — reload
            st.rerun()
    except Exception as e:
        st.error(f'Error polling status: {e}')
        st.code(traceback.format_exc())
        if st.button('Stop monitoring'):
            st.session_state.ds_active_run = None
            st.rerun()
    st.stop()

# ── Resume monitoring ──────────────────────────────────────────────────────

with st.expander('Resume monitoring a previous sync job'):
    resume_id = st.text_input(
        'Run ID', key='ds_resume_run_id',
        placeholder='drive_sync_2026-08-11_15-30-00',
    )
    if st.button('Check status', key='ds_resume_btn') and resume_id.strip():
        st.session_state.ds_active_run = resume_id.strip()
        st.rerun()

# ── Load assignments + clients ─────────────────────────────────────────────

assign_doc = _load_assignments(gcs)

if st.session_state.get('ds_frames') is None:
    with st.spinner('Loading clients from GCS…'):
        frames, load_errors = _load_client_frames()
    st.session_state.ds_frames = frames
    for err in load_errors:
        st.warning(err)

frames: dict[str, pd.DataFrame] = st.session_state.ds_frames

client_labels: dict[str, str] = {}   # client_key → display label
client_norms:  dict[str, str] = {}   # client_key → normalized name
if frames:
    combined = pd.concat(frames.values(), ignore_index=True)
    combined['_key'] = combined.apply(_company_key, axis=1)
    for key, grp in combined.groupby('_key', sort=False):
        name    = str(grp.iloc[0].get('company_name') or '').strip()
        website = str(grp.iloc[0].get('companyWebsite') or '').strip()
        client_labels[key] = f"{name or '—'}  ·  {website or 'no website'}"
        client_norms[key]  = _normalize(name)

# ── 1 · Setup ──────────────────────────────────────────────────────────────

with st.expander('⚙️ Setup — shared drive', expanded=not assign_doc.get('drive_id')):
    st.caption(
        'The shared drive must have **matcher-app@cc-matcher-v1.iam.gserviceaccount.com** '
        'and **matching-job@cc-matcher-v1.iam.gserviceaccount.com** added as Viewer '
        'members. The drive ID is the part of the drive URL after `/folders/` or '
        '`/drive/u/0/folders/` when the shared drive root is open.'
    )
    drive_id_input = st.text_input(
        'Shared drive ID', value=assign_doc.get('drive_id', ''), key='ds_drive_id',
    )
    if st.button('💾 Save drive ID', key='ds_save_drive') and drive_id_input.strip():
        assign_doc['drive_id'] = drive_id_input.strip()
        _save_assignments(gcs, assign_doc)
        st.success('Saved.')
        st.rerun()

drive_id = assign_doc.get('drive_id', '')
if not drive_id:
    st.info('Enter and save the shared drive ID to continue.')
    st.stop()

if not frames:
    st.warning(f'No client parquet files found under {_CLIENTS_PREFIX} in GCS.')

# ── 2 · Scan sections & auto-assign ────────────────────────────────────────

st.divider()
st.subheader('1 · Scan & auto-assign folders')

col_list, col_sel = st.columns([1, 3])
with col_list:
    if st.button('📁 List sections', help='List top-level section folders in the shared drive'):
        try:
            with st.spinner('Listing sections…'):
                svc = _get_drive_service()
                st.session_state.ds_sections = drive_client.list_child_folders(
                    svc, drive_id, drive_id)
        except Exception as e:
            st.error(f'Failed to list sections: {e}')
            st.code(traceback.format_exc())

sections = st.session_state.ds_sections or []
excluded = [s.lower() for s in assign_doc.get('excluded_sections', [])]

if sections:
    with col_sel:
        default_sections = [s['name'] for s in sections
                            if s['name'].lower().strip() not in excluded]
        picked = st.multiselect(
            'Sections to scan', options=[s['name'] for s in sections],
            default=default_sections, key='ds_sections_pick',
            help='"internal projects" is excluded by default — those are not clients.',
        )

    if st.button('🔍 Scan sections & auto-assign', key='ds_scan_btn') and picked:
        section_by_name = {s['name']: s['id'] for s in sections}
        n_new_auto = n_new_unassigned = n_seen = 0
        try:
            svc = _get_drive_service()
            prog = st.progress(0.0)
            newly_matched: dict[str, list[tuple[str, str, float, dict]]] = {}
            for i, sec_name in enumerate(picked):
                prog.progress((i + 1) / len(picked), text=f'Scanning {sec_name}…')
                folders = drive_client.list_child_folders(
                    svc, drive_id, section_by_name[sec_name])
                for folder in folders:
                    fid = folder['id']
                    n_seen += 1
                    if (fid in assign_doc['assignments']
                            or fid in assign_doc['skipped']):
                        continue
                    key, tier, score = _match_folder(
                        _normalize(folder['name']), client_norms)
                    if key:
                        newly_matched.setdefault(key, []).append(
                            (fid, tier, score,
                             {'folder_name': folder['name'], 'section': sec_name}))
                    elif fid not in assign_doc['unassigned']:
                        assign_doc['unassigned'][fid] = {
                            'folder_name': folder['name'],
                            'section':     sec_name,
                            'first_seen':  date.today().isoformat(),
                            'note':        f'no match >= {_FUZZY_THRESHOLD} (best {score:.2f})',
                        }
                        n_new_unassigned += 1
            prog.empty()

            # Collision rule: multiple new folders matching one client are all
            # kept only when every match is exact; otherwise best score wins
            # and the rest go to review. A non-exact match to a client that
            # already has a folder assigned goes to review too.
            for key, matches in newly_matched.items():
                already_assigned = any(
                    a.get('client_key') == key
                    for a in assign_doc['assignments'].values())
                all_exact = all(t == 'exact' for _, t, _, _ in matches)
                if all_exact:
                    keep = matches
                elif already_assigned:
                    keep = []
                else:
                    keep = [max(matches, key=lambda m: m[2])]
                kept_ids = {m[0] for m in keep}
                for fid, tier, score, meta in matches:
                    if fid in kept_ids:
                        assign_doc['assignments'][fid] = {
                            **meta,
                            'client_key':  key,
                            'match_type':  'auto',
                            'assigned_at': date.today().isoformat(),
                        }
                        assign_doc['unassigned'].pop(fid, None)
                        n_new_auto += 1
                    else:
                        assign_doc['unassigned'][fid] = {
                            **meta,
                            'first_seen': date.today().isoformat(),
                            'note': f'{tier} match ({score:.2f}) to already-matched '
                                    f'client — review',
                        }
                        n_new_unassigned += 1

            _save_assignments(gcs, assign_doc)
            st.success(
                f'Scanned {n_seen} folders — {n_new_auto} newly auto-assigned, '
                f'{n_new_unassigned} sent to review. Existing assignments untouched.'
            )
            st.rerun()
        except Exception as e:
            st.error(f'Scan failed: {e}')
            st.code(traceback.format_exc())

# ── 3 · Assignment review ──────────────────────────────────────────────────

n_assigned   = len(assign_doc['assignments'])
n_unassigned = len(assign_doc['unassigned'])
n_skipped    = len(assign_doc['skipped'])

st.caption(
    f'**{n_assigned}** folders assigned · **{n_unassigned}** need review · '
    f'**{n_skipped}** skipped'
)

if n_assigned or n_unassigned or n_skipped:
    with st.expander('📋 Review folder assignments', expanded=bool(n_unassigned)):
        label_by_key = dict(client_labels)
        key_by_label = {v: k for k, v in label_by_key.items()}
        options      = [_NEW_CLIENT, _SKIP] + sorted(key_by_label.keys(), key=str.lower)

        rows = []
        for fid, a in assign_doc['assignments'].items():
            rows.append({
                'folder_id':   fid,
                'folder_name': a.get('folder_name', ''),
                'section':     a.get('section', ''),
                'client':      label_by_key.get(a.get('client_key', ''),
                                                a.get('client_key', '')),
                'match_type':  a.get('match_type', ''),
                'note':        '',
            })
        for fid, u in assign_doc['unassigned'].items():
            rows.append({
                'folder_id':   fid,
                'folder_name': u.get('folder_name', ''),
                'section':     u.get('section', ''),
                'client':      _NEW_CLIENT,
                'match_type':  'unassigned',
                'note':        u.get('note', ''),
            })
        for fid, u in assign_doc['skipped'].items():
            rows.append({
                'folder_id':   fid,
                'folder_name': u.get('folder_name', ''),
                'section':     u.get('section', ''),
                'client':      _SKIP,
                'match_type':  'skipped',
                'note':        '',
            })

        review_df = pd.DataFrame(rows)
        edited = st.data_editor(
            review_df,
            column_config={
                'folder_id':   None,   # hidden
                'folder_name': st.column_config.TextColumn('Folder', disabled=True),
                'section':     st.column_config.TextColumn('Section', disabled=True),
                'client':      st.column_config.SelectboxColumn(
                    'Assigned client', options=options, required=True),
                'match_type':  st.column_config.TextColumn('Match', disabled=True),
                'note':        st.column_config.TextColumn('Note', disabled=True),
            },
            hide_index=True, use_container_width=True, key='ds_review_editor',
        )

        if st.button('💾 Save assignments', key='ds_save_assign'):
            for _, row in edited.iterrows():
                fid    = row['folder_id']
                choice = row['client']
                meta   = (assign_doc['assignments'].get(fid)
                          or assign_doc['unassigned'].get(fid)
                          or assign_doc['skipped'].get(fid) or {})
                base = {'folder_name': meta.get('folder_name', row['folder_name']),
                        'section':     meta.get('section', row['section'])}
                was_key = (assign_doc['assignments'].get(fid) or {}).get('client_key')
                if choice == _SKIP:
                    assign_doc['assignments'].pop(fid, None)
                    assign_doc['unassigned'].pop(fid, None)
                    assign_doc['skipped'][fid] = base
                elif choice == _NEW_CLIENT:
                    assign_doc['assignments'].pop(fid, None)
                    assign_doc['skipped'].pop(fid, None)
                    if fid not in assign_doc['unassigned']:
                        assign_doc['unassigned'][fid] = {
                            **base, 'first_seen': date.today().isoformat(), 'note': ''}
                else:
                    new_key = key_by_label.get(choice, choice)
                    if new_key != was_key:
                        assign_doc['unassigned'].pop(fid, None)
                        assign_doc['skipped'].pop(fid, None)
                        assign_doc['assignments'][fid] = {
                            **base,
                            'client_key':  new_key,
                            'match_type':  'manual',
                            'assigned_at': date.today().isoformat(),
                        }
            _save_assignments(gcs, assign_doc)
            st.success('Assignments saved.')
            st.rerun()

# ── 4 · Sync ───────────────────────────────────────────────────────────────

st.divider()
st.subheader('2 · Sync client profiles')

if not assign_doc['assignments'] and not assign_doc['unassigned']:
    st.info('No folders assigned yet — scan sections above first.')
    st.stop()

sync_state = _load_sync_state(gcs)
folder_last_sync, folder_file_counts = _folder_sync_index(sync_state)
proposed_state: dict = sync_state.get('proposed') or {}

client_folders: dict[str, list[str]] = {}
for _fid, _a in assign_doc['assignments'].items():
    if _a.get('client_key'):
        client_folders.setdefault(_a['client_key'], []).append(_fid)

assigned_keys = sorted(client_folders,
                       key=lambda k: client_labels.get(k, k).lower())

client_last_sync = {
    k: max((folder_last_sync.get(f, '') for f in fids), default='')
    for k, fids in client_folders.items()
}
client_tracked = {
    k: sum(folder_file_counts.get(f, 0) for f in fids)
    for k, fids in client_folders.items()
}
_stale_before = (date.today() - timedelta(days=_STALE_DAYS)).isoformat()

# ── Client selection ───────────────────────────────────────────────────────

st.markdown('**Clients to sync**')

# Quick-select buttons run before the multiselect is created, so writing its
# session-state key here is what sets the widget's value for this run.
qs1, qs2, qs3, qs4, _qs5 = st.columns([1, 1, 1.4, 1.8, 4])
_quick = None
if qs1.button('All', key='ds_pick_all'):
    _quick = assigned_keys
if qs2.button('None', key='ds_pick_none'):
    _quick = []
if qs3.button('Never synced', key='ds_pick_never',
              help='Clients with no synced documents on record.'):
    _quick = [k for k in assigned_keys if not client_last_sync.get(k)]
if qs4.button(f'Stale (> {_STALE_DAYS} days)', key='ds_pick_stale',
              help=f'Never synced, or last synced more than {_STALE_DAYS} days ago.'):
    _quick = [k for k in assigned_keys
              if client_last_sync.get(k, '') < _stale_before]
if _quick is not None:
    st.session_state.ds_sync_pick = _quick
    st.rerun()

# Keep the stored selection valid if assignments changed since it was made
st.session_state.ds_sync_pick = [
    k for k in st.session_state.get('ds_sync_pick', assigned_keys)
    if k in client_folders
]


def _client_option_label(key: str) -> str:
    last = client_last_sync.get(key) or 'never synced'
    n_f  = len(client_folders.get(key, []))
    return (f"{client_labels.get(key, key)}  ·  {last}"
            f"  ·  {n_f} folder{'s' if n_f != 1 else ''}")


sync_selection = st.multiselect(
    'Clients to sync',
    options=assigned_keys,
    format_func=_client_option_label,
    key='ds_sync_pick',
    label_visibility='collapsed',
    help='Pick exactly the clients to process. Only clients with new or changed '
         'Drive documents cost anything — unchanged clients are skipped for free.',
)

with st.expander(f'📊 Sync status of all {len(assigned_keys)} assigned clients'):
    st.dataframe(
        pd.DataFrame([{
            'client':       client_labels.get(k, k),
            'folders':      len(client_folders[k]),
            'files tracked': client_tracked.get(k, 0),
            'last synced':  client_last_sync.get(k) or '—',
            'selected':     k in set(sync_selection),
        } for k in assigned_keys]),
        hide_index=True, use_container_width=True,
    )

# ── New-client proposal selection ──────────────────────────────────────────

unassigned_ids = sorted(
    assign_doc['unassigned'].keys(),
    key=lambda f: str(assign_doc['unassigned'][f].get('folder_name', '')).lower(),
)

new_folder_ids: list[str] = []
if unassigned_ids:
    st.markdown('**New clients to profile** '
                f'({len(unassigned_ids)} unassigned folders)')

    ps1, ps2, ps3, _ps4 = st.columns([1, 1, 1.6, 6.4])
    _pquick = None
    if ps1.button('All', key='ds_prop_all'):
        _pquick = unassigned_ids
    if ps2.button('None', key='ds_prop_none'):
        _pquick = []
    if ps3.button('Never proposed', key='ds_prop_never',
                  help='Folders that have never produced a proposal.'):
        _pquick = [f for f in unassigned_ids if not proposed_state.get(f)]
    if _pquick is not None:
        st.session_state.ds_prop_pick = _pquick
        st.rerun()

    if 'ds_prop_pick' not in st.session_state:
        st.session_state.ds_prop_pick = [
            f for f in unassigned_ids if not proposed_state.get(f)]
    st.session_state.ds_prop_pick = [
        f for f in st.session_state.ds_prop_pick if f in assign_doc['unassigned']]

    def _prop_option_label(fid: str) -> str:
        meta = assign_doc['unassigned'].get(fid, {})
        last = proposed_state.get(fid)
        return (f"{meta.get('folder_name', fid)}  ·  {meta.get('section', '')}"
                f"  ·  {'last proposed ' + last if last else 'never proposed'}")

    new_folder_ids = st.multiselect(
        'Unassigned folders to build a proposed profile from',
        options=unassigned_ids,
        format_func=_prop_option_label,
        key='ds_prop_pick',
        label_visibility='collapsed',
        help='Each selected folder costs a folder download + one Claude call '
             '(~1 min). Defaults to folders never proposed before.',
    )

# ── Run options ────────────────────────────────────────────────────────────

opt1, opt2, opt3 = st.columns([1, 1, 2])
with opt1:
    full_resync = st.checkbox(
        'Full re-scan', key='ds_full_resync',
        help='Ignore file history and reprocess every document.')
with opt2:
    dry_run = st.checkbox(
        'Dry run', key='ds_dry_run',
        help='Report what would change — no profile writes, no file-history updates.')
with opt3:
    budget_label = st.selectbox(
        'Time budget', options=list(_TIME_BUDGETS.keys()),
        index=list(_TIME_BUDGETS).index(_DEFAULT_BUDGET_LABEL),
        key='ds_time_budget',
        help='How long the job may run before it stops gracefully and defers the '
             'rest. The Cloud Run task timeout is 24 h — anything up to that is '
             'safe; longer budgets just let one run finish more clients.')
budget_s = _TIME_BUDGETS[budget_label]

with st.expander('⚙️ Advanced — per-client document caps'):
    a1, a2 = st.columns(2)
    with a1:
        max_docs = st.number_input(
            'Max documents per client', min_value=1, max_value=200, value=40,
            step=5, key='ds_max_docs',
            help='Newest-first. Documents past the cap are deferred to the next run.')
    with a2:
        char_cap = st.number_input(
            'Max characters per client', min_value=20_000, max_value=600_000,
            value=150_000, step=10_000, key='ds_char_cap',
            help='Total extracted text sent to Claude for one client.')

sel_folder_ids = [fid for fid, a in assign_doc['assignments'].items()
                  if a.get('client_key') in set(sync_selection)]

st.caption(
    f'**{len(sync_selection)}** clients ({len(sel_folder_ids)} folders) will be '
    f'checked for changed documents'
    + (f'; **{len(new_folder_ids)}** unassigned folders will produce new-client '
       f'proposals' if new_folder_ids else '')
    + f'. Time budget: **{budget_label}**. '
      'Changed documents are merged into profiles by Claude and re-embedded.'
)

if st.button('🚀 Sync', type='primary', key='ds_sync_btn',
             disabled=not (sel_folder_ids or new_folder_ids)):
    run_id = f"drive_sync_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    try:
        config = {
            'run_id':                run_id,
            'drive_id':              drive_id,
            'folder_ids':            sel_folder_ids,
            'new_client_folder_ids': new_folder_ids,
            'full_resync':           bool(full_resync),
            'dry_run':               bool(dry_run),
            'max_docs_per_client':   int(max_docs),
            'per_client_char_cap':   int(char_cap),
            # Explicitly selected folders are all honoured — the time budget is
            # the only thing that can defer them.
            'max_proposals':         max(len(new_folder_ids), 1),
            'task_timeout_s':        budget_s,
            'model':                 _SYNC_MODEL,
        }
        with st.spinner('Writing job config…'):
            config_blob_path = _write_config(gcs, config)
        with st.spinner('Triggering Cloud Run job…'):
            _trigger_job(_get_credentials(), config_blob_path)
        st.session_state.ds_active_run = run_id
        st.rerun()
    except Exception as e:
        st.error(f'Failed to start sync job: {e}')
        st.code(traceback.format_exc())

# ── 5 · Results ────────────────────────────────────────────────────────────

status = st.session_state.ds_last_status
if not status:
    st.stop()

st.divider()
st.subheader('3 · Last sync results')
st.caption(f"Run ID: `{status.get('run_id', '')}`"
           + ('  ·  **DRY RUN — nothing was written**' if status.get('dry_run') else ''))

m1, m2, m3, m4 = st.columns(4)
m1.metric('Updated',   status.get('clients_updated', 0))
m2.metric('Unchanged', status.get('clients_unchanged', 0))
m3.metric('Errored',   status.get('clients_errored', 0))
m4.metric('Files changed', f"{status.get('files_changed', 0):,}")

n_deferred_props = status.get('proposals_deferred', 0)
if status.get('stopped_early') or n_deferred_props:
    parts = []
    if status.get('stopped_early'):
        budget_h = (status.get('task_timeout_s') or 0) / 3600
        parts.append('the run hit its time budget'
                     + (f' ({budget_h:g} h)' if budget_h else '')
                     + ' and stopped early')
    if n_deferred_props:
        parts.append(f'{n_deferred_props} new-client folders were deferred '
                     f"(cap {status.get('max_proposals', '?')} proposals per run)")
    st.warning('⏳ ' + ' — '.join(parts).capitalize()
               + '. Run **Sync** again to continue where it left off; '
                 'already-synced clients are skipped for free.')

results = status.get('results') or []
if results:
    res_df = pd.DataFrame(results)
    res_df['client'] = res_df['client_key'].map(lambda k: client_labels.get(k, k))
    show_cols = [c for c in ['client', 'outcome', 'files_processed',
                             'summary_changed', 'note'] if c in res_df.columns]
    st.dataframe(res_df[show_cols], hide_index=True, use_container_width=True)

skipped = status.get('files_skipped') or []
if skipped:
    with st.expander(f'⚠️ {len(skipped)} files skipped'):
        st.dataframe(pd.DataFrame(skipped), hide_index=True, use_container_width=True)

# ── 6 · New-client review queue ────────────────────────────────────────────

proposals = [p for p in (status.get('new_client_proposals') or [])
             if not p.get('error') and p.get('proposed_summary')]
failed_proposals = [p for p in (status.get('new_client_proposals') or [])
                    if p.get('error') or not p.get('proposed_summary')]

if failed_proposals:
    with st.expander(f'⚠️ {len(failed_proposals)} folders could not be proposed'):
        for p in failed_proposals:
            st.markdown(f"- **{p.get('folder_name', '?')}** — "
                        f"{p.get('error') or 'no usable summary'}")

if proposals:
    st.divider()
    st.subheader(f'4 · New clients found ({len(proposals)})')
    st.caption(
        'Review each proposal, confirm the company name, and enter the website '
        '(required — it is the client dedup key). Proposals with a website '
        'already filled in are pre-approved — uncheck any you want to hold back. '
        'Approved clients are created in '
        f'`{_CLIENTS_PREFIX}` and their folder becomes a normal assignment.'
    )

    approvals = []
    for p in proposals:
        fid = p['folder_id']
        with st.expander(f"🆕 {p.get('folder_name', fid)}", expanded=False):
            name = st.text_input('Company name', value=p.get('proposed_name', ''),
                                 key=f'ds_np_name_{fid}')
            website = st.text_input(
                'Website *', value=p.get('proposed_website', ''),
                key=f'ds_np_web_{fid}',
                placeholder='https://example.com — required',
            )
            if p.get('website_source') == 'domain_match':
                st.caption('🔗 Website inferred from email/share domains around '
                           'this folder — verify before approving.')
            cand = p.get('candidate_domains') or []
            if cand:
                st.caption('Domains seen in docs & sharing: '
                           + ', '.join(cand[:6]))
            summary = st.text_area(
                'Matching summary (embedded for grant matching)',
                value=p.get('proposed_summary', ''), height=180,
                key=f'ds_np_sum_{fid}',
            )
            if p.get('docs_summary'):
                st.caption('**Docs digest:**')
                st.text(p['docs_summary'][:3000])
            approve = st.checkbox('Approve — create this client',
                                  value=bool((p.get('proposed_website') or '').strip()),
                                  key=f'ds_np_ok_{fid}')
            if approve:
                approvals.append({
                    'folder_id':   fid,
                    'folder_name': p.get('folder_name', ''),
                    'name':        name.strip(),
                    'website':     website.strip(),
                    'summary':     summary.strip(),
                    'docs_data':   p.get('docs_data', ''),
                    'docs_summary': p.get('docs_summary', ''),
                })

    invalid = [a for a in approvals
               if not (a['name'] and a['website'] and a['summary'])]
    if invalid:
        st.warning('Approved clients need a name, website, and summary: '
                   + ', '.join(a['folder_name'] or a['folder_id'] for a in invalid))

    if st.button(f'✅ Create {len(approvals)} approved client(s)',
                 type='primary', key='ds_create_btn',
                 disabled=not approvals or bool(invalid)):
        try:
            tp    = TextProcessor(api_key=st.secrets['openai_api_key'])
            today = date.today().isoformat()
            new_rows = []
            with st.spinner('Embedding and creating clients…'):
                for a in approvals:
                    website = a['website']
                    if not website.startswith(('http://', 'https://')):
                        website = 'https://' + website
                    # float64 to match the dtype of existing rows — pyarrow
                    # cannot mix float32 and float64 ndarrays in one column
                    emb = np.array(tp.get_embedding(a['summary']), dtype=np.float64)
                    new_rows.append({
                        'company_name':       a['name'],
                        'companyWebsite':     website,
                        'summary':            a['summary'],
                        'embeddings':         emb,
                        'uuid':               str(uuid.uuid4()),
                        'scraped_at':         today,
                        'client_docs_data':   a['docs_data'],
                        'client_docs_summary': a['docs_summary'],
                        'docs_updated_at':    today,
                    })
                new_df   = pd.DataFrame(new_rows)
                gcs_path = (f'{_CLIENTS_PREFIX}drive_sync_{today}_'
                            f'{_secrets.token_hex(3)}.parquet')
                BucketManager(_BUCKET, client=gcs).upload_file(gcs_path, new_df)

                # Folder becomes a normal assignment so the next sync updates it
                for a, row in zip(approvals, new_rows):
                    fid = a['folder_id']
                    meta = assign_doc['unassigned'].pop(fid, {})
                    assign_doc['assignments'][fid] = {
                        'folder_name': meta.get('folder_name', a['folder_name']),
                        'section':     meta.get('section', ''),
                        'client_key':  f"{row['company_name']}||{row['companyWebsite']}",
                        'match_type':  'manual',
                        'assigned_at': today,
                    }
                _save_assignments(gcs, assign_doc)

            # Drop handled proposals from the displayed status
            handled = {a['folder_id'] for a in approvals}
            status['new_client_proposals'] = [
                p for p in status.get('new_client_proposals', [])
                if p.get('folder_id') not in handled
            ]
            st.session_state.ds_last_status = status
            st.session_state.pop('ds_frames', None)
            st.success(f'Created {len(new_rows)} client(s) → `{gcs_path}`')
            st.rerun()
        except Exception as e:
            st.error(f'Failed to create clients: {e}')
            st.code(traceback.format_exc())
