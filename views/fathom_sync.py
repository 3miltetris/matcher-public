"""
Fathom Meetings
---------------
Pull client-call transcripts and AI notes from the Fathom notetaker, store
them durably in GCS, and distil them per client into material the Client
Profiles builder can turn into aspects.

A meeting is attributed to a client by the email domains of its EXTERNAL
calendar invitees: a domain matching a client's companyWebsite is automatic,
anything else lands in the review table below and is remembered in
fathom-configs/assignments.json.

The heavy work runs in the fathom-sync-job Cloud Run Job — this page scans
(cheap metadata only), keeps the domain assignments, picks the clients, and
polls the job. Nothing here rewrites a client's matching `summary`: meeting
material reaches grant matching only through the multi-aspect profiles.

Requires `fathom_api_key` in secrets for the scan, and the `fathom-api-key`
Secret Manager secret for the job.
"""

import io
import json
import time
import traceback
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import streamlit as st
from google.cloud import storage

from src.modules import fathom_client as fc
import src.modules.ui_common as uc

# ── Constants ──────────────────────────────────────────────────────────────

_BUCKET          = 'cc-matcher-bucket-jeg-v1'
_CLIENTS_PREFIX  = 'data/all-contacts/clients/'
_CFG_PREFIX      = 'fathom-configs/'
_STATUS_PREFIX   = 'fathom-jobs/'
_ASSIGN_BLOB     = 'fathom-configs/assignments.json'
_SYNC_STATE_BLOB = 'fathom-configs/sync_state.json'
_INDEX_BLOB      = 'data/fathom/meetings_index.parquet'
_JOB_NAME        = 'projects/cc-matcher-v1/locations/us-central1/jobs/fathom-sync-job'
_POLL_INTERVAL   = 10
_SYNC_MODEL      = 'claude-sonnet-4-6'

_DEFAULT_LOOKBACK_DAYS = 90
_SCAN_MAX_PAGES        = 200
_STALE_DAYS            = 30

_SKIP       = '— skip —'
_UNASSIGNED = '— unassigned —'

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


# ── Credentials / IO ───────────────────────────────────────────────────────

def _get_credentials():
    return uc.get_credentials()


def _get_storage_client() -> storage.Client:
    return uc.get_storage_client()


def _fathom_key() -> str:
    return str(st.secrets.get('fathom_api_key') or '').strip()


def _load_client_frames() -> tuple[dict[str, pd.DataFrame], list[str]]:
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


def _empty_assignments() -> dict:
    return {'settings': {'lookback_days': _DEFAULT_LOOKBACK_DAYS}, 'updated_at': '',
            'assignments': {}, 'unassigned': {}, 'skipped': {}}


def _load_assignments(client: storage.Client) -> dict:
    blob = client.bucket(_BUCKET).blob(_ASSIGN_BLOB)
    if not blob.exists():
        return _empty_assignments()
    try:
        doc = json.loads(blob.download_as_text())
    except Exception:
        return _empty_assignments()
    doc.setdefault('assignments', {})
    doc.setdefault('unassigned', {})
    doc.setdefault('skipped', {})
    doc.setdefault('settings', {'lookback_days': _DEFAULT_LOOKBACK_DAYS})
    return doc


def _save_assignments(client: storage.Client, doc: dict) -> None:
    doc['updated_at'] = datetime.now(timezone.utc).isoformat()
    client.bucket(_BUCKET).blob(_ASSIGN_BLOB).upload_from_string(
        json.dumps(doc), content_type='application/json'
    )


def _load_sync_state(client: storage.Client) -> dict:
    """Written by the job: {'meetings': {recording_id: {...}},
    'clients': {client_key: last-synced date}}. Read-only here."""
    blob = client.bucket(_BUCKET).blob(_SYNC_STATE_BLOB)
    if not blob.exists():
        return {'meetings': {}, 'clients': {}}
    try:
        doc = json.loads(blob.download_as_text())
    except Exception:
        return {'meetings': {}, 'clients': {}}
    doc.setdefault('meetings', {})
    doc.setdefault('clients', {})
    return doc


def _load_index(client: storage.Client) -> pd.DataFrame:
    blob = client.bucket(_BUCKET).blob(_INDEX_BLOB)
    if not blob.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
    except Exception:
        return pd.DataFrame()


def _load_meeting_doc(client: storage.Client, blob_path: str) -> dict | None:
    blob = client.bucket(_BUCKET).blob(blob_path)
    if not blob.exists():
        return None
    try:
        return json.loads(blob.download_as_text())
    except Exception:
        return None


def _write_config(client: storage.Client, config: dict) -> str:
    return uc.write_job_config(client, _CFG_PREFIX, config)


def _trigger_job(credentials, config_blob_path: str) -> None:
    uc.trigger_job(credentials, _JOB_NAME, config_blob_path)


def _poll_status(client: storage.Client, run_id: str) -> dict | None:
    return uc.poll_status(client, _STATUS_PREFIX, run_id)


# ── Page ──────────────────────────────────────────────────────────────────
# Body lives in render() so this module can be dispatched to by a parent
# page. Only one render() runs per script run, which is what keeps the
# st.stop() calls below correct.

def render():
    # ── Session state ──────────────────────────────────────────────────────────

    for _k in ('fs_frames', 'fs_active_run', 'fs_last_status', 'fs_scan', 'fs_index'):
        if _k not in st.session_state:
            st.session_state[_k] = None


    # ── Page ───────────────────────────────────────────────────────────────────

    st.title('🎙️ Fathom Meetings')
    st.caption(
        'Ingest client-call transcripts and Fathom notes, store them in GCS, and '
        'distil them into capability material for the multi-aspect client profiles.'
    )

    gcs = _get_storage_client()

    # ── Active run polling ─────────────────────────────────────────────────────

    if st.session_state.fs_active_run:
        run_id = st.session_state.fs_active_run
        st.subheader('🔄 Fathom sync in progress')
        st.caption(f'Run ID: `{run_id}`')
        try:
            status = _poll_status(gcs, run_id)
            if status is None or status.get('state') == 'running':
                done  = (status or {}).get('clients_done', 0)
                total = (status or {}).get('clients_total', 0)
                swept = (status or {}).get('meetings_swept', 0)
                if total:
                    st.progress(min(done / total, 1.0),
                                text=f'{done}/{total} clients processed')
                elif swept:
                    st.info(f'Sweeping meetings… {swept:,} seen so far.')
                else:
                    st.info(f'Job is starting… checking again in {_POLL_INTERVAL}s.')
                if st.button('Cancel monitoring (job keeps running)'):
                    st.session_state.fs_active_run = None
                    st.rerun()
                time.sleep(_POLL_INTERVAL)
                st.rerun()
            elif status.get('error'):
                st.error('Fathom sync job failed.')
                st.code(status['error'], language='text')
                st.session_state.fs_active_run = None
            else:
                st.session_state.fs_last_status = status
                st.session_state.fs_active_run  = None
                st.session_state.pop('fs_frames', None)   # client rows changed — reload
                st.session_state.pop('fs_index', None)
                st.rerun()
        except Exception as e:
            st.error(f'Error polling status: {e}')
            st.code(traceback.format_exc())
            if st.button('Stop monitoring'):
                st.session_state.fs_active_run = None
                st.rerun()
        st.stop()

    # ── Resume monitoring ──────────────────────────────────────────────────────

    with st.expander('Resume monitoring a previous sync job'):
        resume_id = st.text_input(
            'Run ID', key='fs_resume_run_id',
            placeholder='fathom_2026-09-09_10-30-00',
        )
        if st.button('Check status', key='fs_resume_btn') and resume_id.strip():
            st.session_state.fs_active_run = resume_id.strip()
            st.rerun()

    # ── Load state ─────────────────────────────────────────────────────────────

    assign_doc = _load_assignments(gcs)
    sync_state = _load_sync_state(gcs)
    api_key    = _fathom_key()

    if st.session_state.get('fs_frames') is None:
        with st.spinner('Loading clients from GCS…'):
            frames, load_errors = _load_client_frames()
        st.session_state.fs_frames = frames
        for err in load_errors:
            st.warning(err)

    frames = st.session_state.fs_frames

    client_labels:  dict[str, str] = {}   # client_key → display label
    client_domains: dict[str, str] = {}   # client_key → own website domain
    if frames:
        combined = pd.concat(frames.values(), ignore_index=True)
        combined['_key'] = combined.apply(_company_key, axis=1)
        for key, grp in combined.groupby('_key', sort=False):
            name    = str(grp.iloc[0].get('company_name') or '').strip()
            website = str(grp.iloc[0].get('companyWebsite') or '').strip()
            client_labels[key]  = f"{name or '—'}  ·  {website or 'no website'}"
            client_domains[key] = fc.bare_domain(website)

    if not client_labels:
        st.warning('No client companies found in `data/all-contacts/clients/`.')
        st.stop()

    # domain → client_key, own website first, manual assignments winning. Same
    # resolution order the job uses, so the page never claims a different owner.
    domain_owner: dict[str, str] = {}
    for key, domain in client_domains.items():
        if domain:
            domain_owner.setdefault(domain, key)
    for domain, entry in assign_doc['assignments'].items():
        ckey = (entry or {}).get('client_key')
        if ckey:
            domain_owner[domain] = ckey

    # Ingested-call counts per client, straight from the meeting index
    if st.session_state.get('fs_index') is None:
        st.session_state.fs_index = _load_index(gcs)
    index_df = st.session_state.fs_index
    ingested_counts: dict[str, int] = {}
    if isinstance(index_df, pd.DataFrame) and not index_df.empty and 'client_key' in index_df:
        ingested_counts = index_df['client_key'].value_counts().to_dict()

    # Calls seen but not yet ingested, from the last scan's per-domain counts
    seen_counts: dict[str, int] = {}
    for domain, entry in assign_doc['assignments'].items():
        ckey = (entry or {}).get('client_key')
        if ckey:
            seen_counts[ckey] = seen_counts.get(ckey, 0) + int((entry or {}).get('meeting_count') or 0)

    client_last_sync = {k: str(v or '') for k, v in (sync_state.get('clients') or {}).items()}

    # ── 1 · Setup ──────────────────────────────────────────────────────────────

    st.divider()
    st.subheader('1 · Fathom connection')

    if not api_key:
        st.warning(
            'Add `fathom_api_key` to `.streamlit/secrets.toml` (above the '
            '`[gcp_service_account]` block) to scan meetings from this page. The '
            'sync job reads its own copy from the `fathom-api-key` Secret Manager '
            'secret, so a sync can still be triggered without it.'
        )
    else:
        with st.expander('⚙️ Test the connection', expanded=not assign_doc.get('updated_at')):
            st.caption(
                'Fathom API keys are per user: a key sees meetings its owner recorded '
                'plus meetings shared with their team, never other people\'s private '
                'calls. The recorders listed below are what this key can actually see.'
            )
            if st.button('🔌 Test connection', key='fs_test_btn'):
                try:
                    sample = []
                    for meeting in fc.iter_meetings(api_key, max_pages=1):
                        sample.append({
                            'meeting':     fc.meeting_title(meeting),
                            'date':        fc.meeting_date(meeting),
                            'recorded_by': str((meeting.get('recorded_by') or {}).get('email') or ''),
                            'external':    ', '.join(fc.external_domains(meeting)),
                        })
                        if len(sample) >= 10:
                            break
                    if sample:
                        st.success(f'Connected. Showing the {len(sample)} most recent meetings.')
                        st.dataframe(pd.DataFrame(sample), hide_index=True,
                                     use_container_width=True)
                    else:
                        st.warning('Connected, but this key returned no meetings.')
                except fc.FathomAuthError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f'Connection failed: {e}')

    # ── 2 · Scan meetings ──────────────────────────────────────────────────────

    st.divider()
    st.subheader('2 · Scan meetings and match domains')
    st.caption(
        'Metadata-only sweep — no transcripts, no LLM calls, no cost beyond Fathom '
        'API quota. Existing assignments are never overwritten.'
    )

    settings       = assign_doc.get('settings') or {}
    default_window = int(settings.get('lookback_days') or _DEFAULT_LOOKBACK_DAYS)

    sc1, sc2, _sc3 = st.columns([1.2, 1, 4])
    scan_days = sc1.number_input('Scan window (days)', min_value=1, max_value=1095,
                                 value=default_window, step=15, key='fs_scan_days')
    do_scan = sc2.button('🔍 Scan', key='fs_scan_btn', disabled=not api_key,
                         help=None if api_key else 'Needs `fathom_api_key` in secrets')

    if do_scan:
        created_after = (datetime.now(timezone.utc)
                         - timedelta(days=int(scan_days))).strftime('%Y-%m-%dT%H:%M:%SZ')
        try:
            stats: dict[str, dict] = {}
            swept = 0
            with st.spinner('Sweeping Fathom meetings (paced to stay under the rate limit)…'):
                for meeting in fc.iter_meetings(
                    api_key,
                    created_after=created_after,
                    include=('crm_matches',),
                    domains_type='one_or_more_external',
                    max_pages=_SCAN_MAX_PAGES,
                ):
                    swept += 1
                    for domain in fc.external_domains(meeting):
                        entry = stats.setdefault(domain, {
                            'domain': domain, 'meeting_count': 0,
                            'sample_titles': [], 'crm_companies': [],
                        })
                        entry['meeting_count'] += 1
                        if len(entry['sample_titles']) < 3:
                            entry['sample_titles'].append(fc.meeting_title(meeting))
                        for name in fc.crm_company_names(meeting):
                            if (name not in entry['crm_companies']
                                    and len(entry['crm_companies']) < 3):
                                entry['crm_companies'].append(name)

            today   = date.today().isoformat()
            matched = new_unassigned = 0
            for domain, entry in stats.items():
                if domain in assign_doc['skipped']:
                    continue
                owner = domain_owner.get(domain)
                if domain in assign_doc['assignments']:
                    assign_doc['assignments'][domain].update({
                        'meeting_count': entry['meeting_count'],
                        'sample_titles': entry['sample_titles'],
                        'last_seen':     today,
                    })
                    matched += 1
                    continue
                if owner:
                    assign_doc['assignments'][domain] = {
                        'client_key':    owner,
                        'match_type':    'auto',
                        'assigned_at':   today,
                        'meeting_count': entry['meeting_count'],
                        'sample_titles': entry['sample_titles'],
                        'crm_companies': entry['crm_companies'],
                        'last_seen':     today,
                    }
                    matched += 1
                    continue
                prior = assign_doc['unassigned'].get(domain) or {}
                if not prior:
                    new_unassigned += 1
                assign_doc['unassigned'][domain] = {
                    'domain':        domain,
                    'meeting_count': entry['meeting_count'],
                    'sample_titles': entry['sample_titles'],
                    'crm_companies': entry['crm_companies'] or prior.get('crm_companies') or [],
                    'first_seen':    prior.get('first_seen') or today,
                    'last_seen':     today,
                }
            _save_assignments(gcs, assign_doc)
            st.session_state.fs_scan = {
                'swept': swept, 'domains': len(stats),
                'matched': matched, 'new_unassigned': new_unassigned,
                'window_days': int(scan_days), 'scanned_at': today,
            }
            st.rerun()
        except fc.FathomAuthError as e:
            st.error(str(e))
        except fc.RateLimitStalledError as e:
            st.error(f'Fathom rate limit did not clear: {e}')
        except Exception as e:
            st.error(f'Scan failed: {e}')
            st.code(traceback.format_exc())

    scan = st.session_state.get('fs_scan')
    if scan:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric('Meetings swept',   f"{scan['swept']:,}")
        s2.metric('External domains', scan['domains'])
        s3.metric('Matched',          scan['matched'])
        s4.metric('New unmatched',    scan['new_unassigned'])
        st.caption(f"Last scan: {scan['scanned_at']} over {scan['window_days']} days")

    # ── 3 · Review domain assignments ──────────────────────────────────────────

    n_unassigned = len(assign_doc['unassigned'])
    if assign_doc['assignments'] or assign_doc['unassigned'] or assign_doc['skipped']:
        st.divider()
        st.subheader('3 · Domain assignments')
        if n_unassigned:
            st.info(
                f'{n_unassigned} external domain(s) have meetings but no client. '
                'Assign or skip them — skipped domains are never offered again.'
            )
        with st.expander('📋 Review domain assignments', expanded=bool(n_unassigned)):
            label_by_key = dict(client_labels)
            key_by_label = {v: k for k, v in label_by_key.items()}
            options      = [_UNASSIGNED, _SKIP] + sorted(key_by_label.keys(), key=str.lower)

            rows = []
            for domain, a in assign_doc['assignments'].items():
                rows.append({
                    'domain':        domain,
                    'client':        label_by_key.get(a.get('client_key', ''),
                                                      a.get('client_key', '')),
                    'match_type':    a.get('match_type', ''),
                    'meeting_count': int(a.get('meeting_count') or 0),
                    'crm_hint':      ', '.join(a.get('crm_companies') or []),
                    'sample':        ' | '.join(a.get('sample_titles') or []),
                })
            for domain, u in assign_doc['unassigned'].items():
                rows.append({
                    'domain':        domain,
                    'client':        _UNASSIGNED,
                    'match_type':    'unassigned',
                    'meeting_count': int(u.get('meeting_count') or 0),
                    'crm_hint':      ', '.join(u.get('crm_companies') or []),
                    'sample':        ' | '.join(u.get('sample_titles') or []),
                })
            for domain, u in assign_doc['skipped'].items():
                rows.append({
                    'domain':        domain,
                    'client':        _SKIP,
                    'match_type':    'skipped',
                    'meeting_count': int(u.get('meeting_count') or 0),
                    'crm_hint':      ', '.join(u.get('crm_companies') or []),
                    'sample':        ' | '.join(u.get('sample_titles') or []),
                })
            review_df = pd.DataFrame(rows).sort_values(
                'meeting_count', ascending=False, kind='stable'
            ).reset_index(drop=True)

            edited = st.data_editor(
                review_df,
                column_config={
                    'domain':        st.column_config.TextColumn('Domain', disabled=True),
                    'client':        st.column_config.SelectboxColumn(
                        'Assigned client', options=options, required=True),
                    'match_type':    st.column_config.TextColumn('Match', disabled=True),
                    'meeting_count': st.column_config.NumberColumn('Calls', disabled=True),
                    'crm_hint':      st.column_config.TextColumn(
                        'Fathom CRM match', disabled=True,
                        help='Company Fathom itself linked to the call — a hint only'),
                    'sample':        st.column_config.TextColumn(
                        'Example calls', disabled=True),
                },
                hide_index=True, use_container_width=True, key='fs_review_editor',
            )

            if st.button('💾 Save assignments', key='fs_save_assign'):
                today = date.today().isoformat()
                for _, row in edited.iterrows():
                    domain = row['domain']
                    choice = row['client']
                    meta   = (assign_doc['assignments'].get(domain)
                              or assign_doc['unassigned'].get(domain)
                              or assign_doc['skipped'].get(domain) or {})
                    base = {
                        'domain':        domain,
                        'meeting_count': int(meta.get('meeting_count') or 0),
                        'sample_titles': meta.get('sample_titles') or [],
                        'crm_companies': meta.get('crm_companies') or [],
                    }
                    was_key = (assign_doc['assignments'].get(domain) or {}).get('client_key')
                    if choice == _SKIP:
                        assign_doc['assignments'].pop(domain, None)
                        assign_doc['unassigned'].pop(domain, None)
                        assign_doc['skipped'][domain] = base
                    elif choice == _UNASSIGNED:
                        assign_doc['assignments'].pop(domain, None)
                        assign_doc['skipped'].pop(domain, None)
                        if domain not in assign_doc['unassigned']:
                            assign_doc['unassigned'][domain] = {
                                **base, 'first_seen': today}
                    else:
                        new_key = key_by_label.get(choice, choice)
                        if new_key != was_key:
                            assign_doc['unassigned'].pop(domain, None)
                            assign_doc['skipped'].pop(domain, None)
                            assign_doc['assignments'][domain] = {
                                **base,
                                'client_key':  new_key,
                                'match_type':  'manual',
                                'assigned_at': today,
                            }
                _save_assignments(gcs, assign_doc)
                st.success('Assignments saved.')
                st.rerun()

    # ── 4 · Sync ───────────────────────────────────────────────────────────────

    st.divider()
    st.subheader('4 · Sync client meetings')
    st.caption(
        'Transcripts are fetched only for calls that are new since the last run, '
        'then one Claude call per client refreshes that client\'s meeting digest.'
    )

    all_keys = sorted(client_labels, key=lambda k: client_labels[k].lower())


    def _client_option_label(key: str) -> str:
        last = client_last_sync.get(key) or 'never synced'
        seen = seen_counts.get(key, 0)
        got  = int(ingested_counts.get(key, 0) or 0)
        bits = [client_labels.get(key, key), last]
        if seen:
            bits.append(f'{seen} call(s) seen')
        if got:
            bits.append(f'{got} ingested')
        return '  ·  '.join(bits)


    # Quick-pick buttons must render BEFORE the multiselect: they write the
    # widget's session_state key, which is only legal before the widget exists.
    q1, q2, q3, q4, _q5 = st.columns([1, 1, 1.4, 1.6, 4])
    _quick = None
    if q1.button('All', key='fs_pick_all'):
        _quick = all_keys
    if q2.button('None', key='fs_pick_none'):
        _quick = []
    if q3.button('Never synced', key='fs_pick_never',
                 help='Clients the job has never processed'):
        _quick = [k for k in all_keys if not client_last_sync.get(k)]
    if q4.button('Has calls', key='fs_pick_calls',
                 help='Clients with meetings seen by the last scan or already ingested'):
        _quick = [k for k in all_keys if seen_counts.get(k) or ingested_counts.get(k)]
    if _quick is not None:
        st.session_state.fs_sync_pick = _quick
        st.rerun()

    # Keep any stored selection valid if the client list changed underneath it
    _default_pick = [k for k in all_keys if seen_counts.get(k) or ingested_counts.get(k)]
    st.session_state.fs_sync_pick = [
        k for k in st.session_state.get('fs_sync_pick', _default_pick) if k in client_labels
    ]
    sync_selection = st.multiselect(
        'Clients to sync', options=all_keys, format_func=_client_option_label,
        key='fs_sync_pick', label_visibility='collapsed',
        help='A client with no matched meetings in the window costs nothing to include.',
    )

    o1, o2, o3 = st.columns([1.2, 1.2, 1.6])
    lookback = o1.number_input('Look back (days)', min_value=1, max_value=1095,
                               value=default_window, step=15, key='fs_lookback')
    budget_label = o2.selectbox('Time budget', list(_TIME_BUDGETS),
                                index=list(_TIME_BUDGETS).index(_DEFAULT_BUDGET_LABEL),
                                key='fs_budget')
    budget_s   = _TIME_BUDGETS[budget_label]
    full_resync = o3.checkbox(
        'Full re-sync', value=False, key='fs_full',
        help='Re-ingest every call in the window, ignoring what was already synced.')
    dry_run     = o3.checkbox(
        'Dry run', value=False, key='fs_dry',
        help='Fetch and analyse, but write nothing to GCS or the client rows.')

    with st.expander('Advanced'):
        a1, a2, a3 = st.columns(3)
        max_per_client = a1.number_input('Max calls per client', min_value=1, max_value=60,
                                         value=12, step=1, key='fs_max_per_client')
        char_cap       = a2.number_input('Chars per client', min_value=20_000, max_value=400_000,
                                         value=120_000, step=10_000, key='fs_char_cap')
        tr_cap         = a3.number_input('Chars per transcript', min_value=5_000, max_value=120_000,
                                         value=40_000, step=5_000, key='fs_tr_cap')
        max_meetings   = st.number_input('Max new calls this run', min_value=10, max_value=5_000,
                                         value=400, step=50, key='fs_max_meetings',
                                         help='Sweep stops once this many new attributed '
                                              'calls have been found.')

    if st.button('🚀 Sync', type='primary', key='fs_sync_btn',
                 disabled=not sync_selection):
        run_id = f"fathom_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        try:
            config = {
                'run_id':                  run_id,
                'client_keys':             list(sync_selection),
                'lookback_days':           int(lookback),
                'created_after':           None,
                'full_resync':             bool(full_resync),
                'dry_run':                 bool(dry_run),
                'max_meetings':            int(max_meetings),
                'max_meetings_per_client': int(max_per_client),
                'per_client_char_cap':     int(char_cap),
                'transcript_char_cap':     int(tr_cap),
                'task_timeout_s':          budget_s,
                'model':                   _SYNC_MODEL,
            }
            with st.spinner('Writing job config…'):
                config_blob_path = _write_config(gcs, config)
            # Remember the window so the next visit defaults to the same one
            assign_doc.setdefault('settings', {})['lookback_days'] = int(lookback)
            _save_assignments(gcs, assign_doc)
            with st.spinner('Triggering Cloud Run job…'):
                _trigger_job(_get_credentials(), config_blob_path)
            st.session_state.fs_active_run = run_id
            st.rerun()
        except Exception as e:
            st.error(f'Failed to start sync job: {e}')
            st.code(traceback.format_exc())

    # ── 5 · Last sync results ──────────────────────────────────────────────────

    status = st.session_state.fs_last_status
    if status:
        st.divider()
        st.subheader('5 · Last sync results')
        st.caption(f"Run ID: `{status.get('run_id', '')}`"
                   + ('  ·  **DRY RUN — nothing was written**'
                      if status.get('dry_run') else ''))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Updated',   status.get('clients_updated', 0))
        m2.metric('Unchanged', status.get('clients_unchanged', 0))
        m3.metric('Errored',   status.get('clients_errored', 0))
        m4.metric('Calls ingested', f"{status.get('meetings_ingested', 0):,}")

        st.caption(
            f"Swept {status.get('meetings_swept', 0):,} meetings since "
            f"{str(status.get('created_after') or '')[:10]} · "
            f"{status.get('meetings_new', 0):,} new and attributed · "
            f"{status.get('api_calls_used', 0):,} Fathom API calls"
        )

        stopped = status.get('stopped_early')
        if stopped or status.get('meetings_deferred'):
            reason = {
                'timeout':      'the run hit its time budget',
                'rate_limit':   'Fathom rate-limited the sweep',
                'max_meetings': 'the sweep hit its max-new-calls cap',
            }.get(stopped, 'the run stopped early')
            st.warning(
                f"⏳ {reason.capitalize()}"
                + (f" — {status.get('meetings_deferred', 0)} call(s) deferred" if
                   status.get('meetings_deferred') else '')
                + '. Run **Sync** again to continue; already-synced calls are skipped '
                  'for free.'
                + (f" Detail: {status['note']}" if status.get('note') else '')
            )

        results = status.get('results') or []
        if results:
            res_df = pd.DataFrame(results)
            if 'client_key' in res_df.columns:
                res_df['client'] = res_df['client_key'].map(
                    lambda k: client_labels.get(k, k))
            show = [c for c in ['client', 'outcome', 'meetings_processed', 'note']
                    if c in res_df.columns]
            st.dataframe(res_df[show], hide_index=True, use_container_width=True)

        # A dry run writes nothing, so this is the only place its Claude output is
        # visible — without it you cannot tell a working pipeline from a good one.
        preview = status.get('dry_run_preview') or []
        if preview:
            st.markdown('**Dry-run extraction preview** — what would have been written')
            for p in preview:
                with st.expander(f"🧪 {p.get('company_name', '?')} "
                                 f"({len(p.get('meetings') or [])} call(s))"):
                    if p.get('meetings'):
                        st.caption('Calls: ' + ' · '.join(str(m) for m in p['meetings']))
                    st.text(p.get('digest') or '(no digest returned)')
                    counts = p.get('extracted_counts') or {}
                    if counts:
                        st.caption('Extracted fields: ' + ', '.join(
                            f'{k} {v}' for k, v in counts.items() if v))
                    notable = p.get('notable_updates') or []
                    if notable:
                        st.caption('Aspirational / not-yet-confirmed (kept out of the '
                                   'capability arrays):')
                        for n in notable:
                            st.caption(f'• {n}')

        unmatched = status.get('unmatched_domains') or []
        if unmatched:
            with st.expander(f'❓ {len(unmatched)} unmatched domain(s) seen by the job'):
                st.caption('These are already in the review table above — assign one and '
                           'its call history is ingested on the next run.')
                st.dataframe(pd.DataFrame(unmatched), hide_index=True,
                             use_container_width=True)

    # ── 6 · Stored meetings ────────────────────────────────────────────────────

    if isinstance(index_df, pd.DataFrame) and not index_df.empty:
        st.divider()
        st.subheader('6 · Stored transcripts and notes')
        st.caption(f'{len(index_df):,} meetings stored under `data/fathom/meetings/`.')

        with st.expander('📄 Browse stored meetings'):
            keys_present = [k for k in all_keys if k in set(index_df.get('client_key', []))]
            pick = st.selectbox(
                'Client', ['— all —'] + keys_present,
                format_func=lambda k: k if k == '— all —' else client_labels.get(k, k),
                key='fs_browse_client')
            view = (index_df if pick == '— all —'
                    else index_df[index_df['client_key'] == pick])
            show = [c for c in ['meeting_date', 'company_name', 'title', 'duration_min',
                                'n_transcript_lines', 'recorded_by_email', 'url']
                    if c in view.columns]
            st.dataframe(view[show].head(300), hide_index=True, use_container_width=True)

            if not view.empty:
                options = view.head(300).to_dict('records')
                chosen = st.selectbox(
                    'Open a meeting', list(range(len(options))),
                    format_func=lambda i: (f"{options[i].get('meeting_date', '')} · "
                                           f"{options[i].get('title', '')}"),
                    key='fs_browse_meeting')
                rec = options[chosen]
                if rec.get('summary_md'):
                    st.markdown('**Fathom summary**')
                    st.markdown(rec['summary_md'])
                if st.button('Load full transcript', key='fs_load_transcript'):
                    doc = _load_meeting_doc(gcs, str(rec.get('blob_path') or ''))
                    if doc is None:
                        st.error('Stored meeting JSON not found.')
                    else:
                        text = fc.transcript_text(doc.get('transcript'), 200_000)
                        st.download_button(
                            '⬇️ Download meeting JSON',
                            data=json.dumps(doc, ensure_ascii=False, indent=2),
                            file_name=f"fathom_{rec.get('recording_id')}.json",
                            mime='application/json', key='fs_dl_json')
                        st.text_area('Transcript', value=text, height=400,
                                     key='fs_transcript_box')
