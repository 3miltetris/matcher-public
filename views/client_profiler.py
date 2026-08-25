"""
Client Profiles
---------------
Builds multi-aspect capability profiles for the clients in
data/all-contacts/clients/ out of material that already exists on their
rows — the website summary/scrape, Drive document extractions written by
drive-sync-job, and Deep Research output written by the Client Research
view. Claude splits that material into a handful of distinct, independently
searchable aspects; each aspect is embedded separately and the profile is
stored as one row per company in data/client-profiles/profiles.parquet.

Nothing here writes to the client parquets — profiles live in their own
store, so Client Editor / Client Research / Drive Sync can keep rewriting
client rows without touching profiles. When a client's source material
changes, its profile is flagged ⚠️ stale (source fingerprint mismatch) and
can be rebuilt.

Building runs in the `client-profile-job` Cloud Run Job (this view writes a
config to client-profile-configs/ and polls client-profile-jobs/{run_id}/
status.json) so a large batch neither ties up the Streamlit process nor dies
with the browser tab. Editing a single profile still re-embeds in-process —
that's one company and a handful of embeddings.

The Bulk Aspect Match view consumes these profiles.

Deleting is admin-gated (src/modules/access_control.py): section 3 bulk-deletes
profiles for companies that are no longer clients, optionally removing their
contact rows from data/all-contacts/clients/ as well (archived to
data/deleted-clients/ first — see src/modules/client_delete.py).
"""

import io
import json
import time
import traceback
from datetime import date, datetime

import pandas as pd
import streamlit as st
from google.cloud import run_v2, storage
from google.oauth2 import service_account

import src.modules.access_control as ac
import src.modules.aspect_profile as ap
import src.modules.client_delete as cd
from src.modules.Embedding.text_embedder import TextProcessor

# ── Constants ──────────────────────────────────────────────────────────────

_BUCKET         = ap.BUCKET
_CLIENTS_PREFIX = ap.CLIENTS_PREFIX
_CFG_PREFIX     = 'client-profile-configs/'
_STATUS_PREFIX  = 'client-profile-jobs/'
_JOB_NAME       = 'projects/cc-matcher-v1/locations/us-central1/jobs/client-profile-job'
_POLL_INTERVAL  = 10
_JOB_WORKERS    = 4

_STATUS_NONE    = '— none'
_STATUS_CURRENT = '✅ current'
_STATUS_STALE   = '⚠️ stale'


# ── GCS / Cloud Run ────────────────────────────────────────────────────────

def _get_credentials():
    return service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )


def _get_storage_client() -> storage.Client:
    return storage.Client(credentials=_get_credentials())


def _write_config(client: storage.Client, config: dict) -> str:
    blob_path = f"{_CFG_PREFIX}{config['run_id']}.json"
    client.bucket(_BUCKET).blob(blob_path).upload_from_string(
        json.dumps(config), content_type='application/json'
    )
    return blob_path


def _trigger_job(credentials, config_blob_path: str) -> None:
    run_v2.JobsClient(credentials=credentials).run_job(
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


# ── Directory assembly ─────────────────────────────────────────────────────

def _build_directory(combined: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    stored = {}
    if not profiles.empty:
        for _, p in profiles.iterrows():
            stored[p['company_key']] = p

    rows = []
    for key, group in combined.groupby('_key', sort=False):
        merged    = ap.merge_company_row(group)
        available = ap.assemble_source_texts(merged)
        fp        = ap.source_fingerprint(available)
        prof      = stored.get(key)

        if prof is None:
            status = _STATUS_NONE
        elif str(prof.get('source_fingerprint') or '') == fp:
            status = _STATUS_CURRENT
        else:
            status = _STATUS_STALE

        rows.append({
            '_key':        key,
            'company':     str(merged.get('company_name') or key.split('||', 1)[0] or '—'),
            'website':     str(merged.get('companyWebsite') or ''),
            'contacts':    len(group),
            'sources':     ', '.join(available.keys()) or '—',
            'material':    sum(len(t) for t in available.values()),
            'status':      status,
            'aspects':     int(prof['n_aspects']) if prof is not None and pd.notna(prof.get('n_aspects')) else 0,
            'built_at':    str(prof.get('built_at') or '') if prof is not None else '',
            '_fingerprint': fp,
            '_available':  list(available.keys()),
            '_row':        merged,
        })

    return pd.DataFrame(rows).sort_values(
        'company', key=lambda s: s.astype(str).str.lower()
    ).reset_index(drop=True)


# ── Delete report ──────────────────────────────────────────────────────────

def _render_delete_report() -> None:
    """Outcome of the last bulk delete. Rendered in section 3, and also on the
    'no profiles yet' path — deleting the last profile lands there."""
    report = st.session_state.get('cp_delete_report')
    if not report:
        return
    st.success('Deletion complete.')
    st.markdown(cd.format_report(report))
    for err in report['errors']:
        st.warning(err)
    if st.button('Dismiss', key='cp_del_dismiss'):
        st.session_state.cp_delete_report = None
        st.rerun()


# ── Session state ──────────────────────────────────────────────────────────

for _k in ['cp_frames', 'cp_profiles', 'cp_build_summary', 'cp_active_run']:
    if _k not in st.session_state:
        st.session_state[_k] = None
# Bumped after every build so the picker's checkbox state resets instead of
# carrying "build" ticks over onto rows that were just profiled.
if 'cp_build_nonce' not in st.session_state:
    st.session_state.cp_build_nonce = 0


# ── Page ───────────────────────────────────────────────────────────────────

st.title('🧩 Client Profiles')
st.caption(
    'Split each client into independently searchable aspects — built from the '
    'website summary, Drive documents, and Deep Research already on their rows — '
    'and embed each aspect separately for multi-aspect grant matching.'
)

# ── Active run polling ─────────────────────────────────────────────────────
# Placed before the GCS loads below so a poll cycle costs one small status
# read rather than re-downloading every client parquet every 10 seconds.

if st.session_state.cp_active_run:
    run_id = st.session_state.cp_active_run
    st.subheader('🔄 Profile build in progress')
    st.caption(f'Run ID: `{run_id}`')
    try:
        status = _poll_status(_get_storage_client(), run_id)
        if status is None or status.get('state') == 'running':
            done  = (status or {}).get('clients_done', 0)
            total = (status or {}).get('clients_total', 0)
            if total:
                st.progress(done / total, text=f'{done}/{total} clients profiled')
            else:
                st.info(f'Job is starting… checking again in {_POLL_INTERVAL}s.')
            if st.button('Cancel monitoring (job keeps running)'):
                st.session_state.cp_active_run = None
                st.rerun()
            time.sleep(_POLL_INTERVAL)
            st.rerun()
        elif status.get('state') == 'error' or status.get('error'):
            st.error('Profile build job failed.')
            st.code(status.get('error') or 'unknown error', language='text')
            if st.button('Dismiss'):
                st.session_state.cp_active_run = None
                st.rerun()
        else:
            st.session_state.cp_build_summary = {
                'built':    [b.get('company_name') or '—' for b in status.get('built') or []],
                'errors':   list(status.get('errors') or []),
                'deferred': list(status.get('deferred') or []),
                'run_id':   run_id,
                'dry_run':  bool(status.get('dry_run')),
            }
            st.session_state.cp_active_run   = None
            st.session_state.cp_profiles     = None   # rebuilt store — reload
            st.session_state.cp_build_nonce += 1
            st.rerun()
    except Exception as e:
        st.error(f'Error polling status: {e}')
        st.code(traceback.format_exc())
        if st.button('Stop monitoring'):
            st.session_state.cp_active_run = None
            st.rerun()
    st.stop()

with st.expander('Resume monitoring a previous build job'):
    resume_id = st.text_input(
        'Run ID', key='cp_resume_run_id',
        placeholder='client_profile_2026-08-19_10-30-00',
    )
    if st.button('Check status', key='cp_resume_btn') and resume_id.strip():
        st.session_state.cp_active_run = resume_id.strip()
        st.rerun()

col_reload, col_info = st.columns([1, 5])
with col_reload:
    if st.button('↺ Reload', help='Refresh clients and profiles from GCS'):
        st.session_state.cp_frames   = None
        st.session_state.cp_profiles = None
        st.rerun()

if st.session_state.cp_frames is None:
    with st.spinner('Loading clients from GCS…'):
        frames, load_errors = _load_client_frames()
    st.session_state.cp_frames = frames
    for err in load_errors:
        st.warning(err)

if st.session_state.cp_profiles is None:
    with st.spinner('Loading profile store…'):
        try:
            st.session_state.cp_profiles = ap.load_profiles(_get_storage_client())
        except Exception as e:
            st.error(f'Could not load {ap.PROFILES_BLOB}: {e}')
            st.session_state.cp_profiles = ap.empty_profiles_df()

frames: dict[str, pd.DataFrame] = st.session_state.cp_frames
profiles: pd.DataFrame          = st.session_state.cp_profiles

if not frames:
    st.warning(f'No parquet files found under {_CLIENTS_PREFIX} in GCS.')
    _render_delete_report()
    st.stop()

combined = pd.concat(list(frames.values()), ignore_index=True)
combined['_key'] = combined.apply(ap.company_key, axis=1)

directory = _build_directory(combined, profiles)

if st.session_state.get('cp_flash'):
    # Messages written immediately before an st.rerun() are discarded with the
    # rest of the page — hand them to the next run instead.
    st.success(st.session_state.pop('cp_flash'))

with col_info:
    st.info(
        f'{len(directory):,} client companies · '
        f'{int((directory["status"] == _STATUS_CURRENT).sum()):,} current profiles · '
        f'{int((directory["status"] == _STATUS_STALE).sum()):,} stale · '
        f'{int((directory["status"] == _STATUS_NONE).sum()):,} unprofiled'
    )

# ── Section 1 · Build profiles ─────────────────────────────────────────────

st.divider()
st.subheader('1 · Build profiles')

opt_l, opt_r = st.columns([2, 3])

with opt_l:
    target_aspects = st.slider(
        'Target aspects per client', ap.MIN_ASPECTS, ap.MAX_ASPECTS, 4,
        help='Claude aims for this many and may return one fewer or a couple more '
             'depending on how much distinct material a client has.',
    )
    model = st.selectbox('Model', ap.ASPECT_MODELS, index=0)

with opt_r:
    st.markdown('**Source material to use**')
    include_keys = [
        s['key'] for s in ap.SOURCES
        if st.checkbox(s['label'], value=s['default'], key=f'cp_src_{s["key"]}')
    ]
    if not include_keys:
        st.warning('Select at least one source of material.')

# ── Client picker ──────────────────────────────────────────────────────────

f_l, f_r = st.columns([2, 2])
with f_l:
    search = st.text_input('Filter clients', placeholder='name or website…')
with f_r:
    show = st.radio(
        'Show', ['Needs build (none or stale)', 'All', 'Has profile'],
        horizontal=True, key='cp_show',
    )

view = directory
if search.strip():
    q = search.strip().lower()
    view = view[
        view['company'].str.lower().str.contains(q, na=False)
        | view['website'].str.lower().str.contains(q, na=False)
    ]
if show.startswith('Needs'):
    view = view[view['status'] != _STATUS_CURRENT]
elif show == 'Has profile':
    view = view[view['status'] != _STATUS_NONE]

# Clients with no usable material can't be profiled — surface, don't offer.
no_material = view[view['sources'] == '—']
view = view[view['sources'] != '—']

if no_material.empty and view.empty:
    st.info('No clients match the current filter.')
elif view.empty:
    st.info(
        f'No profilable clients match the filter — {len(no_material)} have no '
        'website summary, Drive documents, or research data yet.'
    )

if not view.empty:
    editor_df = view[['company', 'website', 'sources', 'contacts', 'status', 'aspects', 'built_at']].copy()
    editor_df.insert(0, 'build', view['status'] != _STATUS_CURRENT)

    edited = st.data_editor(
        editor_df,
        hide_index=True,
        use_container_width=True,
        height=min(460, 60 + 36 * len(editor_df)),
        disabled=['company', 'website', 'sources', 'contacts', 'status', 'aspects', 'built_at'],
        column_config={
            'build':    st.column_config.CheckboxColumn('Build', help='Build or rebuild this profile'),
            'company':  st.column_config.TextColumn('Client'),
            'website':  st.column_config.TextColumn('Website'),
            'sources':  st.column_config.TextColumn('Material available'),
            'contacts': st.column_config.NumberColumn('Contacts', format='%d'),
            'status':   st.column_config.TextColumn('Profile'),
            'aspects':  st.column_config.NumberColumn('Aspects', format='%d'),
            'built_at': st.column_config.TextColumn('Built'),
        },
        key=f'cp_dir_editor_{st.session_state.cp_build_nonce}_{show}_{search.strip().lower()}',
    )

    selected_keys = view.loc[
        edited.index[edited['build'].fillna(False).to_numpy(dtype=bool)], '_key'
    ].tolist()

    st.caption(
        f'**{len(selected_keys)}** selected · one Claude call and up to '
        f'{target_aspects} embeddings per client, run in the '
        '`client-profile-job` Cloud Run Job — you can leave this page once it '
        'starts and resume monitoring by run ID.'
    )

    if st.button(
        f'🧩 Build {len(selected_keys)} profile{"s" if len(selected_keys) != 1 else ""}',
        type='primary',
        disabled=not selected_keys or not include_keys,
    ):
        run_id = f"client_profile_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        try:
            config = {
                'run_id':         run_id,
                'company_keys':   selected_keys,
                'sources':        include_keys,
                'target_aspects': int(target_aspects),
                'model':          model,
                'concurrency':    _JOB_WORKERS,
                'dry_run':        False,
            }
            with st.spinner('Writing job config…'):
                config_blob_path = _write_config(_get_storage_client(), config)
            with st.spinner('Triggering Cloud Run job…'):
                _trigger_job(_get_credentials(), config_blob_path)
            st.session_state.cp_active_run = run_id
            st.rerun()
        except Exception as e:
            st.error(f'Failed to start the profile build job: {e}')
            st.code(traceback.format_exc())

if not no_material.empty:
    with st.expander(f'{len(no_material)} client(s) with no profilable material'):
        st.dataframe(
            no_material[['company', 'website', 'contacts']],
            hide_index=True, use_container_width=True,
        )
        st.caption(
            'Give these clients a website summary (Client Editor), sync their '
            'Drive folder (Drive Sync), or run Client Research on them first.'
        )

if st.session_state.cp_build_summary:
    summary = st.session_state.cp_build_summary
    if summary['built']:
        st.success(
            f'Built **{len(summary["built"])}** profile'
            f'{"s" if len(summary["built"]) != 1 else ""} → `{ap.PROFILES_BLOB}`'
            + (f'  ·  run `{summary["run_id"]}`' if summary.get('run_id') else '')
        )
    elif not summary['errors']:
        st.info('The job finished without building any profiles.')
    if summary.get('deferred'):
        st.warning(
            f'{len(summary["deferred"])} client(s) hit the job time budget and were '
            'not profiled — build them again to finish: '
            + ', '.join(summary['deferred'][:20])
            + ('…' if len(summary['deferred']) > 20 else '')
        )
    for err in summary['errors']:
        st.warning(err)
    if st.button('Dismiss', key='cp_dismiss'):
        st.session_state.cp_build_summary = None
        st.rerun()

# ── Section 2 · Review & edit ──────────────────────────────────────────────

st.divider()
st.subheader('2 · Review & edit a profile')

profiles = st.session_state.cp_profiles
if profiles.empty:
    st.info('No profiles yet — build some above.')
    _render_delete_report()
    st.stop()

labels = {
    str(p['company_key']): f"{p['company_name'] or '—'}  ·  {int(p['n_aspects'] or 0)} aspects  ·  built {p['built_at']}"
    for _, p in profiles.iterrows()
}
sel_key = st.selectbox(
    'Profile', options=list(labels.keys()), format_func=lambda k: labels[k],
    key='cp_review_key',
)
prof_row = profiles[profiles['company_key'] == sel_key].iloc[0]

meta = st.columns(4)
meta[0].markdown(f"**Website:** {prof_row['companyWebsite'] or '—'}")
meta[1].markdown(f"**Sources:** {prof_row['sources_used'] or '—'}")
meta[2].markdown(f"**Model:** {prof_row['model'] or '—'}")
meta[3].markdown(f"**Built:** {prof_row['built_at'] or '—'}")

current_status = directory.loc[directory['_key'] == sel_key, 'status']
if not current_status.empty and current_status.iloc[0] == _STATUS_STALE:
    st.warning(
        'The client\'s source material has changed since this profile was built — '
        'rebuild it above to pick up the new material.'
    )

new_summary = st.text_area(
    'Profile summary',
    value=str(prof_row['profile_summary'] or ''),
    height=110,
    key=f'cp_summary_{sel_key}',
    help='Context given to the LLM re-ranker in Bulk Aspect Match. Not embedded.',
)

aspects = ap.profile_aspects(prof_row)
aspect_df = pd.DataFrame(
    aspects or [{'label': '', 'kind': 'capability', 'text': '', 'keywords': '', 'evidence': ''}]
)
for col in ('label', 'kind', 'text', 'keywords', 'evidence'):
    if col not in aspect_df.columns:
        aspect_df[col] = ''

st.markdown('**Aspects** — each row is embedded on its own and scored against every grant topic.')
edited_aspects = st.data_editor(
    aspect_df[['label', 'kind', 'text', 'keywords', 'evidence']],
    hide_index=True,
    use_container_width=True,
    num_rows='dynamic',
    column_config={
        'label':    st.column_config.TextColumn('Label', width='medium'),
        'kind':     st.column_config.SelectboxColumn('Kind', options=ap.ASPECT_KINDS, width='small'),
        'text':     st.column_config.TextColumn('Aspect text (embedded)', width='large'),
        'keywords': st.column_config.TextColumn('Keywords (embedded)', width='medium'),
        'evidence': st.column_config.TextColumn('Evidence', width='medium'),
    },
    key=f'cp_aspect_editor_{sel_key}',
)

save_col, del_col = st.columns([1, 1])

with save_col:
    if st.button('💾 Re-embed & save', type='primary', key='cp_save_edits'):
        cleaned = []
        for _, r in edited_aspects.iterrows():
            label = str(r.get('label') or '').strip()
            text  = str(r.get('text') or '').strip()
            if not label or not text:
                continue
            kind = str(r.get('kind') or '').strip().lower()
            cleaned.append({
                'label':    label[:80],
                'kind':     kind if kind in ap.ASPECT_KINDS else 'capability',
                'text':     text,
                'keywords': str(r.get('keywords') or '').strip(),
                'evidence': str(r.get('evidence') or '').strip(),
            })

        if not cleaned:
            st.error('Every aspect needs both a label and aspect text.')
        elif len(cleaned) > ap.MAX_ASPECTS:
            st.error(f'At most {ap.MAX_ASPECTS} aspects per profile.')
        else:
            try:
                tp      = TextProcessor(api_key=st.secrets['openai_api_key'])
                vectors = [tp.get_embedding(ap.aspect_embed_text(a)) for a in cleaned]
                record  = ap.build_profile_record(
                    company_key     = sel_key,
                    company_name    = str(prof_row['company_name'] or ''),
                    website         = str(prof_row['companyWebsite'] or ''),
                    profile_summary = new_summary.strip(),
                    aspects         = cleaned,
                    vectors         = vectors,
                    sources_used    = str(prof_row['sources_used'] or '').split(',') if prof_row['sources_used'] else [],
                    fingerprint     = str(prof_row['source_fingerprint'] or ''),
                    # Recorded once, however many times the profile is edited
                    model           = str(prof_row['model'] or '').replace(' + manual edit', '')
                                      + ' + manual edit',
                    built_at        = date.today().isoformat(),
                )
                merged = ap.upsert_profiles(profiles, [record])
                ap.save_profiles(_get_storage_client(), merged)
                st.session_state.cp_profiles = merged
                st.session_state.cp_flash = (
                    f'Saved {len(cleaned)} aspect(s) for {record["company_name"]}.'
                )
                st.rerun()
            except Exception as e:
                st.error(f'Save failed: {e}')
                st.code(traceback.format_exc())

with del_col:
    if ac.is_admin():
        if st.button('🗑 Delete profile', key='cp_delete'):
            try:
                merged = ap.delete_profile(profiles, sel_key)
                ap.save_profiles(_get_storage_client(), merged)
                st.session_state.cp_profiles = merged
                st.session_state.cp_flash = (
                    f'Profile deleted for {prof_row["company_name"] or sel_key} — '
                    f'the client\'s contact rows are untouched.'
                )
                st.rerun()
            except Exception as e:
                st.error(f'Delete failed: {e}')
    else:
        ac.admin_only_notice('Deleting a profile')


# ── Section 3 · Delete profiles & clients (admin only) ─────────────────────

st.divider()
st.subheader('3 · Delete profiles')

_render_delete_report()

if not ac.is_admin():
    ac.admin_only_notice('Deleting profiles')
    st.stop()

st.caption(
    'Bulk cleanup for companies that are no longer clients. Deleting a profile '
    'only removes it from the profile store — the client keeps its contact rows '
    'and can be re-profiled. Deleting the client as well removes every contact '
    f'row from `{ap.CLIENTS_PREFIX}` (backed up to `{cd.ARCHIVE_PREFIX}` first).'
)

del_labels = {
    str(p['company_key']): f"{p['company_name'] or '—'}  ·  "
                           f"{p['companyWebsite'] or 'no website'}  ·  "
                           f"{int(p['n_aspects'] or 0)} aspects"
    for _, p in profiles.iterrows()
}
if 'cp_del_keys' in st.session_state:
    # Drop selections whose profile is already gone before the widget sees them.
    st.session_state.cp_del_keys = [
        k for k in st.session_state.cp_del_keys if k in del_labels
    ]

del_keys = st.multiselect(
    'Profiles to delete',
    options=list(del_labels.keys()),
    format_func=lambda k: del_labels[k],
    key='cp_del_keys',
)

also_client = st.checkbox(
    'Also delete these clients from data/all-contacts/clients/',
    value=False,
    key='cp_del_rows',
    help='Removes every contact row of the company as well — use this for '
         'companies that are no longer clients at all.',
)
also_drive = st.checkbox(
    'Also clear their Drive Sync folder assignment', value=True,
    key='cp_del_drive', disabled=not also_client,
    help='Marks the folder skipped so Drive Sync neither syncs it nor proposes '
         'it as a new client on the next scan.',
)

if del_keys and also_client:
    counts = cd.count_rows(frames, del_keys)
    st.dataframe(
        pd.DataFrame([
            {'client': str(del_labels[k]).split('  ·  ')[0],
             'contact rows': counts.get(k, 0)}
            for k in del_keys
        ]),
        hide_index=True, use_container_width=True,
    )
    st.warning(
        f'This permanently deletes **{sum(counts.values())}** contact row(s) '
        f'in addition to **{len(del_keys)}** profile(s).'
    )

confirm = st.text_input(
    'Type DELETE to confirm', key='cp_del_confirm', placeholder='DELETE',
)

if st.button(
    f'🗑 Delete {len(del_keys)} '
    f'{"client(s) + profile(s)" if also_client else "profile(s)"} permanently',
    type='primary',
    disabled=not del_keys or confirm.strip().upper() != 'DELETE',
    help='Select at least one profile and type DELETE to enable.',
):
    with st.spinner('Deleting…'):
        try:
            report = cd.delete_clients(
                _get_storage_client(),
                del_keys,
                delete_rows             = also_client,
                delete_profiles         = True,
                clear_drive_assignments = also_client and also_drive,
                actor                   = ac.current_user_email() or 'local-dev',
            )
        except Exception as e:
            st.error(f'Delete failed: {e}')
            st.code(traceback.format_exc())
            st.stop()

    st.session_state.cp_delete_report = report
    st.session_state.cp_profiles = None          # reload both stores from GCS
    st.session_state.cp_frames   = None
    for _k in ('cp_del_keys', 'cp_del_confirm'):
        st.session_state.pop(_k, None)
    st.rerun()
