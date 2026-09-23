"""
Capability Profiles
-------------------
Builds multi-aspect capability profiles for the companies of one pool
(src/modules/pools.py — 🏢 Clients or 🎯 Prospects, picked at the top of the
page) out of material that already exists on their contact rows — the website summary/scrape, Drive document extractions written by
drive-sync-job, and Deep Research output written by the Client Research
view. Claude splits that material into a handful of distinct, independently
searchable aspects, folds together any two that turn out to describe the same
capability, then groups them into the markets they serve — ranked 1st, 2nd,
3rd… by how core each is to the business, plus Defense whenever a DoD use case
is plausible. A second Claude pass then infers *unexplored* markets: customer
worlds the company does NOT serve, reached by linking the aspects it already
has. Those are hypotheses, so they are stored in their own columns and matched
under their own kind, never mixed into the confirmed markets. Every aspect and
every market narrative is embedded separately and the profile is stored as one
row per company in that pool's profile store (profiles.parquet for clients,
prospect_profiles.parquet for prospects).

Nothing here writes to the contact parquets — profiles live in their own
store, so Client Records / Client Research / Drive Sync can keep rewriting
contact rows without touching profiles. When a company's source material
changes, its profile is flagged ⚠️ stale (source fingerprint mismatch) and
can be rebuilt.

Building runs in the `client-profile-job` Cloud Run Job (this view writes a
config to client-profile-configs/ and polls client-profile-jobs/{run_id}/
status.json) so a large batch neither ties up the Streamlit process nor dies
with the browser tab. Editing a single profile still re-embeds in-process —
that's one company and a handful of embeddings.

The Bulk Aspect Match view consumes these profiles.

Deleting is admin-gated (src/modules/access_control.py): section 3 bulk-deletes
profiles for companies we are done with, optionally removing their
contact rows from the pool as well (archived to
data/deleted-clients/ first — see src/modules/client_delete.py).
"""

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
import src.modules.pools as pl
import src.modules.ui_common as uc
from src.modules.Embedding.text_embedder import TextProcessor

# ── Constants ──────────────────────────────────────────────────────────────

_BUCKET         = ap.BUCKET
_CFG_PREFIX     = 'client-profile-configs/'
_STATUS_PREFIX  = 'client-profile-jobs/'
_JOB_NAME       = 'projects/cc-matcher-v1/locations/us-central1/jobs/client-profile-job'
_POLL_INTERVAL  = 10
_JOB_WORKERS    = 4

_STATUS_NONE    = '— none'
_STATUS_CURRENT = '✅ current'
_STATUS_STALE   = '⚠️ stale'
# Built before markets existed: the material is unchanged, but the profile
# can't take part in a market-scoped match run until it is rebuilt.
_STATUS_NOMARKET = '⚠️ no markets'


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


def _load_pool_frames(pool: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    return pl.load_frames(_get_storage_client(), pool)


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

        n_markets = 0
        if prof is not None and pd.notna(prof.get('n_markets')):
            n_markets = int(prof['n_markets'])

        if prof is None:
            status = _STATUS_NONE
        elif str(prof.get('source_fingerprint') or '') != fp:
            status = _STATUS_STALE
        elif n_markets < 1:
            status = _STATUS_NOMARKET
        else:
            status = _STATUS_CURRENT

        rows.append({
            '_key':        key,
            'company':     str(merged.get('company_name') or key.split('||', 1)[0] or '—'),
            'website':     str(merged.get('companyWebsite') or ''),
            'contacts':    len(group),
            'sources':     ', '.join(available.keys()) or '—',
            'material':    sum(len(t) for t in available.values()),
            'status':      status,
            'aspects':     int(prof['n_aspects']) if prof is not None and pd.notna(prof.get('n_aspects')) else 0,
            'markets':     str(prof.get('market_labels') or '') if prof is not None else '',
            'unexplored':  str(prof.get('unexplored_labels') or '') if prof is not None else '',
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

st.title('🧩 Capability Profiles')
st.caption(
    'Split each company into independently searchable aspects — built from the '
    'website summary, documents, meetings and Deep Research already on their rows — '
    'and embed each aspect separately for multi-aspect grant matching.'
)

# Which directory of companies this page is working on. Everything below —
# the material read, the profile store written, the job config, the delete
# sections — follows this one selection. The cached frames/profiles/selection
# are dropped when it changes so one pool's companies can never be shown
# under the other pool's heading.
pool = uc.pool_selector(
    'cp_pool',
    clears=('cp_frames', 'cp_profiles', 'cp_build_summary', 'cp_sel_key',
            'cp_del_keys', 'cp_delete_report'),
    help='Clients are the companies we write proposals for. Prospects are '
         'targets we are pursuing — same profiles, same matching, separate '
         'store.',
)
_NOUN = pl.noun(pool)

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
                'built':      [b.get('company_name') or '—' for b in status.get('built') or []],
                'defense':    sum(1 for b in status.get('built') or [] if b.get('has_defense')),
                'unexplored': sum(int(b.get('n_unexplored') or 0)
                                  for b in status.get('built') or []),
                'errors':     list(status.get('errors') or []),
                # Aspect merges and a failed pass 2 are reported here rather than
                # as errors: the profile was still built and saved.
                'warnings':   list(status.get('warnings') or []),
                # Near-duplicate aspect pairs that did NOT merge — cosine can't
                # tell a repeat from two facets of one platform, so these are
                # for a human to judge and merge in the editor below.
                'near_pairs': [
                    (b.get('company_name') or '—', pair)
                    for b in status.get('built') or []
                    for pair in (b.get('aspect_near_pairs') or [])
                ],
                'deferred':   list(status.get('deferred') or []),
                'run_id':     run_id,
                'dry_run':    bool(status.get('dry_run')),
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
    if st.button('↺ Reload', help=f'Refresh {_NOUN}s and profiles from GCS'):
        st.session_state.cp_frames   = None
        st.session_state.cp_profiles = None
        st.rerun()

# .get(), not attribute access: pool_selector *removes* the keys it clears,
# and the init block above has already run this script pass, so a cleared key
# is absent rather than None.
if st.session_state.get('cp_frames') is None:
    with st.spinner(f'Loading {_NOUN}s from GCS…'):
        frames, load_errors = _load_pool_frames(pool)
    st.session_state.cp_frames = frames
    for err in load_errors:
        st.warning(err)

if st.session_state.get('cp_profiles') is None:
    with st.spinner('Loading profile store…'):
        try:
            st.session_state.cp_profiles = ap.load_profiles(
                _get_storage_client(), pool=pool
            )
        except Exception as e:
            st.error(f'Could not load {ap.profiles_blob(pool)}: {e}')
            st.session_state.cp_profiles = ap.empty_profiles_df()

frames: dict[str, pd.DataFrame] = st.session_state.cp_frames
profiles: pd.DataFrame          = st.session_state.cp_profiles

if not frames:
    st.warning(
        f'No parquet files found under {pl.contacts_prefix(pool)} in GCS. '
        + ('Import companies with the 🎯 Prospects destination in Import '
           'Contacts to start this pool.' if pool == pl.PROSPECTS else '')
    )
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
        f'{len(directory):,} {_NOUN} companies · '
        f'{int((directory["status"] == _STATUS_CURRENT).sum()):,} current profiles · '
        f'{int((directory["status"] == _STATUS_STALE).sum()):,} stale · '
        f'{int((directory["status"] == _STATUS_NOMARKET).sum()):,} without markets · '
        f'{int((directory["status"] == _STATUS_NONE).sum()):,} unprofiled'
    )

# ── Section 1 · Build profiles ─────────────────────────────────────────────

st.divider()
st.subheader('1 · Build profiles')

opt_l, opt_r = st.columns([2, 3])

with opt_l:
    target_aspects = st.slider(
        'Target aspects per client', ap.MIN_ASPECTS, ap.MAX_ASPECTS, 8,
        help='Claude aims for this many and may return one fewer or a couple more '
             'depending on how much distinct material a client has.',
    )
    model = st.selectbox('Model', ap.ASPECT_MODELS, index=0)
    max_markets = st.number_input(
        'Max markets (excluding Defense)', min_value=1, max_value=ap.MAX_MARKETS,
        value=ap.MAX_MARKETS, step=1,
        help='Markets are named from a fixed list so one market means the same '
             'thing for every client. Each aspect is earmarked to the markets '
             'it serves, and Bulk Aspect Match can run one market at a time.',
    )
    assess_defense = st.checkbox(
        'Assess a Defense / DoD use case', value=True,
        help='Adds a Defense market whenever there is a plausible defense '
             'application, even a loose one. When there is none, the reason is '
             'stored on the profile instead. Defense is ranked on the same '
             'scale as every other market and never counts against the cap.',
    )
    assess_unexplored = st.checkbox(
        'Assess unexplored markets', value=True,
        help='A second Claude call per client that links the aspects into '
             'markets the company does NOT serve yet, each with the gap that '
             'would remain. Roughly doubles build time per client. Stored '
             'separately from the confirmed markets and matched under their own '
             'kind in Bulk Aspect Match.',
    )
    max_unexplored = st.number_input(
        'Max unexplored markets', min_value=1, max_value=ap.MAX_UNEXPLORED,
        value=ap.MAX_UNEXPLORED, step=1, disabled=not assess_unexplored,
        help='An empty result is a legitimate answer — a single-product client '
             'may support no market beyond the ones it already serves.',
    )

with opt_r:
    st.markdown('**Source material to use**')
    include_keys = [
        s['key'] for s in ap.SOURCES
        if st.checkbox(s['label'], value=s['default'], key=f'cp_src_{s["key"]}')
    ]
    if not include_keys:
        st.warning('Select at least one source of material.')

    with st.expander('Advanced'):
        merge_threshold = st.slider(
            'Merge markets whose narratives are this similar', 0.80, 0.99,
            ap.MARKET_MERGE_THRESHOLD, 0.01,
            help='Two markets of the same company whose narrative embeddings '
                 'score above this are one market described twice, and are '
                 'folded together at build time. Lower to merge more '
                 'aggressively.',
        )
        aspect_threshold = st.slider(
            'Merge aspects whose vectors are this similar', 0.90, 0.995,
            ap.ASPECT_MERGE_THRESHOLD, 0.005,
            help='Two aspects scoring above this are one capability written '
                 'twice; the longer text survives and absorbs the other\'s '
                 'keywords and market membership. Deliberately higher than the '
                 'market threshold — a company\'s aspects are all descriptions '
                 'of one business in one register, so ada-002 puts genuinely '
                 'distinct capabilities above 0.90 already. Every merge is '
                 'named in the run report; lower this only after checking them.',
        )

# ── Client picker ──────────────────────────────────────────────────────────

f_l, f_r = st.columns([2, 2])
with f_l:
    search = st.text_input(f'Filter {_NOUN}s', placeholder='name or website…')
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

# Companies with no usable material can't be profiled — surface, don't offer.
no_material = view[view['sources'] == '—']
view = view[view['sources'] != '—']

if no_material.empty and view.empty:
    st.info(f'No {_NOUN}s match the current filter.')
elif view.empty:
    st.info(
        f'No profilable {_NOUN}s match the filter — {len(no_material)} have no '
        'website summary, documents, or research data yet.'
    )

if not view.empty:
    editor_df = view[['company', 'website', 'sources', 'contacts', 'status',
                      'aspects', 'markets', 'unexplored', 'built_at']].copy()
    editor_df.insert(0, 'build', view['status'] != _STATUS_CURRENT)

    edited = st.data_editor(
        editor_df,
        hide_index=True,
        use_container_width=True,
        height=min(460, 60 + 36 * len(editor_df)),
        disabled=['company', 'website', 'sources', 'contacts', 'status', 'aspects',
                  'markets', 'unexplored', 'built_at'],
        column_config={
            'build':    st.column_config.CheckboxColumn('Build', help='Build or rebuild this profile'),
            'company':  st.column_config.TextColumn(pl.label(pool).rstrip('s')),
            'website':  st.column_config.TextColumn('Website'),
            'sources':  st.column_config.TextColumn('Material available'),
            'contacts': st.column_config.NumberColumn('Contacts', format='%d'),
            'status':   st.column_config.TextColumn('Profile'),
            'aspects':  st.column_config.NumberColumn('Aspects', format='%d'),
            'markets':  st.column_config.TextColumn('Markets', width='medium'),
            'unexplored': st.column_config.TextColumn('Unexplored', width='medium'),
            'built_at': st.column_config.TextColumn('Built'),
        },
        # The pool is part of the key for the same reason the filter and the
        # search are: data_editor stores its edits by ROW INDEX, so a stale
        # key would re-apply them to a different list of companies.
        key=f'cp_dir_editor_{pool}_{st.session_state.cp_build_nonce}_{show}_{search.strip().lower()}',
    )

    selected_keys = view.loc[
        edited.index[edited['build'].fillna(False).to_numpy(dtype=bool)], '_key'
    ].tolist()

    st.caption(
        f'**{len(selected_keys)}** selected · one Claude call and up to '
        f'{target_aspects} embeddings per company, run in the '
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
                'pool':           pool,
                'company_keys':   selected_keys,
                'sources':        include_keys,
                'target_aspects': int(target_aspects),
                'max_markets':    int(max_markets),
                'assess_defense': bool(assess_defense),
                'assess_unexplored': bool(assess_unexplored),
                'max_unexplored': int(max_unexplored),
                'market_merge_threshold': float(merge_threshold),
                'aspect_merge_threshold': float(aspect_threshold),
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
    with st.expander(f'{len(no_material)} {_NOUN}(s) with no profilable material'):
        st.dataframe(
            no_material[['company', 'website', 'contacts']],
            hide_index=True, use_container_width=True,
        )
        st.caption(
            f'Give these {_NOUN}s a website summary (Client Records), '
            + ('sync their Drive folder (Drive Sync), ' if pl.supports_drive(pool) else '')
            + 'ingest their meetings (Fathom Meetings), or run Deep Research '
              'on them first.'
        )

if st.session_state.get('cp_build_summary'):
    summary = st.session_state.cp_build_summary
    if summary['built']:
        st.success(
            f'Built **{len(summary["built"])}** profile'
            f'{"s" if len(summary["built"]) != 1 else ""} → `{ap.profiles_blob(pool)}`'
            + (f'  ·  {summary["defense"]} with a Defense market'
               if summary.get('defense') else '')
            + (f'  ·  {summary["unexplored"]} unexplored market(s) found'
               if summary.get('unexplored') else '')
            + (f'  ·  run `{summary["run_id"]}`' if summary.get('run_id') else '')
        )
    elif not summary['errors']:
        st.info('The job finished without building any profiles.')
    if summary.get('deferred'):
        st.warning(
            f'{len(summary["deferred"])} company(s) hit the job time budget and were '
            'not profiled — build them again to finish: '
            + ', '.join(summary['deferred'][:20])
            + ('…' if len(summary['deferred']) > 20 else '')
        )
    for err in summary['errors']:
        st.warning(err)
    if summary.get('warnings'):
        with st.expander(f'{len(summary["warnings"])} build note(s) — merged '
                         'aspects and any failed unexplored pass'):
            for note in summary['warnings']:
                st.markdown(f'- {note}')
    if summary.get('near_pairs'):
        with st.expander(
            f'{len(summary["near_pairs"])} near-duplicate aspect pair(s) — review '
            'and merge by hand if they are the same capability'
        ):
            st.caption(
                'These scored close but below the merge threshold. Measurement '
                'showed cosine does not reliably separate a genuine repeat from '
                'two facets of one platform, so nothing was merged automatically '
                '— edit the profile in section 2 to combine any that are.'
            )
            for company, pair in summary['near_pairs']:
                st.markdown(f'- **{company}** · `{pair}`')
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

aspects    = ap.profile_aspects(prof_row)
markets    = ap.profile_markets(prof_row)
unexplored = ap.profile_unexplored(prof_row)

if not markets:
    st.warning(
        'This profile has no markets — it was built before markets existed, or '
        'the build produced none. Rebuild it above to match it by market; it '
        'still works in the whole-company mode of Bulk Aspect Match.'
    )
elif not any(m.get('market') == ap.DEFENSE_MARKET for m in markets):
    st.caption(
        '🎖️ No Defense market: '
        + (str(prof_row.get('dod_assessment') or '') or 'no reason recorded.')
    )

aspect_df = pd.DataFrame(
    aspects or [{'label': '', 'kind': 'capability', 'text': '', 'keywords': '',
                 'evidence': '', 'markets': []}]
)
for col in ('label', 'kind', 'text', 'keywords', 'evidence'):
    if col not in aspect_df.columns:
        aspect_df[col] = ''
# Membership is edited as text; normalize_markets canonicalises it on save.
aspect_df['markets'] = [
    ', '.join(a.get('markets') or []) for a in (aspects or [{}])
]

st.markdown(
    '**Aspects** — each row is embedded on its own and scored against every '
    'grant topic. `Markets` decides which market runs an aspect takes part in.'
)
edited_aspects = st.data_editor(
    aspect_df[['label', 'kind', 'text', 'keywords', 'markets', 'evidence']],
    hide_index=True,
    use_container_width=True,
    num_rows='dynamic',
    column_config={
        'label':    st.column_config.TextColumn('Label', width='medium'),
        'kind':     st.column_config.SelectboxColumn('Aspect type', options=ap.ASPECT_KINDS,
                                                     width='small'),
        'text':     st.column_config.TextColumn('Aspect text (embedded)', width='large'),
        'keywords': st.column_config.TextColumn('Keywords (embedded)', width='medium'),
        'markets':  st.column_config.TextColumn(
            'Markets', width='medium',
            help='Comma-separated, from: ' + ', '.join(ap.MARKET_CATEGORIES)
                 + '. An aspect left blank is put in the primary market.'),
        'evidence': st.column_config.TextColumn('Evidence', width='medium'),
    },
    key=f'cp_aspect_editor_{sel_key}',
)

market_rows = [{
    'market':    str(m.get('market') or ''),
    'tier':      ap.market_tier_rank(m),
    'subtitle':  str(m.get('subtitle') or ''),
    'narrative': str(m.get('narrative') or ''),
    'keywords':  str(m.get('keywords') or ''),
    'aspects':   ', '.join(m.get('aspect_labels') or []),
} for m in markets]
# Profiles built before markets existed start with an empty table rather than
# a seeded row: a placeholder market with no narrative fails the save check
# below, which would block every other edit on the profile until the user
# deleted a row they never added.
market_df = pd.DataFrame(
    market_rows,
    columns=['market', 'tier', 'subtitle', 'narrative', 'keywords', 'aspects'],
)
# Explicit int dtype: an empty frame leaves the column as object, and a
# NumberColumn over an object column renders the rank as text.
market_df['tier'] = (
    pd.to_numeric(market_df['tier'], errors='coerce').fillna(2).astype(int)
)

st.markdown(
    '**Markets** — each narrative is embedded too and scored alongside that '
    "market's aspects, so a Defense use case can pull in topics no single "
    'aspect would. `Tier` is a rank: 1 is the market most core to the business '
    'today. Ranks are renumbered densely on save, so gaps and ties resolve '
    'themselves. The `Aspects` column is derived from the table above.'
)
if market_df.empty:
    st.caption(
        'This profile predates markets. Add rows here to give it some, or '
        'rebuild it above — market-scoped match runs skip it until then.'
    )
edited_markets = st.data_editor(
    market_df,
    hide_index=True,
    use_container_width=True,
    num_rows='dynamic',
    disabled=['aspects'],
    column_config={
        'market':    st.column_config.SelectboxColumn('Market', options=ap.MARKET_CATEGORIES,
                                                      width='medium'),
        'tier':      st.column_config.NumberColumn(
            'Tier', min_value=1, max_value=ap.MAX_MARKET_TIER, step=1,
            format='%d', width='small',
            help='1 = most core to the business today.'),
        'subtitle':  st.column_config.TextColumn('Subtitle', width='medium'),
        'narrative': st.column_config.TextColumn('Market narrative (embedded)', width='large'),
        'keywords':  st.column_config.TextColumn('Keywords (embedded)', width='medium'),
        'aspects':   st.column_config.TextColumn('Aspects (derived)', width='medium'),
    },
    key=f'cp_market_editor_{sel_key}',
)

# -- Unexplored markets -------------------------------------------------
unexplored_df = pd.DataFrame(
    [{
        'market':    str(m.get('market') or ''),
        'tier':      ap.market_tier_rank(m),
        'subtitle':  str(m.get('subtitle') or ''),
        'narrative': str(m.get('narrative') or ''),
        'keywords':  str(m.get('keywords') or ''),
        'rationale': str(m.get('rationale') or ''),
        'aspects':   ', '.join(m.get('aspect_labels') or []),
    } for m in unexplored],
    columns=['market', 'tier', 'subtitle', 'narrative', 'keywords', 'rationale',
             'aspects'],
)
unexplored_df['tier'] = (
    pd.to_numeric(unexplored_df['tier'], errors='coerce').fillna(1).astype(int)
)

st.markdown(
    '**Unexplored markets** — markets this client does **not** serve, inferred '
    'by linking the aspects it does have. Only the narrative is embedded and '
    'scored (there are no aspects earmarked to an unexplored market), and Bulk '
    'Aspect Match re-ranks these with a different prompt that asks how plausibly '
    'the client could extend into a topic. A market listed here must not also '
    'appear in the table above.'
)
if unexplored_df.empty:
    st.caption(
        'None recorded. Either this profile predates the unexplored pass, it was '
        'built with the assessment off, or the aspects genuinely support no '
        'market beyond the ones already served — all three are normal.'
    )
edited_unexplored = st.data_editor(
    unexplored_df,
    hide_index=True,
    use_container_width=True,
    num_rows='dynamic',
    column_config={
        'market':    st.column_config.SelectboxColumn('Market', options=ap.MARKET_CATEGORIES,
                                                      width='medium'),
        'tier':      st.column_config.NumberColumn(
            'Tier', min_value=1, max_value=ap.MAX_MARKET_TIER, step=1,
            format='%d', width='small',
            help='1 = most promising unexplored market.'),
        'subtitle':  st.column_config.TextColumn('Subtitle', width='medium'),
        'narrative': st.column_config.TextColumn('Market narrative (embedded)', width='large'),
        'keywords':  st.column_config.TextColumn('Keywords (embedded)', width='medium'),
        'rationale': st.column_config.TextColumn(
            'Gap remaining (not embedded)', width='large',
            help='Which aspects combine, and what the client would still have '
                 'to build or certify. Shown to the re-ranker as context.'),
        'aspects':   st.column_config.TextColumn(
            'Draws on aspects', width='medium',
            help='Comma-separated aspect labels from the table above. Labels '
                 'that match no aspect are dropped on save.'),
    },
    key=f'cp_unexplored_editor_{sel_key}',
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
                'markets':  [s.strip() for s in str(r.get('markets') or '').split(',')
                             if s.strip()],
            })

        cleaned_markets = []
        missing_narrative = []
        for _, r in edited_markets.iterrows():
            name = ap.canonical_market(r.get('market'))
            if not name:
                continue
            narrative = str(r.get('narrative') or '').strip()
            if not narrative:
                missing_narrative.append(name)
                continue
            cleaned_markets.append({
                'market':    name,
                'tier':      ap.market_tier_rank(r.get('tier')),
                'subtitle':  str(r.get('subtitle') or '').strip()[:160],
                'narrative': narrative,
                'keywords':  str(r.get('keywords') or '').strip(),
            })

        cleaned_unexplored = []
        missing_unexp_narrative = []
        for _, r in edited_unexplored.iterrows():
            name = ap.canonical_market(r.get('market'))
            if not name:
                continue
            narrative = str(r.get('narrative') or '').strip()
            if not narrative:
                # The narrative vector is the only thing an unexplored market is
                # ever scored on, so one without it is unmatchable dead weight.
                missing_unexp_narrative.append(name)
                continue
            cleaned_unexplored.append({
                'market':    name,
                'tier':      ap.market_tier_rank(r.get('tier')),
                'subtitle':  str(r.get('subtitle') or '').strip()[:160],
                'narrative': narrative,
                'keywords':  str(r.get('keywords') or '').strip(),
                'rationale': str(r.get('rationale') or '').strip()[:600],
                'aspects':   [x.strip() for x in str(r.get('aspects') or '').split(',')
                              if x.strip()],
            })

        # normalize_unexplored would silently drop a collision. Say so instead -
        # the user meant one of the two tables, and we can't know which.
        both_tables = sorted(
            {m['market'] for m in cleaned_unexplored}
            & {m['market'] for m in cleaned_markets}
        )

        if not cleaned:
            st.error('Every aspect needs both a label and aspect text.')
        elif len(cleaned) > ap.MAX_ASPECTS:
            st.error(f'At most {ap.MAX_ASPECTS} aspects per profile.')
        elif missing_narrative or missing_unexp_narrative:
            st.error(
                'These markets need a narrative before they can be embedded: '
                + ', '.join(missing_narrative + missing_unexp_narrative)
            )
        elif both_tables:
            st.error(
                'Listed as both served and unexplored: ' + ', '.join(both_tables)
                + '. An unexplored market is one the client does *not* serve — '
                'remove it from one of the two tables.'
            )
        else:
            try:
                cleaned, cleaned_markets = ap.normalize_markets(cleaned, cleaned_markets)
                cleaned_unexplored = ap.normalize_unexplored(
                    cleaned_markets, cleaned_unexplored, ap.MAX_UNEXPLORED, cleaned
                )
                tp      = TextProcessor(api_key=st.secrets['openai_api_key'])
                vectors = [tp.get_embedding(ap.aspect_embed_text(a)) for a in cleaned]
                market_vectors = [
                    tp.get_embedding(ap.market_embed_text(m)) for m in cleaned_markets
                ]
                # Same embed text as a confirmed market: what a solicitation
                # would describe. The rationale is context for the re-ranker,
                # not part of the vector.
                unexplored_vectors = [
                    tp.get_embedding(ap.market_embed_text(m)) for m in cleaned_unexplored
                ]
                record  = ap.build_profile_record(
                    company_key     = sel_key,
                    company_name    = str(prof_row['company_name'] or ''),
                    website         = str(prof_row['companyWebsite'] or ''),
                    profile_summary = new_summary.strip(),
                    aspects         = cleaned,
                    vectors         = vectors,
                    markets         = cleaned_markets,
                    market_vectors  = market_vectors,
                    unexplored         = cleaned_unexplored,
                    unexplored_vectors = unexplored_vectors,
                    dod_assessment  = str(prof_row.get('dod_assessment') or ''),
                    sources_used    = str(prof_row['sources_used'] or '').split(',') if prof_row['sources_used'] else [],
                    fingerprint     = str(prof_row['source_fingerprint'] or ''),
                    # Recorded once, however many times the profile is edited
                    model           = str(prof_row['model'] or '').replace(' + manual edit', '')
                                      + ' + manual edit',
                    built_at        = date.today().isoformat(),
                )
                merged = ap.upsert_profiles(profiles, [record], pool=pool)
                ap.save_profiles(_get_storage_client(), merged, pool=pool)
                st.session_state.cp_profiles = merged
                st.session_state.cp_flash = (
                    f'Saved {len(cleaned)} aspect(s), {len(cleaned_markets)} '
                    f'market(s) and {len(cleaned_unexplored)} unexplored '
                    f'market(s) for {record["company_name"]}.'
                    + ('' if cleaned_markets else
                       ' No markets — this profile is skipped by market-scoped runs.')
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
                ap.save_profiles(_get_storage_client(), merged, pool=pool)
                st.session_state.cp_profiles = merged
                st.session_state.cp_flash = (
                    f'Profile deleted for {prof_row["company_name"] or sel_key} — '
                    f'the {_NOUN}\'s contact rows are untouched.'
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
    f'Bulk cleanup for {_NOUN}s we are done with. Deleting a profile only '
    f'removes it from `{ap.profiles_blob(pool)}` — the {_NOUN} keeps its '
    f'contact rows and can be re-profiled. Deleting the {_NOUN} as well removes '
    f'every contact row from `{pl.contacts_prefix(pool)}` (backed up to '
    f'`{cd.ARCHIVE_PREFIX}` first).'
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
    f'Also delete these {_NOUN}s from {pl.contacts_prefix(pool)}',
    value=False,
    key='cp_del_rows',
    help='Removes every contact row of the company as well — use this for '
         'companies that should leave this pool entirely.',
)
# Prospects never had a Drive folder, so there is nothing to park for them —
# the checkbox is not rendered at all rather than shown doing nothing.
also_drive = False
if pl.supports_drive(pool):
    also_drive = st.checkbox(
        'Also clear their Drive Sync folder assignment', value=True,
        key='cp_del_drive', disabled=not also_client,
        help='Marks the folder skipped so Drive Sync neither syncs it nor '
             'proposes it as a new client on the next scan.',
    )

if del_keys and also_client:
    counts = cd.count_rows(frames, del_keys)
    st.dataframe(
        pd.DataFrame([
            {_NOUN: str(del_labels[k]).split('  ·  ')[0],
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
    f'{f"{_NOUN}(s) + profile(s)" if also_client else "profile(s)"} permanently',
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
                pool                    = pool,
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


# ── Next step ──────────────────────────────────────────────────────────────
# Renders only when the page runs to completion; the guards above st.stop()
# on every path where there is nothing to hand on to.

st.divider()
st.caption('Next step')
st.page_link('views/aspect_match.py', label='Aspect Match', icon='🎯')
st.caption('Score these profiles against grant topics, per capability or per market.')
