"""
Funding Sources
---------------
The master list of funding-source websites the team watches, and the control
panel for the deep-research agent that walks them.

This replaces the hand-maintained "Active Approved Programs — Sources of
Funding" Google Sheet. Each row is one site with a cadence, optional
navigation instructions, and the agency folder its opportunities belong in.
The list lives in deep-research-configs/sources.parquet (see
src/modules/source_registry.py); nothing is stored in the sheet any more.

Running is done by the `deep-research-job` Cloud Run Job — this view writes a
config to deep-research-configs/ and polls deep-research-jobs/{run_id}/
status.json — so a 279-site sweep neither ties up the Streamlit process nor
dies with the browser tab. Cloud Scheduler fires the same job every morning
against deep-research-configs/daily_schedule.json, picking up whatever is due.

Anything the agent finds is embedded and written into the normal grant-topic
store under data/all-topics/processed/{BROAD_AGENCY}/, so it flows into Grant
Search, Bulk Matching and Bulk Aspect Match with no further wiring. Sites that
turn out to expose an API are flagged here rather than scraped forever — the
team should integrate with those directly.
"""

import json
import time
import traceback
from datetime import datetime

import pandas as pd
import streamlit as st
from google.cloud import run_v2, storage
from google.oauth2 import service_account

import src.modules.access_control as ac
import src.modules.source_registry as sr

# ── Constants ──────────────────────────────────────────────────────────────

_BUCKET        = sr.BUCKET
_CFG_PREFIX    = sr.CONFIG_PREFIX
_STATUS_PREFIX = sr.STATUS_PREFIX
_TOPICS_PREFIX = 'data/all-topics/processed/'
_JOB_NAME      = 'projects/cc-matcher-v1/locations/us-central1/jobs/deep-research-job'
_POLL_INTERVAL = 10
_DAILY_CONFIG  = f'{_CFG_PREFIX}daily_schedule.json'

_TIME_BUDGETS = {
    '1 hour': 3_600, '2 hours': 7_200, '4 hours': 14_400,
    '8 hours': 28_800, '12 hours': 43_200, '24 hours': 86_400,
}


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


def _list_agency_folders(client: storage.Client) -> list:
    """The folders under processed/ — the destinations a site can route to."""
    try:
        blobs = client.list_blobs(_BUCKET, prefix=_TOPICS_PREFIX, delimiter='/')
        list(blobs)  # consume the iterator to populate .prefixes
        return sorted(p.replace(_TOPICS_PREFIX, '').strip('/') for p in blobs.prefixes)
    except Exception:
        return []


# ── Flags ──────────────────────────────────────────────────────────────────

def _api_flag_digest(registry: pd.DataFrame) -> dict:
    """Everything an admin needs to know about sites that should not be scraped.

    Kept as one function returning plain data so the same digest can be mailed
    from the job later (views/bulk_matching.py already has a working SMTP
    sender to copy) without touching the rendering below.
    """
    if registry is None or registry.empty:
        return {'api': [], 'login': [], 'paused': []}
    api    = registry[registry['has_api']]
    login  = registry[registry['requires_login']]
    paused = registry[(registry['cadence'] == 'paused') &
                      (registry['consecutive_failures'] >= sr.MAX_CONSECUTIVE_FAILURES)]
    cols = ['source_id', 'name', 'url']
    return {
        'api':    api[cols + ['api_note']].to_dict('records'),
        'login':  login[cols].to_dict('records'),
        'paused': paused[cols + ['last_error']].to_dict('records'),
    }


# ── Session state ──────────────────────────────────────────────────────────

for _k in ('fsrc_registry', 'fsrc_active_run', 'fsrc_last_status', 'fsrc_import'):
    if _k not in st.session_state:
        st.session_state[_k] = None


st.title('🛰️ Funding Sources')
st.caption(
    'The master list of sites the deep-research agent watches, and the runs it makes. '
    'Found opportunities land in the normal grant-topic store, routed to the agency '
    'folder each site is mapped to.'
)


# ── Active run poll ────────────────────────────────────────────────────────
# Kept above the registry load so a poll costs one small status read rather
# than re-reading the whole store every 10 seconds.

if st.session_state.fsrc_active_run:
    run_id = st.session_state.fsrc_active_run
    st.subheader('🔄 Research run in progress')
    st.caption(f'Run ID: `{run_id}`')
    try:
        status = _poll_status(_get_storage_client(), run_id)
        if status is None or status.get('state') == 'running':
            done  = (status or {}).get('sites_done', 0)
            total = (status or {}).get('sites_total', 0)
            if total:
                st.progress(done / total, text=f'{done}/{total} sites checked')
                c1, c2, c3 = st.columns(3)
                c1.metric('New opportunities', status.get('opportunities_new', 0))
                c2.metric('Sites errored', status.get('sites_errored', 0))
                c3.metric('Spend so far', f"${status.get('cost_usd', 0):.2f}")
            else:
                st.info(f'Job is starting… checking again in {_POLL_INTERVAL}s.')
            if st.button('Cancel monitoring (job keeps running)'):
                st.session_state.fsrc_active_run = None
                st.rerun()
            time.sleep(_POLL_INTERVAL)
            st.rerun()
        elif status.get('state') == 'error' or status.get('error'):
            st.error('Deep research job failed.')
            st.code(status.get('error') or 'unknown error', language='text')
            if st.button('Dismiss'):
                st.session_state.fsrc_active_run = None
                st.rerun()
        else:
            st.session_state.fsrc_last_status = status
            st.session_state.fsrc_active_run  = None
            st.session_state.fsrc_registry    = None    # rows were updated
            st.rerun()
    except Exception as e:
        st.error(f'Error polling status: {e}')
        st.code(traceback.format_exc())
        if st.button('Stop monitoring'):
            st.session_state.fsrc_active_run = None
            st.rerun()
    st.stop()


with st.expander('Resume monitoring a previous run'):
    resume_id = st.text_input(
        'Run ID', key='fsrc_resume_run_id',
        placeholder='deep_research_2026-09-16_06-00-00',
    )
    if st.button('Check status', key='fsrc_resume_btn') and resume_id.strip():
        st.session_state.fsrc_active_run = resume_id.strip()
        st.rerun()


# ── Load the registry ──────────────────────────────────────────────────────

if st.session_state.fsrc_registry is None:
    try:
        with st.spinner('Loading the source list…'):
            st.session_state.fsrc_registry = sr.load_sources(_get_storage_client())
    except Exception as e:
        st.error(f'Failed to load the source list: {e}')
        st.code(traceback.format_exc())
        st.stop()

registry = st.session_state.fsrc_registry


# ── Last run summary ───────────────────────────────────────────────────────

if st.session_state.fsrc_last_status:
    s = st.session_state.fsrc_last_status
    st.success(f"Run `{s.get('run_id')}` complete."
               + ('  (dry run — nothing was written)' if s.get('dry_run') else ''))
    m = st.columns(5)
    m[0].metric('Sites checked', s.get('sites_done', 0))
    m[1].metric('Opportunities found', s.get('opportunities_found', 0))
    m[2].metric('New', s.get('opportunities_new', 0))
    m[3].metric('Saved', s.get('opportunities_saved', 0))
    m[4].metric('Spend', f"${s.get('cost_usd', 0):.2f}")

    if s.get('stopped_early'):
        reason = {
            'timeout': 'the run hit its time budget',
        }.get(s['stopped_early'], f"the run stopped early ({s['stopped_early']})")
        st.warning(f'⏳ {reason.capitalize()}. Run again to continue — sites already '
                   'checked today are skipped by their cadence, and opportunities '
                   'already stored are skipped for free.')

    if s.get('gcs_paths'):
        with st.expander(f"📁 Parquets written ({len(s['gcs_paths'])})"):
            for p in s['gcs_paths']:
                st.code(p, language='text')

    rows = s.get('results') or []
    if rows:
        with st.expander(f'Per-site results ({len(rows)})', expanded=False):
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if s.get('dry_run_preview'):
        with st.expander('🔍 Dry-run preview — what would have been saved', expanded=True):
            for prev in s['dry_run_preview']:
                st.markdown(f"**{prev.get('name')}** — {prev.get('new')} new")
                for opp in prev.get('opportunities', []):
                    st.markdown(f"- **{opp.get('title')}**"
                                + (f" · closes {opp.get('close_date')}" if opp.get('close_date') else ''))
                    st.caption((opp.get('description') or '')[:600])

    if st.button('Dismiss summary'):
        st.session_state.fsrc_last_status = None
        st.rerun()
    st.divider()


# ── Empty state / import ───────────────────────────────────────────────────

st.subheader('1 · The source list')

if registry.empty:
    st.info('No sources registered yet. Import the master-list spreadsheet below '
            'to get started.')

with st.expander('📥 Import sites from a spreadsheet',
                 expanded=bool(registry.empty)):
    st.caption('One-time seed from the "Sources of Funding" sheet (export it as CSV). '
               'Rows whose URL is already registered are skipped, so re-importing an '
               'updated export only adds what is new.')
    upload = st.file_uploader('CSV or Excel', type=['csv', 'xlsx', 'xls'],
                              key='fsrc_upload')
    if upload is not None:
        try:
            if upload.name.lower().endswith('.csv'):
                try:
                    raw = pd.read_csv(upload, dtype=str)
                except UnicodeDecodeError:
                    upload.seek(0)
                    raw = pd.read_csv(upload, dtype=str, encoding='latin-1')
            else:
                raw = pd.read_excel(upload, dtype=str)
        except Exception as e:
            st.error(f'Could not read that file: {e}')
            raw = None

        if raw is not None and not raw.empty:
            cols = ['—'] + list(raw.columns)

            def _guess(*needles):
                for c in raw.columns:
                    low = str(c).lower()
                    if any(n in low for n in needles):
                        return cols.index(c)
                return 0

            c1, c2 = st.columns(2)
            with c1:
                url_col  = st.selectbox('URL column *', cols, index=_guess('list', 'url', 'site', 'link'))
                cad_col  = st.selectbox('Cadence column', cols, index=_guess('check', 'cadence', 'frequency'))
                name_col = st.selectbox('Name column', cols, index=_guess('name', 'program', 'organi'))
            with c2:
                last_col = st.selectbox('Last-checked column', cols, index=_guess('last checked', 'last_checked'))
                inst_col = st.selectbox('Instructions column', cols, index=_guess('instruction', 'note', 'how to'))

            if url_col == '—':
                st.warning('A URL column must be mapped.')
            else:
                col_map = {
                    'url':          url_col,
                    'cadence':      None if cad_col  == '—' else cad_col,
                    'name':         None if name_col == '—' else name_col,
                    'last_checked': None if last_col == '—' else last_col,
                    'instructions': None if inst_col == '—' else inst_col,
                }
                try:
                    candidates = sr.import_rows(
                        raw, col_map, added_by=st.session_state.get('user_email', ''))
                except Exception as e:
                    st.error(f'Import failed: {e}')
                    candidates = pd.DataFrame()

                if candidates.empty:
                    st.warning('No rows with a usable URL were found.')
                else:
                    _, would_add, would_skip = sr.merge_import(registry, candidates)
                    st.markdown(f'**{len(candidates)} site(s) read** — '
                                f'{would_add} new, {would_skip} already registered.')
                    st.caption('Routing below is inferred from each host and is a '
                               'suggestion only — correct it in the table after importing. '
                               'Anything unrecognised defaults to CONSORTIUM.')
                    st.dataframe(
                        candidates['broad_agency'].value_counts()
                        .rename_axis('Destination folder').reset_index(name='Sites'),
                        use_container_width=True, hide_index=True,
                    )
                    st.dataframe(
                        candidates[['name', 'url', 'cadence', 'broad_agency',
                                    'last_checked']].head(25),
                        use_container_width=True, hide_index=True,
                    )
                    if would_add and st.button(f'Import {would_add} new site(s)',
                                               type='primary'):
                        try:
                            merged, added, skipped = sr.merge_import(registry, candidates)
                            sr.save_sources(_get_storage_client(), merged)
                            st.session_state.fsrc_registry = None
                            st.success(f'Imported {added} site(s); skipped {skipped} '
                                       'already registered.')
                            st.rerun()
                        except Exception as e:
                            st.error(f'Import failed: {e}')
                            st.code(traceback.format_exc())

if registry.empty:
    st.stop()


# ── Directory + editor ─────────────────────────────────────────────────────

due_mask  = registry.apply(sr.is_due, axis=1)
n_due     = int(due_mask.sum())
n_paused  = int((registry['cadence'] == 'paused').sum())
n_never   = int((registry['last_checked'] == '').sum())

k = st.columns(5)
k[0].metric('Sites', len(registry))
k[1].metric('Due now', n_due)
k[2].metric('Never checked', n_never)
k[3].metric('Paused', n_paused)
k[4].metric('API available', int(registry['has_api'].sum()))

agency_options = sorted(set(_list_agency_folders(_get_storage_client()))
                        | set(registry['broad_agency'])
                        | {sr.DEFAULT_BROAD_AGENCY, 'STATE'})

view = registry.copy()
view.insert(0, 'due', due_mask)

filter_col, search_col = st.columns([1, 2])
with filter_col:
    show = st.selectbox('Show', ['All', 'Due now', 'Never checked', 'Paused',
                                 'Failing', 'Has API', 'Needs login'])
with search_col:
    needle = st.text_input('Search name / URL / notes', key='fsrc_search').strip().lower()

if show == 'Due now':
    view = view[view['due']]
elif show == 'Never checked':
    view = view[view['last_checked'] == '']
elif show == 'Paused':
    view = view[view['cadence'] == 'paused']
elif show == 'Failing':
    view = view[view['consecutive_failures'] > 0]
elif show == 'Has API':
    view = view[view['has_api']]
elif show == 'Needs login':
    view = view[view['requires_login']]

if needle:
    hay = (view['name'].str.lower() + ' ' + view['url'].str.lower() + ' '
           + view['notes'].str.lower())
    view = view[hay.str.contains(needle, regex=False)]

st.caption(f'{len(view)} of {len(registry)} site(s). Edit any cell; add rows at the '
           'bottom of the table. Changes are saved when you press **Save changes**.')

edited = st.data_editor(
    view[['due', 'name', 'url', 'cadence', 'broad_agency', 'sub_agency',
          'enabled', 'max_pages', 'instructions', 'last_checked', 'last_status',
          'consecutive_failures', 'has_api', 'requires_login', 'notes',
          'source_id']],
    use_container_width=True,
    hide_index=True,
    num_rows='dynamic',
    key='fsrc_editor',
    column_config={
        'due':          st.column_config.CheckboxColumn('Due', disabled=True, width='small'),
        'name':         st.column_config.TextColumn('Name', width='medium'),
        'url':          st.column_config.TextColumn('URL', width='large', required=True),
        'cadence':      st.column_config.SelectboxColumn('Cadence', options=sr.CADENCES,
                                                         width='small'),
        'broad_agency': st.column_config.SelectboxColumn(
            'Destination folder', options=agency_options, width='medium',
            help='Which data/all-topics/processed/ folder this site\'s opportunities '
                 'are filed under.'),
        'sub_agency':   st.column_config.TextColumn(
            'Agency label', width='small',
            help='Written into each topic row\'s `agency` column. Defaults to the site name.'),
        'enabled':      st.column_config.CheckboxColumn('On', width='small'),
        'max_pages':    st.column_config.NumberColumn('Pages', min_value=1, max_value=60,
                                                      step=1, width='small'),
        'instructions': st.column_config.TextColumn(
            'Navigation instructions', width='large',
            help='Told to the agent verbatim, e.g. "open each challenge card and read '
                 'the full description".'),
        'last_checked': st.column_config.TextColumn('Last checked', disabled=True,
                                                    width='small'),
        'last_status':  st.column_config.TextColumn('Last result', disabled=True,
                                                    width='small'),
        'consecutive_failures': st.column_config.NumberColumn('Fails', disabled=True,
                                                              width='small'),
        'has_api':        st.column_config.CheckboxColumn('API', width='small'),
        'requires_login': st.column_config.CheckboxColumn('Login', width='small'),
        'notes':          st.column_config.TextColumn('Notes', width='medium'),
        'source_id':      st.column_config.TextColumn('ID', disabled=True, width='small'),
    },
)

save_col, del_col = st.columns([1, 2])
with save_col:
    if st.button('💾 Save changes', type='primary'):
        try:
            rows = []
            for _, r in edited.iterrows():
                row = r.to_dict()
                row.pop('due', None)
                url = sr.normalize_url(row.get('url', ''))
                if not url:
                    continue
                if not str(row.get('source_id') or '').strip():
                    # A row typed into the empty bottom line of the editor comes
                    # back with NaN in every column the user did not fill in —
                    # those must not overwrite the defaults from blank_row().
                    fresh = sr.blank_row(url, added_by=st.session_state.get('user_email', ''))
                    fresh.update({
                        k: v for k, v in row.items()
                        if k != 'source_id' and v is not None and v != ''
                        and not (isinstance(v, float) and pd.isna(v))
                    })
                    fresh['url'] = url
                    row = fresh
                else:
                    row['url'] = url
                rows.append(row)

            client   = _get_storage_client()
            kept_ids = {r['source_id'] for r in rows if r.get('source_id')}
            shown_ids = set(view['source_id'])
            removed   = [sid for sid in shown_ids if sid not in kept_ids]

            sr.upsert_sources(client, rows)
            if removed:
                sr.delete_sources(client, removed)
            st.session_state.fsrc_registry = None
            st.success(f'Saved {len(rows)} site(s)'
                       + (f'; removed {len(removed)}.' if removed else '.'))
            st.rerun()
        except Exception as e:
            st.error(f'Save failed: {e}')
            st.code(traceback.format_exc())
with del_col:
    st.caption('Deleting a row from the table and saving removes that site. '
               'Its stored opportunities are not touched.')

st.divider()


# ── Run ────────────────────────────────────────────────────────────────────

st.subheader('2 · Run the agent')

scope = st.radio(
    'Which sites?',
    [f'Due now ({n_due})', f'All enabled ({int(registry["enabled"].sum())})',
     'Just the ones I pick'],
    horizontal=True,
)

selected_ids = []
if scope.startswith('Just'):
    labels = {
        f"{r['name'] or r['url']} — {r['broad_agency']}": r['source_id']
        for _, r in registry.iterrows()
    }
    chosen = st.multiselect('Sites', sorted(labels), key='fsrc_pick')
    selected_ids = [labels[c] for c in chosen]

adv1, adv2, adv3 = st.columns(3)
with adv1:
    budget_label = st.selectbox('Time budget', list(_TIME_BUDGETS), index=2)
with adv2:
    concurrency = st.slider('Concurrent browsers', 1, 6, 3,
                            help='Each browser holds its own Chromium context '
                                 '(~200 MB). 3 is comfortable at 4 GiB.')
with adv3:
    dry_run = st.checkbox('Dry run', value=False,
                          help='Browses and extracts as normal, and reports a preview, '
                               'but writes no parquets and does not mark sites checked.')

with st.expander('Advanced — per-site budgets'):
    b1, b2 = st.columns(2)
    with b1:
        # A 25-page site needs well over 25 calls — darpa.mil used 45 to reach
        # 17 pages. Too low and the tool budget silently replaces the page cap.
        max_tool_calls = st.slider('Max tool calls per site', 8, 80, 45)
    with b2:
        site_timeout_s = st.slider('Max seconds per site', 60, 900, 480, step=30)
    page_override = st.number_input(
        'Override pages per site (0 = use each site\'s own value)',
        min_value=0, max_value=60, value=0, step=1)

if scope.startswith('Due'):
    n_planned, select_mode = n_due, 'due'
elif scope.startswith('All'):
    n_planned, select_mode = int(registry['enabled'].sum()), 'all'
else:
    n_planned, select_mode = len(selected_ids), 'ids'

# Measured over the first full 277-site sweep (2026-09-16, $56.40 total):
# sites that finished early averaged $0.149, sites that used their whole page
# budget averaged $0.349. Cost tracks pages visited, so a listing-heavy
# selection sits at the top of this range and a state-program selection at the
# bottom.
est_low, est_high = n_planned * 0.15, n_planned * 0.45
st.caption(f'**{n_planned} site(s)** planned · estimated Claude spend '
           f'**${est_low:.2f}–${est_high:.2f}** · browsing alone takes roughly '
           f'{max(1, n_planned * site_timeout_s // (concurrency * 60))} minute(s) '
           'at worst-case per-site time.')

if st.button(f'🛰️ Check {n_planned} site(s)', type='primary',
             disabled=(n_planned == 0)):
    run_id = f"deep_research_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    try:
        config = {
            'run_id':         run_id,
            'select':         select_mode,
            'source_ids':     selected_ids,
            'model':          'claude-sonnet-4-6',
            'concurrency':    int(concurrency),
            'max_tool_calls': int(max_tool_calls),
            'site_timeout_s': int(site_timeout_s),
            'max_pages':      int(page_override) or None,
            'task_timeout_s': _TIME_BUDGETS[budget_label],
            'dry_run':        bool(dry_run),
        }
        with st.spinner('Writing job config…'):
            config_blob_path = _write_config(_get_storage_client(), config)
        with st.spinner('Triggering Cloud Run job…'):
            _trigger_job(_get_credentials(), config_blob_path)
        st.session_state.fsrc_active_run = run_id
        st.rerun()
    except Exception as e:
        st.error(f'Failed to start the deep research job: {e}')
        st.code(traceback.format_exc())

with st.expander('🕕 Daily schedule'):
    st.caption('Cloud Scheduler fires the job every morning against a saved config, '
               'picking up whichever sites are due by their cadence. Save the settings '
               'below to change what that run does.')
    d1, d2 = st.columns(2)
    with d1:
        daily_budget = st.selectbox('Daily time budget', list(_TIME_BUDGETS), index=2,
                                    key='fsrc_daily_budget')
    with d2:
        daily_conc = st.slider('Daily concurrency', 1, 6, 3, key='fsrc_daily_conc')
    if st.button('Save daily schedule config'):
        try:
            _get_storage_client().bucket(_BUCKET).blob(_DAILY_CONFIG).upload_from_string(
                json.dumps({
                    'run_id':         'daily',
                    'select':         'due',
                    'source_ids':     [],
                    'model':          'claude-sonnet-4-6',
                    'concurrency':    int(daily_conc),
                    'max_tool_calls': 45,
                    'site_timeout_s': 480,
                    'max_pages':      None,
                    'task_timeout_s': _TIME_BUDGETS[daily_budget],
                    'dry_run':        False,
                }),
                content_type='application/json',
            )
            st.success(f'Saved to `{_DAILY_CONFIG}`.')
        except Exception as e:
            st.error(f'Could not save: {e}')
    st.caption('One-time setup (Cloud Shell):')
    st.code(
        'gcloud run jobs add-iam-policy-binding deep-research-job \\\n'
        '  --region us-central1 --project cc-matcher-v1 \\\n'
        '  --member="serviceAccount:matching-job@cc-matcher-v1.iam.gserviceaccount.com" \\\n'
        '  --role="roles/run.admin"\n\n'
        'gcloud scheduler jobs create http deep-research-daily \\\n'
        '  --schedule="0 6 * * *" \\\n'
        '  --uri="https://run.googleapis.com/v2/projects/cc-matcher-v1/locations/'
        'us-central1/jobs/deep-research-job:run" \\\n'
        f'  --message-body=\'{{"overrides":{{"containerOverrides":[{{"args":'
        f'["{_DAILY_CONFIG}"]}}]}}}}\' \\\n'
        '  --oauth-service-account-email=matching-job@cc-matcher-v1.iam.gserviceaccount.com \\\n'
        '  --location=us-central1 --time-zone="America/Chicago"',
        language='bash',
    )

st.divider()


# ── Flags ──────────────────────────────────────────────────────────────────

st.subheader('3 · Flags')

digest = _api_flag_digest(registry)

if digest['api']:
    st.warning(
        f"🔌 **{len(digest['api'])} site(s) expose an API or feed.** Scraping these is "
        'wasted effort — a direct integration would be cheaper, faster and more '
        'reliable. Review them and consider building a fetch view like Grants.gov Fetch.'
    )
    st.dataframe(pd.DataFrame(digest['api']), use_container_width=True, hide_index=True)
else:
    st.caption('No sites flagged as having an API yet.')

if digest['login']:
    with st.expander(f"🔒 Login-walled sites ({len(digest['login'])})"):
        st.caption('The agent could not see the listing without credentials. '
                   'Credentialed browsing is not supported — pause these, or have '
                   'someone check them by hand.')
        st.dataframe(pd.DataFrame(digest['login']), use_container_width=True, hide_index=True)

if digest['paused']:
    with st.expander(f"⏸️ Auto-paused after repeated failures ({len(digest['paused'])})"):
        st.caption(f'Paused after {sr.MAX_CONSECUTIVE_FAILURES} consecutive failures so '
                   'they stop consuming budget. Fix the URL or the instructions, then '
                   'set the cadence back.')
        st.dataframe(pd.DataFrame(digest['paused']), use_container_width=True, hide_index=True)

if not ac.is_admin():
    st.caption('Anyone on the team can edit this list and start a run.')
