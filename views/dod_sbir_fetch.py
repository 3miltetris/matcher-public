"""
DoD SBIR Fetch
--------------
Pull open / pre-release DoD SBIR & STTR topics directly from the DSIP public JSON
API (dodsbirsttr.mil/topics-app/) and save them to the canonical topic store under
data/all-topics/processed/DOD-SBIR/. No API key required.

Unlike Grants.gov Fetch there is NO relevance screening: every DoD SBIR topic is a
defense R&D solicitation, so there is nothing to filter at the source — a review
table still lets you drop any topic before saving. Search / fetch / embed logic
lives in src/modules/dod_sbir.py (Streamlit-free, so a future scheduled job can
share it); this view keeps thin st.progress wrappers.
"""

import pandas as pd
import requests
import streamlit as st
from google.cloud import storage

import src.modules.newsletter as nl
import src.modules.ui_common as uc
from src.modules import dod_sbir as ds

# ── GCS ────────────────────────────────────────────────────────────────────

_BUCKET         = 'cc-matcher-bucket-jeg-v1'
_DODSBIR_PREFIX = ds.DODSBIR_PREFIX

_RESERVED_COLS = frozenset({
    'topic_number', 'agency', 'title', 'description', 'open_date', 'close_date',
    'scraped_at', 'grant_summary', 'embeddings', 'award_ceiling', 'status', 'source',
    'is_rolling', 'last_verified_active', 'verify_status',
    *nl.COLUMNS,
})


# ── Storage client ─────────────────────────────────────────────────────────

def _get_storage_client() -> storage.Client:
    return uc.get_storage_client()


# ── thin wrappers (st.progress UI over the shared core) ─────────────────────

def _search(programs, statuses, keyword, max_results) -> tuple[pd.DataFrame, int, dict]:
    with st.spinner('Querying DoD SBIR (DSIP)…'):
        bar = st.progress(0, text='Fetching topic details…')
        df, total, report = ds.search_topics(
            programs=programs, statuses=statuses, keyword=keyword,
            max_results=max_results,
            progress=lambda d, t: bar.progress(d / t, text=f'Fetching topic details… {d}/{t}'),
        )
        bar.empty()
    return df, total, report


def _load_existing_keys(client: storage.Client) -> tuple[set[str], set[str]]:
    return ds.load_existing_keys(client)


def _embed_and_save(df: pd.DataFrame, oai_key: str, anth_key: str,
                    extra_cols: dict[str, str] | None = None) -> str:
    sum_bar = st.progress(0, text='Summarizing topics…')
    emb_bar = st.progress(0, text='Generating embeddings…')
    nl_bar  = st.progress(0, text='Newsletter screening…')
    try:
        path = ds.embed_and_save(
            df, oai_key, anth_key, _get_storage_client(), extra_cols=extra_cols,
            summary_progress=lambda d, t: sum_bar.progress(d / t, text=f'Summarizing… {d}/{t}'),
            embed_progress=lambda d, t: emb_bar.progress(d / t, text=f'Embedding {d}/{t}…'),
            newsletter_progress=lambda d, t: nl_bar.progress(d / t, text=f'Newsletter screening {d}/{t}'),
        )
    finally:
        sum_bar.empty(); emb_bar.empty(); nl_bar.empty()
    return path


# ── Page ────────────────────────────────────────────────────────────────────
# Body lives in render() so the Grant Sources parent page can dispatch to it.

def render():
    for _k in ['dsbir_raw_df', 'dsbir_existing_keys', 'dsbir_detail_report']:
        if _k not in st.session_state:
            st.session_state[_k] = None
    if 'dsbir_custom_cols' not in st.session_state:
        st.session_state.dsbir_custom_cols = []

    st.title('🪖 DoD SBIR Fetch')
    st.caption(
        'Pull open & pre-release DoD SBIR / STTR topics straight from the DSIP public '
        'API and save them to the topic store. Every topic is a defense R&D '
        'solicitation, so there is no relevance screen — review and drop any below '
        'before saving. No API key required.'
    )

    # ── Section 1 · Fetch ──────────────────────────────────────────────────────

    st.subheader('1 · Fetch from DoD SBIR')

    col_p, col_s = st.columns(2)
    with col_p:
        sel_programs = st.multiselect(
            'Program', ds.PROGRAM_OPTIONS, default=ds.PROGRAM_OPTIONS, key='dsb_programs',
            help='SBIR and/or STTR. Filtered client-side on the topic’s own program field.')
    with col_s:
        sel_status_labels = st.multiselect(
            'Release status', list(ds.STATUS_OPTIONS.keys()),
            default=list(ds.STATUS_OPTIONS.keys()), key='dsb_statuses',
            help='Open = accepting proposals now. Pre-release = announced, Q&A/submission '
                 'not yet open (earliest signal).')

    dsb_keyword = st.text_input(
        'Keyword (optional)',
        placeholder='e.g. hypersonics, autonomy, directed energy, counter-UAS',
        key='dsb_keyword',
    )
    dsb_max = st.number_input(
        'Max topics (0 = no cap)', min_value=0, value=2000, step=100, key='dsb_max')

    if st.button('🔍 Fetch from DoD SBIR', type='primary', key='dsb_fetch_btn'):
        statuses = [ds.STATUS_OPTIONS[lbl] for lbl in sel_status_labels]
        if not statuses:
            st.error('Select at least one release status.')
        elif not sel_programs:
            st.error('Select at least one program.')
        else:
            try:
                df_fetched, total, report = _search(
                    programs=sel_programs, statuses=statuses,
                    keyword=dsb_keyword.strip(), max_results=int(dsb_max),
                )
                if df_fetched.empty:
                    st.warning(
                        f'No topics matched (API reported {total:,} total for the selected '
                        'status). Try widening the program/status filters or clearing the keyword.'
                    )
                else:
                    st.caption(f'API total: **{total:,}** — fetched **{len(df_fetched):,}** topics.')
                    st.session_state.dsbir_raw_df        = df_fetched
                    st.session_state.dsbir_existing_keys = None
                    st.session_state.dsbir_detail_report = report
                    st.session_state.dsbir_custom_cols   = []
                    st.rerun()
            except requests.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else '?'
                st.error(f'DoD SBIR API HTTP {code}: {exc}')
            except Exception as exc:
                st.error(f'Fetch failed: {exc}')

    if st.session_state.dsbir_raw_df is None:
        st.stop()

    df_raw = st.session_state.dsbir_raw_df
    st.caption(f'**{len(df_raw):,}** topics loaded.')

    no_desc = int((df_raw['description'].str.strip() == '').sum())
    report  = st.session_state.get('dsbir_detail_report') or {}
    if no_desc > 0:
        failed   = int(report.get('failed', 0))
        statuses = report.get('statuses') or {}
        sample   = report.get('sample_error', '')
        if failed and failed >= no_desc * 0.5:
            # Most blanks are failed detail calls, not genuinely empty topics —
            # name the reason (e.g. http_403/http_429 = the detail endpoint refused
            # the burst) so it is fixable rather than a silent gap.
            st.warning(
                f'{no_desc} of {len(df_raw)} topics have no description because the detail '
                f'endpoint **failed** for {failed} of them (most common: `{sample}`). '
                'Re-fetch to retry, or reduce the batch. Saving now embeds those by title only.',
                icon='⚠️',
            )
            with st.expander('Detail-fetch diagnostics'):
                st.json({'retrieved': report.get('ok', 0), 'empty_200': report.get('empty', 0),
                         'failed': failed, 'statuses': statuses})
        else:
            st.info(
                f'{no_desc} topic(s) returned no detail text from the API — these embed by '
                'title only.', icon='ℹ️',
            )

    _preview_cols = ['title', 'agency', 'program', 'status', 'posted_date', 'close_date']
    st.dataframe(df_raw[_preview_cols].head(5), hide_index=True, use_container_width=True)


    # ── Section 2 · Review & Save ───────────────────────────────────────────────

    st.divider()
    st.subheader('2 · Review & Save')

    # Dedup against the existing DOD-SBIR store.
    if st.session_state.dsbir_existing_keys is None:
        with st.spinner('Checking existing records for duplicates…'):
            try:
                st.session_state.dsbir_existing_keys = _load_existing_keys(_get_storage_client())
            except Exception as e:
                st.warning(f'Could not load existing records for dedup check: {e}')
                st.session_state.dsbir_existing_keys = (set(), set())

    existing_ids, existing_titles = st.session_state.dsbir_existing_keys

    def _is_dup(row: pd.Series) -> bool:
        if str(row.get('notice_id', '')).strip() in existing_ids:
            return True
        if str(row.get('title', '')).strip().lower() in existing_titles:
            return True
        return False

    dup_mask = df_raw.apply(_is_dup, axis=1)
    dupes    = df_raw[dup_mask]
    new_rows = df_raw[~dup_mask].copy()

    m1, m2 = st.columns(2)
    m1.metric('New topics',          f'{len(new_rows):,}')
    m2.metric('Already in the store', f'{len(dupes):,}')

    if not dupes.empty:
        with st.expander(f'Skipped duplicates ({len(dupes)})', expanded=False):
            st.dataframe(
                dupes[['title', 'notice_id', 'agency']].reset_index(drop=True),
                hide_index=True, use_container_width=True,
            )

    if new_rows.empty:
        st.success('All fetched topics are already in the store — nothing new to save.')
        st.stop()

    st.caption(
        'Untick **Include** to drop a topic, or tick **Rolling** for a continuously-open '
        'topic (blank close dates are auto-flagged).'
    )
    _edit = new_rows[['title', 'agency', 'program', 'status', 'close_date']].reset_index(drop=True)
    _edit.insert(0, '_include', True)
    _edit['_is_rolling'] = False
    _ecfg = {
        '_include':    st.column_config.CheckboxColumn('Include', width='small'),
        'title':       st.column_config.TextColumn('Title',   width='large'),
        'agency':      st.column_config.TextColumn('Component', width='small'),
        'program':     st.column_config.TextColumn('Program', width='small'),
        'status':      st.column_config.TextColumn('Status',  width='small'),
        'close_date':  st.column_config.TextColumn('Close',   width='small'),
        '_is_rolling': st.column_config.CheckboxColumn('Rolling', width='small'),
    }
    edited = st.data_editor(
        _edit, hide_index=True, use_container_width=True, column_config=_ecfg,
        disabled=['title', 'agency', 'program', 'status', 'close_date'],
        key='dsbir_review_editor',
    )
    new_rows['_is_rolling'] = edited['_is_rolling'].values
    keep_mask = edited['_include'].values
    save_rows = new_rows[keep_mask].reset_index(drop=True)

    # Custom columns (campaign metadata tagged onto every saved topic).
    with st.expander('➕ Custom columns', expanded=bool(st.session_state.dsbir_custom_cols)):
        st.caption(
            'Tag every saved topic with campaign-specific metadata. Columns are saved to '
            'the parquet and available as optional fields in HubSpot import.'
        )
        for entry in list(st.session_state.dsbir_custom_cols):
            gc1, gc2, gc3 = st.columns([2, 4, 1])
            with gc1:
                st.text(entry['name'])
            with gc2:
                entry['value'] = st.text_input(
                    'Value', value=entry['value'],
                    key=f'dsbfill_{entry["name"]}', label_visibility='collapsed')
            with gc3:
                if st.button('✕', key=f'dsbrm_{entry["name"]}', use_container_width=True):
                    st.session_state.dsbir_custom_cols = [
                        e for e in st.session_state.dsbir_custom_cols if e['name'] != entry['name']]
                    st.rerun()

        if st.session_state.dsbir_custom_cols:
            st.divider()

        gna1, gna2, gna3 = st.columns([2, 4, 1])
        with gna1:
            new_gc_name = st.text_input(
                'Column name', placeholder='e.g. campaign_name',
                key='dsbir_new_gc_name', label_visibility='collapsed')
        with gna2:
            new_gc_val = st.text_input(
                'Value', placeholder='e.g. Spring 2026',
                key='dsbir_new_gc_val', label_visibility='collapsed')
        with gna3:
            if st.button('Add', key='dsbir_add_gc_btn', use_container_width=True):
                clean = new_gc_name.strip().replace(' ', '_').lower()
                existing_names = {e['name'] for e in st.session_state.dsbir_custom_cols}
                if clean and clean not in _RESERVED_COLS and clean not in existing_names:
                    st.session_state.dsbir_custom_cols.append(
                        {'name': clean, 'value': new_gc_val.strip()})
                    st.rerun()

    st.caption(
        f'**{len(save_rows)}** topic(s) will be summarized, embedded '
        f'(`text-embedding-ada-002`), newsletter-screened, and saved to '
        f'`{_DODSBIR_PREFIX}dod_sbir_{{date}}_{{hex}}.parquet`.'
    )

    if save_rows.empty:
        st.warning('Every topic is unticked — nothing to save.')
        st.stop()

    if st.button('💾 Embed & Save', type='primary', key='dsb_save_btn'):
        oai_key  = st.secrets['openai_api_key']
        anth_key = st.secrets['anthropic_api_key']
        extra_cols_dict = {e['name']: e['value'] for e in st.session_state.dsbir_custom_cols} or None
        try:
            path = _embed_and_save(save_rows, oai_key, anth_key, extra_cols=extra_cols_dict)
            st.success(f'Saved **{path}** — {len(save_rows)} topics ready for matching.')
            st.session_state.dsbir_raw_df        = None
            st.session_state.dsbir_existing_keys = None
            st.session_state.dsbir_detail_report = None
            st.session_state.dsbir_custom_cols   = []
            st.rerun()
        except Exception as e:
            st.error(f'Save failed: {e}')
