"""
Grants.gov Fetch
----------------
Query the Grants.gov public search API for federal grant opportunities.
Filter by keyword, posted-date range, opportunity status, funding
instrument, and agency.
Claude Haiku screens each row for relevance, then passing rows are
embedded and saved to GCS under data/all-topics/processed/GRANTS-GOV/.
No API key required — uses the public search2 endpoint.
"""

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import streamlit as st
from google.cloud import storage

import src.modules.newsletter as nl
import src.modules.ui_common as uc
from src.modules import grants_gov as gg

# ── GCS ────────────────────────────────────────────────────────────────────

_BUCKET        = 'cc-matcher-bucket-jeg-v1'
_GRANTS_PREFIX = 'data/all-topics/processed/GRANTS-GOV/'

# Search / screen / summarize / embed logic lives in src/modules/grants_gov.py,
# shared with grants-gov-job. This view keeps thin wrappers that add the
# st.progress UI, plus the rolling re-check (_refresh_rolling_grants) which is
# view-only.
_STATUS_OPTIONS     = gg.STATUS_OPTIONS
_INSTRUMENT_OPTIONS = gg.INSTRUMENT_OPTIONS
_SEARCH_URL         = gg.SEARCH_URL
_DETAIL_WORKERS     = gg.DETAIL_WORKERS

_GRANTS_RESERVED_COLS = frozenset({
    'topic_number', 'agency', 'title', 'description', 'open_date', 'close_date',
    'scraped_at', 'grant_summary', 'embeddings', 'award_ceiling', 'status',
    'sam_confidence', 'sam_reason',
    'is_rolling', 'last_verified_active', 'verify_status',
    *nl.COLUMNS,
})


# ── Storage client ─────────────────────────────────────────────────────────

def _get_storage_client() -> storage.Client:
    return uc.get_storage_client()


# ── Grants.gov helpers ─────────────────────────────────────────────────────
# items_to_df / fetch_description(s) / search / screen / summarize / embed now
# live in src.modules.grants_gov (gg); the wrappers further down add st.progress.

def _lookup_opp_status(number: str) -> tuple[bool, str | None]:
    """Current Grants.gov status for an opportunity number.

    Returns (ok, status): ok=False means the lookup itself failed (leave the row
    untouched); status=None means the number no longer appears in any status
    (treat as gone); otherwise the lowercased oppStatus. All four statuses are
    queried so a closed/archived opportunity is still returned.
    """
    try:
        r = requests.post(
            _SEARCH_URL,
            json={'keyword': number, 'rows': 25, 'startRecord': 0,
                  'oppStatuses': 'posted|forecasted|closed|archived'},
            timeout=20,
        )
        r.raise_for_status()
        hits = (r.json().get('data') or {}).get('oppHits') or []
        for h in hits:
            if str(h.get('number') or '').strip() == str(number).strip():
                return True, str(h.get('oppStatus') or '').strip().lower()
        return True, None
    except Exception:
        return False, None


def _refresh_rolling_grants(gcs: storage.Client) -> dict:
    """Re-check every stored rolling GRANTS-GOV opportunity against the live API.

    Still posted/forecasted → re-stamp active; closed, archived or gone →
    verify_status 'inactive'. Grants.gov returns a definitive status, so (unlike
    the scraped funding sources) there is no consecutive-miss guard. Rewrites only
    the parquets that changed.
    """
    today  = datetime.today().strftime('%Y-%m-%d')
    frames: dict = {}            # blob -> df
    numbers: set = set()
    for blob in gcs.list_blobs(_BUCKET, prefix=_GRANTS_PREFIX):
        if not blob.name.endswith('.parquet'):
            continue
        try:
            df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        except Exception:
            continue
        if 'is_rolling' not in df.columns or 'topic_number' not in df.columns:
            continue
        for col, default in (('last_verified_active', ''), ('verify_status', 'active')):
            if col not in df.columns:
                df[col] = default
        roll   = df['is_rolling'].fillna(False).astype(bool)
        active = df['verify_status'].fillna('').astype(str) != 'inactive'
        if (roll & active).any():
            frames[blob] = df
            numbers.update(df.loc[roll & active, 'topic_number'].astype(str))

    status_map: dict = {}
    if numbers:
        with ThreadPoolExecutor(max_workers=_DETAIL_WORKERS) as pool:
            futs = {pool.submit(_lookup_opp_status, n): n for n in numbers}
            for fut in as_completed(futs):
                n = futs[fut]
                ok, status = fut.result()
                if not ok:
                    continue                      # leave unchanged on lookup error
                status_map[n] = ('active' if status in ('posted', 'forecasted')
                                 else 'inactive')  # closed/archived/None → gone

    refreshed = expired = 0
    for blob, df in frames.items():
        roll   = df['is_rolling'].fillna(False).astype(bool)
        active = df['verify_status'].fillna('').astype(str) != 'inactive'
        dirty  = False
        for idx in df.index[roll & active]:
            res = status_map.get(str(df.at[idx, 'topic_number']))
            if res == 'active':
                df.at[idx, 'last_verified_active'] = today
                df.at[idx, 'verify_status']        = 'active'
                refreshed += 1
                dirty = True
            elif res == 'inactive':
                df.at[idx, 'verify_status'] = 'inactive'
                expired += 1
                dirty = True
        if dirty:
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            buf.seek(0)
            blob.upload_from_file(buf, content_type='application/octet-stream')
    return {'checked': len(numbers), 'refreshed': refreshed, 'expired': expired}


def _search_grants(keyword, statuses, instruments, agencies, date_from, date_to,
                   exclude_expired, max_results) -> tuple[pd.DataFrame, int]:
    """Thin wrapper over gg.search_grants adding a spinner + a fetch progress bar."""
    with st.spinner('Querying Grants.gov…'):
        bar = st.progress(0, text='Fetching descriptions…')
        df, total = gg.search_grants(
            keyword=keyword, statuses=statuses, instruments=instruments,
            agencies=agencies, date_from=date_from, date_to=date_to,
            exclude_expired=exclude_expired, max_results=max_results,
            progress=lambda d, t: bar.progress(d / t, text=f'Fetching descriptions… {d}/{t}'),
        )
        bar.empty()
    return df, total

def _run_screening(df: pd.DataFrame, anth_key: str) -> pd.DataFrame:
    """Thin wrapper over gg.run_screening adding a progress bar."""
    bar = st.progress(0, text='Screening rows…')
    out = gg.run_screening(
        df, anth_key,
        progress=lambda d, t: bar.progress(d / t, text=f'Screening rows… {d}/{t}'))
    bar.empty()
    return out


def _load_existing_keys(client: storage.Client) -> tuple[set[str], set[str]]:
    return gg.load_existing_keys(client)


def _embed_and_save(df: pd.DataFrame, oai_key: str, anth_key: str,
                    extra_cols: dict[str, str] | None = None) -> str:
    """Thin wrapper over gg.embed_and_save adding summary/embed/newsletter bars."""
    sum_bar = st.progress(0, text='Summarizing descriptions…')
    emb_bar = st.progress(0, text='Generating embeddings…')
    nl_bar  = st.progress(0, text='Newsletter screening…')
    try:
        path = gg.embed_and_save(
            df, oai_key, anth_key, _get_storage_client(), extra_cols=extra_cols,
            summary_progress=lambda d, t: sum_bar.progress(d / t, text=f'Summarizing… {d}/{t}'),
            embed_progress=lambda d, t: emb_bar.progress(d / t, text=f'Embedding {d}/{t}…'),
            newsletter_progress=lambda d, t: nl_bar.progress(d / t, text=f'Newsletter screening {d}/{t}'),
        )
    finally:
        sum_bar.empty(); emb_bar.empty(); nl_bar.empty()
    return path


# ── Page ──────────────────────────────────────────────────────────────────
# Body lives in render() so this module can be dispatched to by a parent
# page. Only one render() runs per script run, which is what keeps the
# st.stop() calls below correct.

def render():
    # ── Session state ──────────────────────────────────────────────────────────

    for _k in ['ggov_raw_df', 'ggov_screened_df', 'ggov_existing_keys']:
        if _k not in st.session_state:
            st.session_state[_k] = None
    if 'ggov_custom_cols' not in st.session_state:
        st.session_state.ggov_custom_cols = []


    # ── Page ───────────────────────────────────────────────────────────────────

    st.title('🏦 Grants.gov Fetch')
    st.caption(
        'Search Grants.gov for federal funding opportunities. '
        'Claude screens each result for R&D relevance, then passing rows are embedded '
        'and saved to the topic store. No API key required.'
    )

    with st.expander('🔁 Refresh rolling grants', expanded=False):
        st.caption(
            'Re-check every stored **rolling** GRANTS-GOV opportunity against the live '
            'API. Still posted/forecasted → re-confirmed active; closed, archived or '
            'gone → marked inactive and dropped from matching.'
        )
        if st.button('Refresh now', key='ggov_refresh_rolling'):
            with st.spinner('Checking rolling opportunities against Grants.gov…'):
                try:
                    rep = _refresh_rolling_grants(_get_storage_client())
                    st.success(
                        f"Checked {rep['checked']:,} rolling opportunit(ies): "
                        f"{rep['refreshed']:,} re-confirmed active, "
                        f"{rep['expired']:,} marked inactive."
                    )
                except Exception as e:
                    st.error(f'Refresh failed: {e}')

    # ── Section 1 · Fetch parameters ───────────────────────────────────────────

    st.subheader('1 · Fetch from Grants.gov')

    gg_exclude_expired = st.checkbox(
        'Exclude already-closed opportunities',
        value=True,
        key='gg_exclude_expired',
        help='Hide grants whose close date has already passed. Grants with no close date (common for forecasted) are always kept.',
    )

    gg_use_date_filter = st.checkbox('Filter by posted date', value=False, key='gg_use_date')
    gg_date_from: date | None = None
    gg_date_to:   date | None = None
    if gg_use_date_filter:
        col_df, col_dt = st.columns(2)
        with col_df:
            gg_date_from = st.date_input(
                'Posted from',
                value=date.today() - timedelta(days=7),
                key='gg_date_from',
            )
        with col_dt:
            gg_date_to = st.date_input(
                'Posted to',
                value=date.today(),
                key='gg_date_to',
            )

    gg_keyword = st.text_input(
        'Keyword (optional)',
        placeholder='e.g. SBIR, cybersecurity, artificial intelligence, biotech',
        key='gg_keyword',
    )

    col_s, col_f = st.columns(2)
    with col_s:
        selected_status_labels = st.multiselect(
            'Opportunity status',
            list(_STATUS_OPTIONS.keys()),
            default=['Posted', 'Forecasted'],
            key='gg_statuses',
        )
    with col_f:
        selected_instr_labels = st.multiselect(
            'Funding instruments',
            list(_INSTRUMENT_OPTIONS.keys()),
            default=['Grant', 'Cooperative Agreement'],
            key='gg_instruments',
        )

    col_a, col_m = st.columns(2)
    with col_a:
        gg_agencies_raw = st.text_input(
            'Agency codes (optional, comma-separated)',
            placeholder='e.g. HHS, NSF, DOD, NIH',
            key='gg_agencies',
        )
    with col_m:
        gg_max = st.number_input(
            'Max results (0 = no cap)',
            min_value=0,
            value=200,
            step=25,
            key='gg_max',
        )

    if st.button('🔍 Fetch from Grants.gov', type='primary', key='gg_fetch_btn'):
        statuses    = [_STATUS_OPTIONS[lbl]    for lbl in selected_status_labels]
        instruments = [_INSTRUMENT_OPTIONS[lbl] for lbl in selected_instr_labels]
        agencies    = [a.strip().upper() for a in gg_agencies_raw.split(',') if a.strip()]

        if not statuses:
            st.error('Select at least one opportunity status.')
        elif gg_use_date_filter and gg_date_from and gg_date_to and gg_date_from > gg_date_to:
            st.error('"Posted from" must be on or before "Posted to".')
        else:
            try:
                df_fetched, total = _search_grants(
                    keyword          = gg_keyword.strip(),
                    statuses         = statuses,
                    instruments      = instruments,
                    agencies         = agencies,
                    date_from        = gg_date_from,
                    date_to          = gg_date_to,
                    exclude_expired  = gg_exclude_expired,
                    max_results      = int(gg_max),
                )
                if df_fetched.empty:
                    st.warning(
                        f'No opportunities matched the filters (API returned {total:,} total records). '
                        'Try adjusting the posted-date range or status/instrument filters.'
                    )
                else:
                    st.caption(f'API total: **{total:,}** — after filters: **{len(df_fetched):,}** rows.')
                    st.session_state.ggov_raw_df        = df_fetched
                    st.session_state.ggov_screened_df   = None
                    st.session_state.ggov_existing_keys = None
                    st.session_state.ggov_custom_cols   = []
                    st.rerun()
            except requests.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else '?'
                st.error(f'Grants.gov API HTTP {code}: {exc}')
            except Exception as exc:
                st.error(f'Fetch failed: {exc}')

    if st.session_state.ggov_raw_df is None:
        st.stop()

    df_raw = st.session_state.ggov_raw_df
    st.caption(f'**{len(df_raw):,}** rows loaded.')

    no_desc = (df_raw['description'].str.strip() == '').sum()
    if no_desc > 0:
        st.info(
            f'{no_desc} row(s) have no description text from the API. '
            'These will be screened and embedded by title only.',
            icon='ℹ️',
        )

    _preview_cols = ['title', 'agency', 'posted_date', 'close_date', 'award_ceiling', 'description']
    st.dataframe(df_raw[_preview_cols].head(5), hide_index=True, use_container_width=True)


    # ── Section 2 · Screen with Claude ─────────────────────────────────────────

    st.divider()
    st.subheader('2 · Screen with Claude')

    n_rows   = len(df_raw)
    est_mins = max(1, n_rows // 60)
    screened = st.session_state.ggov_screened_df

    if screened is None:
        st.caption(
            f'Claude Haiku will screen **{n_rows:,}** rows for relevance to R&D small businesses. '
            f'Estimated time: ~{est_mins} min at {gg.SCREEN_WORKERS} concurrent workers.'
        )
        if st.button('⚡ Run Screening', type='primary', key='gg_screen_btn'):
            anth_key = st.secrets['anthropic_api_key']
            try:
                st.session_state.ggov_screened_df = _run_screening(df_raw, anth_key)
                st.rerun()
            except Exception as e:
                st.error(f'Screening failed: {e}')
        st.stop()

    passing  = screened[screened['_import'] == True].copy()
    failing  = screened[screened['_import'] == False].copy()
    pass_pct = len(passing) / len(screened) * 100 if len(screened) > 0 else 0

    m1, m2, m3 = st.columns(3)
    m1.metric('Total rows',   f'{len(screened):,}')
    m2.metric('Passing',      f'{len(passing):,}  ({pass_pct:.0f}%)')
    m3.metric('Filtered out', f'{len(failing):,}')

    if st.button('↺ Re-run screening', key='gg_rescreen_btn'):
        st.session_state.ggov_screened_df = None
        st.rerun()

    _disp_cols = ['title', 'agency', '_confidence', '_reason']
    _disp_cfg  = {
        'title':        st.column_config.TextColumn('Title',      width='medium'),
        'agency':       st.column_config.TextColumn('Agency',     width='small'),
        '_confidence':  st.column_config.TextColumn('Confidence', width='small'),
        '_reason':      st.column_config.TextColumn('Reason',     width='large'),
    }

    with st.expander(f'✅ Passing ({len(passing)})', expanded=True):
        if passing.empty:
            st.info('No rows passed screening.')
        elif '_is_rolling' in passing.columns:
            # Editable Rolling flag so a reviewer can correct the auto-detection.
            _edit_cols = ['title', 'agency', '_is_rolling', '_confidence', '_reason']
            _ecfg = dict(_disp_cfg)
            _ecfg['_is_rolling'] = st.column_config.CheckboxColumn(
                'Rolling', width='small',
                help='Continuously-open / rolling-deadline opportunity. '
                     'Tick to flag one the screener missed.')
            _edited = st.data_editor(
                passing[_edit_cols].reset_index(drop=True),
                hide_index=True,
                use_container_width=True,
                column_config=_ecfg,
                disabled=[c for c in _edit_cols if c != '_is_rolling'],
                key='ggov_passing_editor',
            )
            passing = passing.copy()
            passing['_is_rolling'] = _edited['_is_rolling'].values
        else:
            st.dataframe(
                passing[_disp_cols].reset_index(drop=True),
                hide_index=True,
                use_container_width=True,
                column_config=_disp_cfg,
            )

    with st.expander(f'❌ Filtered out ({len(failing)})', expanded=False):
        if failing.empty:
            st.info('Nothing was filtered out.')
        else:
            st.dataframe(
                failing[_disp_cols].reset_index(drop=True),
                hide_index=True,
                use_container_width=True,
                column_config=_disp_cfg,
            )


    # ── Section 3 · Save ──────────────────────────────────────────────────────

    st.divider()
    st.subheader('3 · Save to topic store')

    if passing.empty:
        st.warning('No passing rows to save — nothing to embed.')
        st.stop()

    if st.session_state.ggov_existing_keys is None:
        with st.spinner('Checking existing records for duplicates…'):
            try:
                existing_ids, existing_titles = _load_existing_keys(_get_storage_client())
                st.session_state.ggov_existing_keys = (existing_ids, existing_titles)
            except Exception as e:
                st.warning(f'Could not load existing records for dedup check: {e}')
                st.session_state.ggov_existing_keys = (set(), set())

    existing_ids, existing_titles = st.session_state.ggov_existing_keys


    def _is_dup(row: pd.Series) -> bool:
        if str(row.get('notice_id', '')).strip() in existing_ids:
            return True
        if str(row.get('title', '')).strip().lower() in existing_titles:
            return True
        return False


    dup_mask = passing.apply(_is_dup, axis=1)
    dupes    = passing[dup_mask]
    new_rows = passing[~dup_mask]

    if not dupes.empty:
        st.info(f'**{len(dupes)}** row(s) skipped — notice ID or title already exists in the store.', icon='ℹ️')
        with st.expander(f'Skipped duplicates ({len(dupes)})', expanded=False):
            st.dataframe(
                dupes[['title', 'notice_id']].reset_index(drop=True),
                hide_index=True,
                use_container_width=True,
            )

    if new_rows.empty:
        st.success('All passing rows are already in the store — nothing new to save.')
        st.stop()

    with st.expander('➕ Custom columns', expanded=bool(st.session_state.ggov_custom_cols)):
        st.caption(
            'Add extra columns to tag every saved topic with campaign-specific metadata. '
            'Columns are saved to the parquet and available as optional fields in HubSpot import.'
        )

        for entry in list(st.session_state.ggov_custom_cols):
            gc1, gc2, gc3 = st.columns([2, 4, 1])
            with gc1:
                st.text(entry['name'])
            with gc2:
                entry['value'] = st.text_input(
                    'Value', value=entry['value'],
                    key=f'gcfill_{entry["name"]}', label_visibility='collapsed',
                )
            with gc3:
                if st.button('✕', key=f'gcrm_{entry["name"]}', use_container_width=True):
                    st.session_state.ggov_custom_cols = [
                        e for e in st.session_state.ggov_custom_cols if e['name'] != entry['name']
                    ]
                    st.rerun()

        if st.session_state.ggov_custom_cols:
            st.divider()

        gna1, gna2, gna3 = st.columns([2, 4, 1])
        with gna1:
            new_gc_name = st.text_input(
                'Column name', placeholder='e.g. campaign_name',
                key='ggov_new_gc_name', label_visibility='collapsed',
            )
        with gna2:
            new_gc_val = st.text_input(
                'Value', placeholder='e.g. Spring 2026',
                key='ggov_new_gc_val', label_visibility='collapsed',
            )
        with gna3:
            if st.button('Add', key='ggov_add_gc_btn', use_container_width=True):
                clean = new_gc_name.strip().replace(' ', '_').lower()
                existing_names = {e['name'] for e in st.session_state.ggov_custom_cols}
                if (
                    clean
                    and clean not in _GRANTS_RESERVED_COLS
                    and clean not in existing_names
                ):
                    st.session_state.ggov_custom_cols.append({'name': clean, 'value': new_gc_val.strip()})
                    st.rerun()

    st.caption(
        f'**{len(new_rows)}** new rows will be summarized, embedded (`text-embedding-ada-002`), and saved to '
        f'`{_GRANTS_PREFIX}grants_gov_{{date}}_{{hex}}.parquet`.'
    )

    if st.button('💾 Embed & Save', type='primary', key='gg_save_btn'):
        oai_key  = st.secrets['openai_api_key']
        anth_key = st.secrets['anthropic_api_key']
        extra_cols_dict = {e['name']: e['value'] for e in st.session_state.ggov_custom_cols} or None
        try:
            path = _embed_and_save(
                new_rows.reset_index(drop=True),
                oai_key,
                anth_key,
                extra_cols=extra_cols_dict,
            )
            st.success(f'Saved **{path}** — {len(new_rows)} topics ready for matching.')
            st.session_state.ggov_raw_df        = None
            st.session_state.ggov_screened_df   = None
            st.session_state.ggov_existing_keys = None
            st.session_state.ggov_custom_cols   = []
            st.rerun()
        except Exception as e:
            st.error(f'Save failed: {e}')
