"""
Grant Search
------------
Cosine-similarity search across grant topics stored in GCS.

Select agencies, apply keyword filters, then search one of two ways:

  Technology description — embed a paragraph and rank topics against it.
  Capability profile     — score ONE company's multi-aspect profile against the
                           selected topics, exactly as Bulk Aspect Match does
                           for the whole directory.

The profile can come from the client store (data/client-profiles/profiles.parquet)
or be built here and now from pasted notes — a call, a deck, a capability
statement, anything the team has about a company we may hold no records for.
A profile built from notes is a normal profile: same prompt, same aspect and
market structure, same embeddings, and it can be saved into the client profile
store so Client Profiles, Bulk Aspect Match and HubSpot Import all see it.

This replaces the old "multi-aspect search", which decomposed the query text
into 2-4 required dimensions and demanded every one of them clear the
threshold. That answered a different question — it narrowed one description —
whereas a company's aspects are alternative capabilities, any one of which
matching is a real hit. The scoring here is the shared implementation in
src/modules/aspect_matching.py, so a single-company search and a directory-wide
run rank the same pair identically.
"""

import io
import traceback
from datetime import date, datetime, timedelta

import anthropic
import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account

import src.modules.aspect_matching as am
import src.modules.anthropic_utils as au
import src.modules.aspect_profile as ap
import src.modules.pools as pl
import src.modules.ui_common as uc
from src.modules.Embedding.text_embedder import TextProcessor
from src.modules.grant_utils import normalize_grant_columns

# ── GCS ────────────────────────────────────────────────────────────────────

_BUCKET        = 'cc-matcher-bucket-jeg-v1'
_TOPICS_PREFIX = 'data/all-topics/processed/'
# Past awards live outside processed/ so nothing enumerating agencies can pick
# them up by accident. Searching them here is opt-in, per search.
_AWARDS_PREFIX = 'data/all-topics/awards/'

# Agency folders that exist in the store but are off by default — niche sources
# the team opts into rather than searches every time.
_DEFAULT_OFF_AGENCIES = {'DEFENSE PATENT HOLIDAY', 'TECHCONNECT'}

# Search modes
_MODE_DESC    = '📝 Technology description'
_MODE_PROFILE = '🧬 Capability profile'

# Where a capability profile comes from
_SRC_CLIENT = '👤 Existing client profile'
_SRC_NOTES  = '📋 Paste notes or source material'

# Match scope within one profile
_SCOPE_ALL    = 'All aspects (whole company)'
_SCOPE_MARKET = 'By market'

# Enough pasted text to be worth a Claude call. Below this the model has
# nothing to split into aspects and will pad, which is the one failure mode
# the aspect prompt is written to avoid.
_MIN_NOTES_CHARS = 200


def _get_storage_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)


def _list_agencies() -> list[str]:
    try:
        client = _get_storage_client()
        blobs = client.list_blobs(_BUCKET, prefix=_TOPICS_PREFIX, delimiter='/')
        list(blobs)
        return sorted(
            p.replace(_TOPICS_PREFIX, '').strip('/')
            for p in blobs.prefixes
        )
    except Exception as e:
        st.error(f'Failed to list agencies: {e}')
        return []


def _list_award_sources() -> list[str]:
    try:
        client = _get_storage_client()
        blobs = client.list_blobs(_BUCKET, prefix=_AWARDS_PREFIX, delimiter='/')
        list(blobs)
        return sorted(
            p.replace(_AWARDS_PREFIX, '').strip('/')
            for p in blobs.prefixes
        )
    except Exception:
        return []


def _load_topics(agencies: list[str], award_sources: list[str] | None = None) -> pd.DataFrame:
    client = _get_storage_client()
    frames = []
    for agency in agencies:
        prefix = f'{_TOPICS_PREFIX}{agency}/'
        for blob in client.list_blobs(_BUCKET, prefix=prefix):
            if blob.name.endswith('.parquet'):
                df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
                df['broad_agency'] = agency
                df['record_kind']  = 'solicitation'
                frames.append(df)
    for src in (award_sources or []):
        prefix = f'{_AWARDS_PREFIX}{src}/'
        for blob in client.list_blobs(_BUCKET, prefix=prefix):
            if blob.name.endswith('.parquet'):
                df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
                df['broad_agency'] = f'{src}-AWARDS'
                df['record_kind']  = 'award'
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    topics = pd.concat(frames, ignore_index=True)
    # Notices marked archived by the SAM.gov revision check are no longer live
    if 'sam_status' in topics.columns:
        topics = topics[topics['sam_status'].fillna('').astype(str) != 'archived'].reset_index(drop=True)
    return topics


# Topic dates arrive as strings in mixed formats (ISO from Topic Importer,
# mm/dd/yyyy from SAM.gov, sometimes with a time suffix) — try each.
_DATE_FORMATS = ('%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%m-%d-%Y', '%b %d, %Y')


def _parse_date_str(value) -> date | None:
    s = str(value).strip()
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        for candidate in (s, s[:10]):
            try:
                return datetime.strptime(candidate, fmt).date()
            except ValueError:
                pass
    return None


def _filter_is_active(f: dict) -> bool:
    if not f.get('column'):
        return False
    if f.get('type') == 'date_range':
        return bool(f.get('date_from') and f.get('date_to'))
    return bool(f.get('keyword', '').strip())


def _filter_mask(df: pd.DataFrame, f: dict) -> pd.Series:
    if f.get('type') == 'date_range':
        d_from, d_to = f['date_from'], f['date_to']
        keep_undated = bool(f.get('include_undated'))

        def _in_range(v) -> bool:
            d = _parse_date_str(v)
            if d is None:
                # Blank, 'Rolling', 'TBD', or an unparseable date string.
                return keep_undated
            return d_from <= d <= d_to

        return df[f['column']].map(_in_range)
    return df[f['column']].astype(str).str.lower().str.contains(
        f['keyword'].lower(), na=False
    )


def _apply_filters(df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    active = [f for f in filters if _filter_is_active(f)]
    if not active:
        return df
    mask = _filter_mask(df, active[0])
    for f in active[1:]:
        m = _filter_mask(df, f)
        mask = (mask & m) if f['operator'] == 'AND' else (mask | m)
    return df[mask]


def _similarity_search(df: pd.DataFrame, query_embedding: list[float], threshold: float) -> pd.DataFrame:
    query_vec = np.array(query_embedding)

    def score(emb):
        try:
            return float(np.dot(np.array(emb), query_vec))
        except Exception:
            return 0.0

    result = df.copy()
    result['similarity_score'] = result['embeddings'].apply(score)
    return (
        result[result['similarity_score'] >= threshold]
        .sort_values('similarity_score', ascending=False)
        .reset_index(drop=True)
    )


# ── Building a profile from pasted notes ───────────────────────────────────

def _claude_json(anth: anthropic.Anthropic, model: str, system: str, user_msg: str, parse):
    """One Claude call with one strict-JSON retry, parsed by `parse`.

    Same contract as client_profile_job._claude_json — a truncated response is
    a hard error rather than a half-parsed profile. No temperature: the
    anthropic 1.x SDK removed it, and passing it raises a local TypeError."""
    last_err = None
    for attempt in range(2):
        content = user_msg if attempt == 0 else (
            user_msg + '\n\nYour previous response was not valid JSON. '
                       'Return ONLY the valid JSON object.'
        )
        resp = anth.messages.create(
            model=model,
            max_tokens=6000,
            system=system,
            messages=[{'role': 'user', 'content': content}],
        )
        if resp.stop_reason == 'max_tokens':
            raise ValueError('Claude hit the output token limit')
        try:
            return parse(au.response_text(resp))
        except ValueError as e:
            last_err = e
    raise ValueError(f'invalid response twice: {last_err}')


def _build_profile_from_notes(
    *,
    company_name: str,
    website: str,
    state: str,
    notes: str,
    target_aspects: int,
    max_markets: int,
    assess_defense: bool,
    assess_unexplored: bool,
    max_unexplored: int,
    model: str,
) -> tuple[dict, list[str]]:
    """(profile record, warnings). Mirrors client_profile_job's per-client build
    so a profile made here is indistinguishable from one the job wrote — same
    prompt, same merges, same normalisation, same record shape."""
    warnings: list[str] = []
    texts = ap.notes_source_texts(notes)
    if not texts:
        raise ValueError('nothing to profile')

    anth = anthropic.Anthropic(api_key=st.secrets['anthropic_api_key'])
    tp   = TextProcessor(api_key=st.secrets['openai_api_key'])

    parsed = _claude_json(
        anth, model,
        ap.build_aspect_system(target_aspects, max_markets, assess_defense),
        ap.build_aspect_user_message(
            {'company_name': company_name, 'website': website, 'state': state}, texts
        ),
        ap.parse_aspect_response,
    )
    summary = parsed['profile_summary']
    aspects = parsed['aspects']
    markets = parsed['markets']

    vectors = [tp.get_embedding(ap.aspect_embed_text(a)) for a in aspects]
    # Fold near-identical aspects together BEFORE market membership is re-derived,
    # so no market is left pointing at an aspect that was merged away.
    aspects, vectors, merges = ap.merge_similar_aspects(
        aspects, vectors, ap.ASPECT_MERGE_THRESHOLD
    )
    warnings += [f'Merged near-identical aspects: {m}' for m in merges]

    market_vectors = [tp.get_embedding(ap.market_embed_text(m)) for m in markets]
    if markets:
        aspects, markets, market_vectors = ap.merge_similar_markets(
            aspects, markets, market_vectors, ap.MARKET_MERGE_THRESHOLD
        )
        by_name = {m['market']: v for m, v in zip(markets, market_vectors)}
        aspects, markets = ap.normalize_markets(aspects, markets, max_markets)
        market_vectors = [by_name[m['market']] for m in markets]

    # Pass 2 — markets the company does NOT serve. Additive and separately
    # guarded: a failure here must never discard the pass-1 work above, which
    # cost a Claude call and every embedding.
    unexplored, unexplored_vectors = [], []
    if assess_unexplored:
        try:
            unexplored = _claude_json(
                anth, model,
                ap.build_unexplored_system(max_unexplored),
                ap.build_unexplored_user_message(
                    {'company_name': company_name, 'website': website, 'state': state},
                    summary, aspects, markets,
                    # stated_intentions() reads notable_updates off client rows;
                    # pasted notes have no such structure, so there are none.
                    [],
                ),
                lambda raw: ap.parse_unexplored_response(
                    raw, markets, aspects, max_unexplored
                ),
            )
            unexplored_vectors = [
                tp.get_embedding(ap.market_embed_text(m)) for m in unexplored
            ]
        except Exception as e:
            unexplored, unexplored_vectors = [], []
            warnings.append(
                f'Unexplored-market pass failed ({type(e).__name__}: {e}) — the '
                'profile is complete apart from its unexplored markets.'
            )

    record = ap.build_profile_record(
        company_key     = ap.company_key(
            {'company_name': company_name, 'companyWebsite': website}
        ),
        company_name    = company_name,
        website         = website,
        profile_summary = summary,
        aspects         = aspects,
        vectors         = vectors,
        sources_used    = list(texts.keys()),
        # Over the pasted material only — it is all this profile was built from,
        # which is exactly what a later staleness check should compare against.
        fingerprint     = ap.source_fingerprint(texts),
        model           = model,
        markets         = markets,
        market_vectors  = market_vectors,
        unexplored         = unexplored,
        unexplored_vectors = unexplored_vectors,
        dod_assessment  = parsed['dod_assessment'],
    )
    return record, warnings


# ── Profile display ────────────────────────────────────────────────────────

def _profile_row(record) -> pd.Series:
    """One profile as a row the shared matcher can read. A record built here and
    a row loaded from profiles.parquet are the same shape, so both paths below
    converge on this."""
    return record if isinstance(record, pd.Series) else pd.Series(record)


def _render_profile(prof: pd.Series) -> None:
    aspects    = ap.profile_aspects(prof)
    markets    = ap.profile_markets(prof)
    unexplored = ap.profile_unexplored(prof)

    if str(prof.get('profile_summary') or '').strip():
        st.markdown(f'**{prof.get("company_name") or "—"}** — {prof["profile_summary"]}')

    c1, c2, c3 = st.columns(3)
    c1.metric('Aspects', len(aspects))
    c2.metric('Markets', len(markets))
    c3.metric('Unexplored markets', len(unexplored))

    with st.expander(f'Aspects ({len(aspects)})', expanded=False):
        if aspects:
            st.dataframe(
                pd.DataFrame([{
                    'Label':    a.get('label', ''),
                    'Kind':     a.get('kind', ''),
                    'Markets':  ', '.join(a.get('markets') or []) if isinstance(a.get('markets'), list)
                                else str(a.get('markets') or ''),
                    'Text':     a.get('text', ''),
                    'Keywords': a.get('keywords', ''),
                } for a in aspects]),
                width='stretch', hide_index=True,
            )
        else:
            st.caption('No aspects — this profile cannot be matched.')

    if markets:
        with st.expander(f'Markets ({len(markets)})', expanded=False):
            st.dataframe(
                pd.DataFrame([{
                    'Tier':      ap.tier_ordinal(ap.market_tier_rank(m)),
                    'Market':    m.get('market', ''),
                    'Subtitle':  m.get('subtitle', ''),
                    'Narrative': m.get('narrative', ''),
                } for m in markets]),
                width='stretch', hide_index=True,
            )
    if unexplored:
        with st.expander(f'Unexplored markets ({len(unexplored)})', expanded=False):
            st.caption(
                'Markets this company does **not** serve, inferred from its aspects. '
                'Matched by a different question — "could they plausibly extend into '
                'this?" — and every result row says which kind it came from.'
            )
            st.dataframe(
                pd.DataFrame([{
                    'Tier':      ap.tier_ordinal(ap.market_tier_rank(m)),
                    'Market':    m.get('market', ''),
                    'Subtitle':  m.get('subtitle', ''),
                    'Narrative': m.get('narrative', ''),
                    'Gap':       m.get('rationale', ''),
                } for m in unexplored]),
                width='stretch', hide_index=True,
            )
    if str(prof.get('dod_assessment') or '').strip():
        st.caption(f'**Defense assessment:** {prof["dod_assessment"]}')


def _unit_options(prof: pd.Series) -> dict[str, am.Unit]:
    """{display label: Unit} for every market of one profile, confirmed and
    unexplored. The kind travels with the index because it selects which flat
    embedding block the index points into."""
    out: dict[str, am.Unit] = {}
    for kind, market, i in am.unit_markets(prof, [am.ROW_CONFIRMED, am.ROW_UNEXPLORED]):
        label = ap.market_label(market)
        if kind == am.ROW_UNEXPLORED:
            label += ' · unexplored'
        out[label] = am.Unit(prof, market, i, kind)
    return out


# ── Session state ──────────────────────────────────────────────────────────

for _k in [
    'gs_topics_df', 'gs_results_df',        # description-mode search
    'gs_profiles',                           # the client profile store
    'gs_profile', 'gs_profile_origin',       # the profile being matched
    'gs_profile_warnings', 'gs_profile_saved',
    'gs_match_results', 'gs_match_meta',     # profile-mode search
]:
    if _k not in st.session_state:
        st.session_state[_k] = None
if 'gs_filters' not in st.session_state:
    st.session_state.gs_filters = [{'column': None, 'type': 'keyword', 'keyword': '', 'operator': 'AND'}]


# ── Page ───────────────────────────────────────────────────────────────────

st.title('🔍 Grant Search')
st.caption(
    'Search grant topics by cosine similarity — against a description you type, '
    'or against one company’s capability profile.'
)

# ── Section 1 · Agency selection ───────────────────────────────────────────

st.subheader('1 · Select agencies')

agencies = _list_agencies()
if not agencies:
    st.warning('No agencies found in GCS.')
    st.stop()

# Quick-select buttons run before the checkboxes are created, so writing their
# session-state keys here is what sets the widget values for this run.
qs1, qs2, _qs3 = st.columns([1, 1, 6])
if qs1.button('Select all', key='gs_agency_all'):
    for agency in agencies:
        st.session_state[f'gs_agency_{agency}'] = True
    st.rerun()
if qs2.button('Deselect all', key='gs_agency_none'):
    for agency in agencies:
        st.session_state[f'gs_agency_{agency}'] = False
    st.rerun()

cols = st.columns(min(len(agencies), 6))
selected = [
    agency for i, agency in enumerate(agencies)
    if cols[i % len(cols)].checkbox(
        agency, value=agency not in _DEFAULT_OFF_AGENCIES, key=f'gs_agency_{agency}'
    )
]

_award_sources = _list_award_sources()
_selected_awards: list[str] = []
if _award_sources:
    if st.checkbox(
        'Include past awards (already-awarded contracts)',
        value=False,
        key='gs_include_awards',
        help=(
            'Adds completed award notices to the pool — useful for asking "who has won '
            'work like this?". They are stored separately from open solicitations and '
            'are never included unless you tick this. Results are tagged in a '
            '`record_kind` column so an award is never mistaken for something to bid on.'
        ),
    ):
        _selected_awards = _award_sources
        st.caption(f'Award sources included: `{", ".join(_award_sources)}`')

if st.button('Load Topics', type='primary', disabled=not selected):
    with st.spinner(f'Loading topics from {len(selected)} agenc{"y" if len(selected) == 1 else "ies"}…'):
        df = _load_topics(selected, _selected_awards)
    if df.empty:
        st.warning('No topics found for the selected agencies.')
    else:
        st.session_state.gs_topics_df     = normalize_grant_columns(df)
        st.session_state.gs_results_df    = None
        st.session_state.gs_match_results = None
        st.session_state.gs_match_meta    = None
        _n_awards = int((df.get('record_kind') == 'award').sum()) if 'record_kind' in df.columns else 0
        st.success(
            f'Loaded **{len(df) - _n_awards:,}** open topics'
            + (f' and **{_n_awards:,}** past awards.' if _n_awards else '.')
        )

if st.session_state.gs_topics_df is None:
    st.stop()

df = st.session_state.gs_topics_df

# ── Section 2 · Filters + preview ─────────────────────────────────────────

st.divider()
st.subheader('2 · Filter topics')

filterable_cols = [c for c in df.columns if c != 'embeddings']

for f in st.session_state.gs_filters:
    if f['column'] not in filterable_cols:
        f['column'] = filterable_cols[0] if filterable_cols else None

for i, f in enumerate(st.session_state.gs_filters):
    if i == 0:
        col_sel, mode_col, val_input, remove_col = st.columns([2, 1.4, 3, 0.5])
    else:
        op_col, col_sel, mode_col, val_input, remove_col = st.columns([1, 2, 1.4, 3, 0.5])
        f['operator'] = op_col.radio(
            'op', ['AND', 'OR'], index=0 if f['operator'] == 'AND' else 1,
            key=f'gs_op_{i}', horizontal=True, label_visibility='collapsed'
        )

    f['column'] = col_sel.selectbox(
        'Column', filterable_cols,
        index=filterable_cols.index(f['column']) if f['column'] in filterable_cols else 0,
        key=f'gs_col_{i}', label_visibility='collapsed'
    )
    mode_label = mode_col.selectbox(
        'Filter type', ['Keyword', 'Date range'],
        index=1 if f.get('type') == 'date_range' else 0,
        key=f'gs_type_{i}', label_visibility='collapsed'
    )
    f['type'] = 'date_range' if mode_label == 'Date range' else 'keyword'
    if f['type'] == 'date_range':
        picked = val_input.date_input(
            'Date range',
            value=(
                f.get('date_from') or date.today() - timedelta(days=30),
                f.get('date_to') or date.today(),
            ),
            key=f'gs_dr_{i}', label_visibility='collapsed',
        )
        if isinstance(picked, tuple) and len(picked) == 2:
            f['date_from'], f['date_to'] = picked
        elif isinstance(picked, tuple) and len(picked) == 1:
            # Mid-selection: only the start date is chosen so far.
            f['date_from'], f['date_to'] = picked[0], None
        f['include_undated'] = val_input.checkbox(
            'Include rows with no date',
            value=bool(f.get('include_undated')),
            key=f'gs_und_{i}',
            help='Also keep rows whose date cell is blank or unreadable '
                 "(e.g. 'Rolling', 'TBD', 'Continuous'), which a date range "
                 'would otherwise drop.',
        )
    else:
        f['keyword'] = val_input.text_input(
            'Keyword', value=f.get('keyword', ''),
            placeholder=f'Filter by {f["column"]}…',
            key=f'gs_kw_{i}', label_visibility='collapsed'
        )
    if remove_col.button('✕', key=f'gs_rm_{i}', disabled=len(st.session_state.gs_filters) == 1):
        st.session_state.gs_filters.pop(i)
        st.rerun()

if st.button('+ Add filter'):
    st.session_state.gs_filters.append(
        {'column': filterable_cols[0], 'type': 'keyword', 'keyword': '', 'operator': 'AND'}
    )
    st.rerun()

filtered = _apply_filters(df, st.session_state.gs_filters)

display_cols = [c for c in filtered.columns if c != 'embeddings']
st.caption(f'**{len(filtered):,}** topics match — showing first 50')
st.dataframe(
    filtered[display_cols].head(50),
    width='stretch',
    hide_index=True,
)

# ── Section 3 · Search ─────────────────────────────────────────────────────

st.divider()
st.subheader('3 · Search')

search_mode = st.radio(
    'Search by', [_MODE_DESC, _MODE_PROFILE],
    horizontal=True, key='gs_search_mode',
    help=(
        'A description is embedded as one vector and ranked directly. A '
        'capability profile splits one company into several independently '
        'embedded aspects, so a topic that matches any one capability is found '
        '— which a single blended vector averages away.'
    ),
)

# ═══ Mode A · Technology description ═══════════════════════════════════════

if search_mode == _MODE_DESC:
    threshold = st.slider('Similarity threshold', 0.0, 1.0, 0.75, 0.01, key='gs_desc_threshold')

    tech_text = st.text_area(
        'Technology description',
        height=120,
        key='gs_tech_text',
        placeholder='Describe the technology or capability you want to match against grant topics…',
    )

    if st.button('🔍 Search', type='primary', disabled=not tech_text.strip()):
        tp = TextProcessor(api_key=st.secrets['openai_api_key'])
        with st.spinner('Generating embedding…'):
            query_embedding = tp.get_embedding(tech_text.strip())
        with st.spinner('Scoring topics…'):
            st.session_state.gs_results_df = _similarity_search(
                filtered, query_embedding, threshold
            )

    if st.session_state.gs_results_df is not None:
        results = st.session_state.gs_results_df
        if results.empty:
            st.warning(
                f'No topics above **{threshold}** similarity threshold. Try lowering it, '
                'or search by capability profile instead — a profile scores each '
                'capability separately rather than blending them into one vector.'
            )
        else:
            st.success(f'**{len(results):,}** topics matched.')

            primary_cols = ['similarity_score']
            # Surface award/solicitation up front rather than leaving it buried among
            # the trailing columns — a past award read as an open opportunity is the
            # one mistake this whole separation exists to prevent.
            if 'record_kind' in results.columns and (results['record_kind'] == 'award').any():
                primary_cols = ['record_kind'] + primary_cols
            other_cols  = [c for c in results.columns
                           if c not in primary_cols and c != 'embeddings']
            result_cols = primary_cols + other_cols

            col_cfg: dict = {
                'similarity_score': st.column_config.NumberColumn('Score', format='%.4f'),
            }
            if 'record_kind' in result_cols:
                col_cfg['record_kind'] = st.column_config.TextColumn('Kind', width='small')

            st.dataframe(
                results[result_cols],
                width='stretch', hide_index=True, column_config=col_cfg,
            )
            st.download_button(
                '⬇ Download CSV',
                results[result_cols].to_csv(index=False).encode('utf-8'),
                file_name=f'grant_search_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv',
                mime='text/csv',
            )

# ═══ Mode B · Capability profile ═══════════════════════════════════════════

else:
    profile_source = st.radio(
        'Profile source', [_SRC_CLIENT, _SRC_NOTES],
        horizontal=True, key='gs_profile_source',
    )

    # ── 3a · Pick or build the profile ─────────────────────────────────────

    if profile_source == _SRC_CLIENT:
        st.markdown('**Select a stored capability profile**')

        # Read-only pick, so both stores can be searched at once — the profile
        # picked carries its own pool, and nothing here writes back to it.
        pick_pools = uc.pool_scope_selector(
            'gs_pick_pools', label='Search which store',
            clears=('gs_profiles',),
            help='Clients, targeted prospects, or both.',
        )

        load_col, refresh_col = st.columns([4, 1])
        if st.session_state.gs_profiles is None:
            with st.spinner('Loading capability profiles…'):
                try:
                    _loaded = [
                        ap.load_profiles(_get_storage_client(), pool=p)
                        for p in pick_pools
                    ]
                    _loaded = [f for f in _loaded if not f.empty]
                    st.session_state.gs_profiles = (
                        pd.concat(_loaded, ignore_index=True) if _loaded
                        else ap.empty_profiles_df()
                    )
                except Exception as e:
                    st.error(
                        'Could not load '
                        + ', '.join(ap.profiles_blob(p) for p in pick_pools)
                        + f': {e}'
                    )
                    st.session_state.gs_profiles = ap.empty_profiles_df()
        if refresh_col.button('↻ Reload', key='gs_profiles_reload'):
            st.session_state.gs_profiles = None
            st.rerun()

        profiles: pd.DataFrame = st.session_state.gs_profiles
        if profiles.empty:
            st.info(
                'No profiles in '
                + ' or '.join(pl.label(p) for p in pick_pools)
                + ' yet. Build them in **Capability Profiles**, or paste notes '
                'below to profile a company ad hoc.'
            )
        else:
            query = load_col.text_input(
                'Search companies', key='gs_profile_query',
                placeholder='Type part of a company name, website or market…',
            ).strip().lower()

            haystack = (
                profiles['company_name'].fillna('').astype(str) + ' ' +
                profiles['companyWebsite'].fillna('').astype(str) + ' ' +
                profiles['market_labels'].fillna('').astype(str) + ' ' +
                profiles['aspect_labels'].fillna('').astype(str)
            ).str.lower()
            matches = profiles[haystack.str.contains(query, na=False)] if query else profiles

            if matches.empty:
                st.warning(f'No profile matches “{query}”.')
            else:
                labels = {
                    f'{pl.pool(r.get("pool"))["icon"]} {r["company_name"]} — '
                    f'{int(r["n_aspects"] or 0)} aspects, '
                    f'{int(r["n_markets"] or 0)} markets': r['company_key']
                    for _, r in matches.iterrows()
                }
                st.caption(f'**{len(matches):,}** of {len(profiles):,} profiles match.')
                picked_label = st.selectbox(
                    'Company', list(labels), key='gs_profile_pick',
                )
                picked_key = labels[picked_label]
                row = profiles[profiles['company_key'] == picked_key]
                if not row.empty:
                    chosen = row.iloc[0]
                    # Switching client clears any results ranked against the old one.
                    prev = st.session_state.gs_profile
                    if prev is None or str(_profile_row(prev).get('company_key')) != str(picked_key) \
                            or st.session_state.gs_profile_origin != 'client':
                        st.session_state.gs_match_results = None
                        st.session_state.gs_match_meta    = None
                    st.session_state.gs_profile          = chosen
                    st.session_state.gs_profile_origin   = 'client'
                    st.session_state.gs_profile_warnings = None
                    st.session_state.gs_profile_saved    = True

    else:  # paste notes
        st.markdown('**Build a capability profile from your own notes**')
        st.caption(
            'Paste anything describing what the company does — call notes, a '
            'capability statement, a deck, website copy. Claude splits it into '
            'independently searchable aspects and the markets they serve, exactly '
            'as the Capability Profiles job does for a client. Everything is '
            'grounded in what you paste; nothing is invented.'
        )

        n1, n2, n3 = st.columns([2, 2, 1])
        notes_name    = n1.text_input('Company name', key='gs_notes_name',
                                      placeholder='Acme Robotics')
        notes_website = n2.text_input('Website', key='gs_notes_site',
                                      placeholder='https://acme.com')
        notes_state   = n3.text_input('State', key='gs_notes_state', placeholder='TX')

        notes = st.text_area(
            'Notes / source material', height=260, key='gs_notes_text',
            placeholder='Paste notes, a capability statement, meeting notes, product '
                        'descriptions…',
        )

        with st.expander('Profile options', expanded=False):
            p1, p2, p3 = st.columns(3)
            target_aspects = p1.slider(
                'Target aspects', ap.MIN_ASPECTS, ap.MAX_ASPECTS, 8, key='gs_notes_aspects',
                help='The prompt aims for this ±2, and returns fewer rather than '
                     'padding when the material does not support more.',
            )
            max_markets = p2.slider(
                'Max markets', 1, ap.MAX_MARKETS, 4, key='gs_notes_markets',
                help='Defense does not count towards this cap.',
            )
            model = p3.selectbox('Model', ap.ASPECT_MODELS, index=0, key='gs_notes_model')

            q1, q2 = st.columns(2)
            assess_defense = q1.checkbox(
                'Assess Defense market', value=True, key='gs_notes_defense',
                help='Adds a Defense market only when the material evidences an actual '
                     'DoD relationship or an active pursuit of one.',
            )
            assess_unexplored = q2.checkbox(
                'Assess unexplored markets', value=True, key='gs_notes_unexplored',
                help='A second Claude call inferring markets the company does NOT serve '
                     'but could extend into. Roughly doubles build time.',
            )
            max_unexplored = st.slider(
                'Max unexplored markets', 1, ap.MAX_UNEXPLORED, ap.MAX_UNEXPLORED,
                key='gs_notes_max_unexplored', disabled=not assess_unexplored,
            )

        notes_ready = len(notes.strip()) >= _MIN_NOTES_CHARS
        if notes.strip() and not notes_ready:
            st.caption(
                f'{len(notes.strip())}/{_MIN_NOTES_CHARS} characters — paste a bit more '
                'before building; too little material makes the model pad.'
            )

        if st.button('🧬 Build capability profile', type='primary', disabled=not notes_ready):
            try:
                with st.spinner('Claude is reading the material and splitting it into aspects…'):
                    record, warns = _build_profile_from_notes(
                        company_name      = notes_name.strip() or 'Untitled company',
                        website           = notes_website.strip(),
                        state             = notes_state.strip(),
                        notes             = notes,
                        target_aspects    = int(target_aspects),
                        max_markets       = int(max_markets),
                        assess_defense    = bool(assess_defense),
                        assess_unexplored = bool(assess_unexplored),
                        max_unexplored    = int(max_unexplored),
                        model             = model,
                    )
                st.session_state.gs_profile          = record
                st.session_state.gs_profile_origin   = 'notes'
                st.session_state.gs_profile_warnings = warns
                st.session_state.gs_profile_saved    = False
                st.session_state.gs_match_results    = None
                st.session_state.gs_match_meta       = None
                st.rerun()
            except Exception as e:
                st.error(f'Profile build failed: {e}')
                st.code(traceback.format_exc())

    # ── The profile in hand ────────────────────────────────────────────────

    if st.session_state.gs_profile is None:
        st.info('Pick a client profile or build one from notes to search with.')
        st.stop()

    prof = _profile_row(st.session_state.gs_profile)
    from_notes = st.session_state.gs_profile_origin == 'notes'

    st.divider()
    st.markdown(
        '#### Capability profile'
        + (' · built from pasted notes' if from_notes
           else f' · from the {pl.label(prof.get("pool"))} store')
    )
    _render_profile(prof)

    for warn in (st.session_state.gs_profile_warnings or []):
        st.warning(warn)

    # ── Save an ad-hoc profile into the client store ───────────────────────

    if from_notes:
        with st.expander(
            '💾 Save as a capability profile',
            expanded=not st.session_state.gs_profile_saved,
        ):
            # A single pool, not a scope: a save has to land in exactly one
            # store. Prospects is the default — a company profiled from pasted
            # notes is usually one we hold no records for at all.
            save_pool = uc.pool_selector(
                'gs_save_pool', label='Save into',
                pools=[pl.PROSPECTS, pl.CLIENTS],
                clears=('gs_profiles',),
                help='Prospects for a company we are targeting; Clients only if '
                     'we already work for it.',
            )
            st.caption(
                f'Writes this profile into `{ap.profiles_blob(save_pool)}`, the '
                'same store the Capability Profiles view edits and Aspect Match '
                'and HubSpot Import read. It does **not** create contact rows — '
                'import those in **Import Contacts** if you want the company in '
                f'the {pl.label(save_pool)} pool proper.'
            )
            s1, s2 = st.columns(2)
            save_name = s1.text_input(
                'Company name', value=str(prof.get('company_name') or ''),
                key='gs_save_name',
            ).strip()
            save_site = s2.text_input(
                'Website', value=str(prof.get('companyWebsite') or ''),
                key='gs_save_site',
            ).strip()

            save_key = ap.company_key(
                {'company_name': save_name, 'companyWebsite': save_site}
            )
            try:
                existing = ap.load_profiles(_get_storage_client(), pool=save_pool)
            except Exception:
                existing = ap.empty_profiles_df()
            collides = (
                not existing.empty
                and 'company_key' in existing.columns
                and (existing['company_key'] == save_key).any()
            )
            if collides:
                st.warning(
                    f'**{save_name}** already has a profile — saving replaces it, '
                    'including any aspects edited by hand in Capability Profiles.'
                )

            if not (save_name and save_site):
                st.caption(
                    'Both a name and a website are needed: together they form the '
                    '`company_key` every other view joins profiles on.'
                )
            if st.button(
                '💾 Save profile', key='gs_save_profile',
                disabled=not (save_name and save_site),
            ):
                try:
                    record = dict(prof)
                    record['company_key']    = save_key
                    record['company_name']   = save_name
                    record['companyWebsite'] = save_site
                    record['built_at']       = date.today().isoformat()
                    gcs    = _get_storage_client()
                    # Re-read rather than trusting the session copy: another
                    # session may have written a profile since this page loaded.
                    latest = ap.load_profiles(gcs, pool=save_pool)
                    merged = ap.upsert_profiles(latest, [record], pool=save_pool)
                    ap.save_profiles(gcs, merged, pool=save_pool)
                    record['pool']                    = save_pool
                    st.session_state.gs_profiles      = None
                    st.session_state.gs_profile       = pd.Series(record)
                    st.session_state.gs_profile_saved = True
                    st.success(
                        f'Saved to {pl.display(save_pool)} — **{save_name}** now '
                        f'has a capability profile with '
                        f'{int(record["n_aspects"] or 0)} aspect(s) and '
                        f'{int(record["n_markets"] or 0)} market(s).'
                    )
                except Exception as e:
                    st.error(f'Save failed: {e}')
                    st.code(traceback.format_exc())

    # ── 3b · Match settings ────────────────────────────────────────────────

    st.divider()
    st.markdown('#### Match this profile against the filtered topics')

    scope = st.radio(
        'Scope', [_SCOPE_ALL, _SCOPE_MARKET], horizontal=True, key='gs_match_scope',
        help=(
            'Whole company scores every aspect at once. By market scores each market '
            'on its own — its earmarked aspects plus the market narrative — so a '
            'defense story is ranked on its own terms instead of averaged in.'
        ),
    )

    unit_options = _unit_options(prof)
    units: list[am.Unit] = []
    if scope == _SCOPE_ALL:
        units = [am.Unit(prof, None, -1, '')]
    elif not unit_options:
        st.warning(
            'This profile has no markets — rebuild it in Capability Profiles, or '
            'search the whole company instead.'
        )
    else:
        # The option list changes whenever a different profile is picked (or a
        # notes profile is rebuilt). A keyed multiselect keeps its stored value
        # over `default`, and Streamlit drops stored values that are no longer
        # options — so a profile switch would leave this empty and grey out the
        # Match button with nothing said. Re-seed it when the options change.
        opt_sig = tuple(unit_options)
        if st.session_state.get('gs_match_markets_sig') != opt_sig:
            st.session_state.gs_match_markets_sig = opt_sig
            st.session_state.gs_match_markets     = list(unit_options)
        picked = st.multiselect(
            'Markets', list(unit_options), default=list(unit_options),
            key='gs_match_markets',
        )
        units = [unit_options[p] for p in picked]

    n_aspects = int(prof.get('n_aspects') or 0)
    o1, o2, o3 = st.columns(3)
    threshold = o1.slider('Aspect similarity threshold', 0.60, 0.95, 0.78, 0.01,
                          key='gs_match_threshold')
    min_hits  = o2.number_input(
        'Aspects that must clear it', min_value=1, max_value=max(1, n_aspects),
        value=1, step=1, key='gs_match_min_hits',
        help='1 = any single capability matching is enough (recommended — the '
             'company’s aspects are different capabilities, not requirements of one '
             'query). The market narrative does not count towards it.',
    )
    top_k = o3.number_input(
        'Top topics per market' if scope == _SCOPE_MARKET else 'Top topics',
        min_value=1, max_value=200, value=25, step=5, key='gs_match_top_k',
    )

    r1, r2, r3 = st.columns(3)
    do_rerank    = r1.checkbox('LLM re-rank', value=True, key='gs_match_rerank')
    rerank_model = r2.selectbox('Re-rank model', am.RERANK_MODELS, index=0,
                                disabled=not do_rerank, key='gs_match_rerank_model')
    min_llm      = r3.number_input('Keep LLM score ≥', min_value=1, max_value=5, value=3,
                                   step=1, disabled=not do_rerank, key='gs_match_min_llm')

    if do_rerank and units:
        st.caption(
            f'Up to **{len(units) * int(top_k):,}** re-rank calls '
            f'({len(units)} unit{"s" if len(units) != 1 else ""} × top {int(top_k)}), '
            f'{am.CONCURRENCY} at a time. Identical pairs are scored once and shared.'
        )

    run = st.button(
        '🎯 Match profile', type='primary',
        disabled=not units or filtered.empty,
    )

    # ── Run ────────────────────────────────────────────────────────────────

    if run:
        # st.stop() raises, so it must not be used inside this handler — every
        # failure path reports and falls through to the results section.
        try:
            with st.spinner('Preparing topic vectors…'):
                topic_matrix, topic_meta = am.stack_topic_embeddings(filtered)

            if topic_matrix.shape[0] == 0:
                st.error('None of the filtered topics carry a usable embedding.')
            else:
                if len(topic_meta) < len(filtered):
                    st.warning(
                        f'{len(filtered) - len(topic_meta):,} topic(s) skipped — '
                        'missing or malformed embedding.'
                    )

                prog = st.progress(0.0, text='Scoring…')
                candidates, skipped = am.match_units(
                    units, topic_matrix, topic_meta,
                    float(threshold), int(min_hits), int(top_k),
                    lambda frac, text: prog.progress(frac, text=text),
                )
                prog.empty()
                del topic_matrix

                for msg in skipped:
                    st.warning(msg)

                unscored, failures, reranked = 0, {}, False
                results = candidates
                if not candidates.empty and do_rerank:
                    rr_prog = st.progress(0.0, text='LLM re-ranking…')
                    results = am.run_rerank(
                        candidates, st.secrets['anthropic_api_key'], rerank_model,
                        lambda done, total: rr_prog.progress(
                            done / total, text=f'LLM re-ranking {done}/{total}…'
                        ),
                    )
                    rr_prog.empty()
                    unscored = int((results['llm_score'] == 0).sum())
                    # A pair scores 0 only when the call failed or the answer was
                    # unparseable. Without this, a re-ranker outage looks exactly
                    # like "nothing matched".
                    failures = (
                        results.loc[results['llm_score'] == 0, 'llm_rationale']
                        .astype(str).value_counts().head(5).to_dict()
                    )
                    reranked = True
                    results  = results[results['llm_score'] >= int(min_llm)]
                    results  = results.sort_values(
                        ['llm_score', 'aspect_score'], ascending=[False, False]
                    )
                elif not candidates.empty:
                    results = candidates.sort_values('aspect_score', ascending=False)

                st.session_state.gs_match_results = results.reset_index(drop=True)
                st.session_state.gs_match_meta = {
                    'client':     str(prof.get('company_name') or ''),
                    'threshold':  float(threshold),
                    'min_hits':   int(min_hits),
                    'top_k':      int(top_k),
                    'units':      len(units),
                    'scope':      scope,
                    'topics':     len(topic_meta),
                    'candidates': len(candidates),
                    'reranked':   reranked,
                    'unscored':   unscored,
                    'failures':   failures,
                    'min_llm':    int(min_llm) if reranked else None,
                }
        except Exception as e:
            st.error(f'Match failed: {e}')
            st.code(traceback.format_exc())

    # ── Results ────────────────────────────────────────────────────────────

    if st.session_state.gs_match_results is not None:
        results = st.session_state.gs_match_results
        meta    = st.session_state.gs_match_meta or {}

        if results.empty:
            # Two very different causes, and they need different fixes: nothing
            # cleared the similarity threshold, or the re-ranker scored/failed
            # everything below the minimum. An unscored pair is stored as 0,
            # which is below every selectable minimum, so a re-ranker outage
            # silently filters away an entire run.
            st.warning(
                f'No topics cleared {meta.get("threshold")} on at least '
                f'{meta.get("min_hits")} aspect(s) across '
                f'{meta.get("topics", 0):,} topics.'
                if not meta.get('candidates')
                else f'Similarity scoring found {meta.get("candidates"):,} candidate(s), '
                     f'but none scored {meta.get("min_llm")} or above — they were '
                     'dropped during re-ranking, not by the similarity threshold.'
            )
            if meta.get('unscored'):
                st.error(
                    f'{meta["unscored"]:,} pair(s) could not be scored at all and were '
                    'stored as 0, which is below every usable minimum.'
                )
                for reason, count in (meta.get('failures') or {}).items():
                    st.caption(f'· {count}× {reason}')
        else:
            st.success(
                f'**{len(results):,}** topic(s) for **{meta.get("client", "")}** '
                f'from {meta.get("candidates", 0):,} candidate(s) across '
                f'{meta.get("units", 0)} unit(s).'
            )
            if meta.get('unscored'):
                st.warning(
                    f'{meta["unscored"]:,} pair(s) could not be scored and were '
                    'excluded — not the same as scoring low.'
                )

            display = am.display_frame(results)
            st.dataframe(
                display, width='stretch', hide_index=True,
                column_config={
                    'aspect_score': st.column_config.NumberColumn('Aspect score', format='%.4f'),
                    'llm_score':    st.column_config.NumberColumn('LLM score', format='%d'),
                    'market_kind':  st.column_config.TextColumn('Kind', width='small'),
                },
            )
            st.download_button(
                '⬇ Download CSV',
                display.to_csv(index=False).encode('utf-8'),
                file_name=(
                    f'aspect_search_{meta.get("client", "profile").replace(" ", "_")}'
                    f'_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv'
                ),
                mime='text/csv',
            )
