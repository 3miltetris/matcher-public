"""
Grant Search
------------
Cosine-similarity search across grant topics stored in GCS.
Select agencies, apply keyword filters, describe your technology, and find matching grants.
"""

import io
import json

import anthropic
import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account

from src.modules.Embedding.text_embedder import TextProcessor
from src.modules.GoogleBucketManager.bucket_manager import BucketManager
from src.modules.grant_utils import normalize_grant_columns

# ── GCS ────────────────────────────────────────────────────────────────────

_BUCKET        = 'cc-matcher-bucket-jeg-v1'
_TOPICS_PREFIX = 'data/all-topics/processed/'


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


def _load_topics(agencies: list[str]) -> pd.DataFrame:
    client = _get_storage_client()
    frames = []
    for agency in agencies:
        prefix = f'{_TOPICS_PREFIX}{agency}/'
        for blob in client.list_blobs(_BUCKET, prefix=prefix):
            if blob.name.endswith('.parquet'):
                df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
                df['broad_agency'] = agency
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _apply_filters(df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    active = [f for f in filters if f['keyword'].strip() and f['column']]
    if not active:
        return df
    mask = df[active[0]['column']].astype(str).str.lower().str.contains(
        active[0]['keyword'].lower(), na=False
    )
    for f in active[1:]:
        m = df[f['column']].astype(str).str.lower().str.contains(
            f['keyword'].lower(), na=False
        )
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


def _decompose_query(query_text: str, anth_client: anthropic.Anthropic) -> list[str]:
    system = (
        'You are a scientific query analyzer for a government grant matching system.\n'
        'Decompose the technology description into 2–4 distinct required dimensions.\n'
        'Each dimension must be an independent necessary condition for a grant to be relevant.\n'
        'If the query is single-dimensional, return exactly one item.\n'
        'Rules:\n'
        '- Each dimension must be independently searchable (not a restatement of another)\n'
        '- Keep each concise (5–15 words)\n'
        '- Return ONLY a JSON array of strings, no other text\n\n'
        'Example:\n'
        'Query: "gene therapy delivery platform for long-term diabetes remission"\n'
        'Output: ["gene therapy viral vector delivery platform", '
        '"diabetes treatment metabolic disease insulin remission"]'
    )
    try:
        response = anth_client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=200,
            system=system,
            messages=[{'role': 'user', 'content': f'Query: "{query_text}"'}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if Claude adds them
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        if (
            isinstance(parsed, list)
            and 1 <= len(parsed) <= 4
            and all(isinstance(a, str) and a.strip() for a in parsed)
        ):
            return [a.strip() for a in parsed]
        st.warning(
            f'Aspect decomposition returned unexpected format — running standard search. '
            f'(Response: `{raw[:200]}`)'
        )
    except anthropic.APIError as e:
        st.warning(f'Claude API error during decomposition — running standard search. ({e})')
    except json.JSONDecodeError as e:
        st.warning(
            f'Could not parse Claude response as JSON — running standard search. '
            f'(Error: {e}; Raw: `{raw[:200]}`)'
        )
    except Exception as e:
        st.warning(f'Aspect decomposition failed — running standard search. ({type(e).__name__}: {e})')
    return [query_text]


def _embed_aspects(aspects: list[str], tp: TextProcessor) -> list[list[float]]:
    return [tp.get_embedding(a) for a in aspects]


def _multi_aspect_search(
    df: pd.DataFrame,
    aspect_embeddings: list[list[float]],
    threshold: float,
) -> pd.DataFrame:
    result = df.copy()
    aspect_cols = []
    for i, emb in enumerate(aspect_embeddings):
        vec = np.array(emb)
        col = f'aspect_{i + 1}_score'
        aspect_cols.append(col)

        def score(e, v=vec):
            try:
                return float(np.dot(np.array(e), v))
            except Exception:
                return 0.0

        result[col] = result['embeddings'].apply(score)

    result['min_aspect_score'] = result[aspect_cols].min(axis=1)
    mask = result[aspect_cols].ge(threshold).all(axis=1)
    return (
        result[mask]
        .sort_values('min_aspect_score', ascending=False)
        .reset_index(drop=True)
    )


def _llm_rerank(
    results: pd.DataFrame,
    query_text: str,
    anth_client: anthropic.Anthropic,
    top_n: int,
    progress,
) -> pd.DataFrame:
    system = (
        'You are evaluating how well a government grant topic matches a technology description.\n'
        'Score the match from 1 to 5:\n'
        '5 = Perfect match — all key requirements align\n'
        '4 = Strong match — most requirements align, minor gaps\n'
        '3 = Moderate match — some alignment, notable gaps\n'
        '2 = Weak match — superficial connection only\n'
        '1 = No match\n\n'
        'Also provide a one-sentence rationale.\n'
        'Return ONLY valid JSON: {"score": <integer 1-5>, "rationale": "<one sentence>"}'
    )
    subset = results.head(top_n).copy()
    total = len(subset)
    scores = []
    rationales = []
    for i, (_, row) in enumerate(subset.iterrows()):
        grant_text = row.get('grant_summary', '')
        try:
            resp = anth_client.messages.create(
                model='claude-haiku-4-5-20251001',
                max_tokens=150,
                system=system,
                messages=[{
                    'role': 'user',
                    'content': f'Technology: {query_text}\n\nGrant: {grant_text}',
                }],
            )
            parsed = json.loads(resp.content[0].text.strip())
            scores.append(int(parsed.get('score', 3)))
            rationales.append(str(parsed.get('rationale', '')))
        except Exception:
            scores.append(3)
            rationales.append('(scoring unavailable)')
        progress.progress((i + 1) / total)

    subset['llm_score'] = scores
    subset['llm_rationale'] = rationales
    return (
        subset[subset['llm_score'] >= 2]
        .sort_values(['llm_score', 'min_aspect_score'], ascending=[False, False])
        .reset_index(drop=True)
    )


def _clear_aspect_widgets(n: int) -> None:
    """Remove stale aspect widget keys so they reinitialise from value= on next render."""
    for j in range(n + 5):
        st.session_state.pop(f'gs_asp_{j}', None)


# ── Session state ──────────────────────────────────────────────────────────

for _k in ['gs_topics_df', 'gs_results_df', 'gs_aspects', 'gs_search_mode', 'gs_results_aspects']:
    if _k not in st.session_state:
        st.session_state[_k] = None
if 'gs_filters' not in st.session_state:
    st.session_state.gs_filters = [{'column': None, 'keyword': '', 'operator': 'AND'}]


# ── Page ───────────────────────────────────────────────────────────────────

st.title('🔍 Grant Search')
st.caption('Search grant topics by cosine similarity to your technology description.')

# ── Section 1 · Agency selection ───────────────────────────────────────────

st.subheader('1 · Select agencies')

agencies = _list_agencies()
if not agencies:
    st.warning('No agencies found in GCS.')
    st.stop()

cols = st.columns(min(len(agencies), 6))
selected = [
    agency for i, agency in enumerate(agencies)
    if cols[i % len(cols)].checkbox(agency, value=True, key=f'gs_agency_{agency}')
]

if st.button('Load Topics', type='primary', disabled=not selected):
    with st.spinner(f'Loading topics from {len(selected)} agenc{"y" if len(selected) == 1 else "ies"}…'):
        df = _load_topics(selected)
    if df.empty:
        st.warning('No topics found for the selected agencies.')
    else:
        st.session_state.gs_topics_df = normalize_grant_columns(df)
        st.session_state.gs_results_df = None
        st.session_state.gs_aspects = None
        st.session_state.gs_search_mode = None
        st.session_state.gs_results_aspects = None
        _clear_aspect_widgets(10)
        st.success(f'Loaded **{len(df):,}** topics.')

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
        col_sel, kw_input, _, remove_col = st.columns([2, 3, 1, 0.5])
    else:
        op_col, col_sel, kw_input, remove_col = st.columns([1, 2, 3, 0.5])
        f['operator'] = op_col.radio(
            'op', ['AND', 'OR'], index=0 if f['operator'] == 'AND' else 1,
            key=f'gs_op_{i}', horizontal=True, label_visibility='collapsed'
        )

    f['column'] = col_sel.selectbox(
        'Column', filterable_cols,
        index=filterable_cols.index(f['column']) if f['column'] in filterable_cols else 0,
        key=f'gs_col_{i}', label_visibility='collapsed'
    )
    f['keyword'] = kw_input.text_input(
        'Keyword', value=f['keyword'],
        placeholder=f'Filter by {f["column"]}…',
        key=f'gs_kw_{i}', label_visibility='collapsed'
    )
    if remove_col.button('✕', key=f'gs_rm_{i}', disabled=len(st.session_state.gs_filters) == 1):
        st.session_state.gs_filters.pop(i)
        st.rerun()

if st.button('+ Add filter'):
    st.session_state.gs_filters.append({'column': filterable_cols[0], 'keyword': '', 'operator': 'AND'})
    st.rerun()

filtered = _apply_filters(df, st.session_state.gs_filters)

display_cols = [c for c in filtered.columns if c != 'embeddings']
st.caption(f'**{len(filtered):,}** topics match — showing first 50')
st.dataframe(
    filtered[display_cols].head(50),
    width='stretch',
    hide_index=True,
)

# ── Section 3 · Similarity search ──────────────────────────────────────────

st.divider()
st.subheader('3 · Technology search')

# ── Mode and options ────────────────────────────────────────────────────────

mode_col, thresh_col = st.columns([3, 1])
with mode_col:
    multi_aspect = st.checkbox(
        'Multi-aspect search (recommended for complex queries)',
        value=True,
        help=(
            'Decomposes your description into independent required dimensions. '
            'A grant must score above the threshold on ALL dimensions — '
            'prevents partial-match results where only one topic area matches.'
        ),
    )

with thresh_col:
    threshold = st.slider('Similarity threshold', 0.0, 1.0, 0.75, 0.01)

llm_rerank = False
rerank_top_n = 20
if multi_aspect:
    rr_col, rn_col = st.columns([3, 1])
    with rr_col:
        llm_rerank = st.checkbox(
            'LLM re-ranking (slower, more accurate — top results only)',
            value=False,
            help='After aspect filtering, Claude scores each surviving grant 1–5 for relevance.',
        )
    if llm_rerank:
        with rn_col:
            rerank_top_n = st.number_input('Re-rank top N', min_value=5, max_value=50, value=20)

# Clear stale results when mode changes
current_mode = 'multi' if multi_aspect else 'single'
if st.session_state.gs_search_mode is not None and st.session_state.gs_search_mode != current_mode:
    st.session_state.gs_results_df = None
    st.session_state.gs_aspects = None
    st.session_state.gs_results_aspects = None
    _clear_aspect_widgets(10)
st.session_state.gs_search_mode = current_mode

# ── Query input ────────────────────────────────────────────────────────────

tech_text = st.text_area(
    'Technology description',
    height=120,
    placeholder='Describe the technology or capability you want to match against grant topics…',
)

query_too_short = len(tech_text.strip().split()) < 5

# ── Phase 1: Decompose (multi-aspect only) ─────────────────────────────────

if multi_aspect and not query_too_short:
    if st.button('Decompose query into aspects', disabled=not tech_text.strip()):
        with st.spinner('Identifying key aspects with Claude…'):
            anth_client = anthropic.Anthropic(api_key=st.secrets['anthropic_api_key'])
            aspects = _decompose_query(tech_text.strip(), anth_client)
        _clear_aspect_widgets(10)
        st.session_state.gs_aspects = aspects
        st.session_state.gs_results_df = None
        st.session_state.gs_results_aspects = None

    if st.session_state.gs_aspects is not None:
        st.markdown('**Edit aspects before searching** — each must score above threshold independently:')

        current_aspects = st.session_state.gs_aspects
        edited_aspects = []
        for i, asp in enumerate(current_aspects):
            a_col, btn_col = st.columns([10, 1])
            val = a_col.text_input(
                f'Aspect {i + 1}',
                value=asp,
                key=f'gs_asp_{i}',
            )
            edited_aspects.append(val)
            if btn_col.button('✕', key=f'gs_asp_rm_{i}', disabled=len(current_aspects) == 1):
                # Read all widget states (including aspects not yet rendered in this pass)
                all_vals = [
                    st.session_state.get(f'gs_asp_{j}', current_aspects[j])
                    for j in range(len(current_aspects))
                ]
                _clear_aspect_widgets(len(current_aspects))
                st.session_state.gs_aspects = [v for j, v in enumerate(all_vals) if j != i]
                st.rerun()

        # Sync edits back (only updates; no auto-removal of empties)
        st.session_state.gs_aspects = edited_aspects

        if st.button('+ Add aspect'):
            _clear_aspect_widgets(len(edited_aspects))
            st.session_state.gs_aspects = edited_aspects + ['']
            st.rerun()

elif multi_aspect and query_too_short:
    st.info('Query is too short for aspect decomposition — will run standard single-aspect search.')

# ── Phase 2: Search ────────────────────────────────────────────────────────

aspects_filled = (
    st.session_state.gs_aspects is not None
    and all(a.strip() for a in st.session_state.gs_aspects)
    and len(st.session_state.gs_aspects) > 0
)
aspects_ready = not multi_aspect or query_too_short or aspects_filled
search_disabled = not tech_text.strip() or not aspects_ready

if st.button('🔍 Search', type='primary', disabled=search_disabled):
    tp = TextProcessor(api_key=st.secrets['openai_api_key'])

    if multi_aspect and not query_too_short and aspects_filled:
        aspects_to_use = st.session_state.gs_aspects
        with st.spinner(f'Embedding {len(aspects_to_use)} aspect(s)…'):
            aspect_embeddings = _embed_aspects(aspects_to_use, tp)
        with st.spinner('Scoring topics against all aspects…'):
            results = _multi_aspect_search(filtered, aspect_embeddings, threshold)

        if llm_rerank and not results.empty:
            anth_client = anthropic.Anthropic(api_key=st.secrets['anthropic_api_key'])
            prog = st.progress(0, text='LLM re-ranking…')
            results = _llm_rerank(results, tech_text.strip(), anth_client, rerank_top_n, prog)
            prog.empty()

        st.session_state.gs_results_df = results
        st.session_state.gs_results_aspects = aspects_to_use
    else:
        with st.spinner('Generating embedding…'):
            query_embedding = tp.get_embedding(tech_text.strip())
        with st.spinner('Scoring topics…'):
            results = _similarity_search(filtered, query_embedding, threshold)
        st.session_state.gs_results_df = results
        st.session_state.gs_results_aspects = None

# ── Results ────────────────────────────────────────────────────────────────

if st.session_state.gs_results_df is not None:
    results = st.session_state.gs_results_df
    used_aspects = st.session_state.gs_results_aspects

    if results.empty:
        st.warning(
            f'No topics above **{threshold}** similarity threshold.'
            + (' Try lowering the threshold, simplifying your query, or editing the aspects.' if used_aspects else '')
        )
    else:
        st.success(f'**{len(results):,}** topics matched.')

        if used_aspects:
            with st.expander(f'Aspects used in search ({len(used_aspects)})', expanded=False):
                for i, asp in enumerate(used_aspects, 1):
                    st.markdown(f'**Aspect {i}:** {asp}')

        aspect_score_cols = [c for c in results.columns if c.startswith('aspect_') and c.endswith('_score')]
        if used_aspects and aspect_score_cols:
            primary_cols = ['min_aspect_score'] + aspect_score_cols
        else:
            primary_cols = ['similarity_score']

        if 'llm_score' in results.columns:
            primary_cols = ['llm_score', 'llm_rationale'] + primary_cols

        other_cols = [
            c for c in results.columns
            if c not in primary_cols and c != 'embeddings'
        ]
        result_cols = primary_cols + other_cols

        col_cfg: dict = {}
        if 'similarity_score' in result_cols:
            col_cfg['similarity_score'] = st.column_config.NumberColumn('Score', format='%.4f')
        if 'min_aspect_score' in result_cols:
            col_cfg['min_aspect_score'] = st.column_config.NumberColumn('Min Aspect Score', format='%.4f')
        if 'llm_score' in result_cols:
            col_cfg['llm_score'] = st.column_config.NumberColumn('LLM Score', format='%d')
        if 'llm_rationale' in result_cols:
            col_cfg['llm_rationale'] = st.column_config.TextColumn('Rationale')
        if used_aspects:
            for i, asp in enumerate(used_aspects, 1):
                col_key = f'aspect_{i}_score'
                if col_key in result_cols:
                    label = f'A{i}: {asp[:25]}…' if len(asp) > 25 else f'A{i}: {asp}'
                    col_cfg[col_key] = st.column_config.NumberColumn(label, format='%.4f')

        st.dataframe(
            results[result_cols],
            width='stretch',
            hide_index=True,
            column_config=col_cfg,
        )
