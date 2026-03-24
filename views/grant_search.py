"""
Grant Search
------------
Cosine-similarity search across grant topics stored in GCS.
Select agencies, apply keyword filters, describe your technology, and find matching grants.
"""

import io

import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account

from src.modules.Embedding.text_embedder import TextProcessor
from src.modules.GoogleBucketManager.bucket_manager import BucketManager

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


# ── Session state ──────────────────────────────────────────────────────────

for _k in ['gs_topics_df', 'gs_results_df']:
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
        st.session_state.gs_topics_df = df
        st.session_state.gs_results_df = None
        st.success(f'Loaded **{len(df):,}** topics.')

if st.session_state.gs_topics_df is None:
    st.stop()

df = st.session_state.gs_topics_df

# ── Section 2 · Filters + preview ─────────────────────────────────────────

st.divider()
st.subheader('2 · Filter topics')

filterable_cols = [c for c in df.columns if c != 'embeddings']

# Ensure existing filter columns are still valid after a reload
for f in st.session_state.gs_filters:
    if f['column'] not in filterable_cols:
        f['column'] = filterable_cols[0] if filterable_cols else None

# Render filter rows
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
    use_container_width=True,
    hide_index=True,
)

# ── Section 3 · Similarity search ──────────────────────────────────────────

st.divider()
st.subheader('3 · Technology search')

tech_col, thresh_col = st.columns([3, 1])
with tech_col:
    tech_text = st.text_area(
        'Technology description',
        height=120,
        placeholder='Describe the technology or capability you want to match against grant topics…',
    )
with thresh_col:
    threshold = st.slider('Similarity threshold', 0.0, 1.0, 0.75, 0.01)

if st.button('🔍 Search', type='primary', disabled=not tech_text.strip()):
    with st.spinner('Generating embedding…'):
        tp = TextProcessor(api_key=st.secrets['openai_api_key'])
        query_embedding = tp.get_embedding(tech_text.strip())
    with st.spinner('Scoring topics…'):
        st.session_state.gs_results_df = _similarity_search(filtered, query_embedding, threshold)

if st.session_state.gs_results_df is not None:
    results = st.session_state.gs_results_df
    if results.empty:
        st.warning(f'No topics above {threshold} similarity threshold.')
    else:
        result_cols = ['similarity_score'] + [c for c in results.columns if c not in ('embeddings', 'similarity_score')]
        st.success(f'**{len(results):,}** topics above {threshold} similarity.')
        st.dataframe(
            results[result_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                'similarity_score': st.column_config.NumberColumn('Score', format='%.4f'),
            },
        )
