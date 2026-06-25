"""
Resume Search
-------------
Cosine-similarity search across processed resume records stored in GCS.
Enter a natural-language description of the candidate you're looking for
and get a ranked list of matching contacts with their expertise summaries.
"""

import io

import numpy as np
_ARRAY_TYPES = (list, np.ndarray)
import pandas as pd
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account

from src.modules.Embedding.text_embedder import TextProcessor

# ── Constants ──────────────────────────────────────────────────────────────

_BUCKET         = 'cc-matcher-bucket-jeg-v1'
_RESUMES_PREFIX = 'data/resumes/'


# ── GCS ────────────────────────────────────────────────────────────────────

def _get_storage_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)


def _load_resumes() -> tuple[pd.DataFrame, list[str]]:
    """Returns (df, errors). df contains only rows with valid embeddings."""
    client = _get_storage_client()
    blobs  = list(client.list_blobs(_BUCKET, prefix=_RESUMES_PREFIX))
    parquet_blobs = [b for b in blobs if b.name.endswith('.parquet')]

    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for blob in parquet_blobs:
        try:
            frames.append(pd.read_parquet(io.BytesIO(blob.download_as_bytes())))
        except Exception as e:
            errors.append(f'{blob.name}: {e}')

    if not frames:
        return pd.DataFrame(), errors

    df = pd.concat(frames, ignore_index=True)
    if 'embeddings' not in df.columns:
        errors.append('embeddings column missing from parquet — re-run the importer embed step')
        return pd.DataFrame(), errors

    before = len(df)
    df = df[df['embeddings'].apply(lambda e: isinstance(e, _ARRAY_TYPES) and len(e) > 0)]
    skipped = before - len(df)
    if skipped:
        errors.append(f'{skipped} row(s) skipped — empty embeddings (contacts with no extractable resume text)')

    return df.reset_index(drop=True), errors


# ── Page ───────────────────────────────────────────────────────────────────

st.title('🔎 Resume Search')
st.caption(
    'Describe the candidate you need and find the best matches from your resume database. '
    'Results are ranked by embedding similarity to your query.'
)

# ── Load ───────────────────────────────────────────────────────────────────

col_reload, col_count = st.columns([1, 5])
with col_reload:
    if st.button('↺ Reload', help='Refresh resume data from GCS'):
        st.rerun()

with st.spinner('Loading resumes from GCS…'):
    resumes_df, load_errors = _load_resumes()

for err in load_errors:
    st.warning(err)

if resumes_df.empty:
    st.warning('No resumes with valid embeddings found in GCS. Use the Resume Importer to add contacts.')
    st.stop()

with col_count:
    st.info(f'{len(resumes_df):,} resumes loaded.')

# ── Filters ────────────────────────────────────────────────────────────────

st.divider()
st.subheader('Search')

kf_col, ex_col = st.columns(2)
with kf_col:
    keyword = st.text_input(
        'Include keyword (optional)',
        placeholder='e.g. machine learning, NIH, Python',
        help='Only resumes whose expertise summary contains this keyword will be searched.',
    )
with ex_col:
    exclude_keywords = st.text_input(
        'Exclude keywords (optional, comma-separated)',
        placeholder='e.g. sales, marketing',
        help='Resumes whose expertise summary contains any of these keywords will be excluded.',
    )

query = st.text_area(
    'Describe the candidate you\'re looking for *',
    placeholder=(
        'e.g. Biomedical engineer with experience in NIH-funded clinical trials '
        'and machine learning for medical imaging.'
    ),
    height=100,
)

s1, s2 = st.columns(2)
with s1:
    top_k = st.slider('Max results', min_value=5, max_value=50, value=10, step=5)
with s2:
    min_score = st.slider('Min similarity', min_value=0.50, max_value=0.99, value=0.70, step=0.01,
                          format='%.2f')

search_btn = st.button('🔍 Search', type='primary', disabled=not query.strip())

if not search_btn:
    st.stop()

query = query.strip()
if not query:
    st.warning('Enter a search query to continue.')
    st.stop()

# ── Apply keyword filter ───────────────────────────────────────────────────

pool = resumes_df.copy()
if keyword.strip():
    kw = keyword.strip().lower()
    mask = pool['expertise_summary'].fillna('').str.lower().str.contains(kw, regex=False)
    pool = pool[mask].reset_index(drop=True)
    if pool.empty:
        st.warning(f'No resumes contain the keyword "{keyword}". Try a broader term.')
        st.stop()

if exclude_keywords.strip():
    ex_terms = [t.strip().lower() for t in exclude_keywords.split(',') if t.strip()]
    summaries = pool['expertise_summary'].fillna('').str.lower()
    exclude_mask = summaries.apply(lambda s: any(t in s for t in ex_terms))
    excluded_count = exclude_mask.sum()
    pool = pool[~exclude_mask].reset_index(drop=True)
    if excluded_count:
        st.info(f'{excluded_count} resume(s) excluded by keyword filter.')
    if pool.empty:
        st.warning('All resumes were excluded by the exclude keyword filter. Try different terms.')
        st.stop()

# ── Embed query & score ────────────────────────────────────────────────────

with st.spinner('Embedding query…'):
    try:
        tp          = TextProcessor(api_key=st.secrets['openai_api_key'])
        query_vec   = np.array(tp.get_embedding(query), dtype=np.float32)
        resume_mat  = np.stack(pool['embeddings'].tolist()).astype(np.float32)
        scores      = resume_mat @ query_vec  # cosine similarity (vectors are unit-normed by ada-002)
    except Exception as e:
        st.error(f'Embedding failed: {e}')
        st.stop()

pool['_score'] = scores

# ── Filter & rank ──────────────────────────────────────────────────────────

results = (
    pool[pool['_score'] >= min_score]
    .sort_values('_score', ascending=False)
    .head(top_k)
    .reset_index(drop=True)
)

if results.empty:
    st.info(
        f'No matches above {min_score:.2f} similarity. '
        'Try lowering the threshold or broadening your query.'
    )
    st.stop()

st.success(f'**{len(results)}** match{"es" if len(results) != 1 else ""} found.')
st.divider()

# ── Display results ────────────────────────────────────────────────────────

for rank, (_, row) in enumerate(results.iterrows(), start=1):
    name    = f"{row.get('firstName', '')} {row.get('lastName', '')}".strip() or '—'
    email   = row.get('email', '—')
    company = row.get('company', '')
    summary = row.get('expertise_summary', '')
    url     = row.get('resume_url', '')
    score   = row['_score']

    with st.container(border=True):
        hcol1, hcol2 = st.columns([5, 1])
        with hcol1:
            header = f'**#{rank} — {name}**'
            if company:
                header += f'  ·  {company}'
            st.markdown(header)
        with hcol2:
            st.metric('Score', f'{score:.3f}')

        info_cols = st.columns(3)
        info_cols[0].markdown(f'📧 {email}')
        phone = row.get('phone', '')
        if phone:
            info_cols[1].markdown(f'📞 {phone}')
        if url:
            info_cols[2].markdown(f'[📄 Resume]({url})')

        if summary:
            st.markdown(summary)

# ── Export ─────────────────────────────────────────────────────────────────

st.divider()
export_cols = [c for c in [
    'email', 'firstName', 'lastName', 'phone', 'company',
    'expertise_summary', 'resume_url', '_score',
] if c in results.columns]
export_df = results[export_cols].rename(columns={'_score': 'similarity_score'})

st.download_button(
    '⬇ Download results CSV',
    data=export_df.to_csv(index=False).encode('utf-8'),
    file_name='resume_search_results.csv',
    mime='text/csv',
)
