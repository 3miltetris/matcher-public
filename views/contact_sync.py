"""
Contact Sync
------------
Upsert contact embeddings from GCS into Pinecone.
Run once (or whenever the contact parquets are updated).
"""

import hashlib
import io
import traceback

import pandas as pd
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account
from pinecone import Pinecone

# ── Constants ────────────────────────────────────────────────────────────────

_BUCKET          = 'cc-matcher-bucket-jeg-v1'
_CONTACTS_PREFIX = 'data/all-contacts/'
_UPSERT_BATCH    = 100


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_storage_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)


def _get_pinecone_index():
    pc = Pinecone(api_key=st.secrets['pinecone_api_key'])
    return pc.Index(st.secrets['pinecone_index'])


def _list_contact_sources() -> list[str]:
    """Return the sub-prefixes directly under data/all-contacts/."""
    client = _get_storage_client()
    blobs  = client.list_blobs(_BUCKET, prefix=_CONTACTS_PREFIX, delimiter='/')
    list(blobs)  # consume iterator to populate blobs.prefixes
    return sorted(p.replace(_CONTACTS_PREFIX, '').strip('/') for p in blobs.prefixes)


def _iter_contact_blobs(sources: list[str]):
    """Yield (blob_name, DataFrame) one parquet at a time for the given sources."""
    client = _get_storage_client()
    for source in sources:
        prefix = f'{_CONTACTS_PREFIX}{source}/'
        for blob in client.list_blobs(_BUCKET, prefix=prefix):
            if blob.name.endswith('.parquet'):
                df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
                df['_source'] = source
                yield blob.name, df


def _make_vector_id(source: str, company_website: str) -> str:
    raw = f'{source}||{company_website}'
    return hashlib.md5(raw.encode('utf-8')).hexdigest()


def _build_vectors(df: pd.DataFrame, source: str) -> list[dict]:
    """Convert a contacts DataFrame into a list of Pinecone vector dicts."""
    vectors = []

    # Normalise summary column name
    if 'company_summary' in df.columns:
        summary_col = 'company_summary'
    elif 'summary' in df.columns:
        summary_col = 'summary'
    else:
        summary_col = None

    for _, row in df.iterrows():
        embedding = row.get('embeddings')
        if embedding is None:
            continue
        # Convert numpy arrays / pandas Series to plain list
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        else:
            embedding = list(embedding)

        if not embedding:
            continue

        website = str(row.get('companyWebsite', '') or '')
        vec_id  = _make_vector_id(source, website)

        raw_summary = str(row.get(summary_col, '') or '') if summary_col else ''
        metadata = {
            'source':          source,
            'companyName':     str(row.get('companyName',  '') or ''),
            'companyWebsite':  website,
            'firstName':       str(row.get('firstName',    '') or ''),
            'lastName':        str(row.get('lastName',     '') or ''),
            'email':           str(row.get('email',        '') or ''),
            'company_summary': raw_summary[:2000],
        }

        vectors.append({'id': vec_id, 'values': embedding, 'metadata': metadata})

    return vectors


# ── Page ─────────────────────────────────────────────────────────────────────

st.title('🔄 Contact Sync')
st.caption('Upsert contact embeddings from GCS into Pinecone.')

# ── Index stats ──────────────────────────────────────────────────────────────

st.subheader('Pinecone index stats')
try:
    _index  = _get_pinecone_index()
    _stats  = _index.describe_index_stats()
    total_vectors = _stats.get('total_vector_count', _stats.get('totalVectorCount', 'N/A'))
    st.metric('Total vectors', f'{total_vectors:,}' if isinstance(total_vectors, int) else total_vectors)
except Exception as _e:
    st.warning(f'Could not fetch index stats: {_e}')

# ── Source selection ─────────────────────────────────────────────────────────

st.divider()
st.subheader('Select contact sources to upsert')

try:
    contact_sources = _list_contact_sources()
except Exception as _e:
    st.error(f'Failed to list contact sources from GCS: {_e}')
    st.code(traceback.format_exc())
    st.stop()

if not contact_sources:
    st.warning('No contact sources found under `data/all-contacts/` in GCS.')
    st.stop()

src_cols = st.columns(min(len(contact_sources), 6))
selected_sources = [
    src
    for i, src in enumerate(contact_sources)
    if src_cols[i % len(src_cols)].checkbox(src, value=True, key=f'cs_src_{src}')
]

# ── Upsert button ─────────────────────────────────────────────────────────────

st.divider()

if st.button('Upsert to Pinecone', type='primary', disabled=not selected_sources):
    try:
        index = _get_pinecone_index()

        # Collect blob list first so we know the total for the progress bar
        with st.spinner('Enumerating parquet files in GCS…'):
            blob_list = list(_iter_contact_blobs(selected_sources))

        if not blob_list:
            st.error('No parquet files found for the selected sources.')
            st.stop()

        total_files    = len(blob_list)
        total_upserted = 0
        bar = st.progress(0, text='Starting…')

        for file_i, (blob_name, df) in enumerate(blob_list):
            source       = df['_source'].iloc[0] if '_source' in df.columns else 'unknown'
            short_name   = blob_name.split('/')[-1]
            bar.progress(
                file_i / total_files,
                text=f'Processing file {file_i + 1}/{total_files}: {short_name}'
            )

            vectors = _build_vectors(df, source)
            if not vectors:
                continue

            # Upsert in batches of _UPSERT_BATCH
            for batch_start in range(0, len(vectors), _UPSERT_BATCH):
                batch = vectors[batch_start : batch_start + _UPSERT_BATCH]
                index.upsert(vectors=batch)
                total_upserted += len(batch)

        bar.progress(1.0, text='Done.')
        st.success(f'Upserted **{total_upserted:,}** vectors across **{total_files}** files.')

        # ── Refresh stats ─────────────────────────────────────────────────
        updated_stats  = index.describe_index_stats()
        updated_total  = updated_stats.get(
            'total_vector_count',
            updated_stats.get('totalVectorCount', 'N/A')
        )
        st.metric(
            'Updated total vectors',
            f'{updated_total:,}' if isinstance(updated_total, int) else updated_total,
        )

    except Exception as e:
        st.error(f'**Error during upsert:** {e}')
        st.code(traceback.format_exc())
