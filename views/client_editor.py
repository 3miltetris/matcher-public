"""
Client Editor
-------------
Edit the company summary for records in data/all-contacts/clients/ and
re-embed the new text. Multiple contact rows share one company summary,
so an edit is applied to every row of the selected company and the
source parquet file(s) are rewritten in place in GCS.
"""

import io

import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account

from src.modules.Embedding.text_embedder import TextProcessor
from src.modules.GoogleBucketManager.bucket_manager import BucketManager

# ── Constants ──────────────────────────────────────────────────────────────

_BUCKET         = 'cc-matcher-bucket-jeg-v1'
_CLIENTS_PREFIX = 'data/all-contacts/clients/'


# ── GCS ────────────────────────────────────────────────────────────────────

def _get_storage_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)


def _load_client_frames() -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Returns ({blob_name: df}, errors). Frames are kept per-blob so edits
    can be written back to the exact file they came from."""
    client = _get_storage_client()
    blobs  = list(client.list_blobs(_BUCKET, prefix=_CLIENTS_PREFIX))

    frames: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for blob in blobs:
        if not blob.name.endswith('.parquet'):
            continue
        try:
            frames[blob.name] = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        except Exception as e:
            errors.append(f'{blob.name}: {e}')

    return frames, errors


def _company_key(row: pd.Series) -> str:
    name    = str(row.get('company_name') or '').strip()
    website = str(row.get('companyWebsite') or '').strip()
    return f'{name}||{website}'


def _group_mask(df: pd.DataFrame, key: str) -> pd.Series:
    name, website = key.split('||', 1)
    names    = df.get('company_name', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    websites = df.get('companyWebsite', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    return (names == name) & (websites == website)


# ── Page ───────────────────────────────────────────────────────────────────

st.title('✏️ Client Editor')
st.caption(
    'Update the company summary for a client and re-embed it. '
    'The change is applied to every contact row of that company and '
    'saved back to the original file in GCS.'
)

# ── Load ───────────────────────────────────────────────────────────────────

col_reload, col_count = st.columns([1, 5])
with col_reload:
    if st.button('↺ Reload', help='Refresh client data from GCS'):
        st.session_state.pop('ce_frames', None)
        st.rerun()

if 'ce_frames' not in st.session_state:
    with st.spinner('Loading clients from GCS…'):
        frames, load_errors = _load_client_frames()
    st.session_state.ce_frames = frames
    for err in load_errors:
        st.warning(err)

frames: dict[str, pd.DataFrame] = st.session_state.ce_frames

if not frames:
    st.warning(f'No parquet files found under {_CLIENTS_PREFIX} in GCS.')
    st.stop()

combined = pd.concat(
    [df.assign(_blob=blob_name) for blob_name, df in frames.items()],
    ignore_index=True,
)

with col_count:
    n_companies = combined.apply(_company_key, axis=1).nunique()
    st.info(f'{len(combined):,} contact rows across {n_companies:,} companies loaded.')

# ── Select company ─────────────────────────────────────────────────────────

st.divider()
st.subheader('Select client')

combined['_key'] = combined.apply(_company_key, axis=1)
groups = (
    combined.groupby('_key', sort=False)
    .agg(
        company_name=('company_name', 'first'),
        companyWebsite=('companyWebsite', 'first'),
        n_contacts=('_key', 'size'),
    )
    .reset_index()
    .sort_values('company_name', key=lambda s: s.fillna('').str.lower())
)

labels = {
    row['_key']: f"{row['company_name'] or '—'}  ·  {row['companyWebsite'] or 'no website'}"
                 f"  ({row['n_contacts']} contact{'s' if row['n_contacts'] != 1 else ''})"
    for _, row in groups.iterrows()
}
selected_key = st.selectbox(
    'Client company',
    options=list(labels.keys()),
    format_func=lambda k: labels[k],
)

group_rows = combined[combined['_key'] == selected_key]
first_row  = group_rows.iloc[0]

# ── Company details ────────────────────────────────────────────────────────

info_cols = st.columns(3)
info_cols[0].markdown(f"**Company:** {first_row.get('company_name') or '—'}")
website = str(first_row.get('companyWebsite') or '').strip()
info_cols[1].markdown(f'**Website:** [{website}]({website})' if website else '**Website:** —')
info_cols[2].markdown(f'**Contacts:** {len(group_rows)}')

contact_cols = [c for c in ['first_name', 'last_name', 'email', 'phone', 'lead_status'] if c in group_rows.columns]
if contact_cols:
    with st.expander('Contacts at this company'):
        st.dataframe(group_rows[contact_cols], hide_index=True, use_container_width=True)

page_text = str(first_row.get('full_text') or first_row.get('page_text') or '').strip()
if page_text:
    with st.expander('Scraped page text (reference)'):
        st.text(page_text[:10000])

# ── Edit summary ───────────────────────────────────────────────────────────

st.divider()
st.subheader('Summary')

current_summary = str(first_row.get('summary') or '').strip()

new_summary = st.text_area(
    'Company summary',
    value=current_summary,
    height=250,
    key=f'ce_summary_{selected_key}',
    help='This text is embedded and used for grant matching — describe what the company actually does.',
)

has_embeddings = isinstance(first_row.get('embeddings'), (list, np.ndarray)) and len(first_row.get('embeddings')) > 0
if not has_embeddings:
    st.warning('This company currently has no embedding — saving will create one.')

changed = new_summary.strip() != current_summary
save_btn = st.button(
    '💾 Re-embed & Save',
    type='primary',
    disabled=not new_summary.strip() or not changed,
    help='Edit the summary to enable saving.' if not changed else None,
)

if not save_btn:
    st.stop()

# ── Re-embed & write back ──────────────────────────────────────────────────

with st.spinner('Embedding new summary…'):
    try:
        tp        = TextProcessor(api_key=st.secrets['openai_api_key'])
        # float64 to match the dtype of existing rows — pyarrow cannot mix
        # float32 and float64 ndarrays in one parquet column
        embedding = np.array(tp.get_embedding(new_summary.strip()), dtype=np.float64)
    except Exception as e:
        st.error(f'Embedding failed: {e}')
        st.stop()

bm = BucketManager(_BUCKET, client=_get_storage_client())

rows_updated  = 0
files_written = []
for blob_name in group_rows['_blob'].unique():
    df   = frames[blob_name]
    mask = _group_mask(df, selected_key)
    if not mask.any():
        continue
    df.loc[mask, 'summary'] = new_summary.strip()
    for idx in df.index[mask]:
        df.at[idx, 'embeddings'] = embedding
    try:
        bm.upload_file(blob_name, df)
    except Exception as e:
        st.error(f'Failed to write {blob_name}: {e}')
        st.stop()
    rows_updated += int(mask.sum())
    files_written.append(blob_name)

st.success(
    f'Updated **{rows_updated}** contact row{"s" if rows_updated != 1 else ""} '
    f'for **{first_row.get("company_name") or website}** and saved to '
    f'{len(files_written)} file{"s" if len(files_written) != 1 else ""} in GCS.'
)
