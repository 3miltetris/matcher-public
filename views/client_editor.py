"""
Client Editor
-------------
Edit the company summary for records in data/all-contacts/clients/ and
re-embed the new text. Multiple contact rows share one company summary,
so an edit is applied to every row of the selected company and the
source parquet file(s) are rewritten in place in GCS.

Admins (see src/modules/access_control.py) additionally get a delete
section for companies that are no longer clients: it removes every contact
row of the company, its multi-aspect profile, and its Drive Sync folder
assignment, archiving the removed rows to data/deleted-clients/ first.
"""

import io
import traceback

import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account

import src.modules.access_control as ac
import src.modules.client_delete as cd
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


def _render_delete_report() -> None:
    """Outcome of the last deletion. Also rendered on the 'no client files'
    path — deleting the last client company lands there."""
    report = st.session_state.get('ce_delete_report')
    if not report:
        return
    st.success('Deletion complete.')
    st.markdown(cd.format_report(report))
    for err in report['errors']:
        st.warning(err)
    if st.button('Dismiss', key='ce_del_dismiss'):
        st.session_state.ce_delete_report = None
        st.rerun()


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
    _render_delete_report()
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

# ── Re-embed & write back ──────────────────────────────────────────────────

if save_btn:
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

# ── Delete clients (admin only) ────────────────────────────────────────────

st.divider()
st.subheader('🗑 Delete clients')

_render_delete_report()

if not ac.is_admin():
    ac.admin_only_notice('Deleting clients')
    st.stop()

st.caption(
    'Removes the company from `data/all-contacts/clients/` entirely — every '
    'contact row, in every file it appears in. Removed rows are backed up to '
    f'`{cd.ARCHIVE_PREFIX}` first, so a mistake can be undone by hand.'
)

# Widget keys are tied to the selected client so switching clients above resets
# the selection and the confirmation — a stale "[Acme] + DELETE" carried over
# onto a different company is exactly the mistake this section must not make.
_del_key   = f'ce_del_keys_{selected_key}'
_conf_key  = f'ce_del_confirm_{selected_key}'

# Drop selections whose client no longer exists (deleted in an earlier pass)
# before the widget sees them.
if _del_key in st.session_state:
    st.session_state[_del_key] = [
        k for k in st.session_state[_del_key] if k in labels
    ]

del_keys = st.multiselect(
    'Clients to delete',
    options=list(labels.keys()),
    default=[selected_key],
    format_func=lambda k: labels[k],
    key=_del_key,
)

opt_l, opt_r = st.columns(2)
with opt_l:
    also_profile = st.checkbox(
        'Also delete their multi-aspect profile', value=True,
        help='Removes the row from data/client-profiles/profiles.parquet so the '
             'client disappears from Bulk Aspect Match.',
    )
with opt_r:
    also_drive = st.checkbox(
        'Also clear their Drive Sync folder assignment', value=True,
        help='Marks the folder skipped so Drive Sync neither syncs it nor '
             'proposes it as a new client on the next scan.',
    )

if del_keys:
    counts = cd.count_rows(frames, del_keys)
    st.dataframe(
        pd.DataFrame([
            {'client': labels[k].split('  (')[0], 'contact rows': counts.get(k, 0)}
            for k in del_keys
        ]),
        hide_index=True, use_container_width=True,
    )
    st.warning(
        f'This permanently deletes **{sum(counts.values())}** contact row(s) '
        f'across **{len(del_keys)}** company/companies.'
    )

confirm = st.text_input(
    'Type DELETE to confirm', key=_conf_key, placeholder='DELETE',
)

if st.button(
    f'🗑 Delete {len(del_keys)} client{"s" if len(del_keys) != 1 else ""} permanently',
    type='primary',
    disabled=not del_keys or confirm.strip().upper() != 'DELETE',
    help='Select at least one client and type DELETE to enable.',
):
    with st.spinner('Deleting…'):
        try:
            report = cd.delete_clients(
                _get_storage_client(),
                del_keys,
                delete_profiles         = also_profile,
                clear_drive_assignments = also_drive,
                actor                   = ac.current_user_email() or 'local-dev',
            )
        except Exception as e:
            st.error(f'Delete failed: {e}')
            st.code(traceback.format_exc())
            st.stop()

    st.session_state.ce_delete_report = report
    st.session_state.pop('ce_frames', None)      # reload clients from GCS
    st.session_state.pop(_del_key, None)         # deleted keys are no longer options
    st.session_state.pop(_conf_key, None)
    st.rerun()
