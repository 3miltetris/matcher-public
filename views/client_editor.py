"""
Company Records
---------------
Edit the company summary for records in one pool (src/modules/pools.py —
🏢 Clients or 🎯 Prospects, picked at the top of the page) and re-embed the
new text. Multiple contact rows share one company summary, so an edit is
applied to every row of the selected company and the source parquet file(s)
are rewritten in place in GCS.

A prospect that signs is **promoted** into the client pool from here: its
contact rows and its capability profile move across, so nothing is re-imported
or re-researched (src/modules/pool_transfer.py).

Admins (see src/modules/access_control.py) additionally get a delete section
for companies that should leave the pool: it removes every contact row of the
company, its multi-aspect profile, and (clients only) its Drive Sync folder
assignment, archiving the removed rows to data/deleted-clients/ first.
"""

import traceback

import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account

import src.modules.access_control as ac
import src.modules.client_delete as cd
import src.modules.pool_transfer as ptr
import src.modules.pools as pl
import src.modules.ui_common as uc
from src.modules.Embedding.text_embedder import TextProcessor
from src.modules.GoogleBucketManager.bucket_manager import BucketManager

# ── Constants ──────────────────────────────────────────────────────────────

_BUCKET         = 'cc-matcher-bucket-jeg-v1'


# ── GCS ────────────────────────────────────────────────────────────────────

def _get_storage_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)


def _load_pool_frames(pool: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """({blob_name: df}, errors). Frames are kept per-blob so edits can be
    written back to the exact file they came from."""
    return pl.load_frames(_get_storage_client(), pool)


_company_key = pl.company_key
_group_mask  = pl.key_mask


def _render_move_report() -> None:
    """Outcome of the last promotion, rendered on the next run: the messages
    are written just before an st.rerun() that discards them."""
    report = st.session_state.get('ce_move_report')
    if not report:
        return
    st.success('Promotion complete.')
    st.markdown(ptr.format_report(report))
    for err in report['errors']:
        st.warning(err)
    if st.button('Dismiss', key='ce_move_dismiss'):
        st.session_state.ce_move_report = None
        st.rerun()


def _render_delete_report() -> None:
    """Outcome of the last deletion. Also rendered on the 'no files at all'
    path — deleting the last company of a pool lands there."""
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

st.title('✏️ Company Records')
st.caption(
    'Update the company summary and re-embed it. The change is applied to '
    'every contact row of that company and saved back to the original file '
    'in GCS.'
)

pool = uc.pool_selector(
    'ce_pool',
    clears=('ce_frames', 'ce_delete_report', 'ce_move_report'),
    help='Clients are the companies we write proposals for. Prospects are '
         'targets we are pursuing — promote one below when it signs.',
)
_NOUN = pl.noun(pool)

# ── Load ───────────────────────────────────────────────────────────────────

col_reload, col_count = st.columns([1, 5])
with col_reload:
    if st.button('↺ Reload', help=f'Refresh {_NOUN} data from GCS'):
        st.session_state.pop('ce_frames', None)
        st.rerun()

if 'ce_frames' not in st.session_state:
    with st.spinner(f'Loading {_NOUN}s from GCS…'):
        frames, load_errors = _load_pool_frames(pool)
    st.session_state.ce_frames = frames
    for err in load_errors:
        st.warning(err)

frames: dict[str, pd.DataFrame] = st.session_state.ce_frames

if not frames:
    st.warning(
        f'No parquet files found under {pl.contacts_prefix(pool)} in GCS.'
        + (' Import companies with the 🎯 Prospects destination in Import '
           'Contacts to start this pool.' if pool == pl.PROSPECTS else '')
    )
    _render_move_report()
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
st.subheader(f'Select {_NOUN}')

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
    f'{pl.label(pool).rstrip("s")} company',
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

# ── Promote prospects to clients ───────────────────────────────────────────
# Deliberately above the delete section: that one st.stop()s for non-admins,
# and promotion is an everyday workflow action, not a privileged one.

_render_move_report()

if pool == pl.PROSPECTS:
    st.divider()
    st.subheader('⬆️ Promote to client')
    st.caption(
        'Moves the company into `data/all-contacts/clients/` with everything '
        'already on it — research, meeting digests, documents — and moves its '
        'capability profile into the client profile store. Nothing is '
        're-imported or re-researched. The rows are written to the client pool '
        'before they are removed from prospects, so a failure mid-way leaves '
        'the company in both pools rather than in neither.'
    )

    _promote_key = f'ce_promote_keys_{selected_key}'
    if _promote_key in st.session_state:
        st.session_state[_promote_key] = [
            k for k in st.session_state[_promote_key] if k in labels
        ]

    promote_keys = st.multiselect(
        'Prospects to promote',
        options=list(labels.keys()),
        default=[selected_key],
        format_func=lambda k: labels[k],
        key=_promote_key,
    )
    move_profile = st.checkbox(
        'Also move their capability profile', value=True, key='ce_promote_profile',
        help='Moves the row from prospect_profiles.parquet into '
             'profiles.parquet, so the company appears in client-scoped Aspect '
             'Match runs and disappears from prospect-scoped ones. Unticking '
             'leaves the profile behind in the prospect store, where it would '
             'no longer have contact rows.',
    )

    if promote_keys:
        counts = cd.count_rows(frames, promote_keys)
        st.dataframe(
            pd.DataFrame([
                {'prospect': labels[k].split('  (')[0], 'contact rows': counts.get(k, 0)}
                for k in promote_keys
            ]),
            hide_index=True, use_container_width=True,
        )

    if st.button(
        f'⬆️ Promote {len(promote_keys)} prospect'
        f'{"s" if len(promote_keys) != 1 else ""} to client',
        type='primary',
        disabled=not promote_keys,
    ):
        with st.spinner('Moving…'):
            try:
                report = ptr.move_companies(
                    _get_storage_client(),
                    promote_keys,
                    source_pool  = pl.PROSPECTS,
                    dest_pool    = pl.CLIENTS,
                    move_profile = move_profile,
                    actor        = ac.current_user_email() or 'local-dev',
                )
            except Exception as e:
                st.error(f'Promotion failed: {e}')
                st.code(traceback.format_exc())
                st.stop()

        st.session_state.ce_move_report = report
        st.session_state.pop('ce_frames', None)      # both pools changed
        st.session_state.pop(_promote_key, None)
        st.rerun()


# ── Delete companies (admin only) ──────────────────────────────────────────

st.divider()
st.subheader(f'🗑 Delete {_NOUN}s')

_render_delete_report()

if not ac.is_admin():
    ac.admin_only_notice(f'Deleting {_NOUN}s')
    st.stop()

st.caption(
    f'Removes the company from `{pl.contacts_prefix(pool)}` entirely — every '
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
    f'{pl.label(pool)} to delete',
    options=list(labels.keys()),
    default=[selected_key],
    format_func=lambda k: labels[k],
    key=_del_key,
)

opt_l, opt_r = st.columns(2)
with opt_l:
    also_profile = st.checkbox(
        'Also delete their multi-aspect profile', value=True,
        help=f'Removes the row from {pl.profiles_blob(pool)} so the company '
             'disappears from Aspect Match.',
    )
with opt_r:
    # Prospects have no Drive folders — the checkbox would do nothing.
    also_drive = False
    if pl.supports_drive(pool):
        also_drive = st.checkbox(
            'Also clear their Drive Sync folder assignment', value=True,
            help='Marks the folder skipped so Drive Sync neither syncs it nor '
                 'proposes it as a new client on the next scan.',
        )

if del_keys:
    counts = cd.count_rows(frames, del_keys)
    st.dataframe(
        pd.DataFrame([
            {_NOUN: labels[k].split('  (')[0], 'contact rows': counts.get(k, 0)}
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
    f'🗑 Delete {len(del_keys)} {_NOUN}{"s" if len(del_keys) != 1 else ""} permanently',
    type='primary',
    disabled=not del_keys or confirm.strip().upper() != 'DELETE',
    help=f'Select at least one {_NOUN} and type DELETE to enable.',
):
    with st.spinner('Deleting…'):
        try:
            report = cd.delete_clients(
                _get_storage_client(),
                del_keys,
                delete_profiles         = also_profile,
                clear_drive_assignments = also_drive,
                actor                   = ac.current_user_email() or 'local-dev',
                pool                    = pool,
            )
        except Exception as e:
            st.error(f'Delete failed: {e}')
            st.code(traceback.format_exc())
            st.stop()

    st.session_state.ce_delete_report = report
    st.session_state.pop('ce_frames', None)      # reload the pool from GCS
    st.session_state.pop(_del_key, None)         # deleted keys are no longer options
    st.session_state.pop(_conf_key, None)
    st.rerun()
