"""
Bulk Matching
-------------
Load contacts and grant topics from GCS, run cosine-similarity matching,
optionally validate with AI and pre-write email copy, then export to CSV.
"""

import io

import numpy as np
import pandas as pd
import streamlit as st
from anthropic import Anthropic
from google.cloud import storage
from google.oauth2 import service_account
from openai import OpenAI

from src.modules.email_generator import generate_subject_line, josiah_copy
from src.modules.matcher import get_matches

# ── GCS ────────────────────────────────────────────────────────────────────

_BUCKET           = 'cc-matcher-bucket-jeg-v1'
_CONTACTS_PREFIX  = 'data/all-contacts/'
_TOPICS_PREFIX    = 'data/all-topics/processed/'
_MIN_THRESHOLD    = 0.82


def _get_storage_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)


def _list_prefixes(prefix: str) -> list[str]:
    try:
        client = _get_storage_client()
        blobs = client.list_blobs(_BUCKET, prefix=prefix, delimiter='/')
        list(blobs)
        return sorted(p.replace(prefix, '').strip('/') for p in blobs.prefixes)
    except Exception as e:
        st.error(f'Failed to list GCS prefixes: {e}')
        return []


def _load_parquets(prefix: str) -> pd.DataFrame:
    client = _get_storage_client()
    frames = []
    for blob in client.list_blobs(_BUCKET, prefix=prefix):
        if blob.name.endswith('.parquet'):
            frames.append(pd.read_parquet(io.BytesIO(blob.download_as_bytes())))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_contacts(sources: list[str]) -> pd.DataFrame:
    frames = []
    for source in sources:
        df = _load_parquets(f'{_CONTACTS_PREFIX}{source}/')
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_topics(agencies: list[str]) -> pd.DataFrame:
    client = _get_storage_client()
    frames = []
    for agency in agencies:
        for blob in client.list_blobs(_BUCKET, prefix=f'{_TOPICS_PREFIX}{agency}/'):
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


def _normalize_columns(contacts: pd.DataFrame, grants: pd.DataFrame):
    """Ensure company_summary and grant_summary exist after loading."""
    contacts = contacts.copy()
    grants   = grants.copy()

    # Contacts: need a 'summary' column for get_matches merge rename
    if 'summary' not in contacts.columns:
        if 'company_summary' in contacts.columns:
            contacts['summary'] = contacts['company_summary']

    # Grants: need 'grant_summary' for AI validation
    if 'grant_summary' not in grants.columns:
        for col in ('description', 'summary'):
            if col in grants.columns:
                grants = grants.rename(columns={col: 'grant_summary'})
                break

    return contacts, grants


def _normalize_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Ensure company_summary and grant_summary exist on the merged result."""
    matches = matches.copy()
    if 'company_summary' not in matches.columns:
        for col in ('summary_x', 'summary', 'company_summary'):
            if col in matches.columns:
                matches = matches.rename(columns={col: 'company_summary'})
                break
    if 'grant_summary' not in matches.columns:
        for col in ('summary_y', 'description'):
            if col in matches.columns:
                matches = matches.rename(columns={col: 'grant_summary'})
                break
    return matches


# ── Session state ───────────────────────────────────────────────────────────

for _k in ['bm_topics_df', 'bm_results_df']:
    if _k not in st.session_state:
        st.session_state[_k] = None
if 'bm_filters' not in st.session_state:
    st.session_state.bm_filters = [{'column': None, 'keyword': '', 'operator': 'AND'}]


# ── Page ────────────────────────────────────────────────────────────────────

st.title('⚙️ Bulk Matching')
st.caption('Match contacts against grant topics using cosine similarity.')

# ── Section 1 · Contact sources ─────────────────────────────────────────────

st.subheader('1 · Select contact sources')

contact_sources = _list_prefixes(_CONTACTS_PREFIX)
if not contact_sources:
    st.warning('No contact sources found in GCS.')
    st.stop()

src_cols = st.columns(min(len(contact_sources), 6))
selected_sources = [
    src for i, src in enumerate(contact_sources)
    if src_cols[i % len(src_cols)].checkbox(src, value=True, key=f'bm_src_{src}')
]

# ── Section 2 · Grant topics ─────────────────────────────────────────────────

st.divider()
st.subheader('2 · Select grant topics')

grant_agencies = _list_prefixes(_TOPICS_PREFIX)
if not grant_agencies:
    st.warning('No grant agencies found in GCS.')
    st.stop()

ag_cols = st.columns(min(len(grant_agencies), 6))
selected_agencies = [
    ag for i, ag in enumerate(grant_agencies)
    if ag_cols[i % len(ag_cols)].checkbox(ag, value=True, key=f'bm_agency_{ag}')
]

if st.button('Load Topics', type='primary', disabled=not selected_agencies):
    with st.spinner('Loading grant topics…'):
        df = _load_topics(selected_agencies)
    if df.empty:
        st.warning('No topics found for the selected agencies.')
    else:
        st.session_state.bm_topics_df = df
        st.session_state.bm_filters   = [{'column': None, 'keyword': '', 'operator': 'AND'}]
        st.session_state.bm_results_df = None
        st.success(f'Loaded **{len(df):,}** grant topics.')

if st.session_state.bm_topics_df is not None:
    df = st.session_state.bm_topics_df
    filterable_cols = [c for c in df.columns if c != 'embeddings']

    for f in st.session_state.bm_filters:
        if f['column'] not in filterable_cols:
            f['column'] = filterable_cols[0] if filterable_cols else None

    for i, f in enumerate(st.session_state.bm_filters):
        if i == 0:
            col_sel, kw_input, _, remove_col = st.columns([2, 3, 1, 0.5])
        else:
            op_col, col_sel, kw_input, remove_col = st.columns([1, 2, 3, 0.5])
            f['operator'] = op_col.radio(
                'op', ['AND', 'OR'], index=0 if f['operator'] == 'AND' else 1,
                key=f'bm_op_{i}', horizontal=True, label_visibility='collapsed'
            )
        f['column'] = col_sel.selectbox(
            'Column', filterable_cols,
            index=filterable_cols.index(f['column']) if f['column'] in filterable_cols else 0,
            key=f'bm_col_{i}', label_visibility='collapsed'
        )
        f['keyword'] = kw_input.text_input(
            'Keyword', value=f['keyword'],
            placeholder=f'Filter by {f["column"]}…',
            key=f'bm_kw_{i}', label_visibility='collapsed'
        )
        if remove_col.button('✕', key=f'bm_rm_{i}', disabled=len(st.session_state.bm_filters) == 1):
            st.session_state.bm_filters.pop(i)
            st.rerun()

    if st.button('+ Add filter', key='bm_add_filter'):
        st.session_state.bm_filters.append({'column': filterable_cols[0], 'keyword': '', 'operator': 'AND'})
        st.rerun()

    filtered_topics = _apply_filters(df, st.session_state.bm_filters)
    display_cols    = [c for c in filtered_topics.columns if c != 'embeddings']
    st.caption(f'**{len(filtered_topics):,}** topics match — showing first 50')
    st.dataframe(filtered_topics[display_cols].head(50), width="stretch", hide_index=True)

# ── Section 3 · Run options ───────────────────────────────────────────────────

st.divider()
st.subheader('3 · Run options')

opt_left, opt_mid, opt_right = st.columns([1, 1, 2])
with opt_left:
    threshold = st.slider(
        'Similarity threshold',
        min_value=_MIN_THRESHOLD, max_value=1.0,
        value=_MIN_THRESHOLD, step=0.01,
    )
with opt_mid:
    ai_validation = st.checkbox('AI validation', value=True)
    prewrite_email = st.checkbox('Pre-write email', value=False)

can_run = selected_sources and st.session_state.bm_topics_df is not None
if st.button('▶ Run Matching', type='primary', disabled=not can_run):
  try:

    # ── Load contacts ───────────────────────────────────────────────────
    with st.spinner('Loading contacts…'):
        contacts = _load_contacts(selected_sources)

    if contacts.empty:
        st.error('No contacts loaded.')
        st.stop()

    if 'embeddings' not in contacts.columns:
        st.error('Contacts are missing an embeddings column.')
        st.stop()

    # ── Prepare grants ──────────────────────────────────────────────────
    grants = _apply_filters(st.session_state.bm_topics_df, st.session_state.bm_filters)

    if grants.empty:
        st.error('No grant topics after filtering.')
        st.stop()

    if 'embeddings' not in grants.columns:
        st.error('Grant topics are missing an embeddings column.')
        st.stop()

    contacts, grants = _normalize_columns(contacts, grants)

    # Drop rows with null embeddings
    contacts = contacts[contacts['embeddings'].notna()].reset_index(drop=True)
    grants   = grants[grants['embeddings'].notna()].reset_index(drop=True)

    if contacts.empty or grants.empty:
        st.error('No valid embeddings found after filtering nulls.')
        st.stop()

    # ── Cosine similarity matching ──────────────────────────────────────
    with st.spinner('Running cosine similarity matching…'):
        matches = get_matches(threshold, grants, contacts)
        matches = _normalize_matches(matches)

    if matches.empty:
        st.warning(f'No matches found above {threshold} similarity threshold.')
        st.stop()

    st.success(f'Found **{len(matches):,}** candidate matches.')

    # ── AI validation ───────────────────────────────────────────────────
    if ai_validation:
        anth_client = Anthropic(api_key=st.secrets['anthropic_api_key'])
        matches['good_match'] = None
        yes_websites: list[str] = []
        bar = st.progress(0, text='AI validation…')

        for i, (idx, row) in enumerate(matches.iterrows()):
            website = row.get('companyWebsite', '')
            if website in yes_websites:
                matches.at[idx, 'good_match'] = 'yes'
            else:
                msg = anth_client.messages.create(
                    model='claude-3-haiku-20240307',
                    max_tokens=15,
                    temperature=0,
                    system=(
                        'Tell me if this company summary and grant summary are aligned. '
                        'Only give a one-word answer. Either "yes" or "no".'
                    ),
                    messages=[{'role': 'user', 'content': (
                        f"company summary: {row.get('company_summary', '')}\n\n"
                        f"grant summary: {row.get('grant_summary', '')}"
                    )}],
                )
                result = msg.content[0].text.strip().lower()
                matches.at[idx, 'good_match'] = result
                if 'yes' in result:
                    yes_websites.append(website)

            bar.progress((i + 1) / len(matches), text=f'AI validation: {i + 1}/{len(matches)}')

        bar.empty()
        yes_count = matches['good_match'].str.contains('yes', na=False).sum()
        st.success(f'AI validation complete — **{yes_count}** good matches.')

    # ── Pre-write email ─────────────────────────────────────────────────
    if prewrite_email:
        if ai_validation:
            email_targets = matches[matches['good_match'].str.contains('yes', na=False)]
        else:
            email_targets = matches

        if email_targets.empty:
            st.warning('No matches to generate emails for.')
        else:
            anth_client  = Anthropic(api_key=st.secrets['anthropic_api_key'])
            openai_client = OpenAI(api_key=st.secrets['openai_api_key'])
            matches['subject_line'] = None
            matches['ai_message']   = None
            bar = st.progress(0, text='Generating email copy…')

            for i, (idx, row) in enumerate(email_targets.iterrows()):
                matches.at[idx, 'subject_line'] = generate_subject_line(
                    company_summary=row.get('company_summary', ''),
                    agency=row.get('agency', row.get('broad_agency', '')),
                    openai_client=openai_client,
                    anth_client=anth_client,
                )
                matches.at[idx, 'ai_message'] = josiah_copy(
                    company_summary=row.get('company_summary', ''),
                    grant_summary=row.get('grant_summary', ''),
                    word_limit=50,
                    anth_client=anth_client,
                )
                bar.progress((i + 1) / len(email_targets), text=f'Generating emails: {i + 1}/{len(email_targets)}')

            bar.empty()
            st.success(f'Email copy generated for **{len(email_targets)}** matches.')

    st.session_state.bm_results_df = matches

# ── Section 4 · Results & export ─────────────────────────────────────────────

if st.session_state.bm_results_df is not None:
    results = st.session_state.bm_results_df
    st.divider()
    st.subheader('4 · Results & export')

    export_cols = [c for c in results.columns if c not in ('embeddings',)]
    selected_cols = st.multiselect(
        'Columns to export',
        options=export_cols,
        default=export_cols,
    )

    if selected_cols:
        preview = results[selected_cols]
        st.dataframe(preview, width="stretch", hide_index=True)

        csv_bytes = preview.to_csv(index=False).encode('utf-8')
        st.download_button(
            '⬇ Download CSV',
            data=csv_bytes,
            file_name='matches.csv',
            mime='text/csv',
            type='primary',
        )
