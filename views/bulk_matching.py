"""
Bulk Matching
-------------
Query Pinecone with grant-topic embeddings to find matching contacts,
optionally validate with Claude AI and pre-write email copy, then export to CSV.
"""

import asyncio
import gc
import io
import random
import traceback
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from anthropic import AsyncAnthropic
from google.cloud import storage
from google.oauth2 import service_account
from openai import AsyncOpenAI  # AsyncOpenAI used inside _generate_email_batch

from src.modules.email_generator import async_generate_subject_line, async_josiah_copy

# ── Constants ────────────────────────────────────────────────────────────────

_BUCKET             = 'cc-matcher-bucket-jeg-v1'
_CONTACTS_PREFIX    = 'data/all-contacts/'
_TOPICS_PREFIX      = 'data/all-topics/processed/'
_MIN_THRESHOLD      = 0.82
_VALIDATION_BATCH   = 10   # concurrent Claude calls during validation
_EMAIL_BATCH        = 5    # rows per batch for email generation (2 API calls each)
_MAX_RETRIES        = 5    # retries on overload / rate-limit
_VALIDATION_SYSTEM  = (
    'Tell me if this company summary and grant summary are aligned. '
    'Only give a one-word answer. Either "yes" or "no".'
)


# ── GCS helpers ──────────────────────────────────────────────────────────────

def _get_storage_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)


def _list_prefixes(prefix: str) -> list[str]:
    try:
        client = _get_storage_client()
        blobs  = client.list_blobs(_BUCKET, prefix=prefix, delimiter='/')
        list(blobs)
        return sorted(p.replace(prefix, '').strip('/') for p in blobs.prefixes)
    except Exception as e:
        st.error(f'Failed to list GCS prefixes under `{prefix}`: {e}')
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
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return pd.concat(frames, ignore_index=True)


def _list_contact_blobs(sources: list[str]) -> list[tuple]:
    """Return (source, blob) pairs without downloading data."""
    client = _get_storage_client()
    result = []
    for source in sources:
        for blob in client.list_blobs(_BUCKET, prefix=f'{_CONTACTS_PREFIX}{source}/'):
            if blob.name.endswith('.parquet'):
                result.append((source, blob))
    return result


# ── Filter helpers ────────────────────────────────────────────────────────────

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


# ── Async helpers ─────────────────────────────────────────────────────────────

async def _validate_rows(
    rows: list[tuple[int, dict]],
    anth_key: str,
) -> list[tuple[int, str]]:
    """Run validation for a batch of (df_index, row_dict) pairs concurrently.
    Client is created inside the coroutine so it lives in the correct event loop."""
    async with AsyncAnthropic(api_key=anth_key) as anth_client:
        async def _one(idx: int, row: dict) -> tuple[int, str]:
            for attempt in range(_MAX_RETRIES):
                try:
                    msg = await anth_client.messages.create(
                        model='claude-3-haiku-20240307',
                        max_tokens=15,
                        temperature=0,
                        system=_VALIDATION_SYSTEM,
                        messages=[{
                            'role': 'user',
                            'content': (
                                f"company summary: {row.get('company_summary', '')}\n\n"
                                f"grant summary: {row.get('grant_summary', '')}"
                            ),
                        }],
                    )
                    return idx, msg.content[0].text.strip().lower()
                except Exception as e:
                    err = str(e)
                    if any(x in err for x in ('529', '429', 'overloaded', 'rate_limit', 'rate limit')):
                        await asyncio.sleep((2 ** attempt) + random.random())
                    else:
                        raise
            return idx, 'no'  # give up after retries — treat as non-match

        return await asyncio.gather(*[_one(idx, row) for idx, row in rows])


async def _generate_email_batch(
    rows: list[tuple[int, dict]],
    openai_key: str,
    anth_key: str,
) -> list[tuple[int, str, str]]:
    """Generate subject line + body concurrently for a batch of rows.
    Clients are created inside the coroutine so they live in the correct event loop."""
    async with AsyncOpenAI(api_key=openai_key) as openai_client, \
               AsyncAnthropic(api_key=anth_key) as anth_client:
        async def _one(idx: int, row: dict) -> tuple[int, str, str]:
            subject, body = await asyncio.gather(
                async_generate_subject_line(
                    company_summary=str(row.get('company_summary', '') or ''),
                    agency=str(row.get('agency', row.get('broad_agency', '')) or ''),
                    openai_client=openai_client,
                    anth_client=anth_client,
                ),
                async_josiah_copy(
                    company_summary=str(row.get('company_summary', '') or ''),
                    grant_summary=str(row.get('grant_summary', '') or ''),
                    word_limit=50,
                    anth_client=anth_client,
                ),
            )
            return idx, subject, body

        return await asyncio.gather(*[_one(idx, row) for idx, row in rows])


# ── Session state ─────────────────────────────────────────────────────────────

for _k in ['bm_topics_df', 'bm_results_df', 'bm_grant_embeddings']:
    if _k not in st.session_state:
        st.session_state[_k] = None
if 'bm_filters' not in st.session_state:
    st.session_state.bm_filters = [{'column': None, 'keyword': '', 'operator': 'AND'}]


# ── Page ──────────────────────────────────────────────────────────────────────

st.title('⚙️ Bulk Matching')
st.caption('Query Pinecone with grant-topic embeddings to find matching contacts.')

# ── Section 1 · Contact sources ───────────────────────────────────────────────

st.subheader('1 · Select contact sources')

contact_sources = _list_prefixes(_CONTACTS_PREFIX)
if not contact_sources:
    st.warning('No contact sources found in GCS.')
    st.stop()

src_cols = st.columns(min(len(contact_sources), 6))
selected_sources = [
    src
    for i, src in enumerate(contact_sources)
    if src_cols[i % len(src_cols)].checkbox(src, value=True, key=f'bm_src_{src}')
]

# ── Section 2 · Grant topics ──────────────────────────────────────────────────

st.divider()
st.subheader('2 · Select grant topics')

grant_agencies = _list_prefixes(_TOPICS_PREFIX)
if not grant_agencies:
    st.warning('No grant agencies found in GCS.')
    st.stop()

ag_cols = st.columns(min(len(grant_agencies), 6))
selected_agencies = [
    ag
    for i, ag in enumerate(grant_agencies)
    if ag_cols[i % len(ag_cols)].checkbox(ag, value=True, key=f'bm_agency_{ag}')
]

if st.button('Load Topics', type='primary', disabled=not selected_agencies):
    with st.spinner('Loading grant topics…'):
        _df = _load_topics(selected_agencies)
    if _df.empty:
        st.warning('No topics found for the selected agencies.')
    else:
        # Normalise grant description column to 'grant_summary'
        if 'grant_summary' not in _df.columns and 'description' in _df.columns:
            _df = _df.rename(columns={'description': 'grant_summary'})
        # Store embeddings as a compact numpy array — much cheaper than a pandas
        # column of Python lists, and frees the per-row list overhead from session state
        if 'embeddings' in _df.columns:
            st.session_state.bm_grant_embeddings = np.stack(_df['embeddings'].values)
            _df = _df.drop(columns=['embeddings'])
        else:
            st.session_state.bm_grant_embeddings = None
        st.session_state.bm_topics_df  = _df
        st.session_state.bm_filters    = [{'column': None, 'keyword': '', 'operator': 'AND'}]
        st.session_state.bm_results_df = None
        st.success(f'Loaded **{len(_df):,}** grant topics.')

if st.session_state.bm_topics_df is not None:
    _topics_df     = st.session_state.bm_topics_df
    filterable_cols = [c for c in _topics_df.columns if c != 'embeddings']

    for f in st.session_state.bm_filters:
        if f['column'] not in filterable_cols:
            f['column'] = filterable_cols[0] if filterable_cols else None

    for i, f in enumerate(st.session_state.bm_filters):
        if i == 0:
            col_sel, kw_input, _, remove_col = st.columns([2, 3, 1, 0.5])
        else:
            op_col, col_sel, kw_input, remove_col = st.columns([1, 2, 3, 0.5])
            f['operator'] = op_col.radio(
                'op', ['AND', 'OR'],
                index=0 if f['operator'] == 'AND' else 1,
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
        st.session_state.bm_filters.append(
            {'column': filterable_cols[0], 'keyword': '', 'operator': 'AND'}
        )
        st.rerun()

    _filtered_preview = _apply_filters(_topics_df, st.session_state.bm_filters)
    _display_cols     = [c for c in _filtered_preview.columns if c != 'embeddings']
    st.caption(f'**{len(_filtered_preview):,}** topics match — showing first 50')
    st.dataframe(_filtered_preview[_display_cols].head(50), width="stretch", hide_index=True)

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
    top_k = st.number_input(
        'Top-K topics per contact',
        min_value=1, max_value=50,
        value=5, step=1,
    )
with opt_mid:
    ai_validation  = st.checkbox('AI validation',  value=True)
    prewrite_email = st.checkbox('Pre-write email', value=False)

can_run = (
    bool(selected_sources)
    and st.session_state.bm_topics_df is not None
    and st.session_state.bm_grant_embeddings is not None
)

if st.button('▶ Run Matching', type='primary', disabled=not can_run):
    try:
        # ── Prepare filtered grants ───────────────────────────────────────
        grants = _apply_filters(st.session_state.bm_topics_df, st.session_state.bm_filters)

        if grants.empty:
            st.error('No grant topics after applying filters.')
            st.stop()

        # ── Build grant matrix from the pre-stacked numpy array ────────────
        # bm_grant_embeddings rows correspond 1-to-1 with bm_topics_df rows
        all_embeddings = st.session_state.bm_grant_embeddings  # (n_all_topics, 1536)
        filtered_indices = grants.index.tolist()
        grant_embeddings = all_embeddings[filtered_indices]     # (n_filtered, 1536)
        del all_embeddings

        if grant_embeddings.shape[0] == 0:
            st.error('All grant topics have null embeddings after filtering.')
            st.stop()

        grant_cols = ['topic_number', 'title', 'agency', 'broad_agency', 'due_date', 'grant_summary']
        grant_meta = grants[[c for c in grant_cols if c in grants.columns]].reset_index(drop=True)

        # ── Stream contacts one file at a time — top-K topics per contact ─
        with st.spinner('Listing contact files…'):
            blob_list = _list_contact_blobs(selected_sources)

        if not blob_list:
            st.error('No contact parquet files found for the selected sources.')
            st.stop()

        match_rows: list[dict] = []
        bar = st.progress(0, text='Matching contacts…')

        for file_i, (source, blob) in enumerate(blob_list):
            df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
            df = df[df['embeddings'].notna()].reset_index(drop=True)
            if df.empty:
                continue

            # Normalise summary column
            if 'company_summary' not in df.columns and 'summary' in df.columns:
                df = df.rename(columns={'summary': 'company_summary'})

            contact_embeddings = np.stack(df['embeddings'].values)  # (n_contacts, 1536)
            scores = np.dot(contact_embeddings, grant_embeddings.T)  # (n_contacts, n_grants)

            for ci in range(len(df)):
                contact_scores = scores[ci]
                above = np.where(contact_scores >= threshold)[0]
                if len(above) == 0:
                    continue
                top_indices = above[np.argsort(contact_scores[above])[::-1][:int(top_k)]]
                contact_row = df.iloc[ci]
                for gi in top_indices:
                    row = {
                        'companyName':     str(contact_row.get('companyName', '') or contact_row.get('company_name', '') or ''),
                        'companyWebsite':  str(contact_row.get('companyWebsite', '') or ''),
                        'firstName':       str(contact_row.get('firstName',      '') or ''),
                        'lastName':        str(contact_row.get('lastName',       '') or ''),
                        'email':           str(contact_row.get('email',          '') or ''),
                        'company_summary': str(contact_row.get('company_summary', '') or ''),
                        'source':          source,
                        'similarity_score': float(contact_scores[gi]),
                    }
                    for col in grant_meta.columns:
                        row[col] = grant_meta.iloc[gi].get(col, '')
                    match_rows.append(row)

            del contact_embeddings, scores
            bar.progress((file_i + 1) / len(blob_list),
                         text=f'File {file_i + 1}/{len(blob_list)}: {blob.name.split("/")[-1]}')

        bar.empty()
        del grant_embeddings, grant_meta

        if not match_rows:
            st.warning(f'No matches found above {threshold} similarity threshold.')
            st.stop()

        matches = pd.DataFrame(match_rows)
        del match_rows
        gc.collect()

        # Parquets store date columns as Timestamps; cast to str so Arrow can serialize them
        for _col in ('scraped_at', 'open_date', 'close_date'):
            if _col in matches.columns:
                matches[_col] = matches[_col].astype(str)

        st.success(f'Found **{len(matches):,}** candidate matches.')

        # ── AI validation (async, deduplicated by website) ────────────────
        if ai_validation:
            anth_key              = st.secrets['anthropic_api_key']
            matches['good_match'] = None

            # One API call per unique website; rows without a website each get their own call
            website_map: dict[str, list[int]] = {}
            no_site_rows: list[tuple[int, dict]] = []
            for idx, row in matches.iterrows():
                site = str(row.get('companyWebsite', '') or '').strip()
                if site:
                    website_map.setdefault(site, []).append(idx)
                else:
                    no_site_rows.append((idx, row.to_dict()))

            unique_tasks = (
                [(indices[0], matches.loc[indices[0]].to_dict()) for indices in website_map.values()]
                + no_site_rows
            )
            total_tasks  = len(unique_tasks)
            done         = 0
            idx_to_result: dict[int, str] = {}
            bar = st.progress(0, text='AI validation…')

            for i in range(0, total_tasks, _VALIDATION_BATCH):
                batch   = unique_tasks[i : i + _VALIDATION_BATCH]
                results = asyncio.run(_validate_rows(batch, anth_key))
                idx_to_result.update(results)
                done += len(batch)
                bar.progress(done / total_tasks, text=f'AI validation: {done}/{total_tasks} unique companies')

            bar.empty()

            # Broadcast results back to all rows
            for site, indices in website_map.items():
                result = idx_to_result.get(indices[0], 'no')
                for idx in indices:
                    matches.at[idx, 'good_match'] = result
            for idx, _ in no_site_rows:
                matches.at[idx, 'good_match'] = idx_to_result.get(idx, 'no')

            # Keep only confirmed matches — drop "no" rows now so export is clean
            matches   = matches[matches['good_match'].str.contains('yes', na=False)].reset_index(drop=True)
            yes_count = len(matches)
            st.success(f'AI validation complete — **{yes_count}** good matches.')

        # ── Pre-write email (async, runs only on the surviving "yes" rows) ─
        if prewrite_email:
            if matches.empty:
                st.warning('No confirmed matches to generate emails for.')
            else:
                anth_key     = st.secrets['anthropic_api_key']
                openai_key   = st.secrets['openai_api_key']
                matches['subject_line'] = None
                matches['ai_message']   = None
                email_rows = [(idx, row.to_dict()) for idx, row in matches.iterrows()]
                total      = len(email_rows)
                done       = 0
                bar = st.progress(0, text='Generating email copy…')

                for i in range(0, total, _EMAIL_BATCH):
                    batch   = email_rows[i : i + _EMAIL_BATCH]
                    results = asyncio.run(_generate_email_batch(batch, openai_key, anth_key))
                    for idx, subject, body in results:
                        matches.at[idx, 'subject_line'] = subject
                        matches.at[idx, 'ai_message']   = body
                    done += len(batch)
                    bar.progress(done / total, text=f'Generating emails: {done}/{total}')

                bar.empty()
                st.success(f'Email copy generated for **{total}** matches.')

        st.session_state.bm_results_df = matches

    except Exception as e:
        st.error(f'**Error:** {e}')
        st.code(traceback.format_exc())

# ── Section 4 · Results & export ──────────────────────────────────────────────

if st.session_state.bm_results_df is not None:
    results = st.session_state.bm_results_df
    st.divider()
    st.subheader('4 · Results & export')

    export_cols   = [c for c in results.columns if c != 'embeddings']
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
