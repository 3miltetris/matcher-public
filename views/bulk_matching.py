"""
Bulk Matching
-------------
Query grant-topic embeddings to find matching contacts,
optionally validate with Claude AI and pre-write email copy,
then save results in 1 000-row segments to GCS.
"""

import asyncio
import gc
import io
import random
import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from anthropic import AsyncAnthropic
from google.cloud import storage
from google.oauth2 import service_account
from openai import AsyncOpenAI

from src.modules.email_generator import async_generate_subject_line, async_josiah_copy

# ── Constants ────────────────────────────────────────────────────────────────

_BUCKET             = 'cc-matcher-bucket-jeg-v1'
_CONTACTS_PREFIX    = 'data/all-contacts/'
_TOPICS_PREFIX      = 'data/all-topics/processed/'
_RESULTS_PREFIX     = 'matching-results/'
_MIN_THRESHOLD      = 0.82
_SEGMENT_SIZE       = 1000  # rows processed end-to-end before freeing memory
_VALIDATION_BATCH   = 10    # concurrent Claude calls during validation
_EMAIL_BATCH        = 5     # rows per batch for email generation (2 API calls each)
_MAX_RETRIES        = 5
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


def _upload_csv(df: pd.DataFrame, blob_path: str, client: storage.Client) -> None:
    blob = client.bucket(_BUCKET).blob(blob_path)
    blob.upload_from_string(df.to_csv(index=False).encode('utf-8'), content_type='text/csv')


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
            return idx, 'no'

        return await asyncio.gather(*[_one(idx, row) for idx, row in rows])


async def _generate_email_batch(
    rows: list[tuple[int, dict]],
    openai_key: str,
    anth_key: str,
) -> list[tuple[int, str, str]]:
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


# ── Segment processor ─────────────────────────────────────────────────────────

def _process_segment(
    segment: pd.DataFrame,
    seg_num: int,
    n_segments: int,
    ai_validation: bool,
    prewrite_email: bool,
    anth_key: str,
    openai_key: str,
    status_placeholder,
    seen_websites: dict[str, str],  # mutated in-place; carries results across segments
) -> pd.DataFrame:
    """
    Run AI validation and/or email pre-write on a single segment DataFrame.
    Returns the processed (and filtered) segment. Frees intermediate objects.

    seen_websites maps companyWebsite → 'yes'/'no' for every site already
    validated in a prior segment, so the same company is never called twice.
    """
    seg_label = f'Segment {seg_num}/{n_segments}'

    # ── AI validation ─────────────────────────────────────────────────────
    if ai_validation:
        segment = segment.copy()
        segment['good_match'] = None

        website_map: dict[str, list[int]] = {}   # site → row indices in this segment
        unique_tasks: list[tuple[int, dict]] = [] # only NEW sites need API calls

        for idx, row in segment.iterrows():
            site = str(row.get('companyWebsite', '') or '').strip()
            if site:
                if site in seen_websites:
                    # Already validated in a prior segment — apply directly
                    segment.at[idx, 'good_match'] = seen_websites[site]
                else:
                    if site not in website_map:
                        website_map[site] = []
                        unique_tasks.append((idx, {
                            'company_summary': row.get('company_summary', ''),
                            'grant_summary':   row.get('grant_summary',   ''),
                        }))
                    website_map[site].append(idx)
            else:
                unique_tasks.append((idx, {
                    'company_summary': row.get('company_summary', ''),
                    'grant_summary':   row.get('grant_summary',   ''),
                }))

        total_tasks   = len(unique_tasks)
        idx_to_result: dict[int, str] = {}

        for i in range(0, total_tasks, _VALIDATION_BATCH):
            batch   = unique_tasks[i : i + _VALIDATION_BATCH]
            results = asyncio.run(_validate_rows(batch, anth_key))
            idx_to_result.update(results)
            done = i + len(batch)
            status_placeholder.progress(
                seg_num / n_segments,
                text=f'{seg_label} · validating {done}/{total_tasks} new companies',
            )
            if done % 100 == 0:
                gc.collect()

        # Broadcast results for new sites and add to seen_websites
        for site, indices in website_map.items():
            result = idx_to_result.get(indices[0], 'no')
            seen_websites[site] = result
            for idx in indices:
                segment.at[idx, 'good_match'] = result
        # Rows without a website each got their own task entry
        for idx in segment.index:
            if segment.at[idx, 'good_match'] is None:
                segment.at[idx, 'good_match'] = idx_to_result.get(idx, 'no')

        del unique_tasks, website_map, idx_to_result
        gc.collect()

        segment = segment[
            segment['good_match'].str.contains('yes', na=False)
        ].reset_index(drop=True)

    if segment.empty:
        return segment

    # ── Pre-write email ───────────────────────────────────────────────────
    if prewrite_email:
        segment = segment.copy()
        segment['subject_line'] = None
        segment['ai_message']   = None

        email_rows = [(idx, row.to_dict()) for idx, row in segment.iterrows()]
        total      = len(email_rows)

        for i in range(0, total, _EMAIL_BATCH):
            batch   = email_rows[i : i + _EMAIL_BATCH]
            results = asyncio.run(_generate_email_batch(batch, openai_key, anth_key))
            for idx, subject, body in results:
                segment.at[idx, 'subject_line'] = subject
                segment.at[idx, 'ai_message']   = body
            done = i + len(batch)
            status_placeholder.progress(
                seg_num / n_segments,
                text=f'{seg_label} · writing emails {done}/{total}',
            )

        del email_rows
        gc.collect()

    return segment


# ── Session state ─────────────────────────────────────────────────────────────

for _k in ['bm_topics_df', 'bm_grant_embeddings', 'bm_run_summary']:
    if _k not in st.session_state:
        st.session_state[_k] = None
if 'bm_filters' not in st.session_state:
    st.session_state.bm_filters = [{'column': None, 'keyword': '', 'operator': 'AND'}]


# ── Page ──────────────────────────────────────────────────────────────────────

st.title('⚙️ Bulk Matching')
st.caption('Match grant-topic embeddings to contacts, validate with AI, and save results to GCS.')

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
        if 'grant_summary' not in _df.columns and 'description' in _df.columns:
            _df = _df.rename(columns={'description': 'grant_summary'})
        if 'embeddings' in _df.columns:
            st.session_state.bm_grant_embeddings = np.stack(_df['embeddings'].values).astype(np.float32)
            _df = _df.drop(columns=['embeddings'])
        else:
            st.session_state.bm_grant_embeddings = None
        st.session_state.bm_topics_df  = _df
        st.session_state.bm_filters    = [{'column': None, 'keyword': '', 'operator': 'AND'}]
        st.session_state.bm_run_summary = None
        st.success(f'Loaded **{len(_df):,}** grant topics.')

if st.session_state.bm_topics_df is not None:
    _topics_df      = st.session_state.bm_topics_df
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
    st.dataframe(_filtered_preview[_display_cols].head(50), width='stretch', hide_index=True)

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
        anth_key   = st.secrets['anthropic_api_key']
        openai_key = st.secrets.get('openai_api_key', '')

        # ── Prepare filtered grant matrix ─────────────────────────────────
        grants = _apply_filters(st.session_state.bm_topics_df, st.session_state.bm_filters)
        if grants.empty:
            st.error('No grant topics after applying filters.')
            st.stop()

        all_embeddings   = st.session_state.bm_grant_embeddings
        filtered_indices = grants.index.tolist()
        grant_embeddings = all_embeddings[filtered_indices]
        del all_embeddings

        if grant_embeddings.shape[0] == 0:
            st.error('All grant topics have null embeddings after filtering.')
            st.stop()

        grant_cols = ['topic_number', 'title', 'agency', 'broad_agency', 'due_date', 'grant_summary']
        grant_meta = grants[[c for c in grant_cols if c in grants.columns]].reset_index(drop=True)

        # ── Stream contacts → collect match rows ──────────────────────────
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

            if 'company_summary' not in df.columns and 'summary' in df.columns:
                df = df.rename(columns={'summary': 'company_summary'})

            contact_embeddings = np.stack(df['embeddings'].values).astype(np.float32)
            scores = np.dot(contact_embeddings, grant_embeddings.T)

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
                    }
                    for col in grant_meta.columns:
                        row[col] = grant_meta.iloc[gi].get(col, '')
                    match_rows.append(row)

            del contact_embeddings, scores
            bar.progress(
                (file_i + 1) / len(blob_list),
                text=f'File {file_i + 1}/{len(blob_list)}: {blob.name.split("/")[-1]}',
            )

        bar.empty()
        del grant_embeddings, grant_meta

        if not match_rows:
            st.warning(f'No matches found above {threshold} similarity threshold.')
            st.stop()

        total_candidates = len(match_rows)
        st.success(f'Found **{total_candidates:,}** candidate matches — processing in segments of {_SEGMENT_SIZE:,}.')

        # ── Process in 1 000-row segments ─────────────────────────────────
        run_id         = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_prefix = f'{_RESULTS_PREFIX}{run_id}/'
        gcs_client     = _get_storage_client()
        n_segments     = (total_candidates + _SEGMENT_SIZE - 1) // _SEGMENT_SIZE
        seg_status     = st.progress(0, text='Starting segments…')
        saved_segments: list[dict] = []
        seen_websites:  dict[str, str] = {}  # shared across segments for dedup

        for seg_i in range(n_segments):
            seg_num   = seg_i + 1
            seg_start = seg_i * _SEGMENT_SIZE
            seg_end   = min(seg_start + _SEGMENT_SIZE, total_candidates)

            # Build segment DataFrame and immediately free the source rows
            segment_rows = match_rows[seg_start:seg_end]
            segment      = pd.DataFrame(segment_rows)
            del segment_rows

            # Cast date columns
            for _col in ('scraped_at', 'open_date', 'close_date'):
                if _col in segment.columns:
                    segment[_col] = segment[_col].astype(str)

            seg_status.progress(
                seg_num / n_segments,
                text=f'Segment {seg_num}/{n_segments} · {len(segment):,} rows',
            )

            segment = _process_segment(
                segment        = segment,
                seg_num        = seg_num,
                n_segments     = n_segments,
                ai_validation  = ai_validation,
                prewrite_email = prewrite_email,
                anth_key       = anth_key,
                openai_key     = openai_key,
                status_placeholder = seg_status,
                seen_websites  = seen_websites,
            )

            if not segment.empty:
                blob_path = f'{results_prefix}segment_{seg_num:03d}.csv'
                _upload_csv(segment, blob_path, gcs_client)
                saved_segments.append({'path': blob_path, 'rows': len(segment)})

            del segment
            gc.collect()

        seg_status.empty()
        del match_rows
        gc.collect()

        total_saved = sum(s['rows'] for s in saved_segments)
        st.session_state.bm_run_summary = {
            'run_id':    run_id,
            'prefix':    results_prefix,
            'segments':  saved_segments,
            'total_rows': total_saved,
        }
        st.success(
            f'Run complete — **{total_saved:,}** rows saved across '
            f'**{len(saved_segments)}** segment(s) to `{results_prefix}`'
        )

    except Exception as e:
        st.error(f'**Error:** {e}')
        st.code(traceback.format_exc())

# ── Section 4 · Results ───────────────────────────────────────────────────────

if st.session_state.bm_run_summary is not None:
    summary = st.session_state.bm_run_summary
    st.divider()
    st.subheader('4 · Results')
    st.caption(f'Run ID: `{summary["run_id"]}` · GCS prefix: `{summary["prefix"]}`')

    if not summary['segments']:
        st.info('No segments were saved (all candidates filtered out).')
    else:
        seg_df = pd.DataFrame(summary['segments'])
        seg_df.index = seg_df.index + 1
        seg_df.index.name = 'Segment'
        st.dataframe(seg_df.rename(columns={'path': 'GCS path', 'rows': 'Rows'}), hide_index=False)
        st.caption(f'Total rows saved: **{summary["total_rows"]:,}**')
