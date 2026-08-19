"""
Bulk Aspect Match
-----------------
Matches the whole client directory against grant topics using the
multi-aspect profiles built in the Client Profiles view.

Each client carries several independently embedded aspects. Every aspect is
scored against every selected grant topic (one numpy matmul per client), a
topic becomes a candidate for that client when enough aspects clear the
similarity threshold, and the top candidates per client are then re-ranked
1-5 by Claude with the matched aspect as context.

Scoring and re-ranking run in this process — keep the page open while a run
is in flight. Results are held in session state, downloadable as CSV, and
written to aspect-match-results/{run_id}/results.csv in GCS.
"""

import asyncio
import io
import json
import random
import re
import traceback
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from anthropic import AsyncAnthropic
from google.cloud import storage
from google.oauth2 import service_account

import src.modules.aspect_profile as ap
from src.modules.grant_utils import normalize_grant_columns

# ── Constants ──────────────────────────────────────────────────────────────

_BUCKET         = ap.BUCKET
_TOPICS_PREFIX  = 'data/all-topics/processed/'
_RESULTS_PREFIX = 'aspect-match-results/'

_RERANK_MODELS = ['claude-haiku-4-5-20251001', 'claude-sonnet-4-6']
_CONCURRENCY   = 15
_MAX_RETRIES   = 5
_CONFIRM_PAIRS = 2500   # above this many re-rank calls, require a confirmation

# Topic columns carried into the results, when present.
_TOPIC_COLS = [
    'topic_number', 'title', 'agency', 'broad_agency', 'due_date', 'close_date',
    'open_date', 'funding_amount', 'grant_summary', 'source',
]


# ── GCS ────────────────────────────────────────────────────────────────────

def _get_storage_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)


def _list_agencies(client: storage.Client) -> list[str]:
    try:
        blobs = client.list_blobs(_BUCKET, prefix=_TOPICS_PREFIX, delimiter='/')
        list(blobs)
        return sorted(p.replace(_TOPICS_PREFIX, '').strip('/') for p in blobs.prefixes)
    except Exception as e:
        st.error(f'Failed to list agencies: {e}')
        return []


def _load_topics(client: storage.Client, agencies: list[str]) -> pd.DataFrame:
    frames = []
    for agency in agencies:
        for blob in client.list_blobs(_BUCKET, prefix=f'{_TOPICS_PREFIX}{agency}/'):
            if blob.name.endswith('.parquet'):
                df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
                df['broad_agency'] = agency
                frames.append(df)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        topics = pd.concat(frames, ignore_index=True)
    # Notices marked archived by the SAM.gov revision check are no longer live
    if 'sam_status' in topics.columns:
        topics = topics[topics['sam_status'].fillna('').astype(str) != 'archived']
    topics = normalize_grant_columns(topics.reset_index(drop=True))
    # Stored as float64; halved to float32 here because the whole topic store
    # is held in session state alongside the scoring matrix. This frame is
    # never written back, so the narrower dtype is local to the view.
    if 'embeddings' in topics.columns:
        topics['embeddings'] = topics['embeddings'].map(
            lambda e: np.asarray(e, dtype=np.float32)
            if isinstance(e, (list, np.ndarray)) else e
        )
    return topics


# ── Filters (same behaviour as Grant Search / Bulk Matching) ───────────────

_DATE_FORMATS = ('%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%m-%d-%Y', '%b %d, %Y')


def _parse_date_str(value) -> date | None:
    s = str(value).strip()
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        for candidate in (s, s[:10]):
            try:
                return datetime.strptime(candidate, fmt).date()
            except ValueError:
                pass
    return None


def _filter_is_active(f: dict) -> bool:
    if not f.get('column'):
        return False
    if f.get('type') == 'date_range':
        return bool(f.get('date_from') and f.get('date_to'))
    return bool(f.get('keyword', '').strip())


def _filter_mask(df: pd.DataFrame, f: dict) -> pd.Series:
    if f.get('type') == 'date_range':
        d_from, d_to = f['date_from'], f['date_to']

        def _in_range(v) -> bool:
            d = _parse_date_str(v)
            return d is not None and d_from <= d <= d_to

        return df[f['column']].map(_in_range)
    return df[f['column']].astype(str).str.lower().str.contains(
        f['keyword'].lower(), na=False
    )


def _apply_filters(df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    active = [f for f in filters if _filter_is_active(f)]
    if not active:
        return df
    mask = _filter_mask(df, active[0])
    for f in active[1:]:
        m = _filter_mask(df, f)
        mask = (mask & m) if f['operator'] == 'AND' else (mask | m)
    return df[mask]


# ── Scoring ────────────────────────────────────────────────────────────────

def _stack_topic_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """(T, dim) float32 matrix + the topic rows it corresponds to. Rows with a
    missing or wrong-length vector are dropped."""
    keep, vecs = [], []
    for idx, emb in df['embeddings'].items():
        if isinstance(emb, (list, np.ndarray)) and len(emb) == ap.EMBED_DIM:
            keep.append(idx)
            vecs.append(np.asarray(emb, dtype=np.float32))
    if not vecs:
        return np.zeros((0, ap.EMBED_DIM), dtype=np.float32), df.iloc[0:0]
    meta = df.loc[keep].drop(columns=['embeddings'], errors='ignore').reset_index(drop=True)
    return np.vstack(vecs), meta


def _match_clients(
    selected: pd.DataFrame,
    topic_matrix: np.ndarray,
    topic_meta: pd.DataFrame,
    threshold: float,
    min_hits: int,
    top_k: int,
    progress=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Candidate (client, topic) rows: a topic qualifies for a client when at
    least `min_hits` of the client's aspects clear `threshold`; the top `top_k`
    per client by best aspect score are kept."""
    meta_cols = [c for c in _TOPIC_COLS if c in topic_meta.columns]
    # Materialised once — .iloc[ti] per candidate builds a Series per row and
    # dominates the loop on large runs.
    meta_records = topic_meta[meta_cols].to_dict('records')
    rows: list[dict] = []
    skipped: list[str] = []
    total = len(selected)

    for n, (_, prof) in enumerate(selected.iterrows(), 1):
        if progress is not None:
            progress.progress(n / total, text=f'Scoring {prof["company_name"]} ({n}/{total})')

        aspects = ap.profile_aspects(prof)
        matrix  = ap.unpack_embeddings(prof)
        if matrix.shape[0] == 0 or len(aspects) != matrix.shape[0]:
            skipped.append(
                f'{prof["company_name"]}: profile has no usable aspect vectors — rebuild it'
            )
            continue

        scores = matrix @ topic_matrix.T          # (n_aspects, T)
        best_i = scores.argmax(axis=0)
        best   = scores.max(axis=0)
        hits   = (scores >= threshold).sum(axis=0)

        qualified = np.where((best >= threshold) & (hits >= min_hits))[0]
        if qualified.size == 0:
            continue
        order = qualified[np.argsort(best[qualified])[::-1][:top_k]]

        for ti in order:
            ai     = int(best_i[ti])
            aspect = aspects[ai]
            row = {
                'client':           prof['company_name'],
                'client_website':   prof['companyWebsite'],
                'aspect_label':     aspect.get('label', ''),
                'aspect_kind':      aspect.get('kind', ''),
                'aspect_score':     round(float(best[ti]), 4),
                'aspects_hit':      int(hits[ti]),
                'aspects_total':    len(aspects),
                'aspect_scores':    json.dumps({
                    aspects[j].get('label', f'aspect_{j + 1}'): round(float(scores[j, ti]), 4)
                    for j in range(len(aspects))
                }),
                '_company_key':     prof['company_key'],
                '_aspect_text':     aspect.get('text', ''),
                '_profile_summary': prof['profile_summary'],
            }
            row.update(meta_records[int(ti)])
            rows.append(row)

    return pd.DataFrame(rows), skipped


# ── LLM re-rank ────────────────────────────────────────────────────────────

_RERANK_SYSTEM = (
    'You are screening a federal grant topic against one specific capability of a '
    'company that a proposal-writing firm represents.\n'
    'Score the fit from 1 to 5:\n'
    '5 = the company could propose to this topic directly with the capability described\n'
    '4 = strong fit with minor gaps\n'
    '3 = plausible fit, but notable gaps or adaptation needed\n'
    '2 = superficial or keyword-level overlap only\n'
    '1 = no fit\n\n'
    'Judge only the capability as described — never assume capabilities that are not stated.\n'
    'Return ONLY valid JSON: {"score": <integer 1-5>, "rationale": "<one sentence>"}'
)


def _rerank_user_message(row: dict) -> str:
    return (
        f"Company: {row.get('client', '')}\n"
        f"Company profile: {str(row.get('_profile_summary') or '')[:1500]}\n\n"
        f"Matched capability — {row.get('aspect_label', '')}:\n"
        f"{str(row.get('_aspect_text') or '')[:2000]}\n\n"
        f"Grant topic: {row.get('title', '')}\n"
        f"Agency: {row.get('agency', '') or row.get('broad_agency', '')}\n"
        f"Topic description:\n{str(row.get('grant_summary') or '')[:6000]}"
    )


def _parse_rerank(text: str) -> tuple[int | None, str]:
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', (text or '').strip())
    start, end = cleaned.find('{'), cleaned.rfind('}')
    if start != -1 and end > start:
        try:
            obj = json.loads(cleaned[start:end + 1])
            score = int(float(obj.get('score')))
            return max(1, min(5, score)), str(obj.get('rationale') or '')[:400]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # Tolerate a stray sentence around the JSON rather than losing the score
    m = re.search(r'"?score"?\s*[:=]\s*([1-5])', cleaned)
    if m:
        r = re.search(r'"?rationale"?\s*[:=]\s*"([^"]*)"', cleaned)
        return int(m.group(1)), (r.group(1)[:400] if r else '')
    return None, '(unparseable response)'


async def _rerank_async(
    rows: list[tuple[int, dict]], api_key: str, model: str, on_done
) -> list[tuple[int, int | None, str]]:
    sem = asyncio.Semaphore(_CONCURRENCY)

    async with AsyncAnthropic(api_key=api_key) as client:
        async def one(idx: int, row: dict) -> tuple[int, int | None, str]:
            async with sem:
                for attempt in range(_MAX_RETRIES):
                    try:
                        resp = await client.messages.create(
                            model=model,
                            max_tokens=250,
                            temperature=0,
                            system=_RERANK_SYSTEM,
                            messages=[{'role': 'user', 'content': _rerank_user_message(row)}],
                        )
                        score, rationale = _parse_rerank(resp.content[0].text)
                        return idx, score, rationale
                    except Exception as e:
                        err = str(e)
                        retryable = any(
                            x in err for x in
                            ('429', '529', 'overloaded', 'rate_limit', 'rate limit', 'timeout')
                        )
                        if retryable and attempt < _MAX_RETRIES - 1:
                            await asyncio.sleep((2 ** attempt) + random.random())
                            continue
                        return idx, None, f'(scoring failed: {type(e).__name__})'
                return idx, None, '(scoring failed: retries exhausted)'

        tasks   = [asyncio.create_task(one(i, r)) for i, r in rows]
        results = []
        for fut in asyncio.as_completed(tasks):
            results.append(await fut)
            on_done(len(results))
        return results


def _run_rerank(candidates: pd.DataFrame, api_key: str, model: str) -> pd.DataFrame:
    rows  = [(int(i), r.to_dict()) for i, r in candidates.iterrows()]
    total = len(rows)
    prog  = st.progress(0.0, text=f'LLM re-ranking 0/{total}…')

    def on_done(done: int) -> None:
        prog.progress(done / total, text=f'LLM re-ranking {done}/{total}…')

    results = asyncio.run(_rerank_async(rows, api_key, model, on_done))
    prog.empty()

    out = candidates.copy()
    out['llm_score']     = 0
    out['llm_rationale'] = ''
    for idx, score, rationale in results:
        # 0 keeps unscored pairs visible but below any usable minimum
        out.at[idx, 'llm_score']     = int(score) if score is not None else 0
        out.at[idx, 'llm_rationale'] = rationale
    return out


# ── Results output ─────────────────────────────────────────────────────────

_DISPLAY_FIRST = [
    'client', 'aspect_label', 'aspect_score', 'aspects_hit', 'aspects_total',
    'llm_score', 'llm_rationale', 'topic_number', 'title', 'agency', 'broad_agency',
]


def _display_frame(df: pd.DataFrame) -> pd.DataFrame:
    internal = [c for c in df.columns if c.startswith('_')]
    first    = [c for c in _DISPLAY_FIRST if c in df.columns]
    rest     = [c for c in df.columns if c not in first and c not in internal]
    return df[first + rest]


def _save_results(client: storage.Client, run_id: str, df: pd.DataFrame) -> str:
    path = f'{_RESULTS_PREFIX}{run_id}/results.csv'
    client.bucket(_BUCKET).blob(path).upload_from_string(
        df.to_csv(index=False).encode('utf-8'), content_type='text/csv'
    )
    return path


# ── Session state ──────────────────────────────────────────────────────────

for _k in ['am_profiles', 'am_topics_df', 'am_results', 'am_run_meta']:
    if _k not in st.session_state:
        st.session_state[_k] = None
if 'am_filters' not in st.session_state:
    st.session_state.am_filters = [{'column': None, 'type': 'keyword', 'keyword': '', 'operator': 'AND'}]


# ── Page ───────────────────────────────────────────────────────────────────

st.title('🎯 Bulk Aspect Match')
st.caption(
    'Score every client aspect against every selected grant topic, keep the '
    'topics that clear the threshold, then have Claude re-rank the survivors.'
)

gcs = _get_storage_client()

if st.session_state.am_profiles is None:
    with st.spinner('Loading client profiles…'):
        try:
            st.session_state.am_profiles = ap.load_profiles(gcs)
        except Exception as e:
            st.error(f'Could not load {ap.PROFILES_BLOB}: {e}')
            st.stop()

profiles: pd.DataFrame = st.session_state.am_profiles

if profiles.empty:
    st.warning(
        'No client profiles found. Build them in the **Client Profiles** view first.'
    )
    st.stop()

# ── Section 1 · Clients ────────────────────────────────────────────────────

st.subheader('1 · Select clients')

head_l, head_r = st.columns([1, 5])
with head_l:
    if st.button('↺ Reload profiles'):
        st.session_state.am_profiles = None
        st.rerun()
with head_r:
    st.info(
        f'{len(profiles):,} profiled client'
        f'{"s" if len(profiles) != 1 else ""} · '
        f'{int(pd.to_numeric(profiles["n_aspects"], errors="coerce").fillna(0).sum()):,} aspects total'
    )

picker = pd.DataFrame({
    'use':      True,
    'client':   profiles['company_name'].fillna('—').astype(str),
    'aspects':  pd.to_numeric(profiles['n_aspects'], errors='coerce').fillna(0).astype(int),
    'labels':   profiles['aspect_labels'].fillna('').astype(str),
    'built_at': profiles['built_at'].fillna('').astype(str),
})

edited = st.data_editor(
    picker,
    hide_index=True,
    use_container_width=True,
    height=min(400, 60 + 36 * len(picker)),
    disabled=['client', 'aspects', 'labels', 'built_at'],
    column_config={
        'use':      st.column_config.CheckboxColumn('Use'),
        'client':   st.column_config.TextColumn('Client'),
        'aspects':  st.column_config.NumberColumn('Aspects', format='%d'),
        'labels':   st.column_config.TextColumn('Aspect labels', width='large'),
        'built_at': st.column_config.TextColumn('Built'),
    },
    key='am_client_picker',
)

selected = profiles.loc[edited.index[edited['use'].fillna(False).to_numpy(dtype=bool)]]
if selected.empty:
    st.warning('Select at least one client.')

max_aspects = int(pd.to_numeric(selected['n_aspects'], errors='coerce').fillna(0).max()) if not selected.empty else 1

# ── Section 2 · Grant topics ───────────────────────────────────────────────

st.divider()
st.subheader('2 · Select grant topics')

agencies = _list_agencies(gcs)
if not agencies:
    st.warning('No grant agencies found in GCS.')
    st.stop()

ag_cols = st.columns(min(len(agencies), 6))
selected_agencies = [
    ag for i, ag in enumerate(agencies)
    if ag_cols[i % len(ag_cols)].checkbox(ag, value=True, key=f'am_agency_{ag}')
]

if st.button('Load Topics', type='primary', disabled=not selected_agencies):
    with st.spinner(f'Loading topics from {len(selected_agencies)} agenc'
                    f'{"y" if len(selected_agencies) == 1 else "ies"}…'):
        topics = _load_topics(gcs, selected_agencies)
    if topics.empty:
        st.warning('No topics found for the selected agencies.')
    else:
        st.session_state.am_topics_df = topics
        st.session_state.am_results   = None
        st.session_state.am_filters   = [{'column': None, 'type': 'keyword', 'keyword': '', 'operator': 'AND'}]
        st.success(f'Loaded **{len(topics):,}** topics.')

if st.session_state.am_topics_df is None:
    st.stop()

topics_df       = st.session_state.am_topics_df
filterable_cols = [c for c in topics_df.columns if c != 'embeddings']

for f in st.session_state.am_filters:
    if f['column'] not in filterable_cols:
        f['column'] = filterable_cols[0] if filterable_cols else None

for i, f in enumerate(st.session_state.am_filters):
    if i == 0:
        col_sel, mode_col, val_input, remove_col = st.columns([2, 1.4, 3, 0.5])
    else:
        op_col, col_sel, mode_col, val_input, remove_col = st.columns([1, 2, 1.4, 3, 0.5])
        f['operator'] = op_col.radio(
            'op', ['AND', 'OR'], index=0 if f['operator'] == 'AND' else 1,
            key=f'am_op_{i}', horizontal=True, label_visibility='collapsed',
        )

    f['column'] = col_sel.selectbox(
        'Column', filterable_cols,
        index=filterable_cols.index(f['column']) if f['column'] in filterable_cols else 0,
        key=f'am_col_{i}', label_visibility='collapsed',
    )
    mode_label = mode_col.selectbox(
        'Filter type', ['Keyword', 'Date range'],
        index=1 if f.get('type') == 'date_range' else 0,
        key=f'am_type_{i}', label_visibility='collapsed',
    )
    f['type'] = 'date_range' if mode_label == 'Date range' else 'keyword'
    if f['type'] == 'date_range':
        picked = val_input.date_input(
            'Date range',
            value=(
                f.get('date_from') or date.today() - timedelta(days=30),
                f.get('date_to') or date.today(),
            ),
            key=f'am_dr_{i}', label_visibility='collapsed',
        )
        if isinstance(picked, tuple) and len(picked) == 2:
            f['date_from'], f['date_to'] = picked
        elif isinstance(picked, tuple) and len(picked) == 1:
            # Mid-selection: only the start date is chosen so far.
            f['date_from'], f['date_to'] = picked[0], None
    else:
        f['keyword'] = val_input.text_input(
            'Keyword', value=f.get('keyword', ''),
            placeholder=f'Filter by {f["column"]}…',
            key=f'am_kw_{i}', label_visibility='collapsed',
        )
    if remove_col.button('✕', key=f'am_rm_{i}', disabled=len(st.session_state.am_filters) == 1):
        st.session_state.am_filters.pop(i)
        st.rerun()

if st.button('+ Add filter', key='am_add_filter'):
    st.session_state.am_filters.append(
        {'column': filterable_cols[0], 'type': 'keyword', 'keyword': '', 'operator': 'AND'}
    )
    st.rerun()

filtered = _apply_filters(topics_df, st.session_state.am_filters)
st.caption(f'**{len(filtered):,}** topics match current filters — showing first 25')
st.dataframe(filtered[filterable_cols].head(25), use_container_width=True, hide_index=True)

# ── Section 3 · Match options ──────────────────────────────────────────────

st.divider()
st.subheader('3 · Match options')

o1, o2, o3 = st.columns(3)
with o1:
    threshold = st.slider('Aspect similarity threshold', 0.60, 0.95, 0.78, 0.01)
with o2:
    min_hits = st.number_input(
        'Aspects that must clear it', min_value=1, max_value=max(1, max_aspects), value=1, step=1,
        help='1 = any single capability matching is enough (recommended — a client\'s '
             'aspects are different capabilities, not requirements of one query). '
             'Raise it to demand topics that touch several of the client\'s capabilities.',
    )
with o3:
    top_k = st.number_input('Top topics per client', min_value=1, max_value=100, value=10, step=1)

r1, r2, r3 = st.columns(3)
with r1:
    do_rerank = st.checkbox('LLM re-rank', value=True)
with r2:
    rerank_model = st.selectbox('Re-rank model', _RERANK_MODELS, index=0, disabled=not do_rerank)
with r3:
    min_llm = st.number_input(
        'Keep LLM score ≥', min_value=1, max_value=5, value=3, step=1, disabled=not do_rerank,
    )

max_pairs = len(selected) * int(top_k)
if do_rerank:
    st.caption(
        f'Up to **{max_pairs:,}** re-rank calls '
        f'({len(selected)} clients × top {int(top_k)}), {_CONCURRENCY} at a time.'
    )
confirm = True
if do_rerank and max_pairs > _CONFIRM_PAIRS:
    confirm = st.checkbox(
        f'I understand this can make up to {max_pairs:,} Claude calls and the page '
        'must stay open until it finishes.',
        value=False,
    )

run = st.button(
    '▶ Run match', type='primary',
    disabled=selected.empty or filtered.empty or not confirm,
)

# ── Run ────────────────────────────────────────────────────────────────────

if run:
    with st.spinner('Preparing topic vectors…'):
        topic_matrix, topic_meta = _stack_topic_embeddings(filtered)

    if topic_matrix.shape[0] == 0:
        st.error('None of the filtered topics carry a usable embedding.')
        run = False
    elif len(topic_meta) < len(filtered):
        st.warning(
            f'{len(filtered) - len(topic_meta):,} topic(s) skipped — missing or '
            'malformed embedding.'
        )

if run:
    # st.stop() raises, so it must not be used inside this handler — every
    # failure path below reports and falls through to the results section.
    try:
        prog = st.progress(0.0, text='Scoring…')
        candidates, skipped = _match_clients(
            selected, topic_matrix, topic_meta,
            float(threshold), int(min_hits), int(top_k), prog,
        )
        prog.empty()
        del topic_matrix

        for msg in skipped:
            st.warning(msg)

        if candidates.empty:
            st.session_state.am_results = candidates
            st.session_state.am_run_meta = {
                'run_id': None, 'threshold': float(threshold), 'min_hits': int(min_hits),
                'top_k': int(top_k), 'clients': len(selected), 'topics': len(topic_meta),
                'candidates': 0, 'reranked': False, 'kept': 0, 'unscored': 0, 'gcs_path': None,
            }
        else:
            reranked = False
            unscored = 0
            results  = candidates
            if do_rerank:
                results  = _run_rerank(candidates, st.secrets['anthropic_api_key'], rerank_model)
                unscored = int((results['llm_score'] == 0).sum())
                reranked = True
                results  = results[results['llm_score'] >= int(min_llm)]
                results  = results.sort_values(
                    ['llm_score', 'aspect_score'], ascending=[False, False]
                ).reset_index(drop=True)
            else:
                results = results.sort_values(
                    ['aspect_score', 'aspects_hit'], ascending=[False, False]
                ).reset_index(drop=True)

            run_id = f'aspect_match_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
            gcs_path = None
            if not results.empty:
                try:
                    gcs_path = _save_results(gcs, run_id, _display_frame(results))
                except Exception as e:
                    st.warning(f'Results not saved to GCS ({e}) — the CSV download still works.')

            st.session_state.am_results  = results
            st.session_state.am_run_meta = {
                'run_id': run_id, 'threshold': float(threshold), 'min_hits': int(min_hits),
                'top_k': int(top_k), 'clients': len(selected), 'topics': len(topic_meta),
                'candidates': len(candidates), 'reranked': reranked,
                'kept': len(results), 'unscored': unscored, 'gcs_path': gcs_path,
            }

    except Exception as e:
        st.error(f'Match run failed: {e}')
        st.code(traceback.format_exc())

# ── Section 4 · Results ────────────────────────────────────────────────────

if st.session_state.am_results is not None:
    results = st.session_state.am_results
    meta    = st.session_state.am_run_meta or {}

    st.divider()
    st.subheader('4 · Results')

    if results.empty:
        st.warning(
            f'No topics cleared {meta.get("threshold")} on at least '
            f'{meta.get("min_hits")} aspect(s)'
            + (' and survived re-ranking.' if meta.get('reranked') else '.')
            + ' Try lowering the threshold or widening the topic selection.'
        )
    else:
        m = st.columns(4)
        m[0].metric('Rows', f'{len(results):,}')
        m[1].metric('Clients matched', f'{results["client"].nunique():,}')
        m[2].metric('Candidates scored', f'{meta.get("candidates", 0):,}')
        m[3].metric('Topics searched', f'{meta.get("topics", 0):,}')

        if meta.get('unscored'):
            st.warning(
                f'{meta["unscored"]} pair(s) could not be scored by the re-ranker '
                '(shown as score 0 and dropped by the minimum score).'
            )
        if meta.get('gcs_path'):
            st.caption(f'Saved to `{meta["gcs_path"]}`')

        display = _display_frame(results)
        col_cfg = {
            'client':        st.column_config.TextColumn('Client'),
            'aspect_label':  st.column_config.TextColumn('Matched aspect'),
            'aspect_score':  st.column_config.NumberColumn('Aspect score', format='%.4f'),
            'aspects_hit':   st.column_config.NumberColumn('Aspects hit', format='%d'),
            'aspects_total': st.column_config.NumberColumn('Aspects', format='%d'),
        }
        if 'llm_score' in display.columns:
            col_cfg['llm_score']     = st.column_config.NumberColumn('LLM score', format='%d')
            col_cfg['llm_rationale'] = st.column_config.TextColumn('Rationale', width='large')

        st.dataframe(display, use_container_width=True, hide_index=True, column_config=col_cfg)

        st.download_button(
            '⬇ Download CSV',
            data=display.to_csv(index=False).encode('utf-8'),
            file_name=f'{meta.get("run_id") or "aspect_match"}.csv',
            mime='text/csv',
        )

        with st.expander('Per-client summary'):
            agg = {
                'topics':            ('client', 'size'),
                'best_aspect_score': ('aspect_score', 'max'),
            }
            if 'llm_score' in results.columns:
                agg['best_llm_score'] = ('llm_score', 'max')
            summary = (
                results.groupby('client')
                .agg(**agg)
                .reset_index()
                .sort_values('topics', ascending=False)
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)
