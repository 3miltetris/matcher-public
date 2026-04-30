"""
SAM.gov Upload
--------------
Two input modes:
  - Upload CSV     : Upload SAM.gov contract opportunity CSVs (original behaviour).
  - Fetch from API : Query the SAM.gov Opportunities API directly by date range,
                     notice type, and optional keyword — no manual export needed.

In both modes Claude Haiku screens each row for relevance to deep tech / R&D small
businesses, then passing rows are embedded and saved to GCS under
data/all-topics/processed/SAM-GOV/.
"""

import io
import json
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st
from anthropic import Anthropic
from bs4 import BeautifulSoup
from google.cloud import storage
from google.oauth2 import service_account

from src.modules.Embedding.text_embedder import TextProcessor
from src.modules.GoogleBucketManager.bucket_manager import BucketManager

# ── GCS ────────────────────────────────────────────────────────────────────

_BUCKET     = 'cc-matcher-bucket-jeg-v1'
_SAM_PREFIX = 'data/all-topics/processed/SAM-GOV/'

_SCREEN_MODEL     = 'claude-haiku-4-5-20251001'
_SCREEN_WORKERS   = 8
_SCREEN_MAX_CHARS = 3000

# ── SAM.gov API ────────────────────────────────────────────────────────────

_SAM_API_BASE  = 'https://api.sam.gov/opportunities/v2/search'
_SAM_DESC_BASE = 'https://api.sam.gov/opportunities/v2/noticedesc'
_SAM_PAGE_SIZE = 1000
_DESC_WORKERS  = 10

_NOTICE_TYPE_OPTIONS = {
    'Presolicitation':                'p',
    'Solicitation':                   'o',
    'Combined Synopsis/Solicitation': 'k',
    'Sources Sought':                 'r',
    'Special Notice':                 's',
}


# ── Storage client ─────────────────────────────────────────────────────────

def _get_storage_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)


# ── Screening prompt ───────────────────────────────────────────────────────

_SCREEN_SYSTEM = """\
You are a grant opportunity screening filter for a consultancy that serves innovative startups, R&D companies, and deep tech small businesses.

You will be given a federal opportunity that has already been pre-filtered to R&D and contract notices. Your only job is to decide: should this opportunity be imported into our matching system?

Answer YES if the opportunity is a realistic fit for startups, small businesses, or deep tech R&D companies.
Answer NO if it is not.

---

OUR IDEAL CLIENTS:
- Startups and small businesses doing technical R&D
- Deep tech ventures (biotech, defense tech, energy, AI/ML, hardware, advanced manufacturing, etc.)
- SBIR/STTR-eligible companies
- Commercialization-stage innovators

IMPORT (YES) if any of the following are true:
- The opportunity is explicitly open to or designed for small businesses or startups
- The NAICS descriptor suggests a field where small R&D companies commonly operate
- The work requires genuine technical innovation, applied research, or novel development
- It is a vehicle type our clients pursue (SBIR, STTR, BAA, IDIQ with small business tracks, etc.)

DO NOT IMPORT (NO) if any of the following are true:
- The opportunity is clearly intended for universities, nonprofits, state/local governments, or large prime contractors with no realistic small business angle
- It is a workforce development, training, community services, or infrastructure project
- It is a generic service procurement with no meaningful R&D or innovation component (e.g., janitorial, facilities, staffing, administrative support)
- The NAICS descriptor is in a sector where deep tech startups almost never compete (e.g., construction, food service, transportation logistics)
- The description makes clear that prior large-scale program experience or clearances beyond typical small business reach are required

When uncertain, only import if the opportunity is clearly relevant. Do not import on weak signals alone.

---

OUTPUT FORMAT:
Respond only with valid JSON. No preamble, no markdown, no explanation outside the JSON.
{"import": true, "confidence": "high", "reason": "One or two sentences explaining the decision."}\
"""


# ── Summarization prompt ──────────────────────────────────────────────────

_SUMMARY_SYSTEM = """\
You are preparing a federal contract opportunity for semantic matching against startup and R&D company profiles.

Summarize the opportunity in 3–5 sentences. Focus exclusively on:
- The specific technical problem, research area, or capability being sought
- Key deliverables or desired technical outcomes
- Relevant domain, technology, or sector (e.g., AI/ML, biotech, defense electronics, advanced manufacturing)

Strip out all procurement boilerplate: FAR clauses, set-aside language, submission deadlines, page limits, administrative instructions, and points of contact. Write in plain technical language. If the description is already short and technical, return it as-is.\
"""


def _summarize_one(title: str, desc: str, anth_key: str) -> str:
    client   = Anthropic(api_key=anth_key)
    user_msg = f"Title: {title}\n\nDescription:\n{str(desc)[:5000]}"
    resp = client.messages.create(
        model=_SCREEN_MODEL,
        max_tokens=400,
        system=_SUMMARY_SYSTEM,
        messages=[{'role': 'user', 'content': user_msg}],
    )
    return resp.content[0].text.strip()


def _summarize_descriptions(titles: list[str], descs: list[str], anth_key: str) -> list[str]:
    results  = [''] * len(descs)
    progress = st.progress(0, text='Summarizing descriptions…')

    with ThreadPoolExecutor(max_workers=_SCREEN_WORKERS) as pool:
        futures = {
            pool.submit(_summarize_one, titles[i], descs[i], anth_key): i
            for i in range(len(descs))
        }
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception:
                results[i] = descs[i]
            done += 1
            progress.progress(done / len(descs), text=f'Summarizing… {done}/{len(descs)}')

    progress.empty()
    return results


# ── Column auto-detection (CSV path) ──────────────────────────────────────

_CANDIDATES: dict[str, list[str]] = {
    'title':        ['title', 'opportunity title', 'solicitation title'],
    'description':  ['description', 'synopsis', 'description/synopsis'],
    'naics_desc':   ['naics desc', 'naics description', 'naics_desc', 'naics_description'],
    'notice_id':    ['notice id', 'notice_id', 'solicitation #', 'sol #', 'solicitation number'],
    'agency':       ['department/ind.agency', 'department', 'agency', 'department name'],
    'posted_date':  ['posted date', 'post date', 'posted_date'],
    'deadline':     ['response deadline', 'response_deadline', 'deadline', 'close date'],
}


def _detect_col(columns: list[str], field: str) -> str | None:
    lower = {c.lower(): c for c in columns}
    for candidate in _CANDIDATES[field]:
        if candidate in lower:
            return lower[candidate]
    return None


# ── SAM.gov API helpers ────────────────────────────────────────────────────

def _fetch_one_desc(notice_id: str, api_key: str) -> str:
    """Fetch and strip HTML from a single SAM.gov opportunity description."""
    try:
        r = requests.get(
            _SAM_DESC_BASE,
            params={'noticeid': notice_id, 'api_key': api_key},
            timeout=20,
        )
        r.raise_for_status()
        content = r.text
        if '<' in content:
            soup = BeautifulSoup(content, 'html.parser')
            return soup.get_text(separator=' ', strip=True)
        return content.strip()
    except Exception:
        return ''


def _fetch_descriptions_batch(notice_ids: list[str], api_key: str) -> list[str]:
    results  = [''] * len(notice_ids)
    progress = st.progress(0, text='Fetching full descriptions…')
    with ThreadPoolExecutor(max_workers=_DESC_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one_desc, nid, api_key): i
            for i, nid in enumerate(notice_ids)
        }
        done = 0
        for future in as_completed(futures):
            i        = futures[future]
            results[i] = future.result()
            done    += 1
            progress.progress(done / len(notice_ids), text=f'Fetching descriptions… {done}/{len(notice_ids)}')
    progress.empty()
    return results


def _search_sam(
    api_key:      str,
    date_from:    str,
    date_to:      str,
    notice_types: list[str],
    keyword:      str,
    max_results:  int,
) -> tuple[list[dict], int]:
    """Paginated SAM.gov opportunity search. Returns (items, total_records)."""
    all_items: list[dict] = []
    offset = 0
    total  = None

    with st.spinner('Querying SAM.gov…'):
        while True:
            params: dict = {
                'api_key':    api_key,
                'postedFrom': date_from,
                'postedTo':   date_to,
                'limit':      _SAM_PAGE_SIZE,
                'offset':     offset,
                'active':     'Yes',
            }
            if notice_types:
                params['ntype'] = ','.join(notice_types)
            if keyword:
                params['keyword'] = keyword

            r = requests.get(_SAM_API_BASE, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            if total is None:
                total = int(data.get('totalRecords', 0))

            page_items = data.get('opportunitiesData') or []
            all_items.extend(page_items)

            if not page_items:
                break
            if max_results and len(all_items) >= max_results:
                all_items = all_items[:max_results]
                break
            offset += len(page_items)
            if offset >= total:
                break

    return all_items, total or 0


def _items_to_df(items: list[dict], api_key: str, fetch_desc: bool) -> tuple[pd.DataFrame, dict]:
    """Convert raw SAM.gov API items to a DataFrame + col_map ready for screening."""
    notice_ids = [item.get('noticeId', '') for item in items]

    if fetch_desc and notice_ids:
        descriptions = _fetch_descriptions_batch(notice_ids, api_key)
    else:
        descriptions = ['' for _ in items]

    df = pd.DataFrame({
        'title':       [item.get('title', '')                                          for item in items],
        'description': descriptions,
        'naics_desc':  [item.get('naicsCode', '')                                      for item in items],
        'notice_id':   [(item.get('solicitationNumber') or item.get('noticeId', ''))   for item in items],
        'agency':      [(item.get('subTier') or item.get('department', ''))            for item in items],
        'posted_date': [item.get('postedDate', '')                                     for item in items],
        'deadline':    [(item.get('responseDeadLine') or '')[:10]                      for item in items],
    })

    col_map = {
        'title':       'title',
        'description': 'description',
        'naics_desc':  'naics_desc',
        'notice_id':   'notice_id',
        'agency':      'agency',
        'posted_date': 'posted_date',
        'deadline':    'deadline',
    }
    return df, col_map


# ── Screening ─────────────────────────────────────────────────────────────

def _screen_one(title: str, desc: str, naics: str, anth_key: str) -> dict:
    client   = Anthropic(api_key=anth_key)
    user_msg = (
        f"Title: {title}\n"
        f"Description: {str(desc)[:_SCREEN_MAX_CHARS]}\n"
        f"NAICS Descriptor: {naics}"
    )
    resp = client.messages.create(
        model=_SCREEN_MODEL,
        max_tokens=200,
        system=_SCREEN_SYSTEM,
        messages=[{'role': 'user', 'content': user_msg}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    return json.loads(raw)


def _run_screening(df: pd.DataFrame, col_map: dict, anth_key: str) -> pd.DataFrame:
    titles = df[col_map['title']].astype(str).tolist()
    descs  = df[col_map['description']].astype(str).tolist()
    naics  = df[col_map['naics_desc']].astype(str).tolist() if col_map.get('naics_desc') else [''] * len(df)

    results  = [None] * len(df)
    progress = st.progress(0, text='Screening rows…')

    with ThreadPoolExecutor(max_workers=_SCREEN_WORKERS) as pool:
        futures = {
            pool.submit(_screen_one, titles[i], descs[i], naics[i], anth_key): i
            for i in range(len(df))
        }
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                results[i] = {'import': False, 'confidence': 'low', 'reason': f'Screening error: {e}'}
            done += 1
            progress.progress(done / len(df), text=f'Screening rows… {done}/{len(df)}')

    progress.empty()

    out = df.copy()
    out['_import']     = [r['import']     for r in results]
    out['_confidence'] = [r['confidence'] for r in results]
    out['_reason']     = [r['reason']     for r in results]
    return out


# ── Existing-record dedup ──────────────────────────────────────────────────

def _load_existing_keys(client: storage.Client) -> tuple[set[str], set[str]]:
    """Return (notice_ids, lower_titles) already in the sam-gov prefix."""
    notice_ids: set[str] = set()
    titles:     set[str] = set()
    blobs = client.list_blobs(_BUCKET, prefix=_SAM_PREFIX)
    for blob in blobs:
        if not blob.name.endswith('.parquet'):
            continue
        try:
            df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()), columns=['topic_number', 'title'])
            notice_ids.update(df['topic_number'].dropna().astype(str).str.strip())
            titles.update(df['title'].dropna().astype(str).str.lower().str.strip())
        except Exception:
            pass
    return notice_ids, titles


# ── Embed + save ───────────────────────────────────────────────────────────

def _embed_and_save(df: pd.DataFrame, col_map: dict, oai_key: str, anth_key: str) -> str:
    tp    = TextProcessor(api_key=oai_key)
    bm    = BucketManager(_BUCKET, client=_get_storage_client())
    today = datetime.today().strftime('%Y-%m-%d')

    out = pd.DataFrame()
    out['topic_number'] = df[col_map['notice_id']].astype(str)   if col_map.get('notice_id')   else ''
    out['agency']       = df[col_map['agency']].astype(str)       if col_map.get('agency')      else 'SAM-GOV'
    out['title']        = df[col_map['title']].astype(str)
    out['description']  = df[col_map['description']].astype(str)
    out['open_date']    = df[col_map['posted_date']].astype(str)  if col_map.get('posted_date') else ''
    out['close_date']   = df[col_map['deadline']].astype(str)     if col_map.get('deadline')    else ''
    out['scraped_at']   = today
    out['sam_confidence'] = df['_confidence'].values
    out['sam_reason']   = df['_reason'].values

    titles    = out['title'].tolist()
    descs     = out['description'].tolist()
    summaries = _summarize_descriptions(titles, descs, anth_key)
    out['grant_summary'] = summaries

    embed_texts = [s if s.strip() else d for s, d in zip(summaries, descs)]
    progress    = st.progress(0, text='Generating embeddings…')
    embeddings  = []
    for i, text in enumerate(embed_texts):
        embeddings.append(tp.get_embedding(text) if text.strip() else None)
        progress.progress((i + 1) / len(embed_texts), text=f'Embedding {i + 1}/{len(embed_texts)}…')
    progress.empty()

    out['embeddings'] = embeddings

    hex_suffix = secrets.token_hex(3)
    gcs_path   = f'{_SAM_PREFIX}sam_gov_{today}_{hex_suffix}.parquet'
    bm.upload_file(gcs_path, out)
    return gcs_path


# ── Session state ──────────────────────────────────────────────────────────

for _k in ['sam_raw_df', 'sam_screened_df', 'sam_existing_keys', 'sam_col_map', 'sam_from_api']:
    if _k not in st.session_state:
        st.session_state[_k] = None


# ── Page ───────────────────────────────────────────────────────────────────

st.title('🏛️ SAM.gov Upload')
st.caption(
    'Screen and embed SAM.gov contract opportunities for the matching pipeline. '
    'Upload exported CSVs or fetch directly from the SAM.gov Opportunities API.'
)

# ── Section 1 · Load ───────────────────────────────────────────────────────

st.subheader('1 · Load opportunities')

tab_csv, tab_api = st.tabs(['📄 Upload CSV', '🔌 Fetch from API'])

# ── Tab: Upload CSV ────────────────────────────────────────────────────────

with tab_csv:
    files = st.file_uploader(
        'SAM.gov export CSV(s)',
        type='csv',
        accept_multiple_files=True,
        label_visibility='collapsed',
    )
    if files:
        frames = []
        for f in files:
            try:
                try:
                    frames.append(pd.read_csv(f, dtype=str, encoding='utf-8'))
                except UnicodeDecodeError:
                    f.seek(0)
                    frames.append(pd.read_csv(f, dtype=str, encoding='latin-1'))
            except Exception as e:
                st.error(f'Could not read **{f.name}**: {e}')
        if frames:
            combined = pd.concat(frames, ignore_index=True).dropna(how='all')
            if (
                st.session_state.sam_raw_df is None
                or st.session_state.sam_from_api
                or len(combined) != len(st.session_state.sam_raw_df)
            ):
                st.session_state.sam_raw_df        = combined
                st.session_state.sam_screened_df   = None
                st.session_state.sam_col_map       = None
                st.session_state.sam_from_api      = False
                st.session_state.sam_existing_keys = None

# ── Tab: Fetch from API ────────────────────────────────────────────────────

with tab_api:
    sam_key = st.secrets.get('sam_gov_api_key')
    if not sam_key:
        st.warning(
            'Add `sam_gov_api_key` to `.streamlit/secrets.toml` to use API fetch. '
            'Register for a free key at **beta.sam.gov → Account Settings → API Keys**.'
        )
    else:
        today_date   = datetime.today().date()
        default_from = today_date - timedelta(days=30)

        col_l, col_r = st.columns(2)
        with col_l:
            api_date_from = st.date_input('Posted from', value=default_from, key='sam_api_date_from')
        with col_r:
            api_date_to = st.date_input('Posted to', value=today_date, key='sam_api_date_to')

        selected_type_labels = st.multiselect(
            'Notice types',
            list(_NOTICE_TYPE_OPTIONS.keys()),
            default=['Solicitation', 'Presolicitation', 'Sources Sought'],
            key='sam_api_notice_types',
        )
        selected_type_codes = [_NOTICE_TYPE_OPTIONS[lbl] for lbl in selected_type_labels]

        api_keyword = st.text_input(
            'Keyword filter (optional)',
            placeholder='e.g. AI, biotech, cybersecurity',
            key='sam_api_keyword',
        )

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            api_max = st.number_input(
                'Max results (0 = no cap)',
                min_value=0,
                value=500,
                step=100,
                key='sam_api_max',
            )
        with col_r2:
            api_fetch_desc = st.checkbox(
                'Fetch full descriptions',
                value=True,
                key='sam_api_fetch_desc',
                help=(
                    'Makes one additional API call per opportunity to retrieve the full '
                    'synopsis text. Slower but significantly improves screening accuracy.'
                ),
            )

        if api_date_from > api_date_to:
            st.error('"Posted from" must be on or before "Posted to".')
        elif st.button('🔍 Fetch from SAM.gov', type='primary', key='sam_api_fetch_btn'):
            try:
                items, total = _search_sam(
                    api_key      = sam_key,
                    date_from    = api_date_from.strftime('%m/%d/%Y'),
                    date_to      = api_date_to.strftime('%m/%d/%Y'),
                    notice_types = selected_type_codes,
                    keyword      = api_keyword.strip(),
                    max_results  = int(api_max),
                )
                if not items:
                    st.warning(f'No active opportunities found for those filters (total records: {total:,}).')
                else:
                    st.caption(f'Found **{total:,}** total records; fetched **{len(items):,}**.')
                    df_api, col_map_api = _items_to_df(items, sam_key, bool(api_fetch_desc))
                    st.session_state.sam_raw_df        = df_api
                    st.session_state.sam_col_map       = col_map_api
                    st.session_state.sam_from_api      = True
                    st.session_state.sam_screened_df   = None
                    st.session_state.sam_existing_keys = None
                    st.rerun()
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else '?'
                if status == 403:
                    st.error('SAM.gov API returned 403 — check that your API key is correct and active.')
                elif status == 429:
                    st.error('SAM.gov API rate limit reached — wait a moment and try again.')
                else:
                    st.error(f'SAM.gov API HTTP {status}: {exc}')
            except Exception as exc:
                st.error(f'Fetch failed: {exc}')


if st.session_state.sam_raw_df is None:
    st.stop()

df_raw   = st.session_state.sam_raw_df
from_api = bool(st.session_state.sam_from_api)
src_label = 'API fetch' if from_api else (f'{len(files)} file(s)' if files else '(previously loaded)')
st.caption(f'**{len(df_raw):,}** rows loaded from {src_label}.')
st.dataframe(df_raw.head(5), hide_index=True, use_container_width=True)


# ── Section 2 · Column mapping (CSV path only) ─────────────────────────────

if not from_api:
    st.divider()
    st.subheader('2 · Column mapping')
    st.caption('Confirm which columns map to each field — auto-detected where possible.')

    cols     = df_raw.columns.tolist()
    none_opt = '— none —'
    col_opts = [none_opt] + cols

    def _sel(field: str, label: str, required: bool = False) -> str | None:
        detected = _detect_col(cols, field)
        idx      = col_opts.index(detected) if detected in col_opts else 0
        val      = st.selectbox(
            label + (' *' if required else ''),
            col_opts,
            index=idx,
            key=f'sam_map_{field}',
        )
        return val if val != none_opt else None

    left_col, right_col = st.columns(2)
    with left_col:
        m_title  = _sel('title',       'Title',              required=True)
        m_desc   = _sel('description', 'Description',        required=True)
        m_naics  = _sel('naics_desc',  'NAICS Descriptor')
        m_notice = _sel('notice_id',   'Notice ID')
    with right_col:
        m_agency = _sel('agency',      'Agency / Department')
        m_posted = _sel('posted_date', 'Posted Date')
        m_dl     = _sel('deadline',    'Response Deadline')

    col_map = {
        'title':       m_title,
        'description': m_desc,
        'naics_desc':  m_naics,
        'notice_id':   m_notice,
        'agency':      m_agency,
        'posted_date': m_posted,
        'deadline':    m_dl,
    }

    if not m_title or not m_desc:
        st.warning('Title and Description columns are required before proceeding.')
        st.stop()

else:
    col_map  = st.session_state.sam_col_map
    m_title  = col_map['title']
    m_desc   = col_map['description']
    m_naics  = col_map.get('naics_desc')
    m_notice = col_map.get('notice_id')
    m_agency = col_map.get('agency')
    m_posted = col_map.get('posted_date')
    m_dl     = col_map.get('deadline')


# ── Section 3 · Screening ──────────────────────────────────────────────────

st.divider()
st.subheader('3 · Screen with Claude')

n_rows   = len(df_raw)
est_mins = max(1, n_rows // 60)
screened = st.session_state.sam_screened_df

if screened is None:
    st.caption(
        f'Claude Haiku will screen **{n_rows:,}** rows for relevance. '
        f'Estimated time: ~{est_mins} min at {_SCREEN_WORKERS} concurrent workers.'
    )
    if st.button('⚡ Run Screening', type='primary'):
        anth_key = st.secrets['anthropic_api_key']
        try:
            st.session_state.sam_screened_df = _run_screening(df_raw, col_map, anth_key)
            st.rerun()
        except Exception as e:
            st.error(f'Screening failed: {e}')
    st.stop()

passing  = screened[screened['_import'] == True].copy()
failing  = screened[screened['_import'] == False].copy()
pass_pct = len(passing) / len(screened) * 100 if len(screened) > 0 else 0

m1, m2, m3 = st.columns(3)
m1.metric('Total rows',   f'{len(screened):,}')
m2.metric('Passing',      f'{len(passing):,}  ({pass_pct:.0f}%)')
m3.metric('Filtered out', f'{len(failing):,}')

if st.button('↺ Re-run screening'):
    st.session_state.sam_screened_df = None
    st.rerun()

_display = [c for c in [m_title, m_desc, '_confidence', '_reason'] if c]
_cfg = {
    m_title:       st.column_config.TextColumn('Title',       width='medium'),
    m_desc:        st.column_config.TextColumn('Description', width='large'),
    '_confidence': st.column_config.TextColumn('Confidence',  width='small'),
    '_reason':     st.column_config.TextColumn('Reason',      width='large'),
}

with st.expander(f'✅ Passing ({len(passing)})', expanded=True):
    if passing.empty:
        st.info('No rows passed screening.')
    else:
        st.dataframe(
            passing[_display].reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
            column_config=_cfg,
        )

with st.expander(f'❌ Filtered out ({len(failing)})', expanded=False):
    if failing.empty:
        st.info('Nothing was filtered out.')
    else:
        st.dataframe(
            failing[_display].reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
            column_config=_cfg,
        )


# ── Section 4 · Save ──────────────────────────────────────────────────────

st.divider()
st.subheader('4 · Save to topic store')

if passing.empty:
    st.warning('No passing rows to save — nothing to embed.')
    st.stop()

if st.session_state.sam_existing_keys is None:
    with st.spinner('Checking existing records for duplicates…'):
        try:
            existing_ids, existing_titles = _load_existing_keys(_get_storage_client())
            st.session_state.sam_existing_keys = (existing_ids, existing_titles)
        except Exception as e:
            st.warning(f'Could not load existing records for dedup check: {e}')
            st.session_state.sam_existing_keys = (set(), set())

existing_ids, existing_titles = st.session_state.sam_existing_keys


def _is_dup(row: pd.Series) -> bool:
    if m_notice:
        if str(row.get(m_notice, '')).strip() in existing_ids:
            return True
    if str(row.get(m_title, '')).strip().lower() in existing_titles:
        return True
    return False


dup_mask = passing.apply(_is_dup, axis=1)
dupes    = passing[dup_mask]
new_rows = passing[~dup_mask]

if not dupes.empty:
    st.info(
        f'**{len(dupes)}** row(s) skipped — notice ID or title already exists in the store.',
        icon='ℹ️',
    )
    with st.expander(f'Skipped duplicates ({len(dupes)})', expanded=False):
        st.dataframe(
            dupes[[m_title] + ([m_notice] if m_notice else [])].reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
        )

if new_rows.empty:
    st.success('All passing rows are already in the store — nothing new to save.')
    st.stop()

st.caption(
    f'**{len(new_rows)}** new rows will be embedded (`text-embedding-ada-002`) and saved to '
    f'`{_SAM_PREFIX}sam_gov_{{date}}_{{hex}}.parquet`.'
)

if st.button('💾 Embed & Save', type='primary'):
    oai_key  = st.secrets['openai_api_key']
    anth_key = st.secrets['anthropic_api_key']
    try:
        path = _embed_and_save(new_rows.reset_index(drop=True), col_map, oai_key, anth_key)
        st.success(f'Saved **{path}** — {len(new_rows)} topics ready for matching.')
        st.session_state.sam_raw_df        = None
        st.session_state.sam_screened_df   = None
        st.session_state.sam_existing_keys = None
        st.session_state.sam_col_map       = None
        st.session_state.sam_from_api      = None
        st.rerun()
    except Exception as e:
        st.error(f'Save failed: {e}')
