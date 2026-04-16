"""
SAM.gov Upload
--------------
Upload SAM.gov contract opportunity CSVs.
Claude Haiku screens each row for relevance to deep tech / R&D small businesses.
Passing rows are embedded and saved to GCS under data/all-topics/sam-gov/.
"""

import json
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import streamlit as st
from anthropic import Anthropic
from google.cloud import storage
from google.oauth2 import service_account

from src.modules.Embedding.text_embedder import TextProcessor
from src.modules.GoogleBucketManager.bucket_manager import BucketManager

# ── GCS ────────────────────────────────────────────────────────────────────

_BUCKET     = 'cc-matcher-bucket-jeg-v1'
_SAM_PREFIX = 'data/all-topics/sam-gov/'

_SCREEN_MODEL     = 'claude-haiku-4-5-20251001'
_SCREEN_WORKERS   = 8
_SCREEN_MAX_CHARS = 3000   # truncate description to keep prompt cost low


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


# ── Column auto-detection ──────────────────────────────────────────────────

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


# ── Existing-record keys (for dedup) ──────────────────────────────────────

def _load_existing_keys(client: storage.Client) -> tuple[set[str], set[str]]:
    """Return (notice_ids, lower_titles) already in the sam-gov prefix."""
    notice_ids: set[str] = set()
    titles:     set[str] = set()
    blobs = client.list_blobs(_BUCKET, prefix=_SAM_PREFIX)
    for blob in blobs:
        if not blob.name.endswith('.parquet'):
            continue
        try:
            import io
            df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()), columns=['topic_number', 'title'])
            notice_ids.update(df['topic_number'].dropna().astype(str).str.strip())
            titles.update(df['title'].dropna().astype(str).str.lower().str.strip())
        except Exception:
            pass
    return notice_ids, titles


# ── Embed + save ───────────────────────────────────────────────────────────

def _embed_and_save(df: pd.DataFrame, col_map: dict, oai_key: str) -> str:
    tp    = TextProcessor(api_key=oai_key)
    bm    = BucketManager(_BUCKET, client=_get_storage_client())
    today = datetime.today().strftime('%Y-%m-%d')

    out = pd.DataFrame()
    out['topic_number'] = df[col_map['notice_id']].astype(str)   if col_map.get('notice_id')    else ''
    out['agency']       = df[col_map['agency']].astype(str)       if col_map.get('agency')       else 'SAM-GOV'
    out['title']        = df[col_map['title']].astype(str)
    out['description']  = df[col_map['description']].astype(str)
    out['open_date']    = df[col_map['posted_date']].astype(str)  if col_map.get('posted_date')  else ''
    out['close_date']   = df[col_map['deadline']].astype(str)     if col_map.get('deadline')     else ''
    out['scraped_at']   = today
    out['sam_confidence'] = df['_confidence'].values
    out['sam_reason']   = df['_reason'].values

    descs    = out['description'].astype(str).tolist()
    progress = st.progress(0, text='Generating embeddings…')
    embeddings = []
    for i, desc in enumerate(descs):
        embeddings.append(tp.get_embedding(desc) if desc.strip() else None)
        progress.progress((i + 1) / len(descs), text=f'Embedding {i + 1}/{len(descs)}…')
    progress.empty()

    out['embeddings'] = embeddings

    hex_suffix = secrets.token_hex(3)
    gcs_path   = f'{_SAM_PREFIX}sam_gov_{today}_{hex_suffix}.parquet'
    bm.upload_file(gcs_path, out)
    return gcs_path


# ── Session state ──────────────────────────────────────────────────────────

for _k in ['sam_raw_df', 'sam_screened_df', 'sam_existing_keys']:
    if _k not in st.session_state:
        st.session_state[_k] = None


# ── Page ───────────────────────────────────────────────────────────────────

st.title('🏛️ SAM.gov Upload')
st.caption(
    'Upload SAM.gov contract opportunity CSVs. '
    'Claude screens each row for relevance to deep tech / R&D small businesses, '
    'then you save passing rows to the topic store with embeddings.'
)

# ── Section 1 · Upload ─────────────────────────────────────────────────────

st.subheader('1 · Upload CSVs')

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
        # Reset downstream state whenever the upload changes
        if (
            st.session_state.sam_raw_df is None
            or len(combined) != len(st.session_state.sam_raw_df)
        ):
            st.session_state.sam_raw_df      = combined
            st.session_state.sam_screened_df = None

if st.session_state.sam_raw_df is None:
    st.stop()

df_raw = st.session_state.sam_raw_df
file_count = len(files) if files else '(previously loaded)'
st.caption(f'**{len(df_raw):,}** rows loaded from {file_count} file(s).')
st.dataframe(df_raw.head(5), hide_index=True, use_container_width=True)

# ── Section 2 · Column mapping ─────────────────────────────────────────────

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
    m_title   = _sel('title',       'Title',              required=True)
    m_desc    = _sel('description', 'Description',        required=True)
    m_naics   = _sel('naics_desc',  'NAICS Descriptor')
    m_notice  = _sel('notice_id',   'Notice ID')
with right_col:
    m_agency  = _sel('agency',      'Agency / Department')
    m_posted  = _sel('posted_date', 'Posted Date')
    m_dl      = _sel('deadline',    'Response Deadline')

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

# ── Section 3 · Screening ──────────────────────────────────────────────────

st.divider()
st.subheader('3 · Screen with Claude')

n_rows    = len(df_raw)
est_mins  = max(1, n_rows // 60)
screened  = st.session_state.sam_screened_df

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

# Results are ready
passing = screened[screened['_import'] == True].copy()
failing = screened[screened['_import'] == False].copy()
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
    m_title:        st.column_config.TextColumn('Title',       width='medium'),
    m_desc:         st.column_config.TextColumn('Description', width='large'),
    '_confidence':  st.column_config.TextColumn('Confidence',  width='small'),
    '_reason':      st.column_config.TextColumn('Reason',      width='large'),
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

# ── Load existing keys once per session ────────────────────────────────────

if st.session_state.sam_existing_keys is None:
    with st.spinner('Checking existing records for duplicates…'):
        try:
            existing_ids, existing_titles = _load_existing_keys(_get_storage_client())
            st.session_state.sam_existing_keys = (existing_ids, existing_titles)
        except Exception as e:
            st.warning(f'Could not load existing records for dedup check: {e}')
            st.session_state.sam_existing_keys = (set(), set())

existing_ids, existing_titles = st.session_state.sam_existing_keys

# ── Apply deduplication ────────────────────────────────────────────────────

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
    oai_key = st.secrets['openai_api_key']
    try:
        path = _embed_and_save(new_rows.reset_index(drop=True), col_map, oai_key)
        st.success(f'Saved **{path}** — {len(new_rows)} topics ready for matching.')
        st.session_state.sam_raw_df       = None
        st.session_state.sam_screened_df  = None
        st.session_state.sam_existing_keys = None
        st.rerun()
    except Exception as e:
        st.error(f'Save failed: {e}')
