"""
Grants.gov Fetch
----------------
Query the Grants.gov public search API for federal grant opportunities.
Filter by keyword, opportunity status, funding instrument, and agency.
Claude Haiku screens each row for relevance, then passing rows are
embedded and saved to GCS under data/all-topics/processed/GRANTS-GOV/.
No API key required — uses the public search2 endpoint.
"""

import io
import json
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta

import pandas as pd
import requests
import streamlit as st
from anthropic import Anthropic
from google.cloud import storage
from google.oauth2 import service_account

from src.modules.Embedding.text_embedder import TextProcessor
from src.modules.GoogleBucketManager.bucket_manager import BucketManager

# ── GCS ────────────────────────────────────────────────────────────────────

_BUCKET        = 'cc-matcher-bucket-jeg-v1'
_GRANTS_PREFIX = 'data/all-topics/processed/GRANTS-GOV/'

_SCREEN_MODEL    = 'claude-haiku-4-5-20251001'
_SCREEN_WORKERS  = 8
_SCREEN_MAX_CHARS = 3000

# ── Grants.gov API ─────────────────────────────────────────────────────────

_SEARCH_URL   = 'https://api.grants.gov/v1/api/search2'
_PAGE_SIZE    = 25   # conservative; grants.gov legacy API

_STATUS_OPTIONS = {
    'Posted':     'posted',
    'Forecasted': 'forecasted',
    'Closed':     'closed',
    'Archived':   'archived',
}

_INSTRUMENT_OPTIONS = {
    'Grant':                 'G',
    'Cooperative Agreement': 'CA',
    'Procurement Contract':  'PC',
    'Other':                 'O',
}

_GRANTS_RESERVED_COLS = frozenset({
    'topic_number', 'agency', 'title', 'description', 'open_date', 'close_date',
    'scraped_at', 'grant_summary', 'embeddings', 'award_ceiling', 'status',
    'sam_confidence', 'sam_reason',
})


# ── Storage client ─────────────────────────────────────────────────────────

def _get_storage_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)


# ── Grants.gov helpers ─────────────────────────────────────────────────────

def _parse_date(s: str) -> date | None:
    for fmt in ('%m/%d/%Y', '%Y-%m-%d', '%Y/%m/%d'):
        try:
            return datetime.strptime(str(s).strip(), fmt).date()
        except (ValueError, TypeError):
            pass
    return None


def _items_to_df(items: list[dict]) -> pd.DataFrame:
    rows = []
    for opp in items:
        synopsis = opp.get('synopsis', {})
        if isinstance(synopsis, dict):
            desc = (
                synopsis.get('synopsisDesc') or
                synopsis.get('fundDesc') or
                synopsis.get('text') or ''
            )
            award_ceiling = str(synopsis.get('awardCeiling') or '')
        else:
            desc = str(synopsis) if synopsis else ''
            award_ceiling = ''

        rows.append({
            'title':         str(opp.get('title') or '').strip(),
            'description':   str(desc).strip(),
            'notice_id':     str(opp.get('number') or opp.get('id') or '').strip(),
            'agency':        str(opp.get('agencyName') or '').strip(),
            'agency_code':   str(opp.get('agencyCode') or '').strip(),
            'posted_date':   str(opp.get('openDate') or '').strip(),
            'close_date':    str(opp.get('closeDate') or '').strip(),
            'award_ceiling': award_ceiling,
            'status':        str(opp.get('oppStatus') or '').strip(),
        })
    return pd.DataFrame(rows)


def _search_grants(
    keyword:     str,
    statuses:    list[str],
    instruments: list[str],
    agencies:    list[str],
    date_from:   date | None,
    date_to:     date | None,
    max_results: int,
) -> tuple[pd.DataFrame, int]:
    all_items: list[dict] = []
    start = 1
    total = None

    with st.spinner('Querying Grants.gov…'):
        while True:
            payload: dict = {
                'oppStatuses': statuses,
                'rows':        _PAGE_SIZE,
                'startRecord': start,
            }
            if keyword:
                payload['keyword'] = keyword
            if instruments:
                payload['fundingInstruments'] = instruments
            if agencies:
                payload['agencies'] = agencies

            r = requests.post(_SEARCH_URL, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()

            if total is None:
                total = int(data.get('hitCount', 0))

            page_items = data.get('oppHits') or []
            all_items.extend(page_items)

            if not page_items:
                break
            if max_results and len(all_items) >= max_results:
                all_items = all_items[:max_results]
                break
            start += len(page_items)
            if start > total:
                break

    df = _items_to_df(all_items)

    # Client-side date filter (API doesn't support date range in POST body)
    if (date_from or date_to) and not df.empty:
        parsed = df['posted_date'].apply(_parse_date)
        mask   = pd.Series(True, index=df.index)
        if date_from:
            mask &= parsed.apply(lambda d: d is None or d >= date_from)
        if date_to:
            mask &= parsed.apply(lambda d: d is None or d <= date_to)
        df = df[mask].reset_index(drop=True)

    return df, total or 0


# ── Screening ─────────────────────────────────────────────────────────────

_SCREEN_SYSTEM = """\
You are a grant opportunity screening filter for a consultancy that serves innovative startups, R&D companies, and deep tech small businesses.

You will be given a federal grant opportunity. Your only job is to decide: should this opportunity be imported into our matching system?

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
- The work requires genuine technical innovation, applied research, or novel development
- It is a vehicle our clients pursue (SBIR, STTR, BAA, cooperative agreement for R&D, etc.)
- The topic area is one where deep tech startups commonly compete

DO NOT IMPORT (NO) if any of the following are true:
- The opportunity is clearly intended for universities, nonprofits, state/local governments, or large prime contractors only
- It is workforce development, training, community services, or social services
- It is a generic service procurement with no meaningful R&D or innovation component
- The description makes clear that large-scale prior program experience beyond typical small business reach is required

When uncertain, only import if the opportunity is clearly relevant.

---

OUTPUT FORMAT:
Respond only with valid JSON. No preamble, no markdown, no explanation outside the JSON.
{"import": true, "confidence": "high", "reason": "One or two sentences explaining the decision."}\
"""


def _screen_one(title: str, desc: str, anth_key: str) -> dict:
    client   = Anthropic(api_key=anth_key)
    user_msg = f"Title: {title}\n\nDescription: {str(desc)[:_SCREEN_MAX_CHARS]}"
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


def _run_screening(df: pd.DataFrame, anth_key: str) -> pd.DataFrame:
    titles  = df['title'].astype(str).tolist()
    descs   = df['description'].astype(str).tolist()
    results = [None] * len(df)
    progress = st.progress(0, text='Screening rows…')

    with ThreadPoolExecutor(max_workers=_SCREEN_WORKERS) as pool:
        futures = {
            pool.submit(_screen_one, titles[i], descs[i], anth_key): i
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


# ── Summarization ──────────────────────────────────────────────────────────

_SUMMARY_SYSTEM = """\
You are preparing a federal grant opportunity for semantic matching against startup and R&D company profiles.

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


def _summarize_all(titles: list[str], descs: list[str], anth_key: str) -> list[str]:
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


# ── Dedup ─────────────────────────────────────────────────────────────────

def _load_existing_keys(client: storage.Client) -> tuple[set[str], set[str]]:
    notice_ids: set[str] = set()
    titles:     set[str] = set()
    blobs = client.list_blobs(_BUCKET, prefix=_GRANTS_PREFIX)
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

def _embed_and_save(
    df: pd.DataFrame,
    oai_key: str,
    anth_key: str,
    extra_cols: dict[str, str] | None = None,
) -> str:
    tp    = TextProcessor(api_key=oai_key)
    bm    = BucketManager(_BUCKET, client=_get_storage_client())
    today = datetime.today().strftime('%Y-%m-%d')

    out = pd.DataFrame()
    out['topic_number']  = df['notice_id'].astype(str)
    out['agency']        = df['agency'].astype(str)
    out['title']         = df['title'].astype(str)
    out['description']   = df['description'].astype(str)
    out['open_date']     = df['posted_date'].astype(str)
    out['close_date']    = df['close_date'].astype(str)
    out['award_ceiling'] = df['award_ceiling'].astype(str)
    out['status']        = df['status'].astype(str)
    out['scraped_at']    = today

    summaries        = _summarize_all(out['title'].tolist(), out['description'].tolist(), anth_key)
    out['grant_summary'] = summaries

    embed_texts = [s if s.strip() else d for s, d in zip(summaries, out['description'].tolist())]
    progress    = st.progress(0, text='Generating embeddings…')
    embeddings  = []
    for i, text in enumerate(embed_texts):
        embeddings.append(tp.get_embedding(text) if text.strip() else None)
        progress.progress((i + 1) / len(embed_texts), text=f'Embedding {i + 1}/{len(embed_texts)}…')
    progress.empty()

    out['embeddings'] = embeddings

    if extra_cols:
        for col_name, col_val in extra_cols.items():
            out[col_name] = col_val

    hex_suffix = secrets.token_hex(3)
    gcs_path   = f'{_GRANTS_PREFIX}grants_gov_{today}_{hex_suffix}.parquet'
    bm.upload_file(gcs_path, out)
    return gcs_path


# ── Session state ──────────────────────────────────────────────────────────

for _k in ['ggov_raw_df', 'ggov_screened_df', 'ggov_existing_keys']:
    if _k not in st.session_state:
        st.session_state[_k] = None
if 'ggov_custom_cols' not in st.session_state:
    st.session_state.ggov_custom_cols = []


# ── Page ───────────────────────────────────────────────────────────────────

st.title('🏦 Grants.gov Fetch')
st.caption(
    'Search Grants.gov for federal funding opportunities. '
    'Claude screens each result for R&D relevance, then passing rows are embedded '
    'and saved to the topic store. No API key required.'
)

# ── Section 1 · Fetch parameters ───────────────────────────────────────────

st.subheader('1 · Fetch from Grants.gov')

col_l, col_r = st.columns(2)
with col_l:
    today_date   = datetime.today().date()
    default_from = today_date - timedelta(days=60)
    gg_date_from = st.date_input('Posted from', value=default_from, key='gg_date_from')
with col_r:
    gg_date_to = st.date_input('Posted to', value=today_date, key='gg_date_to')

gg_keyword = st.text_input(
    'Keyword (optional)',
    placeholder='e.g. SBIR, cybersecurity, artificial intelligence, biotech',
    key='gg_keyword',
)

col_s, col_f = st.columns(2)
with col_s:
    selected_status_labels = st.multiselect(
        'Opportunity status',
        list(_STATUS_OPTIONS.keys()),
        default=['Posted', 'Forecasted'],
        key='gg_statuses',
    )
with col_f:
    selected_instr_labels = st.multiselect(
        'Funding instruments',
        list(_INSTRUMENT_OPTIONS.keys()),
        default=['Grant', 'Cooperative Agreement'],
        key='gg_instruments',
    )

col_a, col_m = st.columns(2)
with col_a:
    gg_agencies_raw = st.text_input(
        'Agency codes (optional, comma-separated)',
        placeholder='e.g. HHS, NSF, DOD, NIH',
        key='gg_agencies',
    )
with col_m:
    gg_max = st.number_input(
        'Max results (0 = no cap)',
        min_value=0,
        value=200,
        step=25,
        key='gg_max',
    )

st.caption(
    'Date range filters are applied client-side after fetch. '
    'For large date windows, set a keyword or lower the max results to stay responsive.'
)

if gg_date_from > gg_date_to:
    st.error('"Posted from" must be on or before "Posted to".')
elif st.button('🔍 Fetch from Grants.gov', type='primary', key='gg_fetch_btn'):
    statuses    = [_STATUS_OPTIONS[lbl]    for lbl in selected_status_labels]
    instruments = [_INSTRUMENT_OPTIONS[lbl] for lbl in selected_instr_labels]
    agencies    = [a.strip().upper() for a in gg_agencies_raw.split(',') if a.strip()]

    if not statuses:
        st.error('Select at least one opportunity status.')
    else:
        try:
            df_fetched, total = _search_grants(
                keyword     = gg_keyword.strip(),
                statuses    = statuses,
                instruments = instruments,
                agencies    = agencies,
                date_from   = gg_date_from,
                date_to     = gg_date_to,
                max_results = int(gg_max),
            )
            if df_fetched.empty:
                st.warning(
                    f'No opportunities matched the filters (API returned {total:,} total records). '
                    'Try a broader date range or fewer filters.'
                )
            else:
                st.caption(f'API total: **{total:,}** — after date filter: **{len(df_fetched):,}** rows.')
                st.session_state.ggov_raw_df        = df_fetched
                st.session_state.ggov_screened_df   = None
                st.session_state.ggov_existing_keys = None
                st.session_state.ggov_custom_cols   = []
                st.rerun()
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else '?'
            st.error(f'Grants.gov API HTTP {code}: {exc}')
        except Exception as exc:
            st.error(f'Fetch failed: {exc}')

if st.session_state.ggov_raw_df is None:
    st.stop()

df_raw = st.session_state.ggov_raw_df
st.caption(f'**{len(df_raw):,}** rows loaded.')

no_desc = (df_raw['description'].str.strip() == '').sum()
if no_desc > 0:
    st.info(
        f'{no_desc} row(s) have no description text from the API. '
        'These will be screened and embedded by title only.',
        icon='ℹ️',
    )

_preview_cols = ['title', 'agency', 'posted_date', 'close_date', 'award_ceiling', 'description']
st.dataframe(df_raw[_preview_cols].head(5), hide_index=True, use_container_width=True)


# ── Section 2 · Screen with Claude ─────────────────────────────────────────

st.divider()
st.subheader('2 · Screen with Claude')

n_rows   = len(df_raw)
est_mins = max(1, n_rows // 60)
screened = st.session_state.ggov_screened_df

if screened is None:
    st.caption(
        f'Claude Haiku will screen **{n_rows:,}** rows for relevance to R&D small businesses. '
        f'Estimated time: ~{est_mins} min at {_SCREEN_WORKERS} concurrent workers.'
    )
    if st.button('⚡ Run Screening', type='primary', key='gg_screen_btn'):
        anth_key = st.secrets['anthropic_api_key']
        try:
            st.session_state.ggov_screened_df = _run_screening(df_raw, anth_key)
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

if st.button('↺ Re-run screening', key='gg_rescreen_btn'):
    st.session_state.ggov_screened_df = None
    st.rerun()

_disp_cols = ['title', 'agency', '_confidence', '_reason']
_disp_cfg  = {
    'title':        st.column_config.TextColumn('Title',      width='medium'),
    'agency':       st.column_config.TextColumn('Agency',     width='small'),
    '_confidence':  st.column_config.TextColumn('Confidence', width='small'),
    '_reason':      st.column_config.TextColumn('Reason',     width='large'),
}

with st.expander(f'✅ Passing ({len(passing)})', expanded=True):
    if passing.empty:
        st.info('No rows passed screening.')
    else:
        st.dataframe(
            passing[_disp_cols].reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
            column_config=_disp_cfg,
        )

with st.expander(f'❌ Filtered out ({len(failing)})', expanded=False):
    if failing.empty:
        st.info('Nothing was filtered out.')
    else:
        st.dataframe(
            failing[_disp_cols].reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
            column_config=_disp_cfg,
        )


# ── Section 3 · Save ──────────────────────────────────────────────────────

st.divider()
st.subheader('3 · Save to topic store')

if passing.empty:
    st.warning('No passing rows to save — nothing to embed.')
    st.stop()

if st.session_state.ggov_existing_keys is None:
    with st.spinner('Checking existing records for duplicates…'):
        try:
            existing_ids, existing_titles = _load_existing_keys(_get_storage_client())
            st.session_state.ggov_existing_keys = (existing_ids, existing_titles)
        except Exception as e:
            st.warning(f'Could not load existing records for dedup check: {e}')
            st.session_state.ggov_existing_keys = (set(), set())

existing_ids, existing_titles = st.session_state.ggov_existing_keys


def _is_dup(row: pd.Series) -> bool:
    if str(row.get('notice_id', '')).strip() in existing_ids:
        return True
    if str(row.get('title', '')).strip().lower() in existing_titles:
        return True
    return False


dup_mask = passing.apply(_is_dup, axis=1)
dupes    = passing[dup_mask]
new_rows = passing[~dup_mask]

if not dupes.empty:
    st.info(f'**{len(dupes)}** row(s) skipped — notice ID or title already exists in the store.', icon='ℹ️')
    with st.expander(f'Skipped duplicates ({len(dupes)})', expanded=False):
        st.dataframe(
            dupes[['title', 'notice_id']].reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
        )

if new_rows.empty:
    st.success('All passing rows are already in the store — nothing new to save.')
    st.stop()

with st.expander('➕ Custom columns', expanded=bool(st.session_state.ggov_custom_cols)):
    st.caption(
        'Add extra columns to tag every saved topic with campaign-specific metadata. '
        'Columns are saved to the parquet and available as optional fields in HubSpot import.'
    )

    for entry in list(st.session_state.ggov_custom_cols):
        gc1, gc2, gc3 = st.columns([2, 4, 1])
        with gc1:
            st.text(entry['name'])
        with gc2:
            entry['value'] = st.text_input(
                'Value', value=entry['value'],
                key=f'gcfill_{entry["name"]}', label_visibility='collapsed',
            )
        with gc3:
            if st.button('✕', key=f'gcrm_{entry["name"]}', use_container_width=True):
                st.session_state.ggov_custom_cols = [
                    e for e in st.session_state.ggov_custom_cols if e['name'] != entry['name']
                ]
                st.rerun()

    if st.session_state.ggov_custom_cols:
        st.divider()

    gna1, gna2, gna3 = st.columns([2, 4, 1])
    with gna1:
        new_gc_name = st.text_input(
            'Column name', placeholder='e.g. campaign_name',
            key='new_gc_name', label_visibility='collapsed',
        )
    with gna2:
        new_gc_val = st.text_input(
            'Value', placeholder='e.g. Spring 2026',
            key='new_gc_val', label_visibility='collapsed',
        )
    with gna3:
        if st.button('Add', key='add_gc_btn', use_container_width=True):
            clean = new_gc_name.strip().replace(' ', '_').lower()
            existing_names = {e['name'] for e in st.session_state.ggov_custom_cols}
            if (
                clean
                and clean not in _GRANTS_RESERVED_COLS
                and clean not in existing_names
            ):
                st.session_state.ggov_custom_cols.append({'name': clean, 'value': new_gc_val.strip()})
                st.rerun()

st.caption(
    f'**{len(new_rows)}** new rows will be summarized, embedded (`text-embedding-ada-002`), and saved to '
    f'`{_GRANTS_PREFIX}grants_gov_{{date}}_{{hex}}.parquet`.'
)

if st.button('💾 Embed & Save', type='primary', key='gg_save_btn'):
    oai_key  = st.secrets['openai_api_key']
    anth_key = st.secrets['anthropic_api_key']
    extra_cols_dict = {e['name']: e['value'] for e in st.session_state.ggov_custom_cols} or None
    try:
        path = _embed_and_save(
            new_rows.reset_index(drop=True),
            oai_key,
            anth_key,
            extra_cols=extra_cols_dict,
        )
        st.success(f'Saved **{path}** — {len(new_rows)} topics ready for matching.')
        st.session_state.ggov_raw_df        = None
        st.session_state.ggov_screened_df   = None
        st.session_state.ggov_existing_keys = None
        st.session_state.ggov_custom_cols   = []
        st.rerun()
    except Exception as e:
        st.error(f'Save failed: {e}')
