"""
Contact Importer
-----------------
Upload a lead spreadsheet from any source, map columns to standard fields,
scrape company websites (aiohttp → Playwright fallback), summarise with
GPT-3.5-turbo, embed with text-embedding-ada-002, and save to GCS under
data/all-contacts/{source}/{source}_{date}_{hex}.parquet.
"""

import asyncio
import io
import secrets
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed as futures_completed
from datetime import datetime

import aiohttp
import pandas as pd
import tldextract
import streamlit as st
from bs4 import BeautifulSoup
from google.cloud import storage
from google.oauth2 import service_account
from openai import OpenAI

from src.modules.Embedding.text_embedder import TextProcessor
from src.modules.GoogleBucketManager.bucket_manager import BucketManager

# ── Constants ──────────────────────────────────────────────────────────────

_BUCKET          = 'cc-matcher-bucket-jeg-v1'
_CONTACTS_ROOT   = 'data/all-contacts/'
_SCRAPE_TIMEOUT  = 15        # seconds (aiohttp)
_PW_TIMEOUT      = 20_000    # ms (Playwright)
_MAX_CONCURRENT  = 20        # semaphore
_PAGE_TEXT_LIMIT = 8_000     # chars

_SUMMARISE_SYSTEM = (
    'Summarise what this company does in 3-5 sentences. '
    'Focus on technology, product, and market. Be factual and concise.'
)

_SOURCE_OPTIONS = ['apollo', 'sba', 'free_alert', 'custom…']

_FIELD_CANDIDATES: dict[str, list[str]] = {
    'companyWebsite': ['website', 'website url', 'url', 'company website', 'companywebsite', 'web', 'homepage', 'domain'],
    'companyName':    ['company', 'company name', 'companyname', 'organization', 'org name', 'account name'],
    'state':          ['state', 'state/province', 'state/territory', 'region', 'province'],
    'firstName':      ['first name', 'firstname', 'first_name', 'given name'],
    'lastName':       ['last name', 'lastname', 'last_name', 'surname', 'family name'],
    'email':          ['email', 'email address', 'work email', 'e-mail'],
    'phone':          ['phone', 'phone number', 'mobile', 'telephone', 'cell'],
    'segment':        ['industry', 'segment', 'vertical', 'sector'],
}


def _detect_col(columns: list[str], field: str) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    for candidate in _FIELD_CANDIDATES[field]:
        if candidate in lower:
            return lower[candidate]
    return None


# ── URL helpers ────────────────────────────────────────────────────────────

def _normalize_url(url) -> str:
    url = str(url or '').strip()
    if not url or url.lower() in ('nan', 'none', ''):
        return ''
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url


def _bare_domain(url: str) -> str:
    ext = tldextract.extract(str(url))
    return f'{ext.domain}.{ext.suffix}'.lower() if ext.domain else ''


# ── GCS helpers ────────────────────────────────────────────────────────────

def _get_storage_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)


def _load_existing_domains(client: storage.Client, source: str) -> set[str]:
    """Return bare domains already stored for this source in GCS."""
    prefix = f'{_CONTACTS_ROOT}{source}/'
    blobs  = client.list_blobs(_BUCKET, prefix=prefix)
    domains: set[str] = set()
    for blob in blobs:
        if not blob.name.endswith('.parquet'):
            continue
        try:
            df = pd.read_parquet(
                io.BytesIO(blob.download_as_bytes()),
                columns=['companyWebsite'],
            )
            for url in df['companyWebsite'].dropna().astype(str):
                d = _bare_domain(url)
                if d:
                    domains.add(d)
        except Exception:
            pass
    return domains


# ── Async scraping ─────────────────────────────────────────────────────────

async def _aiohttp_scrape(session: aiohttp.ClientSession, url: str) -> str:
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=_SCRAPE_TIMEOUT),
            ssl=False,
        ) as resp:
            if resp.status >= 400:
                return 'FAILED'
            html = await resp.text(errors='replace')
            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()
            return ' '.join(soup.get_text(separator=' ').split())[:_PAGE_TEXT_LIMIT]
    except Exception:
        return 'FAILED'


async def _playwright_scrape(url: str) -> str:
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page    = await browser.new_page()
            await page.goto(url, timeout=_PW_TIMEOUT, wait_until='domcontentloaded')
            content = await page.inner_text('body')
            await browser.close()
            return ' '.join(content.split())[:_PAGE_TEXT_LIMIT]
    except Exception:
        return 'ERROR'


async def _scrape_one(sem: asyncio.Semaphore, session: aiohttp.ClientSession, url: str, idx: int) -> dict:
    async with sem:
        text = await _aiohttp_scrape(session, url)
        if text == 'FAILED':
            text = await _playwright_scrape(url)
    return {'_idx': idx, 'page_text': text}


async def _run_scraping(urls: list[str], progress) -> list[str]:
    """Scrape all URLs concurrently (≤20 at once). Returns page_texts in input order."""
    sem     = asyncio.Semaphore(_MAX_CONCURRENT)
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; MatcherBot/1.0)'}
    total   = len(urls)
    ordered = {}
    done_n  = 0

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [
            asyncio.create_task(_scrape_one(sem, session, url, i))
            for i, url in enumerate(urls)
        ]
        for coro in asyncio.as_completed(tasks):
            item              = await coro
            ordered[item['_idx']] = item['page_text']
            done_n           += 1
            progress.progress(done_n / total, text=f'Scraping… {done_n}/{total}')

    return [ordered[i] for i in range(total)]


# ── Summarisation (threaded sync OpenAI) ───────────────────────────────────

def _run_summarization(page_texts: list[str], openai_key: str, progress) -> list[str]:
    client    = OpenAI(api_key=openai_key)
    summaries = [''] * len(page_texts)

    def _one(idx: int, text: str) -> tuple[int, str]:
        if not text or text in ('FAILED', 'ERROR', 'nan', ''):
            return idx, ''
        try:
            resp = client.chat.completions.create(
                model='gpt-3.5-turbo',
                max_tokens=300,
                messages=[
                    {'role': 'system', 'content': _SUMMARISE_SYSTEM},
                    {'role': 'user',   'content': text[:_PAGE_TEXT_LIMIT]},
                ],
            )
            return idx, resp.choices[0].message.content.strip()
        except Exception as e:
            return idx, f'SUMMARY_ERROR: {e}'

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_one, i, t): i for i, t in enumerate(page_texts)}
        done = 0
        for future in futures_completed(futures):
            idx, summary   = future.result()
            summaries[idx] = summary
            done          += 1
            progress.progress(done / len(page_texts), text=f'Summarising… {done}/{len(page_texts)}')

    return summaries


# ── Session state init ─────────────────────────────────────────────────────

for _k in ('ci_raw_df', 'ci_deduped_df', 'ci_scraped_df', 'ci_dedup_source', 'ci_dedup_url_col'):
    if _k not in st.session_state:
        st.session_state[_k] = None


# ── Page ───────────────────────────────────────────────────────────────────

st.title('📋 Contact Importer')
st.caption(
    'Upload a lead spreadsheet (Apollo, SBA, or any source), map columns, '
    'scrape company websites, and add contacts to the matching database.'
)

# ── 1 · Upload ─────────────────────────────────────────────────────────────

st.subheader('1 · Upload spreadsheet')

uploaded = st.file_uploader(
    'Upload CSV or Excel',
    type=['csv', 'xlsx', 'xls'],
    label_visibility='collapsed',
)

if uploaded:
    try:
        if uploaded.name.endswith(('.xlsx', '.xls')):
            raw = pd.read_excel(uploaded, dtype=str)
        else:
            try:
                raw = pd.read_csv(uploaded, dtype=str, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded.seek(0)
                raw = pd.read_csv(uploaded, dtype=str, encoding='latin-1')
        raw = raw.dropna(how='all')

        if (
            st.session_state.ci_raw_df is None
            or len(raw) != len(st.session_state.ci_raw_df)
        ):
            st.session_state.ci_raw_df        = raw
            st.session_state.ci_deduped_df    = None
            st.session_state.ci_scraped_df    = None
            st.session_state.ci_dedup_source  = None
            st.session_state.ci_dedup_url_col = None
    except Exception as e:
        st.error(f'Could not read file: {e}')

if st.session_state.ci_raw_df is None:
    st.stop()

df_raw = st.session_state.ci_raw_df
st.caption(f'**{len(df_raw):,}** rows loaded.')
st.dataframe(df_raw.head(5), hide_index=True, use_container_width=True)

# ── 2 · Source & column mapping ────────────────────────────────────────────

st.divider()
st.subheader('2 · Source & column mapping')

top_l, top_r = st.columns([1, 3])

with top_l:
    src_choice = st.selectbox('Lead source', _SOURCE_OPTIONS, key='ci_src_choice')
    if src_choice == 'custom…':
        source = st.text_input(
            'Custom name',
            placeholder='e.g. linkedin, event_leads',
            key='ci_src_custom',
        ).strip().lower().replace(' ', '_')
    else:
        source = src_choice

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
        key=f'ci_map_{field}',
    )
    return val if val != none_opt else None


with top_r:
    mc1, mc2 = st.columns(2)
    with mc1:
        m_url     = _sel('companyWebsite', 'Website URL', required=True)
        m_name    = _sel('companyName',    'Company name')
        m_state   = _sel('state',          'State')
        m_segment = _sel('segment',        'Industry')
    with mc2:
        m_first = _sel('firstName', 'First name')
        m_last  = _sel('lastName',  'Last name')
        m_email = _sel('email',     'Email')
        m_phone = _sel('phone',     'Phone')

if not source:
    st.warning('Enter a source name to continue.')
    st.stop()
if not m_url:
    st.warning('Website URL column is required.')
    st.stop()

col_map = {
    'companyWebsite': m_url,
    'companyName':    m_name,
    'state':          m_state,
    'segment':        m_segment,
    'firstName':      m_first,
    'lastName':       m_last,
    'email':          m_email,
    'phone':          m_phone,
}

# Invalidate dedup if source or URL column changed since last check
if st.session_state.ci_deduped_df is not None and (
    st.session_state.ci_dedup_source  != source
    or st.session_state.ci_dedup_url_col != m_url
):
    st.session_state.ci_deduped_df = None
    st.session_state.ci_scraped_df = None


# ── 3 · Deduplicate ────────────────────────────────────────────────────────

st.divider()
st.subheader('3 · Deduplicate')


def _build_mapped_df() -> pd.DataFrame:
    out = pd.DataFrame()
    for std_col, src_col in col_map.items():
        if src_col and src_col in df_raw.columns:
            out[std_col] = df_raw[src_col].astype(str)
        else:
            out[std_col] = ''
    out['companyWebsite'] = out['companyWebsite'].apply(_normalize_url)
    return out[out['companyWebsite'] != ''].reset_index(drop=True)


mapped_df = _build_mapped_df()
no_url    = len(df_raw) - len(mapped_df)

if no_url:
    st.caption(f'{no_url:,} row(s) skipped — no URL value.')

if mapped_df.empty:
    st.error('No rows with a valid URL — cannot continue.')
    st.stop()

if st.button('🔍 Check for duplicates', key='ci_dedup_btn'):
    with st.spinner(f'Checking existing records in {_CONTACTS_ROOT}{source}/…'):
        try:
            existing = _load_existing_domains(_get_storage_client(), source)
            mask = mapped_df['companyWebsite'].apply(
                lambda u: _bare_domain(u) not in existing
            )
            st.session_state.ci_deduped_df    = mapped_df[mask].reset_index(drop=True)
            st.session_state.ci_dedup_source  = source
            st.session_state.ci_dedup_url_col = m_url
            st.session_state.ci_scraped_df    = None
            st.rerun()
        except Exception as e:
            st.error(f'Dedup check failed: {e}')

if st.session_state.ci_deduped_df is None:
    st.caption(f'**{len(mapped_df):,}** rows with valid URLs ready to check.')
    st.stop()

deduped_df = st.session_state.ci_deduped_df
n_exist    = len(mapped_df) - len(deduped_df)

dm1, dm2, dm3 = st.columns(3)
dm1.metric('Valid URLs',     f'{len(mapped_df):,}')
dm2.metric('Already stored', f'{n_exist:,}')
dm3.metric('New to import',  f'{len(deduped_df):,}')

if deduped_df.empty:
    st.success('All URLs are already in the contact store — nothing new to import.')
    st.stop()

# ── 4 · Scrape & summarise ─────────────────────────────────────────────────

st.divider()
st.subheader('4 · Scrape & summarise')

if st.session_state.ci_scraped_df is not None:
    scraped_df = st.session_state.ci_scraped_df
    n_ok   = int((scraped_df['scrape_status'] == 'ok').sum())
    n_fail = int((scraped_df['scrape_status'] != 'ok').sum())

    sm1, sm2 = st.columns(2)
    sm1.metric('Scraped OK',       f'{n_ok:,}')
    sm2.metric('Failed / no text', f'{n_fail:,}')

    preview_cols = [c for c in ['companyWebsite', 'companyName', 'scrape_status', 'company_summary']
                    if c in scraped_df.columns]
    st.dataframe(
        scraped_df[preview_cols].head(20),
        hide_index=True,
        use_container_width=True,
        column_config={
            'company_summary': st.column_config.TextColumn('Summary', width='large'),
            'scrape_status':   st.column_config.TextColumn('Status',  width='small'),
        },
    )
    if st.button('↺ Re-scrape'):
        st.session_state.ci_scraped_df = None
        st.rerun()

else:
    n_to_scrape = len(deduped_df)
    st.caption(
        f'**{n_to_scrape:,}** websites will be scraped (aiohttp → Playwright fallback) '
        f'and summarised with GPT-3.5-turbo.'
    )
    if st.button('🕷️ Scrape & Summarise', type='primary', key='ci_scrape_btn'):
        urls    = deduped_df['companyWebsite'].tolist()
        oai_key = st.secrets['openai_api_key']

        scrape_bar = st.progress(0, text='Starting scraper…')
        try:
            if sys.platform == 'win32':
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            page_texts = asyncio.run(_run_scraping(urls, scrape_bar))
        except Exception as e:
            st.error(f'Scraping failed: {e}')
            st.stop()
        scrape_bar.empty()

        summ_bar  = st.progress(0, text='Summarising…')
        summaries = _run_summarization(page_texts, oai_key, summ_bar)
        summ_bar.empty()

        result = deduped_df.copy()
        result['page_text']       = page_texts
        result['company_summary'] = summaries
        result['scrape_status']   = [
            'ok' if t not in ('FAILED', 'ERROR', '', 'nan') else 'failed'
            for t in page_texts
        ]
        st.session_state.ci_scraped_df = result
        st.rerun()
    st.stop()

# ── 5 · Embed & save ──────────────────────────────────────────────────────

st.divider()
st.subheader('5 · Embed & save')

ok_rows = scraped_df[scraped_df['scrape_status'] == 'ok'].reset_index(drop=True)

if ok_rows.empty:
    st.warning('No successfully scraped rows to embed — check URLs and try again.')
    st.stop()

st.caption(
    f'**{len(ok_rows):,}** rows will be embedded (`text-embedding-ada-002`) '
    f'and saved to `{_CONTACTS_ROOT}{source}/`.'
)

if st.button('💾 Embed & Save', type='primary', key='ci_embed_btn'):
    oai_key = st.secrets['openai_api_key']
    tp      = TextProcessor(api_key=oai_key)
    today   = datetime.today().strftime('%Y-%m-%d')

    texts      = ok_rows['company_summary'].tolist()
    embed_bar  = st.progress(0, text='Generating embeddings…')
    embeddings = []
    for i, text in enumerate(texts):
        try:
            embeddings.append(tp.get_embedding(text) if text.strip() else None)
        except Exception:
            embeddings.append(None)
        embed_bar.progress((i + 1) / len(texts), text=f'Embedding {i + 1}/{len(texts)}…')
    embed_bar.empty()

    out = ok_rows.drop(columns=['page_text', 'scrape_status'], errors='ignore').copy()
    out['embeddings'] = embeddings
    out['uuid']       = [str(uuid.uuid4()) for _ in range(len(out))]
    out['scraped_at'] = today

    hex_suffix = secrets.token_hex(3)
    gcs_path   = f'{_CONTACTS_ROOT}{source}/{source}_{today}_{hex_suffix}.parquet'

    try:
        bm = BucketManager(_BUCKET, client=_get_storage_client())
        bm.upload_file(gcs_path, out)
        st.success(f'Saved **{len(out):,}** contacts → `{gcs_path}`')
        for _k in ('ci_raw_df', 'ci_deduped_df', 'ci_scraped_df', 'ci_dedup_source', 'ci_dedup_url_col'):
            st.session_state[_k] = None
        st.rerun()
    except Exception as e:
        st.error(f'Save failed: {e}')
