"""
lead_importer.py – Shared helpers for all lead-source importers.

Functions
---------
load_csvs               – glob + concat raw CSV exports
normalize_columns       – standardise column names
filter_non_commercial   – drop .edu / .gov / .org / .mil domains
dedup_against_existing  – remove URLs already in the parquet store
scrape_pages            – async two-stage aiohttp → Playwright scraper
summarize_companies     – async OpenAI summarisation
embed_summaries         – async OpenAI embedding generation
export_contacts         – save processed contacts to parquet
"""

import asyncio
import glob as glob_module
import os
import uuid
from datetime import datetime
from typing import Optional

import aiohttp
import pandas as pd
import tldextract
from bs4 import BeautifulSoup
from openai import AsyncOpenAI

from src.modules.Embedding.text_embedder import TextProcessor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NON_COMMERCIAL_TLDS = {'.edu', '.gov', '.org', '.mil'}

_SCRAPE_TIMEOUT   = 15        # seconds per URL (aiohttp)
_PLAYWRIGHT_TIMEOUT = 20_000  # ms (Playwright)
_MAX_CONCURRENT   = 20        # semaphore limit for parallel scraping
_PAGE_TEXT_LIMIT  = 8_000     # chars passed to summariser

_SUMMARISE_SYSTEM = (
    "Summarise what this company does in 3-5 sentences. "
    "Focus on technology, product, and market. Be factual and concise."
)

_STANDARD_RENAMES = {
    'company':              'company_name',
    'Company':              'company_name',
    'industry':             'segment',
    'website':              'companyWebsite',
    'Website':              'companyWebsite',
    'Website URL':          'companyWebsite',
    'company_linkedin_url': 'linkedin',
    'first_name':           'firstName',
    'First Name':           'firstName',
    'last_name':            'lastName',
    'Last Name':            'lastName',
    'Email':                'email',
    'Phone Number':         'phone',
}


# ---------------------------------------------------------------------------
# 1. load_csvs
# ---------------------------------------------------------------------------

def load_csvs(input_glob: str, year_filter: Optional[str] = None) -> pd.DataFrame:
    """
    Glob all CSV files matching *input_glob*, optionally restricting to filenames
    that contain *year_filter*, and concatenate into one DataFrame.
    """
    files = glob_module.glob(input_glob, recursive=True)
    if year_filter:
        files = [f for f in files if year_filter in f]
    if not files:
        raise FileNotFoundError(f'No CSV files found matching: {input_glob}')
    return pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)


# ---------------------------------------------------------------------------
# 2. normalize_columns
# ---------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the project-standard names."""
    return df.rename(columns={k: v for k, v in _STANDARD_RENAMES.items() if k in df.columns})


# ---------------------------------------------------------------------------
# 3. filter_non_commercial
# ---------------------------------------------------------------------------

def filter_non_commercial(df: pd.DataFrame, url_col: str = 'companyWebsite') -> pd.DataFrame:
    """Drop rows whose website TLD is in _NON_COMMERCIAL_TLDS."""
    def _is_commercial(url: str) -> bool:
        if not url or pd.isna(url):
            return False
        ext = tldextract.extract(str(url))
        suffix = f'.{ext.suffix}' if ext.suffix else ''
        return suffix not in _NON_COMMERCIAL_TLDS

    return df[df[url_col].apply(_is_commercial)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. dedup_against_existing
# ---------------------------------------------------------------------------

def dedup_against_existing(
    df: pd.DataFrame,
    existing_glob: str,
    url_col: str = 'companyWebsite',
) -> pd.DataFrame:
    """
    Remove rows from *df* whose *url_col* domain already appears in any parquet
    file matched by *existing_glob*.
    """
    files = glob_module.glob(existing_glob, recursive=True)
    if not files:
        return df

    existing_urls: list[str] = []
    for f in files:
        try:
            tmp = pd.read_parquet(f)
            if url_col in tmp.columns:
                existing_urls.extend(tmp[url_col].dropna().astype(str).tolist())
        except Exception:
            pass

    def _bare(url: str) -> str:
        ext = tldextract.extract(str(url))
        return f'{ext.domain}.{ext.suffix}'.lower()

    existing_domains = {_bare(u) for u in existing_urls}
    mask = ~df[url_col].apply(
        lambda u: _bare(str(u)) if pd.notna(u) else ''
    ).isin(existing_domains)
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. scrape_pages  (async)
# ---------------------------------------------------------------------------

async def _aiohttp_scrape(session: aiohttp.ClientSession, url: str) -> str:
    """Fast first-pass scrape using aiohttp + BeautifulSoup."""
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
    """Playwright fallback for JS-rendered pages."""
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=_PLAYWRIGHT_TIMEOUT, wait_until='domcontentloaded')
            content = await page.inner_text('body')
            await browser.close()
            return ' '.join(content.split())[:_PAGE_TEXT_LIMIT]
    except Exception:
        return 'ERROR'


async def _scrape_one(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    url: str,
) -> dict:
    async with sem:
        text = await _aiohttp_scrape(session, url)
        if text == 'FAILED':
            text = await _playwright_scrape(url)
        return {'companyWebsite': url, 'page_text': text}


async def scrape_pages(df: pd.DataFrame, url_col: str = 'companyWebsite') -> pd.DataFrame:
    """
    Async two-stage scraper (aiohttp first, Playwright fallback).
    Returns a DataFrame with columns [url_col, 'page_text'].
    """
    urls = df[url_col].dropna().unique().tolist()
    sem = asyncio.Semaphore(_MAX_CONCURRENT)
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; MatcherBot/1.0)'}

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [_scrape_one(sem, session, u) for u in urls]
        results = await asyncio.gather(*tasks)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 6. summarize_companies  (async)
# ---------------------------------------------------------------------------

async def _summarize_one(client: AsyncOpenAI, page_text: str) -> str:
    try:
        resp = await client.chat.completions.create(
            model='gpt-3.5-turbo',
            max_tokens=300,
            messages=[
                {'role': 'system', 'content': _SUMMARISE_SYSTEM},
                {'role': 'user',   'content': page_text[:_PAGE_TEXT_LIMIT]},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f'SUMMARY_ERROR: {e}'


async def summarize_companies(
    df: pd.DataFrame,
    openai_client: AsyncOpenAI,
    text_col: str = 'page_text',
    out_col: str = 'company_summary',
) -> pd.DataFrame:
    """
    Concurrently summarise each row's *text_col* into *out_col*.
    Rows where page_text is empty, FAILED, or ERROR are skipped.
    """
    df = df.copy()
    tasks, indices = [], []

    for idx, row in df.iterrows():
        text = str(row.get(text_col, ''))
        if text and text not in ('FAILED', 'ERROR', 'nan'):
            tasks.append(_summarize_one(openai_client, text))
            indices.append(idx)

    results = await asyncio.gather(*tasks)

    df[out_col] = ''
    for idx, result in zip(indices, results):
        df.at[idx, out_col] = result

    return df


# ---------------------------------------------------------------------------
# 7. embed_summaries  (async)
# ---------------------------------------------------------------------------

async def embed_summaries(
    df: pd.DataFrame,
    tp: TextProcessor,
    text_col: str = 'company_summary',
    out_col: str = 'embeddings',
) -> pd.DataFrame:
    """
    Generate embeddings for each row's *text_col* via TextProcessor.get_embedding()
    (run in a thread-pool so the sync call doesn't block the event loop).
    """
    df = df.copy()
    loop = asyncio.get_event_loop()

    async def _embed(text: str):
        return await loop.run_in_executor(None, tp.get_embedding, text)

    tasks, indices = [], []
    for idx, row in df.iterrows():
        text = str(row.get(text_col, ''))
        if text and text not in ('', 'nan'):
            tasks.append(_embed(text))
            indices.append(idx)

    results = await asyncio.gather(*tasks)

    df[out_col] = None
    for idx, result in zip(indices, results):
        df.at[idx, out_col] = result

    return df


# ---------------------------------------------------------------------------
# 8. export_contacts
# ---------------------------------------------------------------------------

def export_contacts(
    df: pd.DataFrame,
    export_path: str,
    source_name: str,
    today: Optional[str] = None,
) -> str:
    """
    Attach uuid / scraped_at columns and save to
    ``{export_path}/{source_name}_{today}.parquet``.
    Returns the saved file path.
    """
    if today is None:
        today = datetime.today().strftime('%Y-%m-%d')

    df = df.copy()
    df['uuid']       = [str(uuid.uuid4()) for _ in range(len(df))]
    df['scraped_at'] = today

    os.makedirs(export_path, exist_ok=True)
    out_path = os.path.join(export_path, f'{source_name}_{today}.parquet')
    df.to_parquet(out_path, index=False)
    print(f'Exported {len(df)} contacts → {out_path}')
    return out_path
