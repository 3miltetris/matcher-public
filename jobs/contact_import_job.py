"""
contact_import_job.py — Cloud Run Job for contact importing.

Reads a job config from GCS, then:
  1. Downloads the staged file (CSV or Excel) from GCS
  2. Applies column mapping → standard contact fields, normalizes URLs
  3. Deduplicates against existing records in data/all-contacts/{source}/
  4. Scrapes company websites (async aiohttp → Playwright fallback)
  5. Summarizes scraped text with GPT-3.5-turbo (10 concurrent workers)
  6. Generates text-embedding-ada-002 embeddings (8 concurrent workers)
  7. Saves parquet to data/all-contacts/{source}/{source}_{date}_{hex6}.parquet
  8. Writes contact-import-jobs/{run_id}/status.json

Usage:
    python jobs/contact_import_job.py contact-import-configs/<run_id>.json

Environment variables (injected by Cloud Run from Secret Manager):
    OPENAI_API_KEY

Config schema:
{
  "run_id":        "contact_import_2026-06-25_10-30-00_apollo",
  "source":        "apollo",
  "file_ext":      ".csv",
  "csv_blob_path": "contact-import-uploads/contact_import_2026-06-25_10-30-00_apollo.csv",
  "col_map": {
    "companyWebsite": "Website URL",
    "companyName":    "Company Name",
    "state":          null,
    "segment":        "Industry",
    "firstName":      "First Name",
    "lastName":       "Last Name",
    "email":          "Email",
    "phone":          "Phone Number"
  }
}
"""

import asyncio
import io
import json
import os
import re
import secrets as _secrets
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import aiohttp
import pandas as pd
import tiktoken
import tldextract
from bs4 import BeautifulSoup
from google.cloud import storage
from openai import OpenAI

# ── Constants ──────────────────────────────────────────────────────────────────

_BUCKET            = 'cc-matcher-bucket-jeg-v1'
_CONTACTS_ROOT     = 'data/all-contacts/'
_STATUS_PREFIX     = 'contact-import-jobs/'
_SCRAPE_TIMEOUT    = 15         # seconds (aiohttp)
_PW_TIMEOUT        = 20_000     # ms (Playwright)
_MAX_CONCURRENT    = 20         # scraping semaphore
_PAGE_TEXT_LIMIT   = 8_000      # chars
_SUMMARISE_WORKERS = 10
_EMBED_WORKERS     = 8
_TOKEN_LIMIT       = 7_500

_SUMMARISE_SYSTEM = (
    'Summarise what this company does in 3-5 sentences. '
    'Focus on technology, product, and market. Be factual and concise.'
)

_STANDARD_FIELDS = [
    'companyWebsite', 'companyName', 'state', 'segment',
    'firstName', 'lastName', 'email', 'phone',
]


# ── Secrets ────────────────────────────────────────────────────────────────────

def _get_secret(name: str) -> str:
    env_var = name.upper().replace('-', '_')
    val = os.environ.get(env_var, '')
    if not val:
        raise RuntimeError(f'Environment variable {env_var} is not set.')
    return val


# ── GCS ────────────────────────────────────────────────────────────────────────

def _gcs() -> storage.Client:
    return storage.Client()


def _write_status(client: storage.Client, run_id: str, payload: dict) -> None:
    client.bucket(_BUCKET).blob(f'{_STATUS_PREFIX}{run_id}/status.json').upload_from_string(
        json.dumps(payload), content_type='application/json'
    )


# ── Text cleanup ──────────────────────────────────────────────────────────────

# Matches Excel HYPERLINK formulas, e.g. =HYPERLINK("https://...", "BROCAM LLC")
_HYPERLINK_RE = re.compile(r'=HYPERLINK\s*\(\s*"[^"]*"\s*,\s*"([^"]*)"\s*\)', re.IGNORECASE)

def _strip_hyperlink(val) -> str:
    """Extract display text from an Excel HYPERLINK formula; return text unchanged otherwise."""
    text = str(val) if not isinstance(val, str) else val
    if not text or text.lower() in ('nan', 'none', ''):
        return ''
    m = _HYPERLINK_RE.match(text.strip())
    return m.group(1) if m else text


# ── URL helpers ────────────────────────────────────────────────────────────────

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


# ── Input loading ──────────────────────────────────────────────────────────────

def _load_file(client: storage.Client, blob_path: str, file_ext: str) -> pd.DataFrame:
    blob_bytes = client.bucket(_BUCKET).blob(blob_path).download_as_bytes()
    if file_ext.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(io.BytesIO(blob_bytes), dtype=str)
    try:
        return pd.read_csv(io.BytesIO(blob_bytes), dtype=str, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(blob_bytes), dtype=str, encoding='latin-1')


def _apply_col_map(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    out = pd.DataFrame()
    for std_field in _STANDARD_FIELDS:
        src_col = col_map.get(std_field)
        if src_col and src_col in df.columns:
            out[std_field] = df[src_col].astype(str).apply(_strip_hyperlink)
        else:
            out[std_field] = ''
    out['companyWebsite'] = out['companyWebsite'].apply(_normalize_url)
    return out[out['companyWebsite'] != ''].reset_index(drop=True)


# ── Dedup ──────────────────────────────────────────────────────────────────────

def _load_existing_domains(client: storage.Client, source: str) -> set[str]:
    prefix  = f'{_CONTACTS_ROOT}{source}/'
    domains: set[str] = set()
    for blob in client.list_blobs(_BUCKET, prefix=prefix):
        if not blob.name.endswith('.parquet'):
            continue
        try:
            df = pd.read_parquet(
                io.BytesIO(blob.download_as_bytes()), columns=['companyWebsite']
            )
            for url in df['companyWebsite'].dropna().astype(str):
                d = _bare_domain(url)
                if d:
                    domains.add(d)
        except Exception:
            pass
    return domains


# ── Async scraping ─────────────────────────────────────────────────────────────

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
    # --no-sandbox and --disable-dev-shm-usage are required in Docker/Cloud Run
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage'],
            )
            page = await browser.new_page()
            await page.goto(url, timeout=_PW_TIMEOUT, wait_until='domcontentloaded')
            content = await page.inner_text('body')
            await browser.close()
            return ' '.join(content.split())[:_PAGE_TEXT_LIMIT]
    except Exception:
        return 'ERROR'


async def _scrape_one(
    sem: asyncio.Semaphore, session: aiohttp.ClientSession, url: str, idx: int
) -> dict:
    async with sem:
        text = await _aiohttp_scrape(session, url)
        if text == 'FAILED':
            text = await _playwright_scrape(url)
    return {'_idx': idx, 'page_text': text}


async def _run_scraping(urls: list[str]) -> list[str]:
    sem     = asyncio.Semaphore(_MAX_CONCURRENT)
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; MatcherBot/1.0)'}
    ordered: dict[int, str] = {}
    done_n  = 0
    total   = len(urls)

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [
            asyncio.create_task(_scrape_one(sem, session, url, i))
            for i, url in enumerate(urls)
        ]
        for coro in asyncio.as_completed(tasks):
            item = await coro
            ordered[item['_idx']] = item['page_text']
            done_n += 1
            if done_n % 50 == 0 or done_n == total:
                print(f'  scraping: {done_n}/{total}', flush=True)

    return [ordered[i] for i in range(total)]


# ── Summarization ──────────────────────────────────────────────────────────────

def _summarize_one(idx: int, text: str, client: OpenAI) -> tuple[int, str]:
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


def _run_summarization(page_texts: list[str], openai_key: str) -> list[str]:
    client    = OpenAI(api_key=openai_key)
    summaries = [''] * len(page_texts)
    done      = 0
    total     = len(page_texts)

    with ThreadPoolExecutor(max_workers=_SUMMARISE_WORKERS) as pool:
        futures = {pool.submit(_summarize_one, i, t, client): i for i, t in enumerate(page_texts)}
        for future in as_completed(futures):
            idx, summary   = future.result()
            summaries[idx] = summary
            done          += 1
            if done % 50 == 0 or done == total:
                print(f'  summarizing: {done}/{total}', flush=True)

    return summaries


# ── Embeddings ─────────────────────────────────────────────────────────────────

def _get_embedding(text: str, oai_client: OpenAI, encoding: tiktoken.Encoding) -> list[float] | None:
    if not text.strip():
        return None
    words = text.split()
    while len(encoding.encode(text)) > _TOKEN_LIMIT:
        words = words[:-5]
        if not words:
            return None
        text = ' '.join(words)
    return oai_client.embeddings.create(input=[text], model='text-embedding-ada-002').data[0].embedding


def _embed_all(texts: list[str], openai_key: str) -> list:
    oai_client = OpenAI(api_key=openai_key)
    encoding   = tiktoken.get_encoding('cl100k_base')
    results    = [None] * len(texts)
    done       = 0
    total      = len(texts)

    with ThreadPoolExecutor(max_workers=_EMBED_WORKERS) as pool:
        futures = {
            pool.submit(_get_embedding, text, oai_client, encoding): i
            for i, text in enumerate(texts)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                print(f'  embedding error (index {i}): {e}', flush=True)
            done += 1
            if done % 50 == 0 or done == total:
                print(f'  embedding: {done}/{total}', flush=True)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_blob_path: str) -> None:
    gcs = _gcs()

    print(f'Loading config from {config_blob_path}', flush=True)
    config = json.loads(gcs.bucket(_BUCKET).blob(config_blob_path).download_as_text())

    run_id        = config['run_id']
    source        = config['source']
    file_ext      = config.get('file_ext', '.csv')
    csv_blob_path = config['csv_blob_path']
    col_map       = config['col_map']

    openai_key = _get_secret('openai-api-key')

    # ── Step 1: Load file ──────────────────────────────────────────────────────
    print(f'Loading input file from {csv_blob_path}…', flush=True)
    raw_df = _load_file(gcs, csv_blob_path, file_ext)
    print(f'  {len(raw_df):,} rows loaded', flush=True)

    # ── Step 2: Apply column mapping ───────────────────────────────────────────
    mapped_df    = _apply_col_map(raw_df, col_map)
    rows_fetched = len(mapped_df)
    print(f'  {rows_fetched:,} rows after column mapping and URL normalization', flush=True)

    if mapped_df.empty:
        _write_status(gcs, run_id, {
            'run_id': run_id, 'rows_fetched': 0, 'rows_after_dedup': 0,
            'rows_scraped_ok': 0, 'rows_saved': 0, 'gcs_path': None, 'error': None,
        })
        print('No rows with valid URLs — exiting.', flush=True)
        return

    # ── Step 3: Dedup ──────────────────────────────────────────────────────────
    print(f'Loading existing domains for source "{source}"…', flush=True)
    existing_domains = _load_existing_domains(gcs, source)
    print(f'  {len(existing_domains):,} existing domains', flush=True)

    mask             = mapped_df['companyWebsite'].apply(lambda u: _bare_domain(u) not in existing_domains)
    new_df           = mapped_df[mask].reset_index(drop=True)
    rows_after_dedup = len(new_df)
    print(
        f'  {rows_fetched - rows_after_dedup:,} duplicates removed, '
        f'{rows_after_dedup:,} new rows',
        flush=True,
    )

    if new_df.empty:
        _write_status(gcs, run_id, {
            'run_id': run_id, 'rows_fetched': rows_fetched, 'rows_after_dedup': 0,
            'rows_scraped_ok': 0, 'rows_saved': 0, 'gcs_path': None, 'error': None,
        })
        print('All rows are duplicates — exiting.', flush=True)
        return

    # ── Step 4: Scrape ─────────────────────────────────────────────────────────
    print(f'Scraping {rows_after_dedup:,} websites…', flush=True)
    urls       = new_df['companyWebsite'].tolist()
    page_texts = asyncio.run(_run_scraping(urls))

    scrape_statuses = [
        'ok' if t not in ('FAILED', 'ERROR', '', 'nan') else 'failed'
        for t in page_texts
    ]
    rows_scraped_ok = sum(1 for s in scrape_statuses if s == 'ok')
    print(
        f'  {rows_scraped_ok:,} scraped OK, '
        f'{rows_after_dedup - rows_scraped_ok:,} failed',
        flush=True,
    )

    # ── Step 5: Summarize ──────────────────────────────────────────────────────
    print(f'Summarizing {rows_after_dedup:,} rows…', flush=True)
    summaries = _run_summarization(page_texts, openai_key)

    # ── Step 6: Filter to ok rows, embed ──────────────────────────────────────
    ok_mask      = [s == 'ok' for s in scrape_statuses]
    ok_df        = new_df[ok_mask].reset_index(drop=True)
    ok_summaries = [s for s, ok in zip(summaries, ok_mask) if ok]

    if ok_df.empty:
        _write_status(gcs, run_id, {
            'run_id': run_id, 'rows_fetched': rows_fetched, 'rows_after_dedup': rows_after_dedup,
            'rows_scraped_ok': 0, 'rows_saved': 0, 'gcs_path': None, 'error': None,
        })
        print('No successfully scraped rows — exiting.', flush=True)
        return

    print(f'Generating embeddings for {len(ok_df):,} rows…', flush=True)
    embeddings = _embed_all(ok_summaries, openai_key)

    # ── Step 7: Build output and save ──────────────────────────────────────────
    today = datetime.today().strftime('%Y-%m-%d')
    out   = ok_df.copy()
    out['company_summary'] = ok_summaries
    out['embeddings']      = embeddings
    out['uuid']            = [str(uuid.uuid4()) for _ in range(len(out))]
    out['scraped_at']      = today

    out        = out[out['embeddings'].notna()].reset_index(drop=True)
    rows_saved = len(out)

    hex_suffix = _secrets.token_hex(3)
    gcs_path   = f'{_CONTACTS_ROOT}{source}/{source}_{today}_{hex_suffix}.parquet'

    buf = io.BytesIO()
    out.to_parquet(buf, index=False)
    buf.seek(0)
    gcs.bucket(_BUCKET).blob(gcs_path).upload_from_file(buf, content_type='application/octet-stream')

    print(f'\nDone. {rows_saved:,} contacts saved → {gcs_path}', flush=True)

    _write_status(gcs, run_id, {
        'run_id':          run_id,
        'rows_fetched':    rows_fetched,
        'rows_after_dedup': rows_after_dedup,
        'rows_scraped_ok': rows_scraped_ok,
        'rows_saved':      rows_saved,
        'gcs_path':        gcs_path,
        'error':           None,
    })


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python jobs/contact_import_job.py <config_blob_path>', file=sys.stderr)
        sys.exit(1)

    run_id_fallback = sys.argv[1].split('/')[-1].replace('.json', '')
    try:
        main(sys.argv[1])
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        try:
            _write_status(
                _gcs(),
                run_id_fallback,
                {'run_id': run_id_fallback, 'error': tb, 'rows_saved': 0},
            )
        except Exception:
            pass
        sys.exit(1)
