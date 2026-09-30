"""
contact_import_job.py — Cloud Run Job for contact importing.

Reads a job config from GCS, then:
  1. Downloads the staged file (CSV or Excel) from GCS
  2. Applies column mapping → standard contact fields, normalizes URLs
  3. Deduplicates against existing records in data/all-contacts/{source}/
  4. Builds a company profile per row, by profile_method:
       "scrape" (default) — scrape websites (async aiohttp → Playwright
         fallback), summarize with GPT-3.5-turbo (10 concurrent workers)
       "deep_research" — one background OpenAI Deep Research task per
         unique company domain (technology/R&D focus, shared schema from
         src/modules/tech_research.py); the matching summary is assembled
         from the findings and the full output is stored in
         technology_data / technology_summary / technology_updated_at
  4b. Optionally (financial_research) runs a SECOND Deep Research focus —
       financials, src/modules/finance_research.py — over the same unique
       domains, writing financial_data / financial_summary /
       financials_updated_at. These columns ride along on the first parquet
       write, so nothing has to re-read and rewrite the file afterwards.
  5. Generates text-embedding-ada-002 embeddings (8 concurrent workers)
  6. Saves parquet to data/all-contacts/{source}/{source}_{date}_{hex6}.parquet
  7. Emits finance-research-runs/{run_id}/state.json for the financial phase,
     so the Deep Research view and HubSpot Import's "Financial research run"
     mode see these companies exactly like a manually launched run
  8. Writes contact-import-jobs/{run_id}/status.json

Financial research is ADDITIVE and must never cost the import rows: a company
whose financial task fails or runs out of deadline still saves, just without
the financial columns. That is the opposite of the technology focus, where a
failed task leaves no matching summary to embed and the row is dropped.

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
  },
  "profile_method": "scrape",            // or "deep_research" (technology focus)
  "research_model": "gpt-5.6-terra",     // used by BOTH research focuses
  "financial_research": false,           // true = also run the financial focus
  "max_research_companies": 250,         // spend cap; the overflow is deferred,
                                         //   not researched (0/null = no cap)
  "dedup_all_sources": false,            // true = dedup vs all of data/all-contacts/
  "pool":              null              // "prospects" = write into the prospect pool
                                         //   instead of a plain lead-source folder
}

`financial_research` is deliberately orthogonal to `profile_method`:
profile_method answers "how is the matching summary built" (and only the
technology focus can answer it — finance_research has no build_matching_summary
by design), while financial_research adds diligence columns on top of either.
"""

import asyncio
import io
import json
import os
import re
import secrets as _secrets
import sys
import time
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

import src.modules.finance_research as fr
import src.modules.pools as pl
import src.modules.tech_research as tr

# ── Constants ──────────────────────────────────────────────────────────────────

_BUCKET            = 'cc-matcher-bucket-jeg-v1'
_CONTACTS_ROOT     = 'data/all-contacts/'
_STATUS_PREFIX     = 'contact-import-jobs/'
# Financial runs are recorded in the same store the Deep Research view writes,
# so views/finance_researcher.py and HubSpot Import's financial mode (which
# enumerates this prefix) see an import-launched run like any other.
_FIN_RUNS_PREFIX   = 'finance-research-runs/'
_SCRAPE_TIMEOUT    = 15         # seconds (aiohttp)
_PW_TIMEOUT        = 20_000     # ms (Playwright)
_MAX_CONCURRENT    = 8          # scraping semaphore (Playwright processes ~200MB each)
_PAGE_TEXT_LIMIT   = 8_000      # chars
_SUMMARISE_WORKERS = 10
_EMBED_WORKERS     = 8
_TOKEN_LIMIT       = 7_500

# Deep Research profiling — tasks run server-side in parallel; the job just
# polls. Deadline leaves headroom for embedding + save within the 7200s
# Cloud Run task timeout.
#
# The deadline is anchored at JOB START, not at the start of the research
# phase. On the scrape+financial path the scrape runs first, so a deadline
# measured from the research call could push total runtime past the task
# timeout — which kills the container and loses the import entirely.
_RESEARCH_POLL_S     = 30
_RESEARCH_DEADLINE_S = 6_000
_JOB_START           = time.time()

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

def _dedup_prefixes(source: str, all_sources: bool, pool: str | None) -> list[str]:
    """Which contact prefixes an import is deduplicated against.

    A pool import is checked against its own pool AND the client pool: a company
    we already work for is not a prospect, and importing it as one would give it
    two identities that every downstream join would then have to reconcile.
    Plain lead-source imports keep their original scope."""
    if all_sources:
        return [_CONTACTS_ROOT]
    if pool:
        return sorted({pl.contacts_prefix(pool), pl.contacts_prefix(pl.CLIENTS)})
    return [f'{_CONTACTS_ROOT}{source}/']


def _load_existing_domains(
    client: storage.Client, source: str, all_sources: bool = False,
    pool: str | None = None,
) -> set[str]:
    domains: set[str] = set()
    blobs = [
        b for prefix in _dedup_prefixes(source, all_sources, pool)
        for b in client.list_blobs(_BUCKET, prefix=prefix)
    ]
    for blob in blobs:
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


# ── Profile pipelines ─────────────────────────────────────────────────────────

def _run_scrape_pipeline(new_df: pd.DataFrame, openai_key: str):
    """Standard path: scrape → GPT-3.5 summary. Returns
    (ok_df, ok_summaries, rows_ok)."""
    print(f'Scraping {len(new_df):,} websites…', flush=True)
    urls       = new_df['companyWebsite'].tolist()
    page_texts = asyncio.run(_run_scraping(urls))

    scrape_statuses = [
        'ok' if t not in ('FAILED', 'ERROR', '', 'nan') else 'failed'
        for t in page_texts
    ]
    rows_ok = sum(1 for s in scrape_statuses if s == 'ok')
    print(f'  {rows_ok:,} scraped OK, {len(new_df) - rows_ok:,} failed', flush=True)

    print(f'Summarizing {len(new_df):,} rows…', flush=True)
    summaries = _run_summarization(page_texts, openai_key)

    ok_mask      = [s == 'ok' for s in scrape_statuses]
    ok_df        = new_df[ok_mask].reset_index(drop=True)
    ok_summaries = [s for s, ok in zip(summaries, ok_mask) if ok]
    return ok_df, ok_summaries, rows_ok


def _research_deadline() -> float:
    """Wall-clock instant the research phase must be finished by. Anchored at
    job start so a slow scrape shortens research rather than overrunning the
    Cloud Run task timeout."""
    return _JOB_START + _RESEARCH_DEADLINE_S


def _domain_entries(
    df: pd.DataFrame, max_companies: int | None
) -> tuple[pd.Series, dict[str, dict], int]:
    """(bare-domain per row, {domain: identity entry}, n_deferred).

    One entry per unique domain, in order of first appearance so the spend cap
    is deterministic. Rows whose domain is deferred simply get no research —
    they still import."""
    domains = df['companyWebsite'].apply(_bare_domain)

    entries: dict[str, dict] = {}
    deferred = 0
    for i, d in enumerate(domains):
        if not d or d in entries:
            continue
        if max_companies and len(entries) >= max_companies:
            deferred += 1
            continue
        row = df.iloc[i]
        entries[d] = {
            'company_name': str(row.get('companyName') or row.get('company_name') or ''),
            'website':      str(row.get('companyWebsite') or ''),
            'state':        str(row.get('state') or ''),
        }
    return domains, entries, deferred


def _new_tasks(entries: dict[str, dict]) -> dict[str, dict]:
    """A fresh per-focus task record for each identity entry. Focuses must not
    share dicts — each carries its own response_id, output and cost."""
    return {
        d: {**ident, 'response_id': None, 'output': None,
            'error': None, 'cost_usd': 0.0}
        for d, ident in entries.items()
    }


def _run_research_sets(
    oai: OpenAI, model: str, sets: list[dict], deadline: float
) -> None:
    """Launch every background research task across all focus sets, then poll
    them together against one shared deadline.

    Launching both focuses up front and polling them in one loop costs the same
    wall clock as running one — the tasks execute server-side in parallel.
    Running them sequentially would need two deadlines inside one task timeout
    and would starve whichever focus went second.

    Each set is {'label', 'tasks', 'build_prompt', 'fields'}; results are
    written into the task dicts in place.
    """
    pending: list[tuple[dict, dict]] = []   # (set, task)

    for s in sets:
        print(
            f"Launching {s['label']} Deep Research ({model}) for "
            f"{len(s['tasks'])} unique companies…",
            flush=True,
        )
        for d, task in s['tasks'].items():
            try:
                resp = oai.responses.create(
                    model=model,
                    input=s['build_prompt']({
                        'company_name': task['company_name'] or d,
                        'website':      task['website'],
                        'state':        task['state'],
                    }),
                    background=True,
                    tools=[{'type': 'web_search'}],
                )
                task['response_id'] = resp.id
                pending.append((s, task))
            except Exception as e:
                task['error'] = f'launch failed: {e}'

    total = len(pending)
    while pending and time.time() < deadline:
        time.sleep(_RESEARCH_POLL_S)
        still: list[tuple[dict, dict]] = []
        for s, task in pending:
            try:
                resp = oai.responses.retrieve(task['response_id'])
            except Exception:
                still.append((s, task))   # transient — retry next poll
                continue
            if resp.status in ('queued', 'in_progress'):
                still.append((s, task))
                continue
            if resp.status == 'completed':
                if getattr(resp, 'usage', None):
                    task['cost_usd'] = fr.response_cost_usd(model, resp.usage)
                parsed, err = fr.parse_research_output(
                    oai, resp.output_text or '', fields=s['fields']
                )
                if parsed:
                    task['output'] = parsed
                else:
                    task['error'] = err
            else:
                err = getattr(resp, 'error', None)
                task['error'] = str(err) if err else f'research task {resp.status}'
        pending = still
        print(f'  research: {total - len(pending)}/{total} complete', flush=True)

    for s, task in pending:
        task['error'] = 'research deadline exceeded'
        try:
            oai.responses.cancel(task['response_id'])
        except Exception:
            pass

    for s in sets:
        for d, task in s['tasks'].items():
            if task['error']:
                print(f"  [{s['label']}] {d}: {task['error']}", flush=True)


def _tech_set(entries: dict[str, dict]) -> dict:
    return {
        'label':        'technology',
        'tasks':        _new_tasks(entries),
        'build_prompt': tr.build_research_prompt,
        'fields':       tr.ALL_FIELDS,
    }


def _financial_set(entries: dict[str, dict]) -> dict:
    return {
        'label':        'financial',
        'tasks':        _new_tasks(entries),
        'build_prompt': fr.build_research_prompt,
        'fields':       fr.ALL_FIELDS,
    }


def _apply_tech_results(new_df: pd.DataFrame, domains: pd.Series, tasks: dict):
    """Fan technology results out to contact rows. A company with no usable
    result has no matching summary to embed, so its rows are DROPPED.
    Returns (ok_df, ok_summaries)."""
    today = datetime.today().strftime('%Y-%m-%d')
    ok_mask, summaries, tech_data, tech_digests = [], [], [], []
    for i in range(len(new_df)):
        task = tasks.get(domains.iloc[i])
        out  = task['output'] if task else None
        text = tr.build_matching_summary(out).strip() if out else ''
        if out and text:
            ok_mask.append(True)
            summaries.append(text)
            tech_data.append(json.dumps(out))
            tech_digests.append(tr.build_tech_digest(out))
        else:
            ok_mask.append(False)

    ok_df = new_df[ok_mask].reset_index(drop=True)
    ok_df['technology_data']       = tech_data
    ok_df['technology_summary']    = tech_digests
    ok_df['technology_updated_at'] = today
    return ok_df, summaries


def _apply_financial_results(df: pd.DataFrame, tasks: dict) -> pd.DataFrame:
    """Write financial columns onto the rows of each researched company.

    Additive only — a company with no usable result keeps its row and gets
    empty strings, because the financial focus produces nothing the row needs
    in order to embed and match."""
    today   = datetime.today().strftime('%Y-%m-%d')
    domains = df['companyWebsite'].apply(_bare_domain)

    data, digests, updated = [], [], []
    for d in domains:
        task = tasks.get(d)
        out  = task['output'] if task else None
        if out:
            data.append(json.dumps(out))
            digests.append(fr.build_financial_digest(out))
            updated.append(today)
        else:
            data.append('')
            digests.append('')
            updated.append('')

    out_df = df.copy()
    out_df['financial_data']         = data
    out_df['financial_summary']      = digests
    out_df['financials_updated_at']  = updated
    return out_df


def _write_finres_state(
    client: storage.Client, fin_run_id: str, model: str, pool: str | None,
    tasks: dict, saved_keys: set[str],
) -> None:
    """Emit finance-research-runs/{run_id}/state.json in the exact shape
    views/finance_researcher.py writes, so an import-launched financial run is
    resumable, reviewable, and visible to HubSpot Import's financial mode
    (which enumerates this prefix and requires >=1 company with `output`).

    `applied` is True: the columns were written onto the parquet as it was
    created, so there is nothing left to apply. The view still offers the
    button, and re-applying is idempotent — it writes the same values back.
    """
    companies = []
    for idx, (d, task) in enumerate(sorted(tasks.items())):
        key = pl.company_key({
            'company_name':   task['company_name'],
            'companyWebsite': task['website'],
        })
        companies.append({
            'idx':          idx,
            'key':          key,
            'company_name': task['company_name'] or d,
            'website':      task['website'],
            'response_id':  task['response_id'],
            # The view treats only 'completed'/'error' as terminal; anything
            # else makes it re-poll a response this job already consumed.
            'status':       'completed' if task['output'] else 'error',
            'error':        task['error'],
            'cost_usd':     task['cost_usd'],
            'output':       task['output'],
            # False when the company's rows did not survive the import (e.g.
            # its technology research failed), so Apply cannot silently no-op
            # without explanation.
            'row_saved':    key in saved_keys,
        })

    state = {
        'run_id':     fin_run_id,
        'focus':      'financials',
        'pool':       pool,
        'model':      model,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'applied':    True,
        'origin':     'contact-import-job',
        'companies':  companies,
    }
    client.bucket(_BUCKET).blob(
        f'{_FIN_RUNS_PREFIX}{fin_run_id}/state.json'
    ).upload_from_string(json.dumps(state), content_type='application/json')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_blob_path: str) -> None:
    gcs = _gcs()

    print(f'Loading config from {config_blob_path}', flush=True)
    config = json.loads(gcs.bucket(_BUCKET).blob(config_blob_path).download_as_text())

    run_id         = config['run_id']
    source         = config['source']
    file_ext       = config.get('file_ext', '.csv')
    csv_blob_path  = config['csv_blob_path']
    col_map        = config['col_map']
    profile_method = config.get('profile_method', 'scrape')
    research_model = config.get('research_model', 'gpt-5.6-terra')
    financial      = bool(config.get('financial_research', False))
    # Spend cap on unique companies researched (either focus). 0/None = no cap.
    max_research   = config.get('max_research_companies') or None
    dedup_all      = bool(config.get('dedup_all_sources', False))
    # Destination pool (src/modules/pools.py). None = a plain lead-source
    # folder, which is what every config written before pools existed means.
    pool           = config.get('pool') or None
    if pool and not pl.is_pool(pool):
        raise ValueError(f'config named an unknown pool: {pool!r}')

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
    scope = ('ALL sources' if dedup_all
             else ', '.join(_dedup_prefixes(source, dedup_all, pool)))
    print(f'Loading existing domains for {scope}…', flush=True)
    existing_domains = _load_existing_domains(
        gcs, source, all_sources=dedup_all, pool=pool
    )
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

    # ── Step 4: Build company profiles (scrape or Deep Research) ─────────────
    research_extras: dict = {'profile_method': profile_method}
    fin_tasks: dict = {}

    if profile_method == 'deep_research':
        # Both focuses cover the same unique domains, so plan them once and
        # launch them together. A domain whose technology research fails loses
        # its rows even if its financial research succeeded — that spend is
        # accepted in exchange for not running the two focuses back to back,
        # which would not fit inside one task timeout.
        oai = OpenAI(api_key=openai_key)
        domains, entries, deferred = _domain_entries(new_df, max_research)
        if deferred:
            print(f'  spend cap: {deferred} companies deferred (not researched)', flush=True)

        tech_s = _tech_set(entries)
        sets   = [tech_s] + ([_financial_set(entries)] if financial else [])
        _run_research_sets(oai, research_model, sets, _research_deadline())

        ok_df, ok_summaries = _apply_tech_results(new_df, domains, tech_s['tasks'])
        rows_scraped_ok     = len(ok_df)
        tech_ok = sum(1 for t in tech_s['tasks'].values() if t['output'])
        tech_cost = sum(t['cost_usd'] for t in tech_s['tasks'].values())
        print(
            f'  {tech_ok}/{len(entries)} companies researched OK '
            f'(${tech_cost:,.2f}), {rows_scraped_ok:,} contact rows profiled',
            flush=True,
        )
        research_extras.update({
            'research_model':              research_model,
            'companies_researched':        len(entries),
            'companies_research_ok':       tech_ok,
            'companies_research_deferred': deferred,
            'research_cost_usd':           round(tech_cost, 2),
        })
        if financial:
            fin_tasks = sets[1]['tasks']
        del raw_df, mapped_df, new_df
    else:
        ok_df, ok_summaries, rows_scraped_ok = _run_scrape_pipeline(new_df, openai_key)
        # Free large objects before embedding — page text can be GBs for big imports
        del raw_df, mapped_df, new_df

        if financial and not ok_df.empty:
            # Research only the rows that survived scraping — no wasted spend.
            # The shared deadline has already been eaten into by the scrape.
            oai = OpenAI(api_key=openai_key)
            _, entries, deferred = _domain_entries(ok_df, max_research)
            if deferred:
                print(f'  spend cap: {deferred} companies deferred (not researched)', flush=True)
            fin_s = _financial_set(entries)
            _run_research_sets(oai, research_model, [fin_s], _research_deadline())
            fin_tasks = fin_s['tasks']
            research_extras['companies_research_deferred'] = deferred

    # Financial columns ride along on the first parquet write — no rewrite.
    if fin_tasks:
        ok_df    = _apply_financial_results(ok_df, fin_tasks)
        fin_ok   = sum(1 for t in fin_tasks.values() if t['output'])
        fin_cost = sum(t['cost_usd'] for t in fin_tasks.values())
        print(
            f'  financial: {fin_ok}/{len(fin_tasks)} companies researched OK '
            f'(${fin_cost:,.2f})',
            flush=True,
        )
        research_extras.update({
            'financial_research':     True,
            'research_model':         research_model,
            'companies_financial':    len(fin_tasks),
            'companies_financial_ok': fin_ok,
            'financial_cost_usd':     round(fin_cost, 2),
        })

    # Record the financial phase as a normal Deep Research run. Emitted even
    # when no rows survive: the research was paid for, and the run state keeps
    # it reviewable and importable to HubSpot rather than silently discarded.
    fin_run_id = None
    if fin_tasks:
        fin_run_id = f"finres_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_import"
        research_extras['financial_run_id'] = fin_run_id

    def _emit_fin_run(saved_keys: set[str]) -> None:
        if not fin_run_id:
            return
        try:
            _write_finres_state(
                gcs, fin_run_id, research_model, pool, fin_tasks, saved_keys
            )
            print(f'Financial run recorded → {_FIN_RUNS_PREFIX}{fin_run_id}/state.json',
                  flush=True)
        except Exception as e:
            # The columns are already on the rows; losing the run record must
            # not fail an otherwise good import.
            print(f'WARNING: could not write financial run state: {e}', flush=True)

    if ok_df.empty:
        _emit_fin_run(set())
        _write_status(gcs, run_id, {
            'run_id': run_id, 'rows_fetched': rows_fetched, 'rows_after_dedup': rows_after_dedup,
            'rows_scraped_ok': 0, 'rows_saved': 0, 'gcs_path': None, 'error': None,
            **research_extras,
        })
        print('No successfully profiled rows — exiting.', flush=True)
        return

    # ── Step 5: Embed ──────────────────────────────────────────────────────────
    print(f'Generating embeddings for {len(ok_df):,} rows…', flush=True)
    embeddings = _embed_all(ok_summaries, openai_key)

    # ── Step 6: Build output and save ──────────────────────────────────────────
    today = datetime.today().strftime('%Y-%m-%d')
    out   = ok_df.copy()
    out['company_summary'] = ok_summaries
    out['embeddings']      = embeddings
    out['uuid']            = [str(uuid.uuid4()) for _ in range(len(out))]
    out['scraped_at']      = today

    out        = out[out['embeddings'].notna()].reset_index(drop=True)
    # A pool parquet is written in the clients column convention
    # (company_name / summary) rather than the lead convention
    # (companyName / company_summary), so every pool-aware view and job reads
    # one spelling. normalize_company_columns RENAMES rather than copies, so a
    # later edit to `summary` can never leave a stale `company_summary` behind.
    if pool:
        out = pl.normalize_company_columns(out)
    rows_saved = len(out)

    hex_suffix = _secrets.token_hex(3)
    prefix     = pl.contacts_prefix(pool) if pool else f'{_CONTACTS_ROOT}{source}/'
    gcs_path   = f'{prefix}{source}_{today}_{hex_suffix}.parquet'

    buf = io.BytesIO()
    out.to_parquet(buf, index=False)
    buf.seek(0)
    gcs.bucket(_BUCKET).blob(gcs_path).upload_from_file(buf, content_type='application/octet-stream')

    print(f'\nDone. {rows_saved:,} contacts saved → {gcs_path}', flush=True)

    # Keys are read off the SAVED frame, after the column rename, so they are
    # exactly the identities pl.key_mask will match if anyone re-applies.
    _emit_fin_run({pl.company_key(r) for _, r in out.iterrows()})

    _write_status(gcs, run_id, {
        'run_id':          run_id,
        'rows_fetched':    rows_fetched,
        'rows_after_dedup': rows_after_dedup,
        'rows_scraped_ok': rows_scraped_ok,
        'rows_saved':      rows_saved,
        'gcs_path':        gcs_path,
        'pool':            pool,
        'error':           None,
        **research_extras,
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
