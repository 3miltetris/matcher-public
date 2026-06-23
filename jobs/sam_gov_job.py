"""
sam_gov_job.py — Cloud Run Job for SAM.gov topic ingestion.

Reads a job config from GCS, then:
  1. Loads input data (SAM.gov API or CSV blob uploaded to GCS by Streamlit)
  2. Screens rows with Claude Haiku (async, up to 20 concurrent)
  3. Deduplicates against existing records in data/all-topics/processed/SAM-GOV/
  4. Summarizes passing rows with Claude Haiku (async, up to 20 concurrent)
  5. Generates OpenAI text-embedding-ada-002 embeddings (8 concurrent workers)
  6. Saves a parquet to data/all-topics/processed/SAM-GOV/
  7. Writes sam-gov-jobs/{run_id}/status.json

Usage:
    python jobs/sam_gov_job.py sam-gov-configs/<run_id>.json

Environment variables (injected by Cloud Run from Secret Manager):
    ANTHROPIC_API_KEY
    OPENAI_API_KEY

Config schema:
{
  "run_id":      "sam_gov_2026-06-01_10-30-00",
  "input_mode":  "api" | "csv",

  // CSV mode only:
  "csv_blob_path": "sam-gov-uploads/sam_gov_2026-06-01_10-30-00.csv",
  "col_map": {"title": "Opportunity Title", "description": "Synopsis", ...},

  // API mode only:
  "api_params": {
    "date_from":        "01/01/2026",
    "date_to":          "06/01/2026",
    "notice_types":     ["p", "o", "k", "r"],
    "keyword":          "",
    "max_results":      500,
    "fetch_desc":       true,
    "sam_gov_api_key":  "..."
  },

  "custom_cols": {"campaign_name": "Spring 2026"}
}
"""

import asyncio
import io
import json
import os
import secrets as _secrets
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import requests
import tiktoken
from anthropic import AsyncAnthropic
from bs4 import BeautifulSoup
from google.cloud import storage
from openai import OpenAI

# ── Constants ──────────────────────────────────────────────────────────────────

_BUCKET           = 'cc-matcher-bucket-jeg-v1'
_SAM_STORE_PREFIX = 'data/all-topics/processed/SAM-GOV/'
_STATUS_PREFIX    = 'sam-gov-jobs/'
_SAM_API_BASE     = 'https://api.sam.gov/opportunities/v2/search'
_SAM_DESC_BASE    = 'https://api.sam.gov/opportunities/v2/noticedesc'
_SAM_PAGE_SIZE    = 1000
_DESC_WORKERS     = 6   # SAM.gov allows 10 req/s — 6 workers halves description fetch time
_SAM_PAGE_SLEEP   = 1.0  # seconds between pagination requests
_SAM_MAX_RETRIES  = 6   # max retries on 429
_SCREEN_CONCUR    = 20
_SUMMARY_CONCUR   = 20
_EMBED_WORKERS    = 8
_SCREEN_MODEL     = 'claude-haiku-4-5-20251001'
_SCREEN_MAX_CHARS = 3000
_TOKEN_LIMIT      = 7500

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

_SUMMARY_SYSTEM = """\
You are preparing a federal contract opportunity for semantic matching against startup and R&D company profiles.

Summarize the opportunity in 3–5 sentences. Focus exclusively on:
- The specific technical problem, research area, or capability being sought
- Key deliverables or desired technical outcomes
- Relevant domain, technology, or sector (e.g., AI/ML, biotech, defense electronics, advanced manufacturing)

Strip out all procurement boilerplate: FAR clauses, set-aside language, submission deadlines, page limits, administrative instructions, and points of contact. Write in plain technical language. If the description is already short and technical, return it as-is.\
"""


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


# ── SAM.gov API helpers ────────────────────────────────────────────────────────

def _sam_get(url: str, params: dict, timeout: int = 30) -> requests.Response:
    """GET with exponential backoff on 429 responses."""
    for attempt in range(_SAM_MAX_RETRIES):
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 429:
            wait = 2 ** attempt
            print(f'  SAM.gov 429 — waiting {wait}s before retry {attempt + 1}/{_SAM_MAX_RETRIES}', flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    # Final attempt — let the caller handle any error
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r


def _fetch_one_desc(notice_id: str, api_key: str) -> str:
    try:
        r = _sam_get(_SAM_DESC_BASE, {'noticeid': notice_id, 'api_key': api_key}, timeout=20)
        content = r.text
        if '<' in content:
            return BeautifulSoup(content, 'html.parser').get_text(separator=' ', strip=True)
        return content.strip()
    except Exception:
        return ''


def _fetch_descriptions_batch(notice_ids: list[str], api_key: str) -> list[str]:
    results = [''] * len(notice_ids)
    with ThreadPoolExecutor(max_workers=_DESC_WORKERS) as pool:
        futures = {pool.submit(_fetch_one_desc, nid, api_key): i for i, nid in enumerate(notice_ids)}
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            done += 1
            if done % 200 == 0 or done == len(notice_ids):
                print(f'  descriptions: {done}/{len(notice_ids)}', flush=True)
    return results


def _fetch_from_sam(api_params: dict) -> pd.DataFrame:
    api_key      = api_params['sam_gov_api_key']
    date_from    = api_params['date_from']
    date_to      = api_params['date_to']
    notice_types = api_params.get('notice_types', [])
    keyword      = api_params.get('keyword', '')
    max_results  = int(api_params.get('max_results', 0))
    fetch_desc   = bool(api_params.get('fetch_desc', True))

    all_items: list[dict] = []
    offset = 0
    total: int | None = None

    print(f'Fetching from SAM.gov ({date_from} → {date_to})…', flush=True)
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

        if offset > 0:
            time.sleep(_SAM_PAGE_SLEEP)

        r    = _sam_get(_SAM_API_BASE, params)
        data = r.json()

        if total is None:
            total = int(data.get('totalRecords', 0))
            print(f'  SAM.gov total records: {total:,}', flush=True)

        page_items = data.get('opportunitiesData') or []
        all_items.extend(page_items)

        if not page_items:
            break
        if max_results and len(all_items) >= max_results:
            all_items = all_items[:max_results]
            break
        offset += len(page_items)
        if total and offset >= total:
            break

    print(f'  fetched {len(all_items):,} items', flush=True)

    notice_ids = [item.get('noticeId', '') for item in all_items]
    if fetch_desc and notice_ids:
        print('Fetching full descriptions…', flush=True)
        descriptions = _fetch_descriptions_batch(notice_ids, api_key)
    else:
        descriptions = ['' for _ in all_items]

    return pd.DataFrame({
        'title':       [item.get('title', '')                                                    for item in all_items],
        'description': descriptions,
        'naics_desc':  [item.get('naicsCode', '')                                                for item in all_items],
        'notice_id':   [(item.get('solicitationNumber') or item.get('noticeId', ''))             for item in all_items],
        'agency':      [(item.get('subTier') or item.get('department', ''))                      for item in all_items],
        'posted_date': [item.get('postedDate', '')                                               for item in all_items],
        'deadline':    [(item.get('responseDeadLine') or '')[:10]                                for item in all_items],
    })


def _load_from_csv(client: storage.Client, csv_blob_path: str, col_map: dict) -> pd.DataFrame:
    csv_bytes = client.bucket(_BUCKET).blob(csv_blob_path).download_as_bytes()
    try:
        raw = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, encoding='utf-8')
    except UnicodeDecodeError:
        raw = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, encoding='latin-1')

    n = len(raw)
    def _ser(field: str, default: str = '') -> pd.Series:
        col = col_map.get(field)
        return raw[col].fillna('').astype(str) if col else pd.Series([default] * n, dtype=str)

    return pd.DataFrame({
        'title':       _ser('title'),
        'description': _ser('description'),
        'naics_desc':  _ser('naics_desc'),
        'notice_id':   _ser('notice_id'),
        'agency':      _ser('agency', 'SAM-GOV'),
        'posted_date': _ser('posted_date'),
        'deadline':    _ser('deadline'),
    })


# ── Async screening ────────────────────────────────────────────────────────────

async def _screen_all(df: pd.DataFrame, anth_key: str) -> pd.DataFrame:
    titles = df['title'].astype(str).tolist()
    descs  = df['description'].astype(str).tolist()
    naics  = df['naics_desc'].astype(str).tolist() if 'naics_desc' in df.columns else [''] * len(df)

    sem      = asyncio.Semaphore(_SCREEN_CONCUR)
    results: list[dict | None] = [None] * len(df)
    done_count = 0

    async with AsyncAnthropic(api_key=anth_key) as client:
        async def _one(i: int) -> None:
            nonlocal done_count
            async with sem:
                for attempt in range(5):
                    try:
                        resp = await client.messages.create(
                            model=_SCREEN_MODEL,
                            max_tokens=200,
                            system=_SCREEN_SYSTEM,
                            messages=[{'role': 'user', 'content': (
                                f"Title: {titles[i]}\n"
                                f"Description: {descs[i][:_SCREEN_MAX_CHARS]}\n"
                                f"NAICS Descriptor: {naics[i]}"
                            )}],
                        )
                        raw = resp.content[0].text.strip()
                        if raw.startswith('```'):
                            raw = raw.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
                        results[i] = json.loads(raw)
                        break
                    except Exception as e:
                        err = str(e)
                        if any(x in err for x in ('429', 'overloaded', 'rate_limit', 'rate limit')):
                            await asyncio.sleep(2 ** attempt)
                        else:
                            results[i] = {'import': False, 'confidence': 'low', 'reason': f'Error: {e}'}
                            break
                else:
                    results[i] = {'import': False, 'confidence': 'low', 'reason': 'Max retries exceeded'}

            done_count += 1
            if done_count % 200 == 0 or done_count == len(df):
                print(f'  screening: {done_count}/{len(df)}', flush=True)

        await asyncio.gather(*[_one(i) for i in range(len(df))])

    out = df.copy()
    out['_import']     = [(r['import']                  if r else False) for r in results]
    out['_confidence'] = [(r.get('confidence', 'low')   if r else 'low') for r in results]
    out['_reason']     = [(r.get('reason', '')          if r else '')    for r in results]
    return out


# ── Async summarization ────────────────────────────────────────────────────────

async def _summarize_all(titles: list[str], descs: list[str], anth_key: str) -> list[str]:
    results: list[str] = [''] * len(titles)
    done_count = 0
    sem = asyncio.Semaphore(_SUMMARY_CONCUR)

    async with AsyncAnthropic(api_key=anth_key) as client:
        async def _one(i: int) -> None:
            nonlocal done_count
            async with sem:
                for attempt in range(5):
                    try:
                        resp = await client.messages.create(
                            model=_SCREEN_MODEL,
                            max_tokens=400,
                            system=_SUMMARY_SYSTEM,
                            messages=[{'role': 'user', 'content': (
                                f"Title: {titles[i]}\n\nDescription:\n{descs[i][:5000]}"
                            )}],
                        )
                        results[i] = resp.content[0].text.strip()
                        break
                    except Exception as e:
                        err = str(e)
                        if any(x in err for x in ('429', 'overloaded', 'rate_limit', 'rate limit')):
                            await asyncio.sleep(2 ** attempt)
                        else:
                            results[i] = descs[i]
                            break
                else:
                    results[i] = descs[i]

            done_count += 1
            if done_count % 200 == 0 or done_count == len(titles):
                print(f'  summarizing: {done_count}/{len(titles)}', flush=True)

        await asyncio.gather(*[_one(i) for i in range(len(titles))])

    return results


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
            if done % 200 == 0 or done == len(texts):
                print(f'  embedding: {done}/{len(texts)}', flush=True)

    return results


# ── Dedup ──────────────────────────────────────────────────────────────────────

def _load_existing_keys(client: storage.Client) -> tuple[set[str], set[str]]:
    notice_ids: set[str] = set()
    titles:     set[str] = set()
    for blob in client.list_blobs(_BUCKET, prefix=_SAM_STORE_PREFIX):
        if not blob.name.endswith('.parquet'):
            continue
        try:
            df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()), columns=['topic_number', 'title'])
            notice_ids.update(df['topic_number'].dropna().astype(str).str.strip())
            titles.update(df['title'].dropna().astype(str).str.lower().str.strip())
        except Exception:
            pass
    return notice_ids, titles


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_blob_path: str) -> None:
    gcs = _gcs()

    print(f'Loading config from {config_blob_path}', flush=True)
    config = json.loads(gcs.bucket(_BUCKET).blob(config_blob_path).download_as_text())

    run_id      = config['run_id']
    input_mode  = config['input_mode']
    custom_cols = config.get('custom_cols', {})

    anth_key   = _get_secret('anthropic-api-key')
    openai_key = _get_secret('openai-api-key')

    # ── Step 1: Load input ─────────────────────────────────────────────────────
    print(f'Loading input data (mode={input_mode})…', flush=True)
    if input_mode == 'csv':
        df = _load_from_csv(gcs, config['csv_blob_path'], config['col_map'])
    else:
        df = _fetch_from_sam(config['api_params'])

    rows_fetched = len(df)
    print(f'  {rows_fetched:,} rows loaded', flush=True)

    if df.empty:
        _write_status(gcs, run_id, {
            'run_id': run_id, 'rows_fetched': 0, 'rows_passed_screening': 0,
            'rows_after_dedup': 0, 'rows_saved': 0, 'gcs_path': None, 'error': None,
        })
        print('No rows to process — exiting.', flush=True)
        return

    # ── Step 2: Screen ─────────────────────────────────────────────────────────
    print(f'Screening {rows_fetched:,} rows with Claude…', flush=True)
    df_screened = asyncio.run(_screen_all(df, anth_key))
    passing     = df_screened[df_screened['_import'] == True].copy().reset_index(drop=True)
    rows_passed = len(passing)
    print(
        f'  {rows_passed:,} passed screening '
        f'({rows_fetched - rows_passed:,} filtered out)',
        flush=True,
    )

    if passing.empty:
        _write_status(gcs, run_id, {
            'run_id': run_id, 'rows_fetched': rows_fetched, 'rows_passed_screening': 0,
            'rows_after_dedup': 0, 'rows_saved': 0, 'gcs_path': None, 'error': None,
        })
        print('No rows passed screening — exiting.', flush=True)
        return

    # ── Step 3: Dedup ──────────────────────────────────────────────────────────
    print('Loading existing records for dedup…', flush=True)
    existing_ids, existing_titles = _load_existing_keys(gcs)
    print(f'  {len(existing_ids):,} existing notice IDs, {len(existing_titles):,} existing titles', flush=True)

    def _is_dup(row: pd.Series) -> bool:
        nid = str(row.get('notice_id', '') or '').strip()
        if nid and nid in existing_ids:
            return True
        return str(row.get('title', '') or '').strip().lower() in existing_titles

    dup_mask   = passing.apply(_is_dup, axis=1)
    new_rows   = passing[~dup_mask].reset_index(drop=True)
    rows_dedup = len(new_rows)
    print(
        f'  {len(passing) - rows_dedup:,} duplicates skipped, '
        f'{rows_dedup:,} new rows to process',
        flush=True,
    )

    if new_rows.empty:
        _write_status(gcs, run_id, {
            'run_id': run_id, 'rows_fetched': rows_fetched,
            'rows_passed_screening': rows_passed, 'rows_after_dedup': 0,
            'rows_saved': 0, 'gcs_path': None, 'error': None,
        })
        print('All passing rows already in the store — exiting.', flush=True)
        return

    # ── Step 4: Summarize ──────────────────────────────────────────────────────
    print(f'Summarizing {rows_dedup:,} rows…', flush=True)
    summaries = asyncio.run(_summarize_all(
        new_rows['title'].tolist(),
        new_rows['description'].tolist(),
        anth_key,
    ))

    # Drop rows where the LLM couldn't produce a useful summary (sparse/title-only descriptions)
    _THIN_PHRASES = ("don't have enough technical content", "not enough technical content", "i cannot summarize", "i'm unable to summarize")
    thin_mask = [any(p in s.lower() for p in _THIN_PHRASES) for s in summaries]
    n_thin = sum(thin_mask)
    if n_thin:
        print(f'  {n_thin:,} rows dropped — description too sparse to summarize', flush=True)
        keep = [not t for t in thin_mask]
        new_rows  = new_rows[[i for i, k in enumerate(keep) if k]].reset_index(drop=True)
        summaries = [s for s, k in zip(summaries, keep) if k]
        rows_dedup = len(new_rows)

    if new_rows.empty:
        _write_status(gcs, run_id, {
            'run_id': run_id, 'rows_fetched': rows_fetched,
            'rows_passed_screening': rows_passed, 'rows_after_dedup': 0,
            'rows_saved': 0, 'gcs_path': None, 'error': None,
        })
        print('All rows had insufficient descriptions — exiting.', flush=True)
        return

    # ── Step 5: Embed ──────────────────────────────────────────────────────────
    print(f'Generating embeddings for {rows_dedup:,} rows…', flush=True)
    embed_texts = [s if s.strip() else d for s, d in zip(summaries, new_rows['description'].tolist())]
    embeddings  = _embed_all(embed_texts, openai_key)

    # ── Step 6: Build output ───────────────────────────────────────────────────
    today = datetime.today().strftime('%Y-%m-%d')
    out   = pd.DataFrame({
        'topic_number':   new_rows['notice_id'].astype(str),
        'agency':         new_rows['agency'].astype(str),
        'title':          new_rows['title'].astype(str),
        'description':    new_rows['description'].astype(str),
        'open_date':      new_rows['posted_date'].astype(str),
        'due_date':       new_rows['deadline'].astype(str),
        'scraped_at':     today,
        'sam_confidence': new_rows['_confidence'].values,
        'sam_reason':     new_rows['_reason'].values,
        'grant_summary':  summaries,
        'embeddings':     embeddings,
    })

    for col_name, col_val in custom_cols.items():
        out[col_name] = col_val

    # ── Step 7: Save to GCS ────────────────────────────────────────────────────
    hex_suffix = _secrets.token_hex(3)
    gcs_path   = f'{_SAM_STORE_PREFIX}sam_gov_{today}_{hex_suffix}.parquet'

    buf = io.BytesIO()
    out.to_parquet(buf, index=False)
    buf.seek(0)
    gcs.bucket(_BUCKET).blob(gcs_path).upload_from_file(buf, content_type='application/octet-stream')

    rows_saved = len(out)
    print(f'\nDone. {rows_saved:,} rows saved → {gcs_path}', flush=True)

    _write_status(gcs, run_id, {
        'run_id':                run_id,
        'rows_fetched':          rows_fetched,
        'rows_passed_screening': rows_passed,
        'rows_after_dedup':      rows_dedup,
        'rows_saved':            rows_saved,
        'gcs_path':              gcs_path,
        'error':                 None,
    })


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python jobs/sam_gov_job.py <config_blob_path>', file=sys.stderr)
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
