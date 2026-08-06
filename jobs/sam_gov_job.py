"""
sam_gov_job.py — Cloud Run Job for SAM.gov topic ingestion and revision tracking.

Ingest modes ("api" / "csv") read a job config from GCS, then:
  1. Loads input data (SAM.gov API or CSV blob uploaded to GCS by Streamlit)
  2. Screens rows with Claude Haiku (async, up to 20 concurrent)
  3. Deduplicates against existing records in data/all-topics/processed/SAM-GOV/
     — dup rows whose SAM.gov noticeId changed (amended notices, e.g. revised
     CSOs) are routed to the update path instead of being dropped
  4. Summarizes passing rows with Claude Haiku (async, up to 20 concurrent)
  5. Generates OpenAI text-embedding-ada-002 embeddings (8 concurrent workers)
  6. Saves a parquet to data/all-topics/processed/SAM-GOV/
  7. Writes sam-gov-jobs/{run_id}/status.json

"revision_check" mode sweeps every stored open notice: queries SAM.gov by
solicitation number, detects new versions (noticeId changed), diffs old vs new
content with Claude (including attachment PDF text), re-summarizes/re-embeds
changed notices, rewrites their rows in place, and marks archived notices with
sam_status='archived'.

Usage:
    python jobs/sam_gov_job.py sam-gov-configs/<run_id>.json

Environment variables (injected by Cloud Run from Secret Manager):
    ANTHROPIC_API_KEY
    OPENAI_API_KEY

Config schema:
{
  "run_id":      "sam_gov_2026-06-01_10-30-00",
  "input_mode":  "api" | "csv" | "revision_check",

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

  // revision_check mode only:
  //   "api_params": {"sam_gov_api_key": "...", "include_attachments": true},
  //   "dry_run": true   — report what would change without writing anything

  "custom_cols": {"campaign_name": "Spring 2026"}
}
"""

import asyncio
import io
import json
import os
import re
import secrets as _secrets
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import fitz  # pymupdf — attachment PDF text extraction
import pandas as pd
import pyarrow.parquet as pq
import requests
import tiktoken
from anthropic import Anthropic, AsyncAnthropic
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
_DESC_WORKERS     = 6   # SAM.gov allows 10 req/s — 6 workers is the concurrency cap
_SAM_PAGE_SLEEP   = 3.0  # seconds between pagination requests
_SAM_MAX_RETRIES  = 8   # max retries on throttle 429s (daily-quota 429s abort immediately)
_SAM_RETRY_BASE   = 30  # seconds — throttle 429s clear on a minutes timescale
_SAM_RATE_LOCK    = threading.Lock()
_SAM_LAST_REQ     = [0.0]  # mutable for closure; protected by _SAM_RATE_LOCK
_SAM_MIN_INTERVAL = 1.0 / 8  # 8 req/s — conservative under 10 req/s limit
_SAM_CALL_COUNT   = [0]    # total SAM.gov HTTP requests this run; protected by _SAM_RATE_LOCK
_SAM_QUOTA_FLAG   = threading.Event()  # set once the daily quota 429 is seen — fail fast after
_SCREEN_CONCUR    = 20
_SUMMARY_CONCUR   = 20
_EMBED_WORKERS    = 8
_SCREEN_MODEL     = 'claude-haiku-4-5-20251001'
_SCREEN_MAX_CHARS = 3000
_TOKEN_LIMIT      = 7500

# Revision handling
_ATTACH_MAX_FILES     = 8      # attachments fetched per notice
_ATTACH_MAX_PAGES     = 30     # pages extracted per PDF
_ATTACH_MAX_CHARS     = 40000  # total attachment text per notice
_REV_WORKERS          = 4      # concurrent revised notices (each makes SAM + Claude calls)
_REVCHECK_MAX_WINDOWS = 5      # one-year postedFrom/postedTo windows walked back per solnum lookup
_REVCHECK_STATE_BLOB  = 'sam-gov-configs/revcheck_state.json'  # last-checked cursor per topic_number
_REVCHECK_DEFAULT_BUDGET = 600  # SAM.gov API calls per revision-check run (daily key quota is 1,000)
_NOTICE_ID_RE         = re.compile(r'sam\.gov/opp/([^/]+)/view')

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

_REVISION_SYSTEM = """\
You are comparing two versions of a federal contract opportunity (often a Commercial Solutions Opening) to identify substantive changes between the version we have stored and the latest version on SAM.gov.

You will be given the OLD version (our stored summary and description) and the NEW version (the latest description plus text extracted from attached documents).

Identify:
- Topics, technology areas, or focus areas ADDED in the new version
- Topics, technology areas, or focus areas REMOVED in the new version
- Other substantive changes (deadline, scope, eligibility, funding)

Ignore formatting differences, boilerplate, administrative or points-of-contact changes. Note that the NEW version may include attachment text the OLD version lacked — only report a topic as ADDED if it is genuinely new subject matter, not merely text we did not previously capture for an existing topic.

OUTPUT FORMAT:
Respond only with valid JSON. No preamble, no markdown, no explanation outside the JSON.
{"changed": true, "notes": "One to three sentences describing what changed.", "topics_added": ["..."], "topics_removed": ["..."]}
If nothing substantive changed: {"changed": false, "notes": "No substantive changes.", "topics_added": [], "topics_removed": []}\
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

class QuotaExhaustedError(Exception):
    """SAM.gov daily request quota is spent — it resets only at midnight UTC."""


_API_KEY_RE = re.compile(r'api_key=[^&\s"\']+')


def _redact(text: str) -> str:
    """Strip SAM.gov API keys from text before it reaches logs or status.json —
    requests exceptions embed the full request URL, key included."""
    return _API_KEY_RE.sub('api_key=***', str(text))


def _is_quota_429(r: requests.Response) -> bool:
    """Distinguish daily-quota exhaustion from a transient throttle 429.

    Quota 429s carry a "you have exceeded your quota" body and a Retry-After
    pointing at midnight UTC; throttle 429s clear within seconds to minutes.
    """
    body = r.text.lower()
    if 'quota' in body and 'exceeded' in body:
        return True
    try:
        return int(r.headers.get('Retry-After', 0)) > 1800
    except (TypeError, ValueError):
        return False


def _sam_get(url: str, params: dict, timeout: int = 30) -> requests.Response:
    """GET with proactive rate limiting (8 req/s global) and exponential backoff on 429.

    Raises QuotaExhaustedError as soon as SAM.gov reports the daily quota is
    spent — retrying is pointless until midnight UTC, so callers stop cleanly
    instead of burning the task timeout on doomed backoff waits.
    """
    if _SAM_QUOTA_FLAG.is_set():
        raise QuotaExhaustedError('SAM.gov daily quota already exhausted this run.')
    with _SAM_RATE_LOCK:
        gap = _SAM_MIN_INTERVAL - (time.time() - _SAM_LAST_REQ[0])
        if gap > 0:
            time.sleep(gap)
        _SAM_LAST_REQ[0] = time.time()

    for attempt in range(_SAM_MAX_RETRIES + 1):
        with _SAM_RATE_LOCK:
            _SAM_CALL_COUNT[0] += 1
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 429:
            if _is_quota_429(r):
                _SAM_QUOTA_FLAG.set()
                raise QuotaExhaustedError(
                    'SAM.gov daily request quota exhausted — resets at midnight UTC.'
                )
            if attempt < _SAM_MAX_RETRIES:
                wait = min(_SAM_RETRY_BASE * (2 ** attempt), 600)  # 30, 60, 120, 240, 480, 600, 600, 600s
                print(f'  SAM.gov 429 — waiting {wait}s before retry {attempt + 1}/{_SAM_MAX_RETRIES}', flush=True)
                time.sleep(wait)
                continue
        r.raise_for_status()
        return r
    raise RuntimeError('unreachable')  # loop always returns or raises


def _fetch_one_desc(notice_id: str, api_key: str) -> str:
    try:
        r = _sam_get(_SAM_DESC_BASE, {'noticeid': notice_id, 'api_key': api_key}, timeout=20)
        content = r.text
        if '<' in content:
            return BeautifulSoup(content, 'html.parser').get_text(separator=' ', strip=True)
        return content.strip()
    except QuotaExhaustedError:
        raise
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
    api_key   = api_params['sam_gov_api_key']

    # Support dynamic date range for scheduled daily runs
    if api_params.get('lookback_days'):
        today_dt  = datetime.today()
        date_to   = today_dt.strftime('%m/%d/%Y')
        date_from = (today_dt - timedelta(days=int(api_params['lookback_days']))).strftime('%m/%d/%Y')
    else:
        date_from = api_params['date_from']
        date_to   = api_params['date_to']
    notice_types = api_params.get('notice_types', [])
    keyword      = api_params.get('keyword', '')
    max_results  = int(api_params.get('max_results', 0))

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

    # Descriptions are fetched later, after screening, to avoid burning API quota on rows
    # that will be filtered out.  The caller is responsible for calling _fetch_descriptions_batch
    # on the subset that passes screening when fetch_desc is True.
    return pd.DataFrame({
        'title':       [item.get('title', '')                                                    for item in all_items],
        'description': ['' for _ in all_items],
        'naics_desc':  [item.get('naicsCode', '')                                                for item in all_items],
        'notice_id':   [(item.get('solicitationNumber') or item.get('noticeId', ''))             for item in all_items],
        'agency':      [(item.get('subTier') or item.get('department', ''))                      for item in all_items],
        'posted_date': [item.get('postedDate', '')                                               for item in all_items],
        'deadline':    [(item.get('responseDeadLine') or '')[:10]                                for item in all_items],
        'sam_url':     [f"https://sam.gov/opp/{item['noticeId']}/view" if item.get('noticeId') else '' for item in all_items],
        # Kept for revision detection — the version-specific noticeId and attachment links.
        # Dropped before saving (the output frame selects its columns explicitly).
        '_raw_notice_id':  [item.get('noticeId', '')          for item in all_items],
        '_resource_links': [item.get('resourceLinks') or []   for item in all_items],
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
        'sam_url':     _ser('source_url'),
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


# ── Dedup / store metadata ─────────────────────────────────────────────────────

def _clean(v) -> str:
    if v is None:
        return ''
    try:
        if pd.isna(v):
            return ''
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return '' if s.lower() in ('nan', 'none') else s


def _parse_notice_id(source_url) -> str:
    m = _NOTICE_ID_RE.search(_clean(source_url))
    return m.group(1) if m else ''


def _iso_date(s) -> str:
    """Best-effort YYYY-MM-DD from a stored date string; '' if unparseable."""
    s = _clean(s)[:10]
    if not s:
        return ''
    if '/' in s:
        try:
            return datetime.strptime(s, '%m/%d/%Y').strftime('%Y-%m-%d')
        except ValueError:
            return ''
    return s if len(s) == 10 and s[:4].isdigit() else ''


_META_COLS = ['topic_number', 'title', 'source', 'open_date', 'due_date',
              'grant_summary', 'description', 'notice_version_id', 'sam_status']


def _load_store_meta(client: storage.Client) -> tuple[dict[str, dict], set[str]]:
    """Load SAM-GOV store metadata: (topic_number → record info, all stored titles).

    The per-topic info carries everything revision handling needs — the current
    noticeId (from the notice_version_id column, falling back to the source URL),
    the stored text for diffing, and the parquet blob(s) holding the rows.
    """
    meta:   dict[str, dict] = {}
    titles: set[str] = set()
    for blob in client.list_blobs(_BUCKET, prefix=_SAM_STORE_PREFIX):
        if not blob.name.endswith('.parquet'):
            continue
        try:
            pf   = pq.ParquetFile(io.BytesIO(blob.download_as_bytes()))
            cols = [c for c in _META_COLS if c in set(pf.schema_arrow.names)]
            df   = pf.read(columns=cols).to_pandas()
        except Exception:
            continue
        if 'title' in df.columns:
            titles.update(df['title'].dropna().astype(str).str.lower().str.strip())
        for _, row in df.iterrows():
            tn = _clean(row.get('topic_number'))
            if not tn:
                continue
            entry = meta.get(tn)
            if entry is None:
                entry = meta[tn] = {
                    'title':             _clean(row.get('title')),
                    'open_date':         _clean(row.get('open_date')),
                    'due_date':          _clean(row.get('due_date')),
                    'grant_summary':     _clean(row.get('grant_summary')),
                    'description':       _clean(row.get('description')),
                    'sam_status':        _clean(row.get('sam_status')) or 'active',
                    'notice_version_id': _clean(row.get('notice_version_id')) or _parse_notice_id(row.get('source')),
                    'blobs':             [],
                }
            if blob.name not in entry['blobs']:
                entry['blobs'].append(blob.name)
    return meta, titles


# ── Revision handling ──────────────────────────────────────────────────────────

def _extract_pdf_text(data: bytes, max_chars: int) -> str:
    try:
        chunks: list[str] = []
        total = 0
        with fitz.open(stream=data, filetype='pdf') as doc:
            for i, page in enumerate(doc):
                if i >= _ATTACH_MAX_PAGES or total >= max_chars:
                    break
                text = page.get_text().strip()
                if text:
                    chunks.append(text)
                    total += len(text)
        return '\n'.join(chunks)[:max_chars]
    except Exception:
        return ''


def _fetch_attachments_text(resource_links, api_key: str) -> str:
    """Download a notice's attachments and extract text from the PDFs among them."""
    if resource_links is None or isinstance(resource_links, float):
        return ''
    try:
        links = [str(l).strip() for l in list(resource_links) if str(l).strip()]
    except TypeError:
        return ''
    parts: list[str] = []
    remaining = _ATTACH_MAX_CHARS
    for url in links[:_ATTACH_MAX_FILES]:
        if remaining <= 0:
            break
        try:
            r = _sam_get(url, {'api_key': api_key}, timeout=60)
        except QuotaExhaustedError:
            raise
        except Exception:
            continue
        if not r.content.startswith(b'%PDF'):
            continue
        text = _extract_pdf_text(r.content, remaining)
        if text:
            parts.append(text)
            remaining -= len(text)
    return '\n\n'.join(parts)


def _revision_diff(anth: Anthropic, old_summary: str, old_desc: str, new_text: str) -> dict:
    user_msg = (
        f"OLD VERSION (stored summary):\n{old_summary[:3000]}\n\n"
        f"OLD VERSION (stored description):\n{old_desc[:6000]}\n\n"
        f"NEW VERSION (latest description + attachment text):\n{new_text[:12000]}"
    )
    try:
        resp = anth.messages.create(
            model=_SCREEN_MODEL,
            max_tokens=500,
            system=_REVISION_SYSTEM,
            messages=[{'role': 'user', 'content': user_msg}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
        return json.loads(raw)
    except Exception as e:
        # Default to changed=True so the stored content still gets refreshed.
        return {'changed': True, 'notes': f'Could not diff versions ({e}) — content refreshed.',
                'topics_added': [], 'topics_removed': []}


def _format_revision_notes(diff: dict) -> str:
    parts = [_clean(diff.get('notes'))]
    added   = [str(t).strip() for t in (diff.get('topics_added') or [])  if str(t).strip()]
    removed = [str(t).strip() for t in (diff.get('topics_removed') or []) if str(t).strip()]
    if added:
        parts.append('Topics added: ' + '; '.join(added))
    if removed:
        parts.append('Topics removed: ' + '; '.join(removed))
    return ' | '.join(p for p in parts if p)[:2000]


def _summarize_sync(anth: Anthropic, title: str, text: str) -> str:
    try:
        resp = anth.messages.create(
            model=_SCREEN_MODEL,
            max_tokens=400,
            system=_SUMMARY_SYSTEM,
            messages=[{'role': 'user', 'content': f"Title: {title}\n\nDescription:\n{text[:10000]}"}],
        )
        return resp.content[0].text.strip()
    except Exception:
        return ''


def _process_revisions(
    items: list[dict],
    api_key: str,
    anth_key: str,
    openai_key: str,
    include_attachments: bool,
    dry_run: bool,
) -> tuple[list[dict], list[dict], list[str]]:
    """Fetch latest content for revised notices, diff with Claude, build row updates.

    items: [{topic_number, title, new_notice_id, deadline, resource_links,
             old_summary, old_desc, blobs}]
    Returns (updates for _apply_updates, report entries for status.json,
    topic_numbers deferred because the SAM.gov daily quota ran out mid-fetch —
    those produce no update and get re-detected on the next run).
    In dry_run mode the diff still runs (so the report is useful) but no updates
    are produced.
    """
    anth  = Anthropic(api_key=anth_key)
    oai   = OpenAI(api_key=openai_key)
    enc   = tiktoken.get_encoding('cl100k_base')
    today = datetime.today().strftime('%Y-%m-%d')

    updates: list[dict | None] = [None] * len(items)
    reports: list[dict | None] = [None] * len(items)

    def _one(i: int) -> None:
        item     = items[i]
        new_desc = _fetch_one_desc(item['new_notice_id'], api_key)
        attach   = _fetch_attachments_text(item.get('resource_links'), api_key) if include_attachments else ''
        new_text = '\n\n'.join(t for t in (new_desc, attach) if t).strip()

        if new_text:
            diff = _revision_diff(anth, item['old_summary'], item['old_desc'], new_text)
        else:
            diff = {'changed': False, 'topics_added': [], 'topics_removed': [],
                    'notes': 'New version detected but no description or attachment text could be fetched.'}
        notes = _format_revision_notes(diff)
        reports[i] = {
            'topic_number': item['topic_number'],
            'title':        item['title'],
            'changed':      bool(diff.get('changed')),
            'notes':        notes,
        }
        if dry_run:
            return

        fields: dict = {
            'notice_version_id':  item['new_notice_id'],
            'source':             f"https://sam.gov/opp/{item['new_notice_id']}/view",
            'sam_status':         'active',
            'revised_at':         today,
            'sam_revision_notes': notes,
        }
        if item.get('title'):
            fields['title'] = item['title']
        if item.get('deadline'):
            fields['due_date'] = item['deadline']
        # Only re-summarize/re-embed when the content substantively changed —
        # deadline-only amendments keep the existing summary and embedding.
        if diff.get('changed') and new_text:
            summary   = _summarize_sync(anth, item['title'], new_text)
            embedding = _get_embedding(summary or new_text, oai, enc)
            fields['description'] = new_text
            if summary:
                fields['grant_summary'] = summary
            if embedding is not None:
                fields['embeddings'] = embedding
        updates[i] = {'topic_number': item['topic_number'], 'blobs': item['blobs'], 'fields': fields}

    deferred: list[str] = []
    with ThreadPoolExecutor(max_workers=_REV_WORKERS) as pool:
        futures = {pool.submit(_one, i): i for i in range(len(items))}
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            try:
                future.result()
            except QuotaExhaustedError:
                # Quota died mid-fetch — drop any partial result so nothing
                # half-fetched is written; the notice re-detects next run.
                updates[i] = None
                reports[i] = None
                deferred.append(items[i]['topic_number'])
            except Exception as e:
                reports[i] = {'topic_number': items[i]['topic_number'], 'title': items[i]['title'],
                              'changed': False, 'notes': f'Error processing revision: {_redact(e)}'}
            done += 1
            print(f'  revisions: {done}/{len(items)}', flush=True)

    if deferred:
        print(f'  {len(deferred)} revision(s) deferred — SAM.gov daily quota exhausted', flush=True)
    return [u for u in updates if u], [r for r in reports if r], deferred


def _apply_updates(client: storage.Client, updates: list[dict]) -> int:
    """Rewrite the parquet blob(s) holding each updated notice, patching rows in place.

    Returns the number of distinct notices whose rows were actually updated.
    """
    by_blob: dict[str, list[dict]] = {}
    for u in updates:
        for blob_name in u.get('blobs', []):
            by_blob.setdefault(blob_name, []).append(u)

    updated_topics: set[str] = set()
    for blob_name, blob_updates in by_blob.items():
        blob = client.bucket(_BUCKET).blob(blob_name)
        try:
            df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        except Exception as e:
            print(f'  could not read {blob_name} for update: {e}', flush=True)
            continue
        for col, default in (('notice_version_id', ''), ('sam_status', 'active'),
                             ('revised_at', ''), ('sam_revision_notes', '')):
            if col not in df.columns:
                df[col] = default
        keys    = df['topic_number'].astype(str).str.strip()
        touched = False
        for u in blob_updates:
            mask = keys == u['topic_number']
            if not mask.any():
                continue
            for idx in df.index[mask]:
                for col, val in u['fields'].items():
                    if col not in df.columns:
                        df[col] = ''
                    df.at[idx, col] = val
            touched = True
            updated_topics.add(u['topic_number'])
        if touched:
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            blob.upload_from_string(buf.getvalue(), content_type='application/octet-stream')
            print(f'  rewrote {blob_name} ({len(df)} rows)', flush=True)
    return len(updated_topics)


def _load_revcheck_state(client: storage.Client) -> dict[str, str]:
    """Load the sweep cursor: topic_number → ISO date of its last completed check.

    Advanced only by apply (non-dry) runs, so a dry run and the apply run that
    follows it cover the same chunk of notices.
    """
    try:
        raw = client.bucket(_BUCKET).blob(_REVCHECK_STATE_BLOB).download_as_text()
        return {str(k): str(v) for k, v in (json.loads(raw).get('last_checked') or {}).items()}
    except Exception:
        return {}


def _save_revcheck_state(client: storage.Client, last_checked: dict[str, str]) -> None:
    client.bucket(_BUCKET).blob(_REVCHECK_STATE_BLOB).upload_from_string(
        json.dumps({'last_checked': last_checked}), content_type='application/json'
    )


def _lookup_latest(solnum: str, api_key: str, stored_open_date: str) -> tuple[str, dict | None]:
    """Query SAM.gov for the latest version of a solicitation number.

    The search API requires postedFrom/postedTo (max 1 year apart) and returns
    only the latest active version, so walk one-year windows back from today
    until we find it or pass the original posting date.
    Returns ('found', record) | ('not_found', None) | ('error', None).
    """
    today  = datetime.today()
    open_d = _iso_date(stored_open_date)
    for i in range(_REVCHECK_MAX_WINDOWS):
        w_to   = today - timedelta(days=364 * i)
        w_from = today - timedelta(days=364 * (i + 1))
        params = {
            'api_key':    api_key,
            'solnum':     solnum,
            'postedFrom': w_from.strftime('%m/%d/%Y'),
            'postedTo':   w_to.strftime('%m/%d/%Y'),
            'limit':      10,
            'offset':     0,
        }
        try:
            data = _sam_get(_SAM_API_BASE, params).json()
        except QuotaExhaustedError:
            raise
        except Exception as e:
            print(f'  lookup failed for {solnum}: {_redact(e)}', flush=True)
            return 'error', None
        items = [it for it in (data.get('opportunitiesData') or [])
                 if _clean(it.get('solicitationNumber')) == solnum]
        if items:
            return 'found', max(items, key=lambda it: _clean(it.get('postedDate')))
        if open_d and w_from.strftime('%Y-%m-%d') < open_d:
            break  # searched back past the original posting — it is gone
    return 'not_found', None


def _run_revision_check(gcs: storage.Client, config: dict, run_id: str,
                        anth_key: str, openai_key: str) -> None:
    api_params          = config.get('api_params', {})
    api_key             = api_params['sam_gov_api_key']
    include_attachments = bool(api_params.get('include_attachments', True))
    budget              = int(api_params.get('max_api_calls') or _REVCHECK_DEFAULT_BUDGET)
    dry_run             = bool(config.get('dry_run', False))
    today               = datetime.today().strftime('%Y-%m-%d')

    print(f'Revision check starting (dry_run={dry_run}, budget={budget} API calls)…', flush=True)
    print('Loading stored SAM-GOV records…', flush=True)
    meta, _ = _load_store_meta(gcs)

    candidates = {
        tn: m for tn, m in meta.items()
        if m['sam_status'] != 'archived'
        and (not _iso_date(m['due_date']) or _iso_date(m['due_date']) >= today)
    }
    print(f'  {len(meta):,} stored notices, {len(candidates):,} open candidates to check', flush=True)

    # Sweep least-recently-checked first so a budget-capped run resumes where
    # the previous apply run left off (never-checked notices sort first).
    last_checked = _load_revcheck_state(gcs)
    ordered      = sorted(candidates.items(), key=lambda kv: last_checked.get(kv[0], ''))

    revised:    list[dict] = []
    archived:   list[dict] = []
    backfills:  list[dict] = []
    checked_ok: set[str]   = set()
    stopped_early: str | None = None  # 'quota' | 'budget'
    errors  = 0
    checked = 0
    for tn, m in ordered:
        if _SAM_CALL_COUNT[0] >= budget:
            stopped_early = 'budget'
            break
        checked += 1
        if checked % 25 == 0 or checked == len(candidates):
            print(f'  checked {checked}/{len(candidates)} — '
                  f'{len(revised)} revised, {len(archived)} archived, '
                  f'{_SAM_CALL_COUNT[0]}/{budget} API calls', flush=True)
        try:
            status, rec = _lookup_latest(tn, api_key, m['open_date'])
            if status == 'error':
                errors += 1
                continue
            if status == 'not_found' or _clean((rec or {}).get('active')).lower() == 'no':
                checked_ok.add(tn)
                archived.append({'topic_number': tn, 'title': m['title'], 'blobs': m['blobs']})
                continue
            new_id = _clean(rec.get('noticeId'))
            old_id = m['notice_version_id']
            if not new_id or (old_id and new_id == old_id):
                checked_ok.add(tn)
                continue
            if not old_id:
                # No stored version id (older imports without a source URL) — fall back
                # to comparing description text before treating it as a revision.
                new_desc = _fetch_one_desc(new_id, api_key)
                if ' '.join(new_desc.split()) == ' '.join(m['description'].split()):
                    checked_ok.add(tn)
                    backfills.append({'topic_number': tn, 'blobs': m['blobs'], 'fields': {
                        'notice_version_id': new_id,
                        'source':            f'https://sam.gov/opp/{new_id}/view',
                    }})
                    continue
        except QuotaExhaustedError:
            checked -= 1
            stopped_early = 'quota'
            print('  SAM.gov daily quota exhausted — stopping sweep '
                  '(resumes from here on the next run)', flush=True)
            break
        checked_ok.add(tn)
        revised.append({
            'topic_number':   tn,
            'title':          _clean(rec.get('title')) or m['title'],
            'new_notice_id':  new_id,
            'deadline':       _clean(rec.get('responseDeadLine'))[:10],
            'resource_links': rec.get('resourceLinks') or [],
            'old_summary':    m['grant_summary'],
            'old_desc':       m['description'],
            'blobs':          m['blobs'],
        })

    print(f'  {len(revised)} revised, {len(archived)} archived/gone, {errors} lookup errors', flush=True)

    revisions_detected = len(revised)
    updates:  list[dict] = []
    reports:  list[dict] = []
    deferred: list[str]  = []
    if revised and stopped_early == 'quota':
        # No quota left to fetch revision content — leave these unchecked so the
        # next run re-detects them.
        deferred = [r['topic_number'] for r in revised]
        revised  = []
    elif revised:
        print(f'Processing {len(revised)} revised notices…', flush=True)
        updates, reports, deferred = _process_revisions(
            revised, api_key, anth_key, openai_key, include_attachments, dry_run
        )
    checked_ok.difference_update(deferred)

    rows_updated = 0
    if not dry_run:
        for a in archived:
            updates.append({'topic_number': a['topic_number'], 'blobs': a['blobs'], 'fields': {
                'sam_status':         'archived',
                'sam_revision_notes': f'No longer active on SAM.gov as of {today}.',
            }})
        updates.extend(backfills)
        if updates:
            print(f'Applying updates for {len(updates)} notices…', flush=True)
            rows_updated = _apply_updates(gcs, updates)

        # Advance the sweep cursor (apply runs only — dry runs preview the same
        # chunk the next apply run will process). Prune notices no longer stored.
        for tn in checked_ok:
            last_checked[tn] = today
        _save_revcheck_state(gcs, {tn: d for tn, d in last_checked.items() if tn in meta})

    _write_status(gcs, run_id, {
        'run_id':             run_id,
        'mode':               'revision_check',
        'dry_run':            dry_run,
        'rows_candidates':    len(candidates),
        'rows_checked':       checked,
        'rows_remaining':     len(candidates) - checked,
        'revisions_found':    revisions_detected,
        'revisions_deferred': len(deferred),
        'rows_archived':      len(archived),
        'rows_updated':       rows_updated,
        'lookup_errors':      errors,
        'api_calls_used':     _SAM_CALL_COUNT[0],
        'api_call_budget':    budget,
        'stopped_early':      stopped_early,
        'revisions':          reports[:200],
        'archived':           [{'topic_number': a['topic_number'], 'title': a['title']}
                               for a in archived[:200]],
        'error':              None,
    })
    print('Revision check complete.', flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_blob_path: str) -> None:
    gcs = _gcs()

    print(f'Loading config from {config_blob_path}', flush=True)
    config = json.loads(gcs.bucket(_BUCKET).blob(config_blob_path).download_as_text())

    run_id      = config['run_id']
    # "daily" is the sentinel used by Cloud Scheduler; generate a real ID at runtime
    if run_id == 'daily':
        run_id = f"sam_gov_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}"

    input_mode  = config['input_mode']
    custom_cols = config.get('custom_cols', {})

    anth_key   = _get_secret('anthropic-api-key')
    openai_key = _get_secret('openai-api-key')

    if input_mode == 'revision_check':
        _run_revision_check(gcs, config, run_id, anth_key, openai_key)
        return

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
    store_meta, existing_titles = _load_store_meta(gcs)
    existing_ids = set(store_meta.keys())
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

    # ── Step 3b (API mode): update stored notices whose SAM.gov version changed ─
    # Amendments (e.g. revised CSOs) arrive as duplicates by solicitation number
    # but carry a new noticeId. Instead of dropping them, fetch the new content,
    # diff it with Claude, and rewrite the stored rows in place.
    rows_revised = 0
    revision_reports: list[dict] = []
    if input_mode == 'api' and '_raw_notice_id' in passing.columns:
        rev_items: list[dict] = []
        seen: set[str] = set()
        for _, row in passing[dup_mask].iterrows():
            tn = _clean(row.get('notice_id'))
            stored = store_meta.get(tn)
            if not stored or tn in seen:
                continue
            new_id = _clean(row.get('_raw_notice_id'))
            old_id = stored['notice_version_id']
            if not new_id or not old_id or new_id == old_id:
                continue
            seen.add(tn)
            rev_items.append({
                'topic_number':   tn,
                'title':          _clean(row.get('title')) or stored['title'],
                'new_notice_id':  new_id,
                'deadline':       _clean(row.get('deadline'))[:10],
                'resource_links': row.get('_resource_links'),
                'old_summary':    stored['grant_summary'],
                'old_desc':       stored['description'],
                'blobs':          stored['blobs'],
            })
        if rev_items:
            print(f'{len(rev_items)} stored notices have new versions — updating…', flush=True)
            rev_updates, revision_reports, rev_deferred = _process_revisions(
                rev_items, config['api_params']['sam_gov_api_key'],
                anth_key, openai_key, include_attachments=True, dry_run=False,
            )
            rows_revised = _apply_updates(gcs, rev_updates)
            print(f'  {rows_revised} stored notices updated'
                  + (f', {len(rev_deferred)} deferred (quota)' if rev_deferred else ''),
                  flush=True)

    if new_rows.empty:
        _write_status(gcs, run_id, {
            'run_id': run_id, 'rows_fetched': rows_fetched,
            'rows_passed_screening': rows_passed, 'rows_after_dedup': 0,
            'rows_saved': 0, 'rows_revised': rows_revised,
            'revisions': revision_reports[:200], 'gcs_path': None, 'error': None,
        })
        print('All passing rows already in the store — exiting.', flush=True)
        return

    # ── Step 4 (API mode only): Fetch full descriptions for passing rows ───────
    # Descriptions were intentionally skipped during the initial fetch so we only
    # call the SAM.gov description endpoint for the ~20-40% of rows that survived
    # screening and dedup — typically 5-10x fewer requests than the full fetch.
    if input_mode == 'api' and config.get('api_params', {}).get('fetch_desc', True):
        api_key = config['api_params']['sam_gov_api_key']
        print(f'Fetching descriptions for {rows_dedup:,} passing rows…', flush=True)
        descriptions = _fetch_descriptions_batch(new_rows['notice_id'].tolist(), api_key)
        new_rows = new_rows.copy()
        new_rows['description'] = descriptions

    # ── Step 5: Summarize ──────────────────────────────────────────────────────
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
        new_rows  = new_rows[keep].reset_index(drop=True)
        summaries = [s for s, k in zip(summaries, keep) if k]
        rows_dedup = len(new_rows)

    if new_rows.empty:
        _write_status(gcs, run_id, {
            'run_id': run_id, 'rows_fetched': rows_fetched,
            'rows_passed_screening': rows_passed, 'rows_after_dedup': 0,
            'rows_saved': 0, 'rows_revised': rows_revised,
            'revisions': revision_reports[:200], 'gcs_path': None, 'error': None,
        })
        print('All rows had insufficient descriptions — exiting.', flush=True)
        return

    # ── Step 5: Embed ──────────────────────────────────────────────────────────
    print(f'Generating embeddings for {rows_dedup:,} rows…', flush=True)
    embed_texts = [s if s.strip() else d for s, d in zip(summaries, new_rows['description'].tolist())]
    embeddings  = _embed_all(embed_texts, openai_key)

    # ── Step 6: Build output ───────────────────────────────────────────────────
    today = datetime.today().strftime('%Y-%m-%d')
    if '_raw_notice_id' in new_rows.columns:
        version_ids = new_rows['_raw_notice_id'].astype(str)
    else:
        version_ids = new_rows['sam_url'].astype(str).map(_parse_notice_id) if 'sam_url' in new_rows.columns else ''
    out = pd.DataFrame({
        'topic_number':       new_rows['notice_id'].astype(str),
        'agency':             new_rows['agency'].astype(str),
        'title':              new_rows['title'].astype(str),
        'description':        new_rows['description'].astype(str),
        'open_date':          new_rows['posted_date'].astype(str),
        'due_date':           new_rows['deadline'].astype(str),
        'source':             new_rows['sam_url'].astype(str) if 'sam_url' in new_rows.columns else '',
        'scraped_at':         today,
        'sam_confidence':     new_rows['_confidence'].values,
        'sam_reason':         new_rows['_reason'].values,
        'grant_summary':      summaries,
        'embeddings':         embeddings,
        'notice_version_id':  version_ids,
        'sam_status':         'active',
        'revised_at':         '',
        'sam_revision_notes': '',
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
        'rows_revised':          rows_revised,
        'revisions':             revision_reports[:200],
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
        tb = _redact(traceback.format_exc())
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
