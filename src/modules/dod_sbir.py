"""
dod_sbir.py — Streamlit-free DoD SBIR/STTR (DSIP) search / fetch / embed core.

The DoD SBIR/STTR Innovation Portal (dodsbirsttr.mil/topics-app/) exposes a clean
public JSON API, so a browser agent should never scrape it. This module pulls every
open / pre-release DoD SBIR & STTR topic straight into the canonical grant-topic
store, exactly like `grants_gov.py` does for Grants.gov. Extracted Streamlit-free so
the Grant Sources view AND a future scheduled Cloud Run job share ONE implementation;
progress is reported through optional `progress(done, total)` callables.

Deliberately NO relevance screening (unlike grants_gov): DoD SBIR topics are all
defense R&D solicitations, so a Haiku yes/no screen would pass ~all of them and only
cost calls. Downstream client matching does the real relevance work; the view still
shows a review/edit table so junk can be dropped before save. The summariser is
reused verbatim from grants_gov so the embed text stays identical in shape.

API shape (public, no key):
  search  GET  {SEARCH_URL}?searchParam=<json>&size=100&page=N  -> {total, data:[…]}
               searchParam carries solicitationCycleNames:["openTopics"] +
               topicReleaseStatus:[591,592] (591=Open, 592=Pre-release).
  detail  GET  {DETAIL_URL(topicId)}  -> objective / description / keywords /
               technologyAreas / focusAreas  (the search rows carry no rich text).

The `/topics/api/public/…` path is the real API; the `/topics-app/…` SPA path is
WAF-blocked for non-browser clients. We send a descriptive User-Agent (standard
client identification, not circumvention). Field spellings are read defensively
across a couple of variants because the live shape was verified from a documented
community client rather than a live response (the WAF blocks sandbox probing).
"""

import io
import json
import secrets
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.modules.Embedding.text_embedder import TextProcessor
from src.modules.GoogleBucketManager.bucket_manager import BucketManager
from src.modules import grants_gov as gg           # reuse summarize_all / SUMMARY_SYSTEM
import src.modules.newsletter as nl
import src.modules.sbir_ceilings as sc

BUCKET         = 'cc-matcher-bucket-jeg-v1'
DODSBIR_PREFIX = 'data/all-topics/processed/DOD-SBIR/'

SEARCH_URL = 'https://www.dodsbirsttr.mil/topics/api/public/topics/search'
DETAIL_URL = 'https://www.dodsbirsttr.mil/topics/api/public/topics/{tid}/details'
TOPIC_URL  = 'https://www.dodsbirsttr.mil/topics-app/?topicId={tid}'

PAGE_SIZE      = 100
DETAIL_WORKERS = 5          # modest, so the N-call detail burst is less likely to trip the WAF
MAX_PAGES      = 60          # hard ceiling (60 × 100 = 6k topics) so a bad `total` can't loop forever

RELEASE_OPEN       = 591
RELEASE_PRERELEASE = 592

STATUS_OPTIONS  = {'Open': RELEASE_OPEN, 'Pre-release': RELEASE_PRERELEASE}
_STATUS_LABEL   = {RELEASE_OPEN: 'Open', RELEASE_PRERELEASE: 'Pre-release'}
PROGRAM_OPTIONS = ['SBIR', 'STTR']

_HEADERS = {
    'User-Agent': 'BWCO-Matcher/1.0 (grant-topic ingest; bwcoconsulting.com)',
    'Accept':     'application/json',
}
_TIMEOUT = 30

# The per-topic detail endpoint is hit N times in a burst, so a WAF / rate limiter
# can refuse some. Retry those with backoff instead of silently dropping the text.
_DETAIL_RETRIES = 3
_RETRY_STATUS   = {403, 429, 500, 502, 503, 504}


def _noop(done: int, total: int) -> None:
    """Default progress sink."""


# ── field helpers ─────────────────────────────────────────────────────────────

def _get(d: dict, *keys):
    """First non-empty value across candidate key spellings."""
    for k in keys:
        v = d.get(k)
        if v not in (None, '', [], {}):
            return v
    return None


def _epoch_ms(value) -> str:
    """Epoch-milliseconds -> 'YYYY-MM-DD'; '' for blank / non-positive / unparseable."""
    try:
        ms = float(value)
    except (TypeError, ValueError):
        return ''
    if ms <= 0:
        return ''
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()
    except (ValueError, OverflowError, OSError):
        return ''


def _join(value) -> str:
    """Flatten a keywords / technologyAreas / focusAreas field (str | list[str] |
    list[dict]) into a comma-joined string."""
    if value in (None, '', [], {}):
        return ''
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(_get(value, 'name', 'value', 'text', 'label') or '').strip()
    if isinstance(value, (list, tuple)):
        parts = [_join(v) for v in value]
        return ', '.join(p for p in parts if p)
    return str(value).strip()


def _status_label(code) -> str:
    try:
        return _STATUS_LABEL.get(int(code), str(code or '').strip())
    except (TypeError, ValueError):
        return str(code or '').strip()


def _html_text(value) -> str:
    """Strip HTML from a DSIP rich-text field (objective/description are HTML)."""
    s = str(value or '').strip()
    if not s:
        return ''
    return BeautifulSoup(s, 'html.parser').get_text(' ', strip=True)


def _phase_text(value) -> str:
    """DSIP `phaseHierarchy` is a JSON **string** of phases, each with a
    (HTML) description — where most of the technical substance lives. Return the
    joined phase text, tolerating a dict, a list, or a plain string."""
    if value in (None, '', [], {}):
        return ''
    data = value
    if isinstance(value, str):
        try:
            data = json.loads(value)
        except (ValueError, TypeError):
            return _html_text(value)
    phases = data.get('phases') if isinstance(data, dict) else data
    if not isinstance(phases, (list, tuple)):
        return _html_text(data if isinstance(data, str) else '')
    out = []
    for ph in phases:
        if not isinstance(ph, dict):
            continue
        name = str(_get(ph, 'phase', 'phaseName', 'name') or '').strip()
        body = _html_text(_get(ph, 'description', 'text', 'body'))
        if body:
            out.append(f'{name}: {body}' if name else body)
    return '\n\n'.join(out).strip()


def _detail_text(d: dict) -> str:
    """Assemble the embeddable description from a DSIP detail JSON object."""
    objective = _html_text(_get(d, 'objective'))
    desc      = _html_text(_get(d, 'description', 'topicDescription'))
    phases    = _phase_text(_get(d, 'phaseHierarchy', 'phases'))
    tech      = _join(_get(d, 'technologyAreas', 'technologyArea'))
    focus     = _join(_get(d, 'focusAreas', 'focusArea'))
    keywords  = _join(_get(d, 'keywords'))
    parts = []
    if objective:
        parts.append(f'Objective: {objective}')
    if desc:
        parts.append(desc)
    if phases:
        parts.append(phases)
    if tech:
        parts.append(f'Technology areas: {tech}')
    if focus:
        parts.append(f'Focus areas: {focus}')
    if keywords:
        parts.append(f'Keywords: {keywords}')
    return '\n\n'.join(parts).strip()


# ── search + detail ────────────────────────────────────────────────────────────

def _search_param(statuses: list[int], keyword: str) -> str:
    return json.dumps({
        'searchText':             keyword or None,
        'components':             None,
        'programYear':            None,
        'solicitationCycleNames': ['openTopics'],
        'releaseNumbers':         [],
        'topicReleaseStatus':     list(statuses),
        'modernizationPriorities': [],
        'sortBy':                 'finalTopicCode,asc',
    })


def fetch_detail(topic_id: str) -> tuple[str, str]:
    """Return (assembled description text, status) for one topic. `status` is 'ok',
    'empty' (200 but no text found), 'no_id', 'http_<code>', or 'exc:<type>' — it is
    diagnostic only, so one bad topic never aborts the run. Retries a WAF/rate-limit
    refusal with backoff rather than silently yielding ''."""
    if not topic_id:
        return '', 'no_id'
    last = 'exc:unknown'
    for attempt in range(_DETAIL_RETRIES):
        try:
            r = requests.get(DETAIL_URL.format(tid=topic_id), headers=_HEADERS, timeout=_TIMEOUT)
            if r.status_code in _RETRY_STATUS:
                last = f'http_{r.status_code}'
                time.sleep(0.5 * (attempt + 1))
                continue
            r.raise_for_status()
            d = r.json()
            if isinstance(d, dict) and isinstance(d.get('data'), dict) and 'objective' not in d:
                d = d['data']          # some envelopes wrap the topic in `data`
            text = _detail_text(d) if isinstance(d, dict) else ''
            return (text, 'ok') if text else ('', 'empty')
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else '?'
            return '', f'http_{code}'
        except Exception as e:
            last = f'exc:{type(e).__name__}'
            time.sleep(0.5 * (attempt + 1))
    return '', last


def fetch_all_details(items: list[dict], progress=_noop) -> tuple[list[str], dict]:
    """(texts, report) aligned to `items`. `report` = {ok, empty, failed,
    statuses: {status_label: count}, sample_error} so a run with no descriptions is
    explainable (rate-limited vs wrong field vs empty) instead of silently blank."""
    texts = [''] * len(items)
    report = {'ok': 0, 'empty': 0, 'failed': 0, 'statuses': {}, 'sample_error': ''}
    if not items:
        return texts, report
    counter: Counter = Counter()
    with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as pool:
        futures = {
            pool.submit(fetch_detail, str(_get(it, 'topicId', 'id') or '')): i
            for i, it in enumerate(items)
        }
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            text, status = future.result()
            texts[i] = text
            counter[status] += 1
            if status == 'ok':
                report['ok'] += 1
            elif status == 'empty':
                report['empty'] += 1
            else:
                report['failed'] += 1
                if not report['sample_error']:
                    report['sample_error'] = status
            done += 1
            progress(done, len(items))
    report['statuses'] = dict(counter)
    return texts, report


def items_to_df(items: list[dict], descriptions: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for i, opp in enumerate(items):
        desc     = descriptions[i] if descriptions and i < len(descriptions) else ''
        topic_id = str(_get(opp, 'topicId', 'id') or '').strip()
        cycle    = str(_get(opp, 'cycleName', 'solicitationCycleName') or '').strip()
        rows.append({
            'title':              str(_get(opp, 'topicTitle', 'title') or '').strip(),
            'description':        str(desc).strip(),
            'notice_id':          str(_get(opp, 'topicCode', 'finalTopicCode', 'topicNumber') or topic_id).strip(),
            'agency':             str(_get(opp, 'component', 'componentName', 'agency') or '').strip(),
            'posted_date':        _epoch_ms(_get(opp, 'topicStartDate', 'topicPreReleaseStartDate')),
            'close_date':         _epoch_ms(_get(opp, 'topicEndDate', 'topicClosedDate')),
            'award_ceiling':      '',    # DSIP topics state no per-notice dollar figure
            'status':             _status_label(_get(opp, 'topicStatus', 'topicReleaseStatus', 'releaseStatus')),
            'program':            str(_get(opp, 'program', 'programName') or '').strip().upper(),
            'solicitation_title': str(_get(opp, 'solicitationTitle') or cycle).strip(),
            'source':             TOPIC_URL.format(tid=topic_id) if topic_id else '',
        })
    return pd.DataFrame(rows)


def search_topics(
    programs:    list[str],
    statuses:    list[int],
    keyword:     str,
    max_results: int,
    progress=_noop,
) -> tuple[pd.DataFrame, int, dict]:
    """Page the DSIP search API, filter by program client-side, fetch per-topic
    details for survivors, return (df, total_from_api, detail_report)."""
    param = _search_param(statuses, keyword)
    all_items: list[dict] = []
    total = None
    page = 0
    while page < MAX_PAGES:
        r = requests.get(
            SEARCH_URL, headers=_HEADERS, timeout=_TIMEOUT,
            params={'searchParam': param, 'size': PAGE_SIZE, 'page': page},
        )
        r.raise_for_status()
        body = r.json()
        if total is None:
            total = int(_get(body, 'total', 'totalElements', 'totalRecords') or 0)
        page_items = _get(body, 'data', 'content', 'topics') or []
        if not page_items:
            break
        all_items.extend(page_items)
        if max_results and len(all_items) >= max_results:
            all_items = all_items[:max_results]
            break
        if total and len(all_items) >= total:
            break
        page += 1

    if programs and set(p.upper() for p in programs) != {p.upper() for p in PROGRAM_OPTIONS}:
        want = {p.upper() for p in programs}
        all_items = [it for it in all_items
                     if str(_get(it, 'program', 'programName') or '').strip().upper() in want]

    if not all_items:
        return pd.DataFrame(), total or 0, {'ok': 0, 'empty': 0, 'failed': 0,
                                            'statuses': {}, 'sample_error': ''}
    descriptions, report = fetch_all_details(all_items, progress)
    return items_to_df(all_items, descriptions), total or 0, report


# ── dedup ─────────────────────────────────────────────────────────────────────

def load_existing_keys(client) -> tuple[set[str], set[str]]:
    notice_ids: set[str] = set()
    titles:     set[str] = set()
    for blob in client.list_blobs(BUCKET, prefix=DODSBIR_PREFIX):
        if not blob.name.endswith('.parquet'):
            continue
        try:
            df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()),
                                 columns=['topic_number', 'title'])
            notice_ids.update(df['topic_number'].dropna().astype(str).str.strip())
            titles.update(df['title'].dropna().astype(str).str.lower().str.strip())
        except Exception:
            pass
    return notice_ids, titles


# ── embed + save ──────────────────────────────────────────────────────────────

def embed_and_save(
    df: pd.DataFrame,
    oai_key: str,
    anth_key: str,
    gcs_client,
    extra_cols: dict[str, str] | None = None,
    summary_progress=_noop,
    embed_progress=_noop,
    newsletter_progress=_noop,
) -> str:
    """Summarize → resolve SBIR ceilings → embed → newsletter-tag → save one parquet.

    Writes the canonical grant-topic schema (identical to grants_gov.embed_and_save)
    plus a `source` column, to {DODSBIR_PREFIX}dod_sbir_{date}_{hex}.parquet."""
    tp    = TextProcessor(api_key=oai_key)
    bm    = BucketManager(BUCKET, client=gcs_client)
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
    out['source']        = (df['source'].astype(str) if 'source' in df.columns else '')
    out['scraped_at']    = today

    _rolling_llm = (df['_is_rolling'].fillna(False).astype(bool).values
                    if '_is_rolling' in df.columns else [False] * len(df))
    _blank_cd = out['close_date'].astype(str).str.strip().str.lower().isin(
        ['', 'nan', 'nat', 'none'])
    out['is_rolling']           = (pd.Series(_rolling_llm, index=out.index) | _blank_cd)
    out['last_verified_active'] = today
    out['verify_status']        = 'active'

    summaries = gg.summarize_all(out['title'].tolist(), out['description'].tolist(),
                                 anth_key, summary_progress)
    out['grant_summary'] = summaries

    # Verified SBIR/STTR Phase I/II ceilings from the agency table. DSIP topics carry
    # no per-notice dollar figure, so the DoD table caps apply; the API's own
    # `program` field (SBIR/STTR) drives detection, with title+summary as backup.
    programs = (df['program'].astype(str).tolist() if 'program' in df.columns
                else [''] * len(df))
    ptexts = [f'{t} {s} {p}' for t, s, p in zip(out['title'], summaries, programs)]
    ceils = [sc.resolve_ceilings(
                 broad_agency='DOD', agency=a, program_text=pt,
                 program_type=(p.upper() if p.upper() in sc.PROGRAMS else None))
             for a, pt, p in zip(out['agency'].tolist(), ptexts, programs)]
    for col in sc.CEILING_COLUMNS:
        out[col] = [c[col] for c in ceils]

    embed_texts = [s if s.strip() else d
                   for s, d in zip(summaries, out['description'].tolist())]
    embeddings = []
    for i, text in enumerate(embed_texts):
        embeddings.append(tp.get_embedding(text) if text.strip() else None)
        embed_progress(i + 1, len(embed_texts))
    out['embeddings'] = embeddings

    # Newsletter solicitation_title hint is the parent BAA / cycle name, not the
    # topic title — that is the published solicitation a reader would look up.
    _hints = (df['solicitation_title'].astype(str).tolist()
              if 'solicitation_title' in df.columns else out['title'].tolist())
    nl.tag_frame(out, anth_key, title_hints=_hints, progress=newsletter_progress)

    if extra_cols:
        for col_name, col_val in extra_cols.items():
            out[col_name] = col_val

    hex_suffix = secrets.token_hex(3)
    gcs_path   = f'{DODSBIR_PREFIX}dod_sbir_{today}_{hex_suffix}.parquet'
    bm.upload_file(gcs_path, out)
    return gcs_path
