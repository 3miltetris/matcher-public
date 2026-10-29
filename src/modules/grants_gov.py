"""
grants_gov.py — Streamlit-free Grants.gov search / screen / summarize / embed core.

Extracted from views/grants_gov_fetch.py so the view and the scheduled
grants-gov-job share ONE implementation (the SAM.gov view+job pattern). Progress
is reported through optional `progress(done, total)` callables — the view passes
`st.progress`-backed callbacks, the job passes logging ones or None — so nothing
here imports streamlit. Every function is client/key-injected.

The screening prompt carries the same supply/sole-source/procurement
disqualifiers as the SAM.gov screen: Grants.gov is grants rather than contracts,
but the Procurement Contract instrument can still surface a non-R&D buy, and one
disqualifier list is one fewer thing to drift.
"""

import io
import json
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime

import pandas as pd
import requests
from anthropic import Anthropic
from bs4 import BeautifulSoup

from src.modules.Embedding.text_embedder import TextProcessor
from src.modules.GoogleBucketManager.bucket_manager import BucketManager
import src.modules.anthropic_utils as au
import src.modules.newsletter as nl
import src.modules.sbir_ceilings as sc

BUCKET         = 'cc-matcher-bucket-jeg-v1'
GRANTS_PREFIX  = 'data/all-topics/processed/GRANTS-GOV/'

SCREEN_MODEL     = 'claude-haiku-4-5-20251001'
SCREEN_WORKERS   = 8
SCREEN_MAX_CHARS = 3000

SEARCH_URL    = 'https://api.grants.gov/v1/api/search2'
DETAIL_URL    = 'https://apply07.grants.gov/grantsws/rest/opportunity/details'
PAGE_SIZE     = 25
DETAIL_WORKERS = 8

STATUS_OPTIONS = {
    'Posted':     'posted',
    'Forecasted': 'forecasted',
    'Closed':     'closed',
    'Archived':   'archived',
}

INSTRUMENT_OPTIONS = {
    'Grant':                 'G',
    'Cooperative Agreement': 'CA',
    'Procurement Contract':  'PC',
    'Other':                 'O',
}


def _noop(done: int, total: int) -> None:
    """Default progress sink."""


# ── API ──────────────────────────────────────────────────────────────────────

def parse_date(s) -> date | None:
    for fmt in ('%m/%d/%Y', '%Y-%m-%d', '%Y/%m/%d'):
        try:
            return datetime.strptime(str(s).strip(), fmt).date()
        except (ValueError, TypeError):
            pass
    return None


def _clean_ceiling_str(value) -> str:
    """Raw awardCeiling as a display string; '' for a blank or an unset-0."""
    s = str(value or '').strip()
    return '' if s in ('', '0', '0.0', '0.00') else s


def items_to_df(items: list[dict], descriptions: list[str] | None = None,
                synopses: list[dict] | None = None) -> pd.DataFrame:
    rows = []
    for i, opp in enumerate(items):
        desc = descriptions[i] if descriptions and i < len(descriptions) else ''
        syn  = (synopses[i] if synopses and i < len(synopses) else {}) or {}
        rows.append({
            'title':         str(opp.get('title') or '').strip(),
            'description':   str(desc).strip(),
            'notice_id':     str(opp.get('number') or opp.get('id') or '').strip(),
            'agency':        str(opp.get('agency') or '').strip(),
            'agency_code':   str(opp.get('agencyCode') or '').strip(),
            'posted_date':   str(opp.get('openDate') or '').strip(),
            'close_date':    str(opp.get('closeDate') or '').strip(),
            'award_ceiling': _clean_ceiling_str(syn.get('awardCeiling')),
            'status':        str(opp.get('oppStatus') or '').strip(),
            # Structured ceiling fields from the detail endpoint, consumed by
            # embed_and_save's resolve_ceilings. Kept as hidden *scalar* columns
            # (not a dict) so they survive st.dataframe preview / dedup unharmed,
            # and are not persisted (a fresh `out` frame is built there).
            '_gg_award_ceiling': syn.get('awardCeiling'),
            '_gg_award_floor':   syn.get('awardFloor'),
        })
    return pd.DataFrame(rows)


def fetch_description(opp_id: str, doc_type: str) -> tuple[str, dict]:
    """Return (plain-text description, structured-ceiling fields) for one
    opportunity. The detail endpoint carries `awardCeiling`/`awardFloor`/
    `estimatedFunding` on the synopsis (or forecast) object alongside the text;
    both are returned so the ceiling is captured instead of discarded."""
    try:
        r = requests.post(
            DETAIL_URL,
            data={'oppId': opp_id},
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=15,
        )
        r.raise_for_status()
        d = r.json()
        obj = (d.get('forecast') or {}) if doc_type == 'forecast' else (d.get('synopsis') or {})
        html = (obj.get('forecastDesc') if doc_type == 'forecast' else obj.get('synopsisDesc')) or ''
        text = BeautifulSoup(html, 'html.parser').get_text(separator=' ', strip=True)
        fields = {
            'awardCeiling':     obj.get('awardCeiling'),
            'awardFloor':       obj.get('awardFloor'),
            'estimatedFunding': obj.get('estimatedFunding'),
        }
        return text, fields
    except Exception:
        return '', {}


def fetch_all_descriptions(items: list[dict], progress=_noop) -> tuple[list[str], list[dict]]:
    """(texts, synopsis-field dicts) aligned to `items`."""
    texts    = [''] * len(items)
    synopses = [{} for _ in items]
    if not items:
        return texts, synopses
    with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as pool:
        futures = {
            pool.submit(fetch_description, str(item['id']), item.get('docType', 'synopsis')): i
            for i, item in enumerate(items)
        }
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            texts[i], synopses[i] = future.result()
            done += 1
            progress(done, len(items))
    return texts, synopses


def search_grants(
    keyword:         str,
    statuses:        list[str],
    instruments:     list[str],
    agencies:        list[str],
    date_from:       date | None,
    date_to:         date | None,
    exclude_expired: bool,
    max_results:     int,
    progress=_noop,
) -> tuple[pd.DataFrame, int]:
    """Query search2 (paged), apply the exclude-expired + exact posted-date-range
    client-side filters, fetch descriptions for survivors, return (df, hit_count).
    """
    all_items: list[dict] = []
    start = 1
    total = None

    date_range_days: int | None = None
    if date_from:
        days_back = (datetime.today().date() - date_from).days + 1
        if 1 <= days_back <= 365:
            date_range_days = days_back

    while True:
        payload: dict = {
            'oppStatuses': '|'.join(statuses),
            'rows':        PAGE_SIZE,
            'startRecord': start,
        }
        if keyword:
            payload['keyword'] = keyword
        if instruments:
            payload['fundingInstruments'] = '|'.join(instruments)
        if agencies:
            payload['agencies'] = '|'.join(agencies)
        if date_range_days:
            payload['dateRange'] = str(date_range_days)

        r = requests.post(SEARCH_URL, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json().get('data', {})

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

    if not all_items:
        return pd.DataFrame(), total or 0

    keep  = [True] * len(all_items)
    today = datetime.today().date()
    for i, item in enumerate(all_items):
        if exclude_expired:
            cd = parse_date(str(item.get('closeDate') or ''))
            if cd is not None and cd < today:
                keep[i] = False
                continue
        if date_from and date_to:
            posted = parse_date(str(item.get('openDate') or ''))
            if posted is None or posted < date_from or posted > date_to:
                keep[i] = False

    filtered = [item for item, k in zip(all_items, keep) if k]
    descriptions, synopses = (fetch_all_descriptions(filtered, progress)
                              if filtered else ([], []))
    return items_to_df(filtered, descriptions, synopses), total or 0


# ── screening ─────────────────────────────────────────────────────────────────

SCREEN_SYSTEM = """\
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
- It is a COMMODITY, SUPPLY, or PRODUCT purchase — buying goods, materials, or finished product rather than funding research or development (including requiring the offeror to already hold FDA approval or sell an established commercial product)
- It is a SOLE-SOURCE notice, brand-name justification, or NOTICE OF INTENT TO AWARD, or an award that has ALREADY been made
- The description makes clear that large-scale prior program experience beyond typical small business reach is required

When uncertain, only import if the opportunity is clearly relevant.

Also decide whether this is a ROLLING opportunity: one that accepts applications continuously or on an ongoing basis rather than by a single fixed deadline. Set "is_rolling": true for "applications accepted on an ongoing basis", continuously-open or standing programs, and anything with no close date. Set it false for an opportunity with one specific close date.

---

OUTPUT FORMAT:
Respond only with valid JSON. No preamble, no markdown, no explanation outside the JSON.
{"import": true, "confidence": "high", "is_rolling": false, "reason": "One or two sentences explaining the decision."}\
"""


def screen_one(title: str, desc: str, anth_key: str) -> dict:
    client   = Anthropic(api_key=anth_key)
    user_msg = f"Title: {title}\n\nDescription: {str(desc)[:SCREEN_MAX_CHARS]}"
    resp = client.messages.create(
        model=SCREEN_MODEL,
        max_tokens=200,
        system=SCREEN_SYSTEM,
        messages=[{'role': 'user', 'content': user_msg}],
    )
    raw = au.response_text(resp)
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    return json.loads(raw)


def run_screening(df: pd.DataFrame, anth_key: str, progress=_noop) -> pd.DataFrame:
    titles  = df['title'].astype(str).tolist()
    descs   = df['description'].astype(str).tolist()
    results = [None] * len(df)
    with ThreadPoolExecutor(max_workers=SCREEN_WORKERS) as pool:
        futures = {pool.submit(screen_one, titles[i], descs[i], anth_key): i
                   for i in range(len(df))}
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                results[i] = {'import': False, 'confidence': 'low',
                              'reason': f'Screening error: {e}'}
            done += 1
            progress(done, len(df))
    out = df.copy()
    out['_import']     = [r['import']     for r in results]
    out['_confidence'] = [r['confidence'] for r in results]
    out['_reason']     = [r['reason']     for r in results]
    out['_is_rolling'] = [bool(r.get('is_rolling')) for r in results]
    return out


# ── summarization ─────────────────────────────────────────────────────────────

SUMMARY_SYSTEM = """\
You are preparing a federal grant opportunity for semantic matching against startup and R&D company profiles.

Summarize the opportunity in 3–5 sentences. Focus exclusively on:
- The specific technical problem, research area, or capability being sought
- Key deliverables or desired technical outcomes
- Relevant domain, technology, or sector (e.g., AI/ML, biotech, defense electronics, advanced manufacturing)

Strip out all procurement boilerplate: FAR clauses, set-aside language, submission deadlines, page limits, administrative instructions, and points of contact. Write in plain technical language. If the description is already short and technical, return it as-is.\
"""


def summarize_one(title: str, desc: str, anth_key: str) -> str:
    client   = Anthropic(api_key=anth_key)
    user_msg = f"Title: {title}\n\nDescription:\n{str(desc)[:5000]}"
    resp = client.messages.create(
        model=SCREEN_MODEL,
        max_tokens=400,
        system=SUMMARY_SYSTEM,
        messages=[{'role': 'user', 'content': user_msg}],
    )
    return au.response_text(resp)


def summarize_all(titles: list[str], descs: list[str], anth_key: str,
                  progress=_noop) -> list[str]:
    results = [''] * len(descs)
    with ThreadPoolExecutor(max_workers=SCREEN_WORKERS) as pool:
        futures = {pool.submit(summarize_one, titles[i], descs[i], anth_key): i
                   for i in range(len(descs))}
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception:
                results[i] = descs[i]
            done += 1
            progress(done, len(descs))
    return results


# ── dedup ─────────────────────────────────────────────────────────────────────

def load_existing_keys(client) -> tuple[set[str], set[str]]:
    notice_ids: set[str] = set()
    titles:     set[str] = set()
    for blob in client.list_blobs(BUCKET, prefix=GRANTS_PREFIX):
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
    out['scraped_at']    = today

    _rolling_llm = (df['_is_rolling'].fillna(False).astype(bool).values
                    if '_is_rolling' in df.columns else [False] * len(df))
    _blank_cd = out['close_date'].astype(str).str.strip().str.lower().isin(
        ['', 'nan', 'nat', 'none'])
    out['is_rolling']           = (pd.Series(_rolling_llm, index=out.index) | _blank_cd)
    out['last_verified_active'] = today
    out['verify_status']        = 'active'

    summaries = summarize_all(out['title'].tolist(), out['description'].tolist(),
                              anth_key, summary_progress)
    out['grant_summary'] = summaries

    # Verified ceilings: Grants.gov's structured awardCeiling/awardFloor (captured
    # by fetch_description) when present, else the SBIR/STTR agency table for topics
    # that look like SBIR/STTR. program_text includes the summary so "SBIR"/"STTR"
    # in the body is seen.
    _ceil = (df['_gg_award_ceiling'].tolist() if '_gg_award_ceiling' in df.columns
             else [None] * len(df))
    _floor = (df['_gg_award_floor'].tolist() if '_gg_award_floor' in df.columns
              else [None] * len(df))
    syns = [{'awardCeiling': c, 'awardFloor': f} for c, f in zip(_ceil, _floor)]
    ptexts = [f'{t} {d} {s}' for t, d, s in
              zip(out['title'], out['description'], summaries)]
    ceils = [sc.resolve_ceilings(broad_agency='GRANTS-GOV', agency=a,
                                 program_text=pt, gg_synopsis=syn)
             for a, pt, syn in zip(out['agency'].tolist(), ptexts, syns)]
    for col in sc.CEILING_COLUMNS:
        out[col] = [c[col] for c in ceils]

    embed_texts = [s if s.strip() else d
                   for s, d in zip(summaries, out['description'].tolist())]
    embeddings = []
    for i, text in enumerate(embed_texts):
        embeddings.append(tp.get_embedding(text) if text.strip() else None)
        embed_progress(i + 1, len(embed_texts))
    out['embeddings'] = embeddings

    nl.tag_frame(out, anth_key, title_hints=out['title'].tolist(),
                 progress=newsletter_progress)

    if extra_cols:
        for col_name, col_val in extra_cols.items():
            out[col_name] = col_val

    hex_suffix = secrets.token_hex(3)
    gcs_path   = f'{GRANTS_PREFIX}grants_gov_{today}_{hex_suffix}.parquet'
    bm.upload_file(gcs_path, out)
    return gcs_path


# ── re-verify (liveness) ────────────────────────────────────────────────────
# Shared by the Funding → Grants.gov view's manual button (rolling_only) and the
# scheduled grants-gov-job reverify mode (all open topics). Grants.gov returns a
# definitive oppStatus, so there is no consecutive-miss guard (unlike scraped
# funding sources).

def lookup_opp_status(number: str) -> tuple[bool, str | None]:
    """Current Grants.gov status for an opportunity number.

    Returns (ok, status): ok=False means the lookup itself failed (leave the row
    untouched); status=None means the number no longer appears in any status
    (treat as gone); otherwise the lowercased oppStatus. All four statuses are
    queried so a closed/archived opportunity is still returned.
    """
    try:
        r = requests.post(
            SEARCH_URL,
            json={'keyword': number, 'rows': 25, 'startRecord': 0,
                  'oppStatuses': 'posted|forecasted|closed|archived'},
            timeout=20,
        )
        r.raise_for_status()
        hits = (r.json().get('data') or {}).get('oppHits') or []
        for h in hits:
            if str(h.get('number') or '').strip() == str(number).strip():
                return True, str(h.get('oppStatus') or '').strip().lower()
        return True, None
    except Exception:
        return False, None


def reverify_grants(gcs_client, *, today: str | None = None,
                    rolling_only: bool = True, dry_run: bool = False,
                    progress=_noop) -> dict:
    """Re-check stored GRANTS-GOV opportunities against the live API and update
    `verify_status` / `last_verified_active`.

    Candidates are rows not already `verify_status == 'inactive'`, and with
    `rolling_only=True` additionally `is_rolling` (the manual button's original
    behaviour). `rolling_only=False` re-checks every open topic — the scheduled
    sweep — catching non-rolling grants that closed early or were archived before
    their posted deadline, which the deadline-only expiry in `drop_dead_topics`
    misses. posted/forecasted → active; closed/archived/gone → inactive. Rewrites
    only changed parquets (skipped on `dry_run`).
    """
    today = today or datetime.today().strftime('%Y-%m-%d')
    frames: dict = {}
    numbers: set = set()
    for blob in gcs_client.list_blobs(BUCKET, prefix=GRANTS_PREFIX):
        if not blob.name.endswith('.parquet'):
            continue
        try:
            df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        except Exception:
            continue
        if 'topic_number' not in df.columns:
            continue
        for col, default in (('last_verified_active', ''), ('verify_status', 'active')):
            if col not in df.columns:
                df[col] = default
        active = df['verify_status'].fillna('').astype(str) != 'inactive'
        if rolling_only:
            if 'is_rolling' not in df.columns:
                continue
            sel = active & df['is_rolling'].fillna(False).astype(bool)
        else:
            sel = active
        if sel.any():
            frames[blob] = (df, sel)
            numbers.update(df.loc[sel, 'topic_number'].astype(str))

    status_map: dict = {}
    if numbers:
        done = 0
        with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as pool:
            futs = {pool.submit(lookup_opp_status, n): n for n in numbers}
            for fut in as_completed(futs):
                n = futs[fut]
                ok, status = fut.result()
                done += 1
                progress(done, len(numbers))
                if not ok:
                    continue                       # leave unchanged on lookup error
                status_map[n] = ('active' if status in ('posted', 'forecasted')
                                 else 'inactive')  # closed/archived/None → gone

    refreshed = expired = files_written = 0
    for blob, (df, sel) in frames.items():
        dirty = False
        for idx in df.index[sel]:
            res = status_map.get(str(df.at[idx, 'topic_number']))
            if res == 'active':
                df.at[idx, 'last_verified_active'] = today
                df.at[idx, 'verify_status']        = 'active'
                refreshed += 1
                dirty = True
            elif res == 'inactive':
                df.at[idx, 'verify_status'] = 'inactive'
                expired += 1
                dirty = True
        if dirty and not dry_run:
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            buf.seek(0)
            blob.upload_from_file(buf, content_type='application/octet-stream')
            files_written += 1
    return {'checked': len(numbers), 'refreshed': refreshed, 'expired': expired,
            'files_written': files_written, 'dry_run': dry_run}
