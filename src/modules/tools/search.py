"""
Search tools (Phase 5.1, Tier 1 read-only)
------------------------------------------
search_grants / get_profile / search_resumes. search_companies / get_company
already live in company_lookup.py (written for the Phase 3 agent); the MCP server
imports those directly, so a company lookup is one implementation.

The two embedding searches take an injected `embed_query(text) -> list[float]`
callable rather than an OpenAI client, so they stay Streamlit-free and are tested
with a trivial fake embedder — the MCP server builds `embed_query` once from its
OpenAI key (text-embedding-ada-002, the model every stored vector was built with).
"""

from __future__ import annotations

import io
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.modules import aspect_profile as ap, grant_utils as gu
from src.modules.tools import _common as cm

BUCKET = 'cc-matcher-bucket-jeg-v1'
_TOPICS_PREFIX = 'data/all-topics/processed/'
_RESUMES_PREFIX = 'data/resumes/'
_CONTACTS_PREFIX = 'data/all-contacts/'


# ── helpers ───────────────────────────────────────────────────────────────────

def _matrix(embeddings) -> tuple[np.ndarray, list[int]]:
    """Stack a column of per-row embedding vectors into one (n, dim) matrix,
    keeping only rows whose vector is non-empty and of the modal length. Returns
    the matrix and the original row positions it covers."""
    vecs, rows = [], []
    lengths = {}
    for i, e in enumerate(embeddings):
        if isinstance(e, np.ndarray):
            e = e.tolist()
        if not isinstance(e, list) or not e:
            continue
        lengths[len(e)] = lengths.get(len(e), 0) + 1
        vecs.append((i, e))
    if not vecs:
        return np.zeros((0, 0)), []
    dim = max(lengths, key=lengths.get)
    keep = [(i, v) for i, v in vecs if len(v) == dim]
    rows = [i for i, _ in keep]
    mat = np.asarray([v for _, v in keep], dtype=np.float32)
    return mat, rows


def _num(value):
    """A JSON-safe number (int when whole) or None. Maps NaN/blank/unparseable to
    None so a missing ceiling reads as 'no data', never 0, in the MCP payload."""
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return int(f) if f.is_integer() else f


def _first_str(row, *cols) -> str:
    """First non-blank string across `cols` on a row (NaN/''/'nan'/'None'
    skipped). Contact frames mix conventions — lead folders carry `companyName`/
    `company_summary`, clients/prospects `company_name`/`summary` — and a
    partially-updated file can leave either blank, so a plain `a or b` is unsafe
    (float NaN is truthy)."""
    for c in cols:
        v = row.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        s = str(v).strip()
        if s and s.lower() not in ('nan', 'none'):
            return s
    return ''


def _cosine_top(query_vec, mat: np.ndarray, rows: list[int], limit: int):
    """Top `limit` (row_position, score) by cosine against `query_vec`."""
    if mat.shape[0] == 0:
        return []
    q = np.asarray(query_vec, dtype=np.float32)
    if q.shape[0] != mat.shape[1]:
        return []
    qn = q / (np.linalg.norm(q) or 1.0)
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0] = 1.0
    scores = (mat @ qn) / norms
    order = np.argsort(scores)[::-1][:max(1, int(limit))]
    return [(rows[i], float(scores[i])) for i in order]


def _load_parquets(gcs_client, prefix: str, *, bucket: str) -> pd.DataFrame:
    frames = []
    for blob in gcs_client.list_blobs(bucket, prefix=prefix):
        if not blob.name.endswith('.parquet'):
            continue
        try:
            frames.append(pd.read_parquet(io.BytesIO(blob.download_as_bytes())))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _topic_folders(gcs_client, *, bucket: str) -> dict[str, list]:
    """Group the parquet blobs under processed/ by their agency folder (the first
    path segment after the prefix). One cheap list call — blobs aren't downloaded
    until a folder is scored, so peak memory is one folder, not the whole store."""
    folders: dict[str, list] = {}
    for blob in gcs_client.list_blobs(bucket, prefix=_TOPICS_PREFIX):
        if not blob.name.endswith('.parquet'):
            continue
        rest = blob.name[len(_TOPICS_PREFIX):]
        folder = rest.split('/', 1)[0] if '/' in rest else ''
        if folder:
            folders.setdefault(folder, []).append(blob)
    return folders


def _dedup_key(row) -> str:
    """Identity of an opportunity for cross-folder dedup: its topic number when
    present, else its normalized title. Collapses the same call listed under
    several consortium folders (NIIMBL/AFFOA/NextFlex)."""
    tn = str(row.get('topic_number', '') or '').strip()
    if tn and tn.lower() not in ('nan', 'none'):
        return f'tn::{tn.lower()}'
    title = ' '.join(str(row.get('title', '') or '').split()).lower()
    return f'ti::{title}'


# ── grants ────────────────────────────────────────────────────────────────────

def search_grants(gcs_client, embed_query, query: str, *, agency: str | None = None,
                  open_only: bool = True, include_procurement: bool = False,
                  limit: int = 10, bucket: str = BUCKET) -> dict:
    """Rank grant topics against a typed technology description. `agency`
    restricts to one processed/ folder (e.g. 'DOD'); omit for all — the all-agency
    search streams folder by folder, so it no longer loads the whole store at once.
    `open_only` drops archived/inactive and provably-expired topics (past deadline,
    not rolling), the same filter every reader uses. `include_procurement` (default
    False) keeps commodity/supply buys, sources-sought, sole-source and
    notice-of-intent-to-award notices — otherwise they are dropped as non-fundable.
    The same call listed under several folders is collapsed to one row with
    `also_listed_under`. Each result carries a `record_kind`
    (solicitation/procurement/award), a `status` of open/rolling/unknown and a
    0-1 cosine score."""
    folders = _topic_folders(gcs_client, bucket=bucket)
    if agency:
        want = agency.strip('/')
        folders = {want: folders[want]} if want in folders else {}
    if not folders:
        return {'query': query, 'topics_searched': 0, 'results': []}

    qvec = embed_query(query)
    keep_per_folder = max(2 * int(limit), 20)
    candidates: list[dict] = []
    scored = 0

    for folder, blobs in folders.items():
        frames = []
        for blob in blobs:
            try:
                frames.append(pd.read_parquet(io.BytesIO(blob.download_as_bytes())))
            except Exception:
                continue
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        del frames
        df['broad_agency'] = folder            # folder is the agency identity
        df = gu.normalize_grant_columns(df)
        if open_only:
            df = gu.drop_dead_topics(df).reset_index(drop=True)
        df = gu.add_record_kind(df)
        if not include_procurement and 'record_kind' in df.columns:
            df = df[df['record_kind'] != 'procurement'].reset_index(drop=True)
        if df.empty or 'embeddings' not in df.columns:
            continue
        scored += len(df)
        mat, rowpos = _matrix(list(df['embeddings']))
        for pos, score in _cosine_top(qvec, mat, rowpos, keep_per_folder):
            r = df.iloc[pos]
            candidates.append({
                'topic_number': str(r.get('topic_number', '') or ''),
                'title': cm.cap_text(r.get('title'), 300),
                'agency': str(r.get('agency', '') or ''),
                'broad_agency': folder,
                'record_kind': str(r.get('record_kind', '') or 'solicitation'),
                'due_date': gu.clean_date_display(
                    r.get('due_date', '') or r.get('close_date', '')),
                'status': gu.topic_open_status(r),
                'grant_summary': cm.cap_text(r.get('grant_summary'), 800),
                'award_ceiling_num': _num(r.get('award_ceiling_num')),
                'phase1_ceiling_num': _num(r.get('phase1_ceiling_num')),
                'phase2_ceiling_num': _num(r.get('phase2_ceiling_num')),
                'ceiling_source': str(r.get('ceiling_source', '') or ''),
                'source': cm.cap_text(r.get('source'), 300),
                'score': round(float(score), 3),
                '_key': _dedup_key(r),
            })
        del df, mat

    # Collapse cross-folder duplicates: keep the best-scoring copy, record the
    # other folders/agencies it was listed under.
    best: dict[str, dict] = {}
    for c in candidates:
        key = c['_key']
        cur = best.get(key)
        if cur is None:
            c['also_listed_under'] = []
            best[key] = c
        else:
            keeper = cur if cur['score'] >= c['score'] else c
            other = c if keeper is cur else cur
            alts = set(keeper.get('also_listed_under', []))
            for lbl in (other['agency'] or other['broad_agency'],
                        *other.get('also_listed_under', [])):
                if lbl and lbl != (keeper['agency'] or keeper['broad_agency']):
                    alts.add(lbl)
            keeper['also_listed_under'] = sorted(alts)
            best[key] = keeper

    results = sorted(best.values(), key=lambda c: c['score'], reverse=True)[:max(1, int(limit))]
    for c in results:
        c.pop('_key', None)
    return {'query': query, 'topics_searched': scored, 'results': results}


# ── profiles ──────────────────────────────────────────────────────────────────

def get_profile(gcs_client, company_key: str, *, pools=('clients', 'prospects'),
                bucket: str = BUCKET) -> dict:
    """One company's multi-aspect capability profile: aspect labels + short text,
    confirmed markets (with tier ordinal) and unexplored markets. `{found:
    false}` if no profile exists in the named pools."""
    for pool in pools:
        try:
            df = ap.load_profiles(gcs_client, bucket=bucket, pool=pool)
        except Exception:
            continue
        if df.empty or 'company_key' not in df.columns:
            continue
        hit = df[df['company_key'].astype(str) == str(company_key)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        aspects = [{'label': a.get('label', ''), 'kind': a.get('kind', ''),
                    'text': cm.cap_text(a.get('text'), 400),
                    'keywords': (a.get('keywords') or [])[:12]}
                   for a in ap.profile_aspects(row)]
        markets = [{'market': m.get('market', ''),
                    'tier': ap.tier_ordinal(ap.market_tier_rank(m)),
                    'subtitle': cm.cap_text(m.get('subtitle'), 200)}
                   for m in ap.profile_markets(row)]
        unexplored = [{'market': m.get('market', ''),
                       'rationale': cm.cap_text(m.get('rationale'), 300)}
                      for m in ap.profile_unexplored(row)]
        return {
            'company_key': company_key, 'found': True, 'pool': pool,
            'company_name': str(row.get('company_name', '') or ''),
            'website': str(row.get('companyWebsite', '') or ''),
            'profile_summary': cm.cap_text(row.get('profile_summary'), 800),
            'aspects': aspects, 'markets': markets, 'unexplored_markets': unexplored,
            'built_at': str(row.get('built_at', '') or ''),
        }
    return {'company_key': company_key, 'found': False}


# ── resumes ───────────────────────────────────────────────────────────────────

def search_resumes(gcs_client, embed_query, query: str, *, limit: int = 10,
                   bucket: str = BUCKET) -> dict:
    """Rank individual résumés by their expertise summary against a query. Rows
    with no embedding (summary too thin to embed) are skipped, as in Resume
    Search."""
    df = _load_parquets(gcs_client, _RESUMES_PREFIX, bucket=bucket)
    if df.empty or 'embeddings' not in df.columns:
        return {'query': query, 'results': [], 'resumes_searched': 0}

    mat, rows = _matrix(list(df['embeddings']))
    top = _cosine_top(embed_query(query), mat, rows, limit)
    results = []
    for pos, score in top:
        r = df.iloc[pos]
        name = ' '.join(x for x in [str(r.get('firstName', '') or ''),
                                    str(r.get('lastName', '') or '')] if x).strip()
        results.append({
            'name': name,
            'company': str(r.get('company', '') or ''),
            'email': str(r.get('email', '') or ''),
            'expertise_summary': cm.cap_text(r.get('expertise_summary'), 600),
            'resume_url': str(r.get('resume_url', '') or ''),
            'score': round(score, 3),
        })
    return {'query': query, 'resumes_searched': int(len(df)), 'results': results}


# ── contacts ──────────────────────────────────────────────────────────────────

# Identity/display columns worth reading off a contact parquet. The big scraped
# columns (page_text, full_text, raw JSON blobs) are deliberately NOT read — they
# are dead weight for a semantic search and only slow the parse.
_CONTACT_ID_COLS = ('company_name', 'companyName', 'companyWebsite', 'summary',
                    'company_summary', 'segment', 'industry')
_CONTACTS_WORKERS = 8
# Return BEFORE the claude.ai connector proxy gives up, so a scan that can't
# finish yields honest partial coverage (sources_incomplete) rather than a hard
# proxy timeout with nothing. With the vectorised parse a full scan is normally
# well under this.
_CONTACTS_BUDGET_S = 25.0


def _norm_site(url) -> str:
    """A website reduced to a dedup identity: scheme, leading www. and trailing
    slashes dropped, lowercased. Collapses http/https, www/non-www and trailing-
    slash variants that would otherwise be separate company keys."""
    s = str(url or '').strip().lower()
    for p in ('https://', 'http://'):
        if s.startswith(p):
            s = s[len(p):]
            break
    if s.startswith('www.'):
        s = s[4:]
    return s.rstrip('/')


def _dedup_identity(name: str, website: str) -> str:
    """Normalised (name, site) identity for collapsing the same company listed
    under www/non-www/trailing-slash variants or across folders."""
    return f'{" ".join(str(name).lower().split())}||{_norm_site(website)}'


def _embeddings_matrix(table) -> tuple[np.ndarray, list[int]]:
    """Vectorised (mat float32 [n,dim], row_positions) from a pyarrow table's
    `embeddings` list column — no Python-per-row loop when the vectors are a fixed
    width (the common case), which is what makes a 30k-row blob score in a blink.
    Falls back to the robust `_matrix` for ragged/null data."""
    col = table.column('embeddings').combine_chunks()
    n = len(col)
    if n == 0 or col.null_count == n:
        return np.zeros((0, 0)), []
    try:
        offsets = np.asarray(col.offsets.to_numpy(zero_copy_only=False))
        lengths = np.diff(offsets)
        pos_lengths = lengths[lengths > 0]
        dim = int(np.bincount(pos_lengths).argmax()) if pos_lengths.size else 0
        if dim and col.null_count == 0 and bool((lengths == dim).all()):
            flat = np.asarray(col.values.to_numpy(zero_copy_only=False),
                              dtype=np.float32)
            return flat.reshape(n, dim), list(range(n))
    except Exception:
        pass
    return _matrix(col.to_pylist())


def _read_contact_blob(blob):
    """(status, table) for one contact parquet, reading only the identity/display
    columns + embeddings. status is 'searched' (has an embeddings column),
    'no_embeddings' or 'error'. Never raises. Shared with contact_index's build."""
    try:
        data = blob.download_as_bytes()
        pf = pq.ParquetFile(io.BytesIO(data))
        names = set(pf.schema_arrow.names)
        if 'embeddings' not in names:
            return 'no_embeddings', None
        want = [c for c in (*_CONTACT_ID_COLS, 'embeddings') if c in names]
        return 'searched', pf.read(columns=want)
    except Exception:
        return 'error', None


def _contact_record(table, source: str):
    """A `record(pos) -> dict | None` reader over a contact table: the display
    fields of row `pos` in the shape search_contacts returns (minus `score`), or
    None for a row with neither a name nor a website."""
    cols = {c: (table.column(c).to_pylist() if c in table.schema.names else None)
            for c in _CONTACT_ID_COLS}

    def cell(pos, *candidates):
        for c in candidates:
            vals = cols.get(c)
            if vals is None:
                continue
            v = vals[pos]
            s = '' if v is None else str(v).strip()
            if s and s.lower() not in ('nan', 'none'):
                return s
        return ''

    def record(pos):
        name = cell(pos, 'company_name', 'companyName')
        website = cell(pos, 'companyWebsite')
        if not (name or website):
            return None
        return {
            'company_key': f'{name}||{website}', 'company_name': name,
            'website': website, 'source': source,
            'industry': cell(pos, 'segment', 'industry'),
            'summary': cm.cap_text(cell(pos, 'summary', 'company_summary'), 600),
        }

    return record


def _score_contacts_blob(blob, folder: str, qvec, keep: int):
    """(folder, status, candidates) for one contact parquet scanned live — the
    path for blobs the contact index does not cover yet. Never raises."""
    status, table = _read_contact_blob(blob)
    if status != 'searched':
        return folder, status, []
    mat, rowpos = _embeddings_matrix(table)
    if mat.shape[0] == 0:
        return folder, 'no_embeddings', []
    record = _contact_record(table, folder)
    out = []
    for pos, score in _cosine_top(qvec, mat, rowpos, keep):
        rec = record(pos)
        if rec is not None:
            rec['score'] = round(float(score), 3)
            out.append(rec)
    return folder, 'searched', out


def _index_task(gcs_client, folder: str, blobs: list, qvec, keep: int, bucket: str):
    """Score one folder from its prebuilt index. Returns (candidates, stale_blobs,
    has_embeddings): blobs whose generation the index does not cover (imported
    since the last build, or no index at all) come back as stale for a live scan,
    so results are never staler than the last import."""
    from src.modules.tools import contact_index as ci
    entry = ci.load_index(gcs_client, folder, bucket=bucket)
    if entry is None:
        return [], list(blobs), False
    covered = entry['manifest'].get('blobs', {})
    valid = {b.name for b in blobs if covered.get(b.name) == str(b.generation)}
    stale = [b for b in blobs if b.name not in valid]
    cands, any_rows = ci.score_index(entry, folder, qvec, valid, keep)
    return cands, stale, any_rows


def search_contacts(gcs_client, embed_query, query: str, *, source: str | None = None,
                    limit: int = 10, bucket: str = BUCKET,
                    max_workers: int = _CONTACTS_WORKERS,
                    time_budget_s: float = _CONTACTS_BUDGET_S) -> dict:
    """Semantically rank the companies in `data/all-contacts/` against a query by
    the embedding of their company summary — the same embedding Bulk Matching
    scores grants against, so a hit here is a company that would match a grant
    worded like `query`.

    Covers every contact folder: the lead sources (apollo, sba, …) AND the
    clients/prospects pools. `source` restricts to one folder; omit for all.

    The raw store is multi-GB (mostly vectors), far too much to download per
    query, so each folder is scored from its prebuilt index (contact_index.py,
    rebuilt nightly, cached in-process). Any blob imported since the last build
    is scanned live, so nothing new is missed. Results are deduped to one row per
    company (by a normalised name+domain identity, so www/non-www variants
    collapse) with `also_in` naming the other folders it appears in and a 0-1
    cosine `score`; `company_key` is a real stored key, so get_company /
    get_profile accept it unchanged.

    Everything runs under a wall-clock budget: if the budget runs out, it returns
    what it has and names the unfinished folders in `sources_incomplete`.
    Coverage is reported by folder NAME only — no row/contact counts are ever
    exposed. `scores_note` warns that scores compare only within a source."""
    qvec = embed_query(query)
    want = source.strip().strip('/') if source else None
    keep = max(2 * int(limit), 20)

    by_folder: dict[str, list] = {}
    for blob in gcs_client.list_blobs(bucket, prefix=_CONTACTS_PREFIX):
        if not blob.name.endswith('.parquet'):
            continue
        rest = blob.name[len(_CONTACTS_PREFIX):]
        folder = rest.split('/', 1)[0] if '/' in rest else ''
        if not folder or (want and folder != want):
            continue
        by_folder.setdefault(folder, []).append(blob)

    best: dict[str, dict] = {}
    seen_in: dict[str, set] = {}
    state = {f: {'index_done': False, 'live_total': 0, 'live_done': 0, 'emb': False}
             for f in by_folder}

    def merge(cands):
        for c in cands:
            ident = _dedup_identity(c['company_name'], c['website'])
            seen_in.setdefault(ident, set()).add(c['source'])
            cur = best.get(ident)
            if cur is None or c['score'] > cur['score']:
                c['_ident'] = ident
                best[ident] = c

    deadline = time.monotonic() + float(time_budget_s)
    # NOT a `with` block: its exit joins every queued and running future, which
    # silently turned the budget into "wait for everything" (calls ran ~110s).
    ex = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
    try:
        pending = {ex.submit(_index_task, gcs_client, f, blobs, qvec, keep, bucket):
                   ('index', f) for f, blobs in by_folder.items()}
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            finished, _ = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
            for fut in finished:
                kind, folder = pending.pop(fut)
                st = state[folder]
                if kind == 'index':
                    st['index_done'] = True
                    try:
                        cands, stale, emb = fut.result()
                    except Exception:
                        cands, stale, emb = [], list(by_folder[folder]), False
                    st['emb'] = st['emb'] or emb
                    merge(cands)
                    for blob in stale:
                        pending[ex.submit(_score_contacts_blob, blob, folder, qvec,
                                          keep)] = ('live', folder)
                        st['live_total'] += 1
                else:
                    st['live_done'] += 1
                    try:
                        _, status, cands = fut.result()
                    except Exception:
                        status, cands = 'error', []
                    if status == 'searched':
                        st['emb'] = True
                    merge(cands)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    results = sorted(best.values(), key=lambda c: c['score'], reverse=True)[:max(1, int(limit))]
    for c in results:
        c['also_in'] = sorted(seen_in.get(c.pop('_ident', ''), set()) - {c['source']})

    searched, no_emb, incomplete = [], [], []
    for folder in sorted(state):
        st = state[folder]
        if not st['index_done'] or st['live_done'] < st['live_total']:
            incomplete.append(folder)        # budget ran out before this folder finished
        elif st['emb']:
            searched.append(folder)
        else:
            no_emb.append(folder)

    return {
        'query': query,
        'results': results,
        'sources_searched': searched,
        'sources_no_embeddings': no_emb,
        'sources_incomplete': incomplete,
        'complete': not incomplete,
        'scores_note': 'Cosine scores are comparable only within a source, not '
                       'across sources (pools differ in density).',
    }
