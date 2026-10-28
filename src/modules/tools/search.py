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

import numpy as np
import pandas as pd

from src.modules import aspect_profile as ap, grant_utils as gu
from src.modules.tools import _common as cm

BUCKET = 'cc-matcher-bucket-jeg-v1'
_TOPICS_PREFIX = 'data/all-topics/processed/'
_RESUMES_PREFIX = 'data/resumes/'


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
