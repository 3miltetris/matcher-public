"""
Bulk Aspect Match
-----------------
Matches the whole client directory against grant topics using the
multi-aspect profiles built in the Client Profiles view.

Each client carries several independently embedded aspects, grouped into the
markets those aspects serve. A run scores one *unit* at a time — either the
whole company (every aspect at once, the original behaviour) or one market
(only the aspects earmarked to it, plus the market's own narrative vector).
Every vector in the unit is scored against every selected grant topic (one
numpy matmul per unit), a topic becomes a candidate when enough of them clear
the similarity threshold, and the top candidates per unit are re-ranked 1-5 by
Claude with the matched aspect as context.

Running by market treats each market as its own company, so a client's defense
story is ranked on its own instead of being averaged in with the rest. Markets
are ranked 1st, 2nd, 3rd..., and the tier filter slices on that rank.

A market unit is one of two KINDS, and the distinction is load-bearing:

  confirmed  — a market the client sells into today. Scored on its earmarked
               aspect vectors plus its own narrative.
  unexplored — a market the profile builder inferred by linking the client's
               aspects, which it does NOT serve. Scored on its narrative vector
               alone, and re-ranked by a DIFFERENT prompt: the confirmed prompt
               says "never assume capabilities that are not stated", which is
               exactly what a hypothesis asks for, so it would score every
               unexplored row 1-2 and the minimum-score filter would drop the
               whole run with no error shown. Every result row carries
               `market_kind` so a 4 on a hypothesis is never read as a 4 on a
               capability the client actually has.

Scoring and re-ranking run in this process — keep the page open while a run
is in flight. Results are held in session state, downloadable as CSV, and
written to aspect-match-results/{run_id}/results.csv in GCS.
"""

import asyncio
import io
import json
import random
import re
import traceback
import warnings
from collections import namedtuple
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from anthropic import AsyncAnthropic
from google.cloud import storage
from google.oauth2 import service_account

import src.modules.aspect_profile as ap
from src.modules.grant_utils import normalize_grant_columns

# ── Constants ──────────────────────────────────────────────────────────────

_BUCKET         = ap.BUCKET
_TOPICS_PREFIX  = 'data/all-topics/processed/'
_RESULTS_PREFIX = 'aspect-match-results/'

_RERANK_MODELS = ['claude-haiku-4-5-20251001', 'claude-sonnet-4-6']
_CONCURRENCY   = 15
_MAX_RETRIES   = 5
_CONFIRM_PAIRS = 2500   # above this many re-rank calls, require a confirmation

# Match modes
_MODE_ALL    = 'All aspects (whole company)'
_MODE_MARKET = 'By market'

# Market kind. Confirmed is the default so an ordinary run behaves exactly as
# it always has, and so the unit count doesn't quietly balloon past
# _CONFIRM_PAIRS the first time someone opens the page after a rebuild.
_KIND_CONFIRMED  = 'Confirmed markets'
_KIND_UNEXPLORED = 'Unexplored markets'
_KIND_BOTH       = 'Both'
_KINDS           = [_KIND_CONFIRMED, _KIND_UNEXPLORED, _KIND_BOTH]
_CATEGORY_ALL    = 'All categories'

# Stored on every result row, and what the re-rank prompt selection keys off.
# There is no separate 'Defense only' scope any more: category='Defense' already
# did that, and the old scope existed only to bypass the category filter.
_ROW_CONFIRMED  = 'confirmed'
_ROW_UNEXPLORED = 'unexplored'
_KIND_ROWS      = {
    _KIND_CONFIRMED:  [_ROW_CONFIRMED],
    _KIND_UNEXPLORED: [_ROW_UNEXPLORED],
    _KIND_BOTH:       [_ROW_CONFIRMED, _ROW_UNEXPLORED],
}

# One scoring subject: a whole company (market None, kind ''), or one market of
# one company. `mi` indexes that market's narrative vector inside the profile's
# flat market_embeddings / unexplored_embeddings block.
_Unit = namedtuple('_Unit', 'prof market mi kind')

# Topic columns carried into the results, when present.
_TOPIC_COLS = [
    'topic_number', 'title', 'agency', 'broad_agency', 'due_date', 'close_date',
    'open_date', 'funding_amount', 'grant_summary', 'source',
]


# ── GCS ────────────────────────────────────────────────────────────────────

def _get_storage_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)


def _list_agencies(client: storage.Client) -> list[str]:
    try:
        blobs = client.list_blobs(_BUCKET, prefix=_TOPICS_PREFIX, delimiter='/')
        list(blobs)
        return sorted(p.replace(_TOPICS_PREFIX, '').strip('/') for p in blobs.prefixes)
    except Exception as e:
        st.error(f'Failed to list agencies: {e}')
        return []


def _load_topics(client: storage.Client, agencies: list[str]) -> pd.DataFrame:
    frames = []
    for agency in agencies:
        for blob in client.list_blobs(_BUCKET, prefix=f'{_TOPICS_PREFIX}{agency}/'):
            if blob.name.endswith('.parquet'):
                df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
                df['broad_agency'] = agency
                frames.append(df)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        topics = pd.concat(frames, ignore_index=True)
    # Notices marked archived by the SAM.gov revision check are no longer live
    if 'sam_status' in topics.columns:
        topics = topics[topics['sam_status'].fillna('').astype(str) != 'archived']
    topics = normalize_grant_columns(topics.reset_index(drop=True))
    # Stored as float64; halved to float32 here because the whole topic store
    # is held in session state alongside the scoring matrix. This frame is
    # never written back, so the narrower dtype is local to the view.
    if 'embeddings' in topics.columns:
        topics['embeddings'] = topics['embeddings'].map(
            lambda e: np.asarray(e, dtype=np.float32)
            if isinstance(e, (list, np.ndarray)) else e
        )
    return topics


# ── Filters (same behaviour as Grant Search / Bulk Matching) ───────────────

_DATE_FORMATS = ('%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%m-%d-%Y', '%b %d, %Y')


def _parse_date_str(value) -> date | None:
    s = str(value).strip()
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        for candidate in (s, s[:10]):
            try:
                return datetime.strptime(candidate, fmt).date()
            except ValueError:
                pass
    return None


def _filter_is_active(f: dict) -> bool:
    if not f.get('column'):
        return False
    if f.get('type') == 'date_range':
        return bool(f.get('date_from') and f.get('date_to'))
    return bool(f.get('keyword', '').strip())


def _filter_mask(df: pd.DataFrame, f: dict) -> pd.Series:
    if f.get('type') == 'date_range':
        d_from, d_to = f['date_from'], f['date_to']

        def _in_range(v) -> bool:
            d = _parse_date_str(v)
            return d is not None and d_from <= d <= d_to

        return df[f['column']].map(_in_range)
    return df[f['column']].astype(str).str.lower().str.contains(
        f['keyword'].lower(), na=False
    )


def _apply_filters(df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    active = [f for f in filters if _filter_is_active(f)]
    if not active:
        return df
    mask = _filter_mask(df, active[0])
    for f in active[1:]:
        m = _filter_mask(df, f)
        mask = (mask & m) if f['operator'] == 'AND' else (mask | m)
    return df[mask]


# ── Scoring ────────────────────────────────────────────────────────────────

def _stack_topic_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """(T, dim) float32 matrix + the topic rows it corresponds to. Rows with a
    missing or wrong-length vector are dropped."""
    keep, vecs = [], []
    for idx, emb in df['embeddings'].items():
        if isinstance(emb, (list, np.ndarray)) and len(emb) == ap.EMBED_DIM:
            keep.append(idx)
            vecs.append(np.asarray(emb, dtype=np.float32))
    if not vecs:
        return np.zeros((0, ap.EMBED_DIM), dtype=np.float32), df.iloc[0:0]
    meta = df.loc[keep].drop(columns=['embeddings'], errors='ignore').reset_index(drop=True)
    return np.vstack(vecs), meta


def _unit_markets(prof, kinds: list[str]) -> list[tuple[str, dict, int]]:
    """(kind, market, index-into-its-vector-block) for one profile, restricted
    to the requested kinds. The index is positional within the kind's own flat
    embedding block, which is why the kind has to travel with it."""
    out: list[tuple[str, dict, int]] = []
    if _ROW_CONFIRMED in kinds:
        out += [(_ROW_CONFIRMED, m, i)
                for i, m in enumerate(ap.profile_markets(prof))]
    if _ROW_UNEXPLORED in kinds:
        out += [(_ROW_UNEXPLORED, m, i)
                for i, m in enumerate(ap.profile_unexplored(prof))]
    return out


def _market_counts(selected: pd.DataFrame, kinds: list[str]) -> dict[str, int]:
    """Canonical market name → how many of the selected clients have it, across
    the requested kinds."""
    counts: dict[str, int] = {}
    for _, prof in selected.iterrows():
        names = {str(m.get('market') or '') for _k, m, _i in _unit_markets(prof, kinds)}
        for name in names:
            if name:
                counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _tier_counts(selected: pd.DataFrame, kinds: list[str]) -> dict[int, int]:
    """Tier rank → how many markets of the selected clients sit at it. Drives
    the tier picker, so it only ever offers ranks that exist."""
    counts: dict[int, int] = {}
    for _, prof in selected.iterrows():
        for _k, market, _i in _unit_markets(prof, kinds):
            rank = ap.market_tier_rank(market)
            counts[rank] = counts.get(rank, 0) + 1
    return dict(sorted(counts.items()))


def _plan_units(
    selected: pd.DataFrame,
    by_market: bool,
    kinds: list[str],
    tiers: set[int],
    category: str,
) -> tuple[list[_Unit], list[str]]:
    """One _Unit per scoring subject.

    Whole-company mode gives one unit per client (market None). Market mode
    gives one unit per (client, kind, market) surviving the tier and category
    filters — each is scored, capped and re-ranked as if it were its own
    company. Cheap enough to call on every rerun for the run-size estimate.

    An empty `tiers` means every tier, matching how _CATEGORY_ALL behaves — a
    deselect-everything state would otherwise silently plan zero units."""
    plan: list[_Unit] = []
    skipped: list[str] = []

    for _, prof in selected.iterrows():
        if not by_market:
            plan.append(_Unit(prof, None, -1, ''))
            continue

        if _ROW_CONFIRMED in kinds and not ap.profile_markets(prof):
            skipped.append(
                f'{prof["company_name"]}: profile has no markets — rebuild it in '
                'Client Profiles to match it by market'
            )
        if _ROW_UNEXPLORED in kinds and not ap.profile_unexplored(prof):
            skipped.append(
                f'{prof["company_name"]}: profile has no unexplored markets — '
                'rebuild it in Client Profiles with "Assess unexplored markets" on'
            )

        for kind, market, i in _unit_markets(prof, kinds):
            name = str(market.get('market') or '')
            if tiers and ap.market_tier_rank(market) not in tiers:
                continue
            if category != _CATEGORY_ALL and name != category:
                continue
            plan.append(_Unit(prof, market, i, kind))

    return plan, skipped


def _match_units(
    plan: list[_Unit],
    topic_matrix: np.ndarray,
    topic_meta: pd.DataFrame,
    threshold: float,
    min_hits: int,
    top_k: int,
    progress=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Candidate (unit, topic) rows: a topic qualifies for a unit when at least
    `min_hits` of the unit's *aspect* vectors clear `threshold`; the top `top_k`
    per unit by best score are kept. A confirmed market unit carries its
    earmarked aspect vectors plus the market narrative vector, which for Defense
    is the only place the DoD framing exists.

    An unexplored market unit carries its narrative vector and nothing else.
    Scoring the confirmed aspect vectors under it would mostly return the topics
    its confirmed markets already returned, and _rerank_groups would then share
    one score between two different framings of the same (client, topic, aspect)
    pair — written from whichever framing happened to come first."""
    meta_cols = [c for c in _TOPIC_COLS if c in topic_meta.columns]
    # Materialised once — .iloc[ti] per candidate builds a Series per row and
    # dominates the loop on large runs.
    meta_records = topic_meta[meta_cols].to_dict('records')
    rows: list[dict] = []
    skipped: list[str] = []
    total = len(plan)

    # A client appears once per market, so unpack its stored vectors once.
    aspect_cache: dict[str, tuple] = {}
    market_cache: dict[str, np.ndarray] = {}
    unexp_cache:  dict[str, np.ndarray] = {}

    for n, (prof, market, mi, kind) in enumerate(plan, 1):
        client = str(prof['company_name'])
        label  = f'{client} · {market["market"]}' if market else client
        if kind == _ROW_UNEXPLORED:
            label += ' (unexplored)'
        if progress is not None:
            progress.progress(n / total, text=f'Scoring {label} ({n}/{total})')

        key = str(prof['company_key'])
        if key not in aspect_cache:
            aspect_cache[key] = (ap.profile_aspects(prof), ap.unpack_embeddings(prof))
        aspects, matrix = aspect_cache[key]

        if matrix.shape[0] == 0 or len(aspects) != matrix.shape[0]:
            skipped.append(f'{client}: profile has no usable aspect vectors — rebuild it')
            continue

        if market is None:
            unit_aspects, unit_matrix = aspects, matrix
            n_aspect_vecs = unit_matrix.shape[0]
        elif kind == _ROW_UNEXPLORED:
            # Narrative-only by design (see the docstring). The narrative is the
            # whole hypothesis, so it is also the only thing worth scoring.
            if key not in unexp_cache:
                unexp_cache[key] = ap.unpack_unexplored_embeddings(prof)
            unexp_vectors = unexp_cache[key]
            if not 0 <= mi < unexp_vectors.shape[0]:
                skipped.append(
                    f'{label}: unexplored market has no narrative vector — '
                    'rebuild the profile'
                )
                continue
            unit_aspects = [{
                'label': f'Market: {market.get("market")}',
                'kind':  'market',
                'text':  str(market.get('narrative') or ''),
            }]
            unit_matrix   = unexp_vectors[mi][None, :]
            n_aspect_vecs = 0
        else:
            idx          = ap.market_aspect_indices(aspects, str(market.get('market')))
            unit_aspects = [aspects[i] for i in idx]
            unit_matrix  = matrix[idx] if idx else matrix[:0]

            if key not in market_cache:
                market_cache[key] = ap.unpack_market_embeddings(prof)
            market_vectors = market_cache[key]
            n_aspect_vecs = unit_matrix.shape[0]
            if 0 <= mi < market_vectors.shape[0]:
                unit_aspects = unit_aspects + [{
                    'label': f'Market: {market.get("market")}',
                    'kind':  'market',
                    'text':  str(market.get('narrative') or ''),
                }]
                unit_matrix = np.vstack([unit_matrix, market_vectors[mi][None, :]])

            if unit_matrix.shape[0] == 0:
                skipped.append(
                    f'{label}: market has no aspect or narrative vectors — rebuild the profile'
                )
                continue

        scores  = unit_matrix @ topic_matrix.T     # (n_vectors, T)
        best_i  = scores.argmax(axis=0)
        best    = scores.max(axis=0)
        cleared = scores >= threshold
        # min_hits counts capabilities. The market narrative is one more way of
        # describing the same market, not an extra capability, so it stays out
        # of the count - unless it is all the unit has, which is the case for a
        # market no aspect was earmarked to.
        hits    = (cleared[:n_aspect_vecs] if n_aspect_vecs else cleared).sum(axis=0)
        # A unit with no aspect vectors at all — an unexplored market, or a
        # confirmed market no aspect was earmarked to — is scored on its
        # narrative alone, so there is at most one hit to count. Applying a
        # min_hits above 1 would drop every such unit without saying why.
        effective_hits = min_hits if n_aspect_vecs else 1

        qualified = np.where((best >= threshold) & (hits >= effective_hits))[0]
        if qualified.size == 0:
            continue
        order = qualified[np.argsort(best[qualified])[::-1][:top_k]]

        # The confirmed capabilities an unexplored market says it would draw on.
        # Not scored — this is the evidence the re-ranker judges the extension
        # against, so a hypothesis can't be scored on its own optimism.
        linked_text = ''
        if kind == _ROW_UNEXPLORED:
            linked_text = '\n\n'.join(
                f"{aspects[i].get('label', '')}: {aspects[i].get('text', '')}"
                for i in ap.unexplored_aspect_indices(aspects, market)
            )

        for ti in order:
            ai     = int(best_i[ti])
            aspect = unit_aspects[ai]
            row = {
                'client':           client,
                'client_website':   prof['companyWebsite'],
                'market':           str(market.get('market')) if market else '',
                'market_kind':      kind or _ROW_CONFIRMED,
                'market_tier':      ap.tier_ordinal(ap.market_tier_rank(market)) if market else '',
                'market_subtitle':  str(market.get('subtitle') or '') if market else '',
                'market_rationale': str(market.get('rationale') or '') if market else '',
                'aspect_label':     aspect.get('label', ''),
                'aspect_kind':      aspect.get('kind', ''),
                'aspect_score':     round(float(best[ti]), 4),
                'aspects_hit':      int(hits[ti]),
                'aspects_total':    n_aspect_vecs or len(unit_aspects),
                'aspect_scores':    json.dumps({
                    unit_aspects[j].get('label', f'aspect_{j + 1}'): round(float(scores[j, ti]), 4)
                    for j in range(len(unit_aspects))
                }),
                '_company_key':     prof['company_key'],
                '_aspect_text':     aspect.get('text', ''),
                '_linked_aspects':  linked_text,
                '_profile_summary': prof['profile_summary'],
            }
            row.update(meta_records[int(ti)])
            rows.append(row)

    return pd.DataFrame(rows), skipped


# ── LLM re-rank ────────────────────────────────────────────────────────────

_RERANK_SYSTEM = (
    'You are screening a federal grant topic against one specific capability of a '
    'company that a proposal-writing firm represents.\n'
    'Score the fit from 1 to 5:\n'
    '5 = the company could propose to this topic directly with the capability described\n'
    '4 = strong fit with minor gaps\n'
    '3 = plausible fit, but notable gaps or adaptation needed\n'
    '2 = superficial or keyword-level overlap only\n'
    '1 = no fit\n\n'
    'Judge only the capability as described — never assume capabilities that are not stated.\n'
    'Return ONLY valid JSON: {"score": <integer 1-5>, "rationale": "<one sentence>"}'
)


# The confirmed prompt above asks "could they propose to this TODAY", and
# rightly answers 1 or 2 for anything speculative. Pointing it at an unexplored
# market — a hypothesis by construction — would score the entire kind below any
# selectable minimum, and an unscored/low-scored run shows an empty table with
# no error at all. So unexplored units get their own question, and their own
# scale, judged against the capabilities the client demonstrably has.
_RERANK_UNEXPLORED_SYSTEM = (
    'You are screening a federal grant topic against a market a company does NOT '
    'currently serve, but which a capability analysis suggests it could extend into.\n'
    'You are given the market hypothesis, the gap that analysis says remains, and the '
    'capabilities the company demonstrably has today.\n'
    'Score from 1 to 5 how plausibly this company could pursue this topic by extending '
    'those existing capabilities:\n'
    '5 = a clear extension of a stated capability — adaptation only, no new science\n'
    '4 = a strong extension with one identifiable gap to close\n'
    '3 = plausible, but a real capability, qualification or certification gap stands in the way\n'
    '2 = would require capabilities the company has not demonstrated\n'
    '1 = unrelated to anything the company can do\n\n'
    'Judge the extension ONLY against the listed capabilities — never assume capabilities '
    'that are not stated, and never credit the hypothesis for being ambitious.\n'
    'Return ONLY valid JSON: {"score": <integer 1-5>, "rationale": "<one sentence>"}'
)


def _rerank_user_message(row: dict) -> str:
    if row.get('market_kind') == _ROW_UNEXPLORED:
        subtitle = str(row.get('market_subtitle') or '')
        return (
            f"Company: {row.get('client', '')}\n"
            f"Company profile: {str(row.get('_profile_summary') or '')[:1500]}\n\n"
            f"Market the company does NOT currently serve — {row.get('market', '')}"
            f"{(': ' + subtitle) if subtitle else ''}\n"
            f"{str(row.get('_aspect_text') or '')[:2000]}\n\n"
            f"Gap the analysis says remains: "
            f"{str(row.get('market_rationale') or 'not stated')[:800]}\n\n"
            f"Capabilities the company demonstrably has today:\n"
            f"{str(row.get('_linked_aspects') or '(none recorded)')[:3000]}\n\n"
            f"Grant topic: {row.get('title', '')}\n"
            f"Agency: {row.get('agency', '') or row.get('broad_agency', '')}\n"
            f"Topic description:\n{str(row.get('grant_summary') or '')[:6000]}"
        )
    return (
        f"Company: {row.get('client', '')}\n"
        f"Company profile: {str(row.get('_profile_summary') or '')[:1500]}\n\n"
        f"Matched capability — {row.get('aspect_label', '')}:\n"
        f"{str(row.get('_aspect_text') or '')[:2000]}\n\n"
        f"Grant topic: {row.get('title', '')}\n"
        f"Agency: {row.get('agency', '') or row.get('broad_agency', '')}\n"
        f"Topic description:\n{str(row.get('grant_summary') or '')[:6000]}"
    )


def _parse_rerank(text: str) -> tuple[int | None, str]:
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', (text or '').strip())
    start, end = cleaned.find('{'), cleaned.rfind('}')
    if start != -1 and end > start:
        try:
            obj = json.loads(cleaned[start:end + 1])
            score = int(float(obj.get('score')))
            return max(1, min(5, score)), str(obj.get('rationale') or '')[:400]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # Tolerate a stray sentence around the JSON rather than losing the score
    m = re.search(r'"?score"?\s*[:=]\s*([1-5])', cleaned)
    if m:
        r = re.search(r'"?rationale"?\s*[:=]\s*"([^"]*)"', cleaned)
        return int(m.group(1)), (r.group(1)[:400] if r else '')
    return None, '(unparseable response)'


async def _rerank_async(
    rows: list[tuple[int, dict]], api_key: str, model: str, on_done
) -> list[tuple[int, int | None, str]]:
    sem = asyncio.Semaphore(_CONCURRENCY)

    async with AsyncAnthropic(api_key=api_key) as client:
        async def one(idx: int, row: dict) -> tuple[int, int | None, str]:
            async with sem:
                for attempt in range(_MAX_RETRIES):
                    try:
                        resp = await client.messages.create(
                            model=model,
                            max_tokens=250,
                            # No temperature: the anthropic 1.x SDK removed the
                            # parameter, and it is rejected outright by the newer
                            # models. Determinism comes from the strict JSON
                            # contract in the system prompt instead.
                            system=(
                                _RERANK_UNEXPLORED_SYSTEM
                                if row.get('market_kind') == _ROW_UNEXPLORED
                                else _RERANK_SYSTEM
                            ),
                            messages=[{'role': 'user', 'content': _rerank_user_message(row)}],
                        )
                        score, rationale = _parse_rerank(resp.content[0].text)
                        return idx, score, rationale
                    except Exception as e:
                        err = str(e)
                        retryable = any(
                            x in err for x in
                            ('429', '529', 'overloaded', 'rate_limit', 'rate limit', 'timeout')
                        )
                        if retryable and attempt < _MAX_RETRIES - 1:
                            await asyncio.sleep((2 ** attempt) + random.random())
                            continue
                        return idx, None, f'(scoring failed: {type(e).__name__})'
                return idx, None, '(scoring failed: retries exhausted)'

        tasks   = [asyncio.create_task(one(i, r)) for i, r in rows]
        results = []
        for fut in asyncio.as_completed(tasks):
            results.append(await fut)
            on_done(len(results))
        return results


def _rerank_groups(candidates: pd.DataFrame) -> list[list[int]]:
    """Row indices grouped by (client, topic, matched aspect, market kind).

    In market mode the same aspect can win the same topic under two markets.
    The re-rank prompt carries no market context, so those rows would get
    identical answers — score the pair once and share it.

    The kind IS part of the key: confirmed and unexplored rows are scored by
    different prompts answering different questions, so sharing one score
    across them would stamp a hypothesis score onto a confirmed row."""
    groups: dict[tuple, list[int]] = {}
    for idx, row in candidates.iterrows():
        key = (
            str(row.get('_company_key') or ''),
            # Agency included: one topic_number can belong to two agencies, and
            # blank-numbered rows fall back to titles that repeat across them.
            # Without it two different topics would share one score, written
            # from only the first row's grant_summary.
            str(row.get('broad_agency') or ''),
            str(row.get('agency') or ''),
            str(row.get('topic_number') or row.get('title') or ''),
            str(row.get('aspect_label') or ''),
            str(row.get('market_kind') or ''),
        )
        groups.setdefault(key, []).append(int(idx))
    return list(groups.values())


def _run_rerank(candidates: pd.DataFrame, api_key: str, model: str) -> pd.DataFrame:
    groups = _rerank_groups(candidates)
    rows   = [(n, candidates.loc[g[0]].to_dict()) for n, g in enumerate(groups)]
    total  = len(rows)
    prog   = st.progress(0.0, text=f'LLM re-ranking 0/{total}…')

    def on_done(done: int) -> None:
        prog.progress(done / total, text=f'LLM re-ranking {done}/{total}…')

    results = asyncio.run(_rerank_async(rows, api_key, model, on_done))
    prog.empty()

    out = candidates.copy()
    out['llm_score']     = 0
    out['llm_rationale'] = ''
    for n, score, rationale in results:
        for idx in groups[n]:
            # 0 keeps unscored pairs visible but below any usable minimum
            out.at[idx, 'llm_score']     = int(score) if score is not None else 0
            out.at[idx, 'llm_rationale'] = rationale
    return out


# ── Results output ─────────────────────────────────────────────────────────

_DISPLAY_FIRST = [
    'client', 'market', 'market_kind', 'market_tier', 'aspect_label', 'aspect_score',
    'aspects_hit', 'aspects_total',
    'llm_score', 'llm_rationale', 'topic_number', 'title', 'agency', 'broad_agency',
]


def _display_frame(df: pd.DataFrame) -> pd.DataFrame:
    internal = [c for c in df.columns if c.startswith('_')]
    first    = [c for c in _DISPLAY_FIRST if c in df.columns]
    rest     = [c for c in df.columns if c not in first and c not in internal]
    return df[first + rest]


def _save_results(client: storage.Client, run_id: str, df: pd.DataFrame) -> str:
    path = f'{_RESULTS_PREFIX}{run_id}/results.csv'
    client.bucket(_BUCKET).blob(path).upload_from_string(
        df.to_csv(index=False).encode('utf-8'), content_type='text/csv'
    )
    return path


# ── Session state ──────────────────────────────────────────────────────────

for _k in ['am_profiles', 'am_topics_df', 'am_results', 'am_run_meta']:
    if _k not in st.session_state:
        st.session_state[_k] = None
if 'am_filters' not in st.session_state:
    st.session_state.am_filters = [{'column': None, 'type': 'keyword', 'keyword': '', 'operator': 'AND'}]
# Client picker: a data_editor keeps its own edit state, so All/None flip the
# default and bump the nonce in the widget key to force a fresh table.
if 'am_pick_all' not in st.session_state:
    st.session_state.am_pick_all = True
if 'am_pick_nonce' not in st.session_state:
    st.session_state.am_pick_nonce = 0


# ── Page ───────────────────────────────────────────────────────────────────

st.title('🎯 Bulk Aspect Match')
st.caption(
    'Score every client aspect against every selected grant topic, keep the '
    'topics that clear the threshold, then have Claude re-rank the survivors.'
)

gcs = _get_storage_client()

if st.session_state.am_profiles is None:
    with st.spinner('Loading client profiles…'):
        try:
            st.session_state.am_profiles = ap.load_profiles(gcs)
        except Exception as e:
            st.error(f'Could not load {ap.PROFILES_BLOB}: {e}')
            st.stop()

profiles: pd.DataFrame = st.session_state.am_profiles

if profiles.empty:
    st.warning(
        'No client profiles found. Build them in the **Client Profiles** view first.'
    )
    st.stop()

# ── Section 1 · Clients ────────────────────────────────────────────────────

st.subheader('1 · Select clients')

head_l, head_r = st.columns([1, 5])
with head_l:
    if st.button('↺ Reload profiles'):
        st.session_state.am_profiles = None
        st.rerun()
with head_r:
    st.info(
        f'{len(profiles):,} profiled client'
        f'{"s" if len(profiles) != 1 else ""} · '
        f'{int(pd.to_numeric(profiles["n_aspects"], errors="coerce").fillna(0).sum()):,} aspects total'
    )

qs1, qs2, _qs3 = st.columns([1, 1, 6])
if qs1.button('Select all', key='am_pick_all_btn'):
    st.session_state.am_pick_all    = True
    st.session_state.am_pick_nonce += 1
    st.rerun()
if qs2.button('Deselect all', key='am_pick_none_btn'):
    st.session_state.am_pick_all    = False
    st.session_state.am_pick_nonce += 1
    st.rerun()

picker = pd.DataFrame({
    'use':      st.session_state.am_pick_all,
    'client':   profiles['company_name'].fillna('—').astype(str),
    'aspects':  pd.to_numeric(profiles['n_aspects'], errors='coerce').fillna(0).astype(int),
    'labels':   profiles['aspect_labels'].fillna('').astype(str),
    'built_at': profiles['built_at'].fillna('').astype(str),
})

edited = st.data_editor(
    picker,
    hide_index=True,
    use_container_width=True,
    height=min(400, 60 + 36 * len(picker)),
    disabled=['client', 'aspects', 'labels', 'built_at'],
    column_config={
        'use':      st.column_config.CheckboxColumn('Use'),
        'client':   st.column_config.TextColumn('Client'),
        'aspects':  st.column_config.NumberColumn('Aspects', format='%d'),
        'labels':   st.column_config.TextColumn('Aspect labels', width='large'),
        'built_at': st.column_config.TextColumn('Built'),
    },
    key=f'am_client_picker_{st.session_state.am_pick_nonce}',
)

selected = profiles.loc[edited.index[edited['use'].fillna(False).to_numpy(dtype=bool)]]
if selected.empty:
    st.warning('Select at least one client.')

max_aspects = int(pd.to_numeric(selected['n_aspects'], errors='coerce').fillna(0).max()) if not selected.empty else 1

# ── Section 2 · Grant topics ───────────────────────────────────────────────

st.divider()
st.subheader('2 · Select grant topics')

agencies = _list_agencies(gcs)
if not agencies:
    st.warning('No grant agencies found in GCS.')
    st.stop()

# Quick-select buttons run before the checkboxes are created, so writing their
# session-state keys here is what sets the widget values for this run.
ags1, ags2, _ags3 = st.columns([1, 1, 6])
if ags1.button('Select all', key='am_agency_all'):
    for ag in agencies:
        st.session_state[f'am_agency_{ag}'] = True
    st.rerun()
if ags2.button('Deselect all', key='am_agency_none'):
    for ag in agencies:
        st.session_state[f'am_agency_{ag}'] = False
    st.rerun()

ag_cols = st.columns(min(len(agencies), 6))
selected_agencies = [
    ag for i, ag in enumerate(agencies)
    if ag_cols[i % len(ag_cols)].checkbox(ag, value=True, key=f'am_agency_{ag}')
]

if st.button('Load Topics', type='primary', disabled=not selected_agencies):
    with st.spinner(f'Loading topics from {len(selected_agencies)} agenc'
                    f'{"y" if len(selected_agencies) == 1 else "ies"}…'):
        topics = _load_topics(gcs, selected_agencies)
    if topics.empty:
        st.warning('No topics found for the selected agencies.')
    else:
        st.session_state.am_topics_df = topics
        st.session_state.am_results   = None
        st.session_state.am_filters   = [{'column': None, 'type': 'keyword', 'keyword': '', 'operator': 'AND'}]
        st.success(f'Loaded **{len(topics):,}** topics.')

if st.session_state.am_topics_df is None:
    st.stop()

topics_df       = st.session_state.am_topics_df
filterable_cols = [c for c in topics_df.columns if c != 'embeddings']

for f in st.session_state.am_filters:
    if f['column'] not in filterable_cols:
        f['column'] = filterable_cols[0] if filterable_cols else None

for i, f in enumerate(st.session_state.am_filters):
    if i == 0:
        col_sel, mode_col, val_input, remove_col = st.columns([2, 1.4, 3, 0.5])
    else:
        op_col, col_sel, mode_col, val_input, remove_col = st.columns([1, 2, 1.4, 3, 0.5])
        f['operator'] = op_col.radio(
            'op', ['AND', 'OR'], index=0 if f['operator'] == 'AND' else 1,
            key=f'am_op_{i}', horizontal=True, label_visibility='collapsed',
        )

    f['column'] = col_sel.selectbox(
        'Column', filterable_cols,
        index=filterable_cols.index(f['column']) if f['column'] in filterable_cols else 0,
        key=f'am_col_{i}', label_visibility='collapsed',
    )
    mode_label = mode_col.selectbox(
        'Filter type', ['Keyword', 'Date range'],
        index=1 if f.get('type') == 'date_range' else 0,
        key=f'am_type_{i}', label_visibility='collapsed',
    )
    f['type'] = 'date_range' if mode_label == 'Date range' else 'keyword'
    if f['type'] == 'date_range':
        picked = val_input.date_input(
            'Date range',
            value=(
                f.get('date_from') or date.today() - timedelta(days=30),
                f.get('date_to') or date.today(),
            ),
            key=f'am_dr_{i}', label_visibility='collapsed',
        )
        if isinstance(picked, tuple) and len(picked) == 2:
            f['date_from'], f['date_to'] = picked
        elif isinstance(picked, tuple) and len(picked) == 1:
            # Mid-selection: only the start date is chosen so far.
            f['date_from'], f['date_to'] = picked[0], None
    else:
        f['keyword'] = val_input.text_input(
            'Keyword', value=f.get('keyword', ''),
            placeholder=f'Filter by {f["column"]}…',
            key=f'am_kw_{i}', label_visibility='collapsed',
        )
    if remove_col.button('✕', key=f'am_rm_{i}', disabled=len(st.session_state.am_filters) == 1):
        st.session_state.am_filters.pop(i)
        st.rerun()

if st.button('+ Add filter', key='am_add_filter'):
    st.session_state.am_filters.append(
        {'column': filterable_cols[0], 'type': 'keyword', 'keyword': '', 'operator': 'AND'}
    )
    st.rerun()

filtered = _apply_filters(topics_df, st.session_state.am_filters)
st.caption(f'**{len(filtered):,}** topics match current filters — showing first 25')
st.dataframe(filtered[filterable_cols].head(25), use_container_width=True, hide_index=True)

# ── Section 3 · Match options ──────────────────────────────────────────────

st.divider()
st.subheader('3 · Match options')

mode = st.radio(
    'Match mode', [_MODE_ALL, _MODE_MARKET], horizontal=True, key='am_mode',
    help='Whole company scores every aspect at once. By market scores each '
         'market separately — its earmarked aspects plus the market narrative '
         '— so a client\'s markets compete for topics on their own.',
)
by_market = mode == _MODE_MARKET

kind_label = _KIND_CONFIRMED
kinds      = _KIND_ROWS[_KIND_CONFIRMED]
tiers: set[int] = set()
category   = _CATEGORY_ALL

if by_market:
    m1, m2, m3 = st.columns([2, 2, 2])
    with m1:
        kind_label = st.selectbox(
            'Market kind', _KINDS, index=0, key='am_kind',
            help='Confirmed markets are where the client sells today. Unexplored '
                 'markets are ones the profile builder inferred by linking the '
                 'client\'s aspects — scored on the market narrative alone, and '
                 're-ranked by a different prompt that asks how plausibly the '
                 'client could extend into the topic rather than whether it could '
                 'propose today. Check the `market_kind` column before treating a '
                 'high score as a real fit.',
        )
    kinds = _KIND_ROWS[kind_label]

    counts      = _market_counts(selected, kinds)
    tier_counts = _tier_counts(selected, kinds)
    if not counts:
        st.warning(
            f'None of the selected clients have {kind_label.lower()} yet. Rebuild '
            'their profiles in **Client Profiles** to match by market'
            + (' with "Assess unexplored markets" on.'
               if _ROW_UNEXPLORED in kinds else '.')
        )

    # Both widgets are keyed, and their option lists change with the kind and
    # the client selection — Streamlit raises when a stored value is no longer
    # an option, so prune before the widgets are created.
    tier_opts = list(tier_counts.keys())
    if 'am_tiers' in st.session_state:
        st.session_state.am_tiers = [
            r for r in st.session_state.am_tiers if r in tier_counts
        ]
    with m2:
        tiers = set(st.multiselect(
            'Market tiers', options=tier_opts, default=tier_opts, key='am_tiers',
            format_func=lambda r: f'{ap.tier_ordinal(r)} ({tier_counts[r]} markets)',
            help='Markets are ranked 1st, 2nd, 3rd… by how core they are to the '
                 'business. Leave empty for every tier.',
        ))

    options = [_CATEGORY_ALL] + list(counts.keys())
    if st.session_state.get('am_category') not in options:
        st.session_state.pop('am_category', None)
    with m3:
        category = st.selectbox(
            'Market category', options, index=0, key='am_category',
            format_func=lambda c: c if c == _CATEGORY_ALL else f'{c} ({counts[c]} clients)',
            help='Scoped to the markets the selected clients actually have. '
                 f'Pick {ap.DEFENSE_MARKET} here to run defense on its own.',
        )

plan, plan_skipped = _plan_units(selected, by_market, kinds, tiers, category)

o1, o2, o3 = st.columns(3)
with o1:
    threshold = st.slider('Aspect similarity threshold', 0.60, 0.95, 0.78, 0.01)
with o2:
    min_hits = st.number_input(
        'Aspects that must clear it', min_value=1, max_value=max(1, max_aspects), value=1, step=1,
        help='1 = any single capability matching is enough (recommended — a client\'s '
             'aspects are different capabilities, not requirements of one query). '
             'Raise it to demand topics that touch several of the client\'s capabilities. '
             'The market narrative does not count towards it.',
    )
with o3:
    top_k = st.number_input(
        'Top topics per market' if by_market else 'Top topics per client',
        min_value=1, max_value=100, value=10, step=1,
    )

if by_market:
    n_unexp = sum(1 for u in plan if u.kind == _ROW_UNEXPLORED)
    st.caption(
        f'**{len(plan)}** market unit{"s" if len(plan) != 1 else ""} across '
        f'**{len({str(u.prof["company_key"]) for u in plan})}** client(s) — each is '
        'scored, capped and ranked as if it were its own company.'
        + (f' {n_unexp} of them are unexplored markets, scored on their narrative '
           'alone.' if n_unexp else '')
    )

r1, r2, r3 = st.columns(3)
with r1:
    do_rerank = st.checkbox('LLM re-rank', value=True)
with r2:
    rerank_model = st.selectbox('Re-rank model', _RERANK_MODELS, index=0, disabled=not do_rerank)
with r3:
    min_llm = st.number_input(
        'Keep LLM score ≥', min_value=1, max_value=5, value=3, step=1, disabled=not do_rerank,
    )

max_pairs = len(plan) * int(top_k)
if do_rerank:
    st.caption(
        f'Up to **{max_pairs:,}** re-rank calls '
        f'({len(plan)} {"market unit" if by_market else "client"}'
        f'{"s" if len(plan) != 1 else ""} × top {int(top_k)}), '
        f'{_CONCURRENCY} at a time.'
        + (' Identical (client, topic, aspect) pairs are scored once and shared '
           'across markets, so the real count is usually lower.' if by_market else '')
    )
confirm = True
if do_rerank and max_pairs > _CONFIRM_PAIRS:
    confirm = st.checkbox(
        f'I understand this can make up to {max_pairs:,} Claude calls and the page '
        'must stay open until it finishes.',
        value=False,
    )

for _msg in plan_skipped:
    st.warning(_msg)

run = st.button(
    '▶ Run match', type='primary',
    disabled=selected.empty or filtered.empty or not plan or not confirm,
)

# ── Run ────────────────────────────────────────────────────────────────────

if run:
    with st.spinner('Preparing topic vectors…'):
        topic_matrix, topic_meta = _stack_topic_embeddings(filtered)

    if topic_matrix.shape[0] == 0:
        st.error('None of the filtered topics carry a usable embedding.')
        run = False
    elif len(topic_meta) < len(filtered):
        st.warning(
            f'{len(filtered) - len(topic_meta):,} topic(s) skipped — missing or '
            'malformed embedding.'
        )

if run:
    # st.stop() raises, so it must not be used inside this handler — every
    # failure path below reports and falls through to the results section.
    try:
        prog = st.progress(0.0, text='Scoring…')
        candidates, skipped = _match_units(
            plan, topic_matrix, topic_meta,
            float(threshold), int(min_hits), int(top_k), prog,
        )
        prog.empty()
        del topic_matrix

        for msg in skipped:
            st.warning(msg)

        if candidates.empty:
            st.session_state.am_results = candidates
            st.session_state.am_run_meta = {
                'run_id': None, 'threshold': float(threshold), 'min_hits': int(min_hits),
                'top_k': int(top_k), 'clients': len(selected), 'topics': len(topic_meta),
                'candidates': 0, 'reranked': False, 'kept': 0, 'unscored': 0, 'gcs_path': None,
                'mode': mode, 'kind': kind_label, 'category': category,
                'tiers': sorted(tiers), 'units': len(plan),
            }
        else:
            reranked = False
            unscored = 0
            failures: dict[str, int] = {}
            results  = candidates
            if do_rerank:
                results  = _run_rerank(candidates, st.secrets['anthropic_api_key'], rerank_model)
                unscored = int((results['llm_score'] == 0).sum())
                # A pair scores 0 only when the call failed or the answer was
                # unparseable — the reason is the one thing worth surfacing when
                # a run ends with nothing to show.
                failures = (
                    results.loc[results['llm_score'] == 0, 'llm_rationale']
                    .astype(str).value_counts().head(5).to_dict()
                )
                reranked = True
                results  = results[results['llm_score'] >= int(min_llm)]
                results  = results.sort_values(
                    ['llm_score', 'aspect_score'], ascending=[False, False]
                ).reset_index(drop=True)
            else:
                results = results.sort_values(
                    ['aspect_score', 'aspects_hit'], ascending=[False, False]
                ).reset_index(drop=True)

            run_id = f'aspect_match_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
            gcs_path = None
            if not results.empty:
                try:
                    gcs_path = _save_results(gcs, run_id, _display_frame(results))
                except Exception as e:
                    st.warning(f'Results not saved to GCS ({e}) — the CSV download still works.')

            st.session_state.am_results  = results
            st.session_state.am_run_meta = {
                'run_id': run_id, 'threshold': float(threshold), 'min_hits': int(min_hits),
                'top_k': int(top_k), 'clients': len(selected), 'topics': len(topic_meta),
                'candidates': len(candidates), 'reranked': reranked,
                'kept': len(results), 'unscored': unscored, 'failures': failures,
                'gcs_path': gcs_path,
                'mode': mode, 'kind': kind_label, 'category': category,
                'tiers': sorted(tiers), 'units': len(plan),
                'rerank_calls': len(_rerank_groups(candidates)) if reranked else 0,
            }

    except Exception as e:
        st.error(f'Match run failed: {e}')
        st.code(traceback.format_exc())

# ── Section 4 · Results ────────────────────────────────────────────────────

if st.session_state.am_results is not None:
    results = st.session_state.am_results
    meta    = st.session_state.am_run_meta or {}

    st.divider()
    st.subheader('4 · Results')

    if results.empty:
        st.warning(
            f'No topics cleared {meta.get("threshold")} on at least '
            f'{meta.get("min_hits")} aspect(s)'
            + (' and survived re-ranking.' if meta.get('reranked') else '.')
            + ' Try lowering the threshold or widening the topic selection.'
        )
        # Similarity and re-ranking fail very differently, and lowering the
        # threshold cannot rescue a run the re-ranker dropped.
        if meta.get('candidates'):
            st.info(
                f'Similarity scoring found **{meta["candidates"]:,}** candidate row(s) '
                f'across {meta.get("units", meta.get("clients", 0))} unit(s) — they were '
                'dropped during re-ranking, not by the similarity threshold.'
            )
        if meta.get('unscored'):
            st.error(
                f'**{meta["unscored"]:,}** of them could not be scored by Claude at all. '
                'An unscored pair is stored as 0, which is below every selectable '
                'minimum, so the whole run is filtered away. Reported reasons:'
            )
            for reason, n in (meta.get('failures') or {}).items():
                st.markdown(f'- `{reason}` × {n:,}')
            st.caption(
                'Re-run with **LLM re-rank** unchecked to see the similarity results '
                'with no Claude calls at all.'
            )
        elif meta.get('candidates') and meta.get('kind') == _KIND_UNEXPLORED:
            # An unexplored run scores plausibility of extension, not readiness
            # to propose - it lands lower than a confirmed run by design.
            st.info(
                'This was an **unexplored markets** run: every row is a market '
                'the client does not serve yet, so the re-ranker was asked how '
                'plausibly it could extend into the topic. Those scores sit '
                f'lower than confirmed ones by design — try a minimum of 3 or 2 '
                'before concluding there is nothing there.'
            )
    else:
        m = st.columns(5 if meta.get('mode') == _MODE_MARKET else 4)
        m[0].metric('Rows', f'{len(results):,}')
        m[1].metric('Clients matched', f'{results["client"].nunique():,}')
        if meta.get('mode') == _MODE_MARKET:
            m[2].metric(
                'Markets matched',
                f'{results.groupby(["client", "market_kind", "market"]).ngroups:,}',
            )
        m[-2].metric('Candidates scored', f'{meta.get("candidates", 0):,}')
        m[-1].metric('Topics searched', f'{meta.get("topics", 0):,}')

        if meta.get('mode') == _MODE_MARKET:
            st.caption(
                f'Ran **{meta.get("units", 0)}** market unit(s) · kind '
                f'*{meta.get("kind")}* · tier'
                f'{"s" if len(meta.get("tiers") or []) != 1 else ""} '
                f'*{", ".join(ap.tier_ordinal(t) for t in (meta.get("tiers") or [])) or "all"}*'
                f' · category *{meta.get("category")}*'
                + (f' · {meta["rerank_calls"]:,} re-rank call(s) for {meta.get("candidates", 0):,} '
                   'candidate row(s)' if meta.get('rerank_calls') else '')
            )

        if meta.get('unscored'):
            st.warning(
                f'{meta["unscored"]} pair(s) could not be scored by the re-ranker '
                '(shown as score 0 and dropped by the minimum score)'
                + (': ' + '; '.join(f'{r} × {n}' for r, n in (meta.get('failures') or {}).items())
                   if meta.get('failures') else '.')
            )
        if meta.get('gcs_path'):
            st.caption(f'Saved to `{meta["gcs_path"]}`')

        display = _display_frame(results)
        col_cfg = {
            'client':        st.column_config.TextColumn('Client'),
            'market':        st.column_config.TextColumn('Market'),
            'market_kind':   st.column_config.TextColumn('Kind', width='small'),
            'market_tier':   st.column_config.TextColumn('Tier', width='small'),
            'aspect_label':  st.column_config.TextColumn('Matched aspect'),
            'aspect_score':  st.column_config.NumberColumn('Aspect score', format='%.4f'),
            'aspects_hit':   st.column_config.NumberColumn('Aspects hit', format='%d'),
            'aspects_total': st.column_config.NumberColumn('Aspects', format='%d'),
        }
        if 'llm_score' in display.columns:
            col_cfg['llm_score']     = st.column_config.NumberColumn('LLM score', format='%d')
            col_cfg['llm_rationale'] = st.column_config.TextColumn('Rationale', width='large')

        st.dataframe(display, use_container_width=True, hide_index=True, column_config=col_cfg)

        st.download_button(
            '⬇ Download CSV',
            data=display.to_csv(index=False).encode('utf-8'),
            file_name=f'{meta.get("run_id") or "aspect_match"}.csv',
            mime='text/csv',
        )

        by = (['client', 'market_kind', 'market'] if meta.get('mode') == _MODE_MARKET
              and 'market' in results.columns else ['client'])
        with st.expander('Per-market summary' if len(by) > 1 else 'Per-client summary'):
            agg = {
                'topics':            ('client', 'size'),
                'best_aspect_score': ('aspect_score', 'max'),
            }
            if 'llm_score' in results.columns:
                agg['best_llm_score'] = ('llm_score', 'max')
            summary = (
                results.groupby(by)
                .agg(**agg)
                .reset_index()
                .sort_values('topics', ascending=False)
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)
