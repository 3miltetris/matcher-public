"""
Multi-aspect client profiles
----------------------------
Streamlit-free helpers shared by the Client Profiles view (builds profiles)
and the Bulk Aspect Match view (consumes them).

A client's grant-relevant material is spread across several columns of
data/all-contacts/clients/ — the website summary/scrape, Drive document
extractions written by drive-sync-job, and Deep Research output written by
the Client Research view. A multi-aspect profile distills whatever is
available into a handful of independently searchable aspects (one
capability / technology / domain each), embeds each aspect separately, and
stores one row per company at

    data/client-profiles/profiles.parquet

so grant topics can be scored per aspect instead of against a single
blended company summary.

Aspect vectors are stored FLAT — n_aspects * embedding_dim float64 values
in one list column, with n_aspects/embedding_dim alongside. A flat list of
doubles round-trips through parquet without the dtype ambiguity of nested
list columns (the same reason embeddings elsewhere in this codebase are
written as float64). Use unpack_embeddings() to get the (n_aspects, dim)
matrix back.
"""

import difflib
import hashlib
import io
import json
import re
from datetime import date

import numpy as np
import pandas as pd

import src.modules.finance_research as fr
import src.modules.tech_research as tr
from src.modules.GoogleBucketManager.bucket_manager import BucketManager

# ── Constants ───────────────────────────────────────────────────────────────

BUCKET          = 'cc-matcher-bucket-jeg-v1'
CLIENTS_PREFIX  = 'data/all-contacts/clients/'
PROFILES_PREFIX = 'data/client-profiles/'
PROFILES_BLOB   = f'{PROFILES_PREFIX}profiles.parquet'

ASPECT_MODELS = ['claude-sonnet-4-6', 'claude-haiku-4-5-20251001']
DEFAULT_MODEL = 'claude-sonnet-4-6'

MIN_ASPECTS  = 2
MAX_ASPECTS  = 12
ASPECT_KINDS = ['technology', 'capability', 'product', 'domain', 'market']
EMBED_DIM    = 1536
# Cosine between two aspect vectors of the SAME company above which they are
# one capability written twice, and are merged at build time.
#
# MEASURED, not reasoned: across three real clients rebuilt at 10-11 aspects,
# the whole within-company pairwise distribution topped out at 0.89 / 0.90 /
# 0.92 (median ~0.81). An earlier 0.96 - picked by reasoning about ada-002's
# compressed range rather than by measuring - was unreachable and fired zero
# times, which is strictly worse than not having the feature.
#
# The awkward finding is that cosine does NOT cleanly separate "same capability
# twice" from "two facets of one platform". A 0.90 setting was then measured
# across the whole book (277 clients): it fired 33 times, and roughly 12-14 of
# those merges were WRONG in a systematic way - it absorbs a platform into its
# own application, or merges two genuinely distinct products:
#
#   FOS Myopia Control Contact Lenses  -> FOS Myopia Control Spectacle Lenses
#   Dendritic Cell Immunotherapy ASY-77A -> csHsp70-Targeted ADC Platform
#   Coronary Bifurcation Stenting Platform -> FDA Breakthrough Device IP Portfolio
#   Occupational Radiation Safety Monitoring -> Electronic Polymer Dosimeter Patch
#
# A platform and its applications SHOULD be separate aspects - they match
# different grant topics, which is the entire point of multi-aspect profiling.
# The 383 reported near-misses clustered tightly (median 0.867, p90 0.888, max
# 0.900), so 0.90 sat inside a dense band rather than above it. 0.94 puts the
# auto-merge above that band, where only near-identical text reaches: it fires
# rarely and does little harm when it does. Duplicate detection in practice is
# nearest_aspect_pairs() below - a reviewed list beats a silent deletion.
#
# Every merge is reported WITH its cosine, so the next adjustment can be made
# from data rather than guessed (this round it could not: only the near-misses
# carried scores).
ASPECT_MERGE_THRESHOLD = 0.94
# Pairs at or above this are reported for review even when they do not merge.
ASPECT_NEAR_FLOOR = 0.85

# ── Markets ───────────────────────────────────────────────────────────────
# A market is the customer world a company sells its aspects into. Names come
# from this fixed vocabulary so "Defense" means the same thing for every
# client and one dropdown can span the whole directory; everything
# client-specific lives in the free-form subtitle and narrative. Defense is
# assessed separately and does not count toward MAX_MARKETS.
MARKET_CATEGORIES = [
    'Defense',
    'Aerospace & Space',
    'Health & Life Sciences',
    'Energy & Power',
    'Environment & Climate',
    'Agriculture & Food',
    'Transportation & Mobility',
    'Manufacturing & Industrial',
    'Microelectronics & Semiconductors',
    'Communications & Networking',
    'Cybersecurity & IT',
    'Materials & Chemicals',
    'Water & Infrastructure',
    'Maritime',
    'Public Safety & Emergency Response',
    'Education & Training',
    'Other',
]
DEFENSE_MARKET = 'Defense'
MAX_MARKETS    = 4      # non-defense markets; Defense is additional
# Markets are ranked 1st, 2nd, 3rd... rather than bucketed primary/secondary,
# because everything downstream already wants an ordering. The rank is stored as
# an INTEGER and rendered as an ordinal only at display time - storing "1st"
# would repeat the prose-instead-of-number mistake the HubSpot _num companions
# exist to undo. Profiles written before ranks existed carry 'primary' /
# 'secondary' strings; market_tier_rank() coerces them on read, so no profile
# has to be rebuilt to stay usable.
MAX_MARKET_TIER = MAX_MARKETS + 1        # the non-defense cap, plus Defense
_LEGACY_TIERS   = {'primary': 1, 'secondary': 2}
# Cosine between two market narrative vectors of the SAME company above which
# they are describing one market twice, and are merged at build time.
MARKET_MERGE_THRESHOLD = 0.93

# ── Unexplored markets ────────────────────────────────────────────────────
# A second Claude pass links a company's existing aspects into markets it does
# NOT serve yet. Those are hypotheses, so they get their own columns, their own
# embedding block and their own re-rank prompt - never a status flag on the
# confirmed `markets` array, which every consumer would then have to remember to
# filter. Defense is assessed here too: a client with no confirmed Defense
# market can still surface an unexplored one.
MAX_UNEXPLORED     = 3
MIN_LINKED_ASPECTS = 2

PROFILE_COLUMNS = [
    'company_key', 'company_name', 'companyWebsite',
    'profile_summary', 'aspects', 'aspect_labels',
    'n_aspects', 'embedding_dim', 'aspect_embeddings',
    'markets', 'market_labels', 'n_markets', 'market_embeddings',
    'unexplored_markets', 'unexplored_labels', 'n_unexplored',
    'unexplored_embeddings',
    'dod_assessment',
    'sources_used', 'source_fingerprint', 'model', 'built_at',
]

# Values that mean "no data" in the research/docs columns — the research
# prompts are told to write "Not found" rather than leave a field empty.
_NULLISH = ('', 'not found', 'unknown', 'n/a', 'none', 'nan', '-')


# ── Small helpers ───────────────────────────────────────────────────────────

def _s(value) -> str:
    """Trimmed string, or '' for null-ish values."""
    if value is None:
        return ''
    try:
        if isinstance(value, float) and np.isnan(value):
            return ''
    except TypeError:
        pass
    s = str(value).strip()
    return '' if s.lower() in _NULLISH else s


def _cap(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + '\n…[truncated]'


def _json_obj(value) -> dict:
    raw = _s(value)
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _flatten_kv(obj, prefix: str = '') -> list[str]:
    """Render a nested dict/list of extracted fields as `key: value` lines,
    dropping empty and "Not found" values."""
    lines: list[str] = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            path = f'{prefix}{key}'
            if isinstance(val, (dict, list)):
                lines.extend(_flatten_kv(val, f'{path}.'))
            elif _s(val):
                lines.append(f'{path}: {_s(val)}')
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            if isinstance(val, (dict, list)):
                lines.extend(_flatten_kv(val, f'{prefix}{i}.'))
            elif _s(val):
                lines.append(f'{prefix}{i}: {_s(val)}')
    return lines


def company_key(row) -> str:
    """Same identity used by the clients/ views: name||website."""
    name    = _s(row.get('company_name')) or _s(row.get('companyName'))
    website = _s(row.get('companyWebsite'))
    return f'{name}||{website}'


# ── Per-company material row ────────────────────────────────────────────────

# Columns that can carry profile source material. Shared by the Client
# Profiles view (directory + staleness) and client-profile-job (build).
MATERIAL_COLS = [
    'company_name', 'companyWebsite', 'state',
    'summary', 'company_summary', 'full_text', 'page_text',
    'client_docs_summary', 'client_docs_data',
    'client_meetings_summary', 'client_meetings_data',
    'technology_data', 'technology_summary',
    'financial_data', 'financial_summary',
]


def first_nonempty(values):
    """First value that isn't null-ish, or None."""
    for v in values:
        if _s(v):
            return v
    return None


def merge_company_row(group) -> dict:
    """One representative dict per company. Research/docs columns are written
    to every contact row of a company, but a partially-updated file can leave
    some rows blank — take the first non-empty value per column."""
    out: dict = {}
    for col in MATERIAL_COLS:
        if col in group.columns:
            out[col] = first_nonempty(group[col].tolist())
    return out


# ── Source material extraction ──────────────────────────────────────────────

def _website_text(row) -> str:
    parts = []
    summary = _s(row.get('summary')) or _s(row.get('company_summary'))
    if summary:
        parts.append('Company summary:\n' + summary)
    page = _s(row.get('full_text')) or _s(row.get('page_text'))
    if page:
        parts.append('Scraped website text:\n' + _cap(page, 4000))
    return '\n\n'.join(parts)


def _drive_text(row) -> str:
    parts = []
    digest = _s(row.get('client_docs_summary'))
    if digest:
        parts.append('Drive document digest:\n' + digest)
    data  = _json_obj(row.get('client_docs_data'))
    lines = _flatten_kv(data.get('extracted') or {})
    if lines:
        parts.append('Extracted from Drive documents:\n' + '\n'.join(lines))
    names = [
        _s(f.get('name')) for f in (data.get('source_files') or [])
        if isinstance(f, dict) and _s(f.get('name'))
    ]
    if names:
        parts.append('Source documents: ' + ', '.join(names[:40]))
    return '\n\n'.join(parts)


def _meetings_text(row) -> str:
    """Material distilled from the client's Fathom calls by fathom-sync-job.

    Deliberately digest + extraction only, with no list of meeting titles: the
    titles are provenance (they live in client_meetings_data['meetings'] and
    are shown in the Fathom Meetings view), and folding them in here would
    change source_fingerprint — and so flag every profile stale — every time a
    purely administrative call was ingested."""
    parts = []
    digest = _s(row.get('client_meetings_summary'))
    if digest:
        parts.append('Client meeting digest:\n' + digest)
    data  = _json_obj(row.get('client_meetings_data'))
    lines = _flatten_kv(data.get('extracted') or {})
    if lines:
        parts.append('Extracted from client meetings:\n' + '\n'.join(lines))
    return '\n\n'.join(parts)


def _research_text(row, data_col: str, summary_col: str, fields: list[str]) -> str:
    data = _json_obj(row.get(data_col))
    if data:
        lines = [f'{f}: {_s(data.get(f))}' for f in fields if _s(data.get(f))]
        if lines:
            return '\n'.join(lines)
    return _s(row.get(summary_col))


def _tech_text(row) -> str:
    return _research_text(row, 'technology_data', 'technology_summary', tr.ALL_FIELDS)


def _fin_text(row) -> str:
    return _research_text(row, 'financial_data', 'financial_summary', fr.ALL_FIELDS)


def stated_intentions(row, limit: int = 20) -> list[str]:
    """What the client SAID it intends or is considering, as opposed to
    capability it demonstrably has.

    fathom-sync-job deliberately pushes anything hypothetical, planned, or
    belonging to a third party into extracted['notable_updates'] instead of the
    capability arrays, and drive-sync-job may do the same. Those lines already
    reach the aspect prompt (via _meetings_text flattening all of `extracted`),
    where _RULES now tells the model to ignore them. The unexplored-market pass
    is the one place they are useful: a market the client is already thinking
    about is the strongest candidate there is - as a lead, never as capability."""
    out: list[str] = []
    for col in ('client_meetings_data', 'client_docs_data'):
        extracted = _json_obj(row.get(col)).get('extracted')
        if not isinstance(extracted, dict):
            continue
        updates = extracted.get('notable_updates')
        if isinstance(updates, str):
            updates = [updates]
        if isinstance(updates, np.ndarray):
            updates = list(updates)
        for item in (updates or []):
            text = _s(item)
            if text and text not in out:
                out.append(text[:400])
    return out[:limit]


# Order matters — this is the order sources are shown to the model and in
# the UI. `default` seeds the include-checkboxes in the Client Profiles view;
# financial material is off by default because it describes the company's
# money, not its capabilities.
SOURCES: list[dict] = [
    {'key': 'website',    'label': 'Website summary / scrape',    'default': True,
     'cap':  8000, 'extract': _website_text},
    {'key': 'drive',      'label': 'Drive documents',             'default': True,
     'cap': 14000, 'extract': _drive_text},
    {'key': 'meetings',   'label': 'Client meetings (Fathom)',     'default': True,
     'cap': 14000, 'extract': _meetings_text},
    {'key': 'technology', 'label': 'Deep Research — technology',  'default': True,
     'cap': 14000, 'extract': _tech_text},
    {'key': 'financials', 'label': 'Deep Research — financials',  'default': False,
     'cap':  5000, 'extract': _fin_text},
]

SOURCE_KEYS   = [s['key'] for s in SOURCES]
SOURCE_LABELS = {s['key']: s['label'] for s in SOURCES}


def assemble_source_texts(row, keys: list[str] | None = None) -> dict[str, str]:
    """{source_key: capped source text} for every source with material.
    Pass `keys` to restrict to a subset of SOURCE_KEYS."""
    wanted = set(keys) if keys is not None else set(SOURCE_KEYS)
    out: dict[str, str] = {}
    for src in SOURCES:
        if src['key'] not in wanted:
            continue
        try:
            text = src['extract'](row)
        except Exception:
            text = ''
        if text and text.strip():
            out[src['key']] = _cap(text, src['cap'])
    return out


def source_fingerprint(source_texts: dict[str, str]) -> str:
    """Short digest of the material a profile was built from. Compare against
    a freshly assembled fingerprint (over ALL sources) to detect that the
    client's website/Drive/research data has changed since the last build."""
    payload = json.dumps(
        {k: source_texts[k] for k in sorted(source_texts)}, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


# ── Prompt ──────────────────────────────────────────────────────────────────

_RULES = """You are a technical analyst at a firm that writes federal grant proposals for its clients. You build multi-aspect capability profiles: a client's material is split into a few distinct, independently searchable aspects, each embedded separately and scored against federal R&D grant topics (SBIR/STTR, BAAs, agency solicitations). You then group those aspects into the markets the company sells them into, so each market can be searched on its own.

Split the company described in the material below into {lo}-{hi} aspects (aim for about {target}), then identify the markets those aspects serve.

Aspect rules:
- Ground every aspect in the supplied material. Never invent technologies, products, customers, partners, or certifications.
- Each aspect is ONE independently searchable thing: a core technology, a technical capability, a product or platform line, a scientific domain, or an application/market area. No two aspects may restate each other.
- Write "text" the way a solicitation describes a needed capability: technical, concrete, 40-90 words. No marketing language, no boilerplate about the company being innovative or a leader.
- Financial material (revenue, headcount, funding history) is background only — never make an aspect about finances.
- Lines under "notable_updates" are things the company SAID it plans, is considering, or that belong to a third party. They are unverified: never build an aspect or a market out of them and never let them colour an aspect's text. Only capability the material shows the company has TODAY counts here.
- "keywords": 5-12 comma-separated technical terms a solicitation would use for this aspect.
- "kind": exactly one of technology, capability, product, domain, market.
- "label": at most 6 words, distinct from every other label.
- "evidence": which of the supplied sources support this aspect.
- "markets": comma-separated market names, copied exactly from the markets you return below. Every aspect must serve at least one market.
- If the material only supports fewer aspects than the range, return fewer — never pad with speculation.

Market rules:
- Identify at most {max_markets} markets{defense_extra}. A market is a customer world the company sells into, not a restatement of one capability.
- "market" MUST be copied exactly from this list, choosing the closest fit:
{categories}
- Never return the same market twice, and never split one market into near-identical entries. If two candidate markets would draw on the same aspects and read alike, return one.
- "tier": an integer rank — 1 for the market most core to the business today, then 2, 3 and so on. No two markets may share a rank. Rank Defense on the same scale as every other market: 1 when the company is defense-first, a later rank when the DoD connection is real but adjacent to what it mainly sells.
- "subtitle": at most 12 words naming what this company specifically does in that market.
- "narrative": 40-90 words on what the company offers this market and which of its capabilities it draws on, written the way a solicitation describes a needed capability. Ground it in the material.
- "keywords": 5-12 comma-separated terms a solicitation in this market would use.
- "aspects": the exact labels of the aspects this market draws on.
{defense_rules}
- Return ONLY a valid JSON object. No preamble, no markdown, no code fences."""

_DEFENSE_EXTRA = (
    ', plus a Defense market when the material provides evidence for one — '
    'Defense does not count toward that limit'
)

# A confirmed market is where the company sells TODAY, so Defense has to clear
# an evidence bar like any other. The previous wording ("include Defense
# whenever the connection is real even if it is loose") was written before the
# unexplored pass existed, when a loose dual-use angle had nowhere else to go.
# Measured cost of that: 115 of 282 profiles (41%) carried a confirmed Defense
# market with ZERO defense token anywhere in their source material - a
# music-therapy company had Defense as its second-most-core market. The
# speculative angle now belongs to pass 2, which raises it as an *unexplored*
# Defense market scored by the extension prompt.
_DEFENSE_RULES = """- Include a "Defense" market ONLY when the supplied material shows an actual defense relationship, or an active pursuit of one: DoD funding or an award (including an SBIR/STTR from a defense agency), a named defense solicitation, contract or program the company is pursuing, a defense agency or prime contractor named as a customer or partner, or a product the company itself describes as built for defense use.
- A merely plausible dual-use angle is NOT enough. If the company's technology could in principle serve defense but the material shows no defense funding, pursuit, customer or product, then omit the Defense market and give the one-sentence reason in "dod_assessment" — for example: "Dual-use potential in field-deployable diagnostics, but the material shows no DoD funding, pursuit or customer." The speculative angle is captured elsewhere in this pipeline; do not force it into this list.
- When you do include Defense, its "narrative" must state the concrete use case — who in the department would use it and for what — and rest on the evidence in the material, without inventing programs, contracts, or customers.
- Leave "dod_assessment" empty whenever a Defense market IS included."""

_NO_DEFENSE_RULES = (
    '- Do not include a Defense market unless the supplied material is itself '
    'explicitly about defense customers. Leave "dod_assessment" empty.'
)

_SHAPE = """JSON shape:
{
  "profile_summary": "<2-4 sentences on what this company actually does>",
  "aspects": [
    {
      "label": "<<=6 words>",
      "kind": "<technology|capability|product|domain|market>",
      "text": "<40-90 words, capability-style description>",
      "keywords": "<comma-separated technical terms>",
      "evidence": "<which sources support this>",
      "markets": "<comma-separated market names from the markets list>"
    }
  ],
  "markets": [
    {
      "market": "<exact name from the allowed market list>",
      "tier": <integer rank, 1 = most core to the business>,
      "subtitle": "<<=12 words on what this company does in this market>",
      "narrative": "<40-90 words on the offering and the capabilities behind it>",
      "keywords": "<comma-separated terms a solicitation in this market would use>",
      "aspects": ["<exact aspect label>", "..."]
    }
  ],
  "dod_assessment": "<one sentence, only when there is no defense application>"
}"""


def build_aspect_system(
    target_aspects: int,
    max_markets: int = MAX_MARKETS,
    assess_defense: bool = True,
) -> str:
    target = max(MIN_ASPECTS, min(MAX_ASPECTS, int(target_aspects)))
    lo     = max(MIN_ASPECTS, target - 1)
    hi     = min(MAX_ASPECTS, target + 2)
    return (
        _RULES.format(
            lo=lo, hi=hi, target=target,
            max_markets=max(1, min(MAX_MARKETS, int(max_markets))),
            defense_extra=_DEFENSE_EXTRA if assess_defense else '',
            categories='\n'.join(f'  - {c}' for c in MARKET_CATEGORIES),
            defense_rules=_DEFENSE_RULES if assess_defense else _NO_DEFENSE_RULES,
        )
        + '\n\n' + _SHAPE
    )


def build_aspect_user_message(company: dict, source_texts: dict[str, str]) -> str:
    """company keys: company_name, website, state (all optional)."""
    parts = [
        'COMPANY\n'
        f"Name: {_s(company.get('company_name')) or 'Unknown'}\n"
        f"Website: {_s(company.get('website')) or 'Unknown'}\n"
        f"State: {_s(company.get('state')) or 'Unknown'}"
    ]
    for key in SOURCE_KEYS:
        if key in source_texts:
            parts.append(f'=== SOURCE: {SOURCE_LABELS[key]} ===\n{source_texts[key]}')
    return '\n\n'.join(parts)


# ── Response parsing ────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict:
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', (raw or '').strip())
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end <= start:
        raise ValueError('no JSON object in response')
    obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError('response JSON was not an object')
    return obj


def _name_list(value) -> list[str]:
    """Comma-separated string or list -> trimmed names, order preserved,
    case-insensitively deduped."""
    if value is None:
        return []
    if isinstance(value, str):
        items = value.split(',')
    elif isinstance(value, (list, tuple, np.ndarray)):
        items = list(value)
    else:
        return []
    out, seen = [], set()
    for item in items:
        name = _s(item)
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out


_CATEGORY_BY_LOWER = {c.lower(): c for c in MARKET_CATEGORIES}


def _squash(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', text.lower())


def canonical_market(name) -> str:
    """Closest MARKET_CATEGORIES entry for a free-form market name, else
    'Other'. Empty input returns '' so callers can drop it."""
    raw = _s(name)
    if not raw:
        return ''
    lower = raw.lower()
    if lower in _CATEGORY_BY_LOWER:
        return _CATEGORY_BY_LOWER[lower]
    squashed = _squash(raw)
    for cat in MARKET_CATEGORIES:
        if _squash(cat) == squashed:
            return cat
    close = difflib.get_close_matches(lower, list(_CATEGORY_BY_LOWER), n=1, cutoff=0.85)
    if close:
        return _CATEGORY_BY_LOWER[close[0]]
    # "defense & aerospace", "healthcare / life sciences" -> the category whose
    # leading word the model kept
    for cat in MARKET_CATEGORIES[:-1]:
        head = cat.split(' & ')[0].split(' ')[0].lower()
        if len(head) >= 5 and head in lower:
            return cat
    return 'Other'


def market_tier_rank(market) -> int:
    """Integer tier rank of a market: 1 = most important.

    Accepts a market dict or a bare tier value, and coerces every historical
    form so old profiles need no rebuild - an int, a digit string ("2", "2nd"),
    or the legacy 'primary'/'secondary' buckets. Anything unrecognised sorts
    last. EVERY read of a market's `tier` goes through this."""
    value = market.get('tier') if isinstance(market, dict) else market
    if isinstance(value, bool):
        rank = MAX_MARKET_TIER
    elif isinstance(value, (int, np.integer)):
        rank = int(value)
    elif isinstance(value, (float, np.floating)):
        rank = MAX_MARKET_TIER if np.isnan(value) else int(value)
    else:
        raw = _s(value).lower()
        digits = re.match(r'(\d+)', raw)
        if raw in _LEGACY_TIERS:
            rank = _LEGACY_TIERS[raw]
        elif digits:
            rank = int(digits.group(1))
        else:
            rank = MAX_MARKET_TIER
    return max(1, min(MAX_MARKET_TIER, rank))


_ORDINAL_SUFFIXES = {1: 'st', 2: 'nd', 3: 'rd'}


def tier_ordinal(rank) -> str:
    """1 -> '1st'. Display form only; the stored value stays an int."""
    n = rank if isinstance(rank, int) else market_tier_rank(rank)
    if 11 <= (n % 100) <= 13:
        return f'{n}th'
    return f"{n}{_ORDINAL_SUFFIXES.get(n % 10, 'th')}"


def _renumber_tiers(markets: list[dict]) -> list[dict]:
    """Dense 1..N ranks following the list's current order, so capping, merging
    or a hand-edit can never leave a gap ('1st, 4th' with nothing between)."""
    for i, market in enumerate(markets, start=1):
        market['tier'] = min(i, MAX_MARKET_TIER)
    return markets


def _clean_market(item: dict) -> dict | None:
    market = canonical_market(item.get('market') or item.get('name') or item.get('label'))
    if not market:
        return None
    return {
        'market':        market,
        'tier':          market_tier_rank(item),
        'subtitle':      _s(item.get('subtitle'))[:160],
        'narrative':     _s(item.get('narrative')) or _s(item.get('description')),
        'keywords':      _s(item.get('keywords')),
        'aspect_labels': _name_list(item.get('aspects') or item.get('aspect_labels')),
        'merged_from':   _s(item.get('merged_from')),
    }


def normalize_markets(
    aspects: list[dict], markets: list[dict], max_markets: int = MAX_MARKETS
) -> tuple[list[dict], list[dict]]:
    """Canonicalise market names, collapse repeats of one category, cap the
    non-defense count, resolve aspect-market membership in both directions,
    give every aspect at least one market, and drop markets left with neither
    an aspect nor a narrative of their own.

    Membership is owned by the aspects (`aspect['markets']`); each market's
    `aspect_labels` is derived from them, so editing or deleting an aspect row
    in the Client Profiles editor can never leave a stale pointer behind.

    A market no aspect claims is *kept* as long as it has a narrative: that
    narrative is embedded and scored on its own in market-scoped runs, and for
    Defense it is the only place the DoD framing exists — dropping it because
    the model's `aspects` list did not string-match an aspect label would
    silently delete the whole DoD story."""
    defs: list[dict] = []
    by_name: dict[str, dict] = {}
    for item in markets or []:
        if not isinstance(item, dict):
            continue
        cleaned = _clean_market(item)
        if cleaned is None:
            continue
        existing = by_name.get(cleaned['market'])
        if existing is None:
            by_name[cleaned['market']] = cleaned
            defs.append(cleaned)
            continue
        # Same category twice - one market described twice.
        existing['aspect_labels'] = _name_list(
            existing['aspect_labels'] + cleaned['aspect_labels']
        )
        existing['tier'] = min(existing['tier'], cleaned['tier'])
        if len(cleaned['narrative']) > len(existing['narrative']):
            existing['narrative'] = cleaned['narrative']
            existing['subtitle']  = cleaned['subtitle'] or existing['subtitle']

    if not defs:
        for aspect in aspects:
            aspect['markets'] = []
        return aspects, []

    # Membership claimed by the markets, folded onto the aspects
    label_lookup = {_s(a.get('label')).lower(): a for a in aspects}
    for market in defs:
        for label in market['aspect_labels']:
            aspect = label_lookup.get(label.lower())
            if aspect is not None:
                aspect['markets'] = _name_list(
                    list(aspect.get('markets') or []) + [market['market']]
                )

    # Cap: Defense always survives - it is assessed separately and never counts
    # against max_markets - then the best-ranked non-defense markets.
    _order_of = {id(m): i for i, m in enumerate(defs)}
    ordered = (
        [m for m in defs if m['market'] == DEFENSE_MARKET]
        + sorted(
            (m for m in defs if m['market'] != DEFENSE_MARKET),
            key=lambda m: (m['tier'], _order_of[id(m)]),
        )
    )
    cap  = max(1, min(MAX_MARKETS, int(max_markets)))
    kept, n_other = [], 0
    for market in ordered:
        if market['market'] == DEFENSE_MARKET:
            kept.append(market)
        elif n_other < cap:
            kept.append(market)
            n_other += 1
    allowed = {m['market'] for m in kept}

    # Aspect membership, restricted to the surviving markets. An aspect the
    # model left unassigned lands in the primary market rather than dropping
    # out of every market run.
    fallback = min(kept, key=lambda m: m['tier'])['market']
    for aspect in aspects:
        names = [canonical_market(n) for n in _name_list(aspect.get('markets'))]
        aspect['markets'] = _name_list([n for n in names if n in allowed]) or [fallback]

    for market in kept:
        market['aspect_labels'] = [
            _s(a.get('label')) for a in aspects
            if market['market'] in (a.get('markets') or [])
        ]
    kept = [m for m in kept if m['aspect_labels'] or m['narrative']]
    if not kept:
        for aspect in aspects:
            aspect['markets'] = []
        return aspects, []

    survivors = {m['market'] for m in kept}
    for aspect in aspects:
        aspect['markets'] = [n for n in aspect['markets'] if n in survivors]
    # Rank first, Defense only as the tiebreak: Defense earns its place rather
    # than being pinned ahead of a client's actual core market. It stays
    # cap-exempt above, so it can never be ranked out of the profile.
    kept.sort(key=lambda m: (m['tier'], m['market'] != DEFENSE_MARKET, m['market']))
    _renumber_tiers(kept)
    return aspects, kept


def merge_similar_aspects(
    aspects: list[dict],
    vectors,
    threshold: float = ASPECT_MERGE_THRESHOLD,
) -> tuple[list[dict], list, list[str]]:
    """Fold aspects whose vectors are near-identical into one.

    A richer aspect set starts producing the same capability under two labels.
    Merging rather than dropping means the absorbed aspect's keywords and market
    membership survive on the survivor, so no market silently loses a member and
    no capability is deleted outright. The survivor is the aspect with the longer
    text - the richer description.

    Returns (aspects, vectors, merges), where `merges` are
    "0.951  absorbed -> survivor" strings for the run report: a threshold that is
    eating distinct capabilities is invisible unless the merges are named, and
    unfixable unless their scores are recorded."""
    if len(aspects) < 2 or vectors is None or len(vectors) != len(aspects):
        return aspects, [list(v) for v in (vectors or [])], []

    arr   = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1)
    norms[norms == 0] = 1.0
    unit  = arr / norms[:, None]

    order = sorted(
        range(len(aspects)),
        key=lambda i: (-len(_s(aspects[i].get('text'))), i),
    )
    absorbed: dict[int, tuple[int, float]] = {}
    keep_idx: list[int] = []
    for i in order:
        if i in absorbed:
            continue
        keep_idx.append(i)
        for j in order:
            if j == i or j in absorbed or j in keep_idx:
                continue
            score = float(unit[i] @ unit[j])
            if score >= threshold:
                absorbed[j] = (i, score)

    if not absorbed:
        return aspects, [list(v) for v in vectors], []

    merges: list[str] = []
    for j, (i, score) in sorted(absorbed.items()):
        survivor, gone = aspects[i], aspects[j]
        survivor['keywords'] = ', '.join(_name_list(
            f"{_s(survivor.get('keywords'))},{_s(gone.get('keywords'))}"
        ))
        survivor['markets'] = _name_list(
            list(survivor.get('markets') or []) + list(gone.get('markets') or [])
        )
        survivor['merged_from'] = ', '.join(
            x for x in (_s(survivor.get('merged_from')), _s(gone.get('label'))) if x
        )
        # Cosine first, matching nearest_aspect_pairs()' format, so the two
        # reports can be read against each other when tuning the threshold.
        merges.append(
            f"{score:.3f}  {_s(gone.get('label'))} -> {_s(survivor.get('label'))}"
        )

    keep_idx.sort()
    return (
        [aspects[i] for i in keep_idx],
        [list(vectors[i]) for i in keep_idx],
        merges,
    )


def nearest_aspect_pairs(
    aspects: list[dict],
    vectors,
    floor: float = ASPECT_NEAR_FLOOR,
    limit: int = 3,
) -> list[str]:
    """The closest surviving aspect pairs, as "0.894  A  ~  B" strings.

    Reported per client so a human can see the near-duplicates the threshold
    deliberately did not touch. Measurement showed cosine alone cannot reliably
    tell a genuine repeat from two facets of one platform, so the human-facing
    list is the more trustworthy half of duplicate detection - an auto-merge
    tuned aggressively enough to catch the real repeats would also delete
    legitimate distinctions."""
    if len(aspects) < 2 or vectors is None or len(vectors) != len(aspects):
        return []
    arr   = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1)
    norms[norms == 0] = 1.0
    unit  = arr / norms[:, None]
    sims  = unit @ unit.T

    found = []
    for i in range(len(aspects)):
        for j in range(i + 1, len(aspects)):
            score = float(sims[i, j])
            if score >= floor:
                found.append((score, _s(aspects[i].get('label')),
                              _s(aspects[j].get('label'))))
    found.sort(reverse=True)
    return [f'{sc:.3f}  {a}  ~  {b}' for sc, a, b in found[:limit]]


def merge_similar_markets(
    aspects: list[dict],
    markets: list[dict],
    vectors,
    threshold: float = MARKET_MERGE_THRESHOLD,
) -> tuple[list[dict], list[dict], list]:
    """Fold markets whose narrative vectors are near-identical into one.

    Two different categories can still describe the same business ("Energy &
    Power" and "Environment & Climate" written about one product line). The
    survivor is the Defense / higher-tier / earlier market; it absorbs the
    other's aspect membership and records the absorbed name in `merged_from`.
    Returns (aspects, markets, vectors) aligned with the surviving markets."""
    if len(markets) < 2 or vectors is None or len(vectors) != len(markets):
        return aspects, markets, [list(v) for v in (vectors or [])]

    arr   = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1)
    norms[norms == 0] = 1.0
    unit  = arr / norms[:, None]

    # Survivor preference: better rank, then Defense, then earlier. Defense
    # narratives hold the only DoD framing in the profile, so at equal rank they
    # must never be the absorbed side of a merge.
    order = sorted(
        range(len(markets)),
        key=lambda i: (market_tier_rank(markets[i]),
                       markets[i]['market'] != DEFENSE_MARKET, i),
    )
    absorbed: dict[str, str] = {}
    keep_idx: list[int] = []
    for i in order:
        if markets[i]['market'] in absorbed:
            continue
        keep_idx.append(i)
        for j in order:
            if j == i or j in keep_idx or markets[j]['market'] in absorbed:
                continue
            if float(unit[i] @ unit[j]) >= threshold:
                absorbed[markets[j]['market']] = markets[i]['market']
                markets[i]['merged_from'] = ', '.join(
                    x for x in (markets[i]['merged_from'], markets[j]['market']) if x
                )

    if not absorbed:
        return aspects, markets, [list(v) for v in vectors]

    for aspect in aspects:
        aspect['markets'] = _name_list(
            [absorbed.get(n, n) for n in (aspect.get('markets') or [])]
        )
    keep_idx.sort()
    kept_markets = [markets[i] for i in keep_idx]
    kept_vectors = [list(vectors[i]) for i in keep_idx]
    for market in kept_markets:
        market['aspect_labels'] = [
            _s(a.get('label')) for a in aspects
            if market['market'] in (a.get('markets') or [])
        ]
    return aspects, kept_markets, kept_vectors


def _clean_unexplored(item: dict) -> dict | None:
    market = canonical_market(item.get('market') or item.get('name') or item.get('label'))
    if not market:
        return None
    return {
        'market':        market,
        'tier':          market_tier_rank(item),
        'subtitle':      _s(item.get('subtitle'))[:160],
        'narrative':     _s(item.get('narrative')) or _s(item.get('description')),
        'keywords':      _s(item.get('keywords')),
        'rationale':     _s(item.get('rationale'))[:600],
        'aspect_labels': _name_list(item.get('aspects') or item.get('aspect_labels')),
    }


def _labels_in_text(aspects: list[dict], *texts) -> list[str]:
    """Aspect labels that appear verbatim in the given text.

    Observed on a real build: the model names the aspects it is combining in the
    rationale prose ("The Single-Fiber CLE Optical Platform and Real-Time Edge AI
    Inference Engine combine directly for this use case") while returning an
    empty `aspects` array. Recovering them from the prose matters because the
    linked aspects are the ONLY evidence the Bulk Aspect Match re-ranker judges
    an unexplored market against - with none, it is asked whether a company
    could extend into a topic without being told what the company can do, and
    correctly scores it low."""
    blob = ' '.join(_s(t) for t in texts).lower()
    if not blob:
        return []
    out = []
    for aspect in aspects:
        label = _s(aspect.get('label'))
        if label and label.lower() in blob:
            out.append(label)
    return out


def normalize_unexplored(
    confirmed: list[dict],
    unexplored: list[dict],
    max_unexplored: int = MAX_UNEXPLORED,
    aspects: list[dict] | None = None,
) -> list[dict]:
    """Canonicalise unexplored market names, drop the ones that are not actually
    unexplored, cap the count and dense-rank what survives.

    Dropped: anything canonicalising to 'Other' (too vague to be worth a vector);
    anything the company already serves, which is a contradiction in terms and
    would return the same topics its confirmed market already does; anything
    without a narrative, since the narrative vector is the ONLY thing an
    unexplored market is scored on; and repeats of one category.

    `aspect_labels` is filtered against the real aspect labels so a hallucinated
    label can never reach the re-rank prompt as evidence. MIN_LINKED_ASPECTS is
    asked for in the prompt but deliberately NOT enforced here - a market whose
    labels simply failed to string-match would otherwise be deleted along with
    its narrative, the same trap normalize_markets documents for Defense."""
    served = {_s(m.get('market')) for m in (confirmed or [])}
    known  = {_s(a.get('label')).lower(): _s(a.get('label')) for a in (aspects or [])}

    kept: list[dict] = []
    seen: set[str] = set()
    for item in unexplored or []:
        if not isinstance(item, dict):
            continue
        cleaned = _clean_unexplored(item)
        if cleaned is None or cleaned['market'] == 'Other':
            continue
        if cleaned['market'] in served or cleaned['market'] in seen:
            continue
        if not cleaned['narrative']:
            continue
        if known:
            cleaned['aspect_labels'] = [
                known[l.lower()] for l in cleaned['aspect_labels'] if l.lower() in known
            ]
            if not cleaned['aspect_labels']:
                # Nothing matched (empty array, or labels spelled differently) -
                # fall back to the labels the prose names.
                cleaned['aspect_labels'] = _labels_in_text(
                    aspects, cleaned['narrative'], cleaned['rationale']
                )
        seen.add(cleaned['market'])
        kept.append(cleaned)

    kept.sort(key=lambda m: m['tier'])
    cap = max(0, min(MAX_UNEXPLORED, int(max_unexplored)))
    return _renumber_tiers(kept[:cap])


def parse_aspect_response(raw: str) -> dict:
    """Returns {profile_summary, aspects, markets, dod_assessment}. Raises
    ValueError if the response is not parseable JSON or contains no usable
    aspect. Market names and membership are normalised here, so the result is
    safe to embed and store as-is."""
    obj     = _extract_json(raw)
    summary = _s(obj.get('profile_summary'))

    aspects: list[dict] = []
    seen_labels: set[str] = set()
    for item in obj.get('aspects') or []:
        if not isinstance(item, dict):
            continue
        text  = _s(item.get('text'))
        label = _s(item.get('label')) or text[:60]
        if not text or not label:
            continue
        if label.lower() in seen_labels:
            continue
        seen_labels.add(label.lower())
        kind = _s(item.get('kind')).lower()
        aspects.append({
            'label':    label[:80],
            'kind':     kind if kind in ASPECT_KINDS else 'capability',
            'text':     text,
            'keywords': _s(item.get('keywords')),
            'evidence': _s(item.get('evidence')),
            'markets':  _name_list(item.get('markets')),
        })

    if not aspects:
        raise ValueError('response contained no usable aspects')
    aspects = aspects[:MAX_ASPECTS]

    aspects, markets = normalize_markets(
        aspects, [m for m in (obj.get('markets') or []) if isinstance(m, dict)]
    )
    return {
        'profile_summary': summary,
        'aspects':         aspects,
        'markets':         markets,
        'dod_assessment':  _s(obj.get('dod_assessment'))[:500],
    }


# -- Unexplored-market prompt (pass 2) --------------------------------------
# A SECOND Claude call, deliberately not folded into the first. The aspect
# prompt's whole discipline is "ground everything, never invent"; asking that
# same call to also speculate about markets the company is not in would
# contradict its central rule and contaminate the grounded output. This call
# gets an explicitly speculative licence instead, and its output is stored in
# its own columns so it can never be read back as confirmed capability.
#
# It reads pass 1's structured output rather than the raw material again: the
# question is what the aspects COMBINE into, which is reasoning over the aspect
# set, not another pass over the website.

_UNEXPLORED_RULES = """You are a technical analyst at a firm that writes federal grant proposals for its clients. You are given a client's capability profile: a set of aspects (each an independently searchable capability, technology, product or domain it demonstrably has today) and the markets it already sells into.

Your task is the opposite of building that profile. Identify up to {max_unexplored} markets the company does NOT currently serve but plausibly could, by combining capabilities it already has.

Rules:
- Work only from the aspects listed under ASPECTS. You may combine two or more of them into a new application, but you may never invent a capability, technology, product, customer, partner or certification that is not in that list.
- Every unexplored market must draw on at least {min_linked} of the listed aspects, named exactly. A market resting on a single aspect is usually just that aspect's existing market described again - do not return it.
- "market" MUST be copied exactly from this list, choosing the closest fit:
{categories}
- Never return a market listed under MARKETS ALREADY SERVED, and never return "Other".
- "tier": an integer rank - 1 for the most promising unexplored market, then 2, 3. No two may share a rank. Judge promise by how little the company would have to add, not by how big the market is.
- "subtitle": at most 12 words naming what this company specifically would offer that market.
- "narrative": 40-90 words describing that offering the way a solicitation describes a needed capability, drawing explicitly on the named aspects. Write what the company WOULD offer - never assert that it already serves this market.
- "keywords": 5-12 comma-separated technical terms a solicitation in this market would use.
- "rationale": one or two sentences naming which aspects combine, and what the company would still have to build, qualify or certify to compete there. Be concrete about the gap - this is what a human reads to decide whether to chase it.
- "aspects": the exact labels of the existing aspects this market draws on.
- Anything under STATED INTENTIONS is what the client has SAID it is considering. It is unverified and is never evidence of capability, but a market the client is already thinking about is a strong candidate - prefer it when the listed aspects genuinely support it.
- Assess a Defense market here too when the company does not already serve one: a dual-use application, a component that could go into a defense platform, or a capability a DARPA / AFRL / ONR / DIU / service SBIR programme funds. State the concrete use case in the narrative without inventing programmes, contracts or customers.
- If the aspects genuinely do not support any market beyond the ones already served, return an empty "unexplored_markets" array. Never pad with speculation you cannot tie to a named aspect.
- Return ONLY a valid JSON object. No preamble, no markdown, no code fences."""

_UNEXPLORED_SHAPE = """JSON shape:
{
  "unexplored_markets": [
    {
      "market": "<exact name from the allowed market list>",
      "tier": <integer rank, 1 = most promising>,
      "subtitle": "<<=12 words on what this company would offer here>",
      "narrative": "<40-90 words, capability-style, on what it would offer>",
      "keywords": "<comma-separated terms a solicitation here would use>",
      "rationale": "<which aspects combine, and the gap that remains>",
      "aspects": ["<exact aspect label>", "..."]
    }
  ]
}"""


def build_unexplored_system(max_unexplored: int = MAX_UNEXPLORED) -> str:
    return (
        _UNEXPLORED_RULES.format(
            max_unexplored=max(1, min(MAX_UNEXPLORED, int(max_unexplored))),
            min_linked=MIN_LINKED_ASPECTS,
            categories='\n'.join(
                f'  - {c}' for c in MARKET_CATEGORIES if c != 'Other'
            ),
        )
        + '\n\n' + _UNEXPLORED_SHAPE
    )


def build_unexplored_user_message(
    company: dict,
    profile_summary: str,
    aspects: list[dict],
    markets: list[dict],
    intentions: list[str] | None = None,
) -> str:
    """company keys: company_name, website, state (all optional)."""
    parts = [
        'COMPANY\n'
        f"Name: {_s(company.get('company_name')) or 'Unknown'}\n"
        f"Website: {_s(company.get('website')) or 'Unknown'}\n"
        f"State: {_s(company.get('state')) or 'Unknown'}"
    ]
    if _s(profile_summary):
        parts.append('PROFILE SUMMARY\n' + _s(profile_summary))

    aspect_lines = []
    for i, aspect in enumerate(aspects or [], start=1):
        block = f"{i}. {_s(aspect.get('label'))}"
        kind = _s(aspect.get('kind'))
        if kind:
            block += f' ({kind})'
        if _s(aspect.get('text')):
            block += '\n   ' + _s(aspect.get('text'))
        if _s(aspect.get('keywords')):
            block += '\n   Keywords: ' + _s(aspect.get('keywords'))
        aspect_lines.append(block)
    parts.append(
        'ASPECTS (capabilities this company demonstrably has today)\n'
        + ('\n'.join(aspect_lines) or '(none)')
    )

    served = [
        f'- {market_label(m)}'
        + (f" - {_s(m.get('subtitle'))}" if _s(m.get('subtitle')) else '')
        for m in (markets or [])
    ]
    parts.append(
        'MARKETS ALREADY SERVED (do not return any of these)\n'
        + ('\n'.join(served) or '(none recorded)')
    )

    if intentions:
        parts.append(
            'STATED INTENTIONS (unverified - the company said these; they are '
            'leads, not capability)\n'
            + '\n'.join(f'- {_s(i)}' for i in intentions if _s(i))
        )
    return '\n\n'.join(parts)


def parse_unexplored_response(
    raw: str,
    confirmed: list[dict],
    aspects: list[dict] | None = None,
    max_unexplored: int = MAX_UNEXPLORED,
) -> list[dict]:
    """Normalised unexplored markets, safe to embed and store as-is.

    An empty list is a legitimate answer - a company whose aspects support no
    market beyond the ones it already serves is the expected outcome for a
    single-product client. Only unparseable JSON raises."""
    obj   = _extract_json(raw)
    items = obj.get('unexplored_markets')
    if items is None:
        items = obj.get('markets')
    return normalize_unexplored(
        confirmed, list(items or []), max_unexplored, aspects
    )


def aspect_embed_text(aspect: dict) -> str:
    """Text actually embedded for an aspect — label + description + keywords."""
    parts = [_s(aspect.get('label')), _s(aspect.get('text'))]
    kw = _s(aspect.get('keywords'))
    if kw:
        parts.append(f'Keywords: {kw}')
    return '\n'.join(p for p in parts if p)


def market_embed_text(market: dict) -> str:
    """Text embedded for a market, scored as an extra vector inside its own
    market at match time. For Defense this narrative is the only place the DoD
    use case exists, so no aspect vector can stand in for it."""
    head     = _s(market.get('market'))
    subtitle = _s(market.get('subtitle'))
    parts    = [f'{head} — {subtitle}' if head and subtitle else (head or subtitle),
                _s(market.get('narrative'))]
    kw = _s(market.get('keywords'))
    if kw:
        parts.append(f'Keywords: {kw}')
    return '\n'.join(p for p in parts if p)


def market_label(market: dict) -> str:
    """'Defense (1st)' — display form used in labels and pickers."""
    name = _s(market.get('market')) or '—'
    return f'{name} ({tier_ordinal(market_tier_rank(market))})'


# ── Embedding pack / unpack ─────────────────────────────────────────────────

def pack_embeddings(vectors) -> tuple[list[float], int, int]:
    """(flat float64 list, n_aspects, embedding_dim) for parquet storage."""
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError('expected a non-empty 2-D array of aspect vectors')
    return arr.reshape(-1).tolist(), int(arr.shape[0]), int(arr.shape[1])


def unpack_embeddings(row, dtype=np.float32, col: str = 'aspect_embeddings') -> np.ndarray:
    """(n, embedding_dim) matrix for one of a profile row's flat vector
    columns. Returns an empty (0, dim) array when the row carries none."""
    try:
        dim = int(row.get('embedding_dim') or EMBED_DIM)
    except (TypeError, ValueError):
        dim = EMBED_DIM
    dim = dim if dim > 0 else EMBED_DIM

    flat = row.get(col)
    if flat is None:
        return np.zeros((0, dim), dtype=dtype)
    arr = np.asarray(list(flat), dtype=dtype)
    if arr.size < dim or arr.size % dim:
        return np.zeros((0, dim), dtype=dtype)
    return arr.reshape(-1, dim)


def _json_list(row, col: str) -> list[dict]:
    raw = row.get(col)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or '[]')
        except Exception:
            return []
    if isinstance(raw, np.ndarray):
        raw = list(raw)
    return [a for a in (raw or []) if isinstance(a, dict)]


def profile_aspects(row) -> list[dict]:
    """Aspect dicts for a profile row (the `aspects` column is a JSON string)."""
    return _json_list(row, 'aspects')


def profile_markets(row) -> list[dict]:
    """Market dicts for a profile row. Empty for profiles built before markets
    existed — those are flagged for rebuild in the Client Profiles view."""
    return _json_list(row, 'markets')


def unpack_market_embeddings(row, dtype=np.float32) -> np.ndarray:
    return unpack_embeddings(row, dtype=dtype, col='market_embeddings')


def profile_unexplored(row) -> list[dict]:
    """Unexplored-market dicts for a profile row - markets the company does NOT
    serve, inferred by linking its aspects. Empty for every profile built before
    the unexplored pass existed, and for companies whose aspects support none."""
    return _json_list(row, 'unexplored_markets')


def unpack_unexplored_embeddings(row, dtype=np.float32) -> np.ndarray:
    return unpack_embeddings(row, dtype=dtype, col='unexplored_embeddings')


def unexplored_aspect_indices(aspects: list[dict], market: dict) -> list[int]:
    """Positions of the aspects an unexplored market says it would draw on.

    Unlike a confirmed market these are NOT scored - an unexplored market is
    scored on its narrative vector alone, because the confirmed aspect vectors
    would just return the topics the confirmed markets already returned, and
    _rerank_groups would then share one score between two different framings of
    the same pair. They are looked up only to give the re-ranker the evidence of
    what the company can actually do."""
    # normalize_unexplored() backfills aspect_labels from the narrative and
    # rationale prose when the model returned none, so this is rarely empty.
    wanted = {_s(l).lower() for l in (market.get('aspect_labels') or [])}
    return [
        i for i, a in enumerate(aspects)
        if _s(a.get('label')).lower() in wanted
    ]


def market_aspect_indices(aspects: list[dict], market_name: str) -> list[int]:
    """Positions of the aspects earmarked to a market — the aspect subset a
    market-scoped match run scores with."""
    return [
        i for i, a in enumerate(aspects)
        if market_name in list(a.get('markets') or [])
    ]


# ── Profile records & store ─────────────────────────────────────────────────

def build_profile_record(
    *,
    company_key: str,
    company_name: str,
    website: str,
    profile_summary: str,
    aspects: list[dict],
    vectors,
    sources_used: list[str],
    fingerprint: str,
    model: str,
    markets: list[dict] | None = None,
    market_vectors=None,
    unexplored: list[dict] | None = None,
    unexplored_vectors=None,
    dod_assessment: str = '',
    built_at: str | None = None,
) -> dict:
    flat, n_aspects, dim = pack_embeddings(vectors)
    markets = list(markets or [])
    if markets and market_vectors is not None and len(market_vectors) == len(markets):
        market_flat, n_markets, _ = pack_embeddings(market_vectors)
    else:
        # A profile without vectors for its markets can't be matched per market
        # and would read as "has markets" in the UI - store none at all.
        markets, market_flat, n_markets = [], [], 0
    # Same rule for the unexplored block: its narrative vector is the ONLY thing
    # it is ever scored on, so an unexplored market without one is unmatchable.
    unexplored = list(unexplored or [])
    if (unexplored and unexplored_vectors is not None
            and len(unexplored_vectors) == len(unexplored)):
        unexp_flat, n_unexplored, _ = pack_embeddings(unexplored_vectors)
    else:
        unexplored, unexp_flat, n_unexplored = [], [], 0
    return {
        'company_key':        company_key,
        'company_name':       company_name,
        'companyWebsite':     website,
        'profile_summary':    profile_summary,
        'aspects':            json.dumps(aspects, ensure_ascii=False),
        'aspect_labels':      ' | '.join(_s(a.get('label')) for a in aspects),
        'n_aspects':          n_aspects,
        'embedding_dim':      dim,
        'aspect_embeddings':  flat,
        'markets':            json.dumps(markets, ensure_ascii=False),
        'market_labels':      ' | '.join(market_label(m) for m in markets),
        'n_markets':          n_markets,
        'market_embeddings':  market_flat,
        'unexplored_markets':    json.dumps(unexplored, ensure_ascii=False),
        'unexplored_labels':     ' | '.join(market_label(m) for m in unexplored),
        'n_unexplored':          n_unexplored,
        'unexplored_embeddings': unexp_flat,
        'dod_assessment':     dod_assessment,
        'sources_used':       ','.join(sources_used),
        'source_fingerprint': fingerprint,
        'model':              model,
        'built_at':           built_at or date.today().isoformat(),
    }


def empty_profiles_df() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype='object') for c in PROFILE_COLUMNS})


def load_profiles(gcs_client, bucket: str = BUCKET) -> pd.DataFrame:
    blob = gcs_client.bucket(bucket).blob(PROFILES_BLOB)
    if not blob.exists():
        return empty_profiles_df()
    df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
    for col in PROFILE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df.reset_index(drop=True)


def save_profiles(gcs_client, df: pd.DataFrame, bucket: str = BUCKET) -> None:
    out = df.copy()
    for col in PROFILE_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out = out[PROFILE_COLUMNS].reset_index(drop=True)
    # Written as plain ints so they survive the round-trip as ints, not floats
    for col in ('n_aspects', 'embedding_dim', 'n_markets', 'n_unexplored'):
        out[col] = out[col].fillna(0).astype('int64')
    BucketManager(bucket, client=gcs_client).upload_file(PROFILES_BLOB, out)


def upsert_profiles(existing: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    """Replace any existing rows for the records' company keys, append the rest."""
    if not records:
        return existing
    keys = {r['company_key'] for r in records}
    kept = (
        existing[~existing['company_key'].isin(keys)]
        if not existing.empty and 'company_key' in existing.columns
        else empty_profiles_df()
    )
    merged = pd.concat([kept, pd.DataFrame(records)], ignore_index=True)
    return merged.sort_values(
        'company_name', key=lambda s: s.fillna('').astype(str).str.lower()
    ).reset_index(drop=True)


def delete_profile(existing: pd.DataFrame, key: str) -> pd.DataFrame:
    if existing.empty or 'company_key' not in existing.columns:
        return existing
    return existing[existing['company_key'] != key].reset_index(drop=True)
