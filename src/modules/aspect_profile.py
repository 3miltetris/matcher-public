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
MAX_ASPECTS  = 8
ASPECT_KINDS = ['technology', 'capability', 'product', 'domain', 'market']
EMBED_DIM    = 1536

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
MARKET_TIERS   = ['primary', 'secondary']
MAX_MARKETS    = 4      # non-defense markets; Defense is additional
# Cosine between two market narrative vectors of the SAME company above which
# they are describing one market twice, and are merged at build time.
MARKET_MERGE_THRESHOLD = 0.93

PROFILE_COLUMNS = [
    'company_key', 'company_name', 'companyWebsite',
    'profile_summary', 'aspects', 'aspect_labels',
    'n_aspects', 'embedding_dim', 'aspect_embeddings',
    'markets', 'market_labels', 'n_markets', 'market_embeddings',
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
- "tier": "primary" for markets that are core to the business today, "secondary" for adjacent or opportunistic ones. At least one market must be primary.
- "subtitle": at most 12 words naming what this company specifically does in that market.
- "narrative": 40-90 words on what the company offers this market and which of its capabilities it draws on, written the way a solicitation describes a needed capability. Ground it in the material.
- "keywords": 5-12 comma-separated terms a solicitation in this market would use.
- "aspects": the exact labels of the aspects this market draws on.
{defense_rules}
- Return ONLY a valid JSON object. No preamble, no markdown, no code fences."""

_DEFENSE_EXTRA = (
    ', plus a Defense market when one applies — Defense does not count toward that limit'
)

_DEFENSE_RULES = """- Assess separately whether this company has a plausible defense or DoD application, and include a "Defense" market whenever the connection is real even if it is loose: a dual-use technology, a component that could go into a defense platform, or a capability an office like DARPA, AFRL, ONR, DIU or a service SBIR program funds.
- The Defense market's "narrative" must state the concrete use case — who in the department would use it and for what — without inventing programs, contracts, or customers.
- Only if there is genuinely no defense application, omit the Defense market and give the one-sentence reason in "dod_assessment". Otherwise leave "dod_assessment" empty."""

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
      "tier": "<primary|secondary>",
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


def _clean_market(item: dict) -> dict | None:
    market = canonical_market(item.get('market') or item.get('name') or item.get('label'))
    if not market:
        return None
    tier = _s(item.get('tier')).lower()
    return {
        'market':        market,
        'tier':          tier if tier in MARKET_TIERS else 'secondary',
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
        if cleaned['tier'] == 'primary':
            existing['tier'] = 'primary'
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

    # Cap: Defense always survives, then primaries, then secondaries
    ordered = (
        [m for m in defs if m['market'] == DEFENSE_MARKET]
        + [m for m in defs if m['market'] != DEFENSE_MARKET and m['tier'] == 'primary']
        + [m for m in defs if m['market'] != DEFENSE_MARKET and m['tier'] == 'secondary']
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
    fallback = next((m['market'] for m in kept if m['tier'] == 'primary'), kept[0]['market'])
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
    kept.sort(key=lambda m: (
        m['market'] != DEFENSE_MARKET, m['tier'] != 'primary', m['market']
    ))
    return aspects, kept


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

    order = sorted(
        range(len(markets)),
        key=lambda i: (markets[i]['market'] != DEFENSE_MARKET,
                       markets[i]['tier'] != 'primary', i),
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
    """'Defense (primary)' — display form used in labels and pickers."""
    name = _s(market.get('market')) or '—'
    tier = _s(market.get('tier'))
    return f'{name} ({tier})' if tier else name


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
    for col in ('n_aspects', 'embedding_dim', 'n_markets'):
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
