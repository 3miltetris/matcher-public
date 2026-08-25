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

PROFILE_COLUMNS = [
    'company_key', 'company_name', 'companyWebsite',
    'profile_summary', 'aspects', 'aspect_labels',
    'n_aspects', 'embedding_dim', 'aspect_embeddings',
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

_RULES = """You are a technical analyst at a firm that writes federal grant proposals for its clients. You build multi-aspect capability profiles: a client's material is split into a few distinct, independently searchable aspects, each embedded separately and scored against federal R&D grant topics (SBIR/STTR, BAAs, agency solicitations).

Split the company described in the material below into {lo}-{hi} aspects (aim for about {target}).

Rules:
- Ground every aspect in the supplied material. Never invent technologies, products, customers, partners, or certifications.
- Each aspect is ONE independently searchable thing: a core technology, a technical capability, a product or platform line, a scientific domain, or an application/market area. No two aspects may restate each other.
- Write "text" the way a solicitation describes a needed capability: technical, concrete, 40-90 words. No marketing language, no boilerplate about the company being innovative or a leader.
- Financial material (revenue, headcount, funding history) is background only — never make an aspect about finances.
- "keywords": 5-12 comma-separated technical terms a solicitation would use for this aspect.
- "kind": exactly one of technology, capability, product, domain, market.
- "label": at most 6 words, distinct from every other label.
- "evidence": which of the supplied sources support this aspect.
- If the material only supports fewer aspects than the range, return fewer — never pad with speculation.
- Return ONLY a valid JSON object. No preamble, no markdown, no code fences."""

_SHAPE = """JSON shape:
{
  "profile_summary": "<2-4 sentences on what this company actually does>",
  "aspects": [
    {
      "label": "<<=6 words>",
      "kind": "<technology|capability|product|domain|market>",
      "text": "<40-90 words, capability-style description>",
      "keywords": "<comma-separated technical terms>",
      "evidence": "<which sources support this>"
    }
  ]
}"""


def build_aspect_system(target_aspects: int) -> str:
    target = max(MIN_ASPECTS, min(MAX_ASPECTS, int(target_aspects)))
    lo     = max(MIN_ASPECTS, target - 1)
    hi     = min(MAX_ASPECTS, target + 2)
    return (
        _RULES.format(lo=lo, hi=hi, target=target) + '\n\n' + _SHAPE
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


def parse_aspect_response(raw: str) -> tuple[str, list[dict]]:
    """Returns (profile_summary, aspects). Raises ValueError if the response
    is not parseable JSON or contains no usable aspect."""
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
        })

    if not aspects:
        raise ValueError('response contained no usable aspects')
    return summary, aspects[:MAX_ASPECTS]


def aspect_embed_text(aspect: dict) -> str:
    """Text actually embedded for an aspect — label + description + keywords."""
    parts = [_s(aspect.get('label')), _s(aspect.get('text'))]
    kw = _s(aspect.get('keywords'))
    if kw:
        parts.append(f'Keywords: {kw}')
    return '\n'.join(p for p in parts if p)


# ── Embedding pack / unpack ─────────────────────────────────────────────────

def pack_embeddings(vectors) -> tuple[list[float], int, int]:
    """(flat float64 list, n_aspects, embedding_dim) for parquet storage."""
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError('expected a non-empty 2-D array of aspect vectors')
    return arr.reshape(-1).tolist(), int(arr.shape[0]), int(arr.shape[1])


def unpack_embeddings(row, dtype=np.float32) -> np.ndarray:
    """(n_aspects, embedding_dim) matrix for a profile row. Returns an empty
    (0, dim) array when the row carries no usable vectors."""
    try:
        dim = int(row.get('embedding_dim') or EMBED_DIM)
    except (TypeError, ValueError):
        dim = EMBED_DIM
    dim = dim if dim > 0 else EMBED_DIM

    flat = row.get('aspect_embeddings')
    if flat is None:
        return np.zeros((0, dim), dtype=dtype)
    arr = np.asarray(list(flat), dtype=dtype)
    if arr.size < dim or arr.size % dim:
        return np.zeros((0, dim), dtype=dtype)
    return arr.reshape(-1, dim)


def profile_aspects(row) -> list[dict]:
    """Aspect dicts for a profile row (the `aspects` column is a JSON string)."""
    raw = row.get('aspects')
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or '[]')
        except Exception:
            return []
    if isinstance(raw, np.ndarray):
        raw = list(raw)
    return [a for a in (raw or []) if isinstance(a, dict)]


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
    built_at: str | None = None,
) -> dict:
    flat, n_aspects, dim = pack_embeddings(vectors)
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
    for col in ('n_aspects', 'embedding_dim'):
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
