"""
sbir_ceilings.py — the verified SBIR/STTR Phase I/II award-ceiling reference table
and the single layered resolver every topic writer calls.

SBIR/STTR Phase I and Phase II award ceilings are overwhelmingly a *program/agency*
attribute, not a figure stated in each individual notice — every Army/Navy/AF
Phase I in a given DoD BAA cycle shares roughly one ceiling, NIH has its own
guideline, etc. So a small, committed, dated per-agency table is a more trustworthy
source for a "verified" ceiling than scraping each solicitation. Git history is the
audit trail: who changed a cap, when, and against which source URL.

"Verified" here means **sourced + dated reference cap**, NOT "confirmed for this
specific notice". A per-notice figure (Grants.gov's structured `awardCeiling`, or a
dollar amount actually stated in the solicitation text) is more specific and wins
over the table when present — see `resolve_ceilings`.

┌─────────────────────────────────────────────────────────────────────────────┐
│ ⚠️  THE DOLLAR FIGURES BELOW ARE FY2025-ERA PLACEHOLDERS.                      │
│     BW&CO must author/confirm the real per-agency Phase I/II ceilings and the  │
│     source_url + effective date for each row before this table is treated as   │
│     authoritative in anything client-facing. The structure is final; the       │
│     numbers are a scaffold.                                                     │
└─────────────────────────────────────────────────────────────────────────────┘

Streamlit-free and dependency-light (stdlib + the number parser in
finance_research). Imported by the Grants.gov / SAM.gov / Topic Importer writers
and the one-off backfill job; must never import Streamlit or pandas.
"""

from __future__ import annotations

import json

import src.modules.finance_research as fr

BUCKET = 'cc-matcher-bucket-jeg-v1'
OVERRIDE_BLOB = 'sbir-ceilings/overrides.json'

PROGRAMS = ('SBIR', 'STTR')
DEFAULT_PROGRAM = 'SBIR'

# SBA statutory guideline caps (the "hard" guideline agencies are not to exceed by
# more than ~50% without an approved waiver). Used when the agency is unknown.
# PLACEHOLDER — confirm against https://www.sbir.gov/about and the annual SBA
# program-year adjustment before relying on it.
_FALLBACK = {
    'phase1':    314_363,
    'phase2':    2_095_748,
    'source':    'https://www.sbir.gov/about',
    'effective': '2025-01-01',
    'basis':     'sba_statutory',
}

# Per-agency typical ceilings, keyed on (canonical_agency, program_type).
# `basis` is the fine-grained provenance label written to `ceiling_basis`.
# PLACEHOLDER figures — see the warning banner above.
CEILINGS: dict[tuple[str, str], dict] = {
    ('DOD',  'SBIR'): {'phase1':  250_000, 'phase2': 1_800_000,
                       'source': 'https://www.dodsbirsttr.mil/', 'effective': '2025-01-01', 'basis': 'dod_sbir_table'},
    ('DOD',  'STTR'): {'phase1':  250_000, 'phase2': 1_800_000,
                       'source': 'https://www.dodsbirsttr.mil/', 'effective': '2025-01-01', 'basis': 'dod_sttr_table'},
    ('HHS',  'SBIR'): {'phase1':  314_363, 'phase2': 2_095_748,
                       'source': 'https://seed.nih.gov/', 'effective': '2025-01-01', 'basis': 'nih_sbir_table'},
    ('HHS',  'STTR'): {'phase1':  314_363, 'phase2': 2_095_748,
                       'source': 'https://seed.nih.gov/', 'effective': '2025-01-01', 'basis': 'nih_sttr_table'},
    ('NSF',  'SBIR'): {'phase1':  305_000, 'phase2': 1_000_000,
                       'source': 'https://seedfund.nsf.gov/', 'effective': '2025-01-01', 'basis': 'nsf_sbir_table'},
    ('NSF',  'STTR'): {'phase1':  305_000, 'phase2': 1_000_000,
                       'source': 'https://seedfund.nsf.gov/', 'effective': '2025-01-01', 'basis': 'nsf_sttr_table'},
    ('DOE',  'SBIR'): {'phase1':  200_000, 'phase2': 1_700_000,
                       'source': 'https://science.osti.gov/sbir', 'effective': '2025-01-01', 'basis': 'doe_sbir_table'},
    ('DOE',  'STTR'): {'phase1':  200_000, 'phase2': 1_700_000,
                       'source': 'https://science.osti.gov/sbir', 'effective': '2025-01-01', 'basis': 'doe_sttr_table'},
    ('NASA', 'SBIR'): {'phase1':  150_000, 'phase2':   850_000,
                       'source': 'https://sbir.nasa.gov/', 'effective': '2025-01-01', 'basis': 'nasa_sbir_table'},
    ('NASA', 'STTR'): {'phase1':  150_000, 'phase2':   850_000,
                       'source': 'https://sbir.nasa.gov/', 'effective': '2025-01-01', 'basis': 'nasa_sttr_table'},
    ('USDA', 'SBIR'): {'phase1':  100_000, 'phase2':   650_000,
                       'source': 'https://www.nifa.usda.gov/grants/programs/small-business-innovation-research-program-sbir',
                       'effective': '2025-01-01', 'basis': 'usda_sbir_table'},
    ('DHS',  'SBIR'): {'phase1':  150_000, 'phase2': 1_000_000,
                       'source': 'https://www.dhs.gov/science-and-technology/sbir', 'effective': '2025-01-01', 'basis': 'dhs_sbir_table'},
    ('DOC',  'SBIR'): {'phase1':  175_000, 'phase2':   500_000,
                       'source': 'https://techpartnerships.noaa.gov/sbir/', 'effective': '2025-01-01', 'basis': 'doc_sbir_table'},
    ('DOT',  'SBIR'): {'phase1':  150_000, 'phase2': 1_000_000,
                       'source': 'https://www.transportation.gov/osdbu/SBIR', 'effective': '2025-01-01', 'basis': 'dot_sbir_table'},
    ('ED',   'SBIR'): {'phase1':  250_000, 'phase2': 1_000_000,
                       'source': 'https://ies.ed.gov/sbir/', 'effective': '2025-01-01', 'basis': 'ed_sbir_table'},
    ('EPA',  'SBIR'): {'phase1':  100_000, 'phase2':   400_000,
                       'source': 'https://www.epa.gov/sbir', 'effective': '2025-01-01', 'basis': 'epa_sbir_table'},
}

# Substring → canonical agency. Checked longest-first so "AIR FORCE" wins before a
# bare "FORCE" etc. Agencies that run SBIR/STTR; DARPA/components fold into DOD, and
# NOAA/NIST fold into DOC, because they share the parent department's ceilings.
_AGENCY_ALIASES: dict[str, str] = {
    'DEPARTMENT OF DEFENSE': 'DOD', 'DEFENSE': 'DOD', 'DOD': 'DOD', 'DARPA': 'DOD',
    'ARMY': 'DOD', 'NAVY': 'DOD', 'AIR FORCE': 'DOD', 'SPACE FORCE': 'DOD',
    'SOCOM': 'DOD', 'DEVCOM': 'DOD', 'MDA': 'DOD', 'DHA': 'DOD', 'DLA': 'DOD',
    'HEALTH AND HUMAN SERVICES': 'HHS', 'HHS': 'HHS', 'NIH': 'HHS',
    'NATIONAL INSTITUTES OF HEALTH': 'HHS', 'CDC': 'HHS', 'FDA': 'HHS',
    'BARDA': 'HHS', 'NCI': 'HHS', 'NIAID': 'HHS',
    'NATIONAL SCIENCE FOUNDATION': 'NSF', 'NSF': 'NSF',
    'DEPARTMENT OF ENERGY': 'DOE', 'ENERGY': 'DOE', 'DOE': 'DOE', 'ARPA-E': 'DOE',
    'NASA': 'NASA', 'NATIONAL AERONAUTICS': 'NASA',
    'AGRICULTURE': 'USDA', 'USDA': 'USDA', 'NIFA': 'USDA',
    'HOMELAND SECURITY': 'DHS', 'DHS': 'DHS',
    'COMMERCE': 'DOC', 'DOC': 'DOC', 'NOAA': 'DOC', 'NIST': 'DOC',
    'TRANSPORTATION': 'DOT', 'DOT': 'DOT',
    'EDUCATION': 'ED', 'INSTITUTE OF EDUCATION SCIENCES': 'ED',
    'ENVIRONMENTAL PROTECTION': 'EPA', 'EPA': 'EPA',
}
_ALIAS_ORDER = sorted(_AGENCY_ALIASES, key=len, reverse=True)

# Runtime overrides merged over CEILINGS (the GCS hot-patch seam). Empty by default.
_OVERRIDES: dict[tuple[str, str], dict] = {}


def detect_program(text='', explicit=None) -> str | None:
    """Which SBIR/STTR program a topic is, or None when it is not an SBIR/STTR
    opportunity at all.

    The phase-ceiling table is SBIR/STTR-specific — stamping its figures onto an
    ordinary research grant or a supply-buy contract would be wrong. So the table
    only applies when the topic is identifiably SBIR/STTR: an explicit SBIR/STTR
    `explicit` arg wins, else the program is sniffed from `text` (title + summary +
    funding instrument), else None (no phase ceilings).
    """
    if explicit in PROGRAMS:
        return explicit
    low = str(text or '').lower()
    if 'sttr' in low:
        return 'STTR'
    if 'sbir' in low:
        return 'SBIR'
    return None


def canonical_agency(broad_agency, agency: str = '') -> str | None:
    """Map a folder code and/or an agency string to a canonical SBIR agency key.

    The folder `broad_agency` is authoritative for single-agency folders (DOD, HHS,
    …). For the multi-agency ingest folders (`GRANTS-GOV`, `SAM-GOV`, `CONSORTIUM`,
    `STATE`) there is no single agency, so the row's own `agency`/`agency_code`
    string decides. Returns None when nothing matches (→ SBA fallback).
    """
    folder = str(broad_agency or '').strip().upper()
    _MULTI = {'GRANTS-GOV', 'SAM-GOV', 'CONSORTIUM', 'STATE', 'CUSTOM', ''}
    candidates = []
    if folder not in _MULTI:
        candidates.append(folder)
    candidates.append(str(agency or '').strip().upper())
    for cand in candidates:
        if not cand:
            continue
        if cand in _AGENCY_ALIASES:          # exact code, e.g. "DOD"
            return _AGENCY_ALIASES[cand]
        for alias in _ALIAS_ORDER:           # substring, e.g. "...DEPT OF THE ARMY..."
            if alias in cand:
                return _AGENCY_ALIASES[alias]
    return None


def resolve_agency_row(broad_agency, agency: str = '', program_type: str = DEFAULT_PROGRAM) -> dict:
    """The table row for an agency+program, falling back to the SBA guideline.

    Overrides win over the committed table. The returned dict always carries
    phase1/phase2/source/effective/basis.
    """
    prog = program_type if program_type in PROGRAMS else DEFAULT_PROGRAM
    key = canonical_agency(broad_agency, agency)
    if key is not None:
        for table in (_OVERRIDES, CEILINGS):
            if (key, prog) in table:
                return table[(key, prog)]
            # STTR with no dedicated row falls back to the agency's SBIR row.
            if prog == 'STTR' and (key, 'SBIR') in table:
                return table[(key, 'SBIR')]
    return _FALLBACK


# Below this, a parsed "award ceiling" is almost certainly a parse artifact rather
# than a real figure — a stray percentage ("50% of project fee..." -> 50), a count,
# or a Grants.gov 0. No federal/state R&D award ceiling in this corpus is < $1,000,
# so a sub-floor value is dropped to "no figure" (better than showing "$50").
_MIN_AWARD = 1_000.0


def _positive(value, minimum: float = 0.0) -> float | None:
    """Parse to an upper-bound figure at or above `minimum`, or None. A 0 / blank /
    negative / sub-minimum value reads as 'not specified', never as the number."""
    n = fr.to_upper_bound(value)
    return n if (n is not None and n > 0 and n >= minimum) else None


# The six columns every writer sets (and `grant_utils.CEILING_DEFAULTS` backfills).
CEILING_COLUMNS = (
    'phase1_ceiling_num', 'phase2_ceiling_num',
    'award_ceiling_num', 'award_floor_num',
    'ceiling_source', 'ceiling_basis',
)


def resolve_ceilings(*, broad_agency='', agency='', program_text='',
                     program_type=None, funding_amount_text=None,
                     gg_synopsis=None) -> dict:
    """The single layered ceiling resolver. Returns the six CEILING_COLUMNS.

    - `phase1_ceiling_num` / `phase2_ceiling_num` are the program/agency caps from
      the reference table — populated **only when the topic is SBIR/STTR** (see
      `detect_program`); None for an ordinary grant/contract, so the table's SBIR
      figures are never stamped onto something that is not SBIR.
    - `award_ceiling_num` / `award_floor_num` are the *notice-specific* ceiling/floor
      when one is available, by layer precedence, and `ceiling_source` records which
      layer won:
        1. 'grants_gov'  — Grants.gov structured `awardCeiling`/`awardFloor`
        2. 'stated_text' — a dollar figure stated in the solicitation / funding_amount
        3. 'agency_table'— SBIR/STTR with neither of the above; only the phase caps apply
        '' (empty)       — non-SBIR with no per-notice figure: nothing to show
      `award_ceiling_num` is left None for the table-only case — the phase columns
      carry the table figure, and a consumer that wants a single number reads
      `phase1_ceiling_num`. This keeps `award_ceiling_num` meaning "this notice".

    `program_text` is any text to sniff SBIR/STTR from (title + summary + funding
    instrument). `program_type` ('SBIR'/'STTR') is an explicit override that wins.
    `gg_synopsis` is the Grants.gov detail `synopsis`/`forecast` dict.
    `funding_amount_text` is the Topic-Importer / SAM stated amount (free text).
    """
    program = detect_program(program_text, program_type)
    out = {
        'phase1_ceiling_num': None,
        'phase2_ceiling_num': None,
        'award_ceiling_num':  None,
        'award_floor_num':    None,
        'ceiling_source':     '',
        'ceiling_basis':      '',
    }

    # Phase caps from the table — SBIR/STTR, and only when a KNOWN federal SBIR
    # agency resolves. The blanket SBA-statutory fallback is deliberately NOT used
    # for phase caps: a row that merely mentions "SBIR/STTR" but whose agency does
    # not map to a federal SBIR agency is almost always a state match program, an
    # accelerator, or a passing mention (observed: "SBIR/STTR State Match",
    # "MS-FAST Phase 0", "xTech|Search 10"), and stamping the $314K/$2.1M federal
    # guideline onto it would be wrong. Exact per-notice figures still apply below.
    if program is not None:
        row = resolve_agency_row(broad_agency, agency, program)
        if row is not _FALLBACK:
            out['phase1_ceiling_num'] = float(row['phase1']) if row.get('phase1') else None
            out['phase2_ceiling_num'] = float(row['phase2']) if row.get('phase2') else None
            out['ceiling_source']     = 'agency_table'
            out['ceiling_basis']      = row.get('basis', '')

    # Layer 1 — Grants.gov structured fields (most specific, applies to any program).
    if gg_synopsis:
        ceiling = _positive(gg_synopsis.get('awardCeiling'), _MIN_AWARD)
        floor   = _positive(gg_synopsis.get('awardFloor'), _MIN_AWARD)
        if ceiling is not None or floor is not None:
            out['award_ceiling_num'] = ceiling
            out['award_floor_num']   = floor
            out['ceiling_source']    = 'grants_gov'
            out['ceiling_basis']     = 'grants_gov_synopsis'
            return out

    # Layer 2 — a dollar figure stated in the solicitation text.
    if funding_amount_text:
        stated = _positive(funding_amount_text, _MIN_AWARD)
        if stated is not None:
            out['award_ceiling_num'] = stated
            out['ceiling_source']    = 'stated_text'
            out['ceiling_basis']     = 'stated_text'
            return out

    # Layer 3 — table-only (SBIR) or nothing (non-SBIR): defaults already set.
    return out


# ── Optional GCS override hot-patch (committed table is the default) ───────────

def apply_overrides(mapping: dict) -> None:
    """Merge an override map into the live table. Keys are 'AGENCY|PROGRAM'
    (e.g. 'DOD|SBIR'); values carry phase1/phase2 and optionally source/effective/
    basis. Idempotent; last write wins."""
    for raw_key, val in (mapping or {}).items():
        agency, _, prog = str(raw_key).partition('|')
        key = (agency.strip().upper(), (prog.strip().upper() or DEFAULT_PROGRAM))
        merged = dict(val or {})
        merged.setdefault('basis', f'{key[0].lower()}_{key[1].lower()}_override')
        _OVERRIDES[key] = merged


def load_overrides(client, bucket: str = BUCKET, blob_path: str = OVERRIDE_BLOB) -> bool:
    """Read and apply `sbir-ceilings/overrides.json` if it exists. Returns True when
    an override map was applied. Never raises — a missing/garbled blob leaves the
    committed table in force. Callers (jobs) invoke this once at startup if they
    want hot-patched numbers; the committed table is the default everywhere else."""
    try:
        blob = client.bucket(bucket).blob(blob_path)
        if not blob.exists():
            return False
        apply_overrides(json.loads(blob.download_as_bytes()))
        return True
    except Exception:
        return False
