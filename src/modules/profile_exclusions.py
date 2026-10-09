"""
profile_exclusions.py — companies we no longer work with, kept out of
capability-profile building.

These clients have no real recent Drive activity (only re-filed contracts,
copies, or nothing at all), so profiling them wastes Claude calls and clutters
the Capability Profiles build list. The Capability Profiles view
(views/client_profiler.py) pulls any company whose name matches this list out
of the buildable directory and shows it in a collapsed "excluded" section with
the reason, so the exclusion is visible rather than silent.

Curated by hand (Josiah's Drive-activity review, Oct 2026) — the source of
truth is the code constant below, matched by NORMALIZED company name
(src/modules/folder_match.py::normalize, so legal suffixes / punctuation /
case don't matter). This mirrors the SUPER_ADMINS code-constant pattern in
access_control.py: a small curated list that changes rarely and is edited here
+ redeployed, not from the UI. To exclude another company, add an EXCLUSIONS
row; to bring one back, delete its row.

Streamlit-free, stdlib-only, so it can be imported from a view or a job.
"""

from src.modules.folder_match import normalize

# (company name as it reads in our records, reason it was excluded). The name
# is only ever used normalized, so spelling of punctuation/suffixes is loose;
# a parenthetical alternate name is matched too (see _variants).
EXCLUSIONS: list[tuple[str, str]] = [
    ('AgileDD',                            '2024-09: contracts only (re-filed Apr 2025)'),
    ('AnuBio (Anu Research)',              '2025-01-03: NIH Ph1 work'),
    ('Ari Bio',                            '2025-04-01: company deck; later items re-filed'),
    ('Asylia Therapeutics',               'Only re-filed contracts/copies'),
    ('Avicenna Biosciences',              '2025-04-14: signed SOW for MJF application (reviewed by Josiah)'),
    ('Brylant Logistics',                 '2025-04-09: Illinois DCEO application (reviewed by Josiah)'),
    ('Coeptis Therapeutics',              '2024: award; later items re-filed'),
    ('Cyphra Autonomy',                   '2025-02-05: DoD submission; later re-filed'),
    ('DeepWell DTx',                      '2024 work; later re-uploads only'),
    ('Direct Kinetics Solutions',         '2025-01-28: NASA registration guide'),
    ('EMAlpha',                           '2025-04-18: signed SOW; Apr 2025 submission (reviewed by Josiah)'),
    ('Energy-Water',                      '2024-12-09: assessment doc'),
    ('Euroleader LLC',                    "No client files — BW&CO's own prior entity"),
    ('Euroleader',                        "No client files — BW&CO's own prior entity"),
    ('Guardion',                          '2024-10-09: letters of support'),
    ('Haffner Energy',                    '2024-08-29: competitor grants report'),
    ('InformedDNA',                       '2024-08-21: signed agreement'),
    ('KidsGoEurope',                      '2024-12-13: grant roadmap'),
    ('Kinsa Health',                      '2025-04-04: submission, 2 days before cutoff (reviewed by Josiah)'),
    ('Leshey Foundation',                 '2025-02-03: website updates'),
    ('Medsure Systems',                   '2024-08-21: signed contract'),
    ('NNOXX',                             '2025-04-02: services proposal'),
    ('Northern Permafrost Consulting',    '2025-04-04: NSF pitch final'),
    ('Otinus',                            '2025-04-09: portal info (reviewed by Josiah)'),
    ('Safety Arms Systems',               '2024-10-10: AFWERX P1 submission'),
    ('Tarawa Labs',                       '2024-10-04: NIH submission'),
    ('Texas A&M AgriLife Extension (TALL)', 'No client folder or files found'),
    ('Transat Ventures Lab',              '2025-02-14: services proposal'),
    ('TruDiagnostic',                     '~2024-10: STTR review'),
    ('Valinor Discovery',                 '~2025-01: draft folder'),
    ('Voltela',                           '2023-2024 materials only'),
    ('Warsaw Freedom Institute',          '2025-03-03: grant roadmap'),
    ('Woven Orthopedics',                 '2024-07: SOW and data collection'),
    ('Zaneez',                            '2024-03: finished files'),
]


def _variants(name: str) -> list[str]:
    """Normalized forms a company name should match on. For 'AnuBio (Anu
    Research)' that is the whole thing, the part before the parenthesis, and the
    part inside it — our records may hold any of those spellings."""
    out = {normalize(name)}
    if '(' in name and ')' in name:
        before = name[: name.index('(')]
        inside = name[name.index('(') + 1 : name.rindex(')')]
        out.add(normalize(before))
        out.add(normalize(inside))
    return [v for v in out if v]


# normalized variant -> reason (first spelling wins on a tie)
_INDEX: dict[str, str] = {}
for _display_name, _reason in EXCLUSIONS:
    for _v in _variants(_display_name):
        _INDEX.setdefault(_v, _reason)


def excluded_reason(company_name: str) -> str | None:
    """The reason this company is excluded from profile building, or None.

    Matches on normalized name: exact against any excluded variant, then
    containment either way with the shorter side >= 5 chars (so 'Euroleader'
    catches 'Euroleader LLC', but short tokens can't over-match). Deliberately
    no fuzzy ratio — an exclusion must never swallow a company by coincidence.
    """
    norm = normalize(company_name)
    if not norm:
        return None
    if norm in _INDEX:
        return _INDEX[norm]
    for variant, reason in _INDEX.items():
        shorter = min(variant, norm, key=len)
        if len(shorter) >= 5 and (variant in norm or norm in variant):
            return reason
    return None


def is_excluded(company_name: str) -> bool:
    return excluded_reason(company_name) is not None
