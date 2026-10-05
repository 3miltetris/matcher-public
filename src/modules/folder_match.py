"""
folder_match.py — streamlit-free company-name ↔ Drive folder-name matching.

Moved out of views/drive_sync.py so the DD intake service can dedup
{Company}_INTERNAL folders with exactly the matcher Drive Sync uses to
auto-assign them. jobs/drive_sync_job.py keeps its own _norm_name (a wider
suffix set) because changing it would change which proposals match.
"""

import re
import string
from difflib import SequenceMatcher

LEGAL_SUFFIXES = {'inc', 'llc', 'corp', 'co', 'ltd', 'pllc', 'incorporated',
                  'corporation', 'company'}

FUZZY_THRESHOLD = 0.87
FUZZY_MARGIN    = 0.05

INTERNAL_RE  = re.compile(r'[\s_-]*internal\s*$', re.IGNORECASE)
_PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def normalize(name: str) -> str:
    """Folder or company name → comparable form: strip trailing _INTERNAL,
    lowercase, drop punctuation and legal suffixes, collapse whitespace."""
    text  = INTERNAL_RE.sub('', str(name or '')).lower()
    text  = text.translate(_PUNCT_TABLE)
    words = [w for w in text.split() if w not in LEGAL_SUFFIXES]
    return ' '.join(words)


def match_folder(norm_folder: str, client_norms: dict[str, str]
                 ) -> tuple[str | None, str, float]:
    """Match a normalized folder name against {client_key: normalized_name}.
    Returns (client_key | None, tier, score) — tier in exact|contains|fuzzy|none."""
    if not norm_folder:
        return None, 'none', 0.0
    # (a) exact
    for key, norm in client_norms.items():
        if norm and norm == norm_folder:
            return key, 'exact', 1.0
    # (b) containment either direction, shorter side >= 5 chars
    for key, norm in client_norms.items():
        if not norm:
            continue
        shorter = min(norm, norm_folder, key=len)
        if len(shorter) >= 5 and (norm in norm_folder or norm_folder in norm):
            return key, 'contains', 0.99
    # (c) best ratio >= threshold and clear of runner-up
    scored = sorted(
        ((SequenceMatcher(None, norm_folder, norm).ratio(), key)
         for key, norm in client_norms.items() if norm),
        reverse=True,
    )
    if scored and scored[0][0] >= FUZZY_THRESHOLD:
        if len(scored) == 1 or scored[0][0] - scored[1][0] >= FUZZY_MARGIN:
            return scored[0][1], 'fuzzy', scored[0][0]
    return None, 'none', scored[0][0] if scored else 0.0
