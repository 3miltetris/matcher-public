"""
Bulk-match option discovery (Tier 1, read-only)
-----------------------------------------------
`list_match_options` tells the caller which **contact sources** (the company
pools/lead folders a match run scores) and which **agencies** (the grant-topic
folders) actually exist in the store right now, so a connector can offer real
choices instead of guessing codes. It is the companion of the `run_matching`
trigger: ask this first, let the person pick, then estimate/confirm the run.

Names only — deliberately NO row counts: it is two cheap prefix list calls and
downloads no parquet. Transport-agnostic, Streamlit-free, like every tool here.
"""

from __future__ import annotations

from src.modules import pools as pl
from src.modules.tools import _common as cm

BUCKET = pl.BUCKET

_CONTACTS_PREFIX = 'data/all-contacts/'
_TOPICS_PREFIX = 'data/all-topics/processed/'

# The grant date columns a run_matching date filter may point at (see
# triggers._resolve_matching, which enforces the same allowlist).
DATE_COLUMNS = ['due_date', 'scraped_at']


def list_match_options(gcs_client, *, bucket: str = BUCKET) -> dict:
    """{contact_sources, agencies, date_columns, note}. `contact_sources` is one
    entry per folder under data/all-contacts/ with `kind` ('pool' for
    clients/prospects, else 'lead_source') and `default_selected` (False for
    prospects, mirroring the Bulk Matching view's opt-in). `agencies` are the
    grant-topic folder codes under data/all-topics/processed/."""
    sources = []
    for name in cm.list_subfolders(gcs_client, _CONTACTS_PREFIX, bucket=bucket):
        is_pool = pl.is_pool(name)
        sources.append({
            'name': name,
            'kind': 'pool' if is_pool else 'lead_source',
            'label': pl.pool(name)['label'] if is_pool else name,
            'default_selected': name != pl.PROSPECTS,
        })
    agencies = cm.list_subfolders(gcs_client, _TOPICS_PREFIX, bucket=bucket)
    return {
        'contact_sources': sources,
        'agencies': agencies,
        'date_columns': DATE_COLUMNS,
        'note': ('contact_sources are the company pools/lead folders a match '
                 "scores; agencies are the grant-topic folders. 'prospects' is "
                 'available but off by default (default_selected=false). An '
                 "agency code like SAM-GOV is a grant source, not a contact "
                 'source.'),
    }
