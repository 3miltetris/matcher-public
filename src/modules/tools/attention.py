"""
Attention-queue tool (Phase 5.1, Tier 1 read-only)
--------------------------------------------------
list_attention wraps the attention-queue store (src/modules/attention.py) for the
MCP server. Read-only here; resolving an item is a Tier-3 staged write that goes
through a confirm token, not this module.
"""

from __future__ import annotations

import json

from src.modules import attention as aq
from src.modules.tools import _common as cm


def list_attention(gcs_client, *, kind: str | None = None, status: str = 'open',
                   limit: int = 50, bucket: str = aq.BUCKET) -> dict:
    """Open items in the attention queue (what a scheduled job could not resolve
    on its own), optionally filtered by `kind`. Pass `status=None` for every
    status. Each item carries its parsed `detail`. Also returns per-kind open
    counts so a caller can triage without reading every row."""
    st = None if status in (None, 'all') else status
    df = aq.load(gcs_client, status=st, kind=kind, bucket=bucket)
    counts = aq.open_counts(gcs_client, bucket=bucket)

    items = []
    for r in df.head(max(1, int(limit))).to_dict('records'):
        try:
            detail = json.loads(r.get('detail') or '{}')
        except Exception:
            detail = {}
        items.append({
            'id': r.get('id', ''),
            'kind': r.get('kind', ''),
            'subject_key': r.get('subject_key', ''),
            'subject_label': cm.cap_text(r.get('subject_label'), 200),
            'source_job': r.get('source_job', ''),
            'run_id': r.get('run_id', ''),
            'status': r.get('status', ''),
            'created_at': r.get('created_at', ''),
            'detail': detail,
        })
    return {'open_counts': counts, 'total': int(len(df)),
            'returned': len(items), 'items': items}
