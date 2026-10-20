"""
Attention queue (Stage 11.5 / Phase 2)
--------------------------------------
One store, `data/attention/queue.parquet`, for anything a scheduled job could not
resolve on its own — an unmatched Drive folder, an unattributed Fathom domain, a
profile build error, a deferred client. Instead of being silently dropped (or
emailed into the void during an unattended run), it lands here as an `open` row
and surfaces in the Home page's "Needs attention" panel; the relevant existing
page resolves it.

Streamlit-free (the sync jobs write to it), client-injected like source_registry
and pools. Writes go through gcs_write.read_modify_write so two jobs appending
concurrently never clobber each other — the queue is one blob many jobs touch.
Dedup is on (kind, subject_key) among OPEN rows: a second run re-reporting the
same unresolved thing updates that row rather than piling up duplicates.
"""

import io
import json
import uuid
from datetime import datetime, timezone

import pandas as pd

from src.modules import gcs_write as gw

BUCKET = 'cc-matcher-bucket-jeg-v1'
QUEUE_BLOB = 'data/attention/queue.parquet'

# The kinds in use to start with — not enforced (a new job may add one), but the
# Home panel and the resolving pages key off these.
KINDS = (
    'drive_unmatched_folder',
    'drive_onboarding_proposal',
    'drive_extract_failed',
    'drive_deferred',
    'fathom_unmatched_domain',
    'profile_build_error',
    'profile_refusal_fallback',
)

STATUS_OPEN = 'open'
STATUS_RESOLVED = 'resolved'
STATUS_DISMISSED = 'dismissed'

COLUMNS = [
    'id', 'kind', 'source_job', 'run_id', 'subject_key', 'subject_label',
    'detail', 'created_at', 'status', 'resolved_by', 'resolved_at',
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype='object') for c in COLUMNS})


def _queue_blob(gcs_client, bucket: str):
    return gcs_client.bucket(bucket).blob(QUEUE_BLOB)


def _ensure(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty()
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ''
    return df[COLUMNS]


# ── Writing ─────────────────────────────────────────────────────────────────

def enqueue(gcs_client, items, *, bucket: str = BUCKET) -> int:
    """Upsert attention items. `items` is a list of dicts with keys `kind`,
    `subject_key`, and optionally `subject_label`, `source_job`, `run_id`,
    `detail` (a JSON-serialisable dict). An OPEN row with the same
    (kind, subject_key) is refreshed in place; otherwise a new `open` row is
    added. Returns the number of items written. No-op on an empty list."""
    items = [i for i in (items or []) if i.get('kind') and i.get('subject_key')]
    if not items:
        return 0

    def mutate(df):
        df = _ensure(df)
        rows = df.to_dict('records')
        # index open rows by (kind, subject_key) → position
        open_idx = {
            (r['kind'], r['subject_key']): n
            for n, r in enumerate(rows) if r.get('status') == STATUS_OPEN
        }
        now = _now()
        for item in items:
            key = (item['kind'], str(item['subject_key']))
            payload = {
                'kind':          item['kind'],
                'source_job':    item.get('source_job', ''),
                'run_id':        item.get('run_id', ''),
                'subject_key':   str(item['subject_key']),
                'subject_label': item.get('subject_label', ''),
                'detail':        json.dumps(item.get('detail') or {}, default=str),
            }
            if key in open_idx:
                rows[open_idx[key]].update(payload)  # keep id/created_at/status
            else:
                rows.append({
                    **payload, 'id': uuid.uuid4().hex, 'created_at': now,
                    'status': STATUS_OPEN, 'resolved_by': '', 'resolved_at': '',
                })
                open_idx[key] = len(rows) - 1
        return _ensure(pd.DataFrame(rows))

    gw.read_modify_write(_queue_blob(gcs_client, bucket), mutate)
    return len(items)


def resolve(gcs_client, item_id: str, *, resolved_by: str = '',
            status: str = STATUS_RESOLVED, bucket: str = BUCKET) -> bool:
    """Mark one item resolved/dismissed. Returns True if the id was found."""
    found = {'hit': False}

    def mutate(df):
        df = _ensure(df)
        mask = df['id'].astype(str) == str(item_id)
        if mask.any():
            found['hit'] = True
            df.loc[mask, 'status'] = status
            df.loc[mask, 'resolved_by'] = resolved_by
            df.loc[mask, 'resolved_at'] = _now()
        return df

    gw.read_modify_write(_queue_blob(gcs_client, bucket), mutate)
    return found['hit']


# ── Reading ─────────────────────────────────────────────────────────────────

def load(gcs_client, *, status: str | None = STATUS_OPEN,
         kind: str | None = None, bucket: str = BUCKET) -> pd.DataFrame:
    """The queue as a DataFrame, filtered by status (None = all) and kind."""
    blob = _queue_blob(gcs_client, bucket)
    try:
        df = _ensure(pd.read_parquet(io.BytesIO(blob.download_as_bytes())))
    except Exception:
        return _empty()
    if status is not None:
        df = df[df['status'] == status]
    if kind is not None:
        df = df[df['kind'] == kind]
    return df.reset_index(drop=True)


def open_counts(gcs_client, *, bucket: str = BUCKET) -> dict[str, int]:
    """{kind: open-count} for the Home 'Needs attention' panel."""
    df = load(gcs_client, status=STATUS_OPEN, bucket=bucket)
    if df.empty:
        return {}
    return df.groupby('kind').size().astype(int).to_dict()
