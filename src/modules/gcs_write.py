"""
Generation-guarded GCS writes and a job-level run lock
------------------------------------------------------
Streamlit-free. drive-sync-job (`client_docs_*`, and `summary`/`embeddings` on a
meaningful change), fathom-sync-job (`client_meetings_*`) and Client Editor
(`summary`/`embeddings`) all rewrite the SAME data/all-contacts/clients/*.parquet
files. Each today bulk-loads a frame, mutates a subset of columns, and uploads
the WHOLE file with no precondition — so whichever writer flushes last silently
drops the others' columns. Scheduling (Phase 2) turns that rare race into a
likely one.

`read_modify_write` re-reads the blob AT WRITE TIME and writes under
`if_generation_match`, retrying on the 412; each writer re-applies only its own
column updates onto the freshest frame (never its stale in-memory copy). The run
lock stops two executions of one scheduled job from racing in the first place.

Both are built only on primitives the in-memory GCS fake in tests/gcs_fake.py
already models (`download_as_bytes` / `.generation` / `upload_from_string` with
`if_generation_match`, and BucketManager.upload_json/download_json), so the whole
module is unit-testable without a live bucket. It mirrors the generation-guard
pattern in src/modules/intake/sessions.py::save_session, adding the retry loop
that intake deliberately omits.
"""

import io
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from google.api_core.exceptions import NotFound, PreconditionFailed


class ConcurrencyError(RuntimeError):
    """A generation-guarded write lost the race past its retry budget."""


# list/array-valued columns (embeddings) must be assigned cell-by-cell; pandas
# will not broadcast a sequence into several cells of a column.
_CELLWISE = (list, tuple, np.ndarray)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Generation-guarded read-modify-write ───────────────────────────────────

def read_modify_write(
    blob, mutate_fn, *, retries: int = 5,
    content_type: str = 'application/octet-stream',
) -> int:
    """Re-read `blob` as a parquet DataFrame, apply `mutate_fn`, write it back
    under if_generation_match so a concurrent writer is never clobbered.

    `blob` is a storage.Blob (or the test fake). `mutate_fn(df_or_None) -> df`
    MUST re-apply only this caller's own changes onto the frame it is handed —
    it is called afresh on every retry with the latest frame, so it must never
    fold in a stale in-memory copy. A missing blob is a create (`df=None`,
    generation 0). Returns the new generation; raises ConcurrencyError once
    `retries` 412s are exhausted. Any other error (a bug in mutate_fn, a real
    GCS fault) propagates unchanged.
    """
    last_exc: Exception | None = None
    for _ in range(retries + 1):
        try:
            data = blob.download_as_bytes()
            gen = blob.generation
            df = pd.read_parquet(io.BytesIO(data))
        except NotFound:
            gen, df = 0, None
        new_df = mutate_fn(df)
        buf = io.BytesIO()
        new_df.to_parquet(buf, index=False)
        try:
            blob.upload_from_string(
                buf.getvalue(), content_type=content_type,
                if_generation_match=gen,
            )
            return blob.generation
        except PreconditionFailed as e:
            last_exc = e
            continue
    raise ConcurrencyError(
        f'{getattr(blob, "name", "blob")}: lost the write race after '
        f'{retries} retries'
    ) from last_exc


def apply_column_updates(df, updates_by_key, key_mask_fn):
    """Apply `{company_key: {column: value}}` onto a fresh contact frame.

    The canonical `mutate_fn` for read_modify_write: a sync job accumulates a
    per-company map of the columns it changed, and this re-applies them by key
    onto whatever frame the write path hands back. Scalars go through
    df.loc[mask, col]; list/ndarray values (embeddings) are assigned per row
    with df.at. A key that matches no row in this blob is skipped — the same
    key can live in another blob. The frame is mutated in place and returned.
    """
    if df is None:
        return df
    for key, cols in updates_by_key.items():
        mask = key_mask_fn(df, key)
        if not mask.any():
            continue
        idx = df.index[mask]
        for col, value in cols.items():
            if isinstance(value, _CELLWISE):
                if col not in df.columns:
                    df[col] = pd.Series([None] * len(df), index=df.index, dtype=object)
                elif df[col].dtype != object:
                    df[col] = df[col].astype(object)
                for i in idx:
                    df.at[i, col] = value
            else:
                df.loc[mask, col] = value
    return df


# ── Job-level run lock ──────────────────────────────────────────────────────
# Bucket-based (not BucketManager) so the raw-storage.Client jobs can use it
# without pulling BucketManager into their images. `bucket` is anything with a
# `.blob(path)` returning an object that supports upload_from_string(
# if_generation_match=…) / download_as_bytes / .generation — a real
# storage.Bucket or the test fake.

def lock_blob(job_name: str) -> str:
    return f'locks/{job_name}.json'


def _takeable(record: dict, now: datetime) -> bool:
    """A lock may be taken over when it was explicitly released or has expired."""
    if record.get('released'):
        return True
    try:
        exp = datetime.fromisoformat(record['expires_at'])
    except Exception:
        return True  # unparseable expiry — treat as stale rather than wedge forever
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return now > exp


def _read_lock(blob):
    try:
        data = blob.download_as_bytes()
    except NotFound:
        return None, None
    return json.loads(data), blob.generation


def _write_lock(blob, record: dict, if_generation_match) -> None:
    blob.upload_from_string(
        json.dumps(record), content_type='application/json',
        if_generation_match=if_generation_match,
    )


def acquire_lock(bucket, job_name: str, run_id: str, ttl_seconds: int):
    """Take the run lock for `job_name`, or return None if a live run holds it.

    Returns a token `{bucket, job_name, run_id, generation}` for release_lock,
    or None when another run owns an unexpired, unreleased lock. A released or
    expired lock may be taken over, and the takeover is logged.
    if_generation_match makes two racing acquirers mutually exclusive — only one
    create (or one takeover under a given generation) can win.
    """
    blob = bucket.blob(lock_blob(job_name))
    now = _now()
    record = {
        'run_id':      run_id,
        'acquired_at': now.isoformat(timespec='seconds'),
        'expires_at':  (now + timedelta(seconds=ttl_seconds)).isoformat(timespec='seconds'),
        'released':    False,
    }
    token = {'bucket': bucket, 'job_name': job_name, 'run_id': run_id}
    # 1) create-only — the common case, no lock present.
    try:
        _write_lock(blob, record, if_generation_match=0)
        return {**token, 'generation': blob.generation}
    except PreconditionFailed:
        pass
    # 2) a lock exists — take it over only if released or expired.
    cur, cur_gen = _read_lock(blob)
    if cur is None:
        # vanished between the create attempt and this read — one more create try.
        try:
            _write_lock(blob, record, if_generation_match=0)
            return {**token, 'generation': blob.generation}
        except PreconditionFailed:
            return None
    if not _takeable(cur, now):
        return None
    try:
        _write_lock(blob, record, if_generation_match=cur_gen)
    except PreconditionFailed:
        return None  # another acquirer took it over first
    print(
        f'[lock] {job_name}: run {run_id} took over '
        f'{"released" if cur.get("released") else "stale"} lock from '
        f'run {cur.get("run_id")}',
        flush=True,
    )
    return {**token, 'generation': blob.generation}


def release_lock(token) -> None:
    """Mark the lock released so the next run can take it immediately.

    Best-effort and generation-guarded: if another run already took the lock
    over (its stored run_id no longer matches ours), it is left untouched.
    """
    if not token:
        return
    blob = token['bucket'].blob(lock_blob(token['job_name']))
    cur, cur_gen = _read_lock(blob)
    if cur is None or cur.get('run_id') != token['run_id']:
        return
    cur['released'] = True
    try:
        _write_lock(blob, cur, if_generation_match=cur_gen)
    except PreconditionFailed:
        pass
