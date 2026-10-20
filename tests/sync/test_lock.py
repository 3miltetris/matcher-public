"""Job-level run lock — contention, release, and stale/expired takeover."""

import json
from datetime import datetime, timedelta, timezone

import src.modules.gcs_write as gw

JOB = 'drive-sync-job'
BUCKET = 'test-bucket'


def _bucket(storage):
    return storage.bucket(BUCKET)


def _record(storage):
    cur, _ = gw._read_lock(_bucket(storage).blob(gw.lock_blob(JOB)))
    return cur


def _seed_lock(storage, **rec):
    gw._write_lock(_bucket(storage).blob(gw.lock_blob(JOB)), rec, if_generation_match=0)


def test_acquire_then_contention(storage):
    tok = gw.acquire_lock(_bucket(storage), JOB, run_id='A', ttl_seconds=3600)
    assert tok and tok['run_id'] == 'A'
    # A second live run cannot take it.
    assert gw.acquire_lock(_bucket(storage), JOB, run_id='B', ttl_seconds=3600) is None
    assert _record(storage)['run_id'] == 'A'   # holder untouched


def test_release_lets_next_run_in(storage):
    tok = gw.acquire_lock(_bucket(storage), JOB, run_id='A', ttl_seconds=3600)
    gw.release_lock(tok)
    assert _record(storage)['released'] is True
    tok2 = gw.acquire_lock(_bucket(storage), JOB, run_id='B', ttl_seconds=3600)
    assert tok2 and tok2['run_id'] == 'B'
    assert _record(storage)['released'] is False


def test_stale_lock_is_taken_over(storage):
    past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(timespec='seconds')
    _seed_lock(storage, run_id='dead', acquired_at=past, expires_at=past, released=False)
    tok = gw.acquire_lock(_bucket(storage), JOB, run_id='B', ttl_seconds=3600)
    assert tok and tok['run_id'] == 'B'
    assert _record(storage)['run_id'] == 'B'


def test_unexpired_lock_is_not_taken_over(storage):
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(timespec='seconds')
    _seed_lock(storage, run_id='live', acquired_at=future, expires_at=future, released=False)
    assert gw.acquire_lock(_bucket(storage), JOB, run_id='B', ttl_seconds=3600) is None
    assert _record(storage)['run_id'] == 'live'


def test_release_by_non_owner_is_noop(storage):
    tok_a = gw.acquire_lock(_bucket(storage), JOB, run_id='A', ttl_seconds=3600)
    # Simulate another run having taken the lock over since A acquired it.
    blob = _bucket(storage).blob(gw.lock_blob(JOB))
    _, gen = gw._read_lock(blob)
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(timespec='seconds')
    gw._write_lock(blob, {'run_id': 'B', 'expires_at': future, 'released': False},
                   if_generation_match=gen)
    gw.release_lock(tok_a)                # A no longer owns it
    rec = _record(storage)
    assert rec['run_id'] == 'B'
    assert rec['released'] is False       # not marked released by A
