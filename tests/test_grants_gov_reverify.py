"""Grants.gov liveness re-verification (Phase B) — selection, status mapping, and
the all-open vs rolling-only scope, against the in-memory GCS fake."""

import io

import pandas as pd

import src.modules.grants_gov as gg
from tests.gcs_fake import FakeStorage


def _seed(fake, rows):
    df = pd.DataFrame(rows)
    buf = io.BytesIO(); df.to_parquet(buf, index=False)
    fake.bucket(gg.BUCKET).blob(
        gg.GRANTS_PREFIX + 'grants_gov_2026-01-01_aaaaaa.parquet'
    ).upload_from_string(buf.getvalue())


def _read_back(fake):
    blob = fake.list_blobs(gg.BUCKET, prefix=gg.GRANTS_PREFIX)[0]
    return pd.read_parquet(io.BytesIO(blob.download_as_bytes()))


def test_reverify_all_open_maps_status(monkeypatch):
    fake = FakeStorage()
    _seed(fake, [
        {'topic_number': 'OPEN-1', 'is_rolling': True,  'verify_status': 'active'},
        {'topic_number': 'GONE-1', 'is_rolling': False, 'verify_status': 'active'},
        {'topic_number': 'DEAD-1', 'is_rolling': False, 'verify_status': 'inactive'},
    ])
    statuses = {'OPEN-1': (True, 'posted'), 'GONE-1': (True, 'closed')}
    monkeypatch.setattr(gg, 'lookup_opp_status', lambda n: statuses.get(n, (True, None)))

    res = gg.reverify_grants(fake, rolling_only=False, today='2026-10-29')
    # Both active rows checked (the inactive one is skipped); never the dead one.
    assert res['checked'] == 2
    assert res['refreshed'] == 1 and res['expired'] == 1
    out = _read_back(fake).set_index('topic_number')
    assert out.at['OPEN-1', 'verify_status'] == 'active'
    assert out.at['OPEN-1', 'last_verified_active'] == '2026-10-29'
    assert out.at['GONE-1', 'verify_status'] == 'inactive'
    assert out.at['DEAD-1', 'verify_status'] == 'inactive'   # untouched


def test_reverify_rolling_only_scope(monkeypatch):
    fake = FakeStorage()
    _seed(fake, [
        {'topic_number': 'ROLL-1', 'is_rolling': True,  'verify_status': 'active'},
        {'topic_number': 'DATED-1', 'is_rolling': False, 'verify_status': 'active'},
    ])
    monkeypatch.setattr(gg, 'lookup_opp_status', lambda n: (True, 'posted'))
    res = gg.reverify_grants(fake, rolling_only=True, today='2026-10-29')
    assert res['checked'] == 1          # only the rolling row


def test_reverify_dry_run_writes_nothing(monkeypatch):
    fake = FakeStorage()
    _seed(fake, [{'topic_number': 'GONE-1', 'is_rolling': True, 'verify_status': 'active'}])
    monkeypatch.setattr(gg, 'lookup_opp_status', lambda n: (True, 'closed'))
    res = gg.reverify_grants(fake, rolling_only=False, dry_run=True, today='2026-10-29')
    assert res['expired'] == 1 and res['files_written'] == 0
    assert _read_back(fake).at[0, 'verify_status'] == 'active'   # unchanged on disk


def test_reverify_lookup_failure_leaves_row(monkeypatch):
    fake = FakeStorage()
    _seed(fake, [{'topic_number': 'X-1', 'is_rolling': True, 'verify_status': 'active'}])
    monkeypatch.setattr(gg, 'lookup_opp_status', lambda n: (False, None))  # lookup failed
    res = gg.reverify_grants(fake, rolling_only=False, today='2026-10-29')
    assert res['refreshed'] == 0 and res['expired'] == 0
    assert _read_back(fake).at[0, 'verify_status'] == 'active'
