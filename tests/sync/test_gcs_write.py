"""read_modify_write + apply_column_updates — the generation-guarded parquet write."""

import io

import numpy as np
import pandas as pd
import pytest

import src.modules.gcs_write as gw
import src.modules.pools as pl

BUCKET = 'test-bucket'
NAME = 'data/all-contacts/clients/c.parquet'


def _put(storage, name, df):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    storage.bucket(BUCKET).blob(name).upload_from_string(buf.getvalue())


def _get(storage, name):
    data = storage.bucket(BUCKET).blob(name).download_as_bytes()
    return pd.read_parquet(io.BytesIO(data))


def _blob(storage, name):
    return storage.bucket(BUCKET).blob(name)


def _seed(storage):
    _put(storage, NAME, pd.DataFrame({
        'company_name':    ['Acme', 'Beta'],
        'companyWebsite':  ['acme.com', 'beta.com'],
        'summary':         ['old a', 'old b'],
    }))


# ── read_modify_write ───────────────────────────────────────────────────────

def test_rmw_happy_path(storage):
    _seed(storage)
    gw.read_modify_write(_blob(storage, NAME), lambda df: df.assign(docs='x'))
    out = _get(storage, NAME)
    assert list(out['docs']) == ['x', 'x']
    assert list(out['summary']) == ['old a', 'old b']


def test_rmw_creates_missing_blob(storage):
    def mutate(df):
        assert df is None
        return pd.DataFrame({'company_name': ['New'], 'companyWebsite': ['new.com']})
    gw.read_modify_write(_blob(storage, 'data/all-contacts/clients/new.parquet'), mutate)
    out = _get(storage, 'data/all-contacts/clients/new.parquet')
    assert list(out['company_name']) == ['New']


def test_rmw_retries_on_concurrent_write(storage):
    """A writer that races in between our read and write forces one 412; the
    retry re-reads the fresh frame and the racing column survives alongside ours."""
    _seed(storage)
    calls = {'n': 0}

    def mutate(df):
        calls['n'] += 1
        if calls['n'] == 1:
            # Another job writes its own column between our read and our write.
            other = _get(storage, NAME).assign(meetings='from_other')
            _put(storage, NAME, other)
        return df.assign(docs='from_me')

    gw.read_modify_write(_blob(storage, NAME), mutate)
    out = _get(storage, NAME)
    assert calls['n'] == 2                       # one retry
    assert list(out['docs']) == ['from_me', 'from_me']
    assert list(out['meetings']) == ['from_other', 'from_other']  # not clobbered


def test_rmw_raises_after_retries_exhausted(storage):
    _seed(storage)

    def mutate(df):
        _put(storage, NAME, _get(storage, NAME).assign(x=1))  # always races ahead
        return df.assign(docs='me')

    with pytest.raises(gw.ConcurrencyError):
        gw.read_modify_write(_blob(storage, NAME), mutate, retries=2)


# ── apply_column_updates ────────────────────────────────────────────────────

def test_apply_column_updates_scalars_and_embeddings(storage):
    df = pd.DataFrame({
        'company_name':   ['Acme', 'Beta'],
        'companyWebsite': ['acme.com', 'beta.com'],
        'summary':        ['old a', 'old b'],
    })
    updates = {
        'Acme||acme.com': {'summary': 'new a', 'embeddings': [0.1, 0.2, 0.3]},
    }
    out = gw.apply_column_updates(df, updates, pl.key_mask)
    assert out.loc[0, 'summary'] == 'new a'
    assert list(out.loc[0, 'embeddings']) == [0.1, 0.2, 0.3]
    # Beta untouched; its embeddings cell stays empty, never 0.
    assert out.loc[1, 'summary'] == 'old b'
    assert out.loc[1, 'embeddings'] is None


def test_rmw_writes_embeddings_through_parquet(storage):
    """The drive-sync / client-editor path: a float64 ndarray embedding survives
    the RMW parquet round-trip, and an un-updated row's cell stays null (not 0)."""
    _seed(storage)
    emb = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    updates = {'Acme||acme.com': {'summary': 'new a', 'embeddings': emb}}
    gw.read_modify_write(
        _blob(storage, NAME),
        lambda df: gw.apply_column_updates(df, updates, pl.key_mask),
    )
    out = _get(storage, NAME)
    acme = out[out['company_name'] == 'Acme'].iloc[0]
    assert list(acme['embeddings']) == [0.1, 0.2, 0.3]
    assert acme['summary'] == 'new a'
    beta = out[out['company_name'] == 'Beta'].iloc[0]
    assert beta['embeddings'] is None     # never 0


def test_apply_column_updates_skips_absent_key(storage):
    df = pd.DataFrame({'company_name': ['Acme'], 'companyWebsite': ['acme.com']})
    out = gw.apply_column_updates(df, {'Ghost||ghost.com': {'summary': 'x'}}, pl.key_mask)
    assert 'summary' not in out.columns or out['summary'].isna().all()


def test_concurrent_drive_and_fathom_preserve_both_column_sets(storage):
    """The real hazard: drive-sync and fathom-sync both target one blob. With
    RMW, the interleaved writes keep BOTH client_docs_* and client_meetings_*."""
    _seed(storage)
    drive = {'Acme||acme.com': {'client_docs_summary': 'docs', 'docs_updated_at': '2026-10-20'}}
    fathom = {'Acme||acme.com': {'client_meetings_summary': 'calls', 'meetings_updated_at': '2026-10-20'}}

    first = {'done': False}

    def drive_mutate(df):
        # On drive's first pass, fathom sneaks a full write in underneath it.
        if not first['done']:
            first['done'] = True
            fblob = _blob(storage, NAME)
            gw.read_modify_write(fblob, lambda d: gw.apply_column_updates(d, fathom, pl.key_mask))
        return gw.apply_column_updates(df, drive, pl.key_mask)

    gw.read_modify_write(_blob(storage, NAME), drive_mutate)
    out = _get(storage, NAME)
    row = out[out['company_name'] == 'Acme'].iloc[0]
    assert row['client_docs_summary'] == 'docs'
    assert row['client_meetings_summary'] == 'calls'
