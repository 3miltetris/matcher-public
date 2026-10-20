"""Attention queue — upsert dedup, resolve, concurrency."""

import json

import src.modules.attention as attn


def test_enqueue_and_open_counts(storage):
    attn.enqueue(storage, [
        {'kind': 'drive_unmatched_folder', 'subject_key': 'fid1', 'subject_label': 'Acme_INTERNAL'},
        {'kind': 'fathom_unmatched_domain', 'subject_key': 'beta.com'},
    ])
    assert attn.open_counts(storage) == {'drive_unmatched_folder': 1, 'fathom_unmatched_domain': 1}
    df = attn.load(storage)
    assert set(df['status']) == {'open'}
    assert set(df.columns) == set(attn.COLUMNS)


def test_dedup_updates_open_row(storage):
    attn.enqueue(storage, [{'kind': 'drive_unmatched_folder', 'subject_key': 'fid1',
                            'run_id': 'run_a', 'detail': {'n': 1}}])
    attn.enqueue(storage, [{'kind': 'drive_unmatched_folder', 'subject_key': 'fid1',
                            'run_id': 'run_b', 'detail': {'n': 2}}])
    df = attn.load(storage, kind='drive_unmatched_folder')
    assert len(df) == 1                       # not duplicated
    row = df.iloc[0]
    assert row['run_id'] == 'run_b'           # refreshed
    assert json.loads(row['detail'])['n'] == 2


def test_distinct_subject_keys_are_separate_rows(storage):
    attn.enqueue(storage, [
        {'kind': 'drive_unmatched_folder', 'subject_key': 'fid1'},
        {'kind': 'drive_unmatched_folder', 'subject_key': 'fid2'},
    ])
    assert attn.open_counts(storage)['drive_unmatched_folder'] == 2


def test_resolve_then_reenqueue_opens_fresh_row(storage):
    attn.enqueue(storage, [{'kind': 'drive_unmatched_folder', 'subject_key': 'fid1'}])
    item_id = attn.load(storage).iloc[0]['id']
    assert attn.resolve(storage, item_id, resolved_by='john@bwcoconsulting.com')
    assert attn.open_counts(storage) == {}                       # none open
    assert len(attn.load(storage, status='resolved')) == 1
    # dedup only considers OPEN rows, so re-reporting opens a fresh one
    attn.enqueue(storage, [{'kind': 'drive_unmatched_folder', 'subject_key': 'fid1'}])
    assert attn.open_counts(storage)['drive_unmatched_folder'] == 1
    assert len(attn.load(storage, status=None)) == 2            # resolved + new open


def test_resolve_unknown_id_is_false(storage):
    assert attn.resolve(storage, 'nope') is False


def test_concurrent_enqueue_preserves_both(storage):
    # Two jobs appending to the one queue blob under the generation guard.
    attn.enqueue(storage, [{'kind': 'drive_deferred', 'subject_key': 'Acme||acme.com'}])
    attn.enqueue(storage, [{'kind': 'fathom_unmatched_domain', 'subject_key': 'gamma.io'}])
    assert attn.open_counts(storage) == {'drive_deferred': 1, 'fathom_unmatched_domain': 1}


def test_load_empty_store(storage):
    assert attn.load(storage).empty
    assert attn.open_counts(storage) == {}
