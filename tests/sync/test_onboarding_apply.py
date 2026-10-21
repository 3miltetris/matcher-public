"""onboarding_apply.preview / apply — assignment, skip, new client, overrides."""

import io
import json

import pandas as pd

from src.modules import attention as attn
from src.modules import onboarding as ob
from src.modules import onboarding_apply as oa

BUCKET = 'cc-matcher-bucket-jeg-v1'
ACME = 'Acme Robotics||https://acme.com'


def _assignments(storage):
    blob = storage.bucket(BUCKET).blob('drive-sync-configs/assignments.json')
    return json.loads(blob.download_as_bytes())


def _seed_proposal(storage, **over):
    p = {'folder_id': 'f1', 'folder_name': 'Acme Robotics_INTERNAL',
         'decision': 'existing_client', 'company_key': ACME, 'confidence': 'high',
         'documents': [{'file_id': 'd1', 'include': True},
                       {'file_id': 'd2', 'include': False}]}
    p.update(over)
    ob.save_proposal(storage, p)
    attn.enqueue(storage, [{'kind': 'drive_onboarding_proposal', 'subject_key': 'f1',
                            'subject_label': 'Acme Robotics_INTERNAL'}])
    return p


def test_preview_has_no_side_effects(storage):
    _seed_proposal(storage)
    plan = oa.preview(storage, 'f1')
    assert plan['decision'] == 'existing_client'
    assert plan['client_key'] == ACME
    assert plan['excluded_file_ids'] == ['d2']
    assert not storage.bucket(BUCKET).blob('drive-sync-configs/assignments.json').exists()


def test_apply_existing_client(storage):
    _seed_proposal(storage)
    r = oa.apply(storage, 'f1', actor='john@bwcoconsulting.com')
    assert r['ok'] and r['client_key'] == ACME
    a = _assignments(storage)['assignments']['f1']
    assert a['client_key'] == ACME
    assert a['match_type'] == 'agent_confirmed'
    assert a['excluded_file_ids'] == ['d2']
    # attention item resolved
    assert attn.open_counts(storage) == {}
    assert ob.load_proposal(storage, 'f1')['review_status'] == 'applied'


def test_apply_not_a_client_skips(storage):
    _seed_proposal(storage, decision='not_a_client')
    r = oa.apply(storage, 'f1', actor='john')
    assert r['ok']
    doc = _assignments(storage)
    assert 'f1' in doc['skipped']
    assert 'f1' not in doc['assignments']


def test_apply_unsure_is_noop(storage):
    _seed_proposal(storage, decision='unsure')
    r = oa.apply(storage, 'f1', actor='john')
    assert r['ok'] and r['action'] == 'no change'
    assert not storage.bucket(BUCKET).blob('drive-sync-configs/assignments.json').exists()


def test_apply_new_client_creates_row(storage):
    _seed_proposal(storage, decision='new_client', company_key='',
                   new_company={'company_name': 'New Co', 'website': 'https://new.co'})
    r = oa.apply(storage, 'f1', actor='john')
    assert r['ok'] and r['client_key'] == 'New Co||https://new.co'
    # a clients parquet was written
    blobs = [b.name for b in storage.list_blobs(BUCKET, prefix='data/all-contacts/clients/')]
    assert any('onboard_' in n for n in blobs)
    df = pd.read_parquet(io.BytesIO(
        storage.bucket(BUCKET).blob([n for n in blobs if 'onboard_' in n][0]).download_as_bytes()))
    assert df.iloc[0]['company_name'] == 'New Co'


def test_override_target_company(storage):
    _seed_proposal(storage)
    r = oa.apply(storage, 'f1', overrides={'company_key': 'Beta||https://beta.com'},
                 actor='john')
    assert r['client_key'] == 'Beta||https://beta.com'
    assert _assignments(storage)['assignments']['f1']['client_key'] == 'Beta||https://beta.com'
