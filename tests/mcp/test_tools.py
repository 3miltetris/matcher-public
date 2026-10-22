"""Unit tests for the transport-agnostic Tier-1 tool layer (Phase 5.1).

Everything runs against the in-memory GCS fake — no network, no Streamlit. The
MCP server (services/matcher_mcp) is a thin wrapper over these functions, so
testing them here tests the behaviour that matters.
"""

import json

import pandas as pd

from src.modules import attention as aq, onboarding as ob
from src.modules.tools import (
    _common as cm, attention as t_att, onboarding as t_ob, runs as t_runs,
    search as t_search,
)
from tests.mcp.conftest import put_csv, put_json, put_parquet


# ── _common ───────────────────────────────────────────────────────────────────

def test_cap_text_middle_out():
    out = cm.cap_text('x' * 1000, 100)
    assert len(out) <= 100 + 40 and out.startswith('x') and out.endswith('x')
    assert 'elided' in out


def test_cursor_roundtrip_and_garbage():
    assert cm.decode_cursor(cm.encode_cursor(42)) == 42
    assert cm.decode_cursor(None) == 0
    assert cm.decode_cursor('not-a-cursor') == 0


# ── attention ─────────────────────────────────────────────────────────────────

def test_list_attention(storage):
    aq.enqueue(storage, [
        {'kind': 'drive_unmatched_folder', 'subject_key': 'f1',
         'subject_label': 'Acme_INTERNAL', 'detail': {'section': 'A-H'}},
        {'kind': 'fathom_unmatched_domain', 'subject_key': 'acme.com',
         'subject_label': 'acme.com'},
    ])
    out = t_att.list_attention(storage)
    assert out['total'] == 2 and out['returned'] == 2
    assert out['open_counts'] == {'drive_unmatched_folder': 1,
                                  'fathom_unmatched_domain': 1}
    item = next(i for i in out['items'] if i['kind'] == 'drive_unmatched_folder')
    assert item['detail'] == {'section': 'A-H'}

    only = t_att.list_attention(storage, kind='fathom_unmatched_domain')
    assert only['returned'] == 1 and only['items'][0]['subject_key'] == 'acme.com'


# ── onboarding ────────────────────────────────────────────────────────────────

def test_onboarding_proposals(storage):
    ob.save_proposal(storage, {
        'folder_id': 'fld1', 'folder_name': 'Beta Systems_INTERNAL',
        'decision': 'existing_client', 'confidence': 'high',
        'company_key': 'Beta Systems||https://beta.com', 'review_status': 'open',
        'evidence': [{'claim': 'letterhead says Beta Systems', 'source': 'file:1'}],
        'documents': [{'file_id': '1', 'name': 'capstmt.pdf', 'category': 'technical',
                       'include': True, 'reason': 'capability'}],
    })
    lst = t_ob.list_onboarding_proposals(storage)
    assert lst['total'] == 1
    card = lst['proposals'][0]
    assert card['decision'] == 'existing_client' and card['n_documents'] == 1

    full = t_ob.get_onboarding_proposal(storage, 'fld1')
    assert full['found'] and full['confidence'] == 'high'
    assert full['evidence'][0]['source'] == 'file:1'
    assert full['documents'][0]['include'] is True

    assert t_ob.get_onboarding_proposal(storage, 'missing')['found'] is False


# ── runs ──────────────────────────────────────────────────────────────────────

def test_list_and_get_run_status(storage):
    put_json(storage, 'drive-sync-jobs/drive_sync_2026-10-01/status.json',
             {'run_id': 'drive_sync_2026-10-01', 'state': 'complete',
              'clients_updated': 3})
    put_json(storage, 'drive-sync-jobs/drive_sync_2026-10-02/status.json',
             {'run_id': 'drive_sync_2026-10-02', 'state': 'running', 'mode': 'scheduled',
              'error': None})

    runs = t_runs.list_runs(storage, 'drive-sync-job', limit=5)
    # newest first
    assert [r['run_id'] for r in runs['runs']] == \
        ['drive_sync_2026-10-02', 'drive_sync_2026-10-01']
    assert runs['runs'][0]['state'] == 'running'

    st = t_runs.get_run_status(storage, 'drive-sync-job', 'drive_sync_2026-10-01')
    assert st['found'] and st['status']['clients_updated'] == 3

    assert t_runs.get_run_status(storage, 'drive-sync-job', 'nope')['found'] is False
    assert 'error' in t_runs.list_runs(storage, 'no-such-job')


def test_get_run_status_trims(storage):
    big_list = [{'x': i} for i in range(100)]
    put_json(storage, 'deep-research-jobs/dr_1/status.json',
             {'run_id': 'dr_1', 'state': 'complete', 'results': big_list,
              'error': 'x' * 5000})
    st = t_runs.get_run_status(storage, 'deep-research-job', 'dr_1')['status']
    assert len(st['results']) == 25 and st['results_total'] == 100
    assert len(st['error']) < 5000


def test_get_match_results_filter_and_paginate(storage):
    seg = pd.DataFrame([
        {'companyName': 'Acme', 'companyWebsite': 'https://acme.com',
         'topic_number': 'A1', 'title': 'Sensors', 'good_match': 'yes'},
        {'companyName': 'Acme', 'companyWebsite': 'https://acme.com',
         'topic_number': 'A2', 'title': 'Radar', 'good_match': 'no'},
        {'companyName': 'Beta', 'companyWebsite': 'https://beta.com',
         'topic_number': 'B1', 'title': 'Optics', 'good_match': 'yes'},
    ])
    put_csv(storage, 'matching-results/run1/segment_001.csv', seg)

    allrows = t_runs.get_match_results(storage, 'run1')
    assert allrows['found'] and allrows['total'] == 3

    acme = t_runs.get_match_results(storage, 'run1',
                                    company_key='Acme||https://acme.com')
    assert acme['total'] == 2

    good = t_runs.get_match_results(storage, 'run1', good_only=True)
    assert good['total'] == 2
    assert {r['topic_number'] for r in good['rows']} == {'A1', 'B1'}

    page1 = t_runs.get_match_results(storage, 'run1', limit=2)
    assert page1['returned'] == 2 and page1['next_cursor']
    page2 = t_runs.get_match_results(storage, 'run1', limit=2,
                                     cursor=page1['next_cursor'])
    assert page2['returned'] == 1 and page2['next_cursor'] is None

    assert t_runs.get_match_results(storage, 'ghost')['found'] is False


def test_whats_new(storage):
    put_json(storage, 'drive-sync-jobs/ds1/status.json',
             {'state': 'complete', 'clients_updated': 4, 'attention_written': 7})
    put_json(storage, 'fathom-jobs/f1/status.json',
             {'state': 'complete', 'meetings_ingested': 9})
    aq.enqueue(storage, [{'kind': 'drive_unmatched_folder', 'subject_key': 'x'}])

    out = t_runs.whats_new(storage)
    assert out['jobs']['drive-sync-job']['clients_updated'] == 4
    assert out['jobs']['fathom-sync-job']['meetings_ingested'] == 9
    assert out['jobs']['sam-gov-job']['latest_run'] is None
    assert out['needs_attention'] == {'drive_unmatched_folder': 1}


# ── search: embeddings ────────────────────────────────────────────────────────

def _fake_embedder(vec):
    return lambda _text: list(vec)


def test_search_grants(storage):
    topics = pd.DataFrame([
        {'topic_number': 'T1', 'title': 'Hyperspectral imaging', 'agency': 'ARMY',
         'broad_agency': 'DOD', 'grant_summary': 'imaging sensors', 'due_date': '2026-12-01',
         'embeddings': [1.0, 0.0, 0.0]},
        {'topic_number': 'T2', 'title': 'Battery chemistry', 'agency': 'ARMY',
         'broad_agency': 'DOD', 'grant_summary': 'energy storage', 'due_date': '',
         'embeddings': [0.0, 1.0, 0.0]},
    ])
    put_parquet(storage, 'data/all-topics/processed/DOD/t_2026.parquet', topics)

    out = t_search.search_grants(storage, _fake_embedder([0.9, 0.1, 0.0]),
                                 'imaging', open_only=False, limit=2)
    assert out['topics_searched'] == 2
    assert out['results'][0]['topic_number'] == 'T1'
    assert out['results'][0]['score'] > out['results'][1]['score']

    scoped = t_search.search_grants(storage, _fake_embedder([0, 1, 0]),
                                    'x', agency='DOD', open_only=False)
    assert scoped['results'][0]['topic_number'] == 'T2'


def test_search_resumes(storage):
    res = pd.DataFrame([
        {'firstName': 'Ada', 'lastName': 'Lovelace', 'company': 'Acme',
         'email': 'ada@acme.com', 'expertise_summary': 'optics + sensors',
         'resume_url': 'u1', 'embeddings': [1.0, 0.0]},
        {'firstName': 'Bob', 'lastName': 'Smith', 'company': 'Beta',
         'email': 'bob@beta.com', 'expertise_summary': 'finance',
         'resume_url': 'u2', 'embeddings': [0.0, 1.0]},
    ])
    put_parquet(storage, 'data/resumes/r_2026.parquet', res)
    out = t_search.search_resumes(storage, _fake_embedder([1.0, 0.05]), 'optics')
    assert out['resumes_searched'] == 2
    assert out['results'][0]['name'] == 'Ada Lovelace'


def test_get_profile(storage):
    prof = pd.DataFrame([{
        'company_key': 'Acme||https://acme.com', 'company_name': 'Acme',
        'companyWebsite': 'https://acme.com', 'profile_summary': 'robotics co',
        'aspects': json.dumps([{'label': 'Edge AI', 'kind': 'technology',
                                'text': 'onboard inference', 'keywords': ['ai', 'edge']}]),
        'markets': json.dumps([{'market': 'Defense', 'tier': 1,
                                'subtitle': 'ISR payloads'}]),
        'unexplored_markets': json.dumps([{'market': 'Maritime',
                                           'rationale': 'sensors transfer to hulls'}]),
    }])
    put_parquet(storage, 'data/client-profiles/profiles.parquet', prof)

    out = t_search.get_profile(storage, 'Acme||https://acme.com')
    assert out['found'] and out['pool'] == 'clients'
    assert out['aspects'][0]['label'] == 'Edge AI'
    assert out['markets'][0]['market'] == 'Defense'
    assert out['markets'][0]['tier'] == '1st'
    assert out['unexplored_markets'][0]['market'] == 'Maritime'

    assert t_search.get_profile(storage, 'ghost||x')['found'] is False
