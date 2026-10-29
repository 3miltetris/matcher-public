"""Polling fallback: claim selection, batch isolation, and the endpoint guard."""

from datetime import datetime, timedelta, timezone

import src.modules.roadmap.pipeline as rp
from tests.roadmap.conftest import FakeHubSpot

_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _deps(storage, bm, hubspot):
    return rp.RoadmapDeps(storage_client=storage, bm=bm, embed=lambda t: [0.0],
                          anth=object(), hubspot=hubspot, anthropic_api_key='',
                          bucket='test-bucket', notify_fail=None, now=lambda: _NOW)


def _hs_with_queue():
    hs = FakeHubSpot()
    hs.add('p1', {'email': 'a@x.com', 'event_company_description': 'desc a'})
    hs.add('p2', {'email': 'b@x.com', 'event_company_description': 'desc b'})
    hs.add('done', {'email': 'c@x.com', 'event_company_description': 'desc c',
                    'roadmap_status': 'complete'})
    hs.add('stuck', {'email': 'd@x.com', 'event_company_description': 'desc d',
                     'roadmap_status': 'processing',
                     'roadmap_generated_at': (_NOW - timedelta(minutes=30)).isoformat()})
    hs.add('fresh', {'email': 'e@x.com', 'event_company_description': 'desc e',
                     'roadmap_status': 'processing',
                     'roadmap_generated_at': _NOW.isoformat()})
    return hs


def test_claim_pending_selects_new_and_stale_only(storage, bm):
    hs = _hs_with_queue()
    ids = rp.claim_pending(_deps(storage, bm, hs), limit=10, stale_min=15)
    assert set(ids) == {'p1', 'p2', 'stuck'}        # not 'done', not 'fresh'
    # Every claimed contact is now marked processing.
    for cid in ids:
        assert hs.contacts[cid]['roadmap_status'] == 'processing'
    assert hs.contacts['done']['roadmap_status'] == 'complete'   # untouched


def test_claim_pending_respects_limit(storage, bm):
    hs = _hs_with_queue()
    ids = rp.claim_pending(_deps(storage, bm, hs), limit=1, stale_min=15)
    assert len(ids) == 1


def test_generate_batch_isolates_failures(storage, bm, monkeypatch):
    calls = []

    def _gen(deps, *, contact_id, mode, public_base):
        calls.append(contact_id)
        if contact_id == 'p2':
            raise RuntimeError('boom')
        return {'contact_id': contact_id, 'status': 'complete'}

    monkeypatch.setattr(rp, 'generate_roadmap', _gen)
    out = rp.generate_batch(_deps(storage, bm, FakeHubSpot()), ['p1', 'p2', 'p3'],
                            mode='event', public_base='https://host')
    assert calls == ['p1', 'p2', 'p3']              # p2 failing did not stop p3
    assert [r['contact_id'] for r in out] == ['p1', 'p3']


def test_poll_endpoint_guard_and_count(monkeypatch):
    from fastapi.testclient import TestClient

    import services.intake_web.main as main
    monkeypatch.setenv('ROADMAP_WEBHOOK_SECRET', 's3cret')
    monkeypatch.setattr(main, 'rm_deps', lambda: object())
    monkeypatch.setattr(main.rp, 'claim_pending', lambda deps, limit=10: ['p1', 'p2'])
    ran = {}
    monkeypatch.setattr(main.rp, 'generate_batch',
                        lambda *a, **k: ran.setdefault('n', len(a[1])))
    c = TestClient(main.app)

    assert c.post('/api/roadmap/poll?secret=nope').status_code == 401
    r = c.post('/api/roadmap/poll?secret=s3cret&mode=event&limit=10')
    assert r.status_code == 200 and r.json()['claimed'] == 2
    assert ran.get('n') == 2                         # background batch ran
