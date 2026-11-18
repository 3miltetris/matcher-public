"""Call-booking roadmap: pitched-program exclusion, 3-program writeback, the
rerun-on-materials gate, idempotent reuse, and the poll queue selection.

The ML/fetch stages are faked so no network, Claude, or embedding runs; the logic
under test is the exclusion, the property writeback, and the fingerprint gate."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import src.modules.profile_build as pb
import src.modules.roadmap.booking_hs as bk
import src.modules.roadmap.deck as deck
import src.modules.roadmap.fetch as fetch
import src.modules.roadmap.matcher as mt
import src.modules.roadmap.pipeline as rp
from tests.roadmap.conftest import FakeHubSpot

_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _deps(storage, bm, hubspot):
    return rp.RoadmapDeps(storage_client=storage, bm=bm, embed=lambda t: [0.0],
                          anth=object(), hubspot=hubspot, anthropic_api_key='',
                          bucket='test-bucket', notify_fail=None, now=lambda: _NOW)


# Four candidates; the pitched one (PITCH1) has the TOP score, so excluding it is
# observable — otherwise it would be program #1.
def _candidates():
    return pd.DataFrame([
        {'topic_number': 'PITCH1', 'title': 'Confined Space Monitoring',
         'source': 'https://sam.gov/pitch', 'grant_summary': 'Already pitched.',
         'aspect_label': 'MEMS', 'aspect_score': 0.95, 'agency': 'ARMY'},
        {'topic_number': 'T1', 'title': 'Autonomy Sensors',
         'source': 'https://sam.gov/t1', 'grant_summary': 'Autonomy sensing.',
         'aspect_label': 'MEMS', 'aspect_score': 0.90, 'agency': 'NAVY'},
        {'topic_number': 'T2', 'title': 'Water Treatment',
         'source': 'https://sam.gov/t2', 'grant_summary': 'Clean water.',
         'aspect_label': 'Filters', 'aspect_score': 0.85, 'agency': 'EPA'},
        {'topic_number': 'T3', 'title': 'Edge AI Inference',
         'source': 'https://sam.gov/t3', 'grant_summary': 'Edge compute.',
         'aspect_label': 'Edge', 'aspect_score': 0.80, 'agency': 'DARPA'},
    ])


def _stub(monkeypatch, *, candidates=None, captured=None, build_counter=None,
          website=('MEMS gas sensors for confined-space robotics work.', ''),
          deck_text=('', '')):
    cand = _candidates() if candidates is None else candidates

    def _build(*a, **k):
        if build_counter is not None:
            build_counter['n'] += 1
        return {'profile_summary': 'Sensors company.', 'aspect_labels': 'MEMS'}, []
    monkeypatch.setattr(pb, 'build_profile_from_texts', _build)
    monkeypatch.setattr(fetch, 'fetch_website_text', lambda url, **k: website)
    monkeypatch.setattr(fetch, 'fetch_deck_text', lambda url, tok, **k: deck_text)
    monkeypatch.setattr(mt, 'get_topic_index', lambda *a, **k: object())
    monkeypatch.setattr(mt, 'match_profile', lambda *a, **k: (cand.copy(), []))
    monkeypatch.setattr(mt, 'rerank', lambda c, *a, **k: c)

    def _nar(anth, **k):
        if captured is not None:
            captured['avoid'] = k.get('avoid')
            captured['n_opps'] = len(k.get('opportunities') or [])
        items = {i: {'what_it_is': f'Funds area {i}.', 'why_it_fits': f'Fits via MEMS {i}.'}
                 for i in range(len(k.get('opportunities') or []))}
        return {'why_now': '', 'items': items}, []
    monkeypatch.setattr(deck, 'generate_narrative', _nar)


def _booked(cid='c1', **extra):
    props = {'email': 'ada@acme.com', 'company': 'Acme Robotics',
             'website': 'acme.com', 'call_booked_at': '2026-01-01T09:00:00Z',
             'pitched_program': 'PITCH1'}
    props.update(extra)
    return FakeHubSpot(cid, props)


# ── Exclusion + writeback ─────────────────────────────────────────────────────

def test_pitched_topic_excluded_and_three_programs_written(storage, bm, monkeypatch):
    captured = {}
    _stub(monkeypatch, captured=captured)
    hs = _booked()
    res = rp.generate_booking_roadmap(_deps(storage, bm, hs), contact_id='c1')

    assert res['status'] == 'complete'
    assert res['opportunities'] == 3
    p = hs.patched_by_id['c1']
    # The pitched program is gone; the three slots are the next-best, in order.
    names = [p['call_program_1_name'], p['call_program_2_name'], p['call_program_3_name']]
    assert names == ['Autonomy Sensors', 'Water Treatment', 'Edge AI Inference']
    assert 'Confined Space Monitoring' not in names
    # Each slot carries a 2-sentence summary and the official source link.
    assert p['call_program_1_summary'] == 'Funds area 0. Fits via MEMS 0.'
    assert p['call_program_1_link'] == 'https://sam.gov/t1'
    # The excluded program is echoed for the email, and the copy was told to avoid it.
    assert p['call_pitched_program'] == 'Confined Space Monitoring'
    assert captured['avoid'] == 'Confined Space Monitoring'
    assert captured['n_opps'] == 3
    assert p['call_roadmap_status'] == 'complete'


def test_pitched_matched_by_fuzzy_title(storage, bm, monkeypatch):
    _stub(monkeypatch)
    hs = _booked(pitched_program='confined space')     # a name, not the topic number
    rp.generate_booking_roadmap(_deps(storage, bm, hs), contact_id='c1')
    p = hs.patched_by_id['c1']
    names = [p[f'call_program_{i}_name'] for i in (1, 2, 3)]
    assert 'Confined Space Monitoring' not in names
    assert p['call_pitched_program'] == 'Confined Space Monitoring'


def test_no_match_when_only_candidate_was_the_pitched_one(storage, bm, monkeypatch):
    only_pitched = _candidates().iloc[[0]].reset_index(drop=True)
    _stub(monkeypatch, candidates=only_pitched)
    hs = _booked()
    res = rp.generate_booking_roadmap(_deps(storage, bm, hs), contact_id='c1')
    assert res['status'] == 'no_match'
    p = hs.patched_by_id['c1']
    assert p['call_roadmap_status'] == 'no_match'
    assert p['call_program_1_name'] == ''          # slots cleared, not stale


def test_no_input_when_no_site_or_materials(storage, bm, monkeypatch):
    _stub(monkeypatch, website=('', 'HTTP 404'))
    hs = _booked()
    res = rp.generate_booking_roadmap(_deps(storage, bm, hs), contact_id='c1')
    assert res['status'] == 'no_input'
    assert hs.patched_by_id['c1']['call_roadmap_status'] == 'no_input'


# ── Idempotent reuse + rerun gate ─────────────────────────────────────────────

def test_unchanged_repoll_reuses_without_rebuilding(storage, bm, monkeypatch):
    counter = {'n': 0}
    _stub(monkeypatch, build_counter=counter)
    deps = _deps(storage, bm, _booked())
    first  = rp.generate_booking_roadmap(deps, contact_id='c1')
    second = rp.generate_booking_roadmap(deps, contact_id='c1')
    assert counter['n'] == 1                        # profile built once
    assert first['programs'] == second['programs']  # same result re-written


def test_materials_arrival_triggers_a_rerun(storage, bm, monkeypatch):
    counter = {'n': 0}
    _stub(monkeypatch, build_counter=counter)
    hs = _booked()
    deps = _deps(storage, bm, hs)
    rp.generate_booking_roadmap(deps, contact_id='c1')      # first build, no materials
    assert counter['n'] == 1

    # A materials deck arrives and is readable.
    hs.contacts['c1']['call_materials_url'] = 'https://files.hubspotusercontent.net/d.pdf'
    monkeypatch.setattr(fetch, 'fetch_deck_text',
                        lambda url, tok, **k: ('Pitch deck: thermal imaging too.', ''))
    rp.generate_booking_roadmap(deps, contact_id='c1')      # fingerprint changed -> rebuild
    assert counter['n'] == 2


def test_claim_selects_new_and_changed_only(storage, bm, monkeypatch):
    _stub(monkeypatch)
    hs = _booked('new')                             # brand-new booking, no status
    # Already complete, no materials -> must NOT be re-claimed.
    hs.add('done', {'email': 'd@x.com', 'call_booked_at': 't', 'website': 'd.com',
                    'pitched_program': 'P', 'call_roadmap_status': 'complete'})
    # Complete WITH a materials deck whose fingerprint differs from any record.
    hs.add('changed', {'email': 'e@x.com', 'call_booked_at': 't', 'website': 'e.com',
                       'pitched_program': 'P', 'call_roadmap_status': 'complete',
                       'call_materials_url': 'https://files.hubspotusercontent.net/x.pdf'})

    ids = rp.claim_pending_bookings(_deps(storage, bm, hs), limit=10)
    assert set(ids) == {'new', 'changed'}           # not 'done'
    for cid in ids:
        assert hs.contacts[cid]['call_roadmap_status'] == 'processing'


def test_stale_processing_claim_is_recovered(storage, bm, monkeypatch):
    _stub(monkeypatch)
    hs = FakeHubSpot()
    hs.add('stuck', {'email': 'a@x.com', 'call_booked_at': 't', 'website': 'a.com',
                     'pitched_program': 'P', 'call_roadmap_status': 'processing',
                     'call_roadmap_generated_at': (_NOW - timedelta(minutes=30)).isoformat()})
    hs.add('fresh', {'email': 'b@x.com', 'call_booked_at': 't', 'website': 'b.com',
                     'pitched_program': 'P', 'call_roadmap_status': 'processing',
                     'call_roadmap_generated_at': _NOW.isoformat()})
    ids = rp.claim_pending_bookings(_deps(storage, bm, hs), limit=10, stale_min=15)
    assert ids == ['stuck']                         # fresh claim left alone


def test_failure_marks_error_on_contact(storage, bm, monkeypatch):
    _stub(monkeypatch)
    monkeypatch.setattr(pb, 'build_profile_from_texts',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    hs = _booked()
    res = rp.generate_booking_roadmap(_deps(storage, bm, hs), contact_id='c1')
    assert res['status'] == 'error'
    assert hs.patched_by_id['c1']['call_roadmap_status'] == 'error'
    # The failed attempt is counted against these inputs, so a later poll can stop
    # retrying this contact once the budget is spent. It is recorded as the
    # ATTEMPT's fingerprint — `fingerprint` only ever names a finished result.
    rec, _ = bm.download_json('roadmap/runs/booking_c1.json')
    assert rec['attempts'] == 1 and rec['attempt_fingerprint']
    assert 'fingerprint' not in rec
    # Full timestamp, so the retry's 15-minute staleness wait actually waits.
    assert 'T' in hs.patched_by_id['c1']['call_roadmap_generated_at']


# ── Error recovery + changed-input re-runs (claim queue) ──────────────────────

def _seed_record(bm, cid: str, props: dict, *, attempts: int, fingerprint=None):
    """Write a run record as the pipeline would, so _load_record sees it."""
    fp = fingerprint if fingerprint is not None else rp._booking_fingerprint(
        bk.booking_inputs({**props, '_id': cid}))
    bm.upload_json(f'roadmap/runs/{rp._booking_run_id(cid)}.json',
                   {'run_id': rp._booking_run_id(cid), 'fingerprint': fp,
                    'attempts': attempts})
    return fp


def test_stale_error_recovered_under_budget_not_over(storage, bm, monkeypatch):
    _stub(monkeypatch)
    hs = FakeHubSpot()
    stale = (_NOW - timedelta(minutes=30)).isoformat()
    # Two errored bookings, both stale; one has spent its retry budget.
    under = {'email': 'u@x.com', 'call_booked_at': 't', 'website': 'u.com',
             'pitched_program': 'P', 'call_roadmap_status': 'error',
             'call_roadmap_generated_at': stale}
    over  = {'email': 'o@x.com', 'call_booked_at': 't', 'website': 'o.com',
             'pitched_program': 'P', 'call_roadmap_status': 'error',
             'call_roadmap_generated_at': stale}
    hs.add('under', under)
    hs.add('over', over)
    _seed_record(bm, 'under', under, attempts=1)                       # budget left
    _seed_record(bm, 'over', over, attempts=rp._MAX_BOOKING_ATTEMPTS)  # spent

    ids = rp.claim_pending_bookings(_deps(storage, bm, hs), limit=10, stale_min=15)
    assert ids == ['under']                       # 'over' is left stuck by design


def test_fresh_error_is_not_retried_yet(storage, bm, monkeypatch):
    _stub(monkeypatch)
    hs = FakeHubSpot()
    fresh = {'email': 'f@x.com', 'call_booked_at': 't', 'website': 'f.com',
             'pitched_program': 'P', 'call_roadmap_status': 'error',
             'call_roadmap_generated_at': _NOW.isoformat()}
    hs.add('fresh', fresh)
    _seed_record(bm, 'fresh', fresh, attempts=1)
    ids = rp.claim_pending_bookings(_deps(storage, bm, hs), limit=10, stale_min=15)
    assert ids == []                              # wait out the transient fault first


def test_over_budget_error_retries_again_when_inputs_change(storage, bm, monkeypatch):
    _stub(monkeypatch)
    hs = FakeHubSpot()
    stale = (_NOW - timedelta(minutes=30)).isoformat()
    props = {'email': 'o@x.com', 'call_booked_at': 't', 'website': 'new-site.com',
             'pitched_program': 'P', 'call_roadmap_status': 'error',
             'call_roadmap_generated_at': stale}
    hs.add('o', props)
    # Record is over budget but against a DIFFERENT (old) fingerprint.
    _seed_record(bm, 'o', props, attempts=rp._MAX_BOOKING_ATTEMPTS,
                 fingerprint='old-inputs')
    ids = rp.claim_pending_bookings(_deps(storage, bm, hs), limit=10, stale_min=15)
    assert ids == ['o']                           # new inputs -> fresh attempt


def test_no_input_rerun_when_website_arrives(storage, bm, monkeypatch):
    _stub(monkeypatch)
    hs = FakeHubSpot()
    # Built 'no_input' earlier (empty website); a website has since been added.
    props = {'email': 'n@x.com', 'call_booked_at': 't', 'website': 'added-later.com',
             'pitched_program': 'P', 'call_roadmap_status': 'no_input'}
    hs.add('n', props)
    _seed_record(bm, 'n', props, attempts=1, fingerprint='was-empty')  # old fingerprint
    ids = rp.claim_pending_bookings(_deps(storage, bm, hs), limit=10)
    assert ids == ['n']                           # reinput_candidates picked it up


def test_concurrent_poll_blocked_by_claim_lock(storage, bm, monkeypatch):
    from src.modules import gcs_write
    _stub(monkeypatch)
    hs = _booked('new')                           # a claimable new booking
    # Another poll already holds the booking-poll claim lock.
    held = gcs_write.acquire_lock(storage.bucket('test-bucket'),
                                  rp._BOOKING_POLL_LOCK, 'other-poll', ttl_seconds=180)
    assert held is not None
    ids = rp.claim_pending_bookings(_deps(storage, bm, hs), limit=10)
    assert ids == []                              # did not double-claim
    assert hs.contacts['new'].get('call_roadmap_status') != 'processing'
    # Once the holder releases, the next poll claims normally.
    gcs_write.release_lock(held)
    ids2 = rp.claim_pending_bookings(_deps(storage, bm, hs), limit=10)
    assert ids2 == ['new']


# ── Endpoint ──────────────────────────────────────────────────────────────────

def test_booking_poll_endpoint_guard_and_count(monkeypatch):
    from fastapi.testclient import TestClient

    import services.intake_web.main as main
    monkeypatch.setenv('ROADMAP_WEBHOOK_SECRET', 's3cret')
    monkeypatch.setattr(main, 'rm_deps', lambda: object())
    monkeypatch.setattr(main.rp, 'claim_pending_bookings', lambda deps, limit=10: ['c1', 'c2'])
    ran = {}
    monkeypatch.setattr(main.rp, 'generate_booking_batch',
                        lambda *a, **k: ran.setdefault('n', len(a[1])))
    c = TestClient(main.app)

    assert c.post('/api/roadmap/booking-poll?secret=nope').status_code == 401
    r = c.post('/api/roadmap/booking-poll?secret=s3cret&limit=10')
    assert r.status_code == 200 and r.json()['claimed'] == 2
    assert ran.get('n') == 2


# ── Review fixes: killed re-run, retry clock, crowding, checked memo ──────────

def test_killed_rerun_rebuilds_instead_of_resending_old_programs(storage, bm, monkeypatch):
    counter = {'n': 0}
    _stub(monkeypatch, build_counter=counter)
    hs = _booked()
    deps = _deps(storage, bm, hs)
    assert rp.generate_booking_roadmap(deps, contact_id='c1')['status'] == 'complete'
    assert counter['n'] == 1

    # The founder sends a deck; the re-run is claimed, then the container is
    # killed mid-build (no `except` runs — modelled with SystemExit).
    hs.contacts['c1']['call_materials_url'] = 'https://files.example/deck.pdf'
    monkeypatch.setattr(pb, 'build_profile_from_texts',
                        lambda *a, **k: (_ for _ in ()).throw(SystemExit()))
    with pytest.raises(SystemExit):
        rp.generate_booking_roadmap(deps, contact_id='c1')

    # The recovery poll must BUILD with the deck, not re-send the old programs.
    _stub(monkeypatch, build_counter=counter)
    res = rp.generate_booking_roadmap(deps, contact_id='c1')
    assert counter['n'] == 2                       # rebuilt, not reused
    assert res['status'] == 'complete'


def _error_contact(cid, *, website):
    return {'email': f'{cid}@x.com', 'call_booked_at': 't', 'website': website,
            'pitched_program': 'P', 'call_roadmap_status': 'error',
            'call_roadmap_generated_at': (_NOW - timedelta(minutes=30)).isoformat(),
            'hs_lastmodifieddate': '2026-01-01T11:00:00Z'}


def test_exhausted_errors_do_not_crowd_out_a_retryable_one(storage, bm, monkeypatch):
    _stub(monkeypatch)
    hs = FakeHubSpot()
    for i in range(30):                            # more than a whole claim batch
        props = _error_contact(f'dead{i}', website=f'dead{i}.com')
        hs.add(f'dead{i}', props)
        _seed_record(bm, f'dead{i}', props, attempts=rp._MAX_BOOKING_ATTEMPTS)
    live = _error_contact('live', website='live.com')
    hs.add('live', live)
    _seed_record(bm, 'live', live, attempts=1)

    ids = rp.claim_pending_bookings(_deps(storage, bm, hs), limit=10, stale_min=15)
    assert ids == ['live']


def _no_input_contact(cid, *, website):
    return {'email': f'{cid}@x.com', 'call_booked_at': 't', 'website': website,
            'pitched_program': 'P', 'call_roadmap_status': 'no_input',
            'hs_lastmodifieddate': '2026-01-01T11:00:00Z'}


def test_unchanged_no_input_contacts_do_not_hide_a_changed_one(storage, bm, monkeypatch):
    _stub(monkeypatch)
    hs = FakeHubSpot()
    for i in range(30):
        props = _no_input_contact(f'same{i}', website='')
        hs.add(f'same{i}', props)
        _seed_record(bm, f'same{i}', props, attempts=1)      # fingerprint == inputs
    late = _no_input_contact('late', website='late-site.com')
    hs.add('late', late)
    _seed_record(bm, 'late', late, attempts=1, fingerprint='built-without-site')

    ids = rp.claim_pending_bookings(_deps(storage, bm, hs), limit=10)
    assert ids == ['late']


def test_checked_memo_skips_unchanged_contacts_until_they_change(storage, bm, monkeypatch):
    _stub(monkeypatch)
    hs = FakeHubSpot()
    for i in range(5):
        props = _no_input_contact(f'same{i}', website='')
        hs.add(f'same{i}', props)
        _seed_record(bm, f'same{i}', props, attempts=1)
    deps = _deps(storage, bm, hs)

    reads = []
    real = rp._load_record
    monkeypatch.setattr(rp, '_load_record',
                        lambda d, run_id: reads.append(run_id) or real(d, run_id))

    assert rp.claim_pending_bookings(deps, limit=10) == []
    assert len(reads) == 5                         # first poll examines each once
    reads.clear()
    assert rp.claim_pending_bookings(deps, limit=10) == []
    assert reads == []                             # unchanged: no record reads at all

    # A website arrives on one contact — HubSpot bumps its last-modified time.
    hs.contacts['same3'].update(website='now-has-site.com',
                                hs_lastmodifieddate='2026-01-01T11:30:00Z')
    reads.clear()
    assert rp.claim_pending_bookings(deps, limit=10) == ['same3']
    assert len(reads) == 1
