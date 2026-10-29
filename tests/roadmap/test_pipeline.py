"""End-to-end pipeline wiring, with the ML/render stages faked."""

from datetime import datetime, timezone

import pandas as pd

import src.modules.profile_build as pb
import src.modules.roadmap.deck as deck
import src.modules.roadmap.matcher as mt
import src.modules.roadmap.pipeline as rp
from tests.roadmap.conftest import FakeHubSpot

_NOW = lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)


def _deps(storage, bm, hubspot):
    return rp.RoadmapDeps(
        storage_client=storage, bm=bm, embed=lambda t: [0.0],
        anth=object(), hubspot=hubspot, anthropic_api_key='',
        bucket='test-bucket', notify_fail=None, now=_NOW)


def _stub_generation(monkeypatch, *, opportunities, render_counter=None):
    monkeypatch.setattr(pb, 'build_profile_from_texts',
                        lambda *a, **k: ({'profile_summary': 'Sensors co.',
                                          'aspect_labels': 'MEMS'}, []))
    monkeypatch.setattr(mt, 'get_topic_index', lambda *a, **k: object())
    monkeypatch.setattr(mt, 'match_profile',
                        lambda *a, **k: (pd.DataFrame([{'topic_number': 'T0'}]), []))
    monkeypatch.setattr(mt, 'top_opportunities', lambda *a, **k: opportunities)
    monkeypatch.setattr(deck, 'generate_narrative',
                        lambda *a, **k: ({'why_now': 'Now.', 'items': {}}, []))

    def _render(ctx):
        if render_counter is not None:
            render_counter['n'] += 1
        return b'%PDF-1.4 fake'
    monkeypatch.setattr(deck, 'render_pdf', _render)


_OPPS = [{'title': 'DoD SBIR', 'agency': 'ARMY', 'grant_summary': 'Sensors.',
          'source': 'https://x', 'aspect_label': 'MEMS', 'due_date': '2026-06-01'}]


def test_happy_path_writes_pdf_and_hubspot(storage, bm, hubspot, monkeypatch):
    _stub_generation(monkeypatch, opportunities=_OPPS)
    res = rp.generate_roadmap(_deps(storage, bm, hubspot),
                              contact_id='c123', mode='event',
                              public_base='https://host')
    assert res['status'] == 'complete'
    assert res['opportunities'] == 1

    # PDF kept in GCS as a durable copy...
    pdf_blobs = [n for n in storage.store if n.startswith('roadmap/pdfs/')]
    assert len(pdf_blobs) == 1
    assert storage.store[pdf_blobs[0]]['data'] == b'%PDF-1.4 fake'
    # ...and the public link is the HubSpot Files URL.
    assert len(hubspot.file_uploads) == 1
    assert hubspot.patched['roadmap_pdf_url'] == hubspot.files_url
    assert hubspot.patched['roadmap_status'] == 'complete'


def test_falls_back_to_serve_route_when_files_upload_fails(storage, bm, hubspot, monkeypatch):
    _stub_generation(monkeypatch, opportunities=_OPPS)
    hubspot.files_fail = True
    res = rp.generate_roadmap(_deps(storage, bm, hubspot), contact_id='c123',
                              mode='event', public_base='https://host')
    assert res['status'] == 'complete'
    url = hubspot.patched['roadmap_pdf_url']
    assert url.startswith('https://host/api/roadmap/') and url.endswith('.pdf')

    # Run record persisted.
    rec, _ = bm.download_json(f'roadmap/runs/{res["run_id"]}.json')
    assert rec['status'] == 'complete' and rec['pdf_url'] == url


def test_retry_reuses_pdf(storage, bm, hubspot, monkeypatch):
    counter = {'n': 0}
    _stub_generation(monkeypatch, opportunities=_OPPS, render_counter=counter)
    deps = _deps(storage, bm, hubspot)
    first  = rp.generate_roadmap(deps, contact_id='c123', mode='event',
                                 public_base='https://host')
    second = rp.generate_roadmap(deps, contact_id='c123', mode='event',
                                 public_base='https://host')
    assert counter['n'] == 1                      # PDF rendered once
    assert first['pdf_url'] == second['pdf_url']   # same hosted link
    assert len([n for n in storage.store if n.startswith('roadmap/pdfs/')]) == 1


def test_no_input_still_delivers_empty_deck(storage, bm, monkeypatch):
    # A contact with no description and no deck.
    hs = FakeHubSpot('c999', {'email': 'x@y.com', 'company': 'Thin Co',
                              'event_company_description': ''})
    _stub_generation(monkeypatch, opportunities=_OPPS)   # never reached
    res = rp.generate_roadmap(_deps(storage, bm, hs), contact_id='c999',
                              mode='event', public_base='https://host')
    assert res['status'] == 'no_input'
    assert hs.patched['roadmap_pdf_url'].endswith('.pdf')
    assert hs.patched['roadmap_status'] == 'no_input'


def test_failure_marks_error_on_contact(storage, bm, hubspot, monkeypatch):
    _stub_generation(monkeypatch, opportunities=_OPPS)
    monkeypatch.setattr(deck, 'render_pdf',
                        lambda ctx: (_ for _ in ()).throw(RuntimeError('boom')))
    res = rp.generate_roadmap(_deps(storage, bm, hubspot), contact_id='c123',
                              mode='event', public_base='https://host')
    assert res['status'] == 'error'
    assert hubspot.patched['roadmap_status'] == 'error'
