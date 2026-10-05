import io
import json

import pandas as pd
import pytest

import src.modules.aspect_profile as ap
from src.modules.intake import profile_writer as pw
from src.modules.intake import sessions as ss
from src.modules.intake import submit_pipeline as sp


def _embed(text):
    return [0.1] * 4


def _rows(storage, prefix='data/all-contacts/prospects/'):
    out = {}
    for name, rec in storage.store.items():
        if name.startswith(prefix) and name.endswith('.parquet'):
            out[name] = pd.read_parquet(io.BytesIO(rec['data']))
    return out


# ── profile_writer ───────────────────────────────────────────────────────────

def test_new_prospect_row_is_clients_convention_and_profile_ready(storage, answers):
    res = pw.write_prospect(storage, answers, 'S' * 32, '2026-10-05T12:00:00+00:00', _embed)
    assert res['created'] and res['pool'] == 'prospects'
    (blob, df), = _rows(storage).items()
    row = df.iloc[0]
    assert row['company_name'] == 'Acme Robotics, Inc.'
    assert row['companyWebsite'] == 'https://acme-robotics.com'
    assert row['email'] == 'ada@acme-robotics.com'
    assert list(row['embeddings']) == [0.1] * 4
    texts = ap.assemble_source_texts(ap.merge_company_row(df))
    assert 'MEMS gas sensors' in texts['intake']
    # retry of the same session rewrites the same blob
    pw.write_prospect(storage, answers, 'S' * 32, '2026-10-05T12:00:00+00:00', _embed)
    assert len(_rows(storage)) == 1


def test_existing_client_is_updated_in_place_not_duplicated(storage, answers):
    client = pd.DataFrame([{
        'company_name': 'Acme Robotics', 'companyWebsite': 'https://www.acme-robotics.com/',
        'email': 'ceo@acme-robotics.com', 'summary': 'staff summary',
        'embeddings': [0.9, 0.9],
        'intake_data': json.dumps({'drive_folder_id': 'F1', 'hubspot_company_id': '77'}),
    }])
    buf = io.BytesIO()
    client.to_parquet(buf, index=False)
    storage.bucket('b').blob('data/all-contacts/clients/c.parquet').upload_from_string(buf.getvalue())

    res = pw.write_prospect(storage, answers, 'T' * 32, '2026-10-05T12:00:00+00:00', _embed)
    assert (res['pool'], res['created']) == ('clients', False)
    assert _rows(storage) == {}                          # no prospect written
    df = _rows(storage, 'data/all-contacts/clients/')['data/all-contacts/clients/c.parquet']
    assert len(df) == 2                                  # new email → new contact row
    assert set(df['summary']) == {'staff summary'}       # summary untouched
    data = json.loads(df.iloc[0]['intake_data'])
    assert data['drive_folder_id'] == 'F1' and data['hubspot_company_id'] == '77'
    assert pw.previous_ids(storage, 'clients', res['company_key'])['drive_folder_id'] == 'F1'


# ── submit pipeline ──────────────────────────────────────────────────────────

class Calls:
    def __init__(self):
        self.n = {}

    def hit(self, name):
        self.n[name] = self.n.get(name, 0) + 1


@pytest.fixture
def wired(monkeypatch, storage, bm):
    calls = Calls()
    state = {'hubspot_failures': 0}

    def find_or_create(svc, drive_id, parent, name, known):
        calls.hit('folder')
        return {'folder_id': 'FOLD', 'folder_url': 'u', 'created': True,
                'match': 'none', 'needs_review': False}

    def write_doc(*a, **k):
        calls.hit('doc')
        return {'doc_id': 'DOC', 'doc_url': 'd'}

    def sync(*a, **k):
        calls.hit('hubspot')
        if state['hubspot_failures'] > 0:
            state['hubspot_failures'] -= 1
            raise RuntimeError('HubSpot 502')
        return {'hubspot_company_id': '1', 'hubspot_contact_id': '2',
                'company_created': True, 'contact_created': True}

    def trigger(*a, **k):
        calls.hit('trigger')
        return 'run1'

    monkeypatch.setattr(sp.dfo, 'find_or_create', find_or_create)
    monkeypatch.setattr(sp.dfo, 'write_doc', write_doc)
    monkeypatch.setattr(sp.hubspot_sync, 'sync', sync)
    monkeypatch.setattr(sp.profile_trigger, 'trigger', trigger)
    sent = []
    deps = sp.Deps(storage_client=storage, bm=bm, embed=_embed, sleep=lambda s: None,
                   notify_ok=lambda *a: sent.append(('ok', a[2])),
                   notify_fail=lambda *a: sent.append(('fail', a[2])))
    return deps, calls, state, sent


def test_pipeline_runs_all_steps(wired, bm, answers):
    deps, calls, _, sent = wired
    session = ss.create_session(bm, answers)
    sp.submit(deps, session)
    session = sp.run_remaining(deps, ss.load_session(bm, session['session_id']))
    assert session['status'] == ss.COMPLETE
    assert all(session['steps'][s]['status'] == 'done' for s in sp.STEPS)
    assert sent[-1][0] == 'ok'
    assert bm.exists(f"intake/submissions/{session['session_id']}.json")


def test_failure_then_resume_does_not_repeat_done_steps(wired, bm, answers):
    deps, calls, state, sent = wired
    state['hubspot_failures'] = 3
    session = ss.create_session(bm, answers)
    sp.submit(deps, session)
    session = sp.run_remaining(deps, ss.load_session(bm, session['session_id']))
    assert session['status'] == ss.FAILED and sent[-1] == ('fail', 'hubspot')
    assert calls.n['hubspot'] == 3

    session = sp.run_remaining(deps, ss.load_session(bm, session['session_id']))
    assert session['status'] == ss.COMPLETE
    assert calls.n['folder'] == 1 and calls.n['doc'] == 1 and calls.n['hubspot'] == 4


def test_invalid_answers_write_nothing(wired, bm, storage, answers):
    deps, *_ = wired
    answers.pop('contact_email')
    session = ss.create_session(bm, answers)
    with pytest.raises(sp.ValidationFailed) as e:
        sp.submit(deps, session)
    assert 'contact_email' in e.value.errors
    assert _rows(storage) == {}
