import pytest
from fastapi.testclient import TestClient

from src.modules.intake import submit_pipeline as sp


@pytest.fixture
def client(monkeypatch, storage, bm):
    monkeypatch.setenv('INTAKE_DEV', '1')
    monkeypatch.delenv('TURNSTILE_SECRET', raising=False)
    import services.intake_web.main as main
    monkeypatch.setattr(main, 'DEV', True)
    monkeypatch.setattr(main, '_deps', sp.Deps(storage_client=storage, bm=bm,
                                               embed=lambda t: [0.0] * 4))
    main._hits.clear()
    ran = []
    monkeypatch.setattr(main.sp, 'run_remaining', lambda deps, s: ran.append(s['session_id']))
    c = TestClient(main.app)
    c.ran = ran
    return c


def _start(client):
    r = client.post('/api/sessions', json={
        'company_legal_name': 'Acme Robotics', 'website': 'acme-robotics.com',
        'contact_email': 'ada@acme-robotics.com'})
    assert r.status_code == 200, r.text
    return r.json()['session_id']


def test_schema_and_index(client):
    assert client.get('/').status_code == 200
    secs = client.get('/api/schema').json()['sections']
    assert secs[0]['id'] == '1_company'


def test_full_flow(client, answers):
    sid = _start(client)
    assert client.get(f'/api/sessions/{sid}').json()['answers']['website'] == 'acme-robotics.com'
    r = client.put(f'/api/sessions/{sid}/answers',
                   json={'answers': {**answers, 'not_a_field': 'x'}, 'confirmed_sections': ['1_company']})
    assert r.status_code == 200
    r = client.post(f'/api/sessions/{sid}/submit', json={'answers': {}, 'confirmed_sections': []})
    assert r.json() == {'status': 'submitted'}
    assert client.ran == [sid]
    # resubmits are idempotent, and the draft is closed to edits
    assert client.post(f'/api/sessions/{sid}/submit', json={'answers': {}}).status_code == 200
    assert client.put(f'/api/sessions/{sid}/answers', json={'answers': {}}).status_code == 409


def test_submit_returns_field_errors(client):
    sid = _start(client)
    r = client.post(f'/api/sessions/{sid}/submit', json={'answers': {}})
    assert r.status_code == 422
    assert 'technology_description' in r.json()['errors']


def test_unknown_session_is_404(client):
    assert client.get('/api/sessions/' + 'x' * 32).status_code == 404
    assert client.get('/api/sessions/short').status_code == 404


def test_upload_rejects_bad_type(client):
    sid = _start(client)
    r = client.post(f'/api/sessions/{sid}/upload-urls',
                    json={'files': [{'filename': 'deck.exe', 'size': 10}]})
    assert r.status_code == 400


def test_session_rate_limit(client):
    for _ in range(10):
        _start(client)
    r = client.post('/api/sessions', json={
        'company_legal_name': 'x', 'website': 'x.com', 'contact_email': 'a@x.com'})
    assert r.status_code == 429
