import pytest
from fastapi.testclient import TestClient

from src.modules.intake import allowlist as al
from src.modules.intake import submit_pipeline as sp

EMAIL = 'ada@acme-robotics.com'


@pytest.fixture
def client(monkeypatch, storage, bm):
    monkeypatch.setenv('INTAKE_DEV', '1')
    monkeypatch.delenv('TURNSTILE_SECRET', raising=False)
    monkeypatch.delenv('INTAKE_AUTH_SECRET', raising=False)
    import services.intake_web.main as main
    monkeypatch.setattr(main, 'DEV', True)
    monkeypatch.setattr(main, '_deps', sp.Deps(storage_client=storage, bm=bm,
                                               embed=lambda t: [0.0] * 4))
    main._hits.clear()
    main._allow_cache.clear()
    ran = []
    monkeypatch.setattr(main.sp, 'run_remaining', lambda deps, s: ran.append(s['session_id']))
    sent = []
    monkeypatch.setattr(main.notify, 'smtp_configured', lambda: True)
    monkeypatch.setattr(main.notify, 'sign_in_link',
                        lambda email, link, ttl: sent.append((email, link)) or True)
    al.save(storage, ['acme-robotics.com', 'beta.io'], ['founder@gmail.com'], actor='test')
    c = TestClient(main.app)
    c.ran, c.sent, c.storage, c.main = ran, sent, storage, main
    return c


def _signin(client, email=EMAIL):
    n = len(client.sent)
    r = client.post('/api/auth/request', json={'email': email})
    assert r.status_code == 200, r.text
    assert len(client.sent) == n + 1, 'no link was sent'
    token = client.sent[-1][1].split('?t=', 1)[1]
    r = client.post('/api/auth/verify', json={'token': token})
    assert r.status_code == 200, r.text
    return token


def _start(client):
    r = client.post('/api/sessions', json={
        'company_legal_name': 'Acme Robotics', 'website': 'acme-robotics.com'})
    assert r.status_code == 200, r.text
    return r.json()['session_id']


def test_schema_and_index(client):
    assert client.get('/').status_code == 200
    secs = client.get('/api/schema').json()['sections']
    assert secs[0]['id'] == '1_company'


def test_full_flow(client, answers):
    _signin(client)
    sid = _start(client)
    got = client.get(f'/api/sessions/{sid}').json()['answers']
    assert got['website'] == 'acme-robotics.com' and got['contact_email'] == EMAIL
    r = client.put(f'/api/sessions/{sid}/answers',
                   json={'answers': {**answers, 'not_a_field': 'x'}, 'confirmed_sections': ['1_company']})
    assert r.status_code == 200
    r = client.post(f'/api/sessions/{sid}/submit', json={'answers': {}, 'confirmed_sections': []})
    assert r.json() == {'status': 'submitted'}
    assert client.ran == [sid]
    # resubmits are idempotent, and the draft is closed to edits
    assert client.post(f'/api/sessions/{sid}/submit', json={'answers': {}}).status_code == 200
    assert client.put(f'/api/sessions/{sid}/answers', json={'answers': {}}).status_code == 409
    # a submitted session is no longer the founder's draft
    assert client.get('/api/auth/me').json()['session_id'] is None


def test_contact_email_is_locked_to_sign_in(client, answers):
    _signin(client)
    sid = _start(client)
    client.put(f'/api/sessions/{sid}/answers',
               json={'answers': {**answers, 'contact_email': 'someone@else.com'}})
    assert client.get(f'/api/sessions/{sid}').json()['answers']['contact_email'] == EMAIL


def test_submit_returns_field_errors(client):
    _signin(client)
    sid = _start(client)
    r = client.post(f'/api/sessions/{sid}/submit', json={'answers': {}})
    assert r.status_code == 422
    assert 'technology_description' in r.json()['errors']


def test_unknown_session_is_404(client):
    _signin(client)
    assert client.get('/api/sessions/' + 'x' * 32).status_code == 404
    assert client.get('/api/sessions/short').status_code == 404


def test_upload_rejects_bad_type(client):
    _signin(client)
    sid = _start(client)
    r = client.post(f'/api/sessions/{sid}/upload-urls',
                    json={'files': [{'filename': 'deck.exe', 'size': 10}]})
    assert r.status_code == 400


# ── Access ───────────────────────────────────────────────────────────────────

def test_session_routes_require_sign_in(client):
    _signin(client)
    sid = _start(client)
    client.cookies.clear()
    assert client.post('/api/sessions', json={'company_legal_name': 'x', 'website': 'x.com'}).status_code == 401
    assert client.get(f'/api/sessions/{sid}').status_code == 401
    assert client.put(f'/api/sessions/{sid}/answers', json={'answers': {}}).status_code == 401
    assert client.post(f'/api/sessions/{sid}/submit', json={'answers': {}}).status_code == 401
    assert client.get('/api/auth/me').status_code == 401


def test_forged_cookie_is_rejected(client):
    from src.modules.intake import auth
    client.cookies.set(auth.COOKIE_NAME, auth.sign_cookie(EMAIL, 'not-the-key'))
    assert client.get('/api/auth/me').status_code == 401


def test_another_founders_session_is_404(client):
    _signin(client)
    sid = _start(client)
    _signin(client, 'bob@beta.io')            # replaces the cookie
    assert client.get(f'/api/sessions/{sid}').status_code == 404
    assert client.put(f'/api/sessions/{sid}/answers', json={'answers': {}}).status_code == 404


def test_unapproved_addresses_get_the_same_answer_and_no_email(client):
    for email in ('eve@evil.com', 'someone@gmail.com', 'ada@acme-robotics.co'):
        r = client.post('/api/auth/request', json={'email': email})
        assert r.status_code == 200 and r.json() == {'sent': True}
    assert client.sent == []


def test_subdomain_and_exact_email_entries_are_approved(client):
    _signin(client, 'jo@eng.acme-robotics.com')
    _signin(client, 'Founder@Gmail.com')
    assert client.get('/api/auth/me').json()['email'] == 'founder@gmail.com'


def test_link_is_single_use(client):
    token = _signin(client)
    assert client.post('/api/auth/verify', json={'token': token}).status_code == 400
    assert client.post('/api/auth/verify', json={'token': 'x' * 43}).status_code == 400


def test_expired_link_is_rejected(client, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from src.modules.intake import auth
    client.post('/api/auth/request', json={'email': EMAIL})
    token = client.sent[-1][1].split('?t=', 1)[1]
    later = datetime.now(timezone.utc) + timedelta(minutes=auth.LINK_TTL_MIN + 1)
    monkeypatch.setattr(auth, '_now', lambda: later)
    assert client.post('/api/auth/verify', json={'token': token}).status_code == 400


def test_removing_a_domain_revokes_access(client):
    _signin(client)
    assert client.get('/api/auth/me').status_code == 200
    al.save(client.storage, ['beta.io'], [], actor='test')
    client.main._allow_cache.clear()
    assert client.get('/api/auth/me').status_code == 401


def test_second_start_resumes_the_draft(client):
    _signin(client)
    sid = _start(client)
    r = client.post('/api/sessions', json={'company_legal_name': 'Other', 'website': 'other.com'})
    assert r.json() == {'session_id': sid, 'resumed': True}
    assert client.get('/api/auth/me').json()['session_id'] == sid


def test_legacy_draft_is_claimed_by_its_contact_address(client, bm):
    from src.modules.intake import sessions as ss
    legacy = ss.create_session(bm, {'company_legal_name': 'Acme', 'website': 'acme-robotics.com',
                                    'contact_email': 'Ada@Acme-Robotics.com'})
    other = ss.create_session(bm, {'company_legal_name': 'X', 'website': 'x.com',
                                   'contact_email': 'x@x.com'})
    _signin(client)
    assert client.get(f'/api/sessions/{other["session_id"]}').status_code == 404
    assert client.get(f'/api/sessions/{legacy["session_id"]}').status_code == 200
    assert client.get('/api/auth/me').json()['session_id'] == legacy['session_id']


def test_link_rate_limit(client):
    for _ in range(5):
        client.post('/api/auth/request', json={'email': EMAIL})
    assert client.post('/api/auth/request', json={'email': EMAIL}).status_code == 429


def test_invalid_email_is_400(client):
    assert client.post('/api/auth/request', json={'email': 'not-an-email'}).status_code == 400
