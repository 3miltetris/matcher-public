"""Unit tests for src/modules/source_credentials.py.

Run from the repo root so `src` is importable:
    python -m pytest tests/funding_sources
"""

import json
import types

import src.modules.source_credentials as sc


class NotFound(Exception):
    """Stands in for google.api_core.exceptions.NotFound — source_credentials
    matches it by class name, so the name here must be exactly NotFound."""


class FakeSM:
    """In-memory Secret Manager: a list of payload-bytes versions."""

    def __init__(self):
        self.versions = []

    def access_secret_version(self, request):
        if not self.versions:
            raise NotFound('no enabled version')
        data = self.versions[-1]
        return types.SimpleNamespace(payload=types.SimpleNamespace(data=data))

    def add_secret_version(self, request):
        self.versions.append(request['payload']['data'])
        return types.SimpleNamespace(name=f'v{len(self.versions)}')


def test_missing_secret_is_empty_not_error():
    assert sc.load_credentials(FakeSM()) == {}


def test_set_get_roundtrip():
    sm = FakeSM()
    sc.set_credential(sm, 'a3f9c1', username='team@x.com', password='s3cr3t',
                      login_url='https://x.com/login', actor='john@bwcoconsulting.com')
    data = sc.load_credentials(sm)
    assert set(data) == {'a3f9c1'}
    entry = data['a3f9c1']
    assert entry['username'] == 'team@x.com'
    assert entry['password'] == 's3cr3t'
    assert entry['login_url'] == 'https://x.com/login'
    assert entry['updated_by'] == 'john@bwcoconsulting.com'
    assert entry['updated_at']  # stamped


def test_second_site_merges_not_replaces():
    sm = FakeSM()
    sc.set_credential(sm, 'a', username='u1', password='p1')
    sc.set_credential(sm, 'b', username='u2', password='p2')
    data = sc.load_credentials(sm)
    assert set(data) == {'a', 'b'}


def test_safe_summary_omits_passwords():
    sm = FakeSM()
    sc.set_credential(sm, 'a', username='u1', password='p1', login_url='http://a/login')
    rows = sc.safe_summary(sc.load_credentials(sm))
    assert len(rows) == 1
    row = rows[0]
    assert row['username'] == 'u1'
    assert row['login_url'] == 'http://a/login'
    assert 'password' not in row
    # And nothing in the serialized summary leaks the password.
    assert 'p1' not in json.dumps(rows)


def test_configured_ids_and_credential_for():
    sm = FakeSM()
    sc.set_credential(sm, 'a', username='u1', password='p1')
    data = sc.load_credentials(sm)
    assert sc.configured_ids(data) == {'a'}
    assert sc.credential_for(data, 'a')['password'] == 'p1'
    assert sc.credential_for(data, 'nope') is None


def test_delete_credential():
    sm = FakeSM()
    sc.set_credential(sm, 'a', username='u1', password='p1')
    sc.set_credential(sm, 'b', username='u2', password='p2')
    sc.delete_credential(sm, 'a', actor='john@bwcoconsulting.com')
    data = sc.load_credentials(sm)
    assert set(data) == {'b'}


def test_set_requires_username_and_password():
    sm = FakeSM()
    for bad in [dict(username='', password='p'), dict(username='u', password='')]:
        try:
            sc.set_credential(sm, 'a', **bad)
            assert False, 'expected ValueError'
        except ValueError:
            pass
