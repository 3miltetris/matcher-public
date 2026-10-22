"""Tests for the MCP server's auth gate and audit log (no fastmcp needed)."""

import json

import pytest

from services.matcher_mcp import audit as au, auth


# ── auth gate ─────────────────────────────────────────────────────────────────

def test_is_bwco():
    assert auth.is_bwco('john@bwcoconsulting.com', True)
    assert not auth.is_bwco('john@bwcoconsulting.com', False)   # unverified
    assert not auth.is_bwco('outsider@gmail.com', True)          # wrong domain
    assert not auth.is_bwco('', True)


def test_claims_to_actor():
    a = auth.claims_to_actor({'email': 'A@BWCOConsulting.com', 'email_verified': True})
    assert a.allowed and a.email == 'A@BWCOConsulting.com'
    # string "true" (some IdPs send strings)
    assert auth.claims_to_actor({'email': 'x@bwcoconsulting.com',
                                 'email_verified': 'true'}).allowed
    # Google's userinfo-v2 field name `verified_email` is also accepted
    assert auth.claims_to_actor({'email': 'x@bwcoconsulting.com',
                                 'verified_email': True}).allowed
    # hd alone is NOT enough — the verified flag must be true
    assert not auth.claims_to_actor({'email': 'x@bwcoconsulting.com',
                                     'hd': 'bwcoconsulting.com'}).allowed
    assert not auth.claims_to_actor(None).allowed


def test_resolve_actor_dev(monkeypatch):
    monkeypatch.setenv('MCP_DEV', '1')
    a = auth.resolve_actor()
    assert a.allowed and a.email == 'dev@bwcoconsulting.com'


def test_require_actor_denies_outsider(monkeypatch):
    monkeypatch.delenv('MCP_DEV', raising=False)
    with pytest.raises(auth.AccessDenied):
        auth.require_actor(lambda: {'email': 'x@gmail.com', 'email_verified': True})
    # a verified BW&CO caller passes
    a = auth.require_actor(lambda: {'email': 'ok@bwcoconsulting.com',
                                    'email_verified': True})
    assert a.email == 'ok@bwcoconsulting.com'


def test_require_actor_fails_closed_on_bad_getter(monkeypatch):
    monkeypatch.delenv('MCP_DEV', raising=False)
    with pytest.raises(auth.AccessDenied):
        auth.require_actor(lambda: (_ for _ in ()).throw(RuntimeError('boom')))


# ── audit ─────────────────────────────────────────────────────────────────────

def test_audit_appends_and_hashes(storage):
    assert au.record(storage, email='john@bwcoconsulting.com', tool='whats_new',
                     args={'q': 'secret client name'}, outcome='ok')
    assert au.record(storage, email='john@bwcoconsulting.com', tool='search_grants',
                     args={'query': 'radar'}, outcome='ok', cost_usd=0.01)

    # both lines in one daily blob
    from datetime import datetime, timezone
    day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    raw = storage.bucket(au.BUCKET).blob(f'mcp-audit/{day}.jsonl').download_as_text()
    lines = [json.loads(l) for l in raw.strip().splitlines()]
    assert len(lines) == 2
    assert lines[0]['tool'] == 'whats_new' and lines[0]['email'].endswith('bwcoconsulting.com')
    # the raw argument never appears — only a hash
    assert 'secret client name' not in raw
    assert len(lines[0]['args_hash']) == 16
    assert lines[1]['cost_usd'] == 0.01


def test_audit_hash_stable():
    assert au.args_hash({'a': 1, 'b': 2}) == au.args_hash({'b': 2, 'a': 1})
    assert au.args_hash({'a': 1}) != au.args_hash({'a': 2})
