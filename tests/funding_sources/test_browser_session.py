"""Unit tests for src/modules/browser_session.py and the agent's use of a saved
sign-in (browser_agent.research_site(session=...)).

Run from the repo root so `src` is importable:
    python -m pytest tests/funding_sources
"""

import asyncio
import json
import types

import pytest

import src.modules.browser_agent as ba
import src.modules.browser_session as bs
from tests.funding_sources.test_source_credentials import FakeSM

# Shaped exactly like Steel's GET /v1/sessions/{id}/context (live API, 2026-10-13).
STEEL_CTX = {
    'cookies': [
        {'name': 'marketplace_access_token', 'value': 'SECRET-TOKEN-VALUE',
         'domain': 'marketplace.gocolosseum.org', 'path': '/', 'secure': True,
         'httpOnly': True, 'size': 40, 'expires': 1893456000, 'session': False,
         'priority': 'Medium', 'sameSite': 'Lax', 'sourceScheme': 'Secure',
         'sourcePort': 443},
        {'name': 'sid', 'value': 'abc', 'domain': '.gocolosseum.org', 'path': '/',
         'secure': True, 'httpOnly': False, 'size': 6, 'expires': -1, 'session': True,
         'priority': 'Medium', 'sourceScheme': 'Secure', 'sourcePort': 443},
        {'name': '_ga', 'value': 'GA1.1.123', 'domain': '.google-analytics.com',
         'path': '/', 'secure': False, 'httpOnly': False, 'size': 12,
         'expires': 1893456000, 'session': False},
    ],
    'localStorage': {
        'marketplace.gocolosseum.org': {'auth': '{"jwt":"x"}'},
        'www.googletagmanager.com':    {'gtm': '1'},
    },
    'sessionStorage': {'marketplace.gocolosseum.org': {'tmp': '1'}},
    'indexedDB': {},
}


def test_steel_context_converts_to_playwright_storage_state():
    state = bs.steel_context_to_storage_state(STEEL_CTX)
    by_name = {c['name']: c for c in state['cookies']}
    assert by_name['marketplace_access_token']['expires'] == 1893456000.0
    assert by_name['marketplace_access_token']['sameSite'] == 'Lax'
    assert by_name['marketplace_access_token']['httpOnly'] is True
    # Session cookie -> -1; unknown sameSite is omitted (Playwright default).
    assert by_name['sid']['expires'] == -1
    assert 'sameSite' not in by_name['sid']
    # CDP-only fields are dropped.
    assert 'priority' not in by_name['sid'] and 'size' not in by_name['sid']
    origins = {o['origin']: o['localStorage'] for o in state['origins']}
    assert origins['https://marketplace.gocolosseum.org'] == [
        {'name': 'auth', 'value': '{"jwt":"x"}'}]
    # sessionStorage is not part of Playwright storage_state.
    assert set(state) == {'cookies', 'origins'}


def test_trim_keeps_only_site_domains():
    state = bs.steel_context_to_storage_state(STEEL_CTX)
    trimmed = bs.trim_storage_state(state, ['https://marketplace.gocolosseum.org/challenges'])
    assert {c['name'] for c in trimmed['cookies']} == {'marketplace_access_token', 'sid'}
    assert [o['origin'] for o in trimmed['origins']] == ['https://marketplace.gocolosseum.org']


def test_registrable_domain():
    assert bs.registrable_domain('https://app.portal.example.org/x') == 'example.org'
    assert bs.registrable_domain('.gocolosseum.org') == 'gocolosseum.org'
    assert bs.registrable_domain('') == ''


def test_encode_decode_roundtrip_and_compresses():
    state = bs.steel_context_to_storage_state(STEEL_CTX)
    blob = bs.encode_state(state)
    assert bs.decode_state(blob) == state
    assert bs.decode_state('not base64 gzip') is None


def test_prepare_state_only_trims_when_large():
    small = bs.steel_context_to_storage_state(STEEL_CTX)
    out, trimmed = bs.prepare_state(small, ['https://marketplace.gocolosseum.org'])
    assert not trimmed and out == small
    big = {'cookies': small['cookies'] + [
        # Random, like real tracker ids: a repeated value gzips to nothing.
        {'name': f't{i}', 'value': __import__('os').urandom(200).hex(),
         'domain': f'tracker{i}.com',
         'path': '/', 'expires': -1, 'httpOnly': False, 'secure': False}
        for i in range(60)], 'origins': small['origins']}
    out, trimmed = bs.prepare_state(big, ['https://marketplace.gocolosseum.org'])
    assert trimmed
    assert all('gocolosseum' in c['domain'] for c in out['cookies'])


def test_missing_secret_is_empty():
    assert bs.load_sessions(FakeSM()) == {}


def test_set_load_session_for_roundtrip():
    sm = FakeSM()
    state = bs.steel_context_to_storage_state(STEEL_CTX)
    bs.set_session(sm, '73471d', state, user_agent='UA/1', actor='john@bwcoconsulting.com')
    data = bs.load_sessions(sm)
    assert bs.configured_ids(data) == {'73471d'}
    got = bs.session_for(data, '73471d')
    assert got == {'storage_state': state, 'user_agent': 'UA/1'}
    assert bs.session_for(data, 'nope') is None


def test_summary_never_contains_state():
    sm = FakeSM()
    bs.set_session(sm, 's1', bs.steel_context_to_storage_state(STEEL_CTX))
    rendered = json.dumps(bs.safe_summary(bs.load_sessions(sm)))
    assert 'SECRET-TOKEN-VALUE' not in rendered
    assert 'state_gz' not in rendered
    assert 'marketplace.gocolosseum.org' in rendered   # hosts are fine to show


def test_empty_state_refused():
    with pytest.raises(ValueError):
        bs.set_session(FakeSM(), 's1', {'cookies': [], 'origins': []})


def test_oversized_store_refused_before_write():
    sm = FakeSM()
    huge = {'cookies': [{'name': f'c{i}', 'value': __import__('os').urandom(40).hex(),
                         'domain': 'x.org', 'path': '/', 'expires': -1,
                         'httpOnly': False, 'secure': False} for i in range(1500)],
            'origins': []}
    with pytest.raises(bs.SessionTooLarge):
        bs.set_session(sm, 's1', huge)
    assert sm.versions == []   # nothing written


def test_delete_session():
    sm = FakeSM()
    bs.set_session(sm, 'a', bs.steel_context_to_storage_state(STEEL_CTX))
    bs.set_session(sm, 'b', bs.steel_context_to_storage_state(STEEL_CTX))
    bs.delete_session(sm, 'a')
    assert bs.configured_ids(bs.load_sessions(sm)) == {'b'}


# -- The agent with a saved sign-in -----------------------------------------

class _FakeContext:
    async def new_page(self):
        raise AssertionError('no page should be needed in this test')

    async def close(self):
        pass


class _FakeBrowser:
    def __init__(self):
        self.context_kwargs = None

    async def new_context(self, **kwargs):
        self.context_kwargs = kwargs
        return _FakeContext()


class _FakeAnth:
    """Answers the first turn with report_findings, recording what was sent."""

    def __init__(self, requires_login: bool):
        self.sent = []
        self.requires_login = requires_login
        self.messages = self

    async def create(self, **kwargs):
        self.sent.append(json.dumps(kwargs['messages'], default=str))
        call = types.SimpleNamespace(
            type='tool_use', name='report_findings', id='t1',
            input={'opportunities': [], 'has_api': False,
                   'requires_login': self.requires_login, 'notes': ''})
        usage = types.SimpleNamespace(input_tokens=1, output_tokens=1,
                                      cache_read_input_tokens=0,
                                      cache_creation_input_tokens=0)
        return types.SimpleNamespace(content=[call], usage=usage)


def _run(anth, browser, session):
    site = {'source_id': '73471d', 'name': 'gocolosseum',
            'url': 'https://marketplace.gocolosseum.org', 'max_pages': 5}
    return asyncio.run(ba.research_site(anth, browser, site, session=session))


def test_saved_session_seeds_context_and_stays_out_of_the_conversation():
    state = bs.steel_context_to_storage_state(STEEL_CTX)
    anth, browser = _FakeAnth(requires_login=False), _FakeBrowser()
    result = _run(anth, browser, {'storage_state': state, 'user_agent': 'UA/steel'})
    assert browser.context_kwargs['storage_state'] == state
    assert browser.context_kwargs['user_agent'] == 'UA/steel'
    assert result['session_used'] is True and result['ok'] is True
    assert result['agent_requires_login'] is False
    sent = ''.join(anth.sent)
    assert 'SECRET-TOKEN-VALUE' not in sent
    assert 'already signed in' in sent


def test_agent_wall_with_session_is_reported_as_agent_verdict():
    state = bs.steel_context_to_storage_state(STEEL_CTX)
    result = _run(_FakeAnth(requires_login=True), _FakeBrowser(),
                  {'storage_state': state, 'user_agent': ''})
    assert result['session_used'] and result['agent_requires_login']


def test_no_session_leaves_context_unchanged():
    anth, browser = _FakeAnth(requires_login=False), _FakeBrowser()
    result = _run(anth, browser, None)
    assert 'storage_state' not in browser.context_kwargs
    assert 'MatcherBot' in browser.context_kwargs['user_agent']
    assert result['session_used'] is False
    assert 'already signed in' not in ''.join(anth.sent)


def test_real_chromium_accepts_converted_state():
    """The converted dict must be something Playwright actually loads."""
    pw_api = pytest.importorskip('playwright.async_api')
    state = bs.steel_context_to_storage_state(STEEL_CTX)

    async def go():
        async with pw_api.async_playwright() as p:
            try:
                browser = await p.chromium.launch(headless=True)
            except Exception as e:                            # noqa: BLE001
                pytest.skip(f'chromium not installed: {e}')
            ctx = await browser.new_context(storage_state=state)
            cookies = await ctx.cookies()
            out = await ctx.storage_state()
            await browser.close()
            return cookies, out

    cookies, out = asyncio.run(go())
    assert {c['name'] for c in cookies} >= {'marketplace_access_token', 'sid'}
    origins = {o['origin'] for o in out['origins']}
    assert 'https://marketplace.gocolosseum.org' in origins
