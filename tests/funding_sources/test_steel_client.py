"""Unit tests for src/modules/steel_client.py — the browser-free parts.

The live behaviour (redirects, DNS failures, timeouts all still opening the
window) was verified against Steel on 2026-10-13; these pin the logic that made
users see "could not open the login page in the remote browser (Error)".
"""

import types

import src.modules.steel_client as steel


def test_reason_redacts_api_key_and_drops_method_prefix():
    exc = Exception('BrowserType.connect_over_cdp: WebSocket error: '
                    'wss://connect.steel.dev?apiKey=ste-SECRET123&sessionId=x 401\nCall log: ...')
    msg = steel._reason(exc)
    assert 'ste-SECRET123' not in msg
    assert 'apiKey=***' in msg
    assert not msg.startswith('BrowserType')
    assert '\n' not in msg


def test_reason_keeps_the_actual_navigation_error():
    exc = Exception('Page.goto: net::ERR_NAME_NOT_RESOLVED at https://www.aft3.af.mil/')
    assert steel._reason(exc) == 'net::ERR_NAME_NOT_RESOLVED at https://www.aft3.af.mil/'


def test_redirect_is_recognised_as_landing():
    # Verbatim shape of the LinkedIn failure users hit.
    exc = ('Page.goto: Navigation to "https://www.linkedin.com/company/cpecbrnd/posts/" is '
           'interrupted by another navigation to "https://www.linkedin.com/authwall?trk=bf"')
    assert steel._REDIRECTED in exc


def test_user_agent_reads_browser_level_cdp_not_the_page():
    class _CDP:
        def send(self, method):
            assert method == 'Browser.getVersion'
            return {'userAgent': 'Mozilla/5.0 Chrome/154'}

    browser = types.SimpleNamespace(new_browser_cdp_session=lambda: _CDP())
    assert steel._user_agent(browser) == 'Mozilla/5.0 Chrome/154'


def test_user_agent_failure_is_blank_not_raised():
    def boom():
        raise RuntimeError('cdp gone')
    assert steel._user_agent(types.SimpleNamespace(new_browser_cdp_session=boom)) == ''
