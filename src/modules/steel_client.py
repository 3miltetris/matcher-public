"""
steel_client.py — the Steel remote-browser calls behind the Funding Sources
view's interactive sign-in window (Stage 10).

Flow: ``create_session`` a headful browser → ``open_url`` navigates it to the
site's login page → the view embeds ``live_view_url`` in an iframe and an admin
signs in by hand (authenticator app, SMS, SSO, CAPTCHA — whatever the site
asks) → ``get_context`` pulls the authenticated cookies + localStorage →
``release`` ends the session. ``browser_session`` converts and stores the state;
the deep-research job then replays it in its own headless Chromium, so Steel is
used for the few minutes of a sign-in and **never** at scrape time.

Wire formats were verified against the live API on 2026-10-13: ``POST
/v1/sessions`` (201) returns ``id`` + ``debugUrl``; the debug URL is headful,
honours ``debugConfig.interactive``, and sends no ``X-Frame-Options`` / CSP, so
it embeds; ``GET /v1/sessions/{id}/context`` returns cookies as CDP objects and
localStorage keyed by host; ``POST /v1/sessions/{id}/release`` → 200.
Disconnecting a CDP client leaves the remote session running.

**The debug URL is unauthenticated** — the session UUID is the only thing
protecting it, and anyone holding it can watch or drive the browser while the
admin types a password into it. So it is never logged, it is shown only inside
the IAP-protected admin panel, and the session is released as soon as the
state is captured (or after ``SESSION_TIMEOUT_MS`` at the latest).

Streamlit-free; the API key is passed in by the caller (``st.secrets``).
"""

import concurrent.futures
import re

import requests

API_BASE    = 'https://api.steel.dev/v1'
CONNECT_URL = 'wss://connect.steel.dev'

# Long enough for a human to find their authenticator, short enough that an
# abandoned window (tab closed mid-sign-in) stops billing and closes itself.
SESSION_TIMEOUT_MS = 15 * 60 * 1000
VIEWPORT           = {'width': 1280, 'height': 800}
_HTTP_TIMEOUT_S    = 60
_CONNECT_TIMEOUT_MS = 45_000
# Navigation waits only for the first response ('commit'): the human watches the
# rest of the page load in the live view, and a slow .mil page used to hold the
# admin at a spinner for the full timeout and then fail.
_NAV_TIMEOUT_MS    = 25_000


class SteelError(RuntimeError):
    pass


def _headers(api_key: str) -> dict:
    if not api_key:
        raise SteelError('no Steel API key is configured (st.secrets["steel_api_key"])')
    return {'steel-api-key': api_key, 'Content-Type': 'application/json'}


def _check(resp, what: str):
    if resp.status_code >= 300:
        # The body is Steel's error message; it never contains the API key.
        raise SteelError(f'{what} failed: HTTP {resp.status_code} {resp.text[:300]}')
    try:
        return resp.json()
    except ValueError:
        return {}


def create_session(api_key: str) -> dict:
    """Start an interactive headful session. Returns ``{id, debug_url}``."""
    resp = requests.post(f'{API_BASE}/sessions', headers=_headers(api_key), json={
        'timeout':     SESSION_TIMEOUT_MS,
        'debugConfig': {'interactive': True},
        'dimensions':  VIEWPORT,
    }, timeout=_HTTP_TIMEOUT_S)
    data = _check(resp, 'creating a Steel session')
    if not data.get('id') or not data.get('debugUrl'):
        raise SteelError('Steel returned a session without an id / debugUrl')
    return {'id': data['id'], 'debug_url': data['debugUrl']}


def live_view_url(debug_url: str) -> str:
    """The embeddable, human-drivable view of a session."""
    sep = '&' if '?' in debug_url else '?'
    return f'{debug_url}{sep}interactive=true&showControls=true'


def _redact(text) -> str:
    """Strip the API key out of anything that may echo the CDP connect URL."""
    return re.sub(r'(apiKey=)[^&\s"\']+', r'\1***', str(text))


def _reason(exc: Exception) -> str:
    """One readable line from a Playwright error, key redacted."""
    first = _redact(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return re.sub(r'^(Page|BrowserType)\.\w+:\s*', '', first)[:240]


# A goto "interrupted by another navigation" means the site redirected while
# loading — LinkedIn and lift.technology bounce straight to their own login
# wall. The browser did land somewhere, so it is a success, not an error.
_REDIRECTED = 'interrupted by another navigation'


def _user_agent(browser) -> str:
    """The UA the browser actually sends, read at browser level.

    Not ``page.evaluate('navigator.userAgent')``: evaluate has no timeout and
    waits for an execution context, so on a page still loading after a goto
    timeout it hung the sign-in window for 271s (jacks.jpeocbrnd.army.mil).
    Not the ``userAgent`` field of Steel's session API either: measured
    2026-10-13 it reports Chrome/128 while the browser sends Chrome/154, and
    replaying the wrong one presents a different browser to the site.
    ``Browser.getVersion`` matched ``navigator.userAgent`` exactly.
    """
    try:
        return browser.new_browser_cdp_session().send('Browser.getVersion').get(
            'userAgent', '')
    except Exception:                                         # noqa: BLE001
        return ''


def _open_url_sync(api_key: str, session_id: str, url: str) -> dict:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        # Only a failed connect is fatal — without it there is no browser to show.
        browser = p.chromium.connect_over_cdp(
            f'{CONNECT_URL}?apiKey={api_key}&sessionId={session_id}',
            timeout=_CONNECT_TIMEOUT_MS)
        try:
            ctx  = browser.contexts[0] if browser.contexts else browser.new_context()
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            warning = ''
            try:
                page.goto(url, timeout=_NAV_TIMEOUT_MS, wait_until='commit')
            except Exception as e:                            # noqa: BLE001
                # Navigation is best-effort: a redirect, a slow page, a dead DNS
                # name or a reset must not cost the admin the window — they can
                # carry on from wherever the browser is, or go elsewhere.
                if _REDIRECTED not in str(e):
                    warning = _reason(e)
            # The job replays the state with this same user agent: some sites
            # tie a session to the UA it was issued to.
            return {'user_agent': _user_agent(browser),
                    'url': page.url, 'warning': warning}
        finally:
            # Disconnects only — the remote session (and the page) stay up for
            # the human in the live view.
            browser.close()


def open_url(api_key: str, session_id: str, url: str) -> dict:
    """Point the session at ``url``. Returns ``{user_agent, url, warning}``.

    Never raises for a navigation problem — those come back as ``warning`` (and
    ``url`` is where the browser actually is) so the window still opens. Raises
    ``SteelError`` only when the remote browser cannot be reached at all.

    Run on a worker thread: Playwright's sync API refuses to start inside a
    thread that already has a running asyncio loop, and Streamlit's script
    thread is not guaranteed not to.
    """
    # Not a `with` block: its exit joins the worker, so a stuck worker would
    # hold the page past the timeout below. A worker abandoned here unsticks
    # itself — the caller releases the session, which drops its CDP socket.
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(_open_url_sync, api_key, session_id, url).result(
            timeout=(_CONNECT_TIMEOUT_MS + _NAV_TIMEOUT_MS) / 1000 + 15)
    except concurrent.futures.TimeoutError:
        raise SteelError('the remote browser did not respond in time') from None
    except Exception as e:                                    # noqa: BLE001
        # The reason is shown, with the key redacted — the bare class name
        # ("Error") this used to show was undiagnosable.
        raise SteelError(f'could not reach the remote browser: {_reason(e)}') from None
    finally:
        ex.shutdown(wait=False)


def get_context(api_key: str, session_id: str) -> dict:
    resp = requests.get(f'{API_BASE}/sessions/{session_id}/context',
                        headers=_headers(api_key), timeout=_HTTP_TIMEOUT_S)
    return _check(resp, 'reading the session state')


def session_status(api_key: str, session_id: str) -> str:
    """``live`` while usable; anything else means it timed out or was released."""
    try:
        resp = requests.get(f'{API_BASE}/sessions/{session_id}',
                            headers=_headers(api_key), timeout=_HTTP_TIMEOUT_S)
        return str(_check(resp, 'reading the session').get('status') or '')
    except Exception:                                         # noqa: BLE001
        return ''


def release(api_key: str, session_id: str) -> None:
    """Best-effort: an unreleased session still ends at its timeout."""
    try:
        requests.post(f'{API_BASE}/sessions/{session_id}/release',
                      headers=_headers(api_key), timeout=_HTTP_TIMEOUT_S)
    except Exception:                                         # noqa: BLE001
        pass
