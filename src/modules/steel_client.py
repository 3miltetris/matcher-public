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

import requests

API_BASE    = 'https://api.steel.dev/v1'
CONNECT_URL = 'wss://connect.steel.dev'

# Long enough for a human to find their authenticator, short enough that an
# abandoned window (tab closed mid-sign-in) stops billing and closes itself.
SESSION_TIMEOUT_MS = 15 * 60 * 1000
VIEWPORT           = {'width': 1280, 'height': 800}
_HTTP_TIMEOUT_S    = 60
_NAV_TIMEOUT_MS    = 45_000


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


def _open_url_sync(api_key: str, session_id: str, url: str) -> str:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(
            f'{CONNECT_URL}?apiKey={api_key}&sessionId={session_id}',
            timeout=_NAV_TIMEOUT_MS)
        try:
            ctx  = browser.contexts[0] if browser.contexts else browser.new_context()
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(url, timeout=_NAV_TIMEOUT_MS, wait_until='domcontentloaded')
            # The job replays the state with this same user agent: some sites
            # tie a session to the UA it was issued to.
            return page.evaluate('navigator.userAgent') or ''
        finally:
            # Disconnects only — the remote session (and the page) stay up for
            # the human in the live view.
            browser.close()


def open_url(api_key: str, session_id: str, url: str) -> str:
    """Navigate the session to ``url``; returns the browser's user agent.

    Run on a worker thread: Playwright's sync API refuses to start inside a
    thread that already has a running asyncio loop, and Streamlit's script
    thread is not guaranteed not to.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        try:
            return ex.submit(_open_url_sync, api_key, session_id, url).result(
                timeout=_NAV_TIMEOUT_MS / 1000 + 30)
        except Exception as e:                                # noqa: BLE001
            # Never echo the connect URL — it carries the API key.
            raise SteelError(f'could not open the login page in the remote browser '
                             f'({type(e).__name__})') from None


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
