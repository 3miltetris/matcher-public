"""Tests for PageSession.log_in — the credentialed login step of the browser
agent. These use a fake Playwright page (no real browser), so they run anywhere
bs4 is importable:

    python -m pytest tests/funding_sources

The point under test is the control flow and the security guarantee: the bound
password is filled into the page but never appears in the tool-result string the
model would see.
"""

import asyncio

from src.modules.browser_agent import PageSession

_PASSWORD = 's3cr3t-pw'
_USERNAME = 'team@bwcoconsulting.com'


class FakeLocator:
    def __init__(self, page, kind, count=1):
        self.page, self.kind, self._count = page, kind, count

    @property
    def first(self):
        return self

    async def count(self):
        return self._count

    async def fill(self, value, timeout=None):
        self.page.filled[self.kind] = value

    async def click(self, timeout=None):
        self.page.clicked.append(self.kind)

    async def press(self, key, timeout=None):
        self.page.pressed.append((self.kind, key))


class FakePage:
    def __init__(self, html, *, have_password=True, have_user=True, have_submit=True,
                 have_otp=True):
        self._html = html
        self.url = 'https://site.example/login'
        self._title = 'Members'
        self.filled, self.clicked, self.pressed, self.loadstates = {}, [], [], []
        self.have_password, self.have_user, self.have_submit, self.have_otp = (
            have_password, have_user, have_submit, have_otp)

    def locator(self, sel):
        s = sel.lower()
        if 'password' in s:
            return FakeLocator(self, 'password', 1 if self.have_password else 0)
        if 'submit' in s:
            return FakeLocator(self, 'submit', 1 if self.have_submit else 0)
        if any(t in s for t in ('one-time-code', 'otp', 'code', 'token', 'numeric', 'tel"')):
            return FakeLocator(self, 'otp', 1 if self.have_otp else 0)
        if any(t in s for t in ('email', 'user', 'text', 'autocomplete')):
            return FakeLocator(self, 'user', 1 if self.have_user else 0)
        return FakeLocator(self, 'other', 0)

    def get_by_label(self, text, exact=False):
        return FakeLocator(self, 'label', 0)

    def get_by_placeholder(self, text, exact=False):
        return FakeLocator(self, 'placeholder', 0)

    def get_by_role(self, role, name=None):
        return FakeLocator(self, 'role', 0)

    async def content(self):
        return self._html

    async def title(self):
        return self._title

    async def wait_for_load_state(self, state=None, timeout=None):
        self.loadstates.append(state)


def _session(credentials, page=None, code_fetcher=None):
    s = PageSession(context=None, max_pages=12, credentials=credentials,
                    code_fetcher=code_fetcher)
    s.page = page
    return s


# Post-login banner echoes the username (fine) but never the password.
_LOGGED_IN_HTML = (
    f'<html><body><p>Signed in as {_USERNAME}</p>'
    '<a href="/opportunities">Open opportunities</a></body></html>'
)


def test_login_fills_password_and_never_leaks_it():
    page = FakePage(_LOGGED_IN_HTML)
    s = _session({'username': _USERNAME, 'password': _PASSWORD, 'login_url': ''}, page)
    out = asyncio.run(s.log_in())

    assert 'Submitted the login form' in out
    # The password was typed into the field...
    assert page.filled.get('password') == _PASSWORD
    assert page.filled.get('user') == _USERNAME
    assert page.clicked == ['submit']
    # ...but it must never appear in what the model would see.
    assert _PASSWORD not in out
    # The (non-secret) username in the banner is fine to surface.
    assert _USERNAME in out
    assert s.pages_seen == 1


def test_login_without_credentials_errors():
    s = _session(None, FakePage(_LOGGED_IN_HTML))
    out = asyncio.run(s.log_in())
    assert out.startswith('ERROR') and 'no credentials' in out
    assert 'password' not in s.filled if hasattr(s, 'filled') else True


def test_login_reports_when_no_password_field():
    page = FakePage('<html><body>No form here</body></html>', have_password=False)
    s = _session({'username': _USERNAME, 'password': _PASSWORD, 'login_url': ''}, page)
    out = asyncio.run(s.log_in())
    assert out.startswith('ERROR') and 'password field' in out
    assert _PASSWORD not in out


def test_login_falls_back_to_enter_when_no_submit_button():
    page = FakePage(_LOGGED_IN_HTML, have_submit=False)
    s = _session({'username': _USERNAME, 'password': _PASSWORD, 'login_url': ''}, page)
    out = asyncio.run(s.log_in())
    assert 'Submitted the login form' in out
    assert ('password', 'Enter') in page.pressed
    assert _PASSWORD not in out


# -- get_email_code ---------------------------------------------------------

_CODE = '445566'
# Post-2FA page never echoes the code back.
_VERIFIED_HTML = '<html><body><p>Verified.</p><a href="/opps">Opportunities</a></body></html>'


def test_get_email_code_enters_code_and_never_leaks_it():
    page = FakePage(_VERIFIED_HTML)
    s = _session({'username': _USERNAME, 'password': _PASSWORD}, page,
                 code_fetcher=lambda since: _CODE)
    out = asyncio.run(s.get_email_code())
    assert 'Entered the emailed 2FA code' in out
    assert page.filled.get('otp') == _CODE      # typed into the field
    assert page.clicked == ['submit']
    assert _CODE not in out                      # never surfaced to the model


def test_get_email_code_without_fetcher_errors():
    page = FakePage(_VERIFIED_HTML)
    s = _session({'username': _USERNAME, 'password': _PASSWORD}, page, code_fetcher=None)
    out = asyncio.run(s.get_email_code())
    assert out.startswith('ERROR') and '2FA mailbox' in out


def test_get_email_code_times_out_cleanly(monkeypatch):
    import src.modules.browser_agent as ba
    monkeypatch.setattr(ba, '_OTP_POLL_TOTAL_S', 0.15)
    monkeypatch.setattr(ba, '_OTP_POLL_INTERVAL_S', 0.03)
    page = FakePage(_VERIFIED_HTML)
    s = _session({'username': _USERNAME, 'password': _PASSWORD}, page,
                 code_fetcher=lambda since: None)   # code never arrives
    out = asyncio.run(s.get_email_code())
    assert out.startswith('ERROR') and 'no 2FA code' in out
    assert 'otp' not in page.filled
