"""Unit tests for src/modules/email_otp.py — code extraction and the single
mailbox check, against a fake Gmail service (no network).

    python -m pytest tests/funding_sources
"""

import base64
import time

import src.modules.email_otp as eo


# -- extract_code -----------------------------------------------------------

def test_extract_prefers_code_near_keyword():
    assert eo.extract_code('Your verification code is 481920. Ignore 2026.') == '481920'


def test_extract_plain_number():
    assert eo.extract_code('334455') == '334455'


def test_extract_keyword_wins_over_stray_year():
    # The stray 2026 comes first, but the real code sits next to "code:".
    assert eo.extract_code('(c) 2026 Site Inc.\nYour login code: 778812') == '778812'


def test_extract_none_without_digits():
    assert eo.extract_code('No numbers here at all') is None


def test_extract_custom_regex_alphanumeric():
    assert eo.extract_code('Code: 9FJ2KQ now', regex=r'\b[A-Z0-9]{6}\b') == '9FJ2KQ'


# -- fetch_code against a fake Gmail service --------------------------------

class _Exec:
    def __init__(self, val):
        self._val = val

    def execute(self):
        return self._val


class FakeMessages:
    def __init__(self, listing, by_id):
        self._listing, self._by_id = listing, by_id
        self.last_q = None

    def list(self, userId=None, q=None, maxResults=None):
        self.last_q = q
        return _Exec(self._listing)

    def get(self, userId=None, id=None, format=None):
        return _Exec(self._by_id.get(id, {}))


class FakeUsers:
    def __init__(self, messages):
        self._messages = messages

    def messages(self):
        return self._messages

    def getProfile(self, userId=None):
        return _Exec({'emailAddress': 'funding-2fa@example.com'})


class FakeGmail:
    def __init__(self, listing, by_id):
        self._users = FakeUsers(FakeMessages(listing, by_id))

    def users(self):
        return self._users


def _plain(text):
    data = base64.urlsafe_b64encode(text.encode()).decode()
    return {'mimeType': 'text/plain', 'body': {'data': data}}


def test_fetch_code_reads_plain_body():
    now = int(time.time())
    msg = {'internalDate': str(now * 1000),
           'payload': _plain('Your code is 334455'), 'snippet': ''}
    g = FakeGmail({'messages': [{'id': '1'}]}, {'1': msg})
    assert eo.fetch_code(g, sender='no-reply@site.org', after_epoch=now - 10) == '334455'


def test_fetch_code_none_when_no_gmail():
    assert eo.fetch_code(None, sender='x@y.com', after_epoch=1) is None


def test_fetch_code_skips_stale_message():
    now = int(time.time())
    old = {'internalDate': str((now - 10_000) * 1000),
           'payload': _plain('code 111111'), 'snippet': ''}
    g = FakeGmail({'messages': [{'id': '1'}]}, {'1': old})
    assert eo.fetch_code(g, after_epoch=now) is None


def test_fetch_code_empty_listing():
    g = FakeGmail({'messages': []}, {})
    assert eo.fetch_code(g, after_epoch=int(time.time())) is None
