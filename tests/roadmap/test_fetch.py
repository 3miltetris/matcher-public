"""Deck/website fetch: the HubSpot-only Bearer guard and HTML rejection."""

import src.modules.roadmap.fetch as fetch


class _Resp:
    def __init__(self, status=200, ctype='text/plain', content=b'hello world text',
                 text='hello world text'):
        self.status_code = status
        self.headers = {'Content-Type': ctype}
        self.content = content
        self.text = text


class _FakeRequests:
    def __init__(self, resp):
        self.resp = resp
        self.calls = []

    def get(self, url, headers=None, timeout=None, allow_redirects=None):
        self.calls.append({'url': url, 'headers': headers or {}})
        return self.resp


def test_deck_bearer_only_to_hubspot_hosts(monkeypatch):
    fake = _FakeRequests(_Resp())
    monkeypatch.setattr(fetch, 'requests', fake)
    fetch.fetch_deck_text('https://api.hubspot.com/files/deck.pdf', 'SECRET')
    assert fake.calls[-1]['headers'].get('Authorization') == 'Bearer SECRET'


def test_deck_no_bearer_to_foreign_host(monkeypatch):
    fake = _FakeRequests(_Resp())
    monkeypatch.setattr(fetch, 'requests', fake)
    fetch.fetch_deck_text('https://evil.example.com/deck.pdf', 'SECRET')
    assert 'Authorization' not in fake.calls[-1]['headers']


def test_deck_rejects_html(monkeypatch):
    monkeypatch.setattr(fetch, 'requests',
                        _FakeRequests(_Resp(ctype='text/html', text='<html>')))
    text, reason = fetch.fetch_deck_text('https://hubspot.com/x', 't')
    assert text == '' and 'HTML' in reason


def test_deck_empty_url_is_noop(monkeypatch):
    assert fetch.fetch_deck_text('', 't') == ('', '')


def test_website_fetch_strips_tags(monkeypatch):
    html = '<html><head><style>x{}</style></head><body>We build <b>sensors</b>.</body></html>'
    monkeypatch.setattr(fetch, 'requests',
                        _FakeRequests(_Resp(ctype='text/html', text=html)))
    text, reason = fetch.fetch_website_text('acme.com')
    assert reason == ''
    assert 'We build sensors' in text
    assert '<' not in text and 'x{}' not in text
