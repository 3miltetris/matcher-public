"""
roadmap.fetch — pull text the profile builder can read: the founder's pitch-deck
PDF (from HubSpot) and, on the Instantly path, the company homepage.

Both are best-effort: a failure returns ('', reason), never raises, because a
missing deck or an unreachable site must still let the description-only roadmap
build.
"""

import re

import requests

from src.modules import doc_extract

_UA        = 'MatcherBot/1.0 (+https://www.bwcoconsulting.com)'
_DECK_CAP  = 20000
_SITE_CAP  = 12000
_HS_HOSTS  = ('hubspot', 'hubapi', 'hubspotusercontent')


def _is_hubspot(url: str) -> bool:
    return any(h in url.lower() for h in _HS_HOSTS)


def fetch_deck_text(url: str, hs_token: str | None, *, cap: int = _DECK_CAP) -> tuple[str, str]:
    """(text, reason). The HubSpot Bearer token is sent ONLY to HubSpot hosts,
    so a non-HubSpot file URL can never leak it. Rejects an HTML body (an auth
    redirect), the same trap the resume importer documents."""
    if not url:
        return '', ''
    headers = {'User-Agent': _UA}
    if hs_token and _is_hubspot(url):
        headers['Authorization'] = f'Bearer {hs_token}'
    try:
        r = requests.get(url, headers=headers, timeout=25, allow_redirects=True)
    except Exception as e:
        return '', f'fetch error: {type(e).__name__}'
    if r.status_code != 200:
        return '', f'HTTP {r.status_code}'
    if 'text/html' in r.headers.get('Content-Type', '').lower():
        return '', 'got HTML (auth redirect?)'
    text, ftype = doc_extract.extract_text(r.content, url, r.headers.get('Content-Type', ''))
    return (text[:cap], '') if text else ('', f'no text extracted ({ftype})')


_TAG   = re.compile(r'<(script|style)[^>]*>.*?</\1>', re.S | re.I)
_ANGLE = re.compile(r'<[^>]+>')
_WS    = re.compile(r'\s+')


def fetch_website_text(url: str, *, cap: int = _SITE_CAP) -> tuple[str, str]:
    """(text, reason) for a company homepage. A light requests fetch + tag strip,
    deliberately NOT Playwright: that would add a headless Chromium (~300 MB) to a
    public image for marginal gain on most marketing sites. JS-only sites yield
    little here; wire a Playwright scraper later if the Instantly path needs it."""
    if not url:
        return '', ''
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    try:
        r = requests.get(url, headers={'User-Agent': _UA}, timeout=20, allow_redirects=True)
    except Exception as e:
        return '', f'fetch error: {type(e).__name__}'
    if r.status_code != 200 or 'text/html' not in r.headers.get('Content-Type', '').lower():
        return '', f'HTTP {r.status_code} / {r.headers.get("Content-Type", "")[:40]}'
    body = _TAG.sub(' ', r.text)
    text = _WS.sub(' ', _ANGLE.sub(' ', body)).strip()
    return (text[:cap], '') if text else ('', 'no text extracted')
