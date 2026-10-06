"""
auth.py — magic-link sign-in for the DD intake form.

A founder asks for a link; if their address is on the allowlist
(allowlist.py) we email a single-use token valid for LINK_TTL_MIN minutes.
Only the token's sha256 is stored (`intake/auth/{hash}.json`), so a bucket
read never yields a usable link. Exchanging the token sets a signed cookie
carrying the email and an expiry; every session route requires it.

The token is consumed by a POST from the page, never by the GET the email
link performs: corporate mail scanners (Safe Links, Mimecast) fetch links
before the founder clicks them, and would otherwise burn every token.
"""

import base64
import hashlib
import hmac
import json
import re
import secrets
from datetime import datetime, timedelta, timezone

from google.api_core.exceptions import PreconditionFailed

TOKEN_PREFIX  = 'intake/auth/'
LINK_TTL_MIN  = 30
COOKIE_NAME   = 'intake_auth'
COOKIE_DAYS   = 14                      # matches sessions.EXPIRY_DAYS

# token_urlsafe(32) → 43 chars, 256 bits.
_TOKEN_RE = re.compile(r'^[A-Za-z0-9_-]{43}$')


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def token_blob(token: str) -> str:
    return f'{TOKEN_PREFIX}{_sha(token)}.json'


def issue_link_token(bm, email: str) -> str:
    token = secrets.token_urlsafe(32)
    now = _now()
    bm.upload_json(token_blob(token), {
        'email': email,
        'created_at': now.isoformat(timespec='seconds'),
        'expires_at': (now + timedelta(minutes=LINK_TTL_MIN)).isoformat(timespec='seconds'),
        'used': False,
    }, if_generation_match=0)
    return token


def consume_link_token(bm, token: str) -> str | None:
    """The email the token was issued to, or None when it is malformed,
    unknown, expired or already used. Single use is enforced with the blob
    generation, so two racing clicks cannot both succeed."""
    if not _TOKEN_RE.match(str(token or '')):
        return None
    blob = token_blob(token)
    rec, gen = bm.download_json(blob)
    if not rec or rec.get('used'):
        return None
    try:
        if _now() > datetime.fromisoformat(rec['expires_at']):
            return None
    except Exception:
        return None
    rec['used'] = True
    rec['used_at'] = _now().isoformat(timespec='seconds')
    try:
        bm.upload_json(blob, rec, if_generation_match=gen)
    except PreconditionFailed:
        return None
    return rec.get('email') or None


# ── Cookie ───────────────────────────────────────────────────────────────────

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))


def _sig(payload: str, secret: str) -> str:
    return _b64(hmac.new(secret.encode('utf-8'), payload.encode('ascii'), hashlib.sha256).digest())


def sign_cookie(email: str, secret: str, days: int = COOKIE_DAYS) -> str:
    exp = int((_now() + timedelta(days=days)).timestamp())
    payload = _b64(json.dumps({'e': email, 'x': exp}, separators=(',', ':')).encode('utf-8'))
    return f'{payload}.{_sig(payload, secret)}'


def read_cookie(value: str, secret: str) -> str | None:
    """The signed email, or None when the cookie is missing, forged or expired."""
    try:
        payload, sig = str(value or '').split('.', 1)
        if not hmac.compare_digest(sig, _sig(payload, secret)):
            return None
        data = json.loads(_unb64(payload))
        if int(data['x']) < _now().timestamp():
            return None
        return str(data['e']) or None
    except Exception:
        return None
