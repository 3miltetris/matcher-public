"""
email_otp.py — read one-time 2FA codes out of a shared mailbox (Stage 10).

Some login-walled funding-source sites enforce 2FA and email the code. Team
members forward those code emails into one shared Workspace mailbox; this module
reads that mailbox via the Gmail API and pulls the code out, so
``browser_agent.get_email_code`` can enter it and finish the login unattended.

Streamlit-free and browser-free: the job imports it, builds a Gmail service, and
hands ``browser_agent`` a bound ``code_fetcher`` callback — the agent never
touches Gmail.

Access model (see the plan): a **user-based OAuth refresh token** for the shared
mailbox, stored in Secret Manager as ``funding-2fa-gmail-oauth``:

    {"client_id": "...", "client_secret": "...", "refresh_token": "...",
     "mailbox": "funding-2fa@bwcoconsulting.com"}

The OAuth client is an **Internal** Workspace app, so the restricted
``gmail.readonly`` scope needs no Google verification and the token does not
expire. No service-account key, no domain-wide delegation — consistent with the
repo's "secrets in Secret Manager, ADC for jobs" posture.
"""

import base64
import re

OAUTH_SECRET_ID = 'funding-2fa-gmail-oauth'
PROJECT_ID      = 'cc-matcher-v1'

GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
_TOKEN_URI   = 'https://oauth2.googleapis.com/token'

# 4–8 digit codes cover almost every email OTP. Admins can override per site.
DEFAULT_CODE_REGEX = r'\b\d{4,8}\b'

# How far back from the login timestamp to look, to absorb clock skew / forward
# latency without picking up a stale code.
_GRACE_S = 120

# Words that sit next to the real code in a 2FA email — used to prefer the right
# number when a body contains several.
_OTP_KEYWORDS = re.compile(
    r'code|verif|passcode|one[\s-]?time|\botp\b|\bpin\b|security|authenticat', re.I)


# -- OAuth / Gmail service (lazy imports so the pure helpers test without libs) --

def _secret_name(project_id: str = PROJECT_ID) -> str:
    return f'projects/{project_id}/secrets/{OAUTH_SECRET_ID}/versions/latest'


def _is_missing(exc: Exception) -> bool:
    return type(exc).__name__ in ('NotFound', 'FailedPrecondition')


def load_oauth(sm_client, project_id: str = PROJECT_ID) -> dict:
    """The stored OAuth blob, or ``{}`` if the secret / version does not exist.
    Permission and network errors propagate (the job logs + degrades)."""
    import json
    try:
        resp = sm_client.access_secret_version(request={'name': _secret_name(project_id)})
    except Exception as exc:                                   # noqa: BLE001
        if _is_missing(exc):
            return {}
        raise
    try:
        data = json.loads(resp.payload.data.decode('utf-8'))
    except Exception:                                         # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def build_gmail(oauth: dict):
    """A Gmail API service for the shared mailbox from an OAuth blob.

    Returns None when the blob is missing the fields, so callers can degrade.
    """
    if not oauth or not oauth.get('refresh_token'):
        return None
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    creds = Credentials(
        None,
        refresh_token=oauth['refresh_token'],
        token_uri=_TOKEN_URI,
        client_id=oauth.get('client_id'),
        client_secret=oauth.get('client_secret'),
        scopes=GMAIL_SCOPES,
    )
    return build('gmail', 'v1', credentials=creds, cache_discovery=False)


# -- Code extraction (pure) -------------------------------------------------

def extract_code(text: str, regex: str = '') -> str | None:
    """Pull the OTP out of an email body. Prefers a match sitting next to an OTP
    keyword (``your code is 481920``) over the first stray number."""
    if not text:
        return None
    try:
        rx = re.compile(regex or DEFAULT_CODE_REGEX)
    except re.error:
        rx = re.compile(DEFAULT_CODE_REGEX)

    matches = list(rx.finditer(text))
    if not matches:
        return None

    def _value(m):
        return (m.group(1) if rx.groups else m.group(0)).strip()

    for m in matches:
        window = text[max(0, m.start() - 40):m.start()]
        if _OTP_KEYWORDS.search(window):
            return _value(m)
    return _value(matches[0])


def _b64(data: str) -> str:
    if not data:
        return ''
    try:
        return base64.urlsafe_b64decode(data.encode('utf-8')).decode('utf-8', 'replace')
    except Exception:                                         # noqa: BLE001
        return ''


def _strip_html(html: str) -> str:
    if not html:
        return ''
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(html, 'html.parser').get_text(' ')
    except Exception:                                         # noqa: BLE001
        return re.sub(r'<[^>]+>', ' ', html)


def message_text(payload: dict) -> str:
    """Flatten a Gmail message payload to text (plain preferred, else HTML)."""
    if not isinstance(payload, dict):
        return ''
    mime = payload.get('mimeType', '')
    body = (payload.get('body') or {}).get('data', '')
    parts = payload.get('parts') or []

    if mime == 'text/plain' and body:
        return _b64(body)
    if mime == 'text/html' and body:
        return _strip_html(_b64(body))

    plain, html = '', ''
    for part in parts:
        sub = message_text(part)
        if (part.get('mimeType') == 'text/plain') and sub and not plain:
            plain = sub
        elif (part.get('mimeType') == 'text/html') and sub and not html:
            html = sub
        elif sub and not plain:
            plain = sub
    return plain or html


def _build_query(sender: str, after_epoch: int, subject: str) -> str:
    q = []
    if after_epoch:
        q.append(f'after:{max(0, int(after_epoch) - _GRACE_S)}')
    if sender:
        # From is usually preserved on a Gmail filter-forward, but fall back to
        # the sender string appearing anywhere (body/headers) when it is not.
        q.append(f'(from:{sender} OR "{sender}")')
    if subject:
        q.append(f'subject:({subject})')
    return ' '.join(q) if q else 'newer_than:1h'


# -- The one call the agent's fetcher wraps ---------------------------------

def fetch_code(gmail, *, sender: str = '', after_epoch: int = 0,
               regex: str = '', subject: str = '', max_messages: int = 6) -> str | None:
    """Check the mailbox once for a fresh code. Returns the code or None.

    Never logs the code or the message body.
    """
    if gmail is None:
        return None
    q = _build_query(sender, after_epoch, subject)
    try:
        listing = gmail.users().messages().list(
            userId='me', q=q, maxResults=max_messages).execute()
    except Exception:                                         # noqa: BLE001
        return None
    for meta in (listing.get('messages') or [])[:max_messages]:
        try:
            msg = gmail.users().messages().get(
                userId='me', id=meta['id'], format='full').execute()
        except Exception:                                     # noqa: BLE001
            continue
        # Guard against a stale message the query let through.
        if after_epoch:
            try:
                if int(msg.get('internalDate', 0)) < (int(after_epoch) - _GRACE_S) * 1000:
                    continue
            except (TypeError, ValueError):
                pass
        text = message_text(msg.get('payload') or {}) or msg.get('snippet', '')
        code = extract_code(text, regex)
        if code:
            return code
    return None
