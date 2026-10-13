"""
browser_session.py — captured interactive sign-ins for login-walled funding
sources (Stage 10).

Some funding-source sites guard their member area with an authenticator app,
SMS, SSO or a CAPTCHA. None of those can be automated, and they do not need to
be: a human signs in once in a live browser (the Steel window embedded in the
Funding Sources view, see ``steel_client``), the authenticated browser state —
cookies and localStorage — is captured, and the deep-research job starts that
site's browser context **already signed in** via Playwright's
``new_context(storage_state=...)``. No password or code is ever typed by the
agent for such a site.

A captured state is a live session token — as sensitive as a password — so it
lives in **Secret Manager**, never in the bucket, the registry, a log line or
the model's context, exactly like ``source_credentials``:

    projects/cc-matcher-v1/secrets/funding-source-sessions

Payload: a JSON object keyed by registry ``source_id``::

    {
      "73471d": {
        "state_gz":    "<base64 gzip of the Playwright storage_state JSON>",
        "user_agent":  "Mozilla/5.0 ... Chrome/1xx ...",
        "hosts":       ["marketplace.gocolosseum.org", ...],
        "captured_at": "2026-10-13T15:00:00+00:00",
        "captured_by": "john@bwcoconsulting.com"
      }
    }

It is a **separate** secret from ``funding-source-credentials``: sessions are
recaptured every few weeks while passwords almost never change, a site may have
a session and no stored password at all, and keeping them apart stops one
growing map from pushing the other past the size limit.

**Size is the binding constraint.** A Secret Manager version holds at most
64 KiB, and this map holds every site's state. States are gzipped (cookie and
localStorage JSON compresses ~4x) and, when one is still too big, trimmed to the
hosts that belong to the site (``trim_storage_state``) — third-party tracker
cookies and analytics localStorage are the bulk of a raw capture and carry no
authentication. ``set_session`` refuses a save that would exceed the limit
rather than letting Secret Manager reject it with a less useful error.

The non-secret mirror lives in the registry: ``has_session``,
``session_status`` (``active`` / ``expired``) and ``session_captured_at``. The
job flips ``session_status`` to ``expired`` when the agent hits a login wall
despite an injected session; it never writes this secret.

Streamlit-free and client-injected (the view passes a service-account Secret
Manager client, the job an ADC one) — the ``source_credentials`` split.
"""

import base64
import gzip
import json
from datetime import datetime, timezone
from urllib.parse import urlparse

from src.modules import source_credentials as sc

PROJECT_ID = sc.PROJECT_ID
SECRET_ID  = 'funding-source-sessions'

# Secret Manager's hard limit is 65,536 bytes per version. Leave headroom for
# the JSON envelope so the check here, not the API, is what refuses a save.
MAX_SECRET_BYTES = 60_000
# One site's encoded state above this is trimmed to the site's own hosts before
# it is stored — a raw capture is mostly third-party tracker state.
TRIM_ABOVE_BYTES = 12_000

_ENTRY_FIELDS = ('state_gz', 'user_agent', 'hosts', 'captured_at', 'captured_by')
_SAMESITE = {'strict': 'Strict', 'lax': 'Lax', 'none': 'None'}


class SessionTooLarge(ValueError):
    """The captured state would push the shared secret past its size limit."""


# -- Resource paths ---------------------------------------------------------

def secret_name(project_id: str = PROJECT_ID) -> str:
    return f'projects/{project_id}/secrets/{SECRET_ID}'


def latest_version_name(project_id: str = PROJECT_ID) -> str:
    return f'{secret_name(project_id)}/versions/latest'


# -- Steel context -> Playwright storage_state ------------------------------

def steel_context_to_storage_state(ctx: dict) -> dict:
    """Convert Steel's ``GET /v1/sessions/{id}/context`` payload into the dict
    Playwright's ``new_context(storage_state=...)`` accepts.

    Steel returns cookies as CDP cookie objects and localStorage keyed by host
    (``{"example.com": {"key": "value"}}``, verified against the live API
    2026-10-13). Playwright wants ``expires=-1`` for a session cookie, a
    capitalised ``sameSite`` (omitted when unknown, so Playwright applies its
    default), and localStorage per scheme-qualified origin. ``sessionStorage``
    and ``indexedDB`` are dropped: Playwright's storage_state does not restore
    sessionStorage, and no site in the registry keeps its auth token in
    IndexedDB.
    """
    ctx = ctx or {}
    cookies = []
    for c in ctx.get('cookies') or []:
        if not isinstance(c, dict) or not c.get('name') or not c.get('domain'):
            continue
        expires = c.get('expires')
        if c.get('session') or expires is None or float(expires) <= 0:
            expires = -1
        cookie = {
            'name':     str(c['name']),
            'value':    str(c.get('value') or ''),
            'domain':   str(c['domain']),
            'path':     str(c.get('path') or '/'),
            'expires':  float(expires),
            'httpOnly': bool(c.get('httpOnly')),
            'secure':   bool(c.get('secure')),
        }
        same_site = _SAMESITE.get(str(c.get('sameSite') or '').lower())
        if same_site:
            cookie['sameSite'] = same_site
        cookies.append(cookie)

    origins = []
    for host, items in (ctx.get('localStorage') or {}).items():
        if not isinstance(items, dict) or not items:
            continue
        host = str(host)
        origin = host if '://' in host else f'https://{host}'
        origins.append({
            'origin':       origin,
            'localStorage': [{'name': str(k), 'value': str(v)} for k, v in items.items()],
        })
    return {'cookies': cookies, 'origins': origins}


def state_hosts(state: dict) -> list:
    """Every host a storage_state holds data for — display and trimming input."""
    hosts = {str(c.get('domain') or '').lstrip('.') for c in state.get('cookies') or []}
    hosts |= {urlparse(o.get('origin') or '').hostname or ''
              for o in state.get('origins') or []}
    return sorted(h for h in hosts if h)


def registrable_domain(host_or_url: str) -> str:
    """``app.portal.example.co.uk`` -> ``example.co.uk``.

    Uses tldextract when available (the app image has it). The fallback keeps
    the last two labels, which is right for every .org/.com/.gov/.mil site in
    the registry.
    """
    s = str(host_or_url or '').strip().lower()
    host = urlparse(s).hostname if '://' in s else s.split('/')[0]
    host = (host or '').lstrip('.')
    if not host:
        return ''
    try:
        import tldextract
        ext = tldextract.extract(host)
        if ext.domain and ext.suffix:
            return f'{ext.domain}.{ext.suffix}'
    except Exception:                                         # noqa: BLE001
        pass
    parts = host.split('.')
    return '.'.join(parts[-2:]) if len(parts) >= 2 else host


def trim_storage_state(state: dict, keep_domains) -> dict:
    """Keep only cookies/origins whose host sits under one of ``keep_domains``
    (registrable domains, e.g. the site URL's and the login URL's)."""
    keep = {registrable_domain(d) for d in keep_domains if d}
    keep.discard('')
    if not keep:
        return state

    def _ok(host: str) -> bool:
        return registrable_domain(host) in keep

    return {
        'cookies': [c for c in state.get('cookies') or [] if _ok(c.get('domain', ''))],
        'origins': [o for o in state.get('origins') or [] if _ok(o.get('origin', ''))],
    }


# -- Encoding ---------------------------------------------------------------

def encode_state(state: dict) -> str:
    raw = json.dumps(state, separators=(',', ':')).encode('utf-8')
    return base64.b64encode(gzip.compress(raw, compresslevel=9)).decode('ascii')


def decode_state(blob: str) -> dict | None:
    try:
        data = json.loads(gzip.decompress(base64.b64decode(blob)).decode('utf-8'))
    except Exception:                                         # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    data.setdefault('cookies', [])
    data.setdefault('origins', [])
    return data


# -- Store I/O --------------------------------------------------------------

def load_sessions(sm_client, project_id: str = PROJECT_ID) -> dict:
    """The whole session map. A missing secret / no enabled version is ``{}``;
    permission and network errors propagate (the view shows them, the job
    degrades to no sessions)."""
    try:
        resp = sm_client.access_secret_version(
            request={'name': latest_version_name(project_id)})
    except Exception as exc:                                  # noqa: BLE001
        if sc._is_missing(exc):
            return {}
        raise
    try:
        data = json.loads(resp.payload.data.decode('utf-8'))
    except Exception:                                         # noqa: BLE001
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(sid): {k: e.get(k, '') for k in _ENTRY_FIELDS}
            for sid, e in data.items() if isinstance(e, dict) and e.get('state_gz')}


def _payload(data: dict) -> bytes:
    return json.dumps(data or {}, separators=(',', ':')).encode('utf-8')


def save_sessions(sm_client, data: dict, project_id: str = PROJECT_ID) -> None:
    payload = _payload(data)
    if len(payload) > MAX_SECRET_BYTES:
        raise SessionTooLarge(
            f'the session store would be {len(payload):,} bytes, over the '
            f'{MAX_SECRET_BYTES:,}-byte Secret Manager budget')
    sm_client.add_secret_version(
        request={'parent': secret_name(project_id), 'payload': {'data': payload}})


# -- Mutations --------------------------------------------------------------

def prepare_state(state: dict, keep_domains) -> tuple[dict, bool]:
    """Trim a captured state if its encoding is over ``TRIM_ABOVE_BYTES``.
    Returns ``(state, trimmed)``."""
    if len(encode_state(state)) <= TRIM_ABOVE_BYTES:
        return state, False
    return trim_storage_state(state, keep_domains), True


def set_session(sm_client, source_id: str, state: dict, *, user_agent: str = '',
                actor: str = '', project_id: str = PROJECT_ID) -> dict:
    """Upsert one site's captured state. Returns the new full map. Raises
    ``SessionTooLarge`` (nothing written) if the map would not fit."""
    sid = str(source_id or '').strip()
    if not sid:
        raise ValueError('set_session: a source_id is required')
    if not isinstance(state, dict) or not (state.get('cookies') or state.get('origins')):
        raise ValueError('set_session: the captured browser state is empty — '
                         'sign in fully before saving')
    data = load_sessions(sm_client, project_id)
    data[sid] = {
        'state_gz':    encode_state(state),
        'user_agent':  str(user_agent or '').strip(),
        'hosts':       state_hosts(state),
        'captured_at': datetime.now(timezone.utc).isoformat(),
        'captured_by': str(actor or '').strip(),
    }
    save_sessions(sm_client, data, project_id)
    return data


def delete_session(sm_client, source_id: str, project_id: str = PROJECT_ID) -> dict:
    data = load_sessions(sm_client, project_id)
    if str(source_id) in data:
        data.pop(str(source_id), None)
        save_sessions(sm_client, data, project_id)
    return data


# -- Read helpers -----------------------------------------------------------

def session_for(data: dict, source_id: str) -> dict | None:
    """``{storage_state, user_agent}`` for one site, or ``None``. Used by the job
    to seed a browser context — never rendered or logged."""
    if not isinstance(data, dict):
        return None
    entry = data.get(str(source_id))
    if not isinstance(entry, dict) or not entry.get('state_gz'):
        return None
    state = decode_state(entry['state_gz'])
    if state is None:
        return None
    return {'storage_state': state, 'user_agent': entry.get('user_agent') or ''}


def configured_ids(data: dict) -> set:
    return {sid for sid, e in (data or {}).items()
            if isinstance(e, dict) and e.get('state_gz')}


def store_bytes(data: dict) -> int:
    """Current encoded size of the map, for the view's capacity readout."""
    return len(_payload(data))


def safe_summary(data: dict) -> list:
    """State-free rows for display. The **only** thing the view renders."""
    rows = []
    for sid, e in (data or {}).items():
        if not isinstance(e, dict) or not e.get('state_gz'):
            continue
        rows.append({
            'source_id':   sid,
            'hosts':       list(e.get('hosts') or []),
            'size_bytes':  len(e.get('state_gz') or ''),
            'captured_at': e.get('captured_at', ''),
            'captured_by': e.get('captured_by', ''),
        })
    return sorted(rows, key=lambda r: r['source_id'])
