"""
Fathom meeting-notetaker API client
-----------------------------------
Streamlit-free helpers shared by views/fathom_sync.py (cheap discovery scan)
and jobs/fathom_sync_job.py (full sync), so nothing here may import streamlit.

Fathom's REST API (https://developers.fathom.ai) is read via `requests`
rather than the official `fathom-python` SDK: that SDK is still at 0.0.30
with documented breaking changes, and every other third-party API in this
codebase (HubSpot, SAM.gov, Grants.gov) is called with plain requests.

Rate limits are the binding constraint, not bandwidth:
  * 60 requests / 60 s for ordinary calls
  * 30 requests / 60 s for "heavy" calls -- anything carrying a summary or a
    transcript -- and Fathom warns this can drop to 5 / 60 s under load

so every request goes through one PROCESS-GLOBAL pacing gate (heavy calls
also spend standard quota, hence a single shared gate rather than two).
`fathom_get` additionally backs off on 429/5xx honouring `Retry-After`, and
raises RateLimitStalledError once retries are exhausted so a Cloud Run job
stops cleanly instead of burning its task timeout on doomed waits.
"""

import re
import threading
import time
from datetime import datetime

import requests

# ── Constants ───────────────────────────────────────────────────────────────

API_BASE = 'https://api.fathom.ai/external/v1'

# Payload flags accepted by GET /meetings as include_<name>=true.
INCLUDABLE = ('summary', 'transcript', 'action_items', 'highlights', 'crm_matches')
# Flags that make a request "heavy" in Fathom's rate-limit terms.
_HEAVY_INCLUDES = frozenset({'summary', 'transcript'})

DOMAINS_TYPES = ('all', 'only_internal', 'one_or_more_external')

# Pacing. Deliberately slower than the documented ceilings — Fathom's own docs
# say the heavy allowance can collapse to 5/min when the platform is busy, and
# a job that gets 429-throttled mid-sweep is worse than a slow one.
_STD_MIN_INTERVAL   = 1.05   # ~57 req/min
_HEAVY_MIN_INTERVAL = 2.50   # ~24 req/min
_MAX_RETRIES        = 6
_RETRY_BASE_S       = 5
_MAX_BACKOFF_S      = 180
_RETRY_STATUSES     = (429, 500, 502, 503, 504)

_RATE_LOCK  = threading.Lock()
_LAST_REQ   = [0.0]   # mutable for closure; guarded by _RATE_LOCK
_CALL_COUNT = [0]     # total Fathom HTTP requests this process; same guard

# Domains that are never a client's own website — freemail, our own domain,
# and the boilerplate URLs that show up in every business document.
# Mirrors jobs/drive_sync_job.py::_GENERIC_EMAIL_DOMAINS.
GENERIC_DOMAINS = {
    'gmail.com', 'googlemail.com', 'yahoo.com', 'ymail.com', 'hotmail.com',
    'outlook.com', 'live.com', 'msn.com', 'aol.com', 'icloud.com', 'me.com',
    'mac.com', 'protonmail.com', 'proton.me', 'pm.me', 'zoho.com', 'gmx.com',
    'mail.com', 'comcast.net', 'verizon.net', 'att.net', 'sbcglobal.net',
    'bwcoconsulting.com', 'google.com', 'gserviceaccount.com',
    'sam.gov', 'sbir.gov', 'grants.gov', 'linkedin.com', 'docs.google.com',
    'drive.google.com', 'sba.gov', 'irs.gov',
    # conferencing hosts — these turn up as invitee domains on some calendars
    'zoom.us', 'teams.microsoft.com', 'microsoft.com', 'calendly.com',
    'fathom.video', 'fathom.ai',
}

_SCHEME_RE = re.compile(r'^[a-z][a-z0-9+.-]*://')


# ── Errors ──────────────────────────────────────────────────────────────────

class FathomError(RuntimeError):
    """Any non-recoverable failure talking to the Fathom API."""


class FathomAuthError(FathomError):
    """401/403 — the key is wrong, revoked, or lacks visibility. Never retried."""


class RateLimitStalledError(FathomError):
    """Still 429 after every retry. Callers should stop the sweep, not loop."""


# ── Call accounting ─────────────────────────────────────────────────────────

def call_count() -> int:
    """Fathom HTTP requests made by this process so far (for status payloads)."""
    with _RATE_LOCK:
        return _CALL_COUNT[0]


def reset_call_count() -> None:
    with _RATE_LOCK:
        _CALL_COUNT[0] = 0


# ── HTTP ────────────────────────────────────────────────────────────────────

def _redact(text, api_key: str) -> str:
    """The key travels in a header, not the URL, but exception and response
    text still passes through logs and status.json — strip it if present."""
    out = str(text)
    if api_key and len(api_key) > 6:
        out = out.replace(api_key, '***')
    return out


def _throttle(heavy: bool) -> None:
    """Proactive global pacing gate. One shared timestamp for both tiers: a
    heavy call spends standard quota too, so serialising every call at the
    heavy interval is the only pacing that respects both ceilings."""
    interval = _HEAVY_MIN_INTERVAL if heavy else _STD_MIN_INTERVAL
    while True:
        with _RATE_LOCK:
            now  = time.monotonic()
            wait = _LAST_REQ[0] + interval - now
            if wait <= 0:
                _LAST_REQ[0]    = now
                _CALL_COUNT[0] += 1
                return
        time.sleep(wait)


def _retry_delay(response, attempt: int) -> float:
    """Honour Retry-After when Fathom sends one, else exponential backoff."""
    raw = (response.headers or {}).get('Retry-After', '')
    try:
        after = float(raw)
    except (TypeError, ValueError):
        after = 0.0
    return min(max(after, _RETRY_BASE_S * (2 ** attempt)), _MAX_BACKOFF_S)


def fathom_get(path: str, params: dict | None = None, *, api_key: str,
               heavy: bool = False, timeout: int = 60) -> dict:
    """Rate-limited, retrying GET against the Fathom external API.

    `heavy=True` for anything carrying a summary or transcript payload.
    Raises FathomAuthError (never retried), RateLimitStalledError (retries
    spent on 429s), or FathomError.
    """
    if not api_key:
        raise FathomAuthError('No Fathom API key supplied.')
    url     = f'{API_BASE}/{path.lstrip("/")}'
    headers = {'X-Api-Key': api_key, 'Accept': 'application/json'}
    last    = ''

    for attempt in range(_MAX_RETRIES + 1):
        _throttle(heavy)
        try:
            r = requests.get(url, params=params or None, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last = _redact(e, api_key)
            if attempt >= _MAX_RETRIES:
                raise FathomError(f'GET {path} failed after retries: {last}') from None
            time.sleep(min(_RETRY_BASE_S * (2 ** attempt), _MAX_BACKOFF_S))
            continue

        if r.status_code in (401, 403):
            raise FathomAuthError(
                f'Fathom rejected the API key (HTTP {r.status_code}). Check that the key '
                f'is current and that its user can see the meetings you expect — keys are '
                f'per user, and private unshared calls are never visible.'
            )
        if r.status_code in _RETRY_STATUSES:
            last = f'HTTP {r.status_code}: {_redact(r.text[:300], api_key)}'
            if attempt >= _MAX_RETRIES:
                if r.status_code == 429:
                    raise RateLimitStalledError(
                        f'Fathom still rate-limiting after {_MAX_RETRIES} retries ({last})'
                    )
                raise FathomError(f'GET {path} failed after retries: {last}')
            time.sleep(_retry_delay(r, attempt))
            continue
        if not r.ok:
            raise FathomError(
                f'GET {path} returned HTTP {r.status_code}: {_redact(r.text[:300], api_key)}'
            )
        try:
            payload = r.json()
        except ValueError as e:
            raise FathomError(
                f'GET {path} returned non-JSON: {_redact(e, api_key)}'
            ) from None
        return payload if isinstance(payload, dict) else {'items': payload}

    raise FathomError(f'GET {path} exhausted retries: {last}')


# ── Endpoints ───────────────────────────────────────────────────────────────

def iter_meetings(api_key: str, *, created_after: str | None = None,
                  created_before: str | None = None, include=(),
                  invitee_domains: list[str] | None = None,
                  domains_type: str | None = None,
                  recorded_by: list[str] | None = None,
                  max_pages: int = 400):
    """Yield meeting dicts from GET /meetings, following `next_cursor`.

    `include` is any subset of INCLUDABLE; each member adds include_<name>=true
    and members of _HEAVY_INCLUDES put the whole call on the heavy budget.
    Asking for 'summary' here is far cheaper than one /recordings call per
    meeting — the summary rides along on the page.
    """
    include = tuple(include or ())
    unknown = [name for name in include if name not in INCLUDABLE]
    if unknown:
        raise ValueError(f'unknown include flags: {unknown}')
    if domains_type is not None and domains_type not in DOMAINS_TYPES:
        raise ValueError(f'domains_type must be one of {DOMAINS_TYPES}')

    base: dict = {}
    if created_after:
        base['created_after'] = created_after
    if created_before:
        base['created_before'] = created_before
    if invitee_domains:
        base['calendar_invitees_domains[]'] = list(invitee_domains)
    if domains_type:
        base['calendar_invitees_domains_type'] = domains_type
    if recorded_by:
        base['recorded_by[]'] = list(recorded_by)
    for name in include:
        base[f'include_{name}'] = 'true'

    heavy  = bool(_HEAVY_INCLUDES & set(include))
    cursor = None
    for _ in range(max(1, int(max_pages))):
        params = dict(base)
        if cursor:
            params['cursor'] = cursor
        payload = fathom_get('meetings', params, api_key=api_key, heavy=heavy)
        items   = payload.get('items') or []
        for item in items:
            if isinstance(item, dict):
                yield item
        cursor = payload.get('next_cursor')
        if not cursor or not items:
            return


def get_transcript(api_key: str, recording_id) -> list[dict]:
    """Synchronous transcript for one recording (heavy). `destination_url` is
    deliberately unused — we have no public endpoint for Fathom to POST to."""
    payload = fathom_get(
        f'recordings/{recording_id}/transcript', api_key=api_key, heavy=True
    )
    turns = payload.get('transcript') or []
    return [t for t in turns if isinstance(t, dict)]


# ── Field helpers ───────────────────────────────────────────────────────────

def bare_domain(value) -> str:
    """Lowercased host for an email address, email domain or website URL; ''
    when it is freemail, ours, or not a domain at all. Good enough to line a
    meeting invitee up against a client's companyWebsite."""
    d = str(value or '').strip().lower()
    d = _SCHEME_RE.sub('', d)
    d = d.split('/')[0].split('?')[0].split('#')[0]
    d = d.split('@')[-1].split(':')[0].strip().strip('.')
    for prefix in ('www.', 'mail.', 'email.', 'smtp.'):
        while d.startswith(prefix):
            d = d[len(prefix):]
    if not d or '.' not in d:
        return ''
    if d in GENERIC_DOMAINS or d.endswith('.gserviceaccount.com'):
        return ''
    return d


def external_domains(meeting: dict) -> list[str]:
    """Business domains of the meeting's external calendar invitees, in order,
    deduped. This is how a meeting gets attributed to a client company."""
    out: list[str] = []
    seen: set[str] = set()
    for inv in meeting.get('calendar_invitees') or []:
        if not isinstance(inv, dict) or not inv.get('is_external'):
            continue
        raw = inv.get('email_domain') or inv.get('email') or ''
        d   = bare_domain(raw)
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def meeting_title(meeting: dict) -> str:
    return (str(meeting.get('meeting_title') or '').strip()
            or str(meeting.get('title') or '').strip()
            or 'Untitled meeting')


def meeting_date(meeting: dict) -> str:
    """ISO date (YYYY-MM-DD) the call was recorded."""
    raw = (meeting.get('recording_start_time')
           or meeting.get('scheduled_start_time')
           or meeting.get('created_at') or '')
    return str(raw)[:10]


def duration_minutes(meeting: dict) -> int:
    """Recorded length in whole minutes, 0 when the timestamps are unusable."""
    start, end = meeting.get('recording_start_time'), meeting.get('recording_end_time')
    if not start or not end:
        return 0
    try:
        began  = datetime.fromisoformat(str(start).replace('Z', '+00:00'))
        ended  = datetime.fromisoformat(str(end).replace('Z', '+00:00'))
        return max(0, int((ended - began).total_seconds() // 60))
    except (ValueError, TypeError):
        return 0


def summary_markdown(meeting: dict) -> str:
    summary = meeting.get('default_summary') or {}
    if not isinstance(summary, dict):
        return ''
    return str(summary.get('markdown_formatted') or '').strip()


def action_item_lines(meeting: dict, limit: int = 40) -> list[str]:
    out: list[str] = []
    for item in meeting.get('action_items') or []:
        if not isinstance(item, dict):
            continue
        desc = str(item.get('description') or '').strip()
        if not desc:
            continue
        who  = item.get('assignee') if isinstance(item.get('assignee'), dict) else {}
        name = str((who or {}).get('name') or '').strip()
        out.append(f'{desc} ({name})' if name else desc)
        if len(out) >= limit:
            break
    return out


def attendee_lines(meeting: dict, limit: int = 25) -> list[str]:
    out: list[str] = []
    for inv in meeting.get('calendar_invitees') or []:
        if not isinstance(inv, dict):
            continue
        name  = str(inv.get('name') or '').strip()
        email = str(inv.get('email') or '').strip()
        who   = f'{name} <{email}>' if name and email else (name or email)
        if not who:
            continue
        out.append(f'{who} (external)' if inv.get('is_external') else who)
        if len(out) >= limit:
            break
    return out


def crm_company_names(meeting: dict) -> list[str]:
    """Companies Fathom's own CRM integration matched to the call. Used only as
    a hint in the review table — never to auto-assign a client."""
    matches = meeting.get('crm_matches') or {}
    if not isinstance(matches, dict):
        return []
    out: list[str] = []
    for company in matches.get('companies') or []:
        if isinstance(company, dict):
            name = str(company.get('name') or '').strip()
            if name:
                out.append(name)
    return out


def transcript_text(transcript, char_cap: int = 40_000) -> str:
    """`Speaker: line` text for a transcript, capped by dropping the MIDDLE.

    Meetings front-load context ("here is what we build") and back-load next
    steps; a plain head-truncation throws away exactly the part that says what
    the company is going to do next."""
    lines: list[str] = []
    for turn in transcript or []:
        if not isinstance(turn, dict):
            continue
        text = str(turn.get('text') or '').strip()
        if not text:
            continue
        speaker = turn.get('speaker') if isinstance(turn.get('speaker'), dict) else {}
        name    = str((speaker or {}).get('display_name') or '').strip() or 'Speaker'
        lines.append(f'{name}: {text}')
    joined = '\n'.join(lines)
    cap    = max(500, int(char_cap))
    if len(joined) <= cap:
        return joined
    head = (cap * 2) // 3
    tail = cap - head
    return (joined[:head].rstrip()
            + '\n\n…[middle of transcript truncated]…\n\n'
            + joined[-tail:].lstrip())


def meeting_block(meeting: dict, transcript=None,
                  transcript_char_cap: int = 40_000) -> dict:
    """The per-meeting unit handed to Claude for the per-client digest."""
    return {
        'recording_id':   meeting.get('recording_id'),
        'title':          meeting_title(meeting),
        'date':           meeting_date(meeting),
        'duration_min':   duration_minutes(meeting),
        'attendees':      attendee_lines(meeting),
        'fathom_summary': summary_markdown(meeting),
        'action_items':   action_item_lines(meeting),
        'transcript':     transcript_text(
            transcript if transcript is not None else meeting.get('transcript'),
            transcript_char_cap,
        ),
    }
