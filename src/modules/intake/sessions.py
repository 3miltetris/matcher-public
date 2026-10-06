"""
sessions.py — intake session state in GCS (intake/sessions/{id}.json).

One JSON document per founder session: answers (autosaved), uploads, which
sections the founder confirmed, and — after submit — each pipeline step's
status and output ids, so a retry resumes where it stopped. Writes use the
blob generation as an optimistic lock: an autosave racing the background
submit pipeline fails loudly instead of silently dropping a step record.
"""

import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone

from src.modules.GoogleBucketManager.bucket_manager import BucketManager

BUCKET          = 'cc-matcher-bucket-jeg-v1'
SESSION_PREFIX  = 'intake/sessions/'
UPLOAD_PREFIX   = 'intake/uploads/'
SUBMIT_PREFIX   = 'intake/submissions/'
OWNER_PREFIX    = 'intake/owners/'
EXPIRY_DAYS     = 14

# token_urlsafe(24) → 32 chars, 192 bits.
_ID_RE = re.compile(r'^[A-Za-z0-9_-]{32}$')

DRAFT, SUBMITTED, COMPLETE, FAILED = 'draft', 'submitted', 'complete', 'failed'


class SessionNotFound(KeyError):
    pass


class SessionConflict(RuntimeError):
    """The session changed since it was read (another write won)."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def new_session_id() -> str:
    return secrets.token_urlsafe(24)


def valid_id(session_id: str) -> bool:
    return bool(_ID_RE.match(str(session_id or '')))


def session_blob(session_id: str) -> str:
    return f'{SESSION_PREFIX}{session_id}.json'


def is_expired(session: dict, days: int = EXPIRY_DAYS) -> bool:
    try:
        created = datetime.fromisoformat(session['created_at'])
    except Exception:
        return True
    return datetime.now(timezone.utc) - created > timedelta(days=days)


def create_session(bm: BucketManager, start: dict, owner_email: str = '') -> dict:
    session = {
        'session_id':  new_session_id(),
        'owner_email': owner_email,
        'created_at':  now_iso(),
        'updated_at':  now_iso(),
        'status':      DRAFT,
        'answers':     dict(start),
        'uploads':     [],
        'confirmed_sections': [],
        'steps':       {},
    }
    session['_generation'] = bm.upload_json(
        session_blob(session['session_id']), _strip(session), if_generation_match=0)
    return session


def load_session(bm: BucketManager, session_id: str, allow_expired: bool = False) -> dict:
    if not valid_id(session_id):
        raise SessionNotFound(session_id)
    obj, gen = bm.download_json(session_blob(session_id))
    if obj is None or (not allow_expired and obj.get('status') == DRAFT and is_expired(obj)):
        raise SessionNotFound(session_id)
    obj['_generation'] = gen
    return obj


def save_session(bm: BucketManager, session: dict) -> dict:
    """Write back under the generation it was read at; raises SessionConflict
    if anything else wrote in between."""
    from google.api_core.exceptions import PreconditionFailed
    session['updated_at'] = now_iso()
    try:
        session['_generation'] = bm.upload_json(
            session_blob(session['session_id']), _strip(session),
            if_generation_match=session.get('_generation'))
    except PreconditionFailed as e:
        raise SessionConflict(session['session_id']) from e
    return session


# ── Owner index ──────────────────────────────────────────────────────────────
# intake/owners/{sha256(email)}.json → the founder's current draft, so a
# magic link opened on any device resumes it. Kept outside intake/sessions/
# (and its 14-day lifecycle rule); a pointer to a vanished or finished session
# simply reads as "no draft".

def owner_blob(email: str) -> str:
    return f'{OWNER_PREFIX}{hashlib.sha256(email.encode("utf-8")).hexdigest()}.json'


def owner_draft(bm: BucketManager, email: str) -> dict | None:
    """The email's open draft session, or None."""
    rec, _ = bm.download_json(owner_blob(email))
    if not rec:
        return None
    try:
        session = load_session(bm, rec.get('session_id', ''))
    except SessionNotFound:
        return None
    if session.get('status') != DRAFT or session.get('owner_email') != email:
        return None
    return session


def set_owner_draft(bm: BucketManager, email: str, session_id: str) -> None:
    bm.upload_json(owner_blob(email), {'session_id': session_id, 'updated_at': now_iso()})


def _strip(session: dict) -> dict:
    return {k: v for k, v in session.items() if not k.startswith('_')}
