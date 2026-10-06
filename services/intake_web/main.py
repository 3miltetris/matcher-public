"""
intake-web — the public DD intake service (FastAPI, Cloud Run, unauthenticated).

Routes only; everything else lives in src/modules/intake/. Errors returned to
the founder are generic — details go to the log by session id, never answer
contents.

Access is by magic link: a founder whose address is on the allowlist
(admin-config/intake_allowlist.json, edited in the Matcher's Admin Portal) is
emailed a single-use link, and exchanging it sets a signed cookie that every
session route requires. Each session belongs to the email that created it.

Local run (from the repo root):
    INTAKE_DEV=1 uvicorn services.intake_web.main:app --reload --port 8080
INTAKE_DEV=1 skips Turnstile when TURNSTILE_SECRET is unset, signs cookies with
a throwaway key when INTAKE_AUTH_SECRET is unset, and logs the sign-in link when
SMTP is unset. Never set it in the deployed service.
"""

import logging
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.modules.intake import allowlist as al
from src.modules.intake import auth
from src.modules.intake import notify
from src.modules.intake import schema as sc
from src.modules.intake import schema_export as sx
from src.modules.intake import sessions as ss
from src.modules.intake import submit_pipeline as sp
from src.modules.intake import uploads as up

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')
log = logging.getLogger('intake')

STATIC = Path(__file__).parent / 'static'
DEV    = os.environ.get('INTAKE_DEV') == '1'

app = FastAPI(title='BW&CO intake', docs_url=None, redoc_url=None, openapi_url=None)
app.mount('/static', StaticFiles(directory=STATIC), name='static')

@app.middleware('http')
async def _revalidate(request: Request, call_next):
    """Page and static assets are revalidated on every load (cheap: ETag →
    304), so a redeploy never leaves a browser running an old app.js against
    a new index.html or API."""
    response = await call_next(request)
    if not request.url.path.startswith('/api/'):
        response.headers.setdefault('Cache-Control', 'no-cache')
    return response


_deps: sp.Deps | None = None


def deps() -> sp.Deps:
    global _deps
    if _deps is None:
        _deps = sp.production_deps()
    return _deps


# ── Abuse limits (per instance; Turnstile + the allowlist are the real gates) ─

_LINKS_PER_IP_DAY        = 20
_LINKS_PER_EMAIL_HOUR    = 5
_SESSIONS_PER_EMAIL_DAY  = 10
_WRITES_PER_SESSION_MIN  = 60
_hits: dict[str, deque] = defaultdict(deque)


def _limit(key: str, limit: int, window_s: int) -> None:
    now, q = time.time(), _hits[key]
    while q and now - q[0] > window_s:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, 'Too many requests — please try again later.')
    q.append(now)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get('x-forwarded-for', '')
    return fwd.split(',')[0].strip() or (request.client.host if request.client else '?')


def _verify_turnstile(token: str, ip: str) -> bool:
    secret = os.environ.get('TURNSTILE_SECRET')
    if not secret:
        return DEV
    try:
        r = requests.post('https://challenges.cloudflare.com/turnstile/v0/siteverify',
                          data={'secret': secret, 'response': token, 'remoteip': ip},
                          timeout=10)
        return bool(r.json().get('success'))
    except Exception as e:
        log.warning('turnstile verify failed: %s', type(e).__name__)
        return False


# ── Allowlist + auth ─────────────────────────────────────────────────────────

_ALLOW_TTL_S = 60
_allow_cache: dict = {}
_DEV_SECRET = secrets.token_hex(32)


def _allowlist() -> dict:
    """The allowlist, re-read at most once a minute. A read failure approves
    nobody (and keeps serving the last good copy if there is one)."""
    now = time.time()
    if _allow_cache and now - _allow_cache['at'] < _ALLOW_TTL_S:
        return _allow_cache['doc']
    try:
        doc = al.load(deps().storage_client)
    except Exception as e:
        log.error('allowlist read failed: %s', type(e).__name__)
        doc = _allow_cache.get('doc') or al.empty_doc()
    _allow_cache.update(doc=doc, at=now)
    return doc


def _auth_secret() -> str:
    secret = os.environ.get('INTAKE_AUTH_SECRET')
    if secret:
        return secret
    if DEV:
        return _DEV_SECRET
    log.error('INTAKE_AUTH_SECRET is not set; refusing to sign in')
    raise HTTPException(503, 'Sign-in is temporarily unavailable.')


def require_auth(request: Request) -> str:
    """The signed-in, still-approved email — or 401. Re-checking the
    allowlist here means removing a domain revokes access within a minute."""
    email = auth.read_cookie(request.cookies.get(auth.COOKIE_NAME, ''), _auth_secret())
    if not email or not al.is_approved(_allowlist(), email):
        raise HTTPException(401, 'Please sign in to continue.')
    return email


def _public_base(request: Request) -> str:
    return (os.environ.get('INTAKE_PUBLIC_URL') or str(request.base_url)).rstrip('/')


# ── Sessions ─────────────────────────────────────────────────────────────────

def _session(session_id: str, email: str) -> dict:
    """The caller's own session. Another owner's session reads as not found,
    so a guessed id reveals nothing."""
    try:
        session = ss.load_session(deps().bm, session_id)
    except ss.SessionNotFound:
        raise HTTPException(404, 'This form session has expired or does not exist.')
    owner = session.get('owner_email')
    if not owner and al.norm_email(session['answers'].get(sc.CONTACT_EMAIL)) == email:
        # A draft started before sign-in existed: its own contact address claims it.
        session['owner_email'] = owner = email
        ss.save_session(deps().bm, session)
        ss.set_owner_draft(deps().bm, email, session_id)
    if owner != email:
        raise HTTPException(404, 'This form session has expired or does not exist.')
    return session


def _draft(session_id: str, email: str) -> dict:
    session = _session(session_id, email)
    if session['status'] != ss.DRAFT:
        raise HTTPException(409, 'This form has already been submitted.')
    _limit(f'w:{session_id}', _WRITES_PER_SESSION_MIN, 60)
    return session


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    log.exception('unhandled error on %s', request.url.path)
    return JSONResponse({'detail': 'Something went wrong on our side. Your answers are saved — '
                                   'please try again in a minute.'}, status_code=500)


# ── Pages ────────────────────────────────────────────────────────────────────

@app.get('/', include_in_schema=False)
def index():
    return FileResponse(STATIC / 'index.html')


@app.get('/health', include_in_schema=False)
def health():
    return {'ok': True}


@app.get('/api/config')
def config():
    return {'turnstile_site_key': os.environ.get('TURNSTILE_SITE_KEY', ''),
            'max_files': up.MAX_FILES, 'max_mb': up.MAX_BYTES // (1024 * 1024),
            'extensions': list(up.ALLOWED), 'link_ttl_min': auth.LINK_TTL_MIN}


@app.get('/api/schema')
def get_schema():
    return sx.frontend_schema()


# ── Sign-in ──────────────────────────────────────────────────────────────────

class LinkBody(BaseModel):
    email: str = Field(max_length=300)
    turnstile_token: str = Field('', max_length=4096)


@app.post('/api/auth/request')
def request_link(body: LinkBody, request: Request):
    """Email a sign-in link if the address is approved. The answer is the
    same either way, so this cannot be used to read the allowlist."""
    ip = _client_ip(request)
    _limit(f'ip:{ip}', _LINKS_PER_IP_DAY, 86_400)
    email = al.norm_email(body.email)
    if not al.valid_email(email):
        raise HTTPException(400, 'Please enter a valid email address.')
    if not _verify_turnstile(body.turnstile_token, ip):
        raise HTTPException(400, 'Please complete the verification check.')
    _limit(f'link:{email}', _LINKS_PER_EMAIL_HOUR, 3600)

    if al.is_approved(_allowlist(), email):
        token = auth.issue_link_token(deps().bm, email)
        link = f'{_public_base(request)}/?t={token}'
        if DEV and not notify.smtp_configured():
            log.warning('DEV sign-in link for %s: %s', email, link)
        elif not notify.sign_in_link(email, link, auth.LINK_TTL_MIN):
            log.error('sign-in link email failed for domain %s', al.email_domain(email))
        else:
            log.info('sign-in link sent to domain %s', al.email_domain(email))
    else:
        log.info('sign-in refused for domain %s', al.email_domain(email) or '(freemail)')
    return {'sent': True}


class VerifyBody(BaseModel):
    token: str = Field(max_length=200)


@app.post('/api/auth/verify')
def verify_link(body: VerifyBody, response: Response):
    email = auth.consume_link_token(deps().bm, body.token)
    if not email or not al.is_approved(_allowlist(), email):
        raise HTTPException(400, 'This sign-in link has expired or was already used. '
                                 'Please request a new one.')
    response.set_cookie(auth.COOKIE_NAME, auth.sign_cookie(email, _auth_secret()),
                        max_age=auth.COOKIE_DAYS * 86_400, httponly=True,
                        secure=not DEV, samesite='lax', path='/')
    return {'email': email}


@app.get('/api/auth/me')
def me(request: Request):
    email = require_auth(request)
    draft = ss.owner_draft(deps().bm, email)
    return {'email': email, 'session_id': draft['session_id'] if draft else None}


@app.post('/api/auth/logout')
def logout(response: Response):
    response.delete_cookie(auth.COOKIE_NAME, path='/')
    return {'ok': True}


# ── Sessions ─────────────────────────────────────────────────────────────────

class StartBody(BaseModel):
    company_legal_name: str = Field(max_length=300)
    website: str = Field(max_length=300)


@app.post('/api/sessions')
def create_session(body: StartBody, request: Request):
    email = require_auth(request)
    existing = ss.owner_draft(deps().bm, email)
    if existing:                                  # one open draft per founder
        return {'session_id': existing['session_id'], 'resumed': True}
    _limit(f'sess:{email}', _SESSIONS_PER_EMAIL_DAY, 86_400)
    start = {sc.COMPANY_NAME: body.company_legal_name.strip(),
             sc.WEBSITE: body.website.strip(),
             sc.CONTACT_EMAIL: email}
    session = ss.create_session(deps().bm, start, owner_email=email)
    ss.set_owner_draft(deps().bm, email, session['session_id'])
    log.info('intake %s: session created', session['session_id'])
    return {'session_id': session['session_id'], 'resumed': False}


@app.get('/api/sessions/{session_id}')
def get_session(session_id: str, request: Request):
    s = _session(session_id, require_auth(request))
    return {'status': s['status'], 'answers': s['answers'],
            'confirmed_sections': s.get('confirmed_sections', []),
            'uploads': [{'upload_id': u['upload_id'], 'filename': u['filename']}
                        for u in s.get('uploads', []) if not u.get('rejected')]}


class AnswersBody(BaseModel):
    answers: dict
    confirmed_sections: list[str] = []


def _merge(session: dict, body: AnswersBody) -> None:
    known = {f.id for f in sc.FIELDS if f.type != 'file'}
    for k, v in body.answers.items():
        if k in known:
            session['answers'][k] = v
    # The contact address is the verified sign-in address, never an edit.
    session['answers'][sc.CONTACT_EMAIL] = session['owner_email']
    session['confirmed_sections'] = [s for s in body.confirmed_sections
                                     if s in sc.SECTION_IDS]


@app.put('/api/sessions/{session_id}/answers')
def save_answers(session_id: str, body: AnswersBody, request: Request):
    session = _draft(session_id, require_auth(request))
    _merge(session, body)
    try:
        ss.save_session(deps().bm, session)
    except ss.SessionConflict:
        raise HTTPException(409, 'Saved from another tab — reload to continue.')
    return {'saved_at': session['updated_at']}


class UploadReq(BaseModel):
    files: list[dict] = Field(max_length=up.MAX_FILES)


@app.post('/api/sessions/{session_id}/upload-urls')
def upload_urls(session_id: str, body: UploadReq, request: Request):
    session = _draft(session_id, require_auth(request))
    try:
        urls = up.issue_upload_urls(deps().storage_client, session, body.files)
    except up.UploadError as e:
        raise HTTPException(400, str(e))
    ss.save_session(deps().bm, session)
    return {'uploads': urls}


@app.post('/api/sessions/{session_id}/submit')
def submit(session_id: str, body: AnswersBody, request: Request, background: BackgroundTasks):
    session = _session(session_id, require_auth(request))
    if session['status'] != ss.DRAFT:
        return {'status': 'submitted'}            # idempotent re-click
    _merge(session, body)
    try:
        session = sp.submit(deps(), session)
    except sp.ValidationFailed as e:
        return JSONResponse({'detail': 'Some answers need attention.', 'errors': e.errors},
                            status_code=422)
    background.add_task(_finish, session_id)
    return {'status': 'submitted'}


def _finish(session_id: str) -> None:
    try:
        session = ss.load_session(deps().bm, session_id, allow_expired=True)
        sp.run_remaining(deps(), session)
    except Exception as e:
        log.exception('intake %s: background pipeline crashed', session_id)
        notify.step_failed(session_id, '(unknown)', 'background', type(e).__name__)
