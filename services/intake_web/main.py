"""
intake-web — the public DD intake service (FastAPI, Cloud Run, unauthenticated).

Routes only; everything else lives in src/modules/intake/. Errors returned to
the founder are generic — details go to the log by session id, never answer
contents.

Local run (from the repo root):
    INTAKE_DEV=1 uvicorn services.intake_web.main:app --reload --port 8080
INTAKE_DEV=1 skips Turnstile when TURNSTILE_SECRET is unset. Never set it in
the deployed service.
"""

import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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

_deps: sp.Deps | None = None


def deps() -> sp.Deps:
    global _deps
    if _deps is None:
        _deps = sp.production_deps()
    return _deps


# ── Abuse limits (per instance; Turnstile is the real gate) ──────────────────

_SESSIONS_PER_IP_DAY = 10
_WRITES_PER_SESSION_MIN = 60
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


def _session(session_id: str) -> dict:
    try:
        return ss.load_session(deps().bm, session_id)
    except ss.SessionNotFound:
        raise HTTPException(404, 'This form session has expired or does not exist.')


def _draft(session_id: str) -> dict:
    session = _session(session_id)
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
            'extensions': list(up.ALLOWED)}


@app.get('/api/schema')
def get_schema():
    return sx.frontend_schema()


# ── Sessions ─────────────────────────────────────────────────────────────────

class StartBody(BaseModel):
    company_legal_name: str = Field(max_length=300)
    website: str = Field(max_length=300)
    contact_email: str = Field(max_length=300)
    turnstile_token: str = Field('', max_length=4096)


@app.post('/api/sessions')
def create_session(body: StartBody, request: Request):
    ip = _client_ip(request)
    _limit(f'ip:{ip}', _SESSIONS_PER_IP_DAY, 86_400)
    if not _verify_turnstile(body.turnstile_token, ip):
        raise HTTPException(400, 'Please complete the verification check.')
    start = {sc.COMPANY_NAME: body.company_legal_name.strip(),
             sc.WEBSITE: body.website.strip(),
             sc.CONTACT_EMAIL: body.contact_email.strip()}
    session = ss.create_session(deps().bm, start)
    log.info('intake %s: session created', session['session_id'])
    return {'session_id': session['session_id']}


@app.get('/api/sessions/{session_id}')
def get_session(session_id: str):
    s = _session(session_id)
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
    session['confirmed_sections'] = [s for s in body.confirmed_sections
                                     if s in sc.SECTION_IDS]


@app.put('/api/sessions/{session_id}/answers')
def save_answers(session_id: str, body: AnswersBody):
    session = _draft(session_id)
    _merge(session, body)
    try:
        ss.save_session(deps().bm, session)
    except ss.SessionConflict:
        raise HTTPException(409, 'Saved from another tab — reload to continue.')
    return {'saved_at': session['updated_at']}


class UploadReq(BaseModel):
    files: list[dict] = Field(max_length=up.MAX_FILES)


@app.post('/api/sessions/{session_id}/upload-urls')
def upload_urls(session_id: str, body: UploadReq):
    session = _draft(session_id)
    try:
        urls = up.issue_upload_urls(deps().storage_client, session, body.files)
    except up.UploadError as e:
        raise HTTPException(400, str(e))
    ss.save_session(deps().bm, session)
    return {'uploads': urls}


@app.post('/api/sessions/{session_id}/submit')
def submit(session_id: str, body: AnswersBody, background: BackgroundTasks):
    session = _session(session_id)
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
