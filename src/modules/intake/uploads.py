"""
uploads.py — founder document uploads straight from the browser to GCS.

The service issues V4 signed PUT URLs restricted by content type and size
(x-goog-content-length-range), then re-checks every object on read: size,
declared type and magic bytes. Cloud Run has no private key file, so URLs are
signed through the IAM signBlob API with the runtime service account's own
access token — the account needs roles/iam.serviceAccountTokenCreator on
itself.
"""

import re
import secrets
from datetime import timedelta

import google.auth
from google.auth.transport.requests import Request

from src.modules.intake.sessions import BUCKET, UPLOAD_PREFIX, now_iso

MAX_FILES   = 3
MAX_BYTES   = 25 * 1024 * 1024
URL_EXPIRY  = timedelta(minutes=15)

ALLOWED: dict[str, str] = {
    '.pdf':  'application/pdf',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
}
_MAGIC = {'.pdf': b'%PDF', '.pptx': b'PK\x03\x04', '.docx': b'PK\x03\x04'}


class UploadError(ValueError):
    pass


def safe_filename(name: str) -> str:
    base = re.split(r'[\\/]', str(name or ''))[-1]
    base = re.sub(r'[^A-Za-z0-9._ ()-]+', '_', base).strip(' .')
    return (base or 'document')[:120]


def extension(name: str) -> str:
    m = re.search(r'(\.[A-Za-z0-9]+)$', str(name or ''))
    return m.group(1).lower() if m else ''


def _signing_identity():
    creds, _ = google.auth.default(
        scopes=['https://www.googleapis.com/auth/cloud-platform'])
    creds.refresh(Request())
    email = getattr(creds, 'service_account_email', None)
    if not email or email == 'default':
        raise RuntimeError('signed URLs need a service-account identity')
    return email, creds.token


def issue_upload_urls(storage_client, session: dict, files: list[dict]) -> list[dict]:
    """Validate the requested files, register them on the session (caller
    saves it) and return [{upload_id, filename, url, headers}]. Raises
    UploadError on anything not allowed."""
    current = [u for u in session.get('uploads', []) if not u.get('rejected')]
    if not files or len(current) + len(files) > MAX_FILES:
        raise UploadError(f'Up to {MAX_FILES} files per submission.')
    checked = []
    for f in files:
        name = safe_filename(f.get('filename'))
        ext  = extension(name)
        try:
            size = int(f.get('size') or 0)
        except (TypeError, ValueError):
            size = 0
        if ext not in ALLOWED:
            raise UploadError(f'{name}: only PDF, PPTX or DOCX files are accepted.')
        if not 0 < size <= MAX_BYTES:
            raise UploadError(f'{name}: files must be under 25 MB.')
        checked.append((name, ext, size))
    email, token = _signing_identity()
    bucket = storage_client.bucket(BUCKET)
    out = []
    for name, ext, size in checked:
        ctype     = ALLOWED[ext]
        upload_id = secrets.token_hex(8)
        obj       = f'{UPLOAD_PREFIX}{session["session_id"]}/{upload_id}{ext}'
        headers   = {'Content-Type': ctype,
                     'x-goog-content-length-range': f'0,{MAX_BYTES}'}
        url = bucket.blob(obj).generate_signed_url(
            version='v4', expiration=URL_EXPIRY, method='PUT',
            content_type=ctype, headers={'x-goog-content-length-range': f'0,{MAX_BYTES}'},
            service_account_email=email, access_token=token)
        session.setdefault('uploads', []).append({
            'upload_id': upload_id, 'filename': name, 'object': obj,
            'content_type': ctype, 'declared_size': size, 'issued_at': now_iso(),
        })
        out.append({'upload_id': upload_id, 'filename': name, 'url': url,
                    'headers': headers})
    return out


def verified_uploads(storage_client, session: dict) -> list[dict]:
    """Re-check every registered upload against what actually landed in GCS.
    Returns the uploads that exist and pass; marks the rest `rejected` (and
    deletes a wrong-type object) so nothing unverified reaches Drive."""
    bucket = storage_client.bucket(BUCKET)
    good = []
    for u in session.get('uploads', []):
        if u.get('rejected'):
            continue
        blob = bucket.get_blob(u['object'])
        if blob is None:
            u['rejected'] = 'never uploaded'
            continue
        ext  = extension(u['object'])
        head = blob.download_as_bytes(start=0, end=7)
        if (blob.size or 0) > MAX_BYTES or not head.startswith(_MAGIC.get(ext, b'\0')):
            u['rejected'] = 'failed type/size check'
            blob.delete()
            continue
        u['size'] = blob.size
        good.append(u)
    return good
