"""
drive_client.py — streamlit-free Google Drive v3 helpers for shared drives.

Used by views/drive_sync.py (service-account creds from st.secrets, built
by the caller with scopes=DRIVE_SCOPES) and jobs/drive_sync_job.py (ADC).
The service account must be added as a member (Viewer) of the shared drive.

All API calls route through _execute() for exponential-backoff retries on
rate-limit and transient server errors.
"""

import io
import random
import time

from googleapiclient.discovery import build as _discovery_build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from src.modules import doc_extract

DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

FOLDER_MIME   = 'application/vnd.google-apps.folder'
SHORTCUT_MIME = 'application/vnd.google-apps.shortcut'

# Google-native types → export mime
EXPORT_MIMES = {
    'application/vnd.google-apps.document':     'text/plain',
    'application/vnd.google-apps.spreadsheet':  'text/csv',
    'application/vnd.google-apps.presentation': 'text/plain',
}

BINARY_ALLOW_EXTS = ('.pdf', '.docx', '.doc', '.xlsx', '.txt', '.csv', '.md')
MAX_BINARY_BYTES  = 15_000_000
MAX_DEPTH         = 6

_FILE_FIELDS = ('nextPageToken, files(id, name, mimeType, modifiedTime, size, '
                'shortcutDetails, lastModifyingUser(emailAddress))')

_RETRY_STATUSES = (429, 500, 502, 503, 504)


def build_drive_service(credentials=None):
    """Build a Drive v3 service. Pass explicit credentials (Streamlit path:
    from_service_account_info(..., scopes=DRIVE_SCOPES)); None uses ADC
    (Cloud Run job path)."""
    if credentials is None:
        import google.auth
        credentials, _ = google.auth.default(scopes=DRIVE_SCOPES)
    return _discovery_build('drive', 'v3', credentials=credentials,
                            cache_discovery=False)


def _execute(request, retries: int = 5):
    """Execute a Drive API request with exponential backoff on 403-rate/429/5xx."""
    for attempt in range(retries + 1):
        try:
            return request.execute()
        except HttpError as e:
            status    = e.resp.status if e.resp is not None else 0
            rate_403  = status == 403 and any(
                r in str(e).lower() for r in ('ratelimitexceeded', 'userratelimitexceeded',
                                              'rate limit'))
            if attempt < retries and (status in _RETRY_STATUSES or rate_403):
                time.sleep(2 ** attempt + random.random())
                continue
            raise


def list_child_folders(svc, drive_id: str, parent_id: str) -> list[dict]:
    """Direct child folders of parent_id inside a shared drive → [{'id','name'}]."""
    folders: list[dict] = []
    page_token = None
    while True:
        resp = _execute(svc.files().list(
            q=f"'{parent_id}' in parents and mimeType='{FOLDER_MIME}' and trashed=false",
            corpora='drive', driveId=drive_id,
            includeItemsFromAllDrives=True, supportsAllDrives=True,
            pageSize=200, pageToken=page_token,
            fields='nextPageToken, files(id, name)',
        ))
        folders.extend(resp.get('files', []))
        page_token = resp.get('nextPageToken')
        if not page_token:
            break
    return sorted(folders, key=lambda f: f['name'].lower())


def list_files_recursive(svc, drive_id: str, folder_id: str,
                         max_depth: int = MAX_DEPTH) -> list[dict]:
    """All non-folder files under folder_id (BFS). File shortcuts are resolved
    to their target id/mime; folder shortcuts are skipped (cycle risk).
    Returns [{'id','name','mimeType','modifiedTime','size'}]."""
    files: list[dict] = []
    seen_ids: set[str] = set()
    queue: list[tuple[str, int]] = [(folder_id, 0)]

    while queue:
        current_id, depth = queue.pop(0)
        page_token = None
        while True:
            resp = _execute(svc.files().list(
                q=f"'{current_id}' in parents and trashed=false",
                corpora='drive', driveId=drive_id,
                includeItemsFromAllDrives=True, supportsAllDrives=True,
                pageSize=200, pageToken=page_token,
                fields=_FILE_FIELDS,
            ))
            for f in resp.get('files', []):
                mime = f.get('mimeType', '')
                if mime == FOLDER_MIME:
                    if depth + 1 <= max_depth:
                        queue.append((f['id'], depth + 1))
                    continue
                if mime == SHORTCUT_MIME:
                    target = f.get('shortcutDetails') or {}
                    target_mime = target.get('targetMimeType', '')
                    if target_mime == FOLDER_MIME or not target.get('targetId'):
                        continue
                    f = {**f, 'id': target['targetId'], 'mimeType': target_mime}
                if f['id'] in seen_ids:
                    continue
                seen_ids.add(f['id'])
                files.append({
                    'id':           f['id'],
                    'name':         f.get('name', ''),
                    'mimeType':     f.get('mimeType', ''),
                    'modifiedTime': f.get('modifiedTime', ''),
                    'size':         f.get('size', ''),
                    'last_modified_by': (f.get('lastModifyingUser') or {})
                                        .get('emailAddress', ''),
                })
            page_token = resp.get('nextPageToken')
            if not page_token:
                break
    return files


def list_permission_emails(svc, file_id: str) -> list[str]:
    """Lowercased email addresses of user/group permissions on a file or
    folder (who it is shared with). Viewer membership on the shared drive is
    normally enough to read these; callers should treat failures as
    best-effort (return value simply feeds website inference)."""
    emails: list[str] = []
    page_token = None
    while True:
        resp = _execute(svc.permissions().list(
            fileId=file_id, supportsAllDrives=True,
            pageSize=100, pageToken=page_token,
            fields='nextPageToken, permissions(type, emailAddress)',
        ))
        for p in resp.get('permissions', []):
            if p.get('type') in ('user', 'group') and p.get('emailAddress'):
                emails.append(p['emailAddress'].lower())
        page_token = resp.get('nextPageToken')
        if not page_token:
            break
    return emails


def is_extractable(f: dict) -> bool:
    mime = f.get('mimeType', '')
    if mime in EXPORT_MIMES:
        return True
    name = (f.get('name') or '').lower()
    if not name.endswith(BINARY_ALLOW_EXTS):
        return False
    try:
        size = int(f.get('size') or 0)
    except (TypeError, ValueError):
        size = 0
    return size <= MAX_BINARY_BYTES


def _download(request) -> bytes:
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def download_file_text(svc, f: dict) -> tuple[str, str]:
    """Download/export a Drive file and extract its text.
    Returns (text, note) — note is non-empty only on skip/failure."""
    mime = f.get('mimeType', '')
    name = f.get('name', '')
    try:
        if mime in EXPORT_MIMES:
            export_mime = EXPORT_MIMES[mime]
            content = _download(svc.files().export_media(
                fileId=f['id'], mimeType=export_mime))
            if export_mime in ('text/plain', 'text/csv'):
                text = content.decode('utf-8', errors='replace')[:doc_extract.TEXT_LIMIT]
                return text, ''
            text, _ = doc_extract.extract_text(content, name, export_mime)
            return text, ''
        content = _download(svc.files().get_media(
            fileId=f['id'], supportsAllDrives=True))
        text, ftype = doc_extract.extract_text(content, name, mime)
        if not text:
            return '', f'extraction failed ({ftype})'
        return text, ''
    except HttpError as e:
        status = e.resp.status if e.resp is not None else 0
        if status == 403 and 'exportsizelimitexceeded' in str(e).lower().replace(' ', ''):
            return '', 'export size limit exceeded'
        return '', f'HTTP {status}'
    except Exception as e:
        return '', f'exception: {e}'
