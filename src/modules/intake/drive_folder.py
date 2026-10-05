"""
drive_folder.py — the intake's {Company}_INTERNAL folder in the client shared
drive: find or create it, copy the founder's uploads in, write the responses
Google Doc.

Requires the intake-web@ service account as a Content Manager of the shared
drive and DRIVE_WRITE_SCOPES — the only write access to Drive in this repo.

Reuse, in order: the folder a previous intake stored for this company → a
folder Drive Sync already assigned to it → a single exact/contains name match
anywhere in the drive (src/modules/folder_match, the matcher Drive Sync uses).
Anything weaker or ambiguous creates a NEW folder under the intake parent and
flags the profile for review: writing a founder's documents into another
client's folder is the one outcome that must never happen.
"""

import io
import re

from googleapiclient.http import MediaIoBaseUpload

from src.modules import drive_client as dc
from src.modules import folder_match as fm

_BAD_CHARS = re.compile(r'[/\\:*?"<>|]+')
FOLDER_SUFFIX = '_INTERNAL'
DOC_MIME = 'application/vnd.google-apps.document'


def folder_name(legal_name: str) -> str:
    """'Acme Robotics, Inc.' → 'Acme Robotics, Inc._INTERNAL' — spaces kept
    (the drive's existing {Client Name}_INTERNAL convention)."""
    name = _BAD_CHARS.sub(' ', str(legal_name or ''))
    name = re.sub(r'\s+', ' ', name).strip(' .')
    return f'{name or "Unnamed company"}{FOLDER_SUFFIX}'


def folder_url(folder_id: str) -> str:
    return f'https://drive.google.com/drive/folders/{folder_id}'


def _list(svc, drive_id: str, q: str, fields: str = 'nextPageToken, files(id, name, parents)'):
    out, token = [], None
    while True:
        res = dc._execute(svc.files().list(
            q=q, corpora='drive', driveId=drive_id, includeItemsFromAllDrives=True,
            supportsAllDrives=True, pageSize=1000, fields=fields, pageToken=token))
        out.extend(res.get('files', []))
        token = res.get('nextPageToken')
        if not token:
            return out


def _alive(svc, folder_id: str) -> bool:
    try:
        meta = dc._execute(svc.files().get(fileId=folder_id, supportsAllDrives=True,
                                           fields='id, trashed, mimeType'))
    except Exception:
        return False
    return not meta.get('trashed') and meta.get('mimeType') == dc.FOLDER_MIME


def match_existing(company_name: str, folders: list[dict]) -> tuple[dict | None, str]:
    """(folder, tier) — tier is exact|contains when exactly one folder matches
    at that tier, 'ambiguous' when several do, 'fuzzy' when only a fuzzy match
    exists (reported, never reused), else 'none'."""
    target = fm.normalize(company_name)
    if not target:
        return None, 'none'
    internal = [f for f in folders if fm.INTERNAL_RE.search(f.get('name', ''))]
    norms    = {f['id']: fm.normalize(f['name']) for f in internal}
    exact = [f for f in internal if norms[f['id']] == target]
    if len(exact) == 1:
        return exact[0], 'exact'
    if len(exact) > 1:
        return None, 'ambiguous'
    contains = [f for f in internal if norms[f['id']]
                and len(min(norms[f['id']], target, key=len)) >= 5
                and (norms[f['id']] in target or target in norms[f['id']])]
    if len(contains) == 1:
        return contains[0], 'contains'
    if len(contains) > 1:
        return None, 'ambiguous'
    _, tier, _ = fm.match_folder(target, norms)
    return None, 'fuzzy' if tier == 'fuzzy' else 'none'


def find_or_create(svc, drive_id: str, parent_id: str, company_name: str,
                   known_ids: list[str] = ()) -> dict:
    """{folder_id, folder_url, created, match, needs_review}."""
    for fid in known_ids:
        if fid and _alive(svc, fid):
            return {'folder_id': fid, 'folder_url': folder_url(fid), 'created': False,
                    'match': 'stored', 'needs_review': False}
    folders = _list(svc, drive_id, f"mimeType='{dc.FOLDER_MIME}' and trashed=false")
    found, tier = match_existing(company_name, folders)
    if found:
        return {'folder_id': found['id'], 'folder_url': folder_url(found['id']),
                'created': False, 'match': tier, 'needs_review': False}
    res = dc._execute(svc.files().create(
        body={'name': folder_name(company_name), 'mimeType': dc.FOLDER_MIME,
              'parents': [parent_id]},
        supportsAllDrives=True, fields='id'))
    return {'folder_id': res['id'], 'folder_url': folder_url(res['id']), 'created': True,
            'match': tier, 'needs_review': tier in ('ambiguous', 'fuzzy')}


def _children(svc, drive_id: str, folder_id: str) -> list[dict]:
    return _list(svc, drive_id, f"'{folder_id}' in parents and trashed=false",
                 fields='nextPageToken, files(id, name, mimeType, appProperties)')


def copy_upload(svc, drive_id: str, folder_id: str, filename: str, content: bytes,
                mime: str, upload_id: str, date_prefix: str) -> str:
    """Upload one file into the folder; idempotent by upload_id (stored as an
    appProperty), date-prefixed on a name collision. Returns the file id."""
    existing = _children(svc, drive_id, folder_id)
    for f in existing:
        if (f.get('appProperties') or {}).get('intake_upload_id') == upload_id:
            return f['id']
    name = filename
    if any(f['name'] == name for f in existing):
        name = f'{date_prefix}_{filename}'
    res = dc._execute(svc.files().create(
        body={'name': name, 'parents': [folder_id],
              'appProperties': {'intake_upload_id': upload_id}},
        media_body=MediaIoBaseUpload(io.BytesIO(content), mimetype=mime, resumable=False),
        supportsAllDrives=True, fields='id'))
    return res['id']


def write_doc(svc, drive_id: str, folder_id: str, title: str, html: str,
              session_id: str) -> dict:
    """Create the responses Doc (HTML converted on upload), or replace the
    content of the one this session already wrote. {doc_id, doc_url}."""
    media = MediaIoBaseUpload(io.BytesIO(html.encode('utf-8')), mimetype='text/html',
                              resumable=False)
    for f in _children(svc, drive_id, folder_id):
        if (f.get('appProperties') or {}).get('intake_session_id') == session_id:
            dc._execute(svc.files().update(fileId=f['id'], media_body=media,
                                           supportsAllDrives=True, fields='id'))
            return {'doc_id': f['id'], 'doc_url': f'https://docs.google.com/document/d/{f["id"]}'}
    res = dc._execute(svc.files().create(
        body={'name': title, 'mimeType': DOC_MIME, 'parents': [folder_id],
              'appProperties': {'intake_session_id': session_id}},
        media_body=media, supportsAllDrives=True, fields='id'))
    return {'doc_id': res['id'], 'doc_url': f'https://docs.google.com/document/d/{res["id"]}'}
