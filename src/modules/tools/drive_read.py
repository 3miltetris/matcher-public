"""
Read-only Drive tools for the onboarding agent (Phase 3.4)
----------------------------------------------------------
list_folder / read_document / folder_domain_signals. All read-only, all return
JSON-serialisable dicts, all reduce emails to their DOMAIN (a person's address
never reaches the model). Built on drive_client; the folder-domain harvest uses
the same weighting the view's new-client proposal path uses (permissions ×3,
last-modifier ×2, document text ×1, freemail/own/gov filtered out).

Streamlit-free and injected: the caller passes a Drive service (the job builds
one from ADC). Budgets (how many docs, how many chars) are enforced by the
orchestrator, not here; these functions just cap their own output size.
"""

import re

import tldextract

from src.modules import doc_extract, drive_client as dc

# Domains that are never a company signal.
FREEMAIL = {
    'gmail.com', 'googlemail.com', 'yahoo.com', 'ymail.com', 'outlook.com',
    'hotmail.com', 'live.com', 'msn.com', 'aol.com', 'icloud.com', 'me.com',
    'mac.com', 'proton.me', 'protonmail.com', 'gmx.com', 'mail.com', 'zoho.com',
}
OWN_DOMAINS = {'bwcoconsulting.com'}
_GOV_SUFFIXES = ('.gov', '.mil')

_LIST_CAP = 300
_URL_RE = re.compile(r'https?://([^/\s)>\]"\']+)', re.IGNORECASE)
_EMAIL_RE = re.compile(r'[\w.+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})')


def bare_domain(value: str) -> str:
    """Registrable domain of an email, URL or bare host; '' if none."""
    v = str(value or '').strip().lower()
    if '@' in v:
        v = v.split('@', 1)[1]
    ext = tldextract.extract(v)
    # `registered_domain` is deprecated in newer tldextract; prefer the new name.
    return getattr(ext, 'top_domain_under_public_suffix', '') or ext.registered_domain or ''


def is_business_domain(domain: str) -> bool:
    d = (domain or '').lower()
    if not d or d in FREEMAIL or d in OWN_DOMAINS:
        return False
    return not d.endswith(_GOV_SUFFIXES)


def _truncate_middle(text: str, max_chars: int) -> str:
    """Keep the head and tail, drop the middle — documents front-load who/what
    and back-load signatures/letterheads, both useful for identity."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + '\n…[truncated]…\n' + text[-half:]


# ── Tools ────────────────────────────────────────────────────────────────────

def list_folder(svc, drive_id: str, folder_id: str, *, max_depth: int = 3,
                cap: int = _LIST_CAP) -> dict:
    """Files under a folder: name, mime, size, modifiedTime, and the last
    modifier's email DOMAIN (not address). Capped at `cap` entries."""
    files = dc.list_files_recursive(svc, drive_id, folder_id, max_depth=max_depth)
    out = [{
        'file_id':      f['id'],
        'name':         f.get('name', ''),
        'mimeType':     f.get('mimeType', ''),
        'size':         f.get('size', ''),
        'modifiedTime': f.get('modifiedTime', ''),
        'last_modifier_domain': bare_domain(f.get('last_modified_by', '')),
    } for f in files[:cap]]
    return {'folder_id': folder_id, 'count': len(files),
            'truncated': len(files) > cap, 'files': out}


def read_document(svc, file_id: str, *, max_chars: int = 12000) -> dict:
    """Extracted text of one Drive file, truncated middle-out. Reuses
    doc_extract's 15 MB / permanent-skip rules via drive_client. Returns an
    `error` string (never raises) for an unextractable or missing file."""
    try:
        meta = dc._execute(svc.files().get(
            fileId=file_id, supportsAllDrives=True,
            fields='id,name,mimeType,size'))
    except Exception as e:
        return {'file_id': file_id, 'text': '', 'error': f'metadata: {e}'}
    if not dc.is_extractable(meta):
        return {'file_id': file_id, 'name': meta.get('name', ''), 'text': '',
                'error': f'not extractable ({meta.get("mimeType", "")})'}
    try:
        text, _kind = dc.download_file_text(svc, meta)
    except Exception as e:
        return {'file_id': file_id, 'name': meta.get('name', ''), 'text': '',
                'error': f'extract: {e}'}
    full = len(text or '')
    return {'file_id': file_id, 'name': meta.get('name', ''),
            'text': _truncate_middle(text or '', max_chars),
            'chars': full, 'truncated': full > max_chars, 'error': None}


def folder_domain_signals(svc, drive_id: str, folder_id: str, *,
                          max_docs_scanned: int = 8) -> dict:
    """Ranked business-domain candidates for the folder's owner, by weight:
    share permissions ×3, file last-modifier ×2, domains in document text ×1.
    Freemail / our own / government domains are dropped."""
    weights: dict[str, float] = {}

    def add(domain: str, w: float):
        d = (domain or '').lower()
        if is_business_domain(d):
            weights[d] = weights.get(d, 0.0) + w

    # Folder-level share permissions (×3)
    try:
        for e in dc.list_permission_emails(svc, folder_id):
            add(bare_domain(e), 3.0)
    except Exception:
        pass

    files = dc.list_files_recursive(svc, drive_id, folder_id)
    for f in files:
        add(bare_domain(f.get('last_modified_by', '')), 2.0)   # last modifier ×2

    # Document text: emails + URLs in the first few extractable docs (×1)
    scanned = 0
    for f in files:
        if scanned >= max_docs_scanned:
            break
        if not dc.is_extractable(f):
            continue
        try:
            text, _ = dc.download_file_text(svc, f)
        except Exception:
            continue
        scanned += 1
        for m in _EMAIL_RE.findall(text or ''):
            add(bare_domain(m), 1.0)
        for m in _URL_RE.findall(text or ''):
            add(bare_domain(m), 1.0)

    ranked = sorted(({'domain': d, 'weight': round(w, 1)}
                     for d, w in weights.items()),
                    key=lambda r: -r['weight'])
    return {'folder_id': folder_id, 'docs_scanned': scanned, 'candidates': ranked[:15]}
