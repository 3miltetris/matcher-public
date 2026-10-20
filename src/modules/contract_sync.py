"""
contract_sync.py — streamlit-free core for the signed-contract sync (Stage: contracts).

Pulls signed HubSpot quotes and their Terms & Conditions PDFs into the clients
shared Google Drive, filing each into the client's own {Client}_INTERNAL/Contracts/
folder, or into a top-level Uncategorized-Contracts folder when the company can't
be matched. Idempotent across the whole shared drive via Drive appProperties
(hubspot_quote_id + kind). Ported from the standalone sync_signed_quotes.py and
rewired to the Matcher's real client→folder mapping (GCS drive-sync-configs/
assignments.json — there is NO Firestore), its folder_match normalisation and the
intake service's Drive-write service account (intake-web@).

Matching (no fuzzy — a contract in the wrong client's folder is a confidentiality
breach): exact bare-domain primary, exact normalised-name fallback, each accepted
only when it resolves to exactly one client. Anything else → Uncategorized-Contracts.

This module holds the pure/testable logic and the injected-client Drive/HubSpot
helpers; jobs/contract_sync_job.py is the thin I/O wrapper.
"""

import html
import io
import re
import time

import requests
import tldextract
from googleapiclient.http import MediaIoBaseUpload

from src.modules import drive_client as dc
from src.modules import folder_match as fm
from src.modules.intake import drive_folder as df

HS_BASE = 'https://api.hubapi.com'
GDOC_MIME = 'application/vnd.google-apps.document'

QUOTE_PROPS = ['hs_title', 'hs_esign_date', 'hs_pdf_download_link', 'hs_terms',
               'hs_quote_number']
COMPANY_PROPS = ['name', 'domain', 'website']

# Matches drive.google.com/file/d/<id>, docs.google.com/document/d/<id>, ?id=<id>
DRIVE_ID_RE = re.compile(r'(?:/file/d/|/document/d/|[?&]id=)([A-Za-z0-9_-]{20,})')
# Excel HYPERLINK formulas leak into some HubSpot company names (SBA imports),
# e.g. =HYPERLINK("https://...", "HOLOCHIP CORP") — same shape contact_import strips.
_HYPERLINK_RE = re.compile(r'=HYPERLINK\s*\(\s*"[^"]*"\s*,\s*"([^"]*)"\s*\)', re.IGNORECASE)

UNCATEGORIZED_NAME = 'Uncategorized-Contracts'
CONTRACTS_NAME = 'Contracts'
ROLE_PROP = 'contract_sync_role'

_extract = tldextract.TLDExtract(suffix_list_urls=())  # offline, bundled snapshot


# ---------------------------------------------------------------- text helpers
def fix_text(s: str) -> str:
    """Repair double-encoded UTF-8 ('ProPlateÂ®' -> 'ProPlate®'). Leaves clean text alone."""
    try:
        return s.encode('cp1252').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def strip_hyperlink(val: str) -> str:
    """Display text from an Excel HYPERLINK formula; unchanged otherwise."""
    text = str(val or '')
    m = _HYPERLINK_RE.match(text.strip())
    return m.group(1) if m else text


def safe_name(s: str) -> str:
    return re.sub(r'[\\/]+', '-', fix_text(s or '')).strip() or 'Untitled'


def bare_domain(url_or_domain: str) -> str:
    """'https://www.Acme-Robotics.com/x' or 'acme.com' → 'acme-robotics.com'. '' if none."""
    ext = _extract(str(url_or_domain or '').strip().lower())
    return f'{ext.domain}.{ext.suffix}' if ext.domain and ext.suffix else ''


def extract_term_ids(hs_terms_html: str) -> list[str]:
    """Drive file IDs embedded in a quote's hs_terms HTML, de-duplicated, in order."""
    ids = DRIVE_ID_RE.findall(html.unescape(hs_terms_html or ''))
    return list(dict.fromkeys(ids))


def term_kind(i: int) -> str:
    return 'terms' if i == 0 else f'terms_{i + 1}'


# ---------------------------------------------------------------- HubSpot
def make_hs_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({'Authorization': f'Bearer {token}'})
    return s


def make_hs_call(session: requests.Session):
    """Return a hs(method, path, **kw) -> dict callable with 429/5xx backoff."""
    def hs(method: str, path: str, **kw) -> dict:
        r = None
        for attempt in range(6):
            r = session.request(method, HS_BASE + path, timeout=60, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return {}
    return hs


def iter_signed_quotes(hs, since_ms: int | None):
    """Yield signed quotes (hs_quote_esign_status == SIGNED), oldest first.
    since_ms (epoch millis) limits to quotes signed on/after it when set."""
    filters = [{'propertyName': 'hs_quote_esign_status', 'operator': 'EQ', 'value': 'SIGNED'}]
    if since_ms is not None:
        filters.append({'propertyName': 'hs_esign_date', 'operator': 'GTE', 'value': str(since_ms)})
    after = None
    while True:
        body = {'filterGroups': [{'filters': filters}], 'properties': QUOTE_PROPS,
                'sorts': [{'propertyName': 'hs_esign_date', 'direction': 'ASCENDING'}],
                'limit': 100}
        if after:
            body['after'] = after
        data = hs('POST', '/crm/v3/objects/quotes/search', json=body)
        yield from data.get('results', [])
        after = data.get('paging', {}).get('next', {}).get('after')
        if not after:
            return


def assoc_ids(hs, from_type: str, obj_id: str, to_type: str) -> list[str]:
    results = hs('GET', f'/crm/v4/objects/{from_type}/{obj_id}/associations/{to_type}').get('results', [])
    # Primary association first, if one is labelled
    results.sort(key=lambda r: not any(t.get('label') == 'Primary' for t in r.get('associationTypes', [])))
    return [str(r['toObjectId']) for r in results]


def company_for_quote(hs, quote_id: str) -> dict | None:
    """Resolve a quote to its company: quote→company, else quote→deal→company.
    Returns {'id','name','domain','website'} or None when no company is found."""
    ids = assoc_ids(hs, 'quotes', quote_id, 'companies')
    if not ids:
        for deal_id in assoc_ids(hs, 'quotes', quote_id, 'deals'):
            ids = assoc_ids(hs, 'deals', deal_id, 'companies')
            if ids:
                break
    if not ids:
        return None
    cid = ids[0]
    props = hs('GET', f'/crm/v3/objects/companies/{cid}',
               params={'properties': ','.join(COMPANY_PROPS)}).get('properties', {})
    return {'id': cid, 'name': strip_hyperlink(props.get('name') or ''),
            'domain': props.get('domain') or '', 'website': props.get('website') or ''}


def download_signed_pdf(url: str) -> bytes:
    r = requests.get(url, timeout=120, allow_redirects=True)
    r.raise_for_status()
    if not r.content.startswith(b'%PDF'):
        raise ValueError(f'Download did not return a PDF (content-type {r.headers.get("content-type")})')
    return r.content


# ---------------------------------------------------------------- matching
def client_key_parts(client_key: str) -> tuple[str, str]:
    """'Acme||https://acme.com' → ('Acme', 'https://acme.com')."""
    name, _, website = (client_key or '').partition('||')
    return name, website


def build_assignment_index(assignments: dict) -> dict:
    """From assignments.json's `assignments` dict (folder_id -> {client_key,
    folder_name, ...}) build lookup indexes:
      domain: bare_domain -> set(client_key)
      name:   normalized name -> set(client_key)   (from folder_name AND client_key name)
      folders: client_key -> [ {folder_id, folder_name, assigned_at} ]
    """
    domain: dict[str, set] = {}
    name: dict[str, set] = {}
    folders: dict[str, list] = {}
    for fid, a in (assignments or {}).items():
        ck = a.get('client_key')
        if not ck:
            continue
        folders.setdefault(ck, []).append({
            'folder_id': fid,
            'folder_name': a.get('folder_name', ''),
            'assigned_at': a.get('assigned_at', ''),
        })
        cname, website = client_key_parts(ck)
        d = bare_domain(website)
        if d:
            domain.setdefault(d, set()).add(ck)
        for raw in (a.get('folder_name', ''), cname):
            n = fm.normalize(raw)
            if n:
                name.setdefault(n, set()).add(ck)
    return {'domain': domain, 'name': name, 'folders': folders}


def resolve_client(company: dict, index: dict) -> tuple[str | None, str]:
    """(client_key | None, reason). Exact bare-domain first, then exact
    normalized-name; each accepted only when it hits exactly one client."""
    d = bare_domain(company.get('domain') or company.get('website'))
    if d:
        keys = index['domain'].get(d)
        if keys and len(keys) == 1:
            return next(iter(keys)), 'domain'
        if keys and len(keys) > 1:
            return None, 'ambiguous_domain'
    n = fm.normalize(company.get('name'))
    if n:
        keys = index['name'].get(n)
        if keys and len(keys) == 1:
            return next(iter(keys)), 'name'
        if keys and len(keys) > 1:
            return None, 'ambiguous_name'
    return None, 'no_match'


def pick_folder(client_key: str, index: dict) -> dict | None:
    """The destination _INTERNAL folder for a matched client. A client with
    several folders is still one client (not ambiguous): prefer an _INTERNAL
    name, then the earliest assigned_at, then folder_id — deterministic."""
    folders = index['folders'].get(client_key) or []
    if not folders:
        return None
    return sorted(folders, key=lambda f: (
        not fm.INTERNAL_RE.search(f['folder_name'] or ''),
        f['assigned_at'] or '~',
        f['folder_id'],
    ))[0]


# ---------------------------------------------------------------- Drive
def find_root_folder(svc, drive_id: str, name: str) -> str | None:
    """A folder of the given name directly under the shared drive root (case-insensitive)."""
    want = name.strip().lower()
    for f in dc.list_child_folders(svc, drive_id, drive_id):
        if (f.get('name') or '').strip().lower() == want:
            return f['id']
    return None


def find_or_create_child(svc, drive_id: str, parent_id: str, name: str,
                         role: str, dry: bool, log=None) -> str | None:
    """Find a child folder of parent_id by name (case-insensitive) or create it,
    tagging it appProperties[ROLE_PROP]=role. Returns the folder id (None in a
    dry run when it would be created)."""
    want = name.strip().lower()
    for f in df._children(svc, drive_id, parent_id):
        if f.get('mimeType') == dc.FOLDER_MIME and (f.get('name') or '').strip().lower() == want:
            return f['id']
    if dry:
        if log:
            log('    [dry-run] would create folder %r under %s', name, parent_id)
        return None
    res = dc._execute(svc.files().create(
        body={'name': name, 'mimeType': dc.FOLDER_MIME, 'parents': [parent_id],
              'appProperties': {ROLE_PROP: role}},
        supportsAllDrives=True, fields='id'))
    return res['id']


def find_quote_files(svc, drive_id: str, quote_id: str) -> list[dict]:
    """Every file anywhere in the shared drive tagged with this quote id.
    Returns [{id, name, parents, appProperties}]."""
    q = (f"trashed=false and appProperties has "
         f"{{ key='hubspot_quote_id' and value='{quote_id}' }}")
    return df._list(svc, drive_id, q,
                    fields='nextPageToken, files(id, name, parents, appProperties)')


def upload_pdf(svc, folder_id: str, name: str, data: bytes, props: dict) -> str:
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype='application/pdf', resumable=False)
    res = dc._execute(svc.files().create(
        body={'name': name, 'parents': [folder_id], 'appProperties': props},
        media_body=media, supportsAllDrives=True, fields='id'))
    return res['id']


def copy_terms(svc, src_id: str, folder_id: str, name: str, props: dict) -> str:
    """Copy a T&C source into folder. A Google Doc is exported to PDF; a real
    PDF is server-side copied (no download)."""
    meta = dc._execute(svc.files().get(fileId=src_id, fields='id,name,mimeType',
                                       supportsAllDrives=True))
    if meta['mimeType'] == GDOC_MIME:
        pdf = dc._execute(svc.files().export(fileId=src_id, mimeType='application/pdf'))
        return upload_pdf(svc, folder_id, name, pdf, props)
    res = dc._execute(svc.files().copy(
        fileId=src_id, body={'name': name, 'parents': [folder_id], 'appProperties': props},
        supportsAllDrives=True, fields='id'))
    return res['id']


def move_file(svc, file_id: str, add_parent: str, remove_parent: str, new_name: str) -> None:
    dc._execute(svc.files().update(
        fileId=file_id, addParents=add_parent, removeParents=remove_parent,
        body={'name': new_name}, supportsAllDrives=True, fields='id'))


# ---------------------------------------------------------------- decision
def decide(existing_parent: str | None, dest_folder_id: str | None,
           uncategorized_id: str | None, matched: bool) -> str:
    """Pure skip/move/create decision for one (quote, kind).
      - no existing file                          -> 'create'
      - already in the correct destination        -> 'skip'
      - matched and currently in uncategorized    -> 'move'
      - anything else (already filed somewhere)    -> 'skip'
    """
    if existing_parent is None:
        return 'create'
    if dest_folder_id is not None and existing_parent == dest_folder_id:
        return 'skip'
    if matched and uncategorized_id is not None and existing_parent == uncategorized_id:
        return 'move'
    return 'skip'


def prefixed(name: str, company_name: str, matched: bool) -> str:
    """In uncategorized, prefix the file name with [HubSpot company name]."""
    if matched:
        return name
    cn = safe_name(company_name) if company_name else ''
    return f'[{cn}] {name}' if cn else name


def strip_prefix(name: str) -> str:
    """Drop a leading '[company] ' prefix added for uncategorized."""
    return re.sub(r'^\[[^\]]*\]\s+', '', name or '')
