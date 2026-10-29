"""
roadmap.hubspot_roadmap — read the founder's form answers from HubSpot and write
the roadmap link back.

Builds on the intake service's HubSpot client (src.modules.intake.hubspot_sync).
The trust model: the webhook tells us only *which* contact (an id or email); we
re-read every field from HubSpot with our own token, so a forged webhook payload
cannot inject content, only trigger work (which the endpoint's shared secret and
rate limit guard).

The form's property names are env-configurable, defaulting to the live
"Event Roadmap Form" (HubSpot form 395054d9-…) the boss built:

  ROADMAP_PROP_DESCRIPTION  what-the-company-does field   (default 'event_company_description')
  ROADMAP_PROP_WEBSITE      company URL field             (default 'website')
  ROADMAP_PROP_DECK         pitch-deck file-URL field     (default 'event_pitch_deck')

Output properties written back onto the contact (auto-created on first run):
  roadmap_pdf_url        the hosted PDF link (the boss's HubSpot workflow trigger)
  roadmap_status         'complete' | 'no_match' | 'no_input' | 'error'
  roadmap_generated_at   ISO date
"""

import json
import os

import tldextract

from src.modules.intake.hubspot_sync import HubSpot, HubSpotError

# Where the public roadmap PDFs land in HubSpot's File Manager.
FILES_FOLDER = os.environ.get('ROADMAP_FILES_FOLDER', '/funding-roadmaps')

# Input fields (env-overridable to match the form). Defaults = the live form.
PROP_DESCRIPTION = os.environ.get('ROADMAP_PROP_DESCRIPTION', 'event_company_description')
PROP_WEBSITE     = os.environ.get('ROADMAP_PROP_WEBSITE', 'website')
PROP_DECK        = os.environ.get('ROADMAP_PROP_DECK', 'event_pitch_deck')
PROP_LEAD_SOURCE = os.environ.get('ROADMAP_PROP_LEAD_SOURCE', 'roadmap_lead_source')

# Output fields (fixed — the boss's workflow triggers on roadmap_pdf_url).
PROP_URL          = 'roadmap_pdf_url'
PROP_STATUS       = 'roadmap_status'
PROP_GENERATED_AT = 'roadmap_generated_at'

# roadmap_status values. 'processing' is the in-flight claim the poller sets so a
# contact is picked up exactly once; the rest are terminal.
STATUS_PROCESSING = 'processing'

# The form carries no company-name field, so fall back to the website domain.
_STD_PROPS = ['firstname', 'lastname', 'email', 'mobilephone', 'phone', 'company', 'website']


def company_from_website(url: str) -> str:
    """A display company name from a URL: the registrable-domain stem, spaced
    and title-cased ('acme-robotics.com' -> 'Acme Robotics'). '' when there is
    no usable domain."""
    stem = tldextract.extract(url or '').domain
    return stem.replace('-', ' ').replace('_', ' ').title() if stem else ''


def contact_ref_from_webhook(payload: dict) -> tuple[str | None, str | None]:
    """(contact_id, email) from whatever a HubSpot workflow webhook sends. Tries
    the id under every spelling HubSpot uses (objectId/vid/…), and the email from
    a flat field or a {'value': …} property, so we never depend on one exact
    envelope shape — the id/email is only used to RE-FETCH from HubSpot."""
    if not isinstance(payload, dict):
        return None, None
    cid = None
    for k in ('contact_id', 'objectId', 'object_id', 'vid', 'contactId', 'id'):
        v = payload.get(k)
        if v not in (None, ''):
            cid = str(v)
            break
    if cid is None:
        obj = payload.get('object') or payload.get('contact') or {}
        if isinstance(obj, dict):
            for k in ('objectId', 'id', 'vid'):
                if obj.get(k):
                    cid = str(obj[k])
                    break
    email = payload.get('email')
    props = payload.get('properties')
    if not email and isinstance(props, dict):
        e = props.get('email')
        email = e.get('value') if isinstance(e, dict) else e
    email = str(email).strip().lower() if email else None
    return cid, email

# Property definitions this feature creates on the contact object, in the shape
# hubspot_sync.ensure_properties uses (name, label, type, fieldType).
_OUTPUT_DEFS = [
    (PROP_URL,          'Funding roadmap PDF', 'string', 'text'),
    (PROP_STATUS,       'Roadmap status',      'string', 'text'),
    (PROP_GENERATED_AT, 'Roadmap generated',   'string', 'text'),
]


def ensure_output_properties(hs: HubSpot) -> list[str]:
    """Create the roadmap_* contact properties if missing. Idempotent."""
    have = {p['name'] for p in hs._req('GET', '/crm/v3/properties/contacts')
            .get('results', [])}
    created = []
    for name, label, ptype, field_type in _OUTPUT_DEFS:
        if name in have:
            continue
        hs._req('POST', '/crm/v3/properties/contacts', json={
            'name': name, 'label': label, 'type': ptype,
            'fieldType': field_type, 'groupName': 'contactinformation'})
        created.append(name)
    return created


def get_contact(hs: HubSpot, *, contact_id: str | None = None,
                email: str | None = None) -> dict:
    """The contact's properties as a flat dict, plus its id under '_id'. Looks up
    by id when given, else by email. Raises HubSpotError if neither resolves."""
    props = list(dict.fromkeys(
        _STD_PROPS + [PROP_DESCRIPTION, PROP_WEBSITE, PROP_DECK, PROP_LEAD_SOURCE]))
    if contact_id:
        res = hs._req('GET', f'/crm/v3/objects/contacts/{contact_id}',
                      params={'properties': ','.join(props)})
    elif email:
        res = hs._search('contacts', 'email', str(email).strip().lower(), props)
        if not res:
            raise HubSpotError(f'no contact for email {email}')
    else:
        raise HubSpotError('get_contact needs a contact_id or email')
    out = dict(res.get('properties') or {})
    out['_id'] = res.get('id') or contact_id or ''
    return out


def form_inputs(contact: dict) -> dict:
    """Pull the roadmap inputs out of a fetched contact into a stable shape. The
    form has no company-name field, so the display name falls back to the website
    domain, then to the person's name."""
    website = str(contact.get(PROP_WEBSITE) or '').strip()
    name = (str(contact.get('company') or '').strip()
            or company_from_website(website)
            or ' '.join(x for x in (contact.get('firstname'), contact.get('lastname')) if x).strip())
    return {
        'contact_id':  contact.get('_id', ''),
        'email':       str(contact.get('email') or '').strip().lower(),
        'company':     name,
        'website':     website,
        'description': str(contact.get(PROP_DESCRIPTION) or '').strip(),
        'deck_url':    str(contact.get(PROP_DECK) or '').strip(),
        'lead_source': str(contact.get(PROP_LEAD_SOURCE) or '').strip(),
    }


def upload_public_file(hs: HubSpot, content: bytes, filename: str, *,
                       folder_path: str = FILES_FOLDER) -> tuple[str, str]:
    """Upload bytes to HubSpot Files as a PUBLIC, non-indexable file; return
    (public_url, file_id). The URL is a HubSpot CDN link usable in an email/text
    without auth. Uses the raw session (multipart), not _req (which forces a JSON
    content-type). Needs the `files` scope."""
    options = {'access': 'PUBLIC_NOT_INDEXABLE', 'overwrite': False,
               'duplicateValidationStrategy': 'NONE',
               'duplicateValidationScope': 'ENTIRE_PORTAL'}
    resp = hs.http.post(
        'https://api.hubapi.com/files/v3/files',
        headers={'Authorization': f'Bearer {hs.token}'},
        files={'file': (filename, content, 'application/pdf'),
               'options': (None, json.dumps(options), 'application/json'),
               'folderPath': (None, folder_path)},
        timeout=90)
    if not resp.ok:
        raise HubSpotError(f'files upload -> {resp.status_code}: {resp.text[:300]}')
    data = resp.json()
    url = data.get('url') or ''
    if not url:
        raise HubSpotError(f'files upload returned no url: {str(data)[:200]}')
    return url, str(data.get('id') or '')


def write_roadmap(hs: HubSpot, contact_id: str, *, url: str, status: str,
                  generated_at: str) -> None:
    """Patch the roadmap_* properties onto the contact. Creates the properties
    first (idempotent)."""
    ensure_output_properties(hs)
    hs._req('PATCH', f'/crm/v3/objects/contacts/{contact_id}', json={'properties': {
        PROP_URL: url, PROP_STATUS: status, PROP_GENERATED_AT: generated_at}})


# ── Polling fallback (no workflow webhook available on the HubSpot tier) ──────

def _poll_props() -> list[str]:
    return list(dict.fromkeys([
        'email', PROP_DESCRIPTION, PROP_WEBSITE, PROP_STATUS, PROP_GENERATED_AT]))


def _search(hs: HubSpot, filter_groups: list, limit: int, sorts=None) -> list[dict]:
    body = {'filterGroups': filter_groups, 'properties': _poll_props(),
            'limit': max(1, min(int(limit), 100))}
    if sorts:
        body['sorts'] = sorts
    res = hs._req('POST', '/crm/v3/objects/contacts/search', json=body)
    out = []
    for r in res.get('results', []):
        d = dict(r.get('properties') or {})
        d['_id'] = r.get('id')
        out.append(d)
    return out


def pending_contacts(hs: HubSpot, *, limit: int = 25) -> list[dict]:
    """Contacts that submitted the form (have a description) and have not been
    processed yet (no roadmap_status). Oldest first, so none starve."""
    return _search(hs, [{'filters': [
        {'propertyName': PROP_DESCRIPTION, 'operator': 'HAS_PROPERTY'},
        {'propertyName': PROP_STATUS, 'operator': 'NOT_HAS_PROPERTY'},
    ]}], limit, sorts=[{'propertyName': 'createdate', 'direction': 'ASCENDING'}])


def processing_contacts(hs: HubSpot, *, limit: int = 25) -> list[dict]:
    """Contacts currently claimed (roadmap_status == processing). Used to recover
    claims whose generation was killed before it set a terminal status."""
    return _search(hs, [{'filters': [
        {'propertyName': PROP_STATUS, 'operator': 'EQ', 'value': STATUS_PROCESSING},
    ]}], limit)


def claim(hs: HubSpot, contact_id: str, now_iso: str) -> None:
    """Mark a contact in-flight so the next poll does not pick it again. The
    timestamp doubles as the staleness marker for recovery."""
    ensure_output_properties(hs)
    hs._req('PATCH', f'/crm/v3/objects/contacts/{contact_id}', json={'properties': {
        PROP_STATUS: STATUS_PROCESSING, PROP_GENERATED_AT: now_iso}})
