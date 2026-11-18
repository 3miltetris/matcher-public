"""
roadmap.booking_hs — the HubSpot read/write layer for the call-booking roadmap.

A sibling to hubspot_roadmap.py. Same trust model (a poll names only a contact;
every field is re-read from HubSpot with our token) and same env-overridable
property convention, but a SEPARATE queue, status, and output-property set so the
booking flow and the event/kiosk roadmap flow never touch each other's state.

The booking flow is driven by a Cloud Scheduler poll of HubSpot (the HubSpot tier
has no "send a webhook" workflow action), exactly like the event roadmap's
roadmap-poll. A contact is a booking when BOOKING_PROP_TRIGGER is set (Instantly
mirrors a booked call onto the contact); we exclude the program already pitched in
the cold email (BOOKING_PROP_PITCHED), match the company's website against the
grant store, and write three programs back as structured properties for the boss's
HubSpot/Instantly email workflows to send on.

Input properties (set by Instantly/HubSpot on the contact, env-overridable):
  BOOKING_PROP_TRIGGER      the "call booked" signal to poll on   (default 'call_booked_at')
  BOOKING_PROP_PITCHED      the program already cold-emailed       (default 'pitched_program')
  BOOKING_PROP_WEBSITE      company URL                            (default 'website')
  BOOKING_PROP_MATERIALS    a deck/materials URL sent after booking (default 'call_materials_url')
  BOOKING_PROP_DESCRIPTION  optional existing HubSpot description   (default '' — unused)

Output properties written back (auto-created on first run, fixed names so the
boss's workflow can trigger on them):
  call_roadmap_status        'processing' | 'complete' | 'no_match' | 'no_input' | 'error'
  call_roadmap_generated_at  ISO timestamp (also the stale-claim marker)
  call_pitched_program       echo of the excluded program (for the email's "In addition to X")
  call_program_{1..3}_name / _summary / _link
"""

import os

from src.modules.intake.hubspot_sync import HubSpot, HubSpotError
from src.modules.roadmap.hubspot_roadmap import (company_from_website,  # noqa: F401
                                                 contact_ref_from_webhook)

# Input fields (env-overridable to match whatever HubSpot/Instantly actually set).
PROP_TRIGGER     = os.environ.get('BOOKING_PROP_TRIGGER', 'call_booked_at')
PROP_PITCHED     = os.environ.get('BOOKING_PROP_PITCHED', 'pitched_program')
PROP_WEBSITE     = os.environ.get('BOOKING_PROP_WEBSITE', 'website')
PROP_MATERIALS   = os.environ.get('BOOKING_PROP_MATERIALS', 'call_materials_url')
PROP_DESCRIPTION = os.environ.get('BOOKING_PROP_DESCRIPTION', '')

# Output fields (fixed — the boss's workflow triggers on these).
PROP_STATUS       = 'call_roadmap_status'
PROP_GENERATED_AT = 'call_roadmap_generated_at'
PROP_LASTMOD      = 'hs_lastmodifieddate'   # HubSpot-maintained; changes on any edit
PROP_PITCHED_ECHO = 'call_pitched_program'

STATUS_PROCESSING = 'processing'
STATUS_COMPLETE   = 'complete'
STATUS_NO_MATCH   = 'no_match'
STATUS_NO_INPUT   = 'no_input'
STATUS_ERROR      = 'error'

# How many programs the roadmap emails. Three per the feature request.
N_PROGRAMS = 3

_STD_PROPS = ['firstname', 'lastname', 'email', 'company', 'website']


def program_props(i: int) -> dict[str, str]:
    """The three property names for program slot `i` (1-based)."""
    return {'name': f'call_program_{i}_name',
            'summary': f'call_program_{i}_summary',
            'link': f'call_program_{i}_link'}


# Property definitions this feature creates on the contact, in the (name, label,
# type, fieldType) shape hubspot_sync.ensure_properties uses.
def _output_defs() -> list[tuple[str, str, str, str]]:
    defs = [
        (PROP_STATUS,       'Call roadmap status',    'string', 'text'),
        (PROP_GENERATED_AT, 'Call roadmap generated', 'string', 'text'),
        (PROP_PITCHED_ECHO, 'Pitched program',        'string', 'text'),
    ]
    for i in range(1, N_PROGRAMS + 1):
        p = program_props(i)
        defs += [
            (p['name'],    f'Program {i} name',    'string', 'text'),
            (p['summary'], f'Program {i} summary', 'string', 'textarea'),
            (p['link'],    f'Program {i} link',    'string', 'text'),
        ]
    return defs


def ensure_output_properties(hs: HubSpot) -> list[str]:
    """Create the call_roadmap_* / call_program_* contact properties if missing.
    Idempotent."""
    have = {p['name'] for p in hs._req('GET', '/crm/v3/properties/contacts')
            .get('results', [])}
    created = []
    for name, label, ptype, field_type in _output_defs():
        if name in have:
            continue
        hs._req('POST', '/crm/v3/properties/contacts', json={
            'name': name, 'label': label, 'type': ptype,
            'fieldType': field_type, 'groupName': 'contactinformation'})
        created.append(name)
    return created


def _input_props() -> list[str]:
    props = _STD_PROPS + [PROP_TRIGGER, PROP_PITCHED, PROP_WEBSITE, PROP_MATERIALS]
    if PROP_DESCRIPTION:
        props.append(PROP_DESCRIPTION)
    return list(dict.fromkeys(props))


def get_contact(hs: HubSpot, *, contact_id: str | None = None,
                email: str | None = None) -> dict:
    """The contact's properties as a flat dict, plus its id under '_id'. Looks up
    by id when given, else by email. Raises HubSpotError if neither resolves."""
    props = _input_props()
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


def booking_inputs(contact: dict) -> dict:
    """Pull the booking inputs out of a fetched contact into a stable shape. No
    company-name field, so the display name falls back to the website domain, then
    the person's name."""
    website = str(contact.get(PROP_WEBSITE) or '').strip()
    name = (str(contact.get('company') or '').strip()
            or company_from_website(website)
            or ' '.join(x for x in (contact.get('firstname'), contact.get('lastname')) if x).strip())
    desc = str(contact.get(PROP_DESCRIPTION) or '').strip() if PROP_DESCRIPTION else ''
    return {
        'contact_id':   contact.get('_id', ''),
        'email':        str(contact.get('email') or '').strip().lower(),
        'company':      name,
        'website':      website,
        'description':  desc,
        'pitched':      str(contact.get(PROP_PITCHED) or '').strip(),
        'materials_url': str(contact.get(PROP_MATERIALS) or '').strip(),
    }


def write_booking(hs: HubSpot, contact_id: str, *, programs: list[dict],
                  pitched_name: str, status: str, generated_at: str) -> None:
    """Patch the call_roadmap_* / call_program_* properties onto the contact.
    `programs` is up to N_PROGRAMS dicts {name, summary, link}; unused slots are
    cleared so a re-run with fewer matches never leaves a stale old program."""
    ensure_output_properties(hs)
    props: dict[str, str] = {
        PROP_STATUS: status,
        PROP_GENERATED_AT: generated_at,
        PROP_PITCHED_ECHO: pitched_name or '',
    }
    for i in range(1, N_PROGRAMS + 1):
        p = program_props(i)
        prog = programs[i - 1] if i <= len(programs) else {}
        props[p['name']]    = str(prog.get('name') or '')
        props[p['summary']] = str(prog.get('summary') or '')
        props[p['link']]    = str(prog.get('link') or '')
    hs._req('PATCH', f'/crm/v3/objects/contacts/{contact_id}',
            json={'properties': props})


# ── Polling queue (no workflow webhook on this HubSpot tier) ──────────────────

def _poll_props() -> list[str]:
    return list(dict.fromkeys([
        'email', PROP_TRIGGER, PROP_WEBSITE, PROP_PITCHED, PROP_MATERIALS,
        PROP_STATUS, PROP_GENERATED_AT, PROP_LASTMOD]))


# The re-run / error-recovery searches page through results instead of reading one
# capped page: contacts that will never qualify (unchanged inputs, an exhausted
# retry budget) otherwise fill that page and hide the one that does. Newest-
# modified first, because a genuine change — a website or deck arriving, a fresh
# error — is itself an edit, so qualifying contacts sort to the front. The cap
# bounds the HubSpot calls per poll.
SCAN_CAP = 500
_RECENT_FIRST = [{'propertyName': PROP_LASTMOD, 'direction': 'DESCENDING'}]


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


def _search_iter(hs: HubSpot, filter_groups: list, *, sorts=None,
                 cap: int = SCAN_CAP):
    """Lazily page a contact search (HubSpot's `after` cursor) up to `cap` rows.
    Lazy, so a caller that finds what it needs early stops paging."""
    after, seen = None, 0
    while seen < cap:
        body = {'filterGroups': filter_groups, 'properties': _poll_props(),
                'limit': max(1, min(100, cap - seen))}
        if sorts:
            body['sorts'] = sorts
        if after:
            body['after'] = after
        res = hs._req('POST', '/crm/v3/objects/contacts/search', json=body)
        results = res.get('results', [])
        for r in results:
            d = dict(r.get('properties') or {})
            d['_id'] = r.get('id')
            seen += 1
            yield d
        after = ((res.get('paging') or {}).get('next') or {}).get('after')
        if not after or not results:
            return


def pending_bookings(hs: HubSpot, *, limit: int = 25) -> list[dict]:
    """Contacts with a booked call (trigger set) that have not been processed yet
    (no call_roadmap_status). Oldest first, so none starve."""
    return _search(hs, [{'filters': [
        {'propertyName': PROP_TRIGGER, 'operator': 'HAS_PROPERTY'},
        {'propertyName': PROP_STATUS, 'operator': 'NOT_HAS_PROPERTY'},
    ]}], limit, sorts=[{'propertyName': 'createdate', 'direction': 'ASCENDING'}])


def rerun_candidates(hs: HubSpot, *, cap: int = SCAN_CAP):
    """Already-complete bookings that now carry a materials URL (lazy, newest-
    modified first). Whether a re-run is actually warranted (the materials changed
    since the last build) is decided by comparing the run-record fingerprint —
    this only narrows the search."""
    return _search_iter(hs, [{'filters': [
        {'propertyName': PROP_TRIGGER, 'operator': 'HAS_PROPERTY'},
        {'propertyName': PROP_STATUS, 'operator': 'EQ', 'value': STATUS_COMPLETE},
        {'propertyName': PROP_MATERIALS, 'operator': 'HAS_PROPERTY'},
    ]}], sorts=_RECENT_FIRST, cap=cap)


def reinput_candidates(hs: HubSpot, *, cap: int = SCAN_CAP):
    """Terminal bookings that produced no roadmap because the inputs were too thin
    at build time ('no_input') or matched nothing ('no_match'). If the founder's
    website or materials arrived since, the run-record fingerprint will differ and
    claim_pending_bookings re-runs them — these statuses are only ever written with
    a saved record, so that fingerprint gate is exact (no record ⇒ not re-run).
    Unlike rerun_candidates this does NOT require a materials URL: a bare website
    added to a 'no_input' contact must re-run too. One search (the two statuses
    OR'd as filter groups), lazy and newest-modified first."""
    return _search_iter(hs, [
        {'filters': [
            {'propertyName': PROP_TRIGGER, 'operator': 'HAS_PROPERTY'},
            {'propertyName': PROP_STATUS, 'operator': 'EQ', 'value': status},
        ]} for status in (STATUS_NO_INPUT, STATUS_NO_MATCH)
    ], sorts=_RECENT_FIRST, cap=cap)


def error_bookings(hs: HubSpot, *, cap: int = SCAN_CAP):
    """Bookings whose last run errored (lazy, newest-modified first). Excluded
    from pending/rerun/processing, so without this a transient failure (an
    Anthropic 5xx, a flaky fetch) leaves the contact stuck at 'error' forever.
    claim_pending_bookings recovers them under a staleness gate and a bounded
    retry budget. NOT oldest-first: contacts that exhausted their budget stop
    being edited, so newest-modified sorts them behind the ones still retryable
    instead of letting them hold the first page forever."""
    return _search_iter(hs, [{'filters': [
        {'propertyName': PROP_TRIGGER, 'operator': 'HAS_PROPERTY'},
        {'propertyName': PROP_STATUS, 'operator': 'EQ', 'value': STATUS_ERROR},
    ]}], sorts=_RECENT_FIRST, cap=cap)


def processing_bookings(hs: HubSpot, *, limit: int = 25) -> list[dict]:
    """Contacts currently claimed (status == processing). Used to recover claims
    whose generation was killed before it set a terminal status."""
    return _search(hs, [{'filters': [
        {'propertyName': PROP_STATUS, 'operator': 'EQ', 'value': STATUS_PROCESSING},
    ]}], limit)


def claim(hs: HubSpot, contact_id: str, now_iso: str) -> None:
    """Mark a contact in-flight so the next poll does not pick it again. The
    timestamp doubles as the staleness marker for recovery."""
    ensure_output_properties(hs)
    hs._req('PATCH', f'/crm/v3/objects/contacts/{contact_id}', json={'properties': {
        PROP_STATUS: STATUS_PROCESSING, PROP_GENERATED_AT: now_iso}})
