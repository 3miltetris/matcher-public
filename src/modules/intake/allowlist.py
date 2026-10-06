"""
allowlist.py — who may open the DD intake form.

`admin-config/intake_allowlist.json` holds approved company domains and
approved individual addresses, edited by admins in the Matcher's Admin Portal
and read by intake-web before it emails a sign-in link. It sits under
`admin-config/`, not `intake/`, because lifecycle rules delete parts of
`intake/`.

An address passes when it is listed exactly, or when its registrable domain is
listed (so `jo@eng.acme.com` passes for `acme.com`). Freemail domains can never
be approved as a whole — approving `gmail.com` would open the form to anyone —
so a founder on a freemail address is approved by exact address instead.

Streamlit-free and imported by both the Admin Portal and intake-web, so it pulls
in nothing heavier than tldextract.
"""

import json
import re
from datetime import datetime, timezone

import tldextract

BUCKET = 'cc-matcher-bucket-jeg-v1'
BLOB   = 'admin-config/intake_allowlist.json'

_MAX_HISTORY = 200

# Mailbox providers anyone can sign up to. Kept here rather than imported from
# fathom_client, whose GENERIC_DOMAINS also lists our own and government
# domains and would drag requests into the public image.
FREEMAIL = frozenset({
    'gmail.com', 'googlemail.com', 'yahoo.com', 'ymail.com', 'rocketmail.com',
    'hotmail.com', 'outlook.com', 'live.com', 'msn.com', 'aol.com',
    'icloud.com', 'me.com', 'mac.com', 'protonmail.com', 'proton.me', 'pm.me',
    'zoho.com', 'gmx.com', 'gmx.net', 'mail.com', 'yandex.com', 'hey.com',
    'fastmail.com', 'tutanota.com', 'comcast.net', 'verizon.net', 'att.net',
    'sbcglobal.net', 'cox.net', 'charter.net',
})

_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')

# The bundled public-suffix snapshot: no network fetch at runtime, and the
# same answer in the Streamlit app and the intake service.
_extract = tldextract.TLDExtract(suffix_list_urls=())


def norm_email(value) -> str:
    return str(value or '').strip().lower()


def valid_email(value) -> bool:
    return bool(_EMAIL_RE.match(norm_email(value)))


def normalize_domain(value) -> str:
    """Registrable domain for a domain, URL or email address; '' if none.
    `https://www.Acme.com/about` → `acme.com`, `@eng.acme.co.uk` → `acme.co.uk`."""
    v = str(value or '').strip().lower()
    if '@' in v:
        v = v.rsplit('@', 1)[1]
    ext = _extract(v)
    return f'{ext.domain}.{ext.suffix}' if ext.domain and ext.suffix else ''


def email_domain(email) -> str:
    return normalize_domain(norm_email(email).rsplit('@', 1)[-1]) if '@' in str(email or '') else ''


# ── Store ────────────────────────────────────────────────────────────────────

def empty_doc() -> dict:
    return {'domains': [], 'emails': [], 'updated_at': '', 'updated_by': '', 'history': []}


def _clean(doc: dict) -> dict:
    out = empty_doc()
    if isinstance(doc, dict):
        out.update(doc)
    out['domains'] = sorted({d for d in map(normalize_domain, out.get('domains') or [])
                             if d and d not in FREEMAIL})
    out['emails']  = sorted({e for e in map(norm_email, out.get('emails') or []) if valid_email(e)})
    out['history'] = [h for h in (out.get('history') or []) if isinstance(h, dict)]
    return out


def load(client, bucket: str = BUCKET) -> dict:
    """The allowlist doc. A missing blob is an empty list (nobody approved);
    a read error propagates — callers decide, and must never treat it as
    'everyone approved'."""
    blob = client.bucket(bucket).blob(BLOB)
    if not blob.exists():
        return empty_doc()
    return _clean(json.loads(blob.download_as_bytes()))


def check_entries(domains, emails) -> tuple[list[str], list[str], list[str]]:
    """(domains, emails, problems): normalised entries plus a readable reason
    for each one that was refused."""
    good_d, good_e, problems = set(), set(), []
    for raw in domains:
        raw = str(raw or '').strip()
        if not raw:
            continue
        d = normalize_domain(raw)
        if not d:
            problems.append(f'`{raw}` is not a domain')
        elif d in FREEMAIL:
            problems.append(f'`{d}` is a public email provider — approve individual addresses instead')
        else:
            good_d.add(d)
    for raw in emails:
        raw = str(raw or '').strip()
        if not raw:
            continue
        if valid_email(raw):
            good_e.add(norm_email(raw))
        else:
            problems.append(f'`{raw}` is not an email address')
    return sorted(good_d), sorted(good_e), problems


def save(client, domains, emails, *, actor: str, note: str = '',
         bucket: str = BUCKET) -> dict:
    """Overwrite the allowlist, appending a history entry. Raises ValueError
    listing every refused entry, so nothing is half-saved."""
    domains, emails, problems = check_entries(domains, emails)
    if problems:
        raise ValueError('; '.join(problems))
    current = load(client, bucket)
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    doc = {
        'domains': domains, 'emails': emails,
        'updated_at': stamp, 'updated_by': actor or 'local-dev',
        'history': [*current['history'],
                    {'at': stamp, 'by': actor or 'local-dev', 'note': note}][-_MAX_HISTORY:],
    }
    client.bucket(bucket).blob(BLOB).upload_from_string(
        json.dumps(doc), content_type='application/json')
    return doc


def is_approved(doc: dict, email) -> bool:
    e = norm_email(email)
    if not valid_email(e):
        return False
    if e in set(doc.get('emails') or []):
        return True
    d = email_domain(e)
    return bool(d) and d not in FREEMAIL and d in set(doc.get('domains') or [])
