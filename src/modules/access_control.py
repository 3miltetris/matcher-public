"""
Access control
--------------
Admin gating for the destructive actions in the Streamlit app (deleting
clients and client profiles) and for the Admin Portal view itself.

Identity comes from IAP: app.py copies the verified
`X-Goog-Authenticated-User-Email` address into `st.session_state.user_email`,
and the Cloud Run service blocks unauthenticated access, so an allow-list of
email addresses is all that is needed here.

Roles:
  * **super admin** — `SUPER_ADMINS`, a code constant. Cannot be edited or
    removed from the UI, and is the only role allowed to change the admin
    list.
  * **admin** — listed in `admin-config/admins.json`, managed through the
    Admin Portal view. May run the destructive actions.
  * everyone else — full read/write use of the pipeline, no delete rights.

The local-dev fallback in app.py (app_password, no IAP header) has no email
attached, so it is treated as a super admin — only the password holder can
reach it, and it never happens behind IAP.
"""

import json
from datetime import datetime, timezone

import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account

# ── Constants ───────────────────────────────────────────────────────────────

BUCKET      = 'cc-matcher-bucket-jeg-v1'
ADMINS_BLOB = 'admin-config/admins.json'

SUPER_ADMINS = ('john@bwcoconsulting.com',)

_SESSION_KEY = 'ac_admin_doc'
_MAX_HISTORY = 100


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_storage_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)


def norm_email(value) -> str:
    return str(value or '').strip().lower()


_SUPER = {norm_email(e) for e in SUPER_ADMINS}


def current_user_email() -> str:
    return norm_email(st.session_state.get('user_email'))


def is_local_dev() -> bool:
    """True when the session authenticated through the app_password fallback
    instead of IAP, so no verified email is available."""
    return not current_user_email()


# ── Admin list store ────────────────────────────────────────────────────────

def _empty_doc() -> dict:
    return {'admins': [], 'updated_at': '', 'updated_by': '', 'history': []}


def load_admins(client: storage.Client | None = None, *, refresh: bool = False) -> dict:
    """`admin-config/admins.json` as {admins, updated_at, updated_by, history}.

    Cached in session state — the navigation checks admin status on every
    rerun. Pass refresh=True after a write. A read failure never grants
    rights: the list falls back to empty (SUPER_ADMINS still apply) and the
    error is reported on the doc for the portal to surface."""
    if not refresh and _SESSION_KEY in st.session_state:
        return st.session_state[_SESSION_KEY]

    doc = _empty_doc()
    try:
        blob = (client or _get_storage_client()).bucket(BUCKET).blob(ADMINS_BLOB)
        if blob.exists():
            loaded = json.loads(blob.download_as_text())
            if isinstance(loaded, dict):
                doc.update(loaded)
    except Exception as e:
        doc['error'] = str(e)

    doc['admins']  = sorted({norm_email(e) for e in (doc.get('admins') or []) if norm_email(e)})
    doc['history'] = [h for h in (doc.get('history') or []) if isinstance(h, dict)]
    st.session_state[_SESSION_KEY] = doc
    return doc


def save_admins(
    emails,
    *,
    actor: str,
    note: str = '',
    client: storage.Client | None = None,
) -> dict:
    """Overwrite the admin list, appending a history entry. Super admins are
    never stored in the file — they live in code."""
    gcs     = client or _get_storage_client()
    current = load_admins(gcs)
    stamp   = datetime.now(timezone.utc).isoformat()

    doc = {
        'admins':     sorted({norm_email(e) for e in emails if norm_email(e)} - _SUPER),
        'updated_at': stamp,
        'updated_by': actor or 'local-dev',
        'history': ([*current.get('history', []),
                     {'at': stamp, 'by': actor or 'local-dev', 'note': note}])[-_MAX_HISTORY:],
    }
    gcs.bucket(BUCKET).blob(ADMINS_BLOB).upload_from_string(
        json.dumps(doc), content_type='application/json'
    )
    st.session_state[_SESSION_KEY] = doc
    return doc


def admin_emails(client: storage.Client | None = None) -> list[str]:
    """Every address with admin rights — super admins plus the stored list."""
    return sorted(_SUPER | set(load_admins(client)['admins']))


# ── Checks ──────────────────────────────────────────────────────────────────

def is_super_admin(email: str | None = None) -> bool:
    if email is None:
        if is_local_dev():
            return True
        email = current_user_email()
    return norm_email(email) in _SUPER


def is_admin(email: str | None = None) -> bool:
    if email is None:
        if is_local_dev():
            return True
        email = current_user_email()
    return norm_email(email) in set(admin_emails())


def role_label() -> str:
    if is_local_dev():
        return 'local dev · super admin'
    if is_super_admin():
        return 'super admin'
    if is_admin():
        return 'admin'
    return 'user'


def require_admin(what: str = 'This page') -> None:
    """Stop the script with a notice when the session is not an admin."""
    if is_admin():
        return
    st.error(
        f'🔒 {what} is restricted to administrators. '
        f'Signed in as **{current_user_email() or "unknown"}** — ask a super admin '
        f'({", ".join(SUPER_ADMINS)}) for access via the Admin Portal.'
    )
    st.stop()


def admin_only_notice(what: str = 'Deleting') -> None:
    """Inline caption for a control that is hidden from non-admins."""
    st.caption(
        f'🔒 {what} requires admin access — signed in as '
        f'{current_user_email() or "local dev"} ({role_label()}).'
    )
