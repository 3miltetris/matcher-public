"""
source_credentials.py — per-site login credentials for the Funding Source Watch
(Stage 10).

A handful of the curated funding-source sites hide their listings behind a member
login. This module stores the team's login for those sites so the browser agent
(``src/modules/browser_agent.py``) can sign in and scrape them like any other
site.

Passwords are secrets, so they live in **Secret Manager**, never in the
``sources.parquet`` GCS blob. This follows the Fathom-key precedent in this repo
(secrets go in Secret Manager; the SAM key sitting in a GCS config blob is a
documented wart, not a pattern to copy). One secret holds the whole map:

    projects/cc-matcher-v1/secrets/funding-source-credentials

Its payload is a JSON object keyed by registry ``source_id``:

    {
      "a3f9c1": {
        "username":   "team@bwcoconsulting.com",
        "password":   "...",
        "login_url":  "https://marketplace.example.org/login",
        "updated_at": "2026-10-08T12:00:00+00:00",
        "updated_by": "john@bwcoconsulting.com"
      },
      ...
    }

Streamlit-free and **client-injected**, the same split ``source_registry`` uses:
every function takes an already-built ``SecretManagerServiceClient`` so the view
can pass one built from ``st.secrets`` and the job can pass an ADC one. The
``sources.parquet`` registry carries only a non-secret ``has_credentials`` bool
flag, so the UI and job can tell which sites are set up without reading the
secret.
"""

import json
from datetime import datetime, timezone

PROJECT_ID = 'cc-matcher-v1'
SECRET_ID  = 'funding-source-credentials'

# Fields a stored entry may hold. `password` is deliberately absent from
# `safe_summary` — it must never be rendered. `code_sender` / `code_regex` are
# non-secret 2FA-email config (which address the code comes from, how to parse
# it); they live here so one site's whole login config travels together.
_ENTRY_FIELDS = ('username', 'password', 'login_url', 'code_sender', 'code_regex',
                 'updated_at', 'updated_by')


# -- Resource paths ---------------------------------------------------------

def secret_name(project_id: str = PROJECT_ID) -> str:
    return f'projects/{project_id}/secrets/{SECRET_ID}'


def latest_version_name(project_id: str = PROJECT_ID) -> str:
    return f'{secret_name(project_id)}/versions/latest'


# -- Client builders (lazy imports so the pure helpers can be unit-tested
#    with a fake client in an environment without the library) ------------

def client_from_info(info: dict):
    """A Secret Manager client from a service-account info dict (the view's
    ``st.secrets['gcp_service_account']``)."""
    from google.cloud import secretmanager
    from google.oauth2 import service_account
    creds = service_account.Credentials.from_service_account_info(info)
    return secretmanager.SecretManagerServiceClient(credentials=creds)


def client_adc():
    """A Secret Manager client from Application Default Credentials (the job's
    attached service account)."""
    from google.cloud import secretmanager
    return secretmanager.SecretManagerServiceClient()


# -- Store I/O --------------------------------------------------------------

def _is_missing(exc: Exception) -> bool:
    """True for the "secret / enabled version does not exist yet" cases, which
    mean an empty store rather than an error. Matched by class name so this
    module does not hard-depend on google.api_core at import time."""
    name = type(exc).__name__
    return name in ('NotFound', 'FailedPrecondition')


def load_credentials(sm_client, project_id: str = PROJECT_ID) -> dict:
    """The whole credential map. A missing secret or no enabled version returns
    ``{}`` — the same "missing store is empty, not an error" posture as
    ``source_registry.load_sources``. Permission or network errors propagate so
    the caller can decide (the view surfaces them; the job degrades to ``{}``)."""
    try:
        resp = sm_client.access_secret_version(
            request={'name': latest_version_name(project_id)}
        )
    except Exception as exc:                                   # noqa: BLE001
        if _is_missing(exc):
            return {}
        raise
    try:
        data = json.loads(resp.payload.data.decode('utf-8'))
    except Exception:                                         # noqa: BLE001
        return {}
    if not isinstance(data, dict):
        return {}
    # Keep only well-formed entries.
    out = {}
    for sid, entry in data.items():
        if isinstance(entry, dict):
            out[str(sid)] = {k: entry.get(k, '') for k in _ENTRY_FIELDS}
    return out


def save_credentials(sm_client, data: dict, project_id: str = PROJECT_ID) -> None:
    """Add a new version of the secret holding ``data`` (the secret itself is
    created once via gcloud — see the plan's Infra section). Each save is a new
    version, which doubles as an audit trail."""
    payload = json.dumps(data or {}).encode('utf-8')
    sm_client.add_secret_version(
        request={
            'parent':  secret_name(project_id),
            'payload': {'data': payload},
        }
    )


# -- Mutations --------------------------------------------------------------

def set_credential(
    sm_client,
    source_id: str,
    *,
    username: str,
    password: str,
    login_url: str = '',
    code_sender: str = '',
    code_regex: str = '',
    actor: str = '',
    project_id: str = PROJECT_ID,
) -> dict:
    """Upsert one site's credentials. Returns the new full map. ``code_sender`` /
    ``code_regex`` are non-secret 2FA-email config (blank when the site has no
    email 2FA)."""
    sid = str(source_id or '').strip()
    if not sid:
        raise ValueError('set_credential: a source_id is required')
    if not str(username or '').strip() or not str(password or '').strip():
        raise ValueError('set_credential: username and password are required')

    data = load_credentials(sm_client, project_id)
    data[sid] = {
        'username':    str(username).strip(),
        'password':    str(password),
        'login_url':   str(login_url or '').strip(),
        'code_sender': str(code_sender or '').strip(),
        'code_regex':  str(code_regex or '').strip(),
        'updated_at':  datetime.now(timezone.utc).isoformat(),
        'updated_by':  str(actor or '').strip(),
    }
    save_credentials(sm_client, data, project_id)
    return data


def delete_credential(
    sm_client, source_id: str, *, actor: str = '', project_id: str = PROJECT_ID
) -> dict:
    """Remove one site's credentials. Returns the new full map. A no-op if the
    site had none."""
    data = load_credentials(sm_client, project_id)
    if str(source_id) in data:
        data.pop(str(source_id), None)
        save_credentials(sm_client, data, project_id)
    return data


# -- Read helpers -----------------------------------------------------------

def credential_for(data: dict, source_id: str) -> dict | None:
    """The entry for one site (with its password), or ``None``. Used by the job
    to bind credentials to a browser session — never rendered."""
    if not isinstance(data, dict):
        return None
    entry = data.get(str(source_id))
    return entry if isinstance(entry, dict) and entry.get('password') else None


def configured_ids(data: dict) -> set:
    """Source ids that have a usable (password-bearing) credential."""
    if not isinstance(data, dict):
        return set()
    return {sid for sid, e in data.items()
            if isinstance(e, dict) and e.get('password')}


def safe_summary(data: dict) -> list:
    """Password-free rows for display: ``{source_id, username, login_url,
    updated_at, updated_by}``. The **only** thing the view renders — passwords
    never leave this module."""
    rows = []
    for sid, e in (data or {}).items():
        if not isinstance(e, dict) or not e.get('password'):
            continue
        rows.append({
            'source_id':   sid,
            'username':    e.get('username', ''),
            'login_url':   e.get('login_url', ''),
            'code_sender': e.get('code_sender', ''),
            'code_regex':  e.get('code_regex', ''),
            'updated_at':  e.get('updated_at', ''),
            'updated_by':  e.get('updated_by', ''),
        })
    return sorted(rows, key=lambda r: r['source_id'])
