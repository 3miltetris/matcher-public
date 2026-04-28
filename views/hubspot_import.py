"""
HubSpot Import
--------------
Loads all segment CSVs from a completed matching run in GCS, concatenates
them, and pushes them to HubSpot: contacts are upserted (dedup by email) and
an optional note per contact is created summarising every matched grant.
"""

import io
import json
import math
import time
import traceback
from collections import defaultdict

import pandas as pd
import requests
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account

_BUCKET         = 'cc-matcher-bucket-jeg-v1'
_RESULTS_PREFIX = 'matching-results/'
_HS_BASE        = 'https://api.hubapi.com'
_BATCH_SIZE     = 100   # HubSpot batch endpoint limit
_INTER_BATCH_S  = 0.08  # ~12 req/s — well within HubSpot's 150/10s limit


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _gcs_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)


def _hs_headers() -> dict:
    return {
        'Authorization': f'Bearer {st.secrets["hubspot_api_key"]}',
        'Content-Type':  'application/json',
    }


# ── GCS helpers ───────────────────────────────────────────────────────────────

def _list_completed_runs(client: storage.Client) -> list[str]:
    blobs = client.list_blobs(_BUCKET, prefix=_RESULTS_PREFIX, delimiter='/')
    list(blobs)
    run_ids = sorted(
        (p.replace(_RESULTS_PREFIX, '').strip('/') for p in blobs.prefixes),
        reverse=True,
    )
    completed = []
    for run_id in run_ids:
        status_blob = client.bucket(_BUCKET).blob(f'{_RESULTS_PREFIX}{run_id}/status.json')
        if status_blob.exists():
            status = json.loads(status_blob.download_as_text())
            if not status.get('error'):
                completed.append(run_id)
    return completed


def _load_run(client: storage.Client, run_id: str) -> pd.DataFrame:
    prefix = f'{_RESULTS_PREFIX}{run_id}/'
    blobs  = sorted(
        [b for b in client.list_blobs(_BUCKET, prefix=prefix) if b.name.endswith('.csv')],
        key=lambda b: b.name,
    )
    frames = [pd.read_csv(io.BytesIO(b.download_as_bytes())) for b in blobs]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── HubSpot helpers ───────────────────────────────────────────────────────────

def _hs_post(url: str, payload: dict, retries: int = 4) -> requests.Response:
    for attempt in range(retries):
        resp = requests.post(url, headers=_hs_headers(), json=payload, timeout=30)
        if resp.status_code == 429:
            wait = int(resp.headers.get('Retry-After', 10))
            time.sleep(wait)
            continue
        return resp
    return resp  # return last response after exhausting retries


def _upsert_contacts_batch(batch: list[dict]) -> tuple[dict[str, str], list[str]]:
    """
    Upsert up to 100 contacts by email.
    Returns (email→hs_id map built from the response, error strings).
    """
    inputs = []
    for r in batch:
        props: dict = {}
        for src_col, hs_prop in [
            ('firstName',     'firstname'),
            ('lastName',      'lastname'),
            ('email',         'email'),
            ('companyName',   'company'),
            ('companyWebsite','website'),
        ]:
            v = str(r.get(src_col, '') or '').strip()
            if v:
                props[hs_prop] = v
        inputs.append({'idProperty': 'email', 'id': str(r['email']).strip(), 'properties': props})

    resp = _hs_post(f'{_HS_BASE}/crm/v3/objects/contacts/batch/upsert', {'inputs': inputs})
    if resp.status_code not in (200, 201, 207):
        return {}, [f'HTTP {resp.status_code}: {resp.text[:300]}']

    data = resp.json()
    email_to_id: dict[str, str] = {}
    for obj in data.get('results', []):
        email = obj.get('properties', {}).get('email', '').strip().lower()
        if email:
            email_to_id[email] = obj['id']

    errors = [str(e) for e in data.get('errors', [])]
    return email_to_id, errors


def _build_note_html(run_id: str, matched_rows: list[dict]) -> str:
    lines = [f'<strong>Matcher Run:</strong> {run_id}', '<br>']
    for i, r in enumerate(matched_rows, 1):
        title    = r.get('title',        '')
        agency   = r.get('agency',       '') or r.get('broad_agency', '')
        topic_no = r.get('topic_number', '')
        match    = r.get('good_match',   '')
        summary  = str(r.get('grant_summary', '') or '')[:400]
        subj     = r.get('subject_line', '')
        msg      = r.get('ai_message',   '')

        lines.append(f'<strong>[{i}] {title}</strong>')
        if agency:   lines.append(f'Agency: {agency}')
        if topic_no: lines.append(f'Topic: {topic_no}')
        if match:    lines.append(f'Good Match: {match}')
        if summary:  lines.append(f'<em>{summary}</em>')
        if subj:     lines.append(f'<strong>Subject Line:</strong> {subj}')
        if msg:      lines.append(f'<strong>Email Draft:</strong> {msg}')
        lines.append('<hr>')

    return '<br>'.join(lines)


def _create_notes_batch(note_inputs: list[dict]) -> tuple[int, list[str]]:
    resp = _hs_post(f'{_HS_BASE}/crm/v3/objects/notes/batch/create', {'inputs': note_inputs})
    if resp.status_code not in (200, 201, 207):
        return 0, [f'HTTP {resp.status_code}: {resp.text[:300]}']
    data = resp.json()
    return len(data.get('results', [])), [str(e) for e in data.get('errors', [])]


# ── Page ──────────────────────────────────────────────────────────────────────

st.title('🔗 HubSpot Import')
st.caption('Load a completed matching run and push contacts (+ optional notes) to HubSpot.')

if 'hubspot_api_key' not in st.secrets:
    st.info(
        'Add `hubspot_api_key = "pat-..."` to your `.streamlit/secrets.toml` '
        '(or Streamlit Cloud secrets) to use this view. '
        'Generate a Private App token in HubSpot → Settings → Integrations → Private Apps '
        'with scopes: `crm.objects.contacts.write`, `crm.objects.notes.write`.'
    )
    st.stop()

# ── Run selector ──────────────────────────────────────────────────────────────

gcs = _gcs_client()

with st.spinner('Fetching completed runs from GCS…'):
    runs = _list_completed_runs(gcs)

if not runs:
    st.warning('No completed matching runs found in GCS.')
    st.stop()

run_id = st.selectbox('Select a completed run', runs)

if st.session_state.get('hs_run_id') != run_id:
    st.session_state.hs_df     = None
    st.session_state.hs_run_id = run_id

if st.button('Load run data', type='primary'):
    with st.spinner('Concatenating segment CSVs…'):
        df = _load_run(gcs, run_id)
    if df.empty:
        st.warning('No CSV segments found for this run.')
    else:
        st.session_state.hs_df = df

df: pd.DataFrame | None = st.session_state.get('hs_df')
if df is None:
    st.stop()

# ── Preview ───────────────────────────────────────────────────────────────────

st.divider()
has_email = 'email' in df.columns
valid_mask = (
    df['email'].notna() & (df['email'].astype(str).str.strip() != '')
    if has_email else pd.Series([False] * len(df))
)

c1, c2, c3 = st.columns(3)
c1.metric('Total rows',    f'{len(df):,}')
c2.metric('Unique emails', f'{df["email"].dropna().nunique():,}' if has_email else '—')
c3.metric('Rows with email', f'{valid_mask.sum():,}' if has_email else '—')

st.dataframe(df.head(50), use_container_width=True, hide_index=True)

if not has_email:
    st.error('Run CSV has no `email` column — cannot import to HubSpot.')
    st.stop()

# ── Options ───────────────────────────────────────────────────────────────────

st.divider()
st.subheader('Import options')

skip_no_email = st.checkbox('Skip rows without an email address', value=True)
create_notes  = st.checkbox(
    'Create one note per contact summarising all matched grants',
    value=True,
    help=(
        'Uses HubSpot IDs returned by the upsert step — no extra API calls needed. '
        'Requires `crm.objects.notes.write` scope on your Private App token.'
    ),
)

# ── Trigger import ────────────────────────────────────────────────────────────

st.divider()
if not st.button('▶ Import to HubSpot', type='primary'):
    st.stop()

working = df.copy()
if skip_no_email:
    working = working[valid_mask].reset_index(drop=True)

if working.empty:
    st.warning('No rows with valid emails — nothing to import.')
    st.stop()

rows = working.to_dict('records')
n_batches = math.ceil(len(rows) / _BATCH_SIZE)

# ── Phase 1: Upsert contacts ──────────────────────────────────────────────────

st.subheader('Phase 1 — Upserting contacts')
progress1     = st.progress(0, text='Starting…')
email_to_id:  dict[str, str]  = {}
upsert_errors: list[str]      = []
total_upserted = 0

try:
    for i in range(n_batches):
        batch    = rows[i * _BATCH_SIZE : (i + 1) * _BATCH_SIZE]
        id_map, errs = _upsert_contacts_batch(batch)
        email_to_id.update(id_map)
        upsert_errors.extend(errs)
        total_upserted += len(id_map)
        pct = (i + 1) / n_batches
        progress1.progress(pct, text=f'Upserted {total_upserted:,} contacts ({i+1}/{n_batches} batches)…')
        time.sleep(_INTER_BATCH_S)
except Exception:
    st.error('Unexpected error during contact upsert.')
    st.code(traceback.format_exc())
    st.stop()

progress1.progress(1.0, text=f'Done — {total_upserted:,} contacts processed.')
st.success(f'Contacts upserted: **{total_upserted:,}**')
if upsert_errors:
    with st.expander(f'{len(upsert_errors)} upsert error(s)'):
        for e in upsert_errors[:30]:
            st.text(e)

# ── Phase 2: Create notes ─────────────────────────────────────────────────────

if create_notes:
    st.subheader('Phase 2 — Creating notes')

    # Group all matched rows by email so each contact gets one bundled note
    by_email: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = str(r.get('email', '') or '').strip().lower()
        if key in email_to_id:
            by_email[key].append(r)

    ts_ms = str(int(time.time() * 1000))
    note_inputs: list[dict] = [
        {
            'associations': [{
                'to':    {'id': email_to_id[email_key]},
                'types': [{'associationCategory': 'HUBSPOT_DEFINED', 'associationTypeId': 202}],
            }],
            'properties': {
                'hs_note_body': _build_note_html(run_id, matched_rows),
                'hs_timestamp': ts_ms,
            },
        }
        for email_key, matched_rows in by_email.items()
    ]

    n_note_batches = math.ceil(len(note_inputs) / _BATCH_SIZE) if note_inputs else 0
    progress2      = st.progress(0, text='Starting…')
    notes_ok       = 0
    note_errors:    list[str] = []

    try:
        for i in range(n_note_batches):
            batch    = note_inputs[i * _BATCH_SIZE : (i + 1) * _BATCH_SIZE]
            ok, errs = _create_notes_batch(batch)
            notes_ok    += ok
            note_errors += errs
            pct = (i + 1) / n_note_batches
            progress2.progress(pct, text=f'Created {notes_ok:,} notes ({i+1}/{n_note_batches} batches)…')
            time.sleep(_INTER_BATCH_S)
    except Exception:
        st.error('Unexpected error during note creation.')
        st.code(traceback.format_exc())
        st.stop()

    progress2.progress(1.0, text=f'Done — {notes_ok:,} notes created.')
    st.success(f'Notes created: **{notes_ok:,}**')
    if note_errors:
        with st.expander(f'{len(note_errors)} note error(s)'):
            for e in note_errors[:30]:
                st.text(e)

st.balloons()
st.success(f'Import complete for run `{run_id}`!')
