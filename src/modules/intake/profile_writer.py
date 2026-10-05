"""
profile_writer.py — a submitted intake → the Matcher's company records.

A prospect in the Matcher is contact rows (data/all-contacts/{pool}/) plus a
capability profile built from them by client-profile-job. This module writes
the rows; profile_trigger.py starts the build.

  * The company is looked up by bare website domain across BOTH pools. If it
    is already a client or prospect, the intake columns are written onto its
    existing rows in place — summary and embeddings are left alone (staff or
    research may have written better ones) — and a contact row is added only
    for a new email.
  * Otherwise a new prospect parquet is written in the clients column
    convention, its summary embedded with ada-002 like every other contact row.

The blob name of a new prospect is derived from the session id, so a retried
step rewrites the same file instead of creating a duplicate.
"""

import hashlib
import io
import json
import uuid
from datetime import date

import numpy as np
import pandas as pd
import tldextract

import src.modules.pools as pools
from src.modules.intake import schema as sc
from src.modules.intake import schema_export as sx

INTAKE_COLS = ('intake_data', 'intake_summary', 'intake_updated_at')


def normalize_website(url: str) -> str:
    url = str(url or '').strip()
    if url and not url.lower().startswith(('http://', 'https://')):
        url = 'https://' + url
    return url.rstrip('/')


def bare_domain(url: str) -> str:
    ext = tldextract.extract(str(url or ''))
    return f'{ext.domain}.{ext.suffix}'.lower() if ext.domain else ''


def new_prospect_blob(session_id: str, today: str | None = None) -> str:
    hex6 = hashlib.sha256(session_id.encode()).hexdigest()[:6]
    return f'{pools.contacts_prefix(pools.PROSPECTS)}intake_{today or date.today().isoformat()}_{hex6}.parquet'


def _domain_mask(df: pd.DataFrame, domain: str) -> pd.Series:
    if 'companyWebsite' not in df.columns or not domain:
        return pd.Series(False, index=df.index)
    return df['companyWebsite'].fillna('').astype(str).map(bare_domain) == domain


def find_company(frames_by_pool: dict[str, dict[str, pd.DataFrame]], domain: str
                 ) -> tuple[str, str] | None:
    """(pool, company_key) of an existing company with this domain. Clients
    win over prospects: a company we already work for is never a prospect."""
    for pool in (pools.CLIENTS, pools.PROSPECTS):
        for df in frames_by_pool.get(pool, {}).values():
            hit = df[_domain_mask(df, domain)]
            if not hit.empty:
                return pool, pools.company_key(hit.iloc[0])
    return None


def intake_payload(answers: dict, session_id: str, submitted_at: str) -> dict:
    return {
        'answers':      answers,
        'extracted':    sx.profile_extracted(answers),
        'session_id':   session_id,
        'submitted_at': submitted_at,
        'source':       sx.INTAKE_SOURCE,
    }


# Ids a previous intake wrote back — kept when the company submits again, so a
# re-submission reuses its Drive folder and HubSpot records.
CARRY_KEYS = ('drive_folder_id', 'drive_folder_url', 'hubspot_company_id',
              'hubspot_contact_id')


def _carry_over(raw, payload: dict) -> dict:
    try:
        prev = json.loads(raw) if raw else {}
    except Exception:
        prev = {}
    prev = prev if isinstance(prev, dict) else {}
    return {**{k: prev[k] for k in CARRY_KEYS if prev.get(k)}, **payload}


def _contact_fields(answers: dict) -> dict:
    return {
        'firstName': answers.get(sc.CONTACT_FIRST, ''),
        'lastName':  answers.get(sc.CONTACT_LAST, ''),
        'email':     str(answers.get(sc.CONTACT_EMAIL, '')).lower(),
    }


def _write(storage_client, blob_name: str, df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    storage_client.bucket(pools.BUCKET).blob(blob_name).upload_from_string(
        buf.getvalue(), content_type='application/octet-stream')


def _read(storage_client, blob_name: str) -> pd.DataFrame:
    data = storage_client.bucket(pools.BUCKET).blob(blob_name).download_as_bytes()
    return pools.normalize_company_columns(pd.read_parquet(io.BytesIO(data)))


def _ensure_object_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = ''
        df[c] = df[c].astype(object)
    return df


def write_prospect(storage_client, answers: dict, session_id: str,
                   submitted_at: str, embed) -> dict:
    """Write or update the company's rows. `embed(text) -> list[float]` is
    injected (TextProcessor.get_embedding in production). Returns
    {pool, company_key, blobs, created}."""
    website  = normalize_website(answers.get(sc.WEBSITE))
    domain   = bare_domain(website)
    payload  = intake_payload(answers, session_id, submitted_at)
    digest   = sx.digest(answers)
    today    = submitted_at[:10]
    contact  = _contact_fields(answers)

    frames_by_pool = {p: pools.load_frames(storage_client, p)[0]
                      for p in (pools.CLIENTS, pools.PROSPECTS)}
    found = find_company(frames_by_pool, domain)

    if found is None:
        name    = answers.get(sc.COMPANY_NAME, '').strip()
        summary = sx.embedding_text(answers)
        row = {
            'company_name':      name,
            'companyWebsite':    website,
            'state':             answers.get(sc.COMPANY_STATE, ''),
            'segment':           ' | '.join(answers.get(sc.VERTICALS) or []),
            **contact,
            'summary':           summary,
            'embeddings':        [float(x) for x in np.asarray(embed(summary), dtype=np.float64)],
            'uuid':              str(uuid.uuid4()),
            'scraped_at':        today,
            'source':            'intake',
            'intake_data':       json.dumps(payload, ensure_ascii=False),
            'intake_summary':    digest,
            'intake_updated_at': today,
        }
        blob = new_prospect_blob(session_id, today)
        _write(storage_client, blob, pd.DataFrame([row]))
        return {'pool': pools.PROSPECTS, 'company_key': pools.company_key(row),
                'blobs': [blob], 'created': True}

    pool, key = found
    touched: list[str] = []
    email_seen = False
    blobs = [b for b, df in frames_by_pool[pool].items() if pools.key_mask(df, key).any()]
    for blob in blobs:
        df   = _ensure_object_cols(_read(storage_client, blob), INTAKE_COLS)  # re-read just before write
        mask = pools.key_mask(df, key)
        if not mask.any():
            continue
        if 'email' in df.columns and contact['email']:
            email_seen |= (df.loc[mask, 'email'].fillna('').astype(str).str.lower()
                           == contact['email']).any()
        df.loc[mask, 'intake_data'] = df.loc[mask, 'intake_data'].map(
            lambda raw: json.dumps(_carry_over(raw, payload), ensure_ascii=False))
        df.loc[mask, 'intake_summary']    = digest
        df.loc[mask, 'intake_updated_at'] = today
        if blob == blobs[-1] and not email_seen and contact['email']:
            template = df[mask].iloc[0].to_dict()
            template.update(contact)
            template['uuid'] = str(uuid.uuid4())
            df = pd.concat([df, pd.DataFrame([template])], ignore_index=True)
        _write(storage_client, blob, df)
        touched.append(blob)
    return {'pool': pool, 'company_key': key, 'blobs': touched, 'created': False}


def update_intake_data(storage_client, pool: str, company_key: str, patch: dict) -> list[str]:
    """Merge `patch` into intake_data on every row of the company (Drive and
    HubSpot ids, written back at the end of the pipeline). Re-reads each blob
    immediately before rewriting it."""
    touched = []
    for blob in pools.load_frames(storage_client, pool)[0]:
        df = _read(storage_client, blob)
        if 'intake_data' not in df.columns:
            continue
        mask = pools.key_mask(df, company_key)
        if not mask.any():
            continue
        def _merge(raw):
            try:
                obj = json.loads(raw) if raw else {}
            except Exception:
                obj = {}
            obj.update(patch)
            return json.dumps(obj, ensure_ascii=False)
        df.loc[mask, 'intake_data'] = df.loc[mask, 'intake_data'].map(_merge)
        _write(storage_client, blob, df)
        touched.append(blob)
    return touched


def previous_ids(storage_client, pool: str, company_key: str) -> dict:
    """The CARRY_KEYS ids (Drive folder, HubSpot records) a previous intake
    stored for this company — write_prospect carries them onto the new
    intake_data, so a re-submission reuses its folder and records."""
    for df in pools.load_frames(storage_client, pool)[0].values():
        if 'intake_data' not in df.columns:
            continue
        for raw in df.loc[pools.key_mask(df, company_key), 'intake_data']:
            try:
                obj = json.loads(raw or '{}')
            except Exception:
                continue
            found = {k: obj[k] for k in CARRY_KEYS if obj.get(k)}
            if found:
                return found
    return {}
