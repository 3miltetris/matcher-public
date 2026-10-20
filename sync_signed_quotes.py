#!/usr/bin/env python3
"""
Sync signed HubSpot quotes (e-signature) and their linked Terms & Conditions PDFs
into Google Drive, one folder per company.

  python sync_signed_quotes.py --backfill --dry-run --limit 3   # test first
  python sync_signed_quotes.py --backfill                       # all signed quotes
  python sync_signed_quotes.py                                  # signed in last LOOKBACK_DAYS

Idempotent: every uploaded file is tagged with Drive appProperties
(hubspot_quote_id + kind), so re-runs and overlapping windows never duplicate.
No local state file is needed, so it runs fine as a Cloud Run job.

Env vars:
  HUBSPOT_TOKEN                   Private app token (scopes: crm.objects.quotes.read,
                                  crm.objects.companies.read, crm.objects.deals.read)
  DRIVE_PARENT_FOLDER_ID          Folder that holds the per-company folders
  GOOGLE_APPLICATION_CREDENTIALS  Service account JSON (optional on Cloud Run)
  GOOGLE_IMPERSONATE_USER         Optional: user to impersonate via domain-wide delegation
  LOOKBACK_DAYS                   Window for recurring runs (default 3)
"""
import argparse
import html
import io
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import google.auth
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload

HUBSPOT_TOKEN = os.environ["HUBSPOT_TOKEN"]
PARENT_FOLDER_ID = os.environ["DRIVE_PARENT_FOLDER_ID"]
SA_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
IMPERSONATE = os.environ.get("GOOGLE_IMPERSONATE_USER")
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "3"))

HS_BASE = "https://api.hubapi.com"
FOLDER_MIME = "application/vnd.google-apps.folder"
GDOC_MIME = "application/vnd.google-apps.document"
QUOTE_PROPS = ["hs_title", "hs_esign_date", "hs_pdf_download_link", "hs_terms", "hs_quote_number"]
# Matches drive.google.com/file/d/<id>, docs.google.com/document/d/<id>, ?id=<id>
DRIVE_ID_RE = re.compile(r"(?:/file/d/|/document/d/|[?&]id=)([A-Za-z0-9_-]{20,})")

log = logging.getLogger("quote-sync")
hs_session = requests.Session()
hs_session.headers.update({"Authorization": f"Bearer {HUBSPOT_TOKEN}"})


# ---------------------------------------------------------------- helpers
def fix_text(s: str) -> str:
    """Repair double-encoded UTF-8 (e.g. 'ProPlateÂ®' -> 'ProPlate®'). Leaves clean text alone."""
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def safe_name(s: str) -> str:
    return re.sub(r"[\\/]+", "-", fix_text(s or "")).strip() or "Untitled"


def q_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


# ---------------------------------------------------------------- HubSpot
def hs(method: str, path: str, **kw) -> dict:
    for attempt in range(6):
        r = hs_session.request(method, HS_BASE + path, timeout=60, **kw)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def signed_quotes(since: datetime | None):
    filters = [{"propertyName": "hs_quote_esign_status", "operator": "EQ", "value": "SIGNED"}]
    if since:
        filters.append({"propertyName": "hs_esign_date", "operator": "GTE",
                        "value": str(int(since.timestamp() * 1000))})
    after = None
    while True:
        body = {"filterGroups": [{"filters": filters}], "properties": QUOTE_PROPS,
                "sorts": [{"propertyName": "hs_esign_date", "direction": "ASCENDING"}],
                "limit": 100}
        if after:
            body["after"] = after
        data = hs("POST", "/crm/v3/objects/quotes/search", json=body)
        yield from data.get("results", [])
        after = data.get("paging", {}).get("next", {}).get("after")
        if not after:
            return


def assoc_ids(from_type: str, obj_id: str, to_type: str) -> list[str]:
    results = hs("GET", f"/crm/v4/objects/{from_type}/{obj_id}/associations/{to_type}").get("results", [])
    # Primary association first, if one is labelled
    results.sort(key=lambda r: not any(t.get("label") == "Primary" for t in r.get("associationTypes", [])))
    return [str(r["toObjectId"]) for r in results]


def company_for_quote(quote_id: str) -> tuple[str | None, str]:
    ids = assoc_ids("quotes", quote_id, "companies")
    if not ids:  # fall back to quote -> deal -> company
        for deal_id in assoc_ids("quotes", quote_id, "deals"):
            ids = assoc_ids("deals", deal_id, "companies")
            if ids:
                break
    if not ids:
        return None, "_Unassigned"
    cid = ids[0]
    name = hs("GET", f"/crm/v3/objects/companies/{cid}", params={"properties": "name"})["properties"].get("name")
    return cid, safe_name(name or f"Company {cid}")


def download_signed_pdf(url: str) -> bytes:
    r = requests.get(url, timeout=120, allow_redirects=True)
    r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise ValueError(f"Download did not return a PDF (content-type {r.headers.get('content-type')})")
    return r.content


# ---------------------------------------------------------------- Drive
def drive_service():
    scopes = ["https://www.googleapis.com/auth/drive"]
    if SA_FILE:
        creds = service_account.Credentials.from_service_account_file(SA_FILE, scopes=scopes)
        if IMPERSONATE:
            creds = creds.with_subject(IMPERSONATE)
    else:
        creds, _ = google.auth.default(scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def drive_list(drive, q: str, fields="files(id,name,appProperties)") -> list[dict]:
    return drive.files().list(q=q, fields=fields, pageSize=100, corpora="allDrives",
                              supportsAllDrives=True, includeItemsFromAllDrives=True).execute().get("files", [])


def company_folder(drive, cid: str | None, name: str, cache: dict, dry: bool) -> str | None:
    key = cid or "unassigned"
    if key in cache:
        return cache[key]
    base = f"'{PARENT_FOLDER_ID}' in parents and mimeType='{FOLDER_MIME}' and trashed=false"
    hits = drive_list(drive, f"{base} and appProperties has {{ key='hubspot_company_id' and value='{key}' }}")
    if not hits:
        # Adopt a manually created folder with the same name rather than making a duplicate
        hits = drive_list(drive, f"{base} and name='{q_escape(name)}'")
        if hits and not dry:
            drive.files().update(fileId=hits[0]["id"], body={"appProperties": {"hubspot_company_id": key}},
                                 supportsAllDrives=True).execute()
    if hits:
        folder_id = hits[0]["id"]
    elif dry:
        log.info("  [dry-run] would create folder '%s'", name)
        folder_id = None
    else:
        folder_id = drive.files().create(
            body={"name": name, "mimeType": FOLDER_MIME, "parents": [PARENT_FOLDER_ID],
                  "appProperties": {"hubspot_company_id": key}},
            fields="id", supportsAllDrives=True).execute()["id"]
        log.info("  created folder '%s'", name)
    cache[key] = folder_id
    return folder_id


def synced_kinds(drive, folder_id: str, quote_id: str) -> set[str]:
    q = (f"'{folder_id}' in parents and trashed=false and "
         f"appProperties has {{ key='hubspot_quote_id' and value='{quote_id}' }}")
    return {f.get("appProperties", {}).get("kind", "") for f in drive_list(drive, q)}


def upload_pdf(drive, folder_id: str, name: str, data: bytes, props: dict):
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype="application/pdf", resumable=False)
    drive.files().create(body={"name": name, "parents": [folder_id], "appProperties": props},
                         media_body=media, fields="id", supportsAllDrives=True).execute()


def copy_terms(drive, src_id: str, folder_id: str, name: str, props: dict):
    meta = drive.files().get(fileId=src_id, fields="id,name,mimeType", supportsAllDrives=True).execute()
    if meta["mimeType"] == GDOC_MIME:  # T&C kept as a Google Doc -> export to PDF
        pdf = drive.files().export(fileId=src_id, mimeType="application/pdf").execute()
        upload_pdf(drive, folder_id, name, pdf, props)
    else:  # Real PDF -> server-side copy, no download needed
        drive.files().copy(fileId=src_id, body={"name": name, "parents": [folder_id], "appProperties": props},
                           supportsAllDrives=True).execute()


# ---------------------------------------------------------------- main
def process_quote(drive, quote: dict, cache: dict, dry: bool) -> str:
    qid = quote["id"]
    p = quote["properties"]
    title = safe_name(p.get("hs_title") or f"Quote {qid}")
    date = (p.get("hs_esign_date") or "")[:10] or "undated"
    cid, company = company_for_quote(qid)
    log.info("%s | %s | %s", date, company, title)

    folder_id = company_folder(drive, cid, company, cache, dry)
    done = synced_kinds(drive, folder_id, qid) if folder_id else set()
    base_props = {"hubspot_quote_id": qid}
    did_work = False

    if "signed_quote" not in done:
        name = f"{date} - {title} (Signed).pdf"
        if dry:
            log.info("  [dry-run] would upload '%s'", name)
        else:
            upload_pdf(drive, folder_id, name, download_signed_pdf(p["hs_pdf_download_link"]),
                       {**base_props, "kind": "signed_quote"})
            log.info("  uploaded signed quote")
        did_work = True

    term_ids = DRIVE_ID_RE.findall(html.unescape(p.get("hs_terms") or ""))
    if not term_ids:
        log.warning("  no Drive link found in Terms section")
    for i, src_id in enumerate(dict.fromkeys(term_ids)):
        kind = "terms" if i == 0 else f"terms_{i + 1}"
        if kind in done:
            continue
        name = f"{date} - {title} - Terms and Conditions.pdf"
        if dry:
            log.info("  [dry-run] would copy T&C %s as '%s'", src_id, name)
        else:
            copy_terms(drive, src_id, folder_id, name, {**base_props, "kind": kind, "tc_source_id": src_id})
            log.info("  copied T&C")
        did_work = True

    return "synced" if did_work else "skipped"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true", help="Process every signed quote, not just recent ones")
    ap.add_argument("--dry-run", action="store_true", help="Log actions without writing to Drive")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N quotes (for testing)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    since = None if args.backfill else datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    drive = drive_service()
    cache: dict = {}
    counts = {"synced": 0, "skipped": 0, "failed": 0}

    for n, quote in enumerate(signed_quotes(since), 1):
        try:
            counts[process_quote(drive, quote, cache, args.dry_run)] += 1
        except (requests.HTTPError, HttpError, ValueError, KeyError) as e:
            counts["failed"] += 1
            log.error("  FAILED quote %s: %s", quote["id"], e)
        if args.limit and n >= args.limit:
            break

    log.info("Done: %s", counts)
    sys.exit(1 if counts["failed"] else 0)  # non-zero so Cloud Run / scheduler flags failures


if __name__ == "__main__":
    main()
