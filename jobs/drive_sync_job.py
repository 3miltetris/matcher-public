"""
drive_sync_job.py — Cloud Run Job for the Client Drive Sync feature.

Reads a job config from GCS, then for each assigned client folder group:
  1. Lists files recursively in the client's Drive folder(s)
  2. Diffs file modifiedTime against drive-sync-configs/sync_state.json
     (unchanged clients cost zero downloads and zero LLM calls)
  3. Downloads/extracts changed documents (Google Docs/Sheets exported,
     PDF/DOCX/XLSX/TXT/CSV binaries via src/modules/doc_extract.py)
  4. Claude merges new document info with the current profile → updated
     matching summary + docs digest + structured extraction
  5. Writes client_docs_data / client_docs_summary / docs_updated_at onto
     every contact row of the client; rewrites summary + re-embeds
     (text-embedding-ada-002, float64) only when the change is meaningful
  6. Checkpoints touched parquets + sync_state + interim status every
     10 clients (timeout resilience — a re-trigger resumes via sync_state)

Unassigned folders passed as new_client_folder_ids get a proposal
(name/summary/digest) in the status payload — row creation happens in the
Streamlit view after human review. No parquet writes, no sync_state marks.

Usage:
    python jobs/drive_sync_job.py drive-sync-configs/<run_id>.json

Environment variables (injected by Cloud Run from Secret Manager):
    ANTHROPIC_API_KEY, OPENAI_API_KEY

Config schema:
{
  "run_id":                "drive_sync_2026-08-11_15-30-00",
  "drive_id":              "0ABc...",
  "folder_ids":            ["<assigned client folder ids>"],
  "new_client_folder_ids": ["<unassigned folder ids to propose>"],
  "full_resync":           false,
  "dry_run":               false,
  "max_docs_per_client":   40,
  "per_client_char_cap":   150000,
  "max_proposals":         40,
  "task_timeout_s":        14400,
  "model":                 "claude-sonnet-4-6"
}
"""

import io
import json
import os
import re
import string
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import tiktoken
from anthropic import Anthropic
from google.cloud import storage
from openai import OpenAI

import src.modules.anthropic_utils as au
from src.modules import attention, doc_extract, drive_client, drive_scan
from src.modules import gcs_write as gw
from src.modules import job_control as jc

# ── Constants ──────────────────────────────────────────────────────────────────

_BUCKET          = 'cc-matcher-bucket-jeg-v1'
_CLIENTS_PREFIX  = 'data/all-contacts/clients/'
_CFG_PREFIX      = 'drive-sync-configs/'
_STATUS_PREFIX   = 'drive-sync-jobs/'
_SYNC_STATE_BLOB = 'drive-sync-configs/sync_state.json'
_ASSIGN_BLOB     = 'drive-sync-configs/assignments.json'

_DOWNLOAD_WORKERS   = 4
_CHECKPOINT_EVERY   = 10
_TOKEN_LIMIT        = 7_500
_MAX_SKIPPED_LISTED = 200
_MAX_SOURCE_FILES   = 100
_DEFAULT_MODEL      = 'claude-sonnet-4-6'

# Graceful time budget: stop before Cloud Run's hard task timeout kills the
# container (which would lose all un-checkpointed work AND the status file,
# leaving the UI polling forever). Deferred work is picked up by the next run.
# The window comes from the config (`task_timeout_s`, chosen in the view) and
# must never exceed the deployed --task-timeout, which is Cloud Run's 24 h max.
_DEFAULT_TASK_TIMEOUT_S = 7_200
_MIN_TASK_TIMEOUT_S     = 900
_MAX_TASK_TIMEOUT_S     = 86_400
_DEADLINE_MARGIN_S  = 600
_DEFAULT_MAX_PROPOSALS = 40
# When proposal candidates exist, the client phase may not eat the whole
# budget — reserve this much per planned proposal (capped at half the budget)
# so a heavy client sweep can't starve the proposals phase entirely.
_PROPOSAL_RESERVE_PER_ITEM_S = 90

_MERGE_SYSTEM = (
    'You maintain company profiles for a grant-matching system. The "summary" '
    'field is embedded and cosine-matched against government grant topics — it '
    'must be 3-8 factual sentences about the company\'s technology, products, '
    'R&D capabilities, and market. You receive the current summary, previously '
    'extracted document data, and newly changed documents from the client\'s '
    'data-collection folder. Merge the new information into the profile.\n\n'
    'Respond with ONLY a valid JSON object (no markdown fences) of this shape:\n'
    '{\n'
    '  "no_meaningful_change": false,\n'
    '  "updated_summary": "full replacement summary, or null when no_meaningful_change",\n'
    '  "docs_digest": "5-10 bullet plain-text digest of what the documents reveal "\n'
    '                 "about the company (one bullet per line, prefixed with - )",\n'
    '  "extracted": {\n'
    '    "technologies": [], "products_services": [], "rd_focus": [],\n'
    '    "certifications_registrations": [], "team_and_facilities": "",\n'
    '    "grants_contracts": [], "keywords": [], "notable_updates": []\n'
    '  }\n'
    '}\n\n'
    'Rules:\n'
    '- Set no_meaningful_change=true and updated_summary=null when the new '
    'documents add nothing that would change which grants match this company '
    '(e.g. invoices, NDAs, meeting logistics, boilerplate contracts).\n'
    '- Never rewrite the summary for stylistic reasons — only when the documents '
    'reveal new or corrected facts about what the company does.\n'
    '- Preserve accurate facts from the current summary; the updated summary must '
    'stand alone (it fully replaces the old one).\n'
    '- The "extracted" object should merge previously extracted data with new '
    'findings (you receive the previous data as existing_docs_data).\n'
    '- Escape newlines inside JSON strings properly.'
)

_PROPOSE_SYSTEM = (
    'You create company profiles for a grant-matching system from a client\'s '
    'data-collection documents. The "proposed_summary" field will be embedded and '
    'cosine-matched against government grant topics — it must be 3-8 factual '
    'sentences about the company\'s technology, products, R&D capabilities, and '
    'market.\n\n'
    'Respond with ONLY a valid JSON object (no markdown fences) of this shape:\n'
    '{\n'
    '  "proposed_name": "clean company name",\n'
    '  "proposed_website": "https://... ONLY if literally present in the documents, else \\"\\"",\n'
    '  "proposed_summary": "3-8 factual sentences",\n'
    '  "docs_digest": "5-10 bullet plain-text digest (one bullet per line, prefixed with - )",\n'
    '  "extracted": {\n'
    '    "technologies": [], "products_services": [], "rd_focus": [],\n'
    '    "certifications_registrations": [], "team_and_facilities": "",\n'
    '    "grants_contracts": [], "keywords": [], "notable_updates": []\n'
    '  }\n'
    '}\n\n'
    'Rules:\n'
    '- NEVER invent a website — set proposed_website only when a URL or domain '
    'appears verbatim in the documents, OR when one of the candidate_domains '
    '(email domains of people the folder is shared with or who edited its '
    'documents) clearly belongs to this company. Otherwise leave it "".\n'
    '- Derive the company name from the folder name and documents (strip suffixes '
    'like _INTERNAL).\n'
    '- Escape newlines inside JSON strings properly.'
)


# ── Secrets / GCS ──────────────────────────────────────────────────────────────

def _get_secret(name: str) -> str:
    env_var = name.upper().replace('-', '_')
    val = os.environ.get(env_var, '')
    if not val:
        raise RuntimeError(f'Environment variable {env_var} is not set.')
    return val


def _gcs() -> storage.Client:
    return storage.Client()


def _load_json_blob(client: storage.Client, path: str) -> dict | None:
    blob = client.bucket(_BUCKET).blob(path)
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def _save_json_blob(client: storage.Client, path: str, payload: dict) -> None:
    client.bucket(_BUCKET).blob(path).upload_from_string(
        json.dumps(payload), content_type='application/json'
    )


def _write_status(client: storage.Client, run_id: str, payload: dict) -> None:
    _save_json_blob(client, f'{_STATUS_PREFIX}{run_id}/status.json', payload)


# ── Client frames (clients/ convention: company_name / summary) ───────────────

def _load_client_frames(client: storage.Client) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for blob in client.list_blobs(_BUCKET, prefix=_CLIENTS_PREFIX):
        if not blob.name.endswith('.parquet'):
            continue
        try:
            frames[blob.name] = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        except Exception as e:
            print(f'  WARN could not read {blob.name}: {e}', flush=True)
    return frames


def _group_mask(df: pd.DataFrame, key: str) -> pd.Series:
    name, website = key.split('||', 1)
    names    = df.get('company_name', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    websites = df.get('companyWebsite', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    return (names == name) & (websites == website)


def _name_by_key(frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    """{company_key: company_name} across all client frames — the input to
    drive_scan's name matching (scheduled mode)."""
    out: dict[str, str] = {}
    for df in frames.values():
        if 'company_name' not in df.columns or 'companyWebsite' not in df.columns:
            continue
        names = df['company_name'].fillna('').astype(str).str.strip()
        sites = df['companyWebsite'].fillna('').astype(str).str.strip()
        for name, site in zip(names, sites):
            if name or site:
                out.setdefault(f'{name}||{site}', name)
    return out


# ── Embedding ──────────────────────────────────────────────────────────────────

def _get_embedding(text: str, oai_client: OpenAI, encoding: tiktoken.Encoding) -> list[float] | None:
    if not text.strip():
        return None
    words = text.split()
    while len(encoding.encode(text)) > _TOKEN_LIMIT:
        words = words[:-5]
        if not words:
            return None
        text = ' '.join(words)
    return oai_client.embeddings.create(input=[text], model='text-embedding-ada-002').data[0].embedding


# ── Claude ─────────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[-1]
        raw = raw.rsplit('```', 1)[0].strip()
    start, end = raw.find('{'), raw.rfind('}')
    if start == -1 or end == -1:
        raise ValueError('no JSON object in response')
    return json.loads(raw[start:end + 1])


def _claude_json(anth: Anthropic, model: str, system: str, user_payload: dict) -> dict:
    """Streamed Claude call returning parsed JSON, with one strict-JSON retry."""
    user_msg = json.dumps(user_payload, ensure_ascii=False)
    last_err = None
    for attempt in range(2):
        content = user_msg if attempt == 0 else (
            user_msg + '\n\nYour previous response was not valid JSON. '
                       'Return ONLY the valid JSON object.'
        )
        with anth.messages.stream(
            model=model,
            max_tokens=16000,   # doc-heavy clients (40 docs) can produce long extractions
            system=system,
            messages=[{'role': 'user', 'content': content}],
        ) as stream:
            final = stream.get_final_message()
        if final.stop_reason == 'max_tokens':
            raise ValueError('Claude hit the output token limit')
        try:
            return _extract_json(au.response_text(final))
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
    raise ValueError(f'Claude returned invalid JSON twice: {last_err}')


# ── Drive helpers ──────────────────────────────────────────────────────────────

_thread_local = threading.local()


def _thread_drive_service():
    """googleapiclient's httplib2 transport is NOT thread-safe — sharing one
    service across download workers corrupts memory (segfault). Each worker
    thread builds and reuses its own service instance."""
    if not hasattr(_thread_local, 'svc'):
        _thread_local.svc = drive_client.build_drive_service()
    return _thread_local.svc


def _download_one(f: dict) -> tuple[str, str]:
    return drive_client.download_file_text(_thread_drive_service(), f)


# Skip reasons that are deterministic — retrying next run would fail the same
# way (image-only PDFs, oversized exports, 404s on shortcut targets outside
# the shared drive). These files get marked synced so they stop burning budget
# on every run; a "Full re-scan" retries them if access is later fixed.
_PERMANENT_SKIP_MARKERS = ('extraction failed', 'export size limit',
                           'empty', 'HTTP 404')


def _is_permanent_skip(reason: str) -> bool:
    return any(m in reason for m in _PERMANENT_SKIP_MARKERS)


def _download_changed(files: list[dict], max_docs: int, char_cap: int
                      ) -> tuple[list[dict], list[dict], list[str]]:
    """Download/extract changed files newest-first until caps are hit.
    Returns (documents, skipped, deferred_file_ids). Files skipped by the caps
    are deferred — NOT marked synced, so the next run picks them up."""
    ordered = sorted(files, key=lambda f: f.get('modifiedTime', ''), reverse=True)
    documents: list[dict] = []
    skipped:   list[dict] = []
    deferred:  list[str]  = []
    total_chars = 0

    to_fetch = []
    for f in ordered:
        if len(to_fetch) >= max_docs:
            deferred.append(f['id'])
            continue
        to_fetch.append(f)

    results: dict[str, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(_download_one, f): f['id'] for f in to_fetch}
        for future in as_completed(futures):
            fid = futures[future]
            try:
                results[fid] = future.result()
            except Exception as e:
                results[fid] = ('', f'exception: {e}')

    for f in to_fetch:
        text, note = results.get(f['id'], ('', 'no result'))
        if not text:
            skipped.append({'id': f['id'], 'name': f['name'],
                            'reason': note or 'empty',
                            'modifiedTime': f.get('modifiedTime', ''),
                            'top_folder': f.get('top_folder', '')})
            continue
        if total_chars + len(text) > char_cap:
            deferred.append(f['id'])
            continue
        total_chars += len(text)
        documents.append({
            'id':       f['id'],
            'name':     f['name'],
            'modified': f.get('modifiedTime', ''),
            'text':     text,
        })
    return documents, skipped, deferred


# ── Website inference for new-client proposals ─────────────────────────────────
# Harvest business email/URL domains from around the proposal folder (share
# permissions, last-modifying users, doc text) and auto-fill proposed_website
# when a domain stem matches the company name. Freemail and our own domains
# never qualify.

_GENERIC_EMAIL_DOMAINS = {
    'gmail.com', 'googlemail.com', 'yahoo.com', 'ymail.com', 'hotmail.com',
    'outlook.com', 'live.com', 'msn.com', 'aol.com', 'icloud.com', 'me.com',
    'mac.com', 'protonmail.com', 'proton.me', 'pm.me', 'zoho.com', 'gmx.com',
    'mail.com', 'comcast.net', 'verizon.net', 'att.net', 'sbcglobal.net',
    'bwcoconsulting.com', 'google.com', 'gserviceaccount.com',
    # ubiquitous URLs in business docs that are never the client's site
    'sam.gov', 'sbir.gov', 'grants.gov', 'linkedin.com', 'docs.google.com',
    'drive.google.com', 'sba.gov', 'irs.gov',
}
_EMAIL_RE      = re.compile(r'[a-z0-9._%+-]+@([a-z0-9][a-z0-9.-]*\.[a-z]{2,})', re.I)
_URL_DOMAIN_RE = re.compile(r'(?:https?://|www\.)([a-z0-9][a-z0-9.-]*\.[a-z]{2,})', re.I)
_INTERNAL_RE   = re.compile(r'[\s_-]*internal\s*$', re.I)
_LEGAL_SUFFIXES = {'inc', 'llc', 'corp', 'co', 'ltd', 'pllc', 'incorporated',
                   'corporation', 'company', 'limited', 'lp', 'llp'}
_PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def _norm_name(name: str) -> str:
    """Same normalization as the view's fuzzy folder matcher: strip trailing
    _INTERNAL, lowercase, drop punctuation and legal suffixes."""
    text = _INTERNAL_RE.sub('', str(name or '')).lower()
    text = text.translate(_PUNCT_TABLE)
    return ' '.join(w for w in text.split() if w not in _LEGAL_SUFFIXES)


def _business_domain(raw: str) -> str:
    """Lowercased domain with common subdomains stripped; '' if freemail,
    our own, or otherwise never-a-client-website."""
    d = (raw or '').lower().strip().strip('.')
    for prefix in ('www.', 'mail.', 'email.'):
        if d.startswith(prefix):
            d = d[len(prefix):]
    if not d or '.' not in d:
        return ''
    if d in _GENERIC_EMAIL_DOMAINS or d.endswith('.gserviceaccount.com'):
        return ''
    return d


def _harvest_candidate_domains(perm_emails: list[str], listed_files: list[dict],
                               documents: list[dict]) -> Counter:
    """Business domains seen around a proposal folder, weighted by how strong
    the signal is: folder share permissions ×3, file last-modifying users ×2,
    emails/URLs found in doc text ×1 per occurrence."""
    counts: Counter = Counter()
    for e in perm_emails:
        if '@' in e:
            d = _business_domain(e.rsplit('@', 1)[-1])
            if d:
                counts[d] += 3
    for f in listed_files:
        editor = f.get('last_modified_by') or ''
        if '@' in editor:
            d = _business_domain(editor.rsplit('@', 1)[-1])
            if d:
                counts[d] += 2
    for doc in documents:
        text = doc.get('text', '')
        for regex in (_EMAIL_RE, _URL_DOMAIN_RE):
            for m in regex.finditer(text):
                d = _business_domain(m.group(1))
                if d:
                    counts[d] += 1
    return counts


def _infer_website(company_name: str, folder_name: str,
                   domain_counts: Counter) -> str:
    """Pick the candidate domain whose stem matches the company/folder name
    (exact → containment → acronym → difflib ≥ 0.8); frequency breaks ties.
    Returns 'https://<domain>' or ''."""
    names = {n for n in (_norm_name(company_name), _norm_name(folder_name)) if n}
    if not names or not domain_counts:
        return ''
    compact  = {n.replace(' ', '') for n in names}
    acronyms = {''.join(w[0] for w in n.split()) for n in names
                if len(n.split()) >= 3}
    best, best_score = '', 0.0
    for domain, _count in domain_counts.most_common():
        stem = domain.split('.')[0]
        if len(stem) < 3:
            continue
        if stem in compact:
            score = 1.0
        elif any(min(len(stem), len(c)) >= 5 and (stem in c or c in stem)
                 for c in compact):
            score = 0.9
        elif stem in acronyms:
            score = 0.85
        else:
            score = max((SequenceMatcher(None, stem, c).ratio() for c in compact),
                        default=0.0)
            if score < 0.8:
                score = 0.0
        if score > best_score:
            best, best_score = domain, score
    return f'https://{best}' if best else ''


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_blob_path: str) -> None:
    gcs = _gcs()

    print(f'Loading config from {config_blob_path}', flush=True)
    config = json.loads(gcs.bucket(_BUCKET).blob(config_blob_path).download_as_text())

    run_id       = config['run_id']
    if run_id == 'daily':   # Cloud Scheduler sentinel — stamp a real id per run
        run_id = datetime.now(timezone.utc).strftime('drive_sync_%Y-%m-%d_%H-%M-%S')
    drive_id     = config['drive_id']
    folder_ids   = config.get('folder_ids', [])
    new_folders  = config.get('new_client_folder_ids', [])
    full_resync  = bool(config.get('full_resync', False))
    dry_run      = bool(config.get('dry_run', False))
    max_docs     = int(config.get('max_docs_per_client', 40))
    char_cap     = int(config.get('per_client_char_cap', 150_000))
    max_props    = int(config.get('max_proposals', _DEFAULT_MAX_PROPOSALS))
    model        = config.get('model', _DEFAULT_MODEL)

    # Scheduled mode (Phase 2.2): scan sections -> auto-assign -> sync EVERY
    # assigned client -> attention queue -> chain profile rebuilds. Defaults keep
    # manual behaviour: a config without `mode` runs exactly as before.
    mode             = config.get('mode', 'manual')
    scheduled        = (mode == 'scheduled')
    scan_sections    = config.get('scan_sections', [])
    chain_profiles   = bool(config.get('chain_profiles', True))
    onboard_unmatched = bool(config.get('onboard_unmatched', True))

    task_timeout_s = max(_MIN_TASK_TIMEOUT_S,
                         min(int(config.get('task_timeout_s',
                                            _DEFAULT_TASK_TIMEOUT_S)),
                             _MAX_TASK_TIMEOUT_S))
    work_budget_s = task_timeout_s - _DEADLINE_MARGIN_S
    deadline      = time.monotonic() + work_budget_s
    stopped_early = None
    print(f'Time budget: {work_budget_s}s of a {task_timeout_s}s task timeout.',
          flush=True)

    # Client phase gets a tighter deadline when proposals are planned
    n_props_planned = min(len(new_folders), max_props)
    client_deadline = deadline
    if n_props_planned:
        reserve = min(n_props_planned * _PROPOSAL_RESERVE_PER_ITEM_S,
                      work_budget_s // 2)
        client_deadline = deadline - reserve
        print(f'Reserving {reserve}s of the budget for {n_props_planned} '
              f'planned proposals.', flush=True)

    anth       = Anthropic(api_key=_get_secret('anthropic-api-key'))
    oai        = OpenAI(api_key=_get_secret('openai-api-key'))
    encoding   = tiktoken.get_encoding('cl100k_base')

    assignments_doc = _load_json_blob(gcs, _ASSIGN_BLOB) or {}
    assignments     = assignments_doc.get('assignments', {})
    unassigned      = assignments_doc.get('unassigned', {})

    sync_state = _load_json_blob(gcs, _SYNC_STATE_BLOB) or {'files': {}}
    state_files: dict = sync_state.setdefault('files', {})
    # Proposal rotation cursor: folder_id → last-proposed ISO date. Ensures a
    # large unassigned backlog advances across capped runs instead of
    # re-proposing the same first chunk every time. Advanced on non-dry runs.
    proposed_state: dict = sync_state.setdefault('proposed', {})

    print('Building Drive service (ADC)…', flush=True)
    svc = drive_client.build_drive_service()

    print('Loading client frames…', flush=True)
    frames = _load_client_frames(gcs)
    print(f'  {len(frames)} parquet files', flush=True)

    scan_counts = {'seen': 0, 'auto': 0, 'review': 0}
    if scheduled:
        # Discover + auto-assign new folders, then sync EVERY assigned client.
        norms = drive_scan.client_norms(_name_by_key(frames))
        section_list = [{'id': s, 'name': s} for s in scan_sections]
        if section_list and not dry_run:
            try:
                res = drive_scan.scan_and_assign(
                    svc, drive_id, section_list, assignments_doc, norms)
                _save_json_blob(gcs, _ASSIGN_BLOB, assignments_doc)
                scan_counts = {'seen': res.folders_seen, 'auto': len(res.new_auto),
                               'review': len(res.new_unassigned)}
                print(f'  scan: {scan_counts["seen"]} seen, {scan_counts["auto"]} '
                      f'auto-assigned, {scan_counts["review"]} to review', flush=True)
            except Exception as e:
                print(f'  WARN scheduled scan failed: {e}', flush=True)
        assignments = assignments_doc.get('assignments', {})
        unassigned  = assignments_doc.get('unassigned', {})
        folder_ids  = list(assignments.keys())   # sync all assigned clients
        new_folders = []                          # scheduled runs make no one-shot proposals

    # Group assigned folders by client_key
    groups: dict[str, list[dict]] = {}
    for fid in folder_ids:
        a = assignments.get(fid)
        if not a or not a.get('client_key'):
            print(f'  WARN folder {fid} not in assignments — skipping', flush=True)
            continue
        groups.setdefault(a['client_key'], []).append({'id': fid, **a})

    today   = date.today().isoformat()
    results: list[dict] = []
    skipped_files: list[dict] = []
    proposals: list[dict] = []
    # pending[blob_name][client_key] = {column: value} — the per-company column
    # updates this run has produced but not yet flushed. At each checkpoint they
    # are re-applied onto a FRESHLY re-read copy of each parquet under a
    # generation guard (gcs_write.read_modify_write), so fathom-sync-job's
    # client_meetings_* columns written concurrently are never clobbered — the
    # old whole-file rewrite from the start-of-run snapshot did exactly that.
    pending: dict[str, dict[str, dict]] = {}
    files_scanned = files_changed = 0
    clients_updated = clients_unchanged = clients_errored = 0

    def _checkpoint(state_label: str, done: int, total: int) -> None:
        if not dry_run:
            for blob_name, updates in list(pending.items()):
                blob = gcs.bucket(_BUCKET).blob(blob_name)
                gw.read_modify_write(
                    blob,
                    lambda df, u=updates: gw.apply_column_updates(df, u, _group_mask),
                )
            pending.clear()
            sync_state['updated_at'] = datetime.now(timezone.utc).isoformat()
            _save_json_blob(gcs, _SYNC_STATE_BLOB, sync_state)
        _write_status(gcs, run_id, {
            'run_id': run_id, 'state': state_label, 'dry_run': dry_run,
            'clients_done': done, 'clients_total': total,
            'clients_updated': clients_updated,
            'clients_unchanged': clients_unchanged,
            'clients_errored': clients_errored,
            'files_scanned': files_scanned, 'files_changed': files_changed,
            'error': None,
        })

    total_groups = len(groups)
    group_items  = list(groups.items())
    for n, (client_key, group_folders) in enumerate(group_items, start=1):
        company_name = client_key.split('||', 1)[0]
        if time.monotonic() > client_deadline:
            print(f'Client-phase time budget exhausted at {n}/{total_groups} — '
                  f'deferring the rest (proposals still run).', flush=True)
            stopped_early = 'timeout'
            for key, folders in group_items[n - 1:]:
                results.append({'client_key': key,
                                'folder_ids': [g['id'] for g in folders],
                                'outcome': 'deferred', 'files_processed': 0,
                                'summary_changed': False,
                                'note': 'time budget exhausted — run sync again'})
            break
        print(f'[{n}/{total_groups}] {company_name}', flush=True)
        outcome = {'client_key': client_key,
                   'folder_ids': [g['id'] for g in group_folders],
                   'outcome': 'unchanged', 'files_processed': 0,
                   'summary_changed': False, 'note': ''}
        try:
            # 1. List + diff
            listed: list[dict] = []
            folder_ids_here = set()
            for g in group_folders:
                folder_ids_here.add(g['id'])
                for f in drive_client.list_files_recursive(svc, drive_id, g['id']):
                    listed.append({**f, 'top_folder': g['id']})
            listed = [f for f in listed if drive_client.is_extractable(f)]
            files_scanned += len(listed)

            listed_ids = {f['id'] for f in listed}
            # Prune sync_state entries for vanished files in these folders
            for fid in [fid for fid, meta in state_files.items()
                        if meta.get('folder_id') in folder_ids_here
                        and fid not in listed_ids]:
                del state_files[fid]

            if full_resync:
                changed = listed
            else:
                changed = [f for f in listed
                           if state_files.get(f['id'], {}).get('modifiedTime')
                           != f['modifiedTime']]
            files_changed += len(changed)

            if not changed:
                clients_unchanged += 1
                results.append(outcome)
                continue

            # 2. Download + extract
            documents, skipped, _deferred = _download_changed(
                changed, max_docs, char_cap)
            skipped_files.extend(skipped)

            # Permanently unextractable files are marked synced immediately —
            # they contribute nothing to the profile and would otherwise be
            # re-downloaded and re-failed on every single run
            if not dry_run:
                for sk in skipped:
                    if sk.get('id') and _is_permanent_skip(sk.get('reason', '')):
                        state_files[sk['id']] = {
                            'modifiedTime': sk.get('modifiedTime', ''),
                            'folder_id':    sk.get('top_folder')
                                            or group_folders[0]['id'],
                            'synced_at':    today,
                            'skip_reason':  sk['reason'],
                        }

            if not documents:
                outcome['note'] = 'changed files found but nothing extractable'
                clients_unchanged += 1
                results.append(outcome)
                continue
            outcome['files_processed'] = len(documents)

            # 3. Current profile
            current_summary, existing_docs_data = '', ''
            for df in frames.values():
                mask = _group_mask(df, client_key)
                if mask.any():
                    row = df[mask].iloc[0]
                    current_summary    = str(row.get('summary') or '')
                    existing_docs_data = str(row.get('client_docs_data') or '')
                    break

            # 4. Claude merge
            merged = _claude_json(anth, model, _MERGE_SYSTEM, {
                'company_name':       company_name,
                'current_summary':    current_summary,
                'existing_docs_data': existing_docs_data,
                'new_documents': [
                    {'name': d['name'], 'modified': d['modified'], 'text': d['text']}
                    for d in documents
                ],
            })

            no_change   = bool(merged.get('no_meaningful_change'))
            new_summary = str(merged.get('updated_summary') or '').strip()
            digest      = str(merged.get('docs_digest') or '').strip()
            extracted   = merged.get('extracted') or {}

            do_reembed = (not no_change and new_summary
                          and new_summary != current_summary.strip())

            # 5. Write columns
            if not dry_run:
                # source_files: merge prior list with the docs just processed
                prior_files = []
                try:
                    prior_files = (json.loads(existing_docs_data) or {}).get('source_files', [])
                except Exception:
                    pass
                seen_names = set()
                source_files = []
                for sf in ([{'name': d['name'], 'modified': d['modified']}
                            for d in documents] + prior_files):
                    if sf.get('name') in seen_names:
                        continue
                    seen_names.add(sf.get('name'))
                    source_files.append(sf)
                docs_data = json.dumps({
                    'extracted':    extracted,
                    'source_files': source_files[:_MAX_SOURCE_FILES],
                    'last_run':     run_id,
                })

                new_emb = None
                if do_reembed:
                    emb = _get_embedding(new_summary, oai, encoding)
                    if emb is not None:
                        # float64 to match the dtype of existing rows — pyarrow
                        # cannot mix float32 and float64 ndarrays in one column
                        new_emb = np.array(emb, dtype=np.float64)

                cols: dict = {
                    'client_docs_data':    docs_data,
                    'client_docs_summary': digest,
                    'docs_updated_at':     today,
                }
                if new_emb is not None:
                    cols['summary']    = new_summary
                    cols['embeddings'] = new_emb

                found = False
                for blob_name, df in frames.items():
                    # The start-of-run snapshot tells us which blob holds the
                    # client's rows; the mask is re-derived on the fresh frame at
                    # flush time by apply_column_updates.
                    if not _group_mask(df, client_key).any():
                        continue
                    found = True
                    pending.setdefault(blob_name, {})[client_key] = cols
                if not found:
                    outcome['outcome'] = 'error'
                    outcome['note']    = 'no client rows matched client_key'
                    clients_errored   += 1
                    results.append(outcome)
                    continue

                # Mark files synced only after columns are written
                listed_by_id = {f['id']: f for f in listed}
                for d in documents:
                    src = listed_by_id.get(d['id'], {})
                    state_files[d['id']] = {
                        'modifiedTime': src.get('modifiedTime', d['modified']),
                        'folder_id':    src.get('top_folder', group_folders[0]['id']),
                        'synced_at':    today,
                    }

            outcome['outcome']         = 'updated'
            outcome['summary_changed'] = bool(do_reembed)
            if no_change:
                outcome['note'] = 'docs stored; summary unchanged (no meaningful change)'
            clients_updated += 1
            results.append(outcome)

        except Exception as e:
            print(f'  ERROR {company_name}: {e}', flush=True)
            traceback.print_exc()
            outcome['outcome'] = 'error'
            outcome['note']    = str(e)[:300]
            clients_errored   += 1
            results.append(outcome)

        if n % _CHECKPOINT_EVERY == 0:
            print(f'  checkpoint at {n}/{total_groups}', flush=True)
            _checkpoint('running', n, total_groups)

    # ── New-client proposals ───────────────────────────────────────────────────
    # Capped per run: each proposal costs a folder download + a Claude call
    # (~1 min). Deferred folders stay in `unassigned` and are picked up by
    # the next sync run automatically.
    # Never-proposed folders first, then least-recently-proposed
    new_folders_sorted = sorted(new_folders,
                                key=lambda f: proposed_state.get(f, ''))
    todo_folders       = new_folders_sorted[:max_props]
    proposals_deferred = len(new_folders) - len(todo_folders)
    if proposals_deferred:
        print(f'{proposals_deferred} proposal folders deferred '
              f'(max_proposals={max_props})', flush=True)

    for p_idx, fid in enumerate(todo_folders):
        if time.monotonic() > deadline:
            stopped_early       = 'timeout'
            proposals_deferred += len(todo_folders) - p_idx
            print(f'Time budget exhausted at proposal {p_idx}/{len(todo_folders)} '
                  f'— stopping early.', flush=True)
            break
        meta        = unassigned.get(fid, {})
        folder_name = meta.get('folder_name', fid)
        print(f'[proposal] {folder_name}', flush=True)
        try:
            listed = [f for f in drive_client.list_files_recursive(svc, drive_id, fid)
                      if drive_client.is_extractable(f)]
            files_scanned += len(listed)
            documents, skipped, _ = _download_changed(listed, max_docs, char_cap)
            skipped_files.extend(skipped)
            try:
                perm_emails = drive_client.list_permission_emails(svc, fid)
            except Exception as e:
                print(f'  WARN permissions lookup failed for {folder_name}: {e}',
                      flush=True)
                perm_emails = []
            domain_counts     = _harvest_candidate_domains(perm_emails, listed, documents)
            candidate_domains = [d for d, _ in domain_counts.most_common(10)]
            if not documents:
                # No `continue` here — the rotation/bookkeeping below must
                # still run, or empty folders would hog the per-run cap forever
                proposals.append({
                    'folder_id': fid, 'folder_name': folder_name,
                    'proposed_name': '', 'proposed_website': '',
                    'website_source': '', 'candidate_domains': candidate_domains,
                    'proposed_summary': '', 'docs_summary': '',
                    'docs_data': '', 'error': 'no extractable documents',
                })
            else:
                prop = _claude_json(anth, model, _PROPOSE_SYSTEM, {
                    'folder_name': folder_name,
                    'candidate_domains': candidate_domains,
                    'documents': [
                        {'name': d['name'], 'modified': d['modified'], 'text': d['text']}
                        for d in documents
                    ],
                })
                website        = str(prop.get('proposed_website') or '').strip()
                website_source = 'claude' if website else ''
                if not website:
                    website = _infer_website(str(prop.get('proposed_name') or ''),
                                             folder_name, domain_counts)
                    if website:
                        website_source = 'domain_match'
                proposals.append({
                    'folder_id':         fid,
                    'folder_name':       folder_name,
                    'proposed_name':     str(prop.get('proposed_name') or '').strip(),
                    'proposed_website':  website,
                    'website_source':    website_source,
                    'candidate_domains': candidate_domains,
                    'proposed_summary':  str(prop.get('proposed_summary') or '').strip(),
                    'docs_summary':      str(prop.get('docs_digest') or '').strip(),
                    'docs_data': json.dumps({
                        'extracted':    prop.get('extracted') or {},
                        'source_files': [{'name': d['name'], 'modified': d['modified']}
                                         for d in documents][:_MAX_SOURCE_FILES],
                        'last_run':     run_id,
                    }),
                    'error': None,
                })
        except Exception as e:
            print(f'  ERROR proposal {folder_name}: {e}', flush=True)
            proposals.append({
                'folder_id': fid, 'folder_name': folder_name,
                'proposed_name': '', 'proposed_website': '',
                'website_source': '', 'candidate_domains': [],
                'proposed_summary': '', 'docs_summary': '', 'docs_data': '',
                'error': str(e)[:300],
            })

        if not dry_run:
            proposed_state[fid] = today

        # Interim status after every proposal — the work is expensive, so a
        # crash or kill must never lose the proposals accumulated so far.
        _write_status(gcs, run_id, {
            'run_id': run_id, 'state': 'running', 'dry_run': dry_run,
            'clients_done': total_groups, 'clients_total': total_groups,
            'proposals_done': p_idx + 1, 'proposals_total': len(todo_folders),
            'new_client_proposals': proposals,
            'error': None,
        })

    # ── Scheduled mode: attention queue (2.5) + profile chaining (2.3) ─────────
    profile_run_id = None
    attention_written = 0
    if scheduled and not dry_run:
        items = []
        if onboard_unmatched:
            # Until the Phase 3 onboarding agent exists, unmatched folders land
            # in the attention queue instead of generating one-shot proposals.
            for fid, u in assignments_doc.get('unassigned', {}).items():
                items.append({
                    'kind': 'drive_unmatched_folder', 'subject_key': fid,
                    'subject_label': u.get('folder_name', ''),
                    'source_job': 'drive-sync-job', 'run_id': run_id,
                    'detail': {'section': u.get('section', ''), 'note': u.get('note', '')},
                })
        for r in results:
            ck = r.get('client_key', '')
            if not ck:
                continue
            if r.get('outcome') == 'deferred':
                items.append({'kind': 'drive_deferred', 'subject_key': ck,
                              'subject_label': ck.split('||', 1)[0],
                              'source_job': 'drive-sync-job', 'run_id': run_id,
                              'detail': {'note': r.get('note', '')}})
            elif r.get('outcome') == 'error':
                items.append({'kind': 'drive_extract_failed', 'subject_key': ck,
                              'subject_label': ck.split('||', 1)[0],
                              'source_job': 'drive-sync-job', 'run_id': run_id,
                              'detail': {'note': r.get('note', '')}})
        try:
            attention_written = attention.enqueue(gcs, items)
        except Exception as e:
            print(f'  WARN attention enqueue failed: {e}', flush=True)

        # Chain a profile rebuild for clients whose material changed this run.
        changed_keys = sorted({r.get('client_key') for r in results
                               if r.get('outcome') == 'updated' and r.get('client_key')})
        if chain_profiles and changed_keys:
            try:
                from google.cloud import run_v2
                profile_run_id = datetime.now(timezone.utc).strftime(
                    'client_profile_%Y-%m-%d_%H-%M-%S_drivechain')
                path = jc.write_job_config(gcs, 'client-profile-configs/', {
                    'run_id': profile_run_id, 'pool': 'clients',
                    'company_keys': changed_keys,
                    'sources': ['website', 'drive', 'technology'],
                    'target_aspects': 4, 'max_markets': 4,
                    'assess_defense': True, 'assess_unexplored': True,
                    'max_unexplored': 3, 'model': _DEFAULT_MODEL,
                    'concurrency': 4, 'dry_run': False,
                })
                jc.trigger_job(None, 'client-profile-job', path,
                               jobs_client=run_v2.JobsClient())
                print(f'  chained client-profile-job {profile_run_id} for '
                      f'{len(changed_keys)} client(s)', flush=True)
            except Exception as e:
                profile_run_id = None
                print(f'  WARN profile chaining failed: {e}', flush=True)

    # ── Final checkpoint + status ─────────────────────────────────────────────
    _checkpoint('running', total_groups, total_groups)
    _write_status(gcs, run_id, {
        'run_id':            run_id,
        'state':             'complete',
        'dry_run':           dry_run,
        'mode':              mode,
        'stopped_early':     stopped_early,
        'task_timeout_s':    task_timeout_s,
        'max_proposals':     max_props,
        'clients_total':     total_groups,
        'clients_updated':   clients_updated,
        'clients_unchanged': clients_unchanged,
        'clients_errored':   clients_errored,
        'files_scanned':     files_scanned,
        'files_changed':     files_changed,
        'files_skipped':     skipped_files[:_MAX_SKIPPED_LISTED],
        'results':           results,
        'new_client_proposals': proposals,
        'proposals_deferred':   proposals_deferred,
        'scan':              scan_counts,
        'attention_written': attention_written,
        'profile_run_id':    profile_run_id,
        'error':             None,
    })
    print(f'\nDone. {clients_updated} updated, {clients_unchanged} unchanged, '
          f'{clients_errored} errored, {len(proposals)} proposals '
          f'({proposals_deferred} deferred).', flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python jobs/drive_sync_job.py <config_blob_path>', file=sys.stderr)
        sys.exit(1)

    config_path     = sys.argv[1]
    run_id_fallback = config_path.split('/')[-1].replace('.json', '')
    _gcs_client     = _gcs()

    # Run lock: one drive-sync-job execution at a time. A second run (scheduled
    # or a manual trigger) exits cleanly as skipped_locked rather than racing
    # the parquet writes. ttl = the run's own task timeout + 15 min, so a run
    # killed without releasing is taken over by the next after it could only
    # still be alive. The lock key is the job name, not the run_id.
    _cfg = {}
    try:
        _cfg = json.loads(
            _gcs_client.bucket(_BUCKET).blob(config_path).download_as_text())
    except Exception:
        pass
    _run_id = _cfg.get('run_id', run_id_fallback)
    _ttl = max(_MIN_TASK_TIMEOUT_S,
               min(int(_cfg.get('task_timeout_s', _DEFAULT_TASK_TIMEOUT_S)),
                   _MAX_TASK_TIMEOUT_S)) + 900
    _lock = gw.acquire_lock(_gcs_client.bucket(_BUCKET), 'drive-sync-job', _run_id, _ttl)
    if _lock is None:
        print('Another drive-sync-job run holds the lock — exiting cleanly.', flush=True)
        try:
            _write_status(_gcs_client, _run_id,
                          {'run_id': _run_id, 'state': 'skipped_locked', 'error': None})
        except Exception:
            pass
        sys.exit(0)

    try:
        main(config_path)
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        try:
            _write_status(
                _gcs_client,
                run_id_fallback,
                {'run_id': run_id_fallback, 'state': 'error', 'error': tb},
            )
        except Exception:
            pass
        sys.exit(1)
    finally:
        gw.release_lock(_lock)
