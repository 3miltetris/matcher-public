#!/usr/bin/env python3
"""
contract_sync_job.py — Cloud Run Job: sync signed HubSpot quotes + their Terms &
Conditions PDFs into the clients shared Google Drive.

  python jobs/contract_sync_job.py --backfill --dry-run --limit 10   # review first
  python jobs/contract_sync_job.py --backfill                        # all signed quotes
  python jobs/contract_sync_job.py --reconcile                       # lookback sync + reconcile
  python jobs/contract_sync_job.py --reconcile-only                  # move now-matched uncategorized files

Matched clients: {Client}_INTERNAL/Contracts/. Unmatched: the top-level
Uncategorized-Contracts folder (reused, not re-created). Idempotent across the
whole shared drive via appProperties (hubspot_quote_id + kind), so overlapping
scheduled windows never duplicate. Runs as intake-web@ (Content Manager on the
drive); the client→folder map is GCS drive-sync-configs/assignments.json.

Env:
  HUBSPOT_API_KEY / HUBSPOT_TOKEN   Private app token (scopes: crm.objects.quotes.read,
                                    crm.objects.companies.read, crm.objects.deals.read)
  LOOKBACK_DAYS                     Window for non-backfill runs (default 3)
  CONTRACT_SYNC_BUCKET              GCS bucket (default cc-matcher-bucket-jeg-v1)
  GOOGLE_IMPERSONATE_USER           Optional: impersonate a Workspace user (DWD) to
                                    read T&C sources that live outside the shared drive
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import requests
from googleapiclient.errors import HttpError

from src.modules import contract_sync as cs
from src.modules import drive_client as dc

_BUCKET = os.environ.get('CONTRACT_SYNC_BUCKET', 'cc-matcher-bucket-jeg-v1')
_ASSIGN_BLOB = 'drive-sync-configs/assignments.json'
_STATUS_PREFIX = 'contract-sync-jobs/'
LOOKBACK_DAYS = int(os.environ.get('LOOKBACK_DAYS', '3'))

log = logging.getLogger('contract-sync')


# ---------------------------------------------------------------- GCS
def _gcs():
    from google.cloud import storage
    return storage.Client()


def _load_assignments(gcs) -> tuple[str, dict]:
    raw = gcs.bucket(_BUCKET).blob(_ASSIGN_BLOB).download_as_text()
    doc = json.loads(raw)
    return doc.get('drive_id', ''), doc.get('assignments', {})


def _write_status(gcs, run_id: str, payload: dict) -> None:
    gcs.bucket(_BUCKET).blob(f'{_STATUS_PREFIX}{run_id}/status.json').upload_from_string(
        json.dumps(payload), content_type='application/json')


# ---------------------------------------------------------------- Drive auth
def _drive_service():
    import google.auth
    creds, _ = google.auth.default(scopes=dc.DRIVE_WRITE_SCOPES)
    subject = os.environ.get('GOOGLE_IMPERSONATE_USER')
    if subject and hasattr(creds, 'with_subject'):
        creds = creds.with_subject(subject)
    return dc.build_drive_service(credentials=creds)


# ---------------------------------------------------------------- per-quote
class Ctx:
    def __init__(self, hs, svc, drive_id, index, dry):
        self.hs, self.svc, self.drive_id = hs, svc, drive_id
        self.index, self.dry = index, dry
        self.folder_cache: dict = {}
        self.uncategorized_id = None
        self.uncategorized_resolved = False

    def uncategorized(self) -> str | None:
        if not self.uncategorized_resolved:
            self.uncategorized_id = cs.find_root_folder(self.svc, self.drive_id, cs.UNCATEGORIZED_NAME)
            if self.uncategorized_id is None and not self.dry:
                self.uncategorized_id = cs.find_or_create_child(
                    self.svc, self.drive_id, self.drive_id, cs.UNCATEGORIZED_NAME,
                    'uncategorized', self.dry, log=log.info)
            self.uncategorized_resolved = True
        return self.uncategorized_id

    def contracts_folder(self, client_key: str) -> str | None:
        if client_key in self.folder_cache:
            return self.folder_cache[client_key]
        folder = cs.pick_folder(client_key, self.index)
        cid = None
        if folder:
            cid = cs.find_or_create_child(self.svc, self.drive_id, folder['folder_id'],
                                          cs.CONTRACTS_NAME, 'contracts', self.dry, log=log.info)
        self.folder_cache[client_key] = cid
        return cid


def _existing_by_kind(ctx: Ctx, qid: str) -> dict:
    out = {}
    for f in cs.find_quote_files(ctx.svc, ctx.drive_id, qid):
        kind = (f.get('appProperties') or {}).get('kind', '')
        parents = f.get('parents') or []
        out[kind] = {'id': f['id'], 'parent': parents[0] if parents else None, 'name': f.get('name', '')}
    return out


def process_quote(ctx: Ctx, quote: dict) -> dict:
    qid = quote['id']
    p = quote.get('properties', {})
    title = cs.safe_name(p.get('hs_title') or f'Quote {qid}')
    date = (p.get('hs_esign_date') or '')[:10] or 'undated'

    company = cs.company_for_quote(ctx.hs, qid)
    if company:
        client_key, reason = cs.resolve_client(company, ctx.index)
    else:
        client_key, reason = None, 'no_company'
    matched = client_key is not None
    cname = company['name'] if company else ''
    log.info('%s | %s | %s | %s', date, (cname or '-'), title,
             'MATCHED:' + client_key.split('||')[0] if matched else f'UNCATEGORIZED ({reason})')

    unc = ctx.uncategorized()  # always needed for move detection
    dest = ctx.contracts_folder(client_key) if matched else unc
    existing = _existing_by_kind(ctx, qid)
    base_props = {
        'hubspot_quote_id': qid,
        'hubspot_company_id': company['id'] if company else '',
        'hubspot_company_domain': cs.bare_domain(company.get('domain') or company.get('website')) if company else '',
        'hubspot_company_name': cname,
    }
    did_create = did_move = False

    def handle(kind: str, make_name, do_create):
        nonlocal did_create, did_move
        ex = existing.get(kind)
        action = cs.decide(ex['parent'] if ex else None, dest, unc, matched)
        if action == 'create':
            name = cs.prefixed(make_name(), cname, matched)
            if ctx.dry:
                log.info('    [dry-run] would upload %r', name)
            else:
                do_create(name)
            did_create = True
        elif action == 'move':
            name = make_name()  # unprefixed destination name
            if ctx.dry:
                log.info('    [dry-run] would move %r into %s', ex['name'], dest)
            else:
                cs.move_file(ctx.svc, ex['id'], add_parent=dest, remove_parent=ex['parent'], new_name=name)
                log.info('    reconciled %r', name)
            did_move = True

    # Signed quote PDF
    handle('signed_quote',
           lambda: f'{date} - {title} (Signed).pdf',
           lambda name: cs.upload_pdf(ctx.svc, dest, name,
                                      cs.download_signed_pdf(p['hs_pdf_download_link']),
                                      {**base_props, 'kind': 'signed_quote'}))

    # Terms & Conditions
    term_ids = cs.extract_term_ids(p.get('hs_terms'))
    if not term_ids:
        log.warning('    no Drive link found in Terms section')
    for i, src_id in enumerate(term_ids):
        kind = cs.term_kind(i)
        suffix = '' if i == 0 else f' ({i + 1})'
        handle(kind,
               lambda s=suffix: f'{date} - {title} - Terms and Conditions{s}.pdf',
               lambda name, sid=src_id, k=kind: cs.copy_terms(
                   ctx.svc, sid, dest, name, {**base_props, 'kind': k, 'tc_source_id': sid}))

    return {'matched': matched, 'did_create': did_create, 'did_move': did_move}


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--backfill', action='store_true', help='Process every signed quote')
    ap.add_argument('--dry-run', action='store_true', help='Log actions without writing to Drive')
    ap.add_argument('--limit', type=int, default=0, help='Stop after N quotes (testing)')
    ap.add_argument('--reconcile', action='store_true', help='After the sync pass, move now-matched uncategorized files')
    ap.add_argument('--reconcile-only', action='store_true', help='Only move now-matched uncategorized files')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    token = os.environ.get('HUBSPOT_API_KEY') or os.environ['HUBSPOT_TOKEN']
    run_id = f'contract_sync_{datetime.now(timezone.utc):%Y-%m-%d_%H-%M-%S}'
    gcs = _gcs()
    drive_id, assignments = _load_assignments(gcs)
    if not drive_id:
        raise SystemExit('assignments.json has no drive_id')
    index = cs.build_assignment_index(assignments)
    svc = _drive_service()
    hs = cs.make_hs_call(cs.make_hs_session(token))
    ctx = Ctx(hs, svc, drive_id, index, args.dry_run)

    counts = {'processed': 0, 'synced': 0, 'reconciled': 0, 'skipped': 0,
              'uncategorized': 0, 'failed': 0}

    if not args.reconcile_only:
        since_ms = None
        if not args.backfill:
            since = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
            since_ms = int(since.timestamp() * 1000)
        for n, quote in enumerate(cs.iter_signed_quotes(hs, since_ms), 1):
            counts['processed'] += 1
            try:
                r = process_quote(ctx, quote)
                counts['synced'] += int(r['did_create'])
                counts['reconciled'] += int(r['did_move'])
                counts['uncategorized'] += int(not r['matched'])
                if not (r['did_create'] or r['did_move']):
                    counts['skipped'] += 1
            except (requests.HTTPError, HttpError, ValueError, KeyError) as e:
                counts['failed'] += 1
                log.error('    FAILED quote %s: %s', quote.get('id'), e)
            if args.limit and n >= args.limit:
                break

    if args.reconcile or args.reconcile_only:
        counts['reconciled'] += reconcile_uncategorized(ctx)

    payload = {'run_id': run_id, 'dry_run': args.dry_run,
               'mode': 'reconcile_only' if args.reconcile_only else ('backfill' if args.backfill else 'lookback'),
               'drive_id': drive_id, 'counts': counts, 'error': None}
    try:
        _write_status(gcs, run_id, payload)
    except Exception as e:  # noqa: BLE001 — status write must never mask the run result
        log.warning('status write failed: %s', e)
    log.info('Done: %s', counts)
    sys.exit(1 if counts['failed'] else 0)


def reconcile_uncategorized(ctx: Ctx) -> int:
    """Scan Uncategorized-Contracts; move any file whose stored company domain/
    name now matches a client. No HubSpot calls — re-resolves from appProperties."""
    from src.modules.intake import drive_folder as df
    unc = ctx.uncategorized()
    if not unc:
        return 0
    moved = 0
    for f in df._children(ctx.svc, ctx.drive_id, unc):
        props = f.get('appProperties') or {}
        if not props.get('hubspot_quote_id'):
            continue
        company = {'domain': props.get('hubspot_company_domain', ''),
                   'name': props.get('hubspot_company_name', '')}
        client_key, _ = cs.resolve_client(company, ctx.index)
        if not client_key:
            continue
        dest = ctx.contracts_folder(client_key)
        if not dest:
            continue
        new_name = cs.strip_prefix(f.get('name', ''))
        if ctx.dry:
            log.info('    [dry-run] would reconcile %r -> %s', f.get('name'), client_key.split('||')[0])
        else:
            cs.move_file(ctx.svc, f['id'], add_parent=dest, remove_parent=unc, new_name=new_name)
            log.info('    reconciled %r -> %s', new_name, client_key.split('||')[0])
        moved += 1
    return moved


if __name__ == '__main__':
    main()
