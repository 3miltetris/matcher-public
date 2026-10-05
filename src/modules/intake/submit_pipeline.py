"""
submit_pipeline.py — the intake submit, as idempotent resumable steps.

  submit()         validate → profile_rows        (in the request; the founder
                                                    sees success once rows exist)
  run_remaining()  drive_folder → copy_uploads → gdoc → hubspot →
                   submission_record → writeback → profile_trigger
                                                   (background task)

Each step's status and output ids are recorded in session['steps'] and the
session is saved after every step, so a retry — automatic (3 attempts with
backoff), or manual:

    python -m src.modules.intake.submit_pipeline resume <session_id>

— skips what is done and every step is itself safe to repeat (see the
idempotency notes in each module). Logs carry session ids and step names,
never answers.
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

from src.modules.intake import drive_folder as dfo
from src.modules.intake import gdoc_writer, hubspot_sync, notify
from src.modules.intake import profile_trigger, profile_writer
from src.modules.intake import schema as sc
from src.modules.intake import schema_export as sx
from src.modules.intake import sessions as ss
from src.modules.intake import uploads as up
from src.modules.intake.validation import validate

log = logging.getLogger('intake')

ASSIGN_BLOB = 'drive-sync-configs/assignments.json'
STEPS = ['drive_folder', 'copy_uploads', 'gdoc', 'hubspot', 'submission_record',
         'writeback', 'profile_trigger']
MAX_ATTEMPTS = 3


class ValidationFailed(ValueError):
    def __init__(self, errors: dict):
        super().__init__('invalid answers')
        self.errors = errors


@dataclass
class Deps:
    """Everything external, injectable for tests."""
    storage_client: object
    bm: object
    embed: callable
    drive_svc: object = None
    hubspot: object = None
    jobs_client: object = None
    drive_id: str = field(default_factory=lambda: os.environ.get('INTAKE_SHARED_DRIVE_ID', ''))
    parent_id: str = field(default_factory=lambda: os.environ.get('INTAKE_PARENT_FOLDER_ID', ''))
    sleep: callable = time.sleep
    notify_ok: callable = notify.submission_received
    notify_fail: callable = notify.step_failed


def _done(session: dict, step: str) -> dict | None:
    rec = session.get('steps', {}).get(step)
    return rec if rec and rec.get('status') == 'done' else None


def _company(session: dict) -> str:
    return str(session.get('answers', {}).get(sc.COMPANY_NAME, '')).strip() or '(unnamed)'


# ── In-request part ────────────────────────────────────────────────────────

def submit(deps: Deps, session: dict) -> dict:
    """Validate and write the prospect rows. Raises ValidationFailed; the
    caller then schedules run_remaining(session_id)."""
    clean, errors = validate(session.get('answers', {}))
    if errors:
        raise ValidationFailed(errors)
    if not _done(session, 'profile_rows'):
        session['answers']      = clean
        session['submitted_at'] = session.get('submitted_at') or ss.now_iso()
        res = profile_writer.write_prospect(
            deps.storage_client, clean, session['session_id'],
            session['submitted_at'], deps.embed)
        session.setdefault('steps', {})['profile_rows'] = {
            'status': 'done', 'at': ss.now_iso(), **res}
        session['status'] = ss.SUBMITTED
        ss.save_session(deps.bm, session)
        log.info('intake %s: profile_rows done (%s, created=%s)',
                 session['session_id'], res['pool'], res['created'])
    return session


# ── Background part ────────────────────────────────────────────────────────

def _assigned_folder_ids(deps: Deps, company_key: str) -> list[str]:
    obj, _ = deps.bm.download_json(ASSIGN_BLOB)
    return [fid for fid, a in ((obj or {}).get('assignments') or {}).items()
            if isinstance(a, dict) and a.get('client_key') == company_key]


def _run_step(deps: Deps, session: dict, step: str) -> dict:
    rows   = session['steps']['profile_rows']
    pool, key = rows['pool'], rows['company_key']
    answers = session['answers']
    sid     = session['session_id']

    if step == 'drive_folder':
        prev  = profile_writer.previous_ids(deps.storage_client, pool, key)
        known = [prev.get('drive_folder_id')] + _assigned_folder_ids(deps, key)
        return dfo.find_or_create(deps.drive_svc, deps.drive_id, deps.parent_id,
                                  answers.get(sc.COMPANY_NAME, ''),
                                  [k for k in known if k])

    folder = session['steps']['drive_folder']

    if step == 'copy_uploads':
        copied = {}
        for u in up.verified_uploads(deps.storage_client, session):
            content = deps.bm.download_bytes(u['object'])
            copied[u['upload_id']] = dfo.copy_upload(
                deps.drive_svc, deps.drive_id, folder['folder_id'], u['filename'],
                content, u['content_type'], u['upload_id'],
                session['submitted_at'][:10])
        return {'files': copied}

    if step == 'gdoc':
        html = gdoc_writer.render_html(answers, session['submitted_at'],
                                       session.get('uploads', []))
        return dfo.write_doc(deps.drive_svc, deps.drive_id, folder['folder_id'],
                             gdoc_writer.doc_title(answers, session['submitted_at']),
                             html, sid)

    if step == 'hubspot':
        prev   = profile_writer.previous_ids(deps.storage_client, pool, key)
        ts_ms  = int(datetime.fromisoformat(session['submitted_at']).timestamp() * 1000)
        domain = profile_writer.bare_domain(answers.get(sc.WEBSITE))
        return hubspot_sync.sync(
            answers, domain,
            {'dd_submitted_at': str(ts_ms), 'dd_drive_folder_url': folder['folder_url'],
             'dd_matcher_profile_id': key, 'dd_intake_source': sx.INTAKE_SOURCE},
            known_company_id=prev.get('hubspot_company_id'), client=deps.hubspot)

    if step == 'submission_record':
        blob = f'{ss.SUBMIT_PREFIX}{sid}.json'
        deps.bm.upload_json(blob, {
            'session_id': sid, 'submitted_at': session['submitted_at'],
            'final_answers': answers, 'uploads': session.get('uploads', []),
            'ai_draft': session.get('ai_draft'),          # Phase 2
            'enrichment': session.get('enrichment'),      # Phase 3
            'field_outcomes': None,                       # Phase 2: accepted/edited/rejected
            'pool': pool, 'company_key': key})
        return {'blob': blob}

    if step == 'writeback':
        hs  = session['steps']['hubspot']
        doc = session['steps']['gdoc']
        touched = profile_writer.update_intake_data(deps.storage_client, pool, key, {
            'drive_folder_id':    folder['folder_id'],
            'drive_folder_url':   folder['folder_url'],
            'drive_doc_url':      doc['doc_url'],
            'needs_review':       bool(folder.get('needs_review')),
            'hubspot_company_id': hs['hubspot_company_id'],
            'hubspot_contact_id': hs['hubspot_contact_id'],
        })
        return {'blobs': touched}

    if step == 'profile_trigger':
        return {'run_id': profile_trigger.trigger(deps.storage_client, sid, pool, key,
                                                  jobs_client=deps.jobs_client)}

    raise ValueError(step)


def run_remaining(deps: Deps, session: dict) -> dict:
    sid = session['session_id']
    for step in STEPS:
        if _done(session, step):
            continue
        rec = session['steps'].setdefault(step, {'attempts': 0})
        while True:
            rec['attempts'] = rec.get('attempts', 0) + 1
            try:
                out = _run_step(deps, session, step)
                session['steps'][step] = {'status': 'done', 'at': ss.now_iso(),
                                          'attempts': rec['attempts'], **out}
                ss.save_session(deps.bm, session)
                log.info('intake %s: %s done', sid, step)
                break
            except Exception as e:
                rec.update(status='error', error=f'{type(e).__name__}: {e}'[:500])
                log.warning('intake %s: %s attempt %d failed: %s', sid, step,
                            rec['attempts'], type(e).__name__)
                if rec['attempts'] % MAX_ATTEMPTS == 0:
                    session['status'] = ss.FAILED
                    ss.save_session(deps.bm, session)
                    deps.notify_fail(sid, _company(session), step, rec['error'])
                    return session
                deps.sleep(2 ** rec['attempts'])
    session['status'] = ss.COMPLETE
    ss.save_session(deps.bm, session)
    deps.notify_ok(sid, _company(session), {
        'Drive folder': session['steps']['drive_folder'].get('folder_url'),
        'Responses doc': session['steps']['gdoc'].get('doc_url'),
        'HubSpot company': 'https://app.hubspot.com/contacts/_/record/0-2/'
                           + str(session['steps']['hubspot'].get('hubspot_company_id')),
    })
    return session


# ── Production wiring + manual resume ──────────────────────────────────────

def production_deps() -> Deps:
    from google.cloud import storage

    from src.modules import drive_client
    from src.modules.Embedding.text_embedder import TextProcessor
    from src.modules.GoogleBucketManager.bucket_manager import BucketManager
    client = storage.Client()
    tp = TextProcessor(api_key=os.environ.get('OPENAI_API_KEY'))
    return Deps(storage_client=client, bm=BucketManager(ss.BUCKET, client=client),
                embed=tp.get_embedding,
                drive_svc=drive_client.build_drive_service(scopes=drive_client.DRIVE_WRITE_SCOPES),
                hubspot=hubspot_sync.HubSpot())


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] != 'resume':
        print('usage: python -m src.modules.intake.submit_pipeline resume <session_id>')
        return 2
    logging.basicConfig(level=logging.INFO)
    deps    = production_deps()
    session = ss.load_session(deps.bm, argv[2], allow_expired=True)
    if not _done(session, 'profile_rows'):
        submit(deps, session)
    session = run_remaining(deps, session)
    print(json.dumps({'status': session['status'],
                      'steps': {k: v.get('status') for k, v in session['steps'].items()}}))
    return 0 if session['status'] == ss.COMPLETE else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
