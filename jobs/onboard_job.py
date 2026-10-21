"""
drive-onboard-job — the Drive onboarding agent orchestrator (Phase 3.2/3.10).

For each folder in the batch: create a Managed Agents session against the
pre-synced onboarding agent, send a briefing, run the agent's custom tool calls
IN THIS PROCESS (our GCP/Drive creds never leave here), enforce hard budgets,
capture the terminal submit_proposal, validate it against what the tools actually
surfaced, write the proposal + an attention item, and delete the session.

Config (drive-onboard-configs/{run_id}.json):
{
  "run_id": "...",            # or "daily" sentinel
  "select": "queue",          # "queue" (attention drive_unmatched_folder) | "ids"
  "folder_ids": [...],        # when select == "ids"
  "drive_id": "0A...",
  "max_folders": 10,
  "dry_run": false
}

Env (Secret Manager): ANTHROPIC_API_KEY. Needs anthropic >= 1.15 (Managed Agents).
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone

from google.cloud import storage

from src.modules import attention, drive_client, onboarding as ob
from src.modules.tools import company_lookup, drive_read

_BUCKET = 'cc-matcher-bucket-jeg-v1'
_CFG_PREFIX = 'drive-onboard-configs/'
_STATUS_PREFIX = 'drive-onboard-jobs/'

# Per-folder budgets (orchestrator-enforced, not the prompt — 3.4).
_MAX_TOOL_CALLS = 30
_MAX_DOCS_READ = 12
_MAX_CHARS_TOTAL = 120_000
_WALL_CLOCK_S = 360
_SESSION_BUDGET_CENTS = '100'          # $1 hard cap per folder
_MAX_FOLDERS = 10
_RETRY_AFTER_DAYS = 30                  # don't re-investigate a folder every night
_STALE_SESSION_H = 24


def _get_secret(name: str) -> str:
    val = os.environ.get(name.upper().replace('-', '_'), '')
    if not val:
        raise RuntimeError(f'{name} is not set')
    return val


def _gcs() -> storage.Client:
    return storage.Client()


def _anthropic():
    import anthropic
    c = anthropic.Anthropic(api_key=_get_secret('anthropic-api-key'))
    if not hasattr(c.beta, 'agents'):
        raise SystemExit(f'anthropic {anthropic.__version__} lacks Managed Agents (need >= 1.15)')
    return c


def _write_status(gcs, run_id: str, payload: dict) -> None:
    gcs.bucket(_BUCKET).blob(f'{_STATUS_PREFIX}{run_id}/status.json') \
        .upload_from_string(json.dumps(payload, default=str),
                            content_type='application/json')


def _delete_session(anth, session_id: str) -> None:
    """A running session can't be deleted — interrupt, wait, then delete."""
    try:
        anth.beta.sessions.events.send(session_id=session_id,
                                       events=[{'type': 'user.interrupt'}])
    except Exception:
        pass
    for _ in range(8):
        try:
            if anth.beta.sessions.retrieve(session_id=session_id).status != 'running':
                break
        except Exception:
            break
        time.sleep(3)
    try:
        anth.beta.sessions.delete(session_id=session_id)
    except Exception as e:
        print(f'  WARN session delete failed: {e}', flush=True)


def _sweep_stale_sessions(anth) -> int:
    """Delete any onboarding session older than 24 h (a previous run may have
    died before cleaning up — 0.1(a) requires none linger)."""
    n = 0
    cutoff = datetime.now(timezone.utc) - timedelta(hours=_STALE_SESSION_H)
    try:
        for s in anth.beta.sessions.list():
            md = getattr(s, 'metadata', None) or {}
            if not md.get('onboard_folder'):
                continue
            try:
                created = datetime.fromisoformat(str(s.created_at).replace('Z', '+00:00'))
            except Exception:
                continue
            if created < cutoff:
                _delete_session(anth, s.id)
                n += 1
    except Exception as e:
        print(f'  WARN stale-session sweep failed: {e}', flush=True)
    return n


_BRIEFING = (
    'Investigate the Drive folder with id {folder_id} (named "{folder_name}"). '
    'Use the tools to determine which company it belongs to and which documents '
    'are worth keeping, then call submit_proposal with your decision.'
)


def investigate_folder(anth, gcs, svc, drive_id, agent_cfg, folder_id, folder_name,
                       known_keys_by_pool, *, hide_assignment_for=None) -> dict:
    """Run one session; return a proposal dict (never raises). `hide_assignment_for`
    (a set of company_keys) is forwarded to search_companies so the shadow eval
    can hide a folder's true client from the has_drive_assignment flag."""
    st = {'tool_calls': 0, 'docs_read': 0, 'chars': 0,
          'file_ids': set(), 'folder_file_ids': set(), 'domains': set()}

    def dispatch(name: str, inp: dict) -> dict:
        if name == 'list_folder':
            r = drive_read.list_folder(svc, drive_id, inp.get('folder_id', folder_id))
            for f in r.get('files', []):
                st['file_ids'].add(f['file_id'])
                st['folder_file_ids'].add(f['file_id'])
            return r
        if name == 'read_document':
            st['docs_read'] += 1
            fid = inp.get('file_id', '')
            st['file_ids'].add(fid)
            r = drive_read.read_document(svc, fid)
            st['chars'] += int(r.get('chars') or 0)
            return r
        if name == 'folder_domain_signals':
            r = drive_read.folder_domain_signals(svc, drive_id, inp.get('folder_id', folder_id))
            for c in r.get('candidates', []):
                st['domains'].add(c['domain'])
            return r
        if name == 'search_companies':
            return company_lookup.search_companies(
                gcs, inp.get('query', ''),
                pools=inp.get('pools') or ('clients', 'prospects'),
                hide_assignment_for=hide_assignment_for)
        if name == 'get_company':
            r = company_lookup.get_company(gcs, inp.get('company_key', ''))
            if r.get('website'):
                st['domains'].add(drive_read.bare_domain(r['website']))
            return r
        return {'error': f'unknown tool {name}'}

    def over_budget() -> bool:
        return (st['tool_calls'] >= _MAX_TOOL_CALLS
                or st['docs_read'] >= _MAX_DOCS_READ
                or st['chars'] >= _MAX_CHARS_TOTAL
                or time.monotonic() > deadline)

    base = {'folder_id': folder_id, 'folder_name': folder_name,
            'run_id': run_id, 'prompt_version': agent_cfg.get('prompt_version', '')}
    session = anth.beta.sessions.create(
        agent={'type': 'agent', 'id': agent_cfg['agent_id'],
               'version': agent_cfg['agent_version']},
        environment_id=agent_cfg['env_id'],
        metadata={'onboard_folder': folder_id},
        budget={'type': 'limit',
                'max_list_cost': {'amount': _SESSION_BUDGET_CENTS, 'currency': 'USD'}},
        initial_events=[{'type': 'user.message', 'content': [
            {'type': 'text', 'text': _BRIEFING.format(
                folder_id=folder_id, folder_name=folder_name)}]}],
    )
    deadline = time.monotonic() + _WALL_CLOCK_S
    proposal_input = None
    nudges = 0
    usage = None
    try:
        while True:
            pending = []
            terminal_id = None
            with anth.beta.sessions.events.stream(session_id=session.id) as stream:
                for ev in stream:
                    if ev.type == 'agent.custom_tool_use':
                        if ev.name == ob_TERMINAL:
                            proposal_input = ev.input
                            terminal_id = ev.id
                        else:
                            pending.append(ev)
                    elif ev.type in ('session.status_idle', 'session.status_terminated'):
                        break
            if proposal_input is not None:
                if terminal_id:
                    try:
                        anth.beta.sessions.events.send(session_id=session.id, events=[{
                            'type': 'user.custom_tool_result',
                            'custom_tool_use_id': terminal_id,
                            'content': [{'type': 'text', 'text': 'Received.'}]}])
                    except Exception:
                        pass
                break
            if not pending:
                break
            results = []
            for call in pending:
                st['tool_calls'] += 1
                if over_budget():
                    nudges += 1
                    text = ('Budget reached — call submit_proposal now with what '
                            'you have.')
                else:
                    try:
                        text = json.dumps(dispatch(call.name, call.input or {}), default=str)
                    except Exception as e:
                        text = json.dumps({'error': f'{type(e).__name__}: {e}'})
                results.append({'type': 'user.custom_tool_result',
                                'custom_tool_use_id': call.id,
                                'content': [{'type': 'text', 'text': text}]})
            anth.beta.sessions.events.send(session_id=session.id, events=results)
            if nudges >= 3 and proposal_input is None:
                break
        try:
            usage = anth.beta.sessions.retrieve(session_id=session.id).usage
        except Exception:
            usage = None
    finally:
        _delete_session(anth, session.id)

    cost = 0.0
    if usage is not None and getattr(usage, 'list_cost', None) is not None:
        try:
            cost = int(usage.list_cost.amount) / 100.0
        except Exception:
            cost = 0.0

    if proposal_input is None:
        return {**base, 'decision': 'unsure', 'confidence': 'low', 'evidence': [],
                'outcome': 'budget_exhausted' if nudges else 'error',
                'usage': {'cost_usd': cost},
                'validation_errors': ['no proposal submitted']}

    errs = ob.validate_proposal(
        proposal_input, surfaced_file_ids=st['file_ids'],
        surfaced_domains=st['domains'], folder_file_ids=st['folder_file_ids'],
        known_keys_by_pool=known_keys_by_pool)
    return {**base, **proposal_input,
            'outcome': 'error' if errs else 'proposed',
            'validation_errors': errs, 'usage': {'cost_usd': cost}}


# module-level handles set in main() so investigate_folder can read them
ob_TERMINAL = 'submit_proposal'
run_id = ''


def _select_folders(gcs, config) -> list[tuple[str, str]]:
    """[(folder_id, folder_name)], oldest-first, honouring the 30-day retry gate."""
    max_folders = int(config.get('max_folders', _MAX_FOLDERS))
    state = ob.load_state(gcs)
    cutoff = datetime.now(timezone.utc) - timedelta(days=_RETRY_AFTER_DAYS)

    def recently_done(fid: str) -> bool:
        pa = (state.get(fid) or {}).get('proposed_at')
        if not pa:
            return False
        try:
            return datetime.fromisoformat(str(pa).replace('Z', '+00:00')) > cutoff
        except Exception:
            return False

    if config.get('select') == 'ids':
        picked = [(f, '') for f in config.get('folder_ids', [])]
    else:
        df = attention.load(gcs, kind='drive_unmatched_folder', status='open')
        rows = df.sort_values('created_at').to_dict('records') if not df.empty else []
        picked = [(r['subject_key'], r.get('subject_label', ''))
                  for r in rows if not recently_done(r['subject_key'])]
    return picked[:max_folders]


def main(config_blob_path: str) -> None:
    global run_id
    gcs = _gcs()
    config = json.loads(gcs.bucket(_BUCKET).blob(config_blob_path).download_as_text())
    run_id = config['run_id']
    if run_id == 'daily':
        run_id = datetime.now(timezone.utc).strftime('drive_onboard_%Y-%m-%d_%H-%M-%S')
    drive_id = config['drive_id']
    dry_run = bool(config.get('dry_run', False))

    anth = _anthropic()
    agent_cfg = ob.load_agent_config(gcs)
    if not agent_cfg.get('agent_id') or not agent_cfg.get('env_id'):
        raise RuntimeError('onboarding agent not synced — run '
                           '`python -m agents.drive_onboard.agent sync` first')

    swept = _sweep_stale_sessions(anth)
    svc = drive_client.build_drive_service()
    known_keys_by_pool = {
        'clients': set(company_lookup._companies(gcs, ['clients'], _BUCKET)),
        'prospects': set(company_lookup._companies(gcs, ['prospects'], _BUCKET)),
    }

    folders = _select_folders(gcs, config)
    print(f'[{run_id}] {len(folders)} folder(s), swept {swept} stale sessions', flush=True)

    state = ob.load_state(gcs)
    results = []
    cost_total = 0.0
    for i, (fid, fname) in enumerate(folders, 1):
        try:
            prop = investigate_folder(anth, gcs, svc, drive_id, agent_cfg, fid, fname,
                                      known_keys_by_pool)
        except Exception as e:
            traceback.print_exc()
            prop = {'folder_id': fid, 'folder_name': fname, 'run_id': run_id,
                    'decision': 'unsure', 'confidence': 'low', 'outcome': 'error',
                    'validation_errors': [f'{type(e).__name__}: {e}'],
                    'usage': {'cost_usd': 0.0}}
        cost_total += float(prop.get('usage', {}).get('cost_usd') or 0.0)
        print(f'  [{i}/{len(folders)}] {fid}: {prop.get("decision")} '
              f'({prop.get("confidence")}) outcome={prop.get("outcome")}', flush=True)

        if not dry_run:
            prop['review_status'] = 'open'
            ob.save_proposal(gcs, prop)
            attention.enqueue(gcs, [{
                'kind': 'drive_onboarding_proposal', 'subject_key': fid,
                'subject_label': fname or prop.get('folder_name', ''),
                'source_job': 'drive-onboard-job', 'run_id': run_id,
                'detail': {'decision': prop.get('decision'),
                           'confidence': prop.get('confidence'),
                           'outcome': prop.get('outcome')}}])
            state[fid] = {'proposed_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                          'attempts': (state.get(fid, {}).get('attempts', 0) + 1)}
            ob.save_state(gcs, state)

        results.append({'folder_id': fid, 'decision': prop.get('decision'),
                        'confidence': prop.get('confidence'),
                        'outcome': prop.get('outcome'),
                        'cost_usd': prop.get('usage', {}).get('cost_usd', 0.0)})
        _write_status(gcs, run_id, {
            'run_id': run_id, 'state': 'running', 'dry_run': dry_run,
            'folders_total': len(folders), 'folders_done': i,
            'proposals': results, 'cost_usd_total': round(cost_total, 4),
            'stale_swept': swept, 'error': None})

    _write_status(gcs, run_id, {
        'run_id': run_id, 'state': 'complete', 'dry_run': dry_run,
        'folders_total': len(folders), 'folders_done': len(folders),
        'proposals': results, 'cost_usd_total': round(cost_total, 4),
        'stale_swept': swept, 'stopped_early': None, 'error': None})
    print(f'[{run_id}] complete — {len(results)} proposals, ${cost_total:.4f}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python jobs/onboard_job.py <config_blob_path>', file=sys.stderr)
        sys.exit(1)
    try:
        main(sys.argv[1])
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        try:
            rid = sys.argv[1].split('/')[-1].replace('.json', '')
            _write_status(_gcs(), rid, {'run_id': rid, 'state': 'error', 'error': tb})
        except Exception:
            pass
        sys.exit(1)
