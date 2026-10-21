"""
Phase 3.9 — shadow evaluation of the Drive onboarding agent.

Free ground truth: folders already assigned to a client (`match_type` manual or
auto). Run the agent on N of them with that folder's true client HIDDEN from the
`has_drive_assignment` flag, then compare the agent's proposed company_key to the
truth. Writes proposals to drive-onboard/shadow/{run_id}/ (never the live store).

Go-live bar (PLAN §3.9, John to adjust): ZERO wrong-company answers at `high`
confidence, and <= 1 wrong-company answer overall in 30.

    python -m scripts.eval.onboard_shadow [N]     # default 30

Spends real money (one short Managed Agents session per folder). Requires the
agent to be synced and ANTHROPIC_API_KEY + Drive ADC.
"""

import json
import statistics
import sys
import time
from datetime import datetime, timezone

from google.cloud import storage

import jobs.onboard_job as oj
from src.modules import drive_client, onboarding as ob
from src.modules.tools import company_lookup


def _assigned_folders(gcs, bucket):
    blob = gcs.bucket(bucket).blob('drive-sync-configs/assignments.json')
    doc = json.loads(blob.download_as_bytes()).get('assignments', {})
    return [(fid, a['client_key'], a.get('folder_name', ''))
            for fid, a in doc.items()
            if a.get('client_key') and a.get('match_type') in ('manual', 'auto')]


def main(n: int = 30) -> None:
    bucket = oj._BUCKET
    gcs = storage.Client()
    anth = oj._anthropic()
    agent_cfg = ob.load_agent_config(gcs)
    if not agent_cfg.get('agent_id'):
        raise SystemExit('agent not synced — run `python -m agents.drive_onboard.agent sync`')
    drive_id = None
    # drive_id lives in the drive-sync daily config / assignments doc setup; read it
    # from the sync config if present, else require it as argv[2].
    try:
        cfg = json.loads(gcs.bucket(bucket).blob(
            'drive-sync-configs/daily_schedule.json').download_as_bytes())
        drive_id = cfg.get('drive_id')
    except Exception:
        pass
    if not drive_id and len(sys.argv) > 2:
        drive_id = sys.argv[2]
    if not drive_id:
        raise SystemExit('drive_id unknown — pass it as the 2nd arg')

    svc = drive_client.build_drive_service()
    known = {'clients': set(company_lookup._companies(gcs, ['clients'], bucket)),
             'prospects': set(company_lookup._companies(gcs, ['prospects'], bucket))}

    truth = _assigned_folders(gcs, bucket)[:n]
    run_id = datetime.now(timezone.utc).strftime('shadow_%Y-%m-%d_%H-%M-%S')
    oj.run_id = run_id
    shadow_prefix = f'{ob.SHADOW_PREFIX}{run_id}/'
    print(f'[{run_id}] shadow eval on {len(truth)} assigned folders', flush=True)

    rows, times = [], []
    for i, (fid, true_key, fname) in enumerate(truth, 1):
        t0 = time.monotonic()
        prop = oj.investigate_folder(anth, gcs, svc, drive_id, agent_cfg, fid, fname,
                                     known, hide_assignment_for={true_key})
        dt = time.monotonic() - t0
        times.append(dt)
        proposed = prop.get('company_key', '')
        correct = (prop.get('decision') == 'existing_client' and proposed == true_key)
        wrong_company = (prop.get('decision') in ('existing_client', 'existing_prospect')
                         and proposed and proposed != true_key)
        rec = {'folder_id': fid, 'true_key': true_key,
               'decision': prop.get('decision'), 'proposed': proposed,
               'confidence': prop.get('confidence'), 'outcome': prop.get('outcome'),
               'correct': correct, 'wrong_company': wrong_company,
               'cost_usd': prop.get('usage', {}).get('cost_usd', 0.0), 'secs': round(dt, 1)}
        rows.append(rec)
        gcs.bucket(bucket).blob(f'{shadow_prefix}{fid}.json').upload_from_string(
            json.dumps({**prop, 'truth': true_key, **rec}, default=str),
            content_type='application/json')
        print(f'  [{i}/{len(truth)}] {prop.get("decision")}/{prop.get("confidence")} '
              f'{"OK" if correct else ("WRONG" if wrong_company else "-")} '
              f'${rec["cost_usd"]:.3f} {dt:.0f}s', flush=True)

    n_t = len(rows)
    wrong = [r for r in rows if r['wrong_company']]
    wrong_high = [r for r in wrong if r['confidence'] == 'high']
    report = {
        'run_id': run_id, 'n': n_t,
        'decision_accuracy': round(sum(r['correct'] for r in rows) / n_t, 3) if n_t else 0,
        'wrong_company': len(wrong), 'wrong_company_high_conf': len(wrong_high),
        'unsure_rate': round(sum(r['decision'] == 'unsure' for r in rows) / n_t, 3) if n_t else 0,
        'cost_usd_total': round(sum(r['cost_usd'] for r in rows), 4),
        'cost_per_folder': round(sum(r['cost_usd'] for r in rows) / n_t, 4) if n_t else 0,
        'p50_secs': round(statistics.median(times), 1) if times else 0,
        'p90_secs': round(sorted(times)[int(0.9 * (len(times) - 1))], 1) if times else 0,
        'go_live_bar_met': len(wrong_high) == 0 and len(wrong) <= 1,
        'rows': rows,
    }
    gcs.bucket(bucket).blob(f'{shadow_prefix}report.json').upload_from_string(
        json.dumps(report, indent=2, default=str), content_type='application/json')
    print('\n===== SHADOW REPORT =====', flush=True)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
    print(f'\nGO-LIVE BAR {"MET" if report["go_live_bar_met"] else "NOT MET"} '
          f'(0 wrong@high and <=1 wrong overall). Full report: '
          f'gs://{bucket}/{shadow_prefix}report.json')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 30)
