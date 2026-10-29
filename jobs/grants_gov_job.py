"""
grants_gov_job.py — Cloud Run Job: scheduled Grants.gov ingest.

Grants.gov Fetch was manual-only, so the NIH/NSF/USDA SBIR-STTR omnibus parent
announcements a client needs were never pulled on a schedule. This job runs the
same query/screen/summarize/embed core as the view (src/modules/grants_gov.py)
against a saved config, so a daily Cloud Scheduler run keeps those programs in the
store.

Pipeline:
  1. Load config from grants-gov-configs/{run_id}.json
  2. gg.search_grants  — paged search2 query + description fetch
  3. gg.run_screening  — Claude Haiku relevance screen (keeps `import: true`)
  4. dedup vs the existing GRANTS-GOV store (notice id / title)
  5. gg.embed_and_save — summarize, embed, newsletter-tag, write a parquet
  6. Write grants-gov-jobs/{run_id}/status.json

Config schema (grants-gov-configs/{run_id}.json):
  {
    "run_id":          "grants_gov_2026-10-27_06-00-00",   // or "daily"
    "keyword":         "SBIR STTR",
    "statuses":        ["posted", "forecasted"],
    "instruments":     ["G", "CA", "O"],                   // include "O" for NIH parents
    "agencies":        ["HHS", "NSF", "USDA", "DOD"],      // [] = all agencies
    "lookback_days":   7,                                   // OR date_from/date_to (YYYY-MM-DD)
    "exclude_expired": true,
    "max_results":     0,                                   // 0 = no cap
    "custom_cols":     {}
  }

Usage:
  python jobs/grants_gov_job.py grants-gov-configs/<run_id>.json
"""

import json
import os
import sys
import traceback
from datetime import date, datetime, timedelta

from google.cloud import storage

from src.modules import grants_gov as gg

_BUCKET        = gg.BUCKET
_STATUS_PREFIX = 'grants-gov-jobs/'


def _get_secret(name: str) -> str:
    env_var = name.upper().replace('-', '_')
    val = os.environ.get(env_var, '')
    if not val:
        raise RuntimeError(f'Environment variable {env_var} is not set.')
    return val


def _gcs() -> storage.Client:
    return storage.Client()


def _write_status(client: storage.Client, run_id: str, payload: dict) -> None:
    client.bucket(_BUCKET).blob(f'{_STATUS_PREFIX}{run_id}/status.json').upload_from_string(
        json.dumps(payload, default=str), content_type='application/json')


def _parse_iso(s) -> date | None:
    try:
        return datetime.strptime(str(s).strip(), '%Y-%m-%d').date()
    except (ValueError, TypeError):
        return None


def _run_reverify(gcs: storage.Client, run_id: str, config: dict) -> None:
    """Liveness re-verification mode (no ingest): re-check stored GRANTS-GOV topics
    against the live API and update verify_status. `dry_run` reports without writing;
    `rolling_only` restricts to rolling topics (default False = all open topics)."""
    dry_run      = bool(config.get('dry_run', False))
    rolling_only = bool(config.get('rolling_only', False))
    status = {'run_id': run_id, 'mode': 'reverify', 'state': 'running',
              'dry_run': dry_run, 'rolling_only': rolling_only,
              'checked': 0, 'refreshed': 0, 'expired': 0, 'files_written': 0,
              'error': None}
    _write_status(gcs, run_id, status)
    try:
        res = gg.reverify_grants(gcs, rolling_only=rolling_only, dry_run=dry_run)
        status.update(res)
        status['state'] = 'complete'
        _write_status(gcs, run_id, status)
        print(f'reverify complete: {res}', flush=True)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        status['state'] = 'error'
        status['error'] = tb
        _write_status(gcs, run_id, status)
        raise


def main(config_blob_path: str) -> None:
    gcs = _gcs()
    print(f'Loading config from {config_blob_path}', flush=True)
    config = json.loads(gcs.bucket(_BUCKET).blob(config_blob_path).download_as_text())

    run_id = config['run_id']
    if run_id == 'daily':
        run_id = f"grants_gov_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Liveness re-verification mode is a separate, key-free path.
    if str(config.get('mode', 'ingest') or 'ingest').strip() == 'reverify':
        _run_reverify(gcs, run_id, config)
        return

    anth_key   = _get_secret('anthropic-api-key')
    openai_key = _get_secret('openai-api-key')

    keyword     = str(config.get('keyword', '') or '').strip()
    statuses    = config.get('statuses', ['posted', 'forecasted'])
    instruments = config.get('instruments', ['G', 'CA', 'O'])
    agencies    = config.get('agencies', [])
    exclude_exp = bool(config.get('exclude_expired', True))
    max_results = int(config.get('max_results', 0) or 0)
    custom_cols = config.get('custom_cols', {}) or None

    # lookback_days (scheduled runs) overrides any explicit date window.
    date_from = _parse_iso(config.get('date_from'))
    date_to   = _parse_iso(config.get('date_to'))
    if config.get('lookback_days'):
        date_from = date.today() - timedelta(days=int(config['lookback_days']))
        date_to   = date.today()

    status: dict = {
        'run_id': run_id, 'state': 'running',
        'keyword': keyword, 'agencies': agencies, 'instruments': instruments,
        'rows_fetched': 0, 'rows_passed_screening': 0, 'rows_after_dedup': 0,
        'rows_saved': 0, 'hit_count': 0, 'gcs_path': None, 'error': None,
    }
    _write_status(gcs, run_id, status)

    try:
        # ── fetch ──
        print(f'Querying Grants.gov (keyword={keyword!r}, agencies={agencies})…', flush=True)
        df, hit_count = gg.search_grants(
            keyword=keyword, statuses=statuses, instruments=instruments,
            agencies=agencies, date_from=date_from, date_to=date_to,
            exclude_expired=exclude_exp, max_results=max_results)
        status['rows_fetched'] = len(df)
        status['hit_count']    = hit_count
        _write_status(gcs, run_id, status)
        if df.empty:
            status['state'] = 'complete'
            _write_status(gcs, run_id, status)
            print('No rows returned.', flush=True)
            return

        # ── screen ──
        print(f'Screening {len(df)} rows…', flush=True)
        screened = gg.run_screening(df, anth_key)
        passed = screened[screened['_import'] == True].reset_index(drop=True)  # noqa: E712
        status['rows_passed_screening'] = len(passed)
        _write_status(gcs, run_id, status)
        if passed.empty:
            status['state'] = 'complete'
            _write_status(gcs, run_id, status)
            print('No rows passed screening.', flush=True)
            return

        # ── dedup vs the existing store ──
        existing_ids, existing_titles = gg.load_existing_keys(gcs)
        keep = ~(passed['notice_id'].astype(str).str.strip().isin(existing_ids)
                 | passed['title'].astype(str).str.lower().str.strip().isin(existing_titles))
        new_rows = passed[keep].reset_index(drop=True)
        status['rows_after_dedup'] = len(new_rows)
        _write_status(gcs, run_id, status)
        if new_rows.empty:
            status['state'] = 'complete'
            _write_status(gcs, run_id, status)
            print('All screened rows already in the store.', flush=True)
            return

        # ── embed + save ──
        print(f'Embedding and saving {len(new_rows)} new topics…', flush=True)
        gcs_path = gg.embed_and_save(new_rows, openai_key, anth_key, gcs,
                                     extra_cols=custom_cols)
        status['rows_saved'] = len(new_rows)
        status['gcs_path']   = gcs_path
        status['state']      = 'complete'
        _write_status(gcs, run_id, status)
        print(f'Saved {gcs_path} — {len(new_rows)} topics.', flush=True)

    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        status['state'] = 'error'
        status['error'] = tb
        _write_status(gcs, run_id, status)
        raise


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python jobs/grants_gov_job.py grants-gov-configs/<run_id>.json',
              file=sys.stderr)
        sys.exit(1)
    try:
        main(sys.argv[1])
    except Exception:
        # main() already wrote an error status if it got far enough.
        sys.exit(1)
