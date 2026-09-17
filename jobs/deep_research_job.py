"""
deep_research_job.py — Cloud Run Job for the Funding Source Watch (Stage 10).

Walks the master list of approved funding-source websites on its cadence, has
Claude drive a headless browser over each one, and files whatever open
opportunities it finds into the existing grant-topic store. For each selected
site:

  1. Load the per-source "seen index" and hand its titles to the agent, so the
     agent skips what is already stored instead of paying to re-extract it
  2. Run the browser agent (src/modules/browser_agent.py) with the site's own
     navigation instructions and a hard per-site budget
  3. Drop opportunities already in the seen index; safety-dedup the survivors
     against the titles already in the destination agency folder
  4. Embed each survivor's description (text-embedding-ada-002) and append it to
     the buffer for its destination folder
  5. Update the registry row — last_checked, outcome, has_api, requires_login,
     consecutive_failures — and the seen index

Buffers are flushed to parquet every few sites rather than once at the end, so
a timeout or a crash keeps the extraction already paid for. Everything the
agent found for a site is written before that site is marked checked.

Sites are browsed concurrently (each holds its own browser context); embedding
and GCS writes happen on a worker thread so a flush never stalls the browsers
that are still running.

Usage:
    python jobs/deep_research_job.py deep-research-configs/<run_id>.json

Environment variables (injected by Cloud Run from Secret Manager):
    ANTHROPIC_API_KEY, OPENAI_API_KEY

Config schema:
{
  "run_id":         "deep_research_2026-09-16_06-00-00",   # or "daily" sentinel
  "select":         "due",            # "due" | "all" | "ids"
  "source_ids":     [],               # required when select == "ids"
  "model":          "claude-sonnet-4-6",
  "concurrency":    3,
  "max_tool_calls": 25,
  "site_timeout_s": 240,
  "max_pages":      null,             # override the per-site registry value
  "task_timeout_s": 14400,
  "dry_run":        false
}
"""

import asyncio
import io
import json
import os
import secrets
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import tiktoken
from anthropic import AsyncAnthropic
from google.cloud import storage
from openai import OpenAI

from src.modules import source_registry as sr
from src.modules.browser_agent import (
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_SITE_TIMEOUT_S,
    MODEL as DEFAULT_MODEL,
    launch_browser,
    research_site,
)

_BUCKET         = sr.BUCKET
_TOPICS_PREFIX  = 'data/all-topics/processed/'
_STATUS_PREFIX  = sr.STATUS_PREFIX

_EMBED_WORKERS  = 8
_TOKEN_LIMIT    = 7_500
_EMBED_MODEL    = 'text-embedding-ada-002'

# Graceful time budget: stop handing out sites before Cloud Run's hard task
# timeout kills the container, which would lose both the un-flushed buffers and
# the status file, leaving the view polling forever.
_DEFAULT_TASK_TIMEOUT_S = 14_400
_MIN_TASK_TIMEOUT_S     = 900
_MAX_TASK_TIMEOUT_S     = 86_400
_DEADLINE_MARGIN_S      = 600

_CHECKPOINT_EVERY  = 10
_MIN_CONCURRENCY   = 1
_MAX_CONCURRENCY   = 6
_MAX_DRY_PREVIEWS  = 4
_MAX_RESULT_ROWS   = 400


# -- Infrastructure helpers -------------------------------------------------

def _get_secret(name: str) -> str:
    env_var = name.upper().replace('-', '_')
    val = os.environ.get(env_var, '')
    if not val:
        raise RuntimeError(f'Environment variable {env_var} is not set.')
    return val


def _gcs() -> storage.Client:
    return storage.Client()          # ADC — the job's attached service account


def _write_status(client: storage.Client, run_id: str, payload: dict) -> None:
    client.bucket(_BUCKET).blob(f'{_STATUS_PREFIX}{run_id}/status.json').upload_from_string(
        json.dumps(payload), content_type='application/json'
    )


# -- Embedding (raw openai + tiktoken, as every other job does) -------------

def _get_embedding(text: str, oai_client, encoding):
    if not text or not text.strip():
        return None
    words = text.split()
    while len(encoding.encode(text)) > _TOKEN_LIMIT:
        words = words[:-5]
        if not words:
            return None
        text = ' '.join(words)
    return oai_client.embeddings.create(input=[text], model=_EMBED_MODEL).data[0].embedding


def _embed_all(texts: list, openai_key: str) -> list:
    oai_client = OpenAI(api_key=openai_key)
    encoding   = tiktoken.get_encoding('cl100k_base')
    out = [None] * len(texts)

    def _one(i_text):
        i, text = i_text
        try:
            return i, _get_embedding(text, oai_client, encoding)
        except Exception as e:                                # noqa: BLE001
            print(f'  embedding failed for row {i}: {e}', flush=True)
            return i, None

    with ThreadPoolExecutor(max_workers=_EMBED_WORKERS) as pool:
        for i, vec in pool.map(_one, list(enumerate(texts))):
            out[i] = vec
    return out


# -- Destination store ------------------------------------------------------

def _existing_titles(client: storage.Client, broad_agency: str) -> set:
    """Titles already stored in one destination folder.

    A cheap safety net behind the seen index: it catches an opportunity that
    reached the store through another route (Topic Importer, Grants.gov) and
    would otherwise be duplicated.
    """
    titles = set()
    prefix = f'{_TOPICS_PREFIX}{broad_agency}/'
    for blob in client.list_blobs(_BUCKET, prefix=prefix):
        if not blob.name.endswith('.parquet'):
            continue
        try:
            df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()), columns=['title'])
            titles.update(df['title'].dropna().astype(str).str.lower().str.strip())
        except Exception:
            continue
    return titles


def _build_topic_frame(rows: list, embeddings: list, run_id: str) -> pd.DataFrame:
    """Canonical grant-topic schema plus this pipeline's provenance columns.

    `broad_agency` is deliberately NOT written: every reader injects it from the
    folder name at load time and would overwrite anything stored here.
    """
    today = datetime.today().strftime('%Y-%m-%d')
    out = pd.DataFrame()
    out['topic_number']   = [r.get('topic_number', '') for r in rows]
    out['title']          = [r.get('title', '') for r in rows]
    out['agency']         = [r.get('agency', '') for r in rows]
    out['source']         = [r.get('source', '') for r in rows]
    out['open_date']      = [r.get('open_date', '') for r in rows]
    out['close_date']     = [r.get('close_date', '') for r in rows]
    # `due_date` is what matching_job exports and Bulk Matching filters on;
    # `close_date` is what Grants.gov writes. Both stores exist, so write both.
    out['due_date']       = [r.get('close_date', '') for r in rows]
    out['funding_amount'] = [r.get('funding_amount', '') for r in rows]
    out['scraped_at']     = today
    out['grant_summary']  = [r.get('description', '') for r in rows]
    out['description']    = [r.get('description', '') for r in rows]
    out['embeddings']     = embeddings
    out['record_kind']    = 'solicitation'
    out['source_site']    = [r.get('source_site', '') for r in rows]
    out['source_id']      = [r.get('source_id', '') for r in rows]
    out['source_name']    = [r.get('source_name', '') for r in rows]
    out['first_seen']     = today
    out['deep_research_run_id'] = run_id
    return out


def _save_topics(client: storage.Client, broad_agency: str, df: pd.DataFrame) -> str:
    today = datetime.today().strftime('%Y-%m-%d')
    path  = f'{_TOPICS_PREFIX}{broad_agency}/deep_research_{today}_{secrets.token_hex(3)}.parquet'
    buf   = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    client.bucket(_BUCKET).blob(path).upload_from_file(
        buf, content_type='application/octet-stream'
    )
    return path


# -- Site selection ---------------------------------------------------------

def _select_sites(registry: pd.DataFrame, config: dict) -> pd.DataFrame:
    mode = str(config.get('select') or 'due').lower()
    if registry is None or registry.empty:
        return registry

    if mode == 'ids':
        wanted = set(config.get('source_ids') or [])
        picked = registry[registry['source_id'].isin(wanted)]
    elif mode == 'all':
        picked = registry[registry['enabled']]
    else:
        picked = sr.due_sources(registry)

    # An explicit id list is an operator override and runs even if the site is
    # paused; `due` and `all` respect enabled/cadence.
    return picked.reset_index(drop=True)


# -- The run ----------------------------------------------------------------

async def _browse_all(anth, sites: list, seen_by_id: dict, settings: dict,
                      deadline: float, on_result) -> str:
    """Browse every site, calling `on_result(site, result)` as each finishes."""
    # Imported here rather than at module scope so the job module can be
    # imported (and its pure helpers tested) in an environment without Chromium.
    from playwright.async_api import async_playwright

    stopped_early = None
    sem = asyncio.Semaphore(settings['concurrency'])

    async with async_playwright() as pw:
        browser = await launch_browser(pw)
        try:
            async def _one(site):
                nonlocal stopped_early
                async with sem:
                    if time.monotonic() > deadline:
                        stopped_early = stopped_early or 'timeout'
                        return site, None
                    known = sr.known_titles(seen_by_id.get(site['source_id'], {}))
                    return site, await research_site(
                        anth, browser, site,
                        known=known,
                        model=settings['model'],
                        max_tool_calls=settings['max_tool_calls'],
                        site_timeout_s=settings['site_timeout_s'],
                    )

            tasks = [asyncio.create_task(_one(s)) for s in sites]
            for coro in asyncio.as_completed(tasks):
                site, result = await coro
                await on_result(site, result)
        finally:
            try:
                await browser.close()
            except Exception:
                pass

    return stopped_early


def main(config_blob_path: str) -> None:
    gcs = _gcs()
    print(f'Loading config from {config_blob_path}', flush=True)
    config = json.loads(gcs.bucket(_BUCKET).blob(config_blob_path).download_as_text())

    run_id = config['run_id']
    # "daily" is the Cloud Scheduler sentinel — mint a real ID at runtime so no
    # day's status file overwrites another's.
    if run_id == 'daily':
        run_id = f"deep_research_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}"
    print(f'Run ID: {run_id}', flush=True)

    dry_run  = bool(config.get('dry_run'))
    settings = {
        'model':          str(config.get('model') or DEFAULT_MODEL),
        'concurrency':    max(_MIN_CONCURRENCY,
                              min(int(config.get('concurrency') or 3), _MAX_CONCURRENCY)),
        'max_tool_calls': int(config.get('max_tool_calls') or DEFAULT_MAX_TOOL_CALLS),
        'site_timeout_s': int(config.get('site_timeout_s') or DEFAULT_SITE_TIMEOUT_S),
    }
    task_timeout_s = max(
        _MIN_TASK_TIMEOUT_S,
        min(int(config.get('task_timeout_s') or _DEFAULT_TASK_TIMEOUT_S), _MAX_TASK_TIMEOUT_S),
    )
    deadline = time.monotonic() + (task_timeout_s - _DEADLINE_MARGIN_S)

    registry = sr.load_sources(gcs)
    if registry.empty:
        raise ValueError(
            'the source registry is empty — import the master list in the '
            'Funding Sources view before running this job'
        )
    picked = _select_sites(registry, config)
    if picked.empty:
        print('No sites due. Nothing to do.', flush=True)
        _write_status(gcs, run_id, {
            'run_id': run_id, 'state': 'complete', 'dry_run': dry_run,
            'select': config.get('select', 'due'), 'model': settings['model'],
            'sites_total': 0, 'sites_done': 0, 'sites_ok': 0, 'sites_errored': 0,
            'sites_deferred': 0, 'opportunities_found': 0, 'opportunities_new': 0,
            'opportunities_saved': 0, 'gcs_paths': [], 'api_sites': [],
            'login_walled': [], 'results': [], 'cost_usd': 0.0,
            'stopped_early': None, 'note': 'no sites were due', 'error': None,
        })
        return

    page_override = config.get('max_pages')
    sites = []
    for _, row in picked.iterrows():
        site = row.to_dict()
        if page_override:
            site['max_pages'] = int(page_override)
        sites.append(site)
    total = len(sites)
    print(f'{total} site(s) selected; concurrency={settings["concurrency"]}', flush=True)

    anth_key = _get_secret('anthropic-api-key')
    oai_key  = _get_secret('openai-api-key')
    anth     = AsyncAnthropic(api_key=anth_key)

    # Pre-load every seen index up front: the agent needs the known titles
    # before it browses, and 200-odd small JSON reads are far faster in parallel.
    print('Loading seen indexes…', flush=True)
    seen_by_id = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        for sid, index in zip(
            [s['source_id'] for s in sites],
            pool.map(lambda sid: sr.load_seen(gcs, sid), [s['source_id'] for s in sites]),
        ):
            seen_by_id[sid] = index

    # -- accumulators -------------------------------------------------------
    buffers: dict = {}            # broad_agency -> list of pending topic rows
    registry_updates: dict = {}   # source_id   -> row dict to upsert
    results: list = []
    gcs_paths: list = []
    api_sites: list = []
    login_walled: list = []
    dry_previews: list = []
    title_cache: dict = {}        # broad_agency -> set of existing titles
    counts = {'done': 0, 'ok': 0, 'errored': 0, 'deferred': 0,
              'found': 0, 'new': 0, 'saved': 0}
    cost_total = {'usd': 0.0}
    stopped_early = {'reason': None}

    def _status(state: str) -> dict:
        return {
            'run_id':              run_id,
            'state':               state,
            'dry_run':             dry_run,
            'select':              config.get('select', 'due'),
            'model':               settings['model'],
            'sites_total':         total,
            'sites_done':          counts['done'],
            'sites_ok':            counts['ok'],
            'sites_errored':       counts['errored'],
            'sites_deferred':      counts['deferred'],
            'opportunities_found': counts['found'],
            'opportunities_new':   counts['new'],
            'opportunities_saved': counts['saved'],
            'gcs_paths':           gcs_paths,
            'api_sites':           api_sites,
            'login_walled':        login_walled,
            'results':             results[:_MAX_RESULT_ROWS],
            'dry_run_preview':     dry_previews,
            'cost_usd':            round(cost_total['usd'], 4),
            'stopped_early':       stopped_early['reason'],
            'error':               None,
        }

    def _flush() -> None:
        """Write buffered opportunities, seen indexes and registry updates."""
        if dry_run:
            buffers.clear()
            registry_updates.clear()
            return

        for broad_agency, rows in list(buffers.items()):
            if not rows:
                continue
            print(f'Embedding {len(rows)} new opportunit(ies) for {broad_agency}…', flush=True)
            vectors = _embed_all([r.get('description', '') for r in rows], oai_key)
            frame   = _build_topic_frame(rows, vectors, run_id)
            path    = _save_topics(gcs, broad_agency, frame)
            gcs_paths.append(path)
            counts['saved'] += len(rows)
            print(f'  saved {path}', flush=True)
        buffers.clear()

        for sid in list(registry_updates):
            try:
                sr.save_seen(gcs, sid, seen_by_id.get(sid, {}))
            except Exception as e:                            # noqa: BLE001
                print(f'  seen index write failed for {sid}: {e}', flush=True)
        if registry_updates:
            sr.upsert_sources(gcs, list(registry_updates.values()))
            registry_updates.clear()

    def _handle(site: dict, result) -> None:
        """Post-process one finished site. Runs on a worker thread."""
        sid   = site['source_id']
        name  = site.get('name') or site.get('url')
        today = datetime.today().strftime('%Y-%m-%d')
        row   = dict(site)
        row['last_run_id'] = run_id

        if result is None:
            counts['deferred'] += 1
            counts['done']     += 1
            results.append({'source_id': sid, 'name': name,
                            'status': sr.STATUS_DEFERRED, 'found': 0, 'new': 0,
                            'note': 'time budget spent — re-run to continue'})
            return

        cost_total['usd'] += float(result.get('cost_usd') or 0.0)
        counts['found']   += len(result.get('opportunities') or [])

        row['has_api']        = bool(result.get('has_api'))
        row['api_note']       = str(result.get('api_evidence') or '')[:500]
        row['requires_login'] = bool(result.get('requires_login'))

        if result.get('has_api'):
            api_sites.append({'source_id': sid, 'name': name, 'url': site.get('url'),
                              'evidence': row['api_note']})
        if result.get('requires_login'):
            login_walled.append({'source_id': sid, 'name': name, 'url': site.get('url')})

        if not result.get('ok'):
            counts['errored'] += 1
            counts['done']    += 1
            failures = int(site.get('consecutive_failures') or 0) + 1
            row['consecutive_failures'] = failures
            row['last_status'] = (sr.STATUS_LOGIN if result.get('requires_login')
                                  else sr.STATUS_ERROR)
            row['last_error']  = str(result.get('error') or 'agent did not report')[:500]
            # last_checked is deliberately advanced even on failure, so a broken
            # site does not monopolise every subsequent run.
            row['last_checked'] = today
            if failures >= sr.MAX_CONSECUTIVE_FAILURES and row.get('cadence') != 'paused':
                row['cadence'] = 'paused'
                row['notes'] = (str(row.get('notes') or '') +
                                f' [auto-paused after {failures} consecutive failures '
                                f'on {today}]').strip()
                print(f'  {name}: auto-paused after {failures} failures', flush=True)
            registry_updates[sid] = row
            results.append({'source_id': sid, 'name': name, 'status': row['last_status'],
                            'found': 0, 'new': 0, 'note': row['last_error']})
            return

        # -- dedup ----------------------------------------------------------
        index        = seen_by_id.setdefault(sid, {})
        broad_agency = str(site.get('broad_agency') or sr.DEFAULT_BROAD_AGENCY).strip() \
            or sr.DEFAULT_BROAD_AGENCY
        if broad_agency not in title_cache:
            title_cache[broad_agency] = _existing_titles(gcs, broad_agency)
        known_in_store = title_cache[broad_agency]

        fresh = []
        for opp in result.get('opportunities') or []:
            key = sr.seen_key(sid, opp.get('url', ''), opp.get('title', ''))
            if key in index:
                continue
            if opp.get('title', '').lower().strip() in known_in_store:
                index[key] = {'title': opp.get('title', ''), 'first_seen': today}
                continue
            index[key] = {'title': opp.get('title', ''), 'first_seen': today}
            known_in_store.add(opp.get('title', '').lower().strip())
            fresh.append({
                **opp,
                'agency':      site.get('sub_agency') or name,
                'source':      opp.get('url') or site.get('url', ''),
                'source_site': site.get('url', ''),
                'source_id':   sid,
                'source_name': name,
            })

        counts['new']  += len(fresh)
        counts['ok']   += 1
        counts['done'] += 1
        if fresh:
            buffers.setdefault(broad_agency, []).extend(fresh)

        row['last_checked']         = today
        row['last_status']          = sr.STATUS_OK if fresh else sr.STATUS_NOTHING_FOUND
        row['last_error']           = ''
        row['consecutive_failures'] = 0
        registry_updates[sid]       = row

        # The agent's note is where a wrong URL or a partial sweep gets
        # reported ('the real listing is at ...', 'captured 32 of 42') —
        # truncating it to a sentence threw away the actionable half.
        note = str(result.get('notes') or '')[:800]
        if result.get('stopped_early'):
            note = (note + f" [stopped early: {result['stopped_early']}]").strip()
        results.append({'source_id': sid, 'name': name, 'status': row['last_status'],
                        'found': len(result.get('opportunities') or []),
                        'new': len(fresh), 'note': note})

        if dry_run and len(dry_previews) < _MAX_DRY_PREVIEWS:
            dry_previews.append({
                'name': name, 'url': site.get('url'),
                'new': len(fresh),
                'opportunities': [
                    {'title': o.get('title'), 'close_date': o.get('close_date'),
                     'url': o.get('url'),
                     'description': (o.get('description') or '')[:1200]}
                    for o in fresh[:3]
                ],
            })

        # cache_read is the number that says whether prompt caching engaged;
        # a run where it stays at 0 is paying full price silently.
        print(f'  {name}: {len(result.get("opportunities") or [])} found, '
              f'{len(fresh)} new, {result.get("pages_visited")} page(s), '
              f'${result.get("cost_usd", 0):.3f} '
              f'[cache r={result.get("cache_read_tokens", 0)} '
              f'w={result.get("cache_write_tokens", 0)} '
              f'in={result.get("input_tokens", 0)}]', flush=True)

    # -- drive the run ------------------------------------------------------
    _write_status(gcs, run_id, _status('running'))

    loop_pool = ThreadPoolExecutor(max_workers=1)   # serialises post-processing

    async def _on_result(site, result):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(loop_pool, _handle, site, result)
        if counts['done'] % _CHECKPOINT_EVERY == 0:
            await loop.run_in_executor(loop_pool, _flush)
            _write_status(gcs, run_id, _status('running'))
            print(f'checkpoint: {counts["done"]}/{total} sites', flush=True)

    try:
        reason = asyncio.run(
            _browse_all(anth, sites, seen_by_id, settings, deadline, _on_result)
        )
        if reason:
            stopped_early['reason'] = reason
    finally:
        loop_pool.shutdown(wait=True)

    _flush()

    if counts['deferred']:
        stopped_early['reason'] = stopped_early['reason'] or 'timeout'

    final = _status('complete')
    _write_status(gcs, run_id, final)
    print(
        f'Done. {counts["ok"]} ok, {counts["errored"]} errored, '
        f'{counts["deferred"]} deferred; {counts["new"]} new opportunities, '
        f'{counts["saved"]} saved; ${cost_total["usd"]:.2f}',
        flush=True,
    )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python jobs/deep_research_job.py <config_blob_path>', file=sys.stderr)
        sys.exit(1)

    _run_id_fallback = sys.argv[1].split('/')[-1].replace('.json', '')
    try:
        main(sys.argv[1])
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        try:
            _client = _gcs()
            # Recover the real run_id where possible so the view's poll finds
            # the failure instead of timing out on a status file that never appears.
            _rid = _run_id_fallback
            try:
                _cfg = json.loads(
                    _client.bucket(_BUCKET).blob(sys.argv[1]).download_as_text()
                )
                if _cfg.get('run_id') and _cfg['run_id'] != 'daily':
                    _rid = _cfg['run_id']
            except Exception:
                pass
            _write_status(_client, _rid,
                          {'run_id': _rid, 'state': 'error', 'error': tb})
        except Exception:
            pass
        sys.exit(1)
