"""
client_profile_job.py — Cloud Run Job for the Client Profiles feature.

Builds multi-aspect capability profiles for client companies out of material
that already exists on their rows in data/all-contacts/clients/ (website
summary/scrape, Drive document extractions, Deep Research output). For each
selected company:

  1. Merge its contact rows into one material row (first non-empty per column)
  2. Assemble the selected source texts + fingerprint ALL available material
  3. One Claude call → profile summary + 2-8 independently searchable aspects
  4. Embed each aspect separately (text-embedding-ada-002, float64)
  5. Upsert the profile row into data/client-profiles/profiles.parquet

Clients are processed concurrently (one Claude call each), and the profile
store is re-read from GCS before every save so a profile edited in the
Streamlit view mid-run is never clobbered wholesale. Progress and the
partially-built store are checkpointed every few clients, so a timeout or
crash keeps the profiles already built.

Usage:
    python jobs/client_profile_job.py client-profile-configs/<run_id>.json

Environment variables (injected by Cloud Run from Secret Manager):
    ANTHROPIC_API_KEY, OPENAI_API_KEY

Config schema:
{
  "run_id":         "client_profile_2026-08-19_10-30-00",
  "company_keys":   ["Acme Robotics||https://acme.com", ...],
  "sources":        ["website", "drive", "technology"],
  "target_aspects": 4,
  "model":          "claude-sonnet-4-6",
  "concurrency":    4,
  "dry_run":        false
}
"""

import io
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import tiktoken
from anthropic import Anthropic
from google.cloud import storage
from openai import OpenAI

import src.modules.aspect_profile as ap

# ── Constants ──────────────────────────────────────────────────────────────────

_BUCKET         = ap.BUCKET
_CLIENTS_PREFIX = ap.CLIENTS_PREFIX
_STATUS_PREFIX  = 'client-profile-jobs/'

_TOKEN_LIMIT       = 7_500
_CHECKPOINT_EVERY  = 5
_DEFAULT_WORKERS   = 4
_MAX_WORKERS       = 8

# Graceful time budget: stop before Cloud Run's hard task timeout kills the
# container (which would lose un-checkpointed profiles AND the status file,
# leaving the UI polling forever). Deferred clients are reported for a re-run.
_TASK_TIMEOUT_S    = 7_200
_DEADLINE_MARGIN_S = 300


# ── Secrets / GCS ──────────────────────────────────────────────────────────────

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
        json.dumps(payload), content_type='application/json'
    )


def _load_client_frames(client: storage.Client) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for blob in client.list_blobs(_BUCKET, prefix=_CLIENTS_PREFIX):
        if not blob.name.endswith('.parquet'):
            continue
        try:
            frames.append(pd.read_parquet(io.BytesIO(blob.download_as_bytes())))
        except Exception as e:
            print(f'  WARN could not read {blob.name}: {e}', flush=True)
    return frames


# ── Embedding ──────────────────────────────────────────────────────────────────

def _get_embedding(text: str, oai: OpenAI, encoding: tiktoken.Encoding) -> list[float]:
    text = text.strip()
    if not text:
        raise ValueError('nothing to embed')
    words = text.split()
    while len(encoding.encode(text)) > _TOKEN_LIMIT:
        words = words[:-5]
        if not words:
            raise ValueError('nothing left to embed after token reduction')
        text = ' '.join(words)
    return oai.embeddings.create(
        input=[text], model='text-embedding-ada-002'
    ).data[0].embedding


# ── Claude ─────────────────────────────────────────────────────────────────────

def _claude_aspects(anth: Anthropic, model: str, system: str, user_msg: str
                    ) -> tuple[str, list[dict]]:
    """Aspect generation with one strict-JSON retry."""
    last_err = None
    for attempt in range(2):
        content = user_msg if attempt == 0 else (
            user_msg + '\n\nYour previous response was not valid JSON. '
                       'Return ONLY the valid JSON object.'
        )
        resp = anth.messages.create(
            model=model,
            max_tokens=4000,
            system=system,
            messages=[{'role': 'user', 'content': content}],
        )
        if resp.stop_reason == 'max_tokens':
            raise ValueError('Claude hit the output token limit')
        try:
            return ap.parse_aspect_response(resp.content[0].text)
        except ValueError as e:
            last_err = e
    raise ValueError(f'invalid response twice: {last_err}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_blob_path: str) -> None:
    gcs = _gcs()

    print(f'Loading config from {config_blob_path}', flush=True)
    config = json.loads(gcs.bucket(_BUCKET).blob(config_blob_path).download_as_text())

    run_id       = config['run_id']
    wanted_keys  = list(config.get('company_keys') or [])
    sources      = list(config.get('sources') or ap.SOURCE_KEYS)
    target       = int(config.get('target_aspects', 4))
    model        = config.get('model', ap.DEFAULT_MODEL)
    dry_run      = bool(config.get('dry_run', False))
    workers      = max(1, min(_MAX_WORKERS, int(config.get('concurrency', _DEFAULT_WORKERS))))

    sources = [s for s in sources if s in ap.SOURCE_KEYS]
    if not sources:
        raise ValueError('config contained no valid source keys')
    if not wanted_keys:
        raise ValueError('config contained no company_keys')

    deadline      = time.monotonic() + (_TASK_TIMEOUT_S - _DEADLINE_MARGIN_S)
    stopped_early = None

    anth     = Anthropic(api_key=_get_secret('anthropic-api-key'))
    oai      = OpenAI(api_key=_get_secret('openai-api-key'))
    encoding = tiktoken.get_encoding('cl100k_base')

    system = ap.build_aspect_system(target)

    print('Loading client frames…', flush=True)
    frames = _load_client_frames(gcs)
    if not frames:
        raise RuntimeError(f'no parquet files under {_CLIENTS_PREFIX}')
    combined = pd.concat(frames, ignore_index=True)
    combined['_key'] = combined.apply(ap.company_key, axis=1)
    del frames

    # Material row per requested company
    wanted   = set(wanted_keys)
    material: dict[str, dict] = {}
    for key, group in combined.groupby('_key', sort=False):
        if key in wanted:
            material[key] = ap.merge_company_row(group)
    del combined

    total   = len(wanted_keys)
    built:    list[dict] = []
    records:  list[dict] = []
    errors:   list[str]  = []
    deferred: list[str]  = []
    done = 0

    def _name(key: str) -> str:
        row = material.get(key) or {}
        return str(row.get('company_name') or key.split('||', 1)[0] or key)

    def _status(state: str) -> dict:
        return {
            'run_id':         run_id,
            'state':          state,
            'dry_run':        dry_run,
            'model':          model,
            'sources':        sources,
            'target_aspects': target,
            'clients_total':  total,
            'clients_done':   done,
            'built':          built,
            'errors':         errors,
            'deferred':       deferred,
            'stopped_early':  stopped_early,
            'profiles_blob':  ap.PROFILES_BLOB,
            'error':          None,
        }

    def _save_records() -> None:
        """Re-read the store, upsert everything built this run, write it back.
        Re-reading means a profile edited in the view mid-run survives; the
        run's own records always win for the companies it rebuilt."""
        if dry_run or not records:
            return
        existing = ap.load_profiles(gcs)
        ap.save_profiles(gcs, ap.upsert_profiles(existing, records))

    def _build_one(key: str) -> dict:
        """Runs in a worker thread. Never raises — the outcome is the return."""
        name = _name(key)
        if time.monotonic() > deadline:
            return {'key': key, 'name': name, 'outcome': 'deferred'}
        row = material.get(key)
        if row is None:
            return {'key': key, 'name': name, 'outcome': 'error',
                    'note': 'no client rows matched this company key'}
        try:
            texts = ap.assemble_source_texts(row, sources)
            if not texts:
                return {'key': key, 'name': name, 'outcome': 'error',
                        'note': 'none of the selected sources have material'}
            summary, aspects = _claude_aspects(
                anth, model, system,
                ap.build_aspect_user_message({
                    'company_name': name,
                    'website':      row.get('companyWebsite'),
                    'state':        row.get('state'),
                }, texts),
            )
            vectors = [_get_embedding(ap.aspect_embed_text(a), oai, encoding)
                       for a in aspects]
            record = ap.build_profile_record(
                company_key     = key,
                company_name    = name,
                website         = str(row.get('companyWebsite') or ''),
                profile_summary = summary,
                aspects         = aspects,
                vectors         = vectors,
                sources_used    = list(texts.keys()),
                # Fingerprint over ALL available material, not just the sources
                # used — any later change to any of it should read as stale.
                fingerprint     = ap.source_fingerprint(ap.assemble_source_texts(row)),
                model           = model,
            )
            return {'key': key, 'name': name, 'outcome': 'built', 'record': record}
        except Exception as e:
            traceback.print_exc()
            return {'key': key, 'name': name, 'outcome': 'error', 'note': str(e)[:300]}

    print(f'Building {total} profile(s) with {workers} worker(s), model={model}, '
          f'sources={",".join(sources)}', flush=True)
    _write_status(gcs, run_id, _status('running'))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_build_one, key): key for key in wanted_keys}
        for future in as_completed(futures):
            res  = future.result()
            done += 1
            name = res['name']

            if res['outcome'] == 'built':
                rec = res['record']
                records.append(rec)
                built.append({
                    'company_key':  rec['company_key'],
                    'company_name': rec['company_name'],
                    'n_aspects':    rec['n_aspects'],
                    'sources_used': rec['sources_used'],
                    'aspect_labels': rec['aspect_labels'],
                })
                print(f'[{done}/{total}] {name} → {rec["n_aspects"]} aspects', flush=True)
            elif res['outcome'] == 'deferred':
                stopped_early = 'timeout'
                deferred.append(name)
                print(f'[{done}/{total}] {name} deferred (time budget)', flush=True)
            else:
                errors.append(f'{name}: {res.get("note") or "failed"}')
                print(f'[{done}/{total}] {name} ERROR {res.get("note")}', flush=True)

            if done % _CHECKPOINT_EVERY == 0 and done < total:
                print(f'  checkpoint at {done}/{total}', flush=True)
                try:
                    _save_records()
                except Exception as e:
                    print(f'  WARN checkpoint save failed: {e}', flush=True)
                _write_status(gcs, run_id, _status('running'))

    _save_records()
    _write_status(gcs, run_id, _status('complete'))
    print(f'\nDone. {len(built)} built, {len(errors)} errored, '
          f'{len(deferred)} deferred.', flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python jobs/client_profile_job.py <config_blob_path>',
              file=sys.stderr)
        sys.exit(1)

    run_id_fallback = sys.argv[1].split('/')[-1].replace('.json', '')
    try:
        main(sys.argv[1])
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        try:
            _write_status(_gcs(), run_id_fallback,
                          {'run_id': run_id_fallback, 'state': 'error', 'error': tb})
        except Exception:
            pass
        sys.exit(1)
