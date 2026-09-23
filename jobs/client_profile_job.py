"""
client_profile_job.py — Cloud Run Job for the Client Profiles feature.

Builds multi-aspect capability profiles for the companies of one pool
(src/modules/pools.py — clients or prospects) out of material that already
exists on their contact rows (website summary/scrape, Drive document
extractions, meeting digests, Deep Research output). For each selected
company:

  1. Merge its contact rows into one material row (first non-empty per column)
  2. Assemble the selected source texts + fingerprint ALL available material
  3. Claude pass 1 → profile summary, independently searchable aspects, and the
     markets those aspects serve (ranked 1st, 2nd, 3rd..., up to max_markets
     plus Defense when a DoD use case is plausible), every aspect earmarked to
     at least one market
  4. Embed each aspect, fold together any two that are near-identical, then
     embed each market narrative and fold near-identical markets
     (text-embedding-ada-002, float64)
  5. Claude pass 2 (optional) → unexplored markets: customer worlds the company
     does NOT serve, inferred by linking the aspects it already has. Stored in
     their own columns, so speculation is never read back as capability
  6. Upsert the profile row into that pool's profile store
     (data/client-profiles/profiles.parquet for clients,
     prospect_profiles.parquet for prospects)

Aspect merging happens before market membership is re-derived, so a market can
never end up pointing at an aspect that was folded away. Pass 2 is wrapped in
its own error handling: it is additive, and losing it must never cost the pass-1
work, which is a Claude call plus every embedding. Note that enabling it roughly
doubles per-client latency, so the graceful time budget below is spent twice as
fast.

Clients are processed concurrently (one or two Claude calls each), and the profile
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
  "pool":           "clients",
  "company_keys":   ["Acme Robotics||https://acme.com", ...],
  "sources":        ["website", "drive", "technology"],
  "target_aspects": 4,
  "max_markets":    4,
  "assess_defense": true,
  "assess_unexplored": true,
  "max_unexplored": 3,
  "market_merge_threshold": 0.93,
  "aspect_merge_threshold": 0.96,
  "model":          "claude-sonnet-4-6",
  "concurrency":    4,
  "dry_run":        false
}
"""

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
import src.modules.pools as pl

# ── Constants ──────────────────────────────────────────────────────────────────

_BUCKET         = ap.BUCKET
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


def _load_pool_frames(client: storage.Client, pool: str) -> list[pd.DataFrame]:
    frames, errors = pl.load_frames(client, pool)
    for err in errors:
        print(f'  WARN could not read {err}', flush=True)
    return list(frames.values())


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

def _claude_json(anth: Anthropic, model: str, system: str, user_msg: str, parse):
    """One Claude call with one strict-JSON retry, parsed by `parse`.

    Shared by both passes: aspect+market generation and the unexplored-market
    pass differ only in their prompt and their parser."""
    last_err = None
    for attempt in range(2):
        content = user_msg if attempt == 0 else (
            user_msg + '\n\nYour previous response was not valid JSON. '
                       'Return ONLY the valid JSON object.'
        )
        resp = anth.messages.create(
            model=model,
            # Aspects plus the markets block; a truncated response is a hard
            # error below, so leave headroom rather than lose the run.
            max_tokens=6000,
            system=system,
            messages=[{'role': 'user', 'content': content}],
        )
        if resp.stop_reason == 'max_tokens':
            raise ValueError('Claude hit the output token limit')
        try:
            return parse(resp.content[0].text)
        except ValueError as e:
            last_err = e
    raise ValueError(f'invalid response twice: {last_err}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_blob_path: str) -> None:
    gcs = _gcs()

    print(f'Loading config from {config_blob_path}', flush=True)
    config = json.loads(gcs.bucket(_BUCKET).blob(config_blob_path).download_as_text())

    run_id       = config['run_id']
    # Which directory of companies this run profiles, and therefore which
    # contact prefix it reads and which profile store it writes. Absent in
    # configs written before pools existed → clients, the previous behaviour.
    #
    # NOT named `pool`: `with ThreadPoolExecutor(...) as pool` is the house
    # idiom in every job in this repo (16 occurrences), and one of them lives
    # further down THIS function. A local named `pool` there silently rebinds
    # the name for the closures below — _status(), _save_records() — which are
    # defined before it and called after it. That shipped once: every profile
    # save resolved ap.profiles_blob(<ThreadPoolExecutor>) instead of a pool
    # name, and the run died on json.dumps of the executor in the status file.
    pool_key     = config.get('pool') or ap.DEFAULT_POOL
    wanted_keys  = list(config.get('company_keys') or [])
    sources      = list(config.get('sources') or ap.SOURCE_KEYS)
    target       = int(config.get('target_aspects', 4))
    max_markets  = int(config.get('max_markets', ap.MAX_MARKETS))
    assess_def   = bool(config.get('assess_defense', True))
    assess_unexp = bool(config.get('assess_unexplored', True))
    max_unexp    = int(config.get('max_unexplored', ap.MAX_UNEXPLORED))
    merge_thresh = float(config.get('market_merge_threshold', ap.MARKET_MERGE_THRESHOLD))
    aspect_merge = float(config.get('aspect_merge_threshold', ap.ASPECT_MERGE_THRESHOLD))
    model        = config.get('model', ap.DEFAULT_MODEL)
    dry_run      = bool(config.get('dry_run', False))
    workers      = max(1, min(_MAX_WORKERS, int(config.get('concurrency', _DEFAULT_WORKERS))))

    if not pl.is_pool(pool_key):
        raise ValueError(f'config named an unknown pool: {pool_key!r}')

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

    system     = ap.build_aspect_system(target, max_markets, assess_def)
    unexp_system = ap.build_unexplored_system(max_unexp) if assess_unexp else ''

    print(f'Loading {pool_key} frames…', flush=True)
    frames = _load_pool_frames(gcs, pool_key)
    if not frames:
        raise RuntimeError(f'no parquet files under {pl.contacts_prefix(pool_key)}')
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
    warn_notes: list[str] = []
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
            'pool':           pool_key,
            'model':          model,
            'sources':        sources,
            'target_aspects': target,
            'max_markets':    max_markets,
            'assess_defense': assess_def,
            'assess_unexplored': assess_unexp,
            'max_unexplored': max_unexp,
            'clients_total':  total,
            'clients_done':   done,
            'built':          built,
            'errors':         errors,
            'warnings':       warn_notes,
            'deferred':       deferred,
            'stopped_early':  stopped_early,
            'profiles_blob':  ap.profiles_blob(pool_key),
            'error':          None,
        }

    def _save_records() -> None:
        """Re-read the store, upsert everything built this run, write it back.
        Re-reading means a profile edited in the view mid-run survives; the
        run's own records always win for the companies it rebuilt."""
        if dry_run or not records:
            return
        existing = ap.load_profiles(gcs, pool=pool_key)
        ap.save_profiles(
            gcs, ap.upsert_profiles(existing, records, pool=pool_key), pool=pool_key
        )

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
            parsed = _claude_json(
                anth, model, system,
                ap.build_aspect_user_message({
                    'company_name': name,
                    'website':      row.get('companyWebsite'),
                    'state':        row.get('state'),
                }, texts),
                ap.parse_aspect_response,
            )
            summary = parsed['profile_summary']
            aspects = parsed['aspects']
            markets = parsed['markets']

            vectors = [_get_embedding(ap.aspect_embed_text(a), oai, encoding)
                       for a in aspects]
            # Fold near-identical aspects together BEFORE market membership is
            # re-derived below, so no market is left pointing at an aspect that
            # was merged away. The merges are reported: a threshold that is
            # eating distinct capabilities is invisible otherwise.
            aspects, vectors, merges = ap.merge_similar_aspects(
                aspects, vectors, aspect_merge
            )
            # Reported whether or not anything merged: cosine cannot reliably
            # separate a genuine repeat from two facets of one platform, so the
            # near misses are for a human to judge in the editor.
            near = ap.nearest_aspect_pairs(aspects, vectors)

            market_vectors = [_get_embedding(ap.market_embed_text(m), oai, encoding)
                              for m in markets]
            if markets:
                # Two markets whose narratives read alike are one market
                # described twice; fold them together before storing.
                aspects, markets, market_vectors = ap.merge_similar_markets(
                    aspects, markets, market_vectors, merge_thresh
                )
                by_name = {m['market']: v for m, v in zip(markets, market_vectors)}
                aspects, markets = ap.normalize_markets(aspects, markets, max_markets)
                market_vectors = [by_name[m['market']] for m in markets]

            # Pass 2 - markets the company does NOT serve, inferred by
            # linking the aspects it does have. Additive and separately
            # guarded: a failure here must never discard the pass-1 work above,
            # which cost a Claude call and every embedding.
            unexplored, unexplored_vectors, unexp_err = [], [], ''
            if assess_unexp:
                try:
                    unexplored = _claude_json(
                        anth, model, unexp_system,
                        ap.build_unexplored_user_message(
                            {
                                'company_name': name,
                                'website':      row.get('companyWebsite'),
                                'state':        row.get('state'),
                            },
                            summary, aspects, markets, ap.stated_intentions(row),
                        ),
                        lambda raw: ap.parse_unexplored_response(
                            raw, markets, aspects, max_unexp
                        ),
                    )
                    unexplored_vectors = [
                        _get_embedding(ap.market_embed_text(m), oai, encoding)
                        for m in unexplored
                    ]
                except Exception as e:
                    traceback.print_exc()
                    unexplored, unexplored_vectors = [], []
                    unexp_err = str(e)[:300]

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
                markets         = markets,
                market_vectors  = market_vectors,
                unexplored         = unexplored,
                unexplored_vectors = unexplored_vectors,
                dod_assessment  = parsed['dod_assessment'],
            )
            return {'key': key, 'name': name, 'outcome': 'built', 'record': record,
                    'merges': merges, 'near_pairs': near,
                    'unexplored_error': unexp_err}
        except Exception as e:
            traceback.print_exc()
            return {'key': key, 'name': name, 'outcome': 'error', 'note': str(e)[:300]}

    print(f'Building {total} profile(s) with {workers} worker(s), model={model}, '
          f'sources={",".join(sources)}, max_markets={max_markets}, '
          f'assess_defense={assess_def}, assess_unexplored={assess_unexp} '
          f'(max {max_unexp})', flush=True)
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
                    'company_key':   rec['company_key'],
                    'company_name':  rec['company_name'],
                    'n_aspects':     rec['n_aspects'],
                    'sources_used':  rec['sources_used'],
                    'aspect_labels': rec['aspect_labels'],
                    'n_markets':     rec['n_markets'],
                    'market_labels': rec['market_labels'],
                    'has_defense':   ap.DEFENSE_MARKET in [
                        m.get('market') for m in ap.profile_markets(rec)
                    ],
                    'n_unexplored':     rec['n_unexplored'],
                    'unexplored_labels': rec['unexplored_labels'],
                    'aspect_merges':    res.get('merges') or [],
                    'aspect_near_pairs': res.get('near_pairs') or [],
                })
                for merge in (res.get('merges') or []):
                    warn_notes.append(f'{name}: merged near-identical aspects {merge}')
                if res.get('unexplored_error'):
                    warn_notes.append(
                        f'{name}: unexplored-market pass failed '
                        f'({res["unexplored_error"]}) - the profile was saved '
                        'without unexplored markets'
                    )
                print(f'[{done}/{total}] {name} → {rec["n_aspects"]} aspects, '
                      f'{rec["n_markets"]} markets [{rec["market_labels"]}], '
                      f'{rec["n_unexplored"]} unexplored '
                      f'[{rec["unexplored_labels"]}]', flush=True)
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
          f'{len(deferred)} deferred, {len(warn_notes)} warning(s).', flush=True)


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
