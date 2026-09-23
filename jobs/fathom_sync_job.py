"""
fathom_sync_job.py — Cloud Run Job for the Fathom Meetings feature.

Reads a job config from GCS, then:
  1. Sweeps GET /meetings once (cheap: summaries + action items + CRM matches
     ride along on each page, so only transcripts cost a per-meeting call)
  2. Attributes each meeting to a client by its external calendar-invitee
     domains — manual assignments in fathom-configs/assignments.json first,
     then the client's own companyWebsite domain
  3. Fetches transcripts for new attributed meetings of the selected clients
  4. Stores every ingested meeting verbatim at data/fathom/meetings/{id}.json
     and one row per meeting in data/fathom/meetings_index.parquet
  5. One Claude call per client with new meetings: prior digest + prior
     extraction + the new calls → refreshed digest + extraction
  6. Writes client_meetings_data / client_meetings_summary /
     meetings_updated_at onto every contact row of the client
  7. Checkpoints touched parquets + sync_state + interim status every 10
     clients (timeout resilience — a re-trigger resumes via sync_state)

This job NEVER touches `summary` or `embeddings`. Meeting material reaches
grant matching only through the multi-aspect client profiles (Stage 8), which
read it as the `meetings` source in src/modules/aspect_profile.py.

Unattributed meetings are deliberately NOT marked synced: their domains are
recorded in assignments.json for review, and assigning one later ingests its
backlog on the next run without needing a full re-sync.

Only meetings with at least one external invitee are swept — an internal-only
call has no external domain and so could never be attributed to a client.

Usage:
    python jobs/fathom_sync_job.py fathom-configs/<run_id>.json

Environment variables (injected by Cloud Run from Secret Manager):
    ANTHROPIC_API_KEY, FATHOM_API_KEY

Config schema:
{
  "run_id":                  "fathom_2026-09-09_10-30-00",
  "client_keys":             ["Acme Robotics||https://acme.com"],
  "lookback_days":           90,
  "created_after":           null,
  "full_resync":             false,
  "dry_run":                 false,
  "max_meetings":            400,
  "max_meetings_per_client": 12,
  "per_client_char_cap":     120000,
  "transcript_char_cap":     40000,
  "task_timeout_s":          14400,
  "model":                   "claude-sonnet-4-6"
}
"""

import io
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from anthropic import Anthropic
from google.cloud import storage

import src.modules.anthropic_utils as au
from src.modules import fathom_client as fc
import src.modules.pools as pl

# ── Constants ──────────────────────────────────────────────────────────────────

_BUCKET          = 'cc-matcher-bucket-jeg-v1'
_CFG_PREFIX      = 'fathom-configs/'
_STATUS_PREFIX   = 'fathom-jobs/'
_SYNC_STATE_BLOB = 'fathom-configs/sync_state.json'
_ASSIGN_BLOB     = 'fathom-configs/assignments.json'
_MEETINGS_PREFIX = 'data/fathom/meetings/'
_INDEX_BLOB      = 'data/fathom/meetings_index.parquet'

_TRANSCRIPT_WORKERS = 2      # heavy-budget calls; the pacing gate serialises anyway
_CHECKPOINT_EVERY   = 10
_DEFAULT_MODEL      = 'claude-sonnet-4-6'
_MAX_LISTED         = 200    # cap on list-shaped fields inside status.json
_MAX_DRY_PREVIEWS   = 3      # clients whose digest a dry run reports back
_DRY_PREVIEW_CHARS  = 4_000

_DEFAULT_LOOKBACK_DAYS   = 90
_DEFAULT_MAX_MEETINGS    = 400
_DEFAULT_MAX_PER_CLIENT  = 12
_DEFAULT_CHAR_CAP        = 120_000
_DEFAULT_TRANSCRIPT_CAP  = 40_000

# Graceful time budget: stop before Cloud Run's hard task timeout kills the
# container, which would lose un-checkpointed work AND the status file and
# leave the UI polling forever. Deferred work resumes on the next run.
_DEFAULT_TASK_TIMEOUT_S = 14_400
_MIN_TASK_TIMEOUT_S     = 900
_MAX_TASK_TIMEOUT_S     = 86_400
_DEADLINE_MARGIN_S      = 600

_INDEX_COLUMNS = [
    'recording_id', 'client_key', 'company_name', 'title', 'meeting_date',
    'created_at', 'duration_min', 'recorded_by_email', 'external_domains',
    'n_transcript_lines', 'summary_md', 'url', 'blob_path', 'run_id', 'synced_at',
]

_MERGE_SYSTEM = (
    'You maintain company profiles for a federal grant-matching system. You are '
    'given a client company\'s existing meeting-derived data and the notes and '
    'transcripts of new calls between our consultants and that client. Distil '
    'what the calls reveal about the COMPANY — its technology, capabilities, '
    'products, R&D work, facilities, past federal awards, and the markets it '
    'serves — so it can be matched against government grant topics (SBIR/STTR, '
    'BAAs, agency solicitations).\n\n'
    'Respond with ONLY a valid JSON object (no markdown fences) of this shape:\n'
    '{\n'
    '  "no_meaningful_change": false,\n'
    '  "meetings_digest": "5-12 bullet plain-text digest of what these calls reveal "\n'
    '                     "about the COMPANY - its capability, technology and market "\n'
    '                     "position - not about our engagement with it (one bullet per "\n'
    '                     "line, prefixed with - )",\n'
    '  "extracted": {\n'
    '    "technologies": [], "capabilities": [], "products_services": [],\n'
    '    "rd_projects": [], "federal_programs_agencies": [], "past_awards": [],\n'
    '    "target_markets": [], "customers_partners": [],\n'
    '    "certifications_registrations": [], "facilities_equipment": [],\n'
    '    "team_expertise": "", "keywords": [], "notable_updates": []\n'
    '  }\n'
    '}\n\n'
    'Rules:\n'
    '- CONFIRMED vs ASPIRATIONAL is the most important distinction. The main '
    '"extracted" arrays are only for capability the company demonstrably HAS '
    'today — built, deployed, funded, staffed, or otherwise stated as fact. '
    'Anything hypothetical, planned, hoped-for, "we are thinking about", or '
    'belonging to a third party goes in "notable_updates", phrased so its status '
    'is unmistakable ("plans to…", "exploring…").\n'
    '- Never invent technologies, awards, customers, partners or certifications. '
    'Transcripts are speech: if something is garbled or ambiguous, leave it out.\n'
    '- Ignore meeting logistics, scheduling, the pricing of our own services, '
    'contract admin and small talk. None of that is capability.\n'
    '- These are consulting calls, so much of what is said concerns the '
    'ENGAGEMENT rather than the company. Write about the company, not about the '
    'work we are doing for it. Exclude our deliverables and task assignments '
    '("the Phase 3 partner analysis is in preparation", "the GUI developer will '
    'record a demo video"), proposal and submission process status, filing '
    'deadlines, portal mechanics, and who owes whom what by when. Keep the '
    'SUBSTANCE those tasks are about: which agencies, offices and programs the '
    'company is pursuing belongs in "federal_programs_agencies", and the '
    'technology being written up or demonstrated belongs in "technologies" — '
    'but never the project-management wrapper around either.\n'
    '- "past_awards": federal grants/contracts the company says it has won, with '
    'agency and program where stated (e.g. "AF SBIR Phase II, 2024").\n'
    '- "federal_programs_agencies": agencies, offices or programs the company is '
    'pursuing or has already engaged with.\n'
    '- "keywords": 8-20 technical terms a solicitation in this company\'s field '
    'would use.\n'
    '- "team_expertise": one or two sentences on the technical team\'s background; '
    '"" when the calls say nothing about it.\n'
    '- Merge new findings INTO existing_meetings_data: the returned "extracted" '
    'object fully REPLACES the stored one, so carry forward everything still '
    'accurate and drop nothing that was previously established.\n'
    '- Set no_meaningful_change=true only when the new calls reveal nothing about '
    'what the company does or wants to pursue (purely scheduling or admin calls). '
    'Leave "meetings_digest" and "extracted" empty in that case.\n'
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


# ── Company frames (clients/ convention: company_name / summary) ──────────────

def _load_client_frames(client: storage.Client) -> dict[str, pd.DataFrame]:
    """Every pool in one {blob_name: frame} dict.

    A meeting is attributed by the external invitee's domain, and that domain
    may belong to a client or to a targeted prospect — the sweep cannot know
    which, and it costs one paginated pass either way, so both pools are always
    resolved. Blob names are unique per prefix, so the merged dict still writes
    each row back to the exact file it came from and a prospect's digest lands
    in the prospect parquet."""
    frames: dict[str, pd.DataFrame] = {}
    for pool_key in pl.POOL_KEYS:
        pool_frames, errors = pl.load_frames(client, pool_key)
        for err in errors:
            print(f'  WARN could not read {err}', flush=True)
        frames.update(pool_frames)
    return frames


_group_mask = pl.key_mask


def _client_identities(frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    """client_key -> company_name for every company in either pool."""
    out: dict[str, str] = {}
    for df in frames.values():
        if 'company_name' not in df.columns:
            continue
        names    = df['company_name'].fillna('').astype(str).str.strip()
        websites = (df.get('companyWebsite', pd.Series('', index=df.index))
                    .fillna('').astype(str).str.strip())
        for name, site in zip(names, websites):
            if name:
                out[f'{name}||{site}'] = name
    return out


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
            max_tokens=16000,   # a 12-call client can produce a long extraction
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


# ── Meeting persistence ────────────────────────────────────────────────────────

def _raw_meeting_doc(meeting: dict, transcript: list[dict], client_key: str,
                     run_id: str) -> dict:
    """Everything Fathom gave us about one call, stored verbatim so the
    extraction can be redone later without spending API calls again."""
    return {
        'recording_id':     meeting.get('recording_id'),
        'client_key':       client_key,
        'title':            fc.meeting_title(meeting),
        'meeting_date':     fc.meeting_date(meeting),
        'created_at':       meeting.get('created_at'),
        'duration_min':     fc.duration_minutes(meeting),
        'url':              meeting.get('url'),
        'share_url':        meeting.get('share_url'),
        'meeting_url':      meeting.get('meeting_url'),
        'meeting_type':     meeting.get('meeting_type'),
        'recorded_by':      meeting.get('recorded_by'),
        'calendar_invitees': meeting.get('calendar_invitees'),
        'external_domains': fc.external_domains(meeting),
        'fathom_summary':   fc.summary_markdown(meeting),
        'action_items':     meeting.get('action_items'),
        'crm_matches':      meeting.get('crm_matches'),
        'transcript':       transcript,
        'run_id':           run_id,
        'synced_at':        datetime.now(timezone.utc).isoformat(),
    }


def _index_row(meeting: dict, transcript: list[dict], client_key: str,
               company_name: str, blob_path: str, run_id: str) -> dict:
    recorded_by = meeting.get('recorded_by') or {}
    return {
        'recording_id':       str(meeting.get('recording_id') or ''),
        'client_key':         client_key,
        'company_name':       company_name,
        'title':              fc.meeting_title(meeting),
        'meeting_date':       fc.meeting_date(meeting),
        'created_at':         str(meeting.get('created_at') or ''),
        'duration_min':       fc.duration_minutes(meeting),
        'recorded_by_email':  str((recorded_by or {}).get('email') or ''),
        'external_domains':   ', '.join(fc.external_domains(meeting)),
        'n_transcript_lines': len(transcript or []),
        'summary_md':         fc.summary_markdown(meeting),
        'url':                str(meeting.get('url') or ''),
        'blob_path':          blob_path,
        'run_id':             run_id,
        'synced_at':          datetime.now(timezone.utc).isoformat(),
    }


def _upsert_index(client: storage.Client, rows: list[dict]) -> None:
    """Replace any existing rows for these recording_ids, append the rest."""
    if not rows:
        return
    blob = client.bucket(_BUCKET).blob(_INDEX_BLOB)
    if blob.exists():
        existing = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
    else:
        existing = pd.DataFrame({c: pd.Series(dtype='object') for c in _INDEX_COLUMNS})
    new = pd.DataFrame(rows)
    if not existing.empty and 'recording_id' in existing.columns:
        ids      = set(new['recording_id'].astype(str))
        existing = existing[~existing['recording_id'].astype(str).isin(ids)]
    merged = pd.concat([existing, new], ignore_index=True)
    for col in _INDEX_COLUMNS:
        if col not in merged.columns:
            merged[col] = ''
    merged = merged[_INDEX_COLUMNS].sort_values(
        'meeting_date', ascending=False, kind='stable'
    ).reset_index(drop=True)
    buf = io.BytesIO()
    merged.to_parquet(buf, index=False)
    buf.seek(0)
    blob.upload_from_file(buf, content_type='application/octet-stream')


# ── Main ───────────────────────────────────────────────────────────────────────

def main(config_blob_path: str) -> None:
    gcs    = _gcs()
    config = _load_json_blob(gcs, config_blob_path)
    if config is None:
        raise RuntimeError(f'Config blob not found: {config_blob_path}')

    run_id         = config['run_id']
    wanted_keys    = [k for k in (config.get('client_keys') or []) if k]
    lookback_days  = max(1, int(config.get('lookback_days') or _DEFAULT_LOOKBACK_DAYS))
    created_after  = str(config.get('created_after') or '').strip()
    full_resync    = bool(config.get('full_resync', False))
    dry_run        = bool(config.get('dry_run', False))
    max_meetings   = max(1, int(config.get('max_meetings') or _DEFAULT_MAX_MEETINGS))
    max_per_client = max(1, int(config.get('max_meetings_per_client')
                                or _DEFAULT_MAX_PER_CLIENT))
    char_cap       = max(5_000, int(config.get('per_client_char_cap') or _DEFAULT_CHAR_CAP))
    tr_cap         = max(1_000, int(config.get('transcript_char_cap')
                                    or _DEFAULT_TRANSCRIPT_CAP))
    model          = config.get('model') or _DEFAULT_MODEL
    task_timeout_s = max(_MIN_TASK_TIMEOUT_S,
                         min(int(config.get('task_timeout_s') or _DEFAULT_TASK_TIMEOUT_S),
                             _MAX_TASK_TIMEOUT_S))
    deadline = time.monotonic() + (task_timeout_s - _DEADLINE_MARGIN_S)

    if not created_after:
        created_after = (
            datetime.now(timezone.utc) - timedelta(days=lookback_days)
        ).strftime('%Y-%m-%dT%H:%M:%SZ')

    fathom_key = _get_secret('fathom-api-key')
    anth       = Anthropic(api_key=_get_secret('anthropic-api-key'))
    today      = date.today().isoformat()

    print(f'[{run_id}] dry_run={dry_run} full_resync={full_resync} '
          f'created_after={created_after} clients={len(wanted_keys) or "all"}', flush=True)

    # ── State ─────────────────────────────────────────────────────────────────
    assignments_doc = _load_json_blob(gcs, _ASSIGN_BLOB) or {}
    assignments     = assignments_doc.setdefault('assignments', {})
    unassigned      = assignments_doc.setdefault('unassigned', {})
    skipped         = assignments_doc.setdefault('skipped', {})
    assignments_doc.setdefault('settings', {})

    sync_state    = _load_json_blob(gcs, _SYNC_STATE_BLOB) or {}
    state_meetings = sync_state.setdefault('meetings', {})
    state_clients  = sync_state.setdefault('clients', {})

    frames     = _load_client_frames(gcs)
    identities = _client_identities(frames)
    if not identities:
        raise RuntimeError(
            'No companies found in '
            + ' or '.join(pl.contacts_prefix(k) for k in pl.POOL_KEYS)
        )

    # domain → client_key: the client's own website first, manual assignments win
    domain_map: dict[str, str] = {}
    for key in identities:
        _, website = key.split('||', 1)
        domain = fc.bare_domain(website)
        if domain:
            domain_map.setdefault(domain, key)
    for domain, entry in assignments.items():
        ckey = (entry or {}).get('client_key')
        if ckey:
            domain_map[fc.bare_domain(domain) or domain] = ckey
    skipped_domains = {fc.bare_domain(d) or d for d in skipped}

    wanted = set(wanted_keys) if wanted_keys else None

    _write_status(gcs, run_id, {
        'run_id': run_id, 'state': 'running', 'dry_run': dry_run,
        'meetings_swept': 0, 'clients_done': 0, 'clients_total': 0, 'error': None,
    })

    # ── Phase 1: discovery sweep ──────────────────────────────────────────────
    per_client: dict[str, list[dict]] = {}
    unmatched: dict[str, dict] = {}
    swept = 0
    new_attributed = 0
    stopped_early = None
    sweep_note = ''
    try:
        for meeting in fc.iter_meetings(
            fathom_key,
            created_after=created_after,
            include=('summary', 'action_items', 'crm_matches'),
            domains_type='one_or_more_external',
        ):
            swept += 1
            rid = str(meeting.get('recording_id') or '')
            if not rid:
                continue
            if not full_resync and rid in state_meetings:
                continue

            domains = fc.external_domains(meeting)
            ckey = next((domain_map[d] for d in domains if d in domain_map), None)
            if ckey is None:
                for domain in domains:
                    if domain in skipped_domains:
                        continue
                    entry = unmatched.setdefault(domain, {
                        'domain': domain, 'meeting_count': 0,
                        'sample_titles': [], 'crm_companies': [],
                    })
                    entry['meeting_count'] += 1
                    if len(entry['sample_titles']) < 3:
                        entry['sample_titles'].append(fc.meeting_title(meeting))
                    for name in fc.crm_company_names(meeting):
                        if name not in entry['crm_companies'] and len(entry['crm_companies']) < 3:
                            entry['crm_companies'].append(name)
                continue

            if wanted is not None and ckey not in wanted:
                continue
            per_client.setdefault(ckey, []).append(meeting)
            new_attributed += 1
            if new_attributed >= max_meetings:
                stopped_early = 'max_meetings'
                sweep_note = f'sweep stopped at max_meetings={max_meetings}'
                break
            if time.monotonic() > deadline:
                stopped_early = 'timeout'
                sweep_note = 'time budget spent during the meeting sweep'
                break
    except fc.RateLimitStalledError as e:
        stopped_early = 'rate_limit'
        sweep_note = str(e)
        print(f'  WARN {sweep_note}', flush=True)

    print(f'[{run_id}] swept {swept} meetings, {new_attributed} new across '
          f'{len(per_client)} clients, {len(unmatched)} unmatched domains', flush=True)

    # Unmatched domains are the review queue — merge, never clobber.
    if not dry_run and unmatched:
        for domain, entry in unmatched.items():
            if domain in assignments or domain in skipped:
                continue
            prior = unassigned.get(domain) or {}
            unassigned[domain] = {
                'domain':        domain,
                'meeting_count': int(prior.get('meeting_count') or 0) + entry['meeting_count'],
                'sample_titles': (entry['sample_titles'] or prior.get('sample_titles') or [])[:3],
                'crm_companies': (entry['crm_companies'] or prior.get('crm_companies') or [])[:3],
                'first_seen':    prior.get('first_seen') or today,
                'last_seen':     today,
            }
        assignments_doc['updated_at'] = datetime.now(timezone.utc).isoformat()
        _save_json_blob(gcs, _ASSIGN_BLOB, assignments_doc)

    # ── Phase 2: per-client ingest ────────────────────────────────────────────
    client_order = sorted(per_client, key=lambda k: identities.get(k, k).lower())
    results: list[dict] = []
    touched_blobs: set[str] = set()
    index_rows: list[dict] = []
    dry_previews: list[dict] = []
    meetings_ingested = 0
    meetings_deferred = 0
    clients_updated = clients_unchanged = clients_errored = 0

    def _checkpoint(state_label: str, done: int, total: int) -> None:
        if not dry_run:
            for blob_name in touched_blobs:
                buf = io.BytesIO()
                frames[blob_name].to_parquet(buf, index=False)
                buf.seek(0)
                gcs.bucket(_BUCKET).blob(blob_name).upload_from_file(
                    buf, content_type='application/octet-stream')
            touched_blobs.clear()
            _upsert_index(gcs, index_rows)
            index_rows.clear()
            sync_state['updated_at'] = datetime.now(timezone.utc).isoformat()
            _save_json_blob(gcs, _SYNC_STATE_BLOB, sync_state)
        _write_status(gcs, run_id, {
            'run_id': run_id, 'state': state_label, 'dry_run': dry_run,
            'clients_done': done, 'clients_total': total,
            'clients_updated': clients_updated,
            'clients_unchanged': clients_unchanged,
            'clients_errored': clients_errored,
            'meetings_swept': swept, 'meetings_new': new_attributed,
            'meetings_ingested': meetings_ingested,
            'meetings_deferred': meetings_deferred,
            'api_calls_used': fc.call_count(),
            'error': None,
        })

    total = len(client_order)
    for n, client_key in enumerate(client_order, start=1):
        company_name = identities.get(client_key, client_key.split('||', 1)[0])

        if time.monotonic() > deadline:
            stopped_early = stopped_early or 'timeout'
            for remaining in client_order[n - 1:]:
                results.append({
                    'client_key': remaining,
                    'company_name': identities.get(remaining, remaining),
                    'outcome': 'deferred', 'meetings_processed': 0,
                    'note': 'time budget spent — re-run to continue',
                })
                meetings_deferred += len(per_client.get(remaining, []))
            break

        outcome = {'client_key': client_key, 'company_name': company_name,
                   'outcome': 'unchanged', 'meetings_processed': 0, 'note': ''}
        try:
            ordered = sorted(
                per_client[client_key],
                key=lambda m: str(m.get('created_at') or ''), reverse=True,
            )
            take    = ordered[:max_per_client]
            overflow = len(ordered) - len(take)
            if overflow:
                meetings_deferred += overflow
                outcome['note'] = f'{overflow} older call(s) deferred to the next run'

            # Transcripts (heavy). A failure is non-fatal: the Fathom summary
            # alone still carries useful signal.
            with ThreadPoolExecutor(max_workers=_TRANSCRIPT_WORKERS) as pool:
                fetched = list(pool.map(
                    lambda m: _safe_transcript(fathom_key, m), take
                ))

            blocks: list[dict] = []
            used_chars = 0
            processed: list[tuple[dict, list[dict]]] = []
            for meeting, transcript in zip(take, fetched):
                block = fc.meeting_block(meeting, transcript, tr_cap)
                cost  = len(block['transcript']) + len(block['fathom_summary'])
                if blocks and used_chars + cost > char_cap:
                    meetings_deferred += 1
                    continue
                used_chars += cost
                blocks.append(block)
                processed.append((meeting, transcript))

            if not blocks:
                outcome['note'] = (outcome['note'] + '; no usable call content').strip('; ')
                clients_unchanged += 1
                results.append(outcome)
                continue

            # Prior state for the merge
            current_digest, existing_data = '', {}
            for df in frames.values():
                mask = _group_mask(df, client_key)
                if not mask.any():
                    continue
                row = df.loc[mask].iloc[0]
                current_digest = str(row.get('client_meetings_summary') or '')
                try:
                    existing_data = json.loads(str(row.get('client_meetings_data') or '{}'))
                except (ValueError, TypeError):
                    existing_data = {}
                break

            merged = _claude_json(anth, model, _MERGE_SYSTEM, {
                'company_name':           company_name,
                'current_digest':         current_digest,
                'existing_meetings_data': (existing_data or {}).get('extracted') or {},
                'new_meetings':           blocks,
            })
            no_change = bool(merged.get('no_meaningful_change'))
            digest    = str(merged.get('meetings_digest') or '').strip()
            extracted = merged.get('extracted') or {}

            # Store the raw calls regardless — they were fetched and paid for.
            prior_list = [m for m in ((existing_data or {}).get('meetings') or [])
                          if isinstance(m, dict)]
            seen_ids   = {str(m.get('recording_id')) for m in prior_list}
            new_list: list[dict] = []
            for meeting, transcript in processed:
                rid       = str(meeting.get('recording_id') or '')
                blob_path = f'{_MEETINGS_PREFIX}{rid}.json'
                if not dry_run:
                    _save_json_blob(gcs, blob_path,
                                    _raw_meeting_doc(meeting, transcript, client_key, run_id))
                    index_rows.append(_index_row(meeting, transcript, client_key,
                                                 company_name, blob_path, run_id))
                meetings_ingested += 1
                if rid not in seen_ids:
                    new_list.append({
                        'recording_id': rid,
                        'title':        fc.meeting_title(meeting),
                        'date':         fc.meeting_date(meeting),
                        'url':          str(meeting.get('url') or ''),
                    })

            meetings_list = (new_list + prior_list)[:_MAX_LISTED]
            if no_change or not digest:
                # Nothing new about the company — keep the established profile
                # material so the aspect-profile fingerprint does not churn.
                digest    = current_digest
                extracted = (existing_data or {}).get('extracted') or extracted
                outcome['outcome'] = 'unchanged'
            else:
                outcome['outcome'] = 'updated'

            # A dry run writes nothing, so without this the digest it just paid
            # Claude to produce is unobservable — "the pipeline ran" and "the
            # extraction is any good" would need two separate runs to answer.
            if dry_run and len(dry_previews) < _MAX_DRY_PREVIEWS:
                dry_previews.append({
                    'company_name':     company_name,
                    'meetings':         [f"{b['date']} {b['title']}" for b in blocks],
                    'digest':           digest[:_DRY_PREVIEW_CHARS],
                    'extracted_counts': {
                        k: (len(v) if isinstance(v, list) else (1 if str(v).strip() else 0))
                        for k, v in (extracted or {}).items()
                    },
                    'notable_updates':  [
                        str(u)[:300] for u in ((extracted or {}).get('notable_updates') or [])[:5]
                    ],
                })

            meetings_data = json.dumps({
                'extracted': extracted,
                'meetings':  meetings_list,
                'last_run':  run_id,
            }, ensure_ascii=False)

            found = False
            for blob_name, df in frames.items():
                mask = _group_mask(df, client_key)
                if not mask.any():
                    continue
                found = True
                df.loc[mask, 'client_meetings_data']    = meetings_data
                df.loc[mask, 'client_meetings_summary'] = digest
                df.loc[mask, 'meetings_updated_at']     = today
                touched_blobs.add(blob_name)
            if not found:
                outcome['outcome'] = 'error'
                outcome['note']    = 'no client rows matched client_key'
                clients_errored += 1
                results.append(outcome)
                continue

            # Mark synced only after the columns are written.
            for meeting, _transcript in processed:
                rid = str(meeting.get('recording_id') or '')
                if rid:
                    state_meetings[rid] = {
                        'client_key': client_key,
                        'created_at': str(meeting.get('created_at') or ''),
                        'synced_at':  today,
                    }
            state_clients[client_key] = today

            outcome['meetings_processed'] = len(processed)
            if outcome['outcome'] == 'updated':
                clients_updated += 1
            else:
                clients_unchanged += 1
            print(f'  [{n}/{total}] {company_name}: {outcome["outcome"]} '
                  f'({len(processed)} calls)', flush=True)

        except fc.RateLimitStalledError as e:
            stopped_early = 'rate_limit'
            outcome['outcome'] = 'deferred'
            outcome['note']    = f'Fathom rate limit: {e}'
            results.append(outcome)
            print(f'  ABORT {company_name}: {e}', flush=True)
            break
        except Exception as e:
            clients_errored += 1
            outcome['outcome'] = 'error'
            outcome['note']    = f'{type(e).__name__}: {e}'
            print(f'  ERROR {company_name}: {e}', flush=True)

        results.append(outcome)
        if n % _CHECKPOINT_EVERY == 0:
            _checkpoint('running', n, total)

    _checkpoint('running', total, total)

    _write_status(gcs, run_id, {
        'run_id': run_id, 'state': 'complete', 'dry_run': dry_run,
        'stopped_early': stopped_early, 'note': sweep_note,
        'created_after': created_after,
        'meetings_swept': swept, 'meetings_new': new_attributed,
        'meetings_ingested': meetings_ingested,
        'meetings_deferred': meetings_deferred,
        'clients_total': total, 'clients_done': total,
        'clients_updated': clients_updated,
        'clients_unchanged': clients_unchanged,
        'clients_errored': clients_errored,
        'api_calls_used': fc.call_count(),
        'results': results[:_MAX_LISTED],
        'unmatched_domains': sorted(
            unmatched.values(), key=lambda e: -e['meeting_count']
        )[:_MAX_LISTED],
        'model': model,
        'dry_run_preview': dry_previews,
        'error': None,
    })
    print(f'[{run_id}] complete — {clients_updated} updated, '
          f'{clients_unchanged} unchanged, {clients_errored} errored, '
          f'{meetings_ingested} calls ingested, {fc.call_count()} Fathom calls',
          flush=True)


def _safe_transcript(api_key: str, meeting: dict) -> list[dict]:
    """Transcript for one meeting; [] on anything but a rate-limit stall, which
    must propagate so the whole sweep stops instead of silently degrading every
    remaining client to summary-only material."""
    rid = meeting.get('recording_id')
    if not rid:
        return []
    try:
        return fc.get_transcript(api_key, rid)
    except fc.RateLimitStalledError:
        raise
    except Exception as e:
        print(f'  WARN transcript {rid} failed: {e}', flush=True)
        return []


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python jobs/fathom_sync_job.py <config_blob_path>', flush=True)
        sys.exit(2)
    cfg_path = sys.argv[1]
    run_id_fallback = cfg_path.split('/')[-1].replace('.json', '')
    try:
        main(cfg_path)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        try:
            client = _gcs()
            cfg    = _load_json_blob(client, cfg_path) or {}
            _write_status(client, cfg.get('run_id') or run_id_fallback, {
                'run_id': cfg.get('run_id') or run_id_fallback,
                'state': 'error', 'error': tb,
            })
        except Exception as inner:
            print(f'Could not write error status: {inner}', flush=True)
        sys.exit(1)
