"""
Tier-2 trigger guardrails for the Matcher MCP server (Phase 5.5)
---------------------------------------------------------------
The five trigger tools — run_matching, build_profiles, sync_drive, sync_fathom,
run_onboarding_agent — let a consultant start a Cloud Run job from claude.ai.
Unlike the read-only Tier-1 tools, a trigger spends real money, so every one goes
through a two-step confirmation enforced here in code, not in the prompt:

  1. estimate(...) validates the arguments and returns {estimate, confirm_token}
     WITHOUT doing anything. The token is an HMAC over (tool, normalized args,
     email, expiry) with a 10-minute TTL — it binds the confirmation to exactly
     the request that was estimated and to the caller who estimated it.
  2. trigger(..., confirm_token) re-derives the token from the arguments it is
     handed and compares it. Only then does it check the caps, write the job
     config and run the job, returning the run_id.

Three caps that no token can bypass, checked at trigger time:
  • a per-tool unit limit (e.g. build_profiles <= 25 companies);
  • a per-user daily spend cap (default $25, admin-config/mcp_limits.json),
    tracked in a generation-guarded daily ledger (mcp-spend/{date}.json);
  • the job run lock — a job already running returns {already_running, run_id}
    instead of starting a duplicate.

The cost/time figures are deliberately ROUGH and labelled as such: they bound
spend and set expectations, they are not billing. Actual cost is reported by each
job in its own status.json. This module is transport-agnostic, Streamlit-free and
unit-tested with the GCS fake; the MCP server (services/matcher_mcp/main.py) is a
thin wrapper that injects the OAuth-verified email and the HMAC secret.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from google.api_core.exceptions import NotFound, PreconditionFailed

from src.modules import gcs_write as gw, job_control as jc
from src.modules.tools import runs as t_runs

BUCKET = 'cc-matcher-bucket-jeg-v1'

TOKEN_TTL_S = 600  # 10 minutes, per the plan (§5.5)

_LIMITS_BLOB = 'admin-config/mcp_limits.json'
_SPEND_PREFIX = 'mcp-spend/'
_SPEND_RETRIES = 6
_TOKENS_KEY = '__tokens__'     # reserved ledger key (never a valid email) that
_MAX_LEDGER_TOKENS = 500       # records the confirm tokens already committed today

DEFAULT_DAILY_CAP_USD = 25.0


class SpendCapExceeded(RuntimeError):
    """Committing this charge would push the user past their daily cap. Raised
    inside the generation-guarded write, so a cap check made against a read taken
    before a concurrent commit cannot be out-raced."""
    def __init__(self, new_total: float):
        super().__init__(f'daily spend cap would be exceeded (${new_total:.2f})')
        self.new_total = new_total


class TokenReplayed(RuntimeError):
    """This confirm token was already committed today — a replay. The first
    trigger already started the run and charged its spend."""

_SYNC_MODEL = 'claude-sonnet-4-6'  # mirrors drive_sync / fathom_sync views


# ── HMAC confirm token ────────────────────────────────────────────────────────

def _normalize(tool: str, args: dict) -> str:
    """The canonical string a token signs: the tool name plus its validated
    arguments as sorted-key JSON. Two logically identical requests normalise to
    the same string (lists are sorted/deduped by the resolvers before they get
    here), so the confirm call re-derives the same token from the same args."""
    return json.dumps({'tool': tool, 'args': args}, sort_keys=True, default=str)


def make_token(tool: str, norm: str, email: str, secret: str, *,
               now: float | None = None, ttl: int = TOKEN_TTL_S) -> str:
    """`{expiry}.{hmac}` — expiry in plaintext (so verify can reject a stale
    token without server state), the HMAC covering tool+args+email+expiry."""
    expiry = int((now if now is not None else time.time())) + ttl
    msg = f'{tool}|{norm}|{email.lower()}|{expiry}'.encode('utf-8')
    sig = hmac.new(secret.encode('utf-8'), msg, hashlib.sha256).hexdigest()
    return f'{expiry}.{sig}'


def verify_token(token: str, tool: str, norm: str, email: str, secret: str, *,
                 now: float | None = None) -> tuple[bool, str]:
    """(ok, reason). Fails closed on a malformed, expired, tampered or
    wrong-arguments token. Uses a constant-time compare."""
    now = now if now is not None else time.time()
    try:
        exp_str, sig = str(token).split('.', 1)
        expiry = int(exp_str)
    except Exception:
        return False, 'malformed confirm_token'
    if now > expiry:
        return False, ('confirm_token expired — call again without confirm_token '
                       'to get a fresh estimate and token')
    msg = f'{tool}|{norm}|{email.lower()}|{expiry}'.encode('utf-8')
    good = hmac.new(secret.encode('utf-8'), msg, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(good, sig):
        return False, ('confirm_token does not match these arguments — the '
                       'request changed, or the token is for a different caller')
    return True, ''


# ── spend ledger + limits ─────────────────────────────────────────────────────

def _today() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')


def load_limits(gcs_client, *, bucket: str = BUCKET) -> dict:
    """The spend caps. admin-config/mcp_limits.json may set `default_daily_cap_usd`,
    a `user_daily_cap_usd` map of per-user overrides, and a `unit_caps` map of
    per-tool unit-limit overrides. A missing/garbled file yields the baked-in
    defaults (never no cap)."""
    out = {'default_daily_cap_usd': DEFAULT_DAILY_CAP_USD,
           'user_daily_cap_usd': {}, 'unit_caps': {}}
    try:
        raw = gcs_client.bucket(bucket).blob(_LIMITS_BLOB).download_as_bytes()
        doc = json.loads(raw)
    except Exception:
        return out
    if isinstance(doc.get('default_daily_cap_usd'), (int, float)):
        out['default_daily_cap_usd'] = float(doc['default_daily_cap_usd'])
    if isinstance(doc.get('user_daily_cap_usd'), dict):
        out['user_daily_cap_usd'] = {str(k).lower(): float(v)
                                     for k, v in doc['user_daily_cap_usd'].items()
                                     if isinstance(v, (int, float))}
    if isinstance(doc.get('unit_caps'), dict):
        out['unit_caps'] = {str(k): int(v) for k, v in doc['unit_caps'].items()
                            if isinstance(v, (int, float))}
    return out


def daily_cap_for(email: str, limits: dict) -> float:
    return float(limits.get('user_daily_cap_usd', {}).get(
        email.lower(), limits.get('default_daily_cap_usd', DEFAULT_DAILY_CAP_USD)))


def today_spend(gcs_client, email: str, *, bucket: str = BUCKET) -> float:
    """What this email has already committed in triggers today (UTC). 0 if the
    ledger does not exist yet."""
    blob = gcs_client.bucket(bucket).blob(f'{_SPEND_PREFIX}{_today()}.json')
    try:
        doc = json.loads(blob.download_as_bytes())
    except Exception:
        return 0.0
    return float(doc.get(email.lower(), 0.0) or 0.0)


def add_spend(gcs_client, email: str, amount: float, *, cap: float | None = None,
              token_sig: str | None = None, bucket: str = BUCKET) -> float:
    """Add `amount` to this email's committed spend for today — and, in the SAME
    generation-guarded write, enforce the cap and reject a replayed token, so
    neither can be out-raced by a concurrent trigger. Returns the new total.

    Two concurrent triggers never lose an increment (that is what the generation
    guard gave us before); on top of that:
      • `cap` (when given): if adding `amount` would exceed it, SpendCapExceeded
        is raised and NOTHING is written. Because the check is re-evaluated after
        every 412-retry against the freshly re-read total, a commit that another
        trigger landed between our read and our write is counted — closing the
        read-then-commit TOCTOU in trigger().
      • `token_sig` (when given): if it is already in today's ledger the token is
        a replay — TokenReplayed is raised before any charge, so a confirm token
        reused inside its TTL can neither double-charge nor double-trigger. The
        winner of a same-token race records the sig; the loser 412-retries, now
        sees it recorded, and raises.
    Raises ConcurrencyError if the guarded write cannot land within the retry
    budget — the caller must NOT trigger if spend cannot be recorded."""
    blob = gcs_client.bucket(bucket).blob(f'{_SPEND_PREFIX}{_today()}.json')
    key = email.lower()
    for _ in range(_SPEND_RETRIES):
        try:
            cur = json.loads(blob.download_as_bytes())
            gen = blob.generation
            if not isinstance(cur, dict):
                cur, gen = {}, 0
        except NotFound:
            cur, gen = {}, 0
        except Exception:
            cur, gen = {}, 0
        consumed = cur.get(_TOKENS_KEY)
        consumed = list(consumed) if isinstance(consumed, list) else []
        if token_sig and token_sig in consumed:
            raise TokenReplayed('this confirmation was already used today')
        new_total = round(float(cur.get(key, 0.0) or 0.0) + float(amount), 6)
        if cap is not None and new_total > cap:
            raise SpendCapExceeded(new_total)
        cur[key] = new_total
        if token_sig:
            cur[_TOKENS_KEY] = (consumed + [token_sig])[-_MAX_LEDGER_TOKENS:]
        try:
            blob.upload_from_string(json.dumps(cur),
                                    content_type='application/json',
                                    if_generation_match=gen)
            return new_total
        except PreconditionFailed:
            continue
    # Could not commit the ledger after retries — surface it; the caller must
    # NOT trigger if spend cannot be recorded (that would uncap the cap).
    raise gw.ConcurrencyError('could not update the MCP spend ledger')


# ── run-lock / already-running check ──────────────────────────────────────────

def running_run_id(gcs_client, job: str, *, bucket: str = BUCKET) -> str | None:
    """The run_id of an in-flight run of `job`, or None. Checks the job run lock
    first, then the newest status.json if its state reads as running. matching-job
    and the two sync jobs (drive-sync, fathom-sync) hold a run lock while executing
    and are seen by the lock check; client-profile-job and drive-onboard-job do NOT
    take a lock, so for those the status.json fallback is the only in-flight signal
    — keep it. The single-use confirm token closes the remaining
    replay-after-completion gap that a run_id or a status check alone could not."""
    lock = gcs_client.bucket(bucket).blob(gw.lock_blob(job))
    try:
        rec = json.loads(lock.download_as_bytes())
        if not gw._takeable(rec, datetime.now(timezone.utc)):
            return str(rec.get('run_id') or 'unknown')
    except Exception:
        pass
    prefix = t_runs.STATUS_PREFIXES.get(job)
    if prefix:
        ids = t_runs._run_ids(gcs_client, prefix, bucket=bucket)
        if ids:
            st = jc.poll_status(gcs_client, prefix, ids[0], bucket=bucket) or {}
            if str(st.get('state', '')).lower() in ('running', 'started'):
                return ids[0]
    return None


# ── per-tool specs ────────────────────────────────────────────────────────────
# Each resolver validates the raw args and returns (norm_args, units, extra,
# error): norm_args is what the token signs (JSON-stable), units is what the cap
# and the estimate count, extra carries anything build_config needs that is not
# in norm_args (resolved folder ids, the drive id). An error string short-circuits
# with no token and no side effect.

@dataclass(frozen=True)
class Spec:
    job: str
    status_prefix: str
    cfg_prefix: str
    run_id_prefix: str
    unit_cap: int
    cost_per_unit: float
    base_cost: float
    minutes_per_unit: float
    base_minutes: float
    unit_noun: str


def _as_list(v) -> list[str]:
    if isinstance(v, str):
        v = [p.strip() for p in v.split(',')]
    elif not isinstance(v, (list, tuple)):
        return []
    seen, out = set(), []
    for x in v:
        s = str(x).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _load_drive_assignments(gcs_client, bucket: str) -> dict:
    try:
        raw = gcs_client.bucket(bucket).blob(
            'drive-sync-configs/assignments.json').download_as_bytes()
        return json.loads(raw)
    except Exception:
        return {}


# run_matching ─────────────────────────────────────────────────────────────────

def _resolve_matching(gcs_client, args: dict, bucket: str):
    agencies = _as_list(args.get('agencies'))
    sources = _as_list(args.get('sources'))
    if not agencies:
        return None, 0, {}, 'at least one agency is required (e.g. ["DOD","HHS"])'
    if not sources:
        return None, 0, {}, 'at least one contact source is required (e.g. ["apollo"])'
    norm = {
        'agencies': sorted(agencies),
        'sources': sorted(sources),
        'threshold': round(float(args.get('threshold', 0.82)), 4),
        'top_k': int(args.get('top_k', 5)),
        'ai_validation': bool(args.get('ai_validation', True)),
        'prewrite_email': bool(args.get('prewrite_email', False)),
    }
    return norm, 1, {}, None


def _config_matching(norm: dict, extra: dict, run_id: str) -> dict:
    return {
        'run_id': run_id,
        'threshold': norm['threshold'],
        'top_k': norm['top_k'],
        'sources': norm['sources'],
        'agencies': norm['agencies'],
        'topic_filters': [],
        'ai_validation': norm['ai_validation'],
        'prewrite_email': norm['prewrite_email'],
        # {} not None: matching_job does config.get('email_config', {}), which
        # returns the value when the key is present, so None would reach
        # email_config.get(...) and crash the run when prewrite_email is on.
        'email_config': {},
    }


def _run_id_matching(norm: dict) -> str:
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    ag = '-'.join(norm['agencies']) or 'none'
    src = '-'.join(norm['sources']) or 'none'
    return f'{ts}_ag-{ag}_src-{src}'


# build_profiles ───────────────────────────────────────────────────────────────

def _resolve_build_profiles(gcs_client, args: dict, bucket: str):
    keys = _as_list(args.get('company_keys'))
    if not keys:
        return None, 0, {}, 'company_keys is required (each "Name||https://site")'
    from src.modules import aspect_profile as ap
    pool = str(args.get('pool', 'clients') or 'clients')
    if pool not in ('clients', 'prospects'):
        return None, 0, {}, "pool must be 'clients' or 'prospects'"
    sources = _as_list(args.get('sources')) or [
        s['key'] for s in ap.SOURCES if s.get('default')]
    norm = {
        'company_keys': sorted(keys),
        'pool': pool,
        'sources': sorted(sources),
        'target_aspects': int(args.get('target_aspects', 4)),
        'max_markets': int(args.get('max_markets', ap.MAX_MARKETS)),
        'assess_defense': bool(args.get('assess_defense', True)),
        'assess_unexplored': bool(args.get('assess_unexplored', True)),
        'max_unexplored': int(args.get('max_unexplored', ap.MAX_UNEXPLORED)),
        'model': str(args.get('model', _SYNC_MODEL) or _SYNC_MODEL),
    }
    return norm, len(keys), {}, None


def _config_build_profiles(norm: dict, extra: dict, run_id: str) -> dict:
    from src.modules import aspect_profile as ap
    return {
        'run_id': run_id,
        'pool': norm['pool'],
        'company_keys': norm['company_keys'],
        'sources': norm['sources'],
        'target_aspects': norm['target_aspects'],
        'max_markets': norm['max_markets'],
        'assess_defense': norm['assess_defense'],
        'assess_unexplored': norm['assess_unexplored'],
        'max_unexplored': norm['max_unexplored'],
        'market_merge_threshold': ap.MARKET_MERGE_THRESHOLD,
        'aspect_merge_threshold': ap.ASPECT_MERGE_THRESHOLD,
        'model': norm['model'],
        'concurrency': 4,
        'dry_run': False,
    }


# sync_drive ───────────────────────────────────────────────────────────────────

def _resolve_sync_drive(gcs_client, args: dict, bucket: str):
    keys = set(_as_list(args.get('company_keys')))
    if not keys:
        return None, 0, {}, 'company_keys is required'
    doc = _load_drive_assignments(gcs_client, bucket)
    drive_id = str(doc.get('drive_id', '') or '')
    if not drive_id:
        return None, 0, {}, ('no shared drive is configured — set it in the Drive '
                             'Sync view first')
    assignments = doc.get('assignments', {})
    folder_ids = sorted(fid for fid, a in assignments.items()
                        if a.get('client_key') in keys)
    resolved_keys = sorted({a['client_key'] for a in assignments.values()
                            if a.get('client_key') in keys})
    if not folder_ids:
        return None, 0, {}, ('none of those companies has an assigned Drive folder '
                             '— assign one in the Drive Sync view first')
    norm = {
        'company_keys': resolved_keys,
        'full_resync': bool(args.get('full_resync', False)),
        'dry_run': bool(args.get('dry_run', False)),
    }
    extra = {'drive_id': drive_id, 'folder_ids': folder_ids}
    return norm, len(resolved_keys), extra, None


def _config_sync_drive(norm: dict, extra: dict, run_id: str) -> dict:
    return {
        'run_id': run_id,
        'drive_id': extra['drive_id'],
        'folder_ids': extra['folder_ids'],
        'new_client_folder_ids': [],
        'full_resync': norm['full_resync'],
        'dry_run': norm['dry_run'],
        'max_docs_per_client': 40,
        'per_client_char_cap': 150000,
        'max_proposals': 1,
        'task_timeout_s': 14400,
        'model': _SYNC_MODEL,
    }


# sync_fathom ──────────────────────────────────────────────────────────────────

def _resolve_sync_fathom(gcs_client, args: dict, bucket: str):
    keys = _as_list(args.get('company_keys'))
    if not keys:
        return None, 0, {}, 'company_keys is required'
    norm = {
        'company_keys': sorted(keys),
        'lookback_days': int(args.get('lookback_days', 90)),
        'full_resync': bool(args.get('full_resync', False)),
        'dry_run': bool(args.get('dry_run', False)),
    }
    return norm, len(keys), {}, None


def _config_sync_fathom(norm: dict, extra: dict, run_id: str) -> dict:
    return {
        'run_id': run_id,
        'client_keys': norm['company_keys'],
        'lookback_days': norm['lookback_days'],
        'created_after': None,
        'full_resync': norm['full_resync'],
        'dry_run': norm['dry_run'],
        'max_meetings': 400,
        'max_meetings_per_client': 12,
        'per_client_char_cap': 120000,
        'transcript_char_cap': 40000,
        'task_timeout_s': 14400,
        'model': _SYNC_MODEL,
    }


# run_onboarding_agent ─────────────────────────────────────────────────────────

def _resolve_onboarding(gcs_client, args: dict, bucket: str):
    fids = _as_list(args.get('folder_ids'))
    if not fids:
        return None, 0, {}, 'folder_ids is required (unmatched Drive folder ids)'
    doc = _load_drive_assignments(gcs_client, bucket)
    drive_id = str(doc.get('drive_id', '') or '')
    if not drive_id:
        return None, 0, {}, 'no shared drive is configured'
    norm = {'folder_ids': sorted(fids), 'dry_run': bool(args.get('dry_run', False))}
    return norm, len(fids), {'drive_id': drive_id}, None


def _config_onboarding(norm: dict, extra: dict, run_id: str) -> dict:
    return {
        'run_id': run_id,
        'select': 'ids',
        'folder_ids': norm['folder_ids'],
        'drive_id': extra['drive_id'],
        'max_folders': len(norm['folder_ids']),
        'dry_run': norm['dry_run'],
    }


_SPECS = {
    'run_matching': Spec(
        job='matching-job', status_prefix='matching-results/',
        cfg_prefix='job-configs/', run_id_prefix='',
        unit_cap=1, cost_per_unit=0.0, base_cost=2.0,
        minutes_per_unit=0.0, base_minutes=8.0, unit_noun='run'),
    'build_profiles': Spec(
        job='client-profile-job', status_prefix='client-profile-jobs/',
        cfg_prefix='client-profile-configs/', run_id_prefix='client_profile',
        unit_cap=25, cost_per_unit=0.18, base_cost=0.0,
        minutes_per_unit=1.5, base_minutes=1.0, unit_noun='company'),
    'sync_drive': Spec(
        job='drive-sync-job', status_prefix='drive-sync-jobs/',
        cfg_prefix='drive-sync-configs/', run_id_prefix='drive_sync',
        unit_cap=50, cost_per_unit=0.06, base_cost=0.0,
        minutes_per_unit=1.2, base_minutes=2.0, unit_noun='company'),
    'sync_fathom': Spec(
        job='fathom-sync-job', status_prefix='fathom-jobs/',
        cfg_prefix='fathom-configs/', run_id_prefix='fathom',
        unit_cap=50, cost_per_unit=0.06, base_cost=0.0,
        minutes_per_unit=1.2, base_minutes=5.0, unit_noun='company'),
    'run_onboarding_agent': Spec(
        job='drive-onboard-job', status_prefix='drive-onboard-jobs/',
        cfg_prefix='drive-onboard-configs/', run_id_prefix='drive_onboard',
        unit_cap=10, cost_per_unit=0.28, base_cost=0.0,
        minutes_per_unit=6.0, base_minutes=1.0, unit_noun='folder'),
}

_RESOLVERS = {
    'run_matching': _resolve_matching,
    'build_profiles': _resolve_build_profiles,
    'sync_drive': _resolve_sync_drive,
    'sync_fathom': _resolve_sync_fathom,
    'run_onboarding_agent': _resolve_onboarding,
}
_CONFIGURERS = {
    'run_matching': _config_matching,
    'build_profiles': _config_build_profiles,
    'sync_drive': _config_sync_drive,
    'sync_fathom': _config_sync_fathom,
    'run_onboarding_agent': _config_onboarding,
}

TOOLS = tuple(_SPECS)


def _unit_cap(tool: str, limits: dict) -> int:
    return int(limits.get('unit_caps', {}).get(tool, _SPECS[tool].unit_cap))


def _estimate_cost_minutes(spec: Spec, units: int) -> tuple[float, int]:
    cost = round(spec.base_cost + spec.cost_per_unit * units, 2)
    minutes = int(round(spec.base_minutes + spec.minutes_per_unit * units))
    return cost, minutes


# ── public: estimate / trigger ────────────────────────────────────────────────

def estimate(gcs_client, tool: str, args: dict, email: str, *, secret: str,
             now: float | None = None, bucket: str = BUCKET) -> dict:
    """Step 1. Validate `args`, return {estimate:{units, est_cost_usd,
    est_minutes}, confirm_token, ...} with NO side effect, or {error}. The token
    is minted here and must be passed back to trigger() unchanged along with the
    same args."""
    if tool not in _SPECS:
        return {'error': f'unknown trigger tool {tool!r}'}
    spec = _SPECS[tool]
    try:
        norm, units, extra, err = _RESOLVERS[tool](gcs_client, args or {}, bucket)
    except Exception as e:  # noqa: BLE001
        return {'error': f'could not validate arguments: {type(e).__name__}: {e}'}
    if err:
        return {'error': err}

    limits = load_limits(gcs_client, bucket=bucket)
    cap = _unit_cap(tool, limits)
    if units > cap:
        return {'error': f'{units} {spec.unit_noun}s exceeds the per-call limit of '
                         f'{cap} for {tool}; split it into smaller calls'}

    cost, minutes = _estimate_cost_minutes(spec, units)
    daily_cap = daily_cap_for(email, limits)
    spent = today_spend(gcs_client, email, bucket=bucket)
    norm_str = _normalize(tool, norm)
    token = make_token(tool, norm_str, email, secret, now=now)

    out = {
        'tool': tool,
        'estimate': {'units': units, 'unit': spec.unit_noun,
                     'est_cost_usd': cost, 'est_minutes': minutes,
                     'rough': True},
        'spend_today_usd': round(spent, 2),
        'daily_cap_usd': round(daily_cap, 2),
        'confirm_token': token,
        'expires_in_s': TOKEN_TTL_S,
        'next': ('Call this tool again with the SAME arguments plus confirm_token '
                 'to start the run.'),
    }
    if spent + cost > daily_cap:
        out['warning'] = (f'this would reach ${round(spent + cost, 2)} of your '
                          f'${round(daily_cap, 2)} daily cap and will be refused at '
                          'confirm; ask an admin to raise mcp_limits.json')
    return out


def trigger(gcs_client, tool: str, args: dict, email: str, confirm_token: str, *,
            secret: str, jobs_client=None, credentials=None,
            now: float | None = None, bucket: str = BUCKET) -> dict:
    """Step 2. Re-derive and verify the token against `args`, enforce the unit cap
    and the daily spend cap, refuse if the job is already running, then write the
    config and run the job. Returns {run_id, job, ...}, {already_running, run_id}
    or {error}. The spend is committed BEFORE the job is triggered, so a crash
    after trigger can only over-count, never let a run escape the cap."""
    if tool not in _SPECS:
        return {'error': f'unknown trigger tool {tool!r}'}
    spec = _SPECS[tool]
    try:
        norm, units, extra, err = _RESOLVERS[tool](gcs_client, args or {}, bucket)
    except Exception as e:  # noqa: BLE001
        return {'error': f'could not validate arguments: {type(e).__name__}: {e}'}
    if err:
        return {'error': err}

    norm_str = _normalize(tool, norm)
    ok, reason = verify_token(confirm_token or '', tool, norm_str, email, secret,
                              now=now)
    if not ok:
        return {'error': reason}

    limits = load_limits(gcs_client, bucket=bucket)
    cap = _unit_cap(tool, limits)
    if units > cap:
        return {'error': f'{units} {spec.unit_noun}s exceeds the per-call limit of '
                         f'{cap} for {tool}'}

    cost, minutes = _estimate_cost_minutes(spec, units)
    daily_cap = daily_cap_for(email, limits)
    # Advisory fast-fail on a clearly-over-cap request (friendly message, no write
    # attempt). The race-safe enforcement is add_spend(cap=…) below.
    spent = today_spend(gcs_client, email, bucket=bucket)
    if spent + cost > daily_cap:
        return {'error': f'daily spend cap reached: ${round(spent, 2)} already '
                         f'committed today + ${cost} estimate exceeds your '
                         f'${round(daily_cap, 2)} cap. Try again tomorrow or ask '
                         'an admin to raise mcp_limits.json'}

    busy = running_run_id(gcs_client, spec.job, bucket=bucket)
    if busy:
        return {'already_running': True, 'job': spec.job, 'run_id': busy,
                'note': f'{spec.job} is already running (run {busy}); not starting '
                        'a duplicate'}

    # Commit spend first — if the ledger write fails we must not trigger. The cap
    # and the token's single use are enforced inside this one guarded write, so a
    # concurrent confirm cannot slip past the cap and a replayed token cannot
    # double-charge or double-trigger.
    token_sig = hashlib.sha256((confirm_token or '').encode('utf-8')).hexdigest()
    try:
        new_total = add_spend(gcs_client, email, cost, cap=daily_cap,
                              token_sig=token_sig, bucket=bucket)
    except SpendCapExceeded:
        return {'error': f'daily spend cap reached: a concurrent run used the '
                         f'remaining budget, so ${cost} would exceed your '
                         f'${round(daily_cap, 2)} cap. Try again tomorrow or ask '
                         'an admin to raise mcp_limits.json'}
    except TokenReplayed:
        return {'error': 'this confirmation was already used — the run was already '
                         'started. Poll its status, or call again without '
                         'confirm_token for a fresh estimate.'}
    except Exception as e:  # noqa: BLE001
        return {'error': f'could not record spend, not triggering: {e}'}

    run_id = (_run_id_matching(norm) if tool == 'run_matching'
              else f'{spec.run_id_prefix}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    config = _CONFIGURERS[tool](norm, extra, run_id)
    try:
        jc.write_job_config(gcs_client, spec.cfg_prefix, config, bucket=bucket)
        jc.trigger_job(credentials, spec.job, f'{spec.cfg_prefix}{run_id}.json',
                       jobs_client=jobs_client)
    except Exception as e:  # noqa: BLE001
        return {'error': f'triggered spend but the job failed to start: '
                         f'{type(e).__name__}: {e}', 'run_id': run_id,
                'spend_today_usd': round(new_total, 2)}

    return {
        'triggered': True, 'job': spec.job, 'run_id': run_id,
        'units': units, 'est_cost_usd': cost, 'est_minutes': minutes,
        'spend_today_usd': round(new_total, 2), 'daily_cap_usd': round(daily_cap, 2),
        'status_prefix': spec.status_prefix,
        'note': f'started {spec.job}; poll with get_run_status("{spec.job}", '
                f'"{run_id}")',
    }
