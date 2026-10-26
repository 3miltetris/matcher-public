"""Unit tests for the Tier-2 trigger guardrails (Phase 5.5).

Everything runs against the in-memory GCS fake and a fake Cloud Run jobs client
that records run_job calls — no network, no real triggers. Covers token
make/verify (expiry, tamper, wrong caller), per-call unit caps, the per-user
daily spend cap, the already-running guard, argument normalization and the
config each tool writes.
"""

import json
import time

import pytest

from src.modules import gcs_write as gw
from src.modules.tools import triggers as tg
from tests.mcp.conftest import put_json

SECRET = 'test-hmac-secret'
EMAIL = 'john@bwcoconsulting.com'


class FakeJobs:
    """Records jc.trigger_job's run_job(request=...) calls instead of running."""
    def __init__(self):
        self.calls = []

    def run_job(self, request=None):
        self.calls.append(request)
        return object()


def _drive_assignments(storage, *, drive_id='0ADRIVE', keys=None):
    keys = keys or {}
    assignments = {f'fld_{k}': {'client_key': ck, 'folder_name': f'{k}_INTERNAL'}
                   for k, ck in keys.items()}
    put_json(storage, 'drive-sync-configs/assignments.json',
             {'drive_id': drive_id, 'assignments': assignments})


# ── token ─────────────────────────────────────────────────────────────────────

def test_token_roundtrip_and_expiry():
    norm = tg._normalize('build_profiles', {'company_keys': ['a', 'b']})
    tok = tg.make_token('build_profiles', norm, EMAIL, SECRET, now=1000)
    ok, _ = tg.verify_token(tok, 'build_profiles', norm, EMAIL, SECRET, now=1100)
    assert ok
    # expired (now past the 600s TTL)
    bad, reason = tg.verify_token(tok, 'build_profiles', norm, EMAIL, SECRET,
                                  now=1000 + tg.TOKEN_TTL_S + 1)
    assert not bad and 'expired' in reason


def test_token_tamper_and_wrong_caller():
    norm = tg._normalize('build_profiles', {'company_keys': ['a']})
    tok = tg.make_token('build_profiles', norm, EMAIL, SECRET, now=1000)
    # changed arguments → different norm → rejected
    norm2 = tg._normalize('build_profiles', {'company_keys': ['a', 'evil']})
    assert not tg.verify_token(tok, 'build_profiles', norm2, EMAIL, SECRET,
                               now=1100)[0]
    # different caller → rejected
    assert not tg.verify_token(tok, 'build_profiles', norm, 'someone@bwcoconsulting.com',
                               SECRET, now=1100)[0]
    # flipped signature byte → rejected
    assert not tg.verify_token(tok[:-1] + ('0' if tok[-1] != '0' else '1'),
                               'build_profiles', norm, EMAIL, SECRET, now=1100)[0]
    # malformed
    assert not tg.verify_token('garbage', 'build_profiles', norm, EMAIL, SECRET)[0]


# ── estimate ──────────────────────────────────────────────────────────────────

def test_estimate_build_profiles(storage):
    keys = ['Acme||https://acme.com', 'Beta||https://beta.com']
    out = tg.estimate(storage, 'build_profiles', {'company_keys': keys}, EMAIL,
                      secret=SECRET, now=1000)
    assert out['estimate']['units'] == 2 and out['estimate']['unit'] == 'company'
    assert out['estimate']['est_cost_usd'] > 0 and out['confirm_token']
    assert out['daily_cap_usd'] == tg.DEFAULT_DAILY_CAP_USD
    # the token verifies for the same (normalized) args
    norm, units, _e, err = tg._RESOLVERS['build_profiles'](storage,
                                                           {'company_keys': keys}, tg.BUCKET)
    assert err is None
    ok, _ = tg.verify_token(out['confirm_token'], 'build_profiles',
                            tg._normalize('build_profiles', norm), EMAIL, SECRET, now=1001)
    assert ok


def test_estimate_unit_cap(storage):
    keys = [f'C{i}||https://c{i}.com' for i in range(30)]
    out = tg.estimate(storage, 'build_profiles', {'company_keys': keys}, EMAIL,
                      secret=SECRET)
    assert 'error' in out and 'per-call limit of 25' in out['error']
    assert 'confirm_token' not in out


def test_estimate_missing_args(storage):
    assert 'error' in tg.estimate(storage, 'run_matching', {'agencies': []}, EMAIL,
                                  secret=SECRET)
    assert 'error' in tg.estimate(storage, 'build_profiles', {}, EMAIL, secret=SECRET)


def test_estimate_warns_near_cap(storage):
    put_json(storage, 'mcp-spend/' + tg._today() + '.json', {EMAIL: 24.5})
    out = tg.estimate(storage, 'run_matching',
                      {'agencies': ['DOD'], 'sources': ['apollo']}, EMAIL, secret=SECRET)
    # base cost 2.0 pushes 24.5 over the $25 default cap
    assert 'warning' in out and out['confirm_token']


# ── sync_drive resolution ─────────────────────────────────────────────────────

def test_estimate_sync_drive_resolves_folders(storage):
    _drive_assignments(storage, keys={'acme': 'Acme||https://acme.com',
                                       'beta': 'Beta||https://beta.com'})
    out = tg.estimate(storage, 'sync_drive',
                      {'company_keys': ['Acme||https://acme.com']}, EMAIL,
                      secret=SECRET)
    assert out['estimate']['units'] == 1

    # a company with no assigned folder is a clear error, not a silent no-op
    bad = tg.estimate(storage, 'sync_drive',
                      {'company_keys': ['Ghost||https://ghost.com']}, EMAIL,
                      secret=SECRET)
    assert 'error' in bad and 'assigned Drive folder' in bad['error']


# ── trigger ───────────────────────────────────────────────────────────────────

def test_trigger_requires_valid_token(storage):
    args = {'company_keys': ['Acme||https://acme.com']}
    # no token
    assert 'error' in tg.trigger(storage, 'build_profiles', args, EMAIL, '',
                                 secret=SECRET, jobs_client=FakeJobs())


def test_trigger_success_writes_config_and_runs(storage):
    args = {'company_keys': ['Acme||https://acme.com', 'Beta||https://beta.com']}
    est = tg.estimate(storage, 'build_profiles', args, EMAIL, secret=SECRET, now=1000)
    jobs = FakeJobs()
    out = tg.trigger(storage, 'build_profiles', args, EMAIL, est['confirm_token'],
                     secret=SECRET, jobs_client=jobs, now=1001)
    assert out['triggered'] and out['job'] == 'client-profile-job'
    assert out['run_id'].startswith('client_profile_')
    assert len(jobs.calls) == 1
    # config was staged
    cfg_path = f"client-profile-configs/{out['run_id']}.json"
    cfg = json.loads(storage.bucket(tg.BUCKET).blob(cfg_path).download_as_text())
    assert sorted(cfg['company_keys']) == sorted(args['company_keys'])
    assert cfg['pool'] == 'clients' and cfg['dry_run'] is False
    # spend recorded
    assert out['spend_today_usd'] == est['estimate']['est_cost_usd']
    assert tg.today_spend(storage, EMAIL) == est['estimate']['est_cost_usd']


def test_trigger_normalization_order_independent(storage):
    """Estimating with keys in one order and confirming in another still verifies
    — the resolver sorts before signing."""
    est = tg.estimate(storage, 'build_profiles',
                      {'company_keys': ['B||https://b.com', 'A||https://a.com']},
                      EMAIL, secret=SECRET, now=1000)
    jobs = FakeJobs()
    out = tg.trigger(storage, 'build_profiles',
                     {'company_keys': ['A||https://a.com', 'B||https://b.com']},
                     EMAIL, est['confirm_token'], secret=SECRET, jobs_client=jobs,
                     now=1001)
    assert out['triggered'] and len(jobs.calls) == 1


def test_trigger_spend_cap_refuses(storage):
    # cap below the 2-company estimate (0.36) so the confirm is refused
    put_json(storage, 'admin-config/mcp_limits.json', {'default_daily_cap_usd': 0.2})
    args = {'company_keys': ['Acme||https://acme.com', 'Beta||https://beta.com']}
    est = tg.estimate(storage, 'build_profiles', args, EMAIL, secret=SECRET, now=1000)
    jobs = FakeJobs()
    out = tg.trigger(storage, 'build_profiles', args, EMAIL, est['confirm_token'],
                     secret=SECRET, jobs_client=jobs, now=1001)
    assert 'error' in out and 'cap' in out['error'].lower()
    assert len(jobs.calls) == 0            # never triggered
    assert tg.today_spend(storage, EMAIL) == 0.0   # never charged


def test_trigger_already_running_via_lock(storage):
    # a live lock for the job → already_running, no trigger
    bucket = storage.bucket(tg.BUCKET)
    tok = gw.acquire_lock(bucket, 'client-profile-job', 'existing_run', 3600)
    assert tok
    args = {'company_keys': ['Acme||https://acme.com']}
    est = tg.estimate(storage, 'build_profiles', args, EMAIL, secret=SECRET, now=1000)
    jobs = FakeJobs()
    out = tg.trigger(storage, 'build_profiles', args, EMAIL, est['confirm_token'],
                     secret=SECRET, jobs_client=jobs, now=1001)
    assert out.get('already_running') and out['run_id'] == 'existing_run'
    assert len(jobs.calls) == 0


def test_trigger_user_cap_override(storage):
    put_json(storage, 'admin-config/mcp_limits.json',
             {'default_daily_cap_usd': 1.0,
              'user_daily_cap_usd': {EMAIL: 100.0}})
    args = {'company_keys': ['Acme||https://acme.com']}
    est = tg.estimate(storage, 'build_profiles', args, EMAIL, secret=SECRET, now=1000)
    assert est['daily_cap_usd'] == 100.0
    out = tg.trigger(storage, 'build_profiles', args, EMAIL, est['confirm_token'],
                     secret=SECRET, jobs_client=FakeJobs(), now=1001)
    assert out['triggered']


def test_run_onboarding_agent_config(storage):
    _drive_assignments(storage, drive_id='0AONBOARD')
    args = {'folder_ids': ['fldA', 'fldB']}
    est = tg.estimate(storage, 'run_onboarding_agent', args, EMAIL, secret=SECRET,
                      now=1000)
    assert est['estimate']['units'] == 2 and est['estimate']['unit'] == 'folder'
    out = tg.trigger(storage, 'run_onboarding_agent', args, EMAIL,
                     est['confirm_token'], secret=SECRET, jobs_client=FakeJobs(),
                     now=1001)
    cfg = json.loads(storage.bucket(tg.BUCKET).blob(
        f"drive-onboard-configs/{out['run_id']}.json").download_as_text())
    assert cfg['select'] == 'ids' and cfg['drive_id'] == '0AONBOARD'
    assert sorted(cfg['folder_ids']) == ['fldA', 'fldB'] and cfg['max_folders'] == 2


def test_run_matching_config_and_run_id(storage):
    args = {'agencies': ['DOD', 'HHS'], 'sources': ['apollo'], 'top_k': 7}
    est = tg.estimate(storage, 'run_matching', args, EMAIL, secret=SECRET, now=1000)
    out = tg.trigger(storage, 'run_matching', args, EMAIL, est['confirm_token'],
                     secret=SECRET, jobs_client=FakeJobs(), now=1001)
    assert out['triggered'] and 'ag-DOD-HHS' in out['run_id']
    cfg = json.loads(storage.bucket(tg.BUCKET).blob(
        f"job-configs/{out['run_id']}.json").download_as_text())
    assert cfg['top_k'] == 7 and cfg['agencies'] == ['DOD', 'HHS']


def test_add_spend_accumulates(storage):
    assert tg.add_spend(storage, EMAIL, 1.5) == 1.5
    assert tg.add_spend(storage, EMAIL, 2.0) == 3.5
    assert tg.today_spend(storage, EMAIL) == 3.5
    # a different user has their own ledger entry
    assert tg.add_spend(storage, 'other@bwcoconsulting.com', 0.5) == 0.5
    assert tg.today_spend(storage, EMAIL) == 3.5


def test_add_spend_enforces_cap_atomically(storage):
    # the cap is checked inside the guarded write: a charge that would exceed it
    # raises and writes nothing, so a concurrent commit can't be out-raced.
    assert tg.add_spend(storage, EMAIL, 0.6, cap=1.0) == 0.6
    with pytest.raises(tg.SpendCapExceeded):
        tg.add_spend(storage, EMAIL, 0.6, cap=1.0)
    assert tg.today_spend(storage, EMAIL) == 0.6


def test_add_spend_rejects_replayed_token(storage):
    sig = 'deadbeef'
    assert tg.add_spend(storage, EMAIL, 0.5, token_sig=sig) == 0.5
    with pytest.raises(tg.TokenReplayed):
        tg.add_spend(storage, EMAIL, 0.5, token_sig=sig)
    # the replay charged nothing
    assert tg.today_spend(storage, EMAIL) == 0.5
    # a different token still commits
    assert tg.add_spend(storage, EMAIL, 0.5, token_sig='feedface') == 1.0


def test_trigger_token_is_single_use(storage):
    """Replaying the same confirm token inside its TTL must not start a second
    run or charge a second time (findings #3/#4)."""
    args = {'company_keys': ['Acme||https://acme.com']}
    est = tg.estimate(storage, 'build_profiles', args, EMAIL, secret=SECRET, now=1000)
    jobs = FakeJobs()
    first = tg.trigger(storage, 'build_profiles', args, EMAIL, est['confirm_token'],
                       secret=SECRET, jobs_client=jobs, now=1001)
    assert first['triggered']
    charged = tg.today_spend(storage, EMAIL)
    replay = tg.trigger(storage, 'build_profiles', args, EMAIL, est['confirm_token'],
                        secret=SECRET, jobs_client=jobs, now=1002)
    assert 'error' in replay and 'already used' in replay['error']
    assert len(jobs.calls) == 1                       # no duplicate run
    assert tg.today_spend(storage, EMAIL) == charged  # no double charge


def test_trigger_matching_already_running_via_lock(storage):
    """matching-job now holds a run lock, so a concurrent run_matching is caught
    by running_run_id just like the sync jobs (finding #4)."""
    bucket = storage.bucket(tg.BUCKET)
    assert gw.acquire_lock(bucket, 'matching-job', 'existing_match', 3600)
    args = {'agencies': ['DOD'], 'sources': ['apollo']}
    est = tg.estimate(storage, 'run_matching', args, EMAIL, secret=SECRET, now=1000)
    jobs = FakeJobs()
    out = tg.trigger(storage, 'run_matching', args, EMAIL, est['confirm_token'],
                     secret=SECRET, jobs_client=jobs, now=1001)
    assert out.get('already_running') and out['run_id'] == 'existing_match'
    assert len(jobs.calls) == 0


def test_run_matching_email_config_is_dict(storage):
    """email_config must be {} not None, or matching_job crashes with
    prewrite_email on (finding #1)."""
    args = {'agencies': ['DOD'], 'sources': ['apollo'], 'prewrite_email': True}
    est = tg.estimate(storage, 'run_matching', args, EMAIL, secret=SECRET, now=1000)
    out = tg.trigger(storage, 'run_matching', args, EMAIL, est['confirm_token'],
                     secret=SECRET, jobs_client=FakeJobs(), now=1001)
    cfg = json.loads(storage.bucket(tg.BUCKET).blob(
        f"job-configs/{out['run_id']}.json").download_as_text())
    assert cfg['email_config'] == {} and cfg['prewrite_email'] is True
