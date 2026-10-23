"""
Matcher MCP server — Tier 1 (read-only) (Phase 5)
=================================================
A public, streamable-HTTP MCP server the BW&CO team reaches from claude.ai as a
custom connector, to search the Matcher and check what the scheduled pipeline
produced without opening Streamlit. Tier 1 is READ-ONLY: no triggers (Tier 2),
no writes (Tier 3), no admin operations.

Design (see docs/spikes/mcp-connector.md for the researched answers):
  • Transport: streamable HTTP, served at /mcp. Discovery via
    /.well-known/oauth-protected-resource (RFC 9728); DCR per RFC 7591.
  • Auth: FastMCP `GoogleProvider` bridges Google (no DCR) to Claude (expects
    DCR), runs the Google login and issues the server's own JWT. The BW&CO
    restriction is layered on top in auth.py — GoogleProvider alone admits any
    Google account.
  • Tools: thin wrappers over the tested, transport-agnostic functions in
    src/modules/tools/ (+ company_lookup). Each wrapper gates on a verified BW&CO
    actor, caps its output, and writes one audit line.

The tool LOGIC lives in src/modules/tools and is unit-tested there; this module
is the wiring. `fastmcp` is imported lazily inside build_server so the module
imports cleanly without the dependency (tests exercise auth.py / audit.py, not a
live server). Run locally:
    pip install "fastmcp>=3.3,<4"
    MCP_DEV=1 python -m services.matcher_mcp.main        # dev identity, no OAuth
    # point the MCP Inspector at http://127.0.0.1:8080/mcp
"""

from __future__ import annotations

import json
import os

from google.cloud import storage

from services.matcher_mcp import audit as au, auth
from src.modules.tools import (
    attention as t_att, company_lookup as t_co, onboarding as t_ob,
    runs as t_runs, search as t_search, triggers as t_trig,
)

_BUCKET = 'cc-matcher-bucket-jeg-v1'
_EMBED_MODEL = 'text-embedding-ada-002'


# ── shared context (storage + query embedder) ────────────────────────────────

def _storage() -> storage.Client:
    return storage.Client()  # ADC on Cloud Run


def _confirm_secret() -> str:
    """The HMAC key the Tier-2 confirm tokens are signed with. Required in
    production (a Secret Manager secret mounted as MCP_CONFIRM_SECRET) so a token
    minted by one Cloud Run instance verifies on another; a fixed dev key in
    MCP_DEV so the Inspector can exercise the two-step locally."""
    s = os.environ.get('MCP_CONFIRM_SECRET', '')
    if not s and os.environ.get('MCP_DEV') == '1':
        return 'mcp-dev-confirm-secret'
    return s


def _make_embed_query():
    """embed_query(text) -> list[float] using text-embedding-ada-002, the model
    every stored vector was built with. Built once; reused across calls."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    def embed_query(text: str) -> list[float]:
        resp = client.embeddings.create(model=_EMBED_MODEL, input=text or '')
        return list(resp.data[0].embedding)

    return embed_query


# ── FastMCP auth + claims integration point ──────────────────────────────────

def _build_auth():
    """GoogleProvider for production; None in MCP_DEV (no OAuth, dev identity).
    Verify the GoogleProvider kwargs against the installed fastmcp on first
    deploy (the house 'check the signature before trusting it' rule)."""
    if os.environ.get('MCP_DEV') == '1':
        return None
    from fastmcp.server.auth.providers.google import GoogleProvider
    return GoogleProvider(
        client_id=os.environ['GOOGLE_CLIENT_ID'],
        client_secret=os.environ['GOOGLE_CLIENT_SECRET'],
        base_url=os.environ['MCP_BASE_URL'],  # the server's own public URL
        # GoogleProvider requests NO user info by default — without the email
        # scope the email claim is absent and the BW&CO gate fails closed on
        # every caller. openid + userinfo.email are the minimum the gate needs.
        required_scopes=['openid', 'https://www.googleapis.com/auth/userinfo.email'],
        # The OAuth callback path the Google client must list as its authorized
        # redirect URI: {MCP_BASE_URL}/auth/callback.
        redirect_path='/auth/callback',
    )


def _current_claims() -> dict:
    """The verified Google claims for the in-flight request, from FastMCP's auth
    dependency. THE integration point: the attribute carrying the ID-token claims
    differs across fastmcp versions, so this reads a few known shapes and fails
    closed. Confirm against the installed fastmcp on first deploy."""
    try:
        from fastmcp.server.dependencies import get_access_token
        tok = get_access_token()
    except Exception:
        return {}
    if tok is None:
        return {}
    for attr in ('claims', 'extra', 'user'):
        val = getattr(tok, attr, None)
        if isinstance(val, dict) and ('email' in val or 'email_verified' in val):
            return val
    # Some providers expose the fields directly on the token object.
    email = getattr(tok, 'email', None)
    if email:
        return {'email': email, 'email_verified': getattr(tok, 'email_verified', True)}
    return {}


# ── tool wrapper: actor gate + call + output cap + audit ──────────────────────

def _emit(result) -> str:
    """Serialise a tool result to a JSON string capped at the connector's result
    budget (20k chars); halve the longest list field until it fits and say so."""
    text = json.dumps(result, default=str)
    if len(text) <= 20_000:
        return text
    if isinstance(result, dict):
        trimmed = dict(result)
        while len(text) > 20_000:
            lists = [(k, v) for k, v in trimmed.items()
                     if isinstance(v, list) and len(v) > 1]
            if not lists:
                break
            k, v = max(lists, key=lambda kv: len(json.dumps(kv[1], default=str)))
            trimmed[k] = v[:max(1, len(v) // 2)]
            trimmed['_truncated'] = True
            text = json.dumps(trimmed, default=str)
        return text if len(text) <= 20_000 else text[:20_000]
    return text[:20_000]


def build_server():
    from fastmcp import FastMCP

    gcs = _storage()
    embed_query = _make_embed_query()
    secret = _confirm_secret()
    mcp = FastMCP(name='matcher-mcp', auth=_build_auth())

    _jobs_box: dict = {}

    def _jobs_client():
        """A Cloud Run JobsClient (ADC), built once, only when a trigger fires —
        so the read-only Tier-1 tools and MCP_DEV never need Cloud Run creds."""
        if 'c' not in _jobs_box:
            from google.cloud import run_v2
            _jobs_box['c'] = run_v2.JobsClient()
        return _jobs_box['c']

    def run_tool(tool: str, impl, audit_args: dict):
        """Gate on a verified BW&CO actor, run `impl()`, audit, return a dict.
        An AccessDenied or any error is audited and returned as an error dict
        (never a stack trace to the model)."""
        try:
            actor = auth.require_actor(_current_claims)
        except auth.AccessDenied as e:
            au.record(gcs, email='', tool=tool, args=audit_args, outcome='denied')
            return {'error': str(e)}
        try:
            result = impl()
            au.record(gcs, email=actor.email, tool=tool, args=audit_args,
                      outcome='ok')
            return result
        except Exception as e:  # noqa: BLE001
            au.record(gcs, email=actor.email, tool=tool, args=audit_args,
                      outcome='error')
            return {'error': f'{type(e).__name__}: {e}'}

    # ── search ────────────────────────────────────────────────────────────────
    @mcp.tool()
    def search_grants(query: str, agency: str = '', open_only: bool = True,
                      limit: int = 10) -> str:
        """Rank open grant topics against a typed technology description. Set
        `agency` to one folder code (e.g. 'DOD', 'HHS') to narrow, or leave empty
        for all. Returns the top matches with a 0-1 relevance score."""
        args = {'query': query, 'agency': agency, 'open_only': open_only,
                'limit': limit}
        r = run_tool('search_grants', lambda: t_search.search_grants(
            gcs, embed_query, query, agency=(agency or None),
            open_only=open_only, limit=limit), args)
        return _emit(r)

    @mcp.tool()
    def search_companies(query: str, pools: str = 'clients,prospects',
                         limit: int = 10) -> str:
        """Find a company by name and get its company_key. `pools` is a
        comma-separated subset of clients,prospects."""
        args = {'query': query, 'pools': pools, 'limit': limit}
        pl = [p.strip() for p in pools.split(',') if p.strip()]
        r = run_tool('search_companies', lambda: t_co.search_companies(
            gcs, query, pools=tuple(pl), limit=limit), args)
        return _emit(r)

    @mcp.tool()
    def get_company(company_key: str) -> str:
        """A company's summary, website, pool, Drive folder assignments and any
        intake record. `company_key` is 'Name||https://site'."""
        args = {'company_key': company_key}
        r = run_tool('get_company', lambda: t_co.get_company(gcs, company_key),
                     args)
        return _emit(r)

    @mcp.tool()
    def get_profile(company_key: str) -> str:
        """A company's multi-aspect capability profile: aspects, confirmed
        markets and unexplored markets."""
        args = {'company_key': company_key}
        r = run_tool('get_profile', lambda: t_search.get_profile(gcs, company_key),
                     args)
        return _emit(r)

    @mcp.tool()
    def search_resumes(query: str, limit: int = 10) -> str:
        """Rank individual résumés by expertise against a natural-language query."""
        args = {'query': query, 'limit': limit}
        r = run_tool('search_resumes', lambda: t_search.search_resumes(
            gcs, embed_query, query, limit=limit), args)
        return _emit(r)

    # ── runs ──────────────────────────────────────────────────────────────────
    @mcp.tool()
    def list_runs(job: str, limit: int = 10) -> str:
        """Recent runs of a job (run_id + state). `job` is e.g. 'drive-sync-job',
        'matching-job', 'deep-research-job'."""
        args = {'job': job, 'limit': limit}
        r = run_tool('list_runs', lambda: t_runs.list_runs(gcs, job, limit=limit),
                     args)
        return _emit(r)

    @mcp.tool()
    def get_run_status(job: str, run_id: str) -> str:
        """One run's status.json (counts, state, errors), trimmed to the result
        budget."""
        args = {'job': job, 'run_id': run_id}
        r = run_tool('get_run_status',
                     lambda: t_runs.get_run_status(gcs, job, run_id), args)
        return _emit(r)

    @mcp.tool()
    def get_match_results(run_id: str, company_key: str = '', good_only: bool = False,
                          limit: int = 25, cursor: str = '') -> str:
        """Page a matching run's result rows. Filter to one `company_key` or to
        AI-validated good matches. Pass the returned `next_cursor` to continue."""
        args = {'run_id': run_id, 'company_key': company_key,
                'good_only': good_only, 'limit': limit, 'cursor': cursor}
        r = run_tool('get_match_results', lambda: t_runs.get_match_results(
            gcs, run_id, company_key=(company_key or None), good_only=good_only,
            limit=limit, cursor=(cursor or None)), args)
        return _emit(r)

    @mcp.tool()
    def whats_new() -> str:
        """A digest of what the scheduled pipeline produced (latest run of each
        scheduled job) plus the open attention-queue counts."""
        r = run_tool('whats_new', lambda: t_runs.whats_new(gcs), {})
        return _emit(r)

    # ── attention + onboarding ────────────────────────────────────────────────
    @mcp.tool()
    def list_attention(kind: str = '', status: str = 'open', limit: int = 50) -> str:
        """Open items in the attention queue (what a scheduled job could not
        resolve). Filter by `kind`; `status='all'` for every status."""
        args = {'kind': kind, 'status': status, 'limit': limit}
        r = run_tool('list_attention', lambda: t_att.list_attention(
            gcs, kind=(kind or None), status=status, limit=limit), args)
        return _emit(r)

    @mcp.tool()
    def list_onboarding_proposals(status: str = 'open', limit: int = 50) -> str:
        """The Drive onboarding agent's proposals (compact). `status='all'` for
        every status."""
        args = {'status': status, 'limit': limit}
        r = run_tool('list_onboarding_proposals',
                     lambda: t_ob.list_onboarding_proposals(
                         gcs, status=(None if status == 'all' else status),
                         limit=limit), args)
        return _emit(r)

    @mcp.tool()
    def get_onboarding_proposal(folder_id: str) -> str:
        """One onboarding proposal in full: decision, confidence, evidence,
        conflicts and the document include/exclude table."""
        args = {'folder_id': folder_id}
        r = run_tool('get_onboarding_proposal',
                     lambda: t_ob.get_onboarding_proposal(gcs, folder_id), args)
        return _emit(r)

    # ── Tier-2 triggers (two-step confirmation) ───────────────────────────────
    def run_trigger(tool: str, args: dict, confirm_token: str):
        """Shared wrapper for the five trigger tools. With no confirm_token it
        returns an estimate + a token (no side effect); with a confirm_token it
        verifies, enforces the caps + run lock, triggers the job and audits the
        committed estimate cost. Gating, audit and error-hiding mirror run_tool."""
        try:
            actor = auth.require_actor(_current_claims)
        except auth.AccessDenied as e:
            au.record(gcs, email='', tool=tool, args=args, outcome='denied')
            return _emit({'error': str(e)})
        if not secret:
            au.record(gcs, email=actor.email, tool=tool, args=args, outcome='error')
            return _emit({'error': 'server misconfigured: MCP_CONFIRM_SECRET is '
                                   'not set, so triggers are disabled'})
        try:
            if not confirm_token:
                res = t_trig.estimate(gcs, tool, args, actor.email, secret=secret)
                au.record(gcs, email=actor.email, tool=tool, args=args,
                          outcome='estimate')
            else:
                res = t_trig.trigger(gcs, tool, args, actor.email, confirm_token,
                                     secret=secret, jobs_client=_jobs_client())
                outcome = ('triggered' if res.get('triggered')
                           else 'already_running' if res.get('already_running')
                           else 'error' if 'error' in res else 'ok')
                au.record(gcs, email=actor.email, tool=tool, args=args,
                          outcome=outcome, cost_usd=float(res.get('est_cost_usd') or 0.0))
            return _emit(res)
        except Exception as e:  # noqa: BLE001
            au.record(gcs, email=actor.email, tool=tool, args=args, outcome='error')
            return _emit({'error': f'{type(e).__name__}: {e}'})

    @mcp.tool()
    def run_matching(agencies: str, sources: str, threshold: float = 0.82,
                     top_k: int = 5, ai_validation: bool = True,
                     prewrite_email: bool = False, confirm_token: str = '') -> str:
        """Start a bulk matching run (contacts × grant topics). `agencies` and
        `sources` are comma-separated (e.g. 'DOD,HHS' / 'apollo,sba'). Call once
        WITHOUT confirm_token to get an estimate and a confirm_token, then call
        again with the SAME arguments plus that confirm_token to start the run."""
        args = {'agencies': agencies, 'sources': sources, 'threshold': threshold,
                'top_k': top_k, 'ai_validation': ai_validation,
                'prewrite_email': prewrite_email}
        return run_trigger('run_matching', args, confirm_token)

    @mcp.tool()
    def build_profiles(company_keys: str, pool: str = 'clients', sources: str = '',
                       target_aspects: int = 4, assess_unexplored: bool = True,
                       confirm_token: str = '') -> str:
        """(Re)build multi-aspect capability profiles for companies (≤25 per call).
        `company_keys` is comma-separated 'Name||https://site' keys. `pool` is
        clients or prospects; `sources` optionally narrows the material. Two-step:
        call without confirm_token for the estimate, then with it to start."""
        args = {'company_keys': company_keys, 'pool': pool, 'sources': sources,
                'target_aspects': target_aspects, 'assess_unexplored': assess_unexplored}
        return run_trigger('build_profiles', args, confirm_token)

    @mcp.tool()
    def sync_drive(company_keys: str, full_resync: bool = False,
                   dry_run: bool = False, confirm_token: str = '') -> str:
        """Sync the Google Drive folders of specific clients (only companies that
        already have an assigned folder). `company_keys` is comma-separated.
        Two-step: estimate first, then confirm with the token."""
        args = {'company_keys': company_keys, 'full_resync': full_resync,
                'dry_run': dry_run}
        return run_trigger('sync_drive', args, confirm_token)

    @mcp.tool()
    def sync_fathom(company_keys: str, lookback_days: int = 90,
                    full_resync: bool = False, dry_run: bool = False,
                    confirm_token: str = '') -> str:
        """Ingest recent Fathom meetings for specific companies. `company_keys` is
        comma-separated; `lookback_days` bounds the window. Two-step: estimate
        first, then confirm with the token."""
        args = {'company_keys': company_keys, 'lookback_days': lookback_days,
                'full_resync': full_resync, 'dry_run': dry_run}
        return run_trigger('sync_fathom', args, confirm_token)

    @mcp.tool()
    def run_onboarding_agent(folder_ids: str, dry_run: bool = False,
                             confirm_token: str = '') -> str:
        """Run the Drive onboarding agent on unmatched folders (≤10 per call) to
        propose which company each belongs to. `folder_ids` is comma-separated
        Drive folder ids (from list_attention 'drive_unmatched_folder' items).
        Two-step: estimate first, then confirm with the token."""
        args = {'folder_ids': folder_ids, 'dry_run': dry_run}
        return run_trigger('run_onboarding_agent', args, confirm_token)

    return mcp


if __name__ == '__main__':
    server = build_server()
    # FastMCP 3.x takes host/port as run() kwargs (forwarded into
    # transport_kwargs); it does NOT read FASTMCP_HOST/PORT here, so Cloud Run
    # needs them passed explicitly — bind 0.0.0.0:$PORT, not the 127.0.0.1:8000
    # default (which fails the Cloud Run startup probe).
    server.run(transport='streamable-http', host='0.0.0.0',
               port=int(os.environ.get('PORT', '8080')))
