# The Matcher in Claude — `matcher-mcp` connector

The **Matcher MCP server** (`services/matcher_mcp/`) lets the BW&CO team use the
Matcher from **claude.ai** as a custom connector: searching grants and companies,
checking what last night's scheduled runs produced, and reviewing onboarding
proposals — without opening Streamlit. **Tier 1 is read-only** (no triggers, no
writes, no admin). Tiers 2–3 (triggers, staged writes) are later PRs.

This is the project's **second public endpoint** (after `intake-web`): claude.ai
calls it from Anthropic's infrastructure, so IAP cannot gate it — auth is OAuth on
the server itself, restricted to **verified `@bwcoconsulting.com` Google
identities** (`services/matcher_mcp/auth.py`).

---

## Architecture (one paragraph)

Streamable HTTP MCP server (FastMCP) at `/mcp`. Auth is FastMCP's **`GoogleProvider`**
(an `OAuthProxy`): it shows claude.ai a Dynamic-Client-Registration interface, bridges
to a **pre-registered Google OAuth client**, runs the Google login, and issues the
server's own JWT. Each tool gates on a verified BW&CO actor, caps its output at 20k
chars, paginates large sets with an opaque cursor, and writes one audit line to
`mcp-audit/{date}.jsonl` (arguments **hashed**, never stored). The tool logic is the
transport-agnostic, unit-tested functions in `src/modules/tools/` — the server is
just the wiring. See `docs/spikes/mcp-connector.md` for the researched OAuth/connector
answers.

**Tier 1 tools:** `search_grants`, `search_companies`, `get_company`, `get_profile`,
`search_resumes`, `list_runs`, `get_run_status`, `get_match_results`, `whats_new`,
`list_attention`, `list_onboarding_proposals`, `get_onboarding_proposal`.

---

## DEPLOYED 2026-10-22

- **Service URL:** `https://matcher-mcp-998055288572.us-central1.run.app` (Cloud Run gave the new service the newer `-<projectNumber>.<region>.run.app` URL format, **not** the `-76zevaohwa-uc` format the older services use — so the URL could not be predicted; it was read off the first deploy).
- **MCP endpoint (connector URL):** `https://matcher-mcp-998055288572.us-central1.run.app/mcp`
- **Google OAuth redirect URI:** `https://matcher-mcp-998055288572.us-central1.run.app/auth/callback`
- Public via **`--no-invoker-iam-check`** (`run.googleapis.com/invoker-iam-disabled: true`), because the org's Domain-Restricted-Sharing policy blocks an `allUsers` IAM binding — the same mechanism `intake-web` uses.
- Deployed revision serves OAuth discovery correctly (`/.well-known/oauth-protected-resource/mcp` → 200, scopes `openid`+`userinfo.email`; `/mcp` → 401 with the `WWW-Authenticate` resource_metadata pointer).
- **Remaining to use it:** (1) set the Google OAuth client's redirect URI to the one above, (2) an org owner adds the connector in claude.ai (URL `…/mcp`, "Register automatically").

## One-time setup (reference — mostly done 2026-10-22)

### 1. Google OAuth client (identity)
In the `cc-matcher-v1` Google Cloud console → **APIs & Services → Credentials →
Create OAuth client ID → Web application**:
- Authorized redirect URI: **`{MCP_BASE_URL}/auth/callback`** (the deployed Cloud Run
  URL + `/auth/callback`; redeploy-and-update once the URL is known, or use a custom
  domain so the URL is stable).
- Restrict to the BW&CO Google Workspace if possible (the server re-checks the domain
  regardless — the `hd` hint is not the control).
- Store the client id/secret in Secret Manager: `mcp-google-client-id`,
  `mcp-google-client-secret`.

### 2. Service account + least-privilege IAM (plan §5.3)
```
gcloud iam service-accounts create matcher-mcp --display-name="Matcher MCP server" \
  --project cc-matcher-v1
# Read the data it serves + write only the audit log (Tier 1 writes nothing else):
gcloud projects add-iam-policy-binding cc-matcher-v1 \
  --member="serviceAccount:matcher-mcp@cc-matcher-v1.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"        # scope to the served prefixes + mcp-audit/ when buckets are split
gcloud projects add-iam-policy-binding cc-matcher-v1 \
  --member="serviceAccount:matcher-mcp@cc-matcher-v1.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```
(Tier 2 will add `roles/run.admin` on the specific jobs it triggers; Tier 3 adds write
on `drive-sync-configs/assignments.json` + the clients prefix. Not needed for Tier 1.)

### 3. Build + deploy
```
gcloud builds submit --config cloudbuild.matcher_mcp.yaml --project cc-matcher-v1 .

gcloud run deploy matcher-mcp \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/matcher-mcp:latest \
  --region us-central1 --project cc-matcher-v1 \
  --service-account matcher-mcp@cc-matcher-v1.iam.gserviceaccount.com \
  --no-invoker-iam-check --max-instances 2 --memory 1Gi \
  --set-secrets=OPENAI_API_KEY=openai-api-key:latest,GOOGLE_CLIENT_ID=mcp-google-client-id:latest,GOOGLE_CLIENT_SECRET=mcp-google-client-secret:latest \
  --set-env-vars=MCP_BASE_URL=https://matcher-mcp-998055288572.us-central1.run.app
```
Two gotchas learned on the first deploy:
- **Public access:** `--allow-unauthenticated` tries an `allUsers` IAM binding, which the
  DRS org policy **refuses**. Use **`--no-invoker-iam-check`** instead (sets
  `invoker-iam-disabled: true`), the same mechanism `intake-web` uses.
- **`MCP_BASE_URL`** must equal the service's own public URL, and the URL is **not
  predictable** (new services get the `-<projectNumber>` format). Deploy once, read the
  `Service URL` off the output, then `gcloud run services update matcher-mcp
  --set-env-vars=MCP_BASE_URL=<that url>` and set the Google redirect URI to
  `<that url>/auth/callback`.
- **Port:** the server binds `0.0.0.0:$PORT` via `run(..., host=..., port=...)` — FastMCP
  3.x ignores `FASTMCP_HOST/PORT`, so a container that only sets those env vars binds
  `127.0.0.1:8000` and fails the Cloud Run startup probe.

### 4. Verify the fastmcp signatures (house rule — before trusting a deploy)
```python
import inspect
from fastmcp.server.auth.providers.google import GoogleProvider
from fastmcp.server.dependencies import get_access_token
print(sorted(inspect.signature(GoogleProvider.__init__).parameters))
```
Confirm `_build_auth`'s kwargs and `_current_claims`'s token-claims shape in
`services/matcher_mcp/main.py` match the installed `fastmcp` (3.3.x).

### 5. Add the connector in claude.ai (org owner — plan open question #5)
Settings → Connectors → **Add custom connector** → URL `https://<url>/mcp` →
Authentication → OAuth client → **Register automatically** (DCR). Sign in with a
BW&CO Google account; the first connect negotiates the MCP protocol version (record
it) and `whoami`-equivalent access is gated to `@bwcoconsulting.com`. Then add the
`bwco-matcher` skill separately (§5.7).

---

## Local development
```
pip install "fastmcp>=3.3,<4"
MCP_DEV=1 OPENAI_API_KEY=sk-... python -m services.matcher_mcp.main   # dev identity, no OAuth
# point the MCP Inspector at http://127.0.0.1:8080/mcp
```
`MCP_DEV=1` returns a fixed `dev@bwcoconsulting.com` identity so the transport + tools
can be exercised before OAuth is wired. **Never set `MCP_DEV` on the deployed service.**

## Tests
```
python -m pytest tests/mcp     # tool logic + auth gate + audit (no fastmcp needed)
```
