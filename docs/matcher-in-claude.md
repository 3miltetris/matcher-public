# The Matcher in Claude — `matcher-mcp` connector

The **Matcher MCP server** (`services/matcher_mcp/`) lets the BW&CO team use the
Matcher from **claude.ai** as a custom connector: searching grants and companies,
checking what last night's scheduled runs produced, and reviewing onboarding
proposals — without opening Streamlit. **Tier 1 is read-only** (no triggers, no
writes, no admin). **Tier 2 (job triggers, behind a two-step confirm token +
per-user daily spend cap) is built and tested — see the Tier 2 section below for
the IAM + secret it needs before it works.** Tier 3 (staged writes) is a later PR.

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

**Tier 1 tools:** `search_grants`, `search_companies` (match on company NAME),
`search_contacts` (semantic/embedding search of the contact store — lead sources
+ clients/prospects — for companies whose summary fits a technology description),
`get_company`, `get_profile`,
`search_resumes`, `list_runs`, `get_run_status`, `get_match_results`, `whats_new`,
`list_attention`, `list_onboarding_proposals`, `get_onboarding_proposal`,
`list_match_options` (the available contact sources + agency folders, by name, to
feed a `run_matching` request).

**`search_contacts` runs off a prebuilt index.** `data/all-contacts/` is ~3.7 GB
(~79% of it embedding vectors), far too much to download per query — scanning it
live took 100+ s and timed out on apollo/sba/sam/vc/crunchbase. The
**`contact-index-job`** (the matcher-mcp image running
`python -m src.modules.tools.contact_index`) condenses each source into
unit-normalised float16 vectors + a small metadata table under
`data/contact-index/{source}/` (~1.2 GB total; apollo ~196k companies). The server
loads it on boot and caches it; any blob imported since the last build is scanned
live, so a new import is searchable immediately. Measured from GCP: cold ~20 s for
all 14 sources, warm apollo ~1.5 s. **4 GiB is required** (raised from 1 GiB): the
cached index is ~1 GB, loads are single-flight and copy-free, but a 1 GiB instance
would not hold it. A source whose blobs are unchanged is skipped on rebuild, unless
its manifest predates the current `INDEX_FORMAT` (then it is rebuilt).
Companies are deduped on their website **host**, not name+URL: the same firm is
"ionX Technologies, Inc." in doe_teaming_partners and "IONX TECHNOLOGIES INC" in
sam. Each build writes `hosts.json` so `also_in` names every folder holding the
company, even ones the search was not scoped to — 2,315 companies sit in more than
one of just doe/sam/sec_raise/sba, so this is what prevents double outreach. Run it by hand after a big import with
`gcloud run jobs execute contact-index-job --region us-central1 --project cc-matcher-v1`
(add `--args=-m,src.modules.tools.contact_index,--force` to rebuild everything).
Nightly schedule (3 AM Central):
```
gcloud scheduler jobs create http contact-index-daily --schedule="0 3 * * *" \
  --uri="https://run.googleapis.com/v2/projects/cc-matcher-v1/locations/us-central1/jobs/contact-index-job:run" \
  --http-method=POST --message-body='{}' \
  --oauth-service-account-email=matching-job@cc-matcher-v1.iam.gserviceaccount.com \
  --location=us-central1 --time-zone="America/Chicago" --project cc-matcher-v1
```

**Tier 2 tools (triggers):** `run_matching`, `build_profiles`, `sync_drive`,
`sync_fathom`, `run_onboarding_agent`. `run_matching` takes the match axes
(`agencies`, `sources`) plus an optional single grant **date filter**
(`date_column` = `due_date`/`scraped_at`, `date_from`/`date_to` ISO, one-sided
allowed, `include_undated`) and `include_procurement`; the date filter becomes a
`date_range` `topic_filter` the matching job already applies. Each is **two-step**:
calling it without a
`confirm_token` validates the arguments and returns `{estimate, confirm_token}`
with no side effect; calling it again with the SAME arguments plus that token
starts the Cloud Run job. The token is an HMAC over `(tool, normalized args,
email, 10-min expiry)` — a changed argument, a different caller or an expired
token all fail closed (`src/modules/tools/triggers.py`). Three caps no token can
bypass, enforced in code: a per-call unit limit per tool (`build_profiles` ≤ 25
companies, `run_onboarding_agent` ≤ 10 folders, the syncs ≤ 50, matching = 1
run), a **per-user daily spend cap** (default $25, `admin-config/mcp_limits.json`,
tracked in `mcp-spend/{date}.json`), and the existing job **run lock** (an
already-running job returns `{already_running, run_id}` instead of a duplicate).
The committed estimate is written to the spend ledger **before** the job is
triggered, and the audit line records the per-call `cost_usd`.

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
(Tier 3 will add write on `drive-sync-configs/assignments.json` + the clients prefix.
Tier 2's `roles/run.admin` grants + HMAC secret are in the **Tier 2 setup** section below.)

### 3. Build + deploy
```
gcloud builds submit --config cloudbuild.matcher_mcp.yaml --project cc-matcher-v1 .

gcloud run deploy matcher-mcp \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/matcher-mcp:latest \
  --region us-central1 --project cc-matcher-v1 \
  --service-account matcher-mcp@cc-matcher-v1.iam.gserviceaccount.com \
  --no-invoker-iam-check --max-instances 2 --memory 4Gi --cpu 2 \
  --set-secrets=OPENAI_API_KEY=openai-api-key:latest,GOOGLE_CLIENT_ID=mcp-google-client-id:latest,GOOGLE_CLIENT_SECRET=mcp-google-client-secret:latest,MCP_JWT_SIGNING_KEY=mcp-jwt-signing-key:latest,MCP_STORAGE_ENCRYPTION_KEY=mcp-storage-encryption-key:latest \
  --set-env-vars=MCP_BASE_URL=https://matcher-mcp-998055288572.us-central1.run.app
```
`MCP_JWT_SIGNING_KEY` + `MCP_STORAGE_ENCRYPTION_KEY` are **required** (the service
refuses to boot without them) — they are what keep the connector connected across
cold starts; see **§ Persistent OAuth** below for why and the one-time setup.
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

### 3b. Tier 2 setup (triggers) — ✅ DONE 2026-10-23

Done and live on revision **`matcher-mcp-00004-jlk`**: `mcp-confirm-secret` created,
`matcher-mcp@` granted `run.admin` on all 5 triggerable jobs, service redeployed
with `MCP_CONFIRM_SECRET` mounted. Verified: OAuth discovery 200, `/mcp` 401,
`MCP_CONFIRM_SECRET` present on the revision. (claude.ai caches the tool list, so
reconnect the connector once to pick up the 5 new trigger tools.) The commands
that did it, for reference / a rebuild:

```
# (a) the HMAC key the confirm tokens are signed with. One stable secret so a token
#     minted on one Cloud Run instance verifies on another (max-instances 2).
python -c "import secrets; print(secrets.token_urlsafe(48))" \
  | tr -d '\n' | gcloud secrets create mcp-confirm-secret --data-file=- --project cc-matcher-v1

# (b) run.admin on each triggerable job (runWithOverrides needs run.admin, not invoker —
#     the same rule the views' matcher-app@ grants follow).
for J in matching-job client-profile-job drive-sync-job fathom-sync-job drive-onboard-job; do
  gcloud run jobs add-iam-policy-binding "$J" --region us-central1 --project cc-matcher-v1 \
    --member="serviceAccount:matcher-mcp@cc-matcher-v1.iam.gserviceaccount.com" \
    --role="roles/run.admin"
done

# (c) OPTIONAL — override the $25/day default or a user's cap. Omit the blob for defaults.
#     {"default_daily_cap_usd":25,"user_daily_cap_usd":{"x@bwcoconsulting.com":50},
#      "unit_caps":{"build_profiles":25}}

# (d) redeploy mounting the new secret (keep the existing --set-secrets + --set-env-vars).
gcloud run services update matcher-mcp --region us-central1 --project cc-matcher-v1 \
  --update-secrets=MCP_CONFIRM_SECRET=mcp-confirm-secret:latest
```

If `MCP_CONFIRM_SECRET` is unset the trigger tools refuse with a clear "server
misconfigured" message rather than running uncapped — fail closed. Without the
`run.admin` grants the config is still staged but the `run_job` call 403s, and the
tool reports "triggered spend but the job failed to start" (so re-grant, then the
same run_id can be re-triggered).

### 3c. Persistent OAuth — so the connector stays connected (LIVE 2026-11-11, rev `matcher-mcp-00008-xdh`)

Done and live: Firestore `(default)` database created (Native, us-central1, free
tier), secrets `mcp-jwt-signing-key` + `mcp-storage-encryption-key` created,
`matcher-mcp@` granted `roles/datastore.user`, service redeployed with both mounted.
Verified: OAuth discovery 200, `/mcp` 401, both secrets present on the revision, and
the service boots (so `GoogleProvider` accepts the kwargs on the installed fastmcp).

**Symptom it fixes:** having to reconnect the connector in claude.ai on almost every
use. The connector was not the problem — the **server was discarding its OAuth state**.
FastMCP's `GoogleProvider` (an `OAuthProxy`) on Linux/Cloud Run (1) signs the JWTs it
issues with a key generated **fresh at each process start** (no keyring on Linux → an
ephemeral salt), and (2) keeps the proxy↔Google token mapping and the DCR client
registration **in memory**. `matcher-mcp` runs `--max-instances 2` with **no min
instances**, so it scales to zero when idle; every cold start after a gap wipes both,
the token claude.ai still holds "resolves to nothing" (`invalid_grant` on refresh), and
claude.ai is forced to redo the whole Google login. A stable signing key alone is **not
enough** — the refresh token is only a reference into a key-value store that also has to
survive the restart.

**The fix (both parts, in `services/matcher_mcp/main.py`):**
- a **stable `jwt_signing_key`** from Secret Manager (`mcp-jwt-signing-key`), so issued
  tokens stay valid across restarts; and
- a persistent, encrypted **`client_storage`** — a **Firestore** `FirestoreStore`
  (serverless, ADC, ~$0 at this volume) wrapped in a `FernetEncryptionWrapper` keyed by
  `mcp-storage-encryption-key`, so the token↔Google mapping and client registration
  survive cold starts (and redeploys) and refresh keeps working. Data lives in the
  Firestore `mcp_oauth` collection, encrypted at rest.

`_build_auth()` **requires both** env vars in production and refuses to boot without
them (fail loud, never silent reversion), and it signature-checks that the installed
fastmcp's `GoogleProvider` accepts `jwt_signing_key` + `client_storage`, raising an
"upgrade fastmcp" message if not. Dependencies: `py-key-value-aio[firestore]` +
`cryptography` (added to `requirements.matcher_mcp.txt`).

**One-time setup** (run once; `! gcloud auth login` first if tokens are stale — local
gcloud defaults to the wrong project, so every command passes `--project cc-matcher-v1`):
```bash
# (a) Enable Firestore and create the database in Native mode (one region; us-central1
#     to sit with everything else). Skip the create if a (default) database exists.
gcloud services enable firestore.googleapis.com --project cc-matcher-v1
gcloud firestore databases create --location=us-central1 --project cc-matcher-v1

# (b) Two secrets. Generate fresh values if rebuilding — the signing key is any strong
#     random string; the encryption key MUST be a urlsafe-base64 32-byte Fernet key
#     (python -c "import os,base64;print(base64.urlsafe_b64encode(os.urandom(32)).decode())").
printf %s '<MCP_JWT_SIGNING_KEY value>' \
  | gcloud secrets create mcp-jwt-signing-key --data-file=- --project cc-matcher-v1
printf %s '<MCP_STORAGE_ENCRYPTION_KEY value>' \
  | gcloud secrets create mcp-storage-encryption-key --data-file=- --project cc-matcher-v1

# (c) Let matcher-mcp@ read/write Firestore (secretAccessor it already has from §2).
gcloud projects add-iam-policy-binding cc-matcher-v1 \
  --member="serviceAccount:matcher-mcp@cc-matcher-v1.iam.gserviceaccount.com" \
  --role="roles/datastore.user"

# (d) Rebuild + redeploy mounting the two new secrets (keep the existing ones).
gcloud builds submit --config cloudbuild.matcher_mcp.yaml --project cc-matcher-v1 .
gcloud run services update matcher-mcp --region us-central1 --project cc-matcher-v1 \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/matcher-mcp:latest \
  --update-secrets=MCP_JWT_SIGNING_KEY=mcp-jwt-signing-key:latest,MCP_STORAGE_ENCRYPTION_KEY=mcp-storage-encryption-key:latest
```
After this, reconnect the connector **once** in claude.ai (to mint a token under the new
stable key); it should then stay connected. **Never rotate these two secrets casually** —
rotating either invalidates every live session and forces everyone to reconnect once.

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
`bwco-matcher` skill separately (below).

### 6. The `bwco-matcher` skill
The connector exposes the tools; the **`bwco-matcher` skill** (`skills/bwco-matcher/
SKILL.md`) teaches the team which tool to use for each workflow ("what grants fit
client X", reading match scores, working an onboarding proposal) and what still has
to be done in Streamlit. The connector and skill are **separate** artifacts in
claude.ai (no single bundle — spike Q3). Add the skill via claude.ai **Settings →
Capabilities → Skills → Upload** (or package it into an org plugin alongside the
connector if/when that path is confirmed). The per-tool detail lives in each tool's
own description, so the skill stays short; update it whenever a Tier-2/3 tool ships
(the "what this connector can't do yet" list).

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
