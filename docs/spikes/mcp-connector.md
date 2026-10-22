# Spike 0.3 — claude.ai custom-connector auth

**Goal:** stand up a hello-world remote MCP server, add it to claude.ai as a
custom connector, and have one `whoami` tool return the signed-in user's verified
`@bwcoconsulting.com` email (`PLAN-drive-onboarding-agent.md` §0.3). This
de-risks Phase 5 (`matcher-mcp`); it is **parallelizable and does not block
Phase 1**, and adding the connector **needs the org owner** (plan open question #5).

**Status:** scaffold written (`scripts/spikes/mcp_whoami_server.py`) — transport +
`whoami` tool run locally under `MCP_DEV=1`. The five questions below are
**doc-research items**, answered from current docs, **not written from memory**:
read the support.claude.com custom-connectors / remote-MCP articles and the MCP
authorization spec before wiring auth.

**Run / verify:**
```
pip install "mcp[cli]" starlette uvicorn
MCP_DEV=1 python scripts/spikes/mcp_whoami_server.py      # http://127.0.0.1:8080/mcp
# point the MCP Inspector at it, call whoami  -> dev@bwcoconsulting.com
```
Deploy to Cloud Run `--allow-unauthenticated` (claude.ai calls from Anthropic
infra, so IAP can't gate it — auth is OAuth on the server itself), then the org
owner adds it in claude.ai and calls whoami.

---

## The five questions (fill from docs, then from the real connector)

### 1 — OAuth flow + redirect URI
Which flow does claude.ai support for a custom connector: **dynamic client
registration**, or a **pre-registered client id + secret** entered when the
connector is added? What redirect URI must the server allow? Identity provider
is **Google**; accept a caller only if the verified ID-token `email` ends in
`@bwcoconsulting.com` **and** `email_verified` is true, checked server-side (the
`hd` hint is a convenience, not a control — same rule as `services/intake_web`).
_Answer (2026-10-22, from current docs):_ claude.ai supports **all three** OAuth-client
identification modes, chosen in the "Add connector → Authentication → OAuth client"
dialog: **Claude's published identity** (a Client ID Metadata Document Anthropic
hosts), **Register automatically** (OAuth 2.0 **Dynamic Client Registration**,
RFC 7591), or a **pre-registered client id** the owner enters. On connect, claude.ai
first fetches `GET /.well-known/oauth-protected-resource` (RFC 9728) and follows its
`authorization_servers`. **Our design:** the server is its own Authorization Server
via FastMCP's **`GoogleProvider`** (an `OAuthProxy`): it presents a DCR-compliant
`/register` to Claude and bridges to **Google** using a *pre-registered Google OAuth
client* (one `GOOGLE_CLIENT_ID`/`SECRET`), runs the Google login, and issues the
server's own JWT (never passing Google's token through). So in the claude.ai dialog
choose **Register automatically**. **Redirect URI to allow in the Google Cloud OAuth
client:** the proxy's own fixed callback, `{MCP_BASE_URL}/auth/callback` (FastMCP's
`redirect_path` default); Claude's own dynamic callback is stored and forwarded by the
proxy, so Google only ever sees the proxy's. The BW&CO/`email_verified` gate is layered
on in `services/matcher_mcp/auth.py` (GoogleProvider alone admits any Google account).

### 2 — Transport + protocol version
Which MCP transport and protocol version does claude.ai expect? (Streamable HTTP
is assumed; the scaffold serves `/mcp`.) Record the exact protocol version string.
_Answer (2026-10-22):_ **Streamable HTTP** — confirmed the expected remote transport
(the older HTTP+SSE transport is legacy). FastMCP serves it at `/mcp`
(`server.run(transport="streamable-http")`). The **protocol version is negotiated**
per connection via the `MCP-Protocol-Version` header during `initialize`; the SDK
sends the current revision, so there is no version string to hard-code. **Record the
negotiated version printed on the first real connect** when the connector is added.

### 3 — Org-wide install + plugin packaging
Can an org owner add the connector once for the whole organization? Can it be
packaged in a plugin alongside the `bwco-matcher` skill (plan §5.7), or must the
connector and skill be added separately?
_Answer (2026-10-22):_ On Team/Enterprise plans an **organization admin/owner** adds a
custom connector org-wide (Settings → Connectors / admin console), so individuals need
not each add it. There is **no single "connector + skill" bundle** to add in one click
in the connector dialog — the connector (remote MCP URL) and the `bwco-matcher` skill
are added as **separate** artifacts. **Confirm the exact admin path and whether the
plan allows org-wide custom connectors on first install** (needs the org owner — plan
open question #5). The skill ships separately (plan §5.7).

### 4 — Response-size limits per tool result
What is the max size of a single tool result claude.ai will accept? This sets the
hard cap for every Phase 5 tool's output (plan §5.4 assumes a ~20k-char default,
truncate-middle-out).
_Answer (2026-10-22):_ No single published hard byte-limit for a tool result; the
practical constraint is the model's context budget, so **keep results small**. We
adopt the plan's **20,000-char cap** (`src/modules/tools/_common.MAX_RESULT_CHARS`),
cap every free-text field middle-out, and **paginate** large sets with an opaque
cursor — never return a whole parquet. `services/matcher_mcp/main._emit` enforces the
20k ceiling on the serialized result as a backstop. Revisit if a real connector
rejects a result under this cap.

### 5 — MCP Python SDK version to pin
Which `mcp` SDK version should we pin? Per CLAUDE.md **"SDK version pins"**, pin
`mcp>=<x.y>,<<x+1>` with an explicit upper bound before adding it to any
`requirements*.txt`, and record the signature check used (the same discipline as
the `anthropic`/`openai` pins).
_Answer (2026-10-22):_ Pin the **standalone `fastmcp>=3.3.0,<4`** (not the mcp SDK's
bundled FastMCP 1.x) — it carries `GoogleProvider` / `OAuthProxy`, which the OAuth
bridge needs. Latest is 3.3.1 (May 2026). `requirements.matcher_mcp.txt` has the pin.
**Signature check before any version bump / first deploy** (the house rule — verify,
don't deploy-and-hope):
```python
import inspect
from fastmcp.server.auth.providers.google import GoogleProvider
from fastmcp.server.dependencies import get_access_token  # claims integration point
print(sorted(inspect.signature(GoogleProvider.__init__).parameters))  # client_id/secret/base_url
```
Confirm the `GoogleProvider` kwargs in `main._build_auth` and the token-claims shape
`main._current_claims` reads (it tries `claims`/`extra`/`user`/direct `email`).

---

## Exit criterion
Connector added in claude.ai; `whoami` returns the caller's real verified
`@bwcoconsulting.com` email; a non-BW&CO / unverified Google account is refused.
Record the deployed URL and the org-owner steps in `docs/matcher-in-claude.md`
(plan §5.7) when Phase 5 starts.

## Feeds into Phase 5
- The OAuth/identity answers become `services/matcher_mcp/auth.py` (plan §5.2);
  `access_control.is_admin(email)` is made Streamlit-free there too.
- The response-size limit (Q4) sets every tool's output cap (plan §5.4).
- The SDK pin (Q5) is added to `requirements.matcher_mcp.txt`.
