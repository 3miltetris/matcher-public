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
_Answer:_

### 2 — Transport + protocol version
Which MCP transport and protocol version does claude.ai expect? (Streamable HTTP
is assumed; the scaffold serves `/mcp`.) Record the exact protocol version string.
_Answer:_

### 3 — Org-wide install + plugin packaging
Can an org owner add the connector once for the whole organization? Can it be
packaged in a plugin alongside the `bwco-matcher` skill (plan §5.7), or must the
connector and skill be added separately?
_Answer:_

### 4 — Response-size limits per tool result
What is the max size of a single tool result claude.ai will accept? This sets the
hard cap for every Phase 5 tool's output (plan §5.4 assumes a ~20k-char default,
truncate-middle-out).
_Answer:_

### 5 — MCP Python SDK version to pin
Which `mcp` SDK version should we pin? Per CLAUDE.md **"SDK version pins"**, pin
`mcp>=<x.y>,<<x+1>` with an explicit upper bound before adding it to any
`requirements*.txt`, and record the signature check used (the same discipline as
the `anthropic`/`openai` pins).
_Answer:_

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
