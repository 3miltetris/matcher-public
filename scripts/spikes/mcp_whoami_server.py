"""
Phase 0.3 spike — hello-world MCP server for a claude.ai custom connector.

THROWAWAY SCAFFOLD. Goal (PLAN-drive-onboarding-agent.md §0.3): stand up the
smallest remote MCP server claude.ai can add as a custom connector, with one
`whoami` tool that returns the signed-in user's verified BW&CO email — proving
the OAuth/identity path end to end before Phase 5 (`matcher-mcp`) is built.

HONEST SCOPE NOTE. The claude-api skill covers the Anthropic *client* side, not
building an MCP *server* for a connector, and the claude.ai connector OAuth flow
is a moving target. So the five questions in docs/spikes/mcp-connector.md are
**doc-research items** (support.claude.com custom-connectors / remote-MCP
articles + the MCP authorization spec), NOT things to write from memory. This
file is the transport + tool skeleton; the AUTH section is the integration point
you fill in from that research. Do not treat the stub as a security control.

Transport: streamable HTTP (what claude.ai expects). Built on the MCP Python SDK
(`mcp`) FastMCP server. Per CLAUDE.md "SDK version pins", when `mcp` is added to
a requirements file, pin it `>=<x>,<<x+1>` — decide the exact pin in 0.3 Q5.

Run locally:
    pip install "mcp[cli]" starlette uvicorn
    MCP_DEV=1 python scripts/spikes/mcp_whoami_server.py      # DEV identity, no OAuth
Then point the MCP Inspector at http://127.0.0.1:8080/mcp and call whoami.

Deploy (Cloud Run, public — claude.ai calls it from Anthropic infra, so IAP
cannot gate it; auth is OAuth on the server itself). Add the connector in
claude.ai as an ORG OWNER (open question #5 in the plan) and confirm whoami
returns your verified @bwcoconsulting.com address.
"""

from __future__ import annotations

import os

# ── Identity (the 0.3 research integration point) ────────────────────────────
ALLOWED_DOMAIN = "bwcoconsulting.com"


class Identity:
    """The verified caller. In production this comes from the OAuth layer:
    claude.ai completes an OAuth flow against our server (flow + redirect URI =
    0.3 Q1), and every tool call carries a verified Google ID token whose
    `email` + `email_verified` we check SERVER-SIDE (the `hd`/domain hint is a
    convenience, never the control — same rule as services/intake_web)."""

    def __init__(self, email: str, verified: bool):
        self.email = email
        self.verified = verified

    @property
    def allowed(self) -> bool:
        return (self.verified
                and self.email.lower().endswith("@" + ALLOWED_DOMAIN))


def resolve_identity() -> Identity:
    """Return the verified caller for the current request.

    TODO(0.3): replace the DEV stub with the real extraction once the research
    settles how claude.ai delivers the OAuth identity to the MCP server
    (bearer token in the request → verify Google ID token → email/email_verified).
    The MCP Python SDK exposes the authenticated context to tool handlers; wire
    this to that context. Until then, MCP_DEV=1 returns a fixed dev identity so
    the transport + tool can be exercised with the MCP Inspector.
    """
    if os.environ.get("MCP_DEV") == "1":
        return Identity(email=f"dev@{ALLOWED_DOMAIN}", verified=True)
    # Production path — not implemented in the spike; fail closed.
    return Identity(email="", verified=False)


# ── MCP server ───────────────────────────────────────────────────────────────
def build_server():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("matcher-mcp-spike")

    @mcp.tool()
    def whoami() -> str:
        """Return the verified BW&CO identity of the signed-in caller, or an
        access-denied message. This is the 0.3 exit criterion: added as a
        connector in claude.ai, it must return the caller's real
        @bwcoconsulting.com email."""
        ident = resolve_identity()
        if not ident.allowed:
            return ("access denied: a verified @bwcoconsulting.com Google "
                    "identity is required (got "
                    f"{ident.email or 'no identity'}, verified={ident.verified})")
        return f"signed in as {ident.email} (verified BW&CO)"

    return mcp


if __name__ == "__main__":
    server = build_server()
    # FastMCP serves streamable HTTP at /mcp. Host/port are picked up from the
    # environment on Cloud Run (PORT); default to 8080 locally.
    os.environ.setdefault("FASTMCP_PORT", os.environ.get("PORT", "8080"))
    os.environ.setdefault("FASTMCP_HOST", "0.0.0.0")
    server.run(transport="streamable-http")
