"""
Identity gate for the Matcher MCP server (Phase 5.2)
----------------------------------------------------
The server is public (claude.ai calls it from Anthropic's infrastructure, so IAP
cannot gate it — docs/spikes/mcp-connector.md). Auth is OAuth on the server
itself: FastMCP's `GoogleProvider` bridges Google (which does not do Dynamic
Client Registration) to Claude (which expects it), runs the Google login, and
issues the server's own JWT. GoogleProvider by itself admits ANY Google account,
so this module adds the BW&CO restriction on top: a caller is accepted only when
the verified ID-token `email` ends in @bwcoconsulting.com AND `email_verified` is
true, checked server-side — the `hd` hint is a convenience, never the control
(the same rule as services/intake_web).

Pure and Streamlit-free: the FastMCP auth dependency is injected by main.py as a
`claims` dict, so the gate itself is unit-tested without standing up OAuth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ALLOWED_DOMAIN = 'bwcoconsulting.com'


@dataclass(frozen=True)
class Actor:
    """The verified caller behind a tool invocation."""
    email: str
    verified: bool

    @property
    def allowed(self) -> bool:
        return self.verified and self.email.lower().endswith('@' + ALLOWED_DOMAIN)


def is_bwco(email: str, verified: bool) -> bool:
    return Actor(str(email or ''), bool(verified)).allowed


def claims_to_actor(claims: dict | None) -> Actor:
    """Map a verified Google ID-token claims dict to an Actor. We trust `email`
    only when `email_verified` is truthy; the hosted-domain `hd` claim is NOT a
    substitute (a personal Google account can set no `hd`, and `hd` alone does
    not prove the address). Returns a disallowed Actor for missing/garbled
    claims — fail closed."""
    claims = claims or {}
    email = str(claims.get('email', '') or '')
    verified = claims.get('email_verified')
    verified = bool(verified) if not isinstance(verified, str) \
        else verified.strip().lower() == 'true'
    return Actor(email=email, verified=verified)


class AccessDenied(Exception):
    """Raised when a non-BW&CO / unverified caller reaches a tool."""


def resolve_actor(get_claims=None) -> Actor:
    """The caller for the current request.

    DEV (`MCP_DEV=1`): a fixed verified BW&CO identity, so the transport and
    tools can be exercised with the MCP Inspector before OAuth is wired.

    PROD: `get_claims()` returns the verified Google claims FastMCP extracted from
    the request's bearer token (main.py injects this from the FastMCP auth
    dependency — the single integration point, verified against the installed
    fastmcp on first deploy). Fail closed if it yields nothing.
    """
    if os.environ.get('MCP_DEV') == '1':
        return Actor(email=f'dev@{ALLOWED_DOMAIN}', verified=True)
    try:
        claims = get_claims() if get_claims else None
    except Exception:
        claims = None
    return claims_to_actor(claims)


def require_actor(get_claims=None) -> Actor:
    """resolve_actor, but raise AccessDenied unless the caller is a verified
    BW&CO identity. Every tool calls this first."""
    actor = resolve_actor(get_claims)
    if not actor.allowed:
        raise AccessDenied(
            'a verified @bwcoconsulting.com Google identity is required '
            f'(got {actor.email or "no identity"}, verified={actor.verified})')
    return actor
