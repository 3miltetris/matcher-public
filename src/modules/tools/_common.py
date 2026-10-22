"""
Shared helpers for the transport-agnostic tool layer (Phase 5.1)
----------------------------------------------------------------
Every Tier-1 tool returns a small JSON-serialisable dict. Two things every tool
needs and must do identically: cap the size of any free-text field (claude.ai
rejects an oversized tool result — see docs/spikes/mcp-connector.md Q4), and
page large result sets with an opaque cursor rather than returning a whole
parquet. These live here so a tool is written once and behaves the same whether
it is called by the MCP server or any other caller.

Streamlit-free, stdlib-only.
"""

from __future__ import annotations

import base64
import json

# The hard cap on a single tool result's free text, from the plan (§5.4) and the
# connector research (Q4). Individual tools cap their own long fields below this
# so the assembled result stays well under whatever claude.ai enforces.
MAX_RESULT_CHARS = 20_000


def cap_text(text, limit: int = 2_000) -> str:
    """Truncate a single free-text field MIDDLE-OUT (keep the head and the tail,
    drop the middle) and say so, so neither the opening nor the closing context
    is lost. Mirrors strategy_session.cap_middle / aspect_matching's row capping.
    Non-strings are coerced; None becomes ''."""
    s = '' if text is None else str(text)
    if len(s) <= limit:
        return s
    if limit <= 32:
        return s[:limit]
    keep = limit - 20
    head = keep // 2
    tail = keep - head
    return f'{s[:head]}\n…[{len(s) - keep} chars elided]…\n{s[-tail:]}'


def encode_cursor(offset: int) -> str:
    """An opaque pagination cursor over a 0-based offset."""
    raw = json.dumps({'o': int(offset)}).encode('utf-8')
    return base64.urlsafe_b64encode(raw).decode('ascii')


def decode_cursor(cursor) -> int:
    """The offset a cursor points at; 0 for a missing/garbled cursor (so a bad
    cursor restarts from the top rather than erroring the whole call)."""
    if not cursor:
        return 0
    try:
        raw = base64.urlsafe_b64decode(str(cursor).encode('ascii'))
        return max(0, int(json.loads(raw).get('o', 0)))
    except Exception:
        return 0
