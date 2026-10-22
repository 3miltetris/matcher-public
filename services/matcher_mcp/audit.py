"""
Audit log for the Matcher MCP server (Phase 5.2)
------------------------------------------------
Every tool call appends one line to `mcp-audit/{date}.jsonl`:
`{ts, email, tool, args_hash, outcome, cost_usd}`. Arguments are HASHED, never
stored, because they can contain client material (a company name, a query). The
daily blob is written under a generation guard so two concurrent appends never
clobber each other, mirroring gcs_write.read_modify_write (which is parquet-only,
hence the small text-append variant here).

Best-effort: an audit failure is logged and swallowed — it must never break the
tool call it is recording.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from google.api_core.exceptions import PreconditionFailed

BUCKET = 'cc-matcher-bucket-jeg-v1'
_PREFIX = 'mcp-audit/'
_RETRIES = 5


def args_hash(args) -> str:
    """A stable sha256 of the call arguments — enough to correlate repeated calls
    without ever storing the arguments themselves."""
    try:
        blob = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        blob = repr(args)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def _blob(gcs_client, bucket: str):
    day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    return gcs_client.bucket(bucket).blob(f'{_PREFIX}{day}.jsonl')


def record(gcs_client, *, email: str, tool: str, args=None, outcome: str,
           cost_usd: float = 0.0, bucket: str = BUCKET) -> bool:
    """Append one audit line. Returns True on success; never raises."""
    line = json.dumps({
        'ts': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'email': email, 'tool': tool, 'args_hash': args_hash(args or {}),
        'outcome': outcome, 'cost_usd': round(float(cost_usd or 0.0), 6),
    }) + '\n'
    blob = _blob(gcs_client, bucket)
    for _ in range(_RETRIES):
        try:
            try:
                cur = blob.download_as_bytes()
                gen = blob.generation
            except Exception:
                cur, gen = b'', 0
            blob.upload_from_string(cur + line.encode('utf-8'),
                                    content_type='application/x-ndjson',
                                    if_generation_match=gen)
            return True
        except PreconditionFailed:
            continue  # someone else appended; re-read and retry
        except Exception as e:  # noqa: BLE001 — audit must not break the call
            print(f'[mcp-audit] WARN failed to record {tool}: {e}', flush=True)
            return False
    return False
