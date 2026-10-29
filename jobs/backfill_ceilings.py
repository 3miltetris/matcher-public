"""
backfill_ceilings.py — one-off, idempotent backfill of the verified SBIR/STTR
ceiling columns onto every existing processed grant-topic parquet.

Why a durable one-off rewrite rather than a load-time fill: the agency-table
figures are what let the roadmap (which matches against *all* processed parquets)
show typical dollars immediately, and a rewrite stores them once instead of
recomputing on every read. Writes go through `gcs_write.read_modify_write`, so a
concurrent ingest writing the same blob is never clobbered.

Resolution per row is `sbir_ceilings.resolve_ceilings` using only the row's own
stored fields (agency, title, grant_summary, and any stated funding_amount /
award_ceiling text) — so SBIR/STTR rows get their agency phase caps, a row with a
stated dollar figure gets that exact amount, and a non-SBIR row gets nothing.

Rows that already carry an exact per-notice source ('grants_gov' / 'stated_text',
written by the live ingest path) are **preserved** — the backfill has no synopsis
to re-derive them from and must never downgrade them to a table figure. Re-running
is therefore safe and deterministic.

Usage (from the repo root / in the job image):
    python -m jobs.backfill_ceilings [--dry-run] [--prefix FOLDER] [--limit N]

--dry-run reports per-folder counts without writing. --prefix restricts to one
broad-agency folder (e.g. GRANTS-GOV). Awards (data/all-topics/awards/) are skipped
— a completed award has no Phase I/II ceiling to assign.
"""

from __future__ import annotations

import argparse
import io
import sys

import pandas as pd
from google.cloud import storage

import src.modules.grant_utils as gu
import src.modules.gcs_write as gw
import src.modules.sbir_ceilings as sc

BUCKET = 'cc-matcher-bucket-jeg-v1'
PROCESSED_PREFIX = 'data/all-topics/processed/'

# Per-notice sources the backfill must not overwrite (it cannot re-derive them).
_PRESERVE = frozenset({'grants_gov', 'stated_text'})


def _broad_agency(blob_name: str) -> str:
    """The folder under processed/ a parquet lives in, e.g. 'DOD', 'GRANTS-GOV'."""
    rest = blob_name[len(PROCESSED_PREFIX):]
    return rest.split('/', 1)[0] if '/' in rest else ''


def _col_list(df: pd.DataFrame, name: str, n: int) -> list:
    return df[name].astype(str).tolist() if name in df.columns else [''] * n


def _eq(a, b) -> bool:
    """Equal, treating None and NaN as the same missing value — so a value that
    round-tripped through parquet as NaN is not seen as a change vs a fresh None."""
    a_na = a is None or (isinstance(a, float) and a != a)
    b_na = b is None or (isinstance(b, float) and b != b)
    if a_na or b_na:
        return a_na and b_na
    return a == b


def _apply_ceilings(df: pd.DataFrame, broad_agency: str) -> tuple[pd.DataFrame, int]:
    """Set the six ceiling columns in place; return (df, rows_changed). Preserves
    rows whose existing ceiling_source is an exact per-notice source."""
    if df is None or df.empty:
        return (df if df is not None else pd.DataFrame()), 0
    df = gu.ensure_ceiling_columns(df)
    n = len(df)
    agency  = _col_list(df, 'agency', n)
    title   = _col_list(df, 'title', n)
    summ    = _col_list(df, 'grant_summary', n)
    fa      = _col_list(df, 'funding_amount', n)
    ac      = _col_list(df, 'award_ceiling', n)
    old_src = _col_list(df, 'ceiling_source', n)

    cols = {c: list(df[c]) for c in sc.CEILING_COLUMNS}
    changed = 0
    for i in range(n):
        if old_src[i] in _PRESERVE:          # keep exact per-notice figures
            continue
        stated = fa[i] if fa[i].strip() else ac[i]
        res = sc.resolve_ceilings(
            broad_agency=broad_agency, agency=agency[i],
            program_text=f'{title[i]} {summ[i]}',
            funding_amount_text=(stated or None),
        )
        row_changed = False
        for c in sc.CEILING_COLUMNS:
            if not _eq(cols[c][i], res[c]):
                cols[c][i] = res[c]
                row_changed = True
        if row_changed:
            changed += 1
    for c in sc.CEILING_COLUMNS:
        df[c] = cols[c]
    return df, changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--prefix', default='', help='restrict to one broad-agency folder')
    ap.add_argument('--limit', type=int, default=0, help='process at most N parquets')
    args = ap.parse_args()

    client = storage.Client()
    bucket = client.bucket(BUCKET)
    prefix = PROCESSED_PREFIX + (args.prefix.strip('/') + '/' if args.prefix else '')
    blobs = [b for b in client.list_blobs(BUCKET, prefix=prefix)
             if b.name.endswith('.parquet')]
    if args.limit:
        blobs = blobs[:args.limit]

    print(f'{"DRY-RUN " if args.dry_run else ""}backfill over {len(blobs)} parquet(s) '
          f'under {prefix}', flush=True)

    totals = {'files': 0, 'rows': 0, 'rows_changed': 0, 'files_written': 0, 'errors': 0}
    per_folder: dict[str, int] = {}
    for blob in blobs:
        folder = _broad_agency(blob.name)
        try:
            if args.dry_run:
                df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
                _, changed = _apply_ceilings(df, folder)
                wrote = False
            else:
                captured = {'changed': 0}

                def _mutate(df, _folder=folder, _cap=captured):
                    out, ch = _apply_ceilings(df, _folder)
                    _cap['changed'] = ch
                    return out

                gw.read_modify_write(bucket.blob(blob.name), _mutate)
                changed = captured['changed']
                df = None
                wrote = changed > 0
            totals['files'] += 1
            totals['rows_changed'] += changed
            per_folder[folder] = per_folder.get(folder, 0) + changed
            if wrote:
                totals['files_written'] += 1
            print(f'  {blob.name}: {changed} row(s) {"would change" if args.dry_run else "changed"}',
                  flush=True)
        except Exception as e:          # one bad parquet must not abort the sweep
            totals['errors'] += 1
            print(f'  ERROR {blob.name}: {e}', flush=True)

    print('\nSummary:', flush=True)
    for folder in sorted(per_folder):
        print(f'  {folder}: {per_folder[folder]} row(s) changed', flush=True)
    print(f'  files={totals["files"]} files_written={totals["files_written"]} '
          f'rows_changed={totals["rows_changed"]} errors={totals["errors"]}', flush=True)
    return 1 if totals['errors'] else 0


if __name__ == '__main__':
    sys.exit(main())
