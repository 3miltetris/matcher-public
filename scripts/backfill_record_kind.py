"""
backfill_record_kind.py — one-off: reclassify existing grant topics and backfill
the solicitation link.

A coworker review found the matcher's top matches dominated by SAM.gov *supply
buys*, *sole-source / notice-of-intent-to-award* notices and already-awarded
contracts — all stored as ordinary solicitations because the ingest screen never
excluded them and the classification fields were dropped. Going forward the SAM
screen rejects them and the parquet keeps `notice_type`/`set_aside`; this script
cleans the rows already in the store.

For each parquet under the target prefix it:
  - sets `record_kind` (solicitation / procurement / award) via
    `grant_utils.classify_record_kind`, so match-time readers can filter
    procurement out, and
  - backfills a blank `source` from the stored SAM `notice_version_id`
    (`https://sam.gov/opp/{id}/view`) so the solicitation link is present.

It is PATTERN-BASED: the dropped `notice_type`/`set_aside` cannot be recovered
from the stored row, so supply buys whose summary was laundered into R&D prose
and whose title has no PSC code are not caught here — only re-fetching from
SAM.gov would recover those, which is a separate, quota-spending pass. This
reclassifies what is deterministically identifiable and is safe to re-run.

Dry-run by default (reports counts, writes nothing). Pass --apply to write.

    python -m scripts.backfill_record_kind                 # dry run, SAM-GOV
    python -m scripts.backfill_record_kind --all           # dry run, every folder
    python -m scripts.backfill_record_kind --apply         # write SAM-GOV
    python -m scripts.backfill_record_kind --all --apply   # write every folder

Needs GCS credentials (ADC: `gcloud auth application-default login`, or
GOOGLE_APPLICATION_CREDENTIALS).
"""

import argparse
import io

import pandas as pd
from google.cloud import storage

from src.modules import grant_utils as gu
from src.modules import gcs_write

BUCKET = 'cc-matcher-bucket-jeg-v1'
PROCESSED_PREFIX = 'data/all-topics/processed/'
SAM_PREFIX = 'data/all-topics/processed/SAM-GOV/'


def _reconstruct_source(row) -> str:
    """A SAM.gov opp URL from the stored version id, when source is blank."""
    nid = str(row.get('notice_version_id', '') or '').strip()
    if nid and nid.lower() not in ('nan', 'none', ''):
        return f'https://sam.gov/opp/{nid}/view'
    return ''


def _process_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (new_df, stats) — new_df has refreshed record_kind + backfilled
    source. Never mutates the input."""
    out = df.copy()
    before_kind = (out['record_kind'].astype(str).tolist()
                   if 'record_kind' in out.columns else [''] * len(out))
    out = gu.add_record_kind(out)
    after_kind = out['record_kind'].tolist()

    source_filled = 0
    if 'source' not in out.columns:
        out['source'] = ''
    blank = out['source'].map(
        lambda v: str(v).strip().lower() in ('', 'nan', 'none', 'null', '<na>'))
    for i in out.index[blank]:
        url = _reconstruct_source(out.loc[i])
        if url:
            out.at[i, 'source'] = url
            source_filled += 1
        else:
            out.at[i, 'source'] = ''   # scrub the junk token even if unrecoverable

    stats = {
        'rows': len(out),
        'to_procurement': sum(1 for b, a in zip(before_kind, after_kind)
                              if a == 'procurement' and b != 'procurement'),
        'to_award': sum(1 for b, a in zip(before_kind, after_kind)
                        if a == 'award' and b != 'award'),
        'source_filled': source_filled,
    }
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--all', action='store_true',
                    help='every processed/ folder (default: SAM-GOV only)')
    ap.add_argument('--apply', action='store_true',
                    help='write changes back (default: dry run)')
    ap.add_argument('--project', default=None, help='GCP project for the client')
    args = ap.parse_args()

    client = storage.Client(project=args.project) if args.project else storage.Client()
    prefix = PROCESSED_PREFIX if args.all else SAM_PREFIX

    totals = {'files': 0, 'rows': 0, 'to_procurement': 0, 'to_award': 0,
              'source_filled': 0, 'files_changed': 0}
    for blob in client.list_blobs(BUCKET, prefix=prefix):
        if not blob.name.endswith('.parquet'):
            continue
        df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        if df.empty:
            continue
        new_df, stats = _process_frame(df)
        changed = (stats['to_procurement'] or stats['to_award']
                   or stats['source_filled'])
        totals['files'] += 1
        for k in ('rows', 'to_procurement', 'to_award', 'source_filled'):
            totals[k] += stats[k]
        if changed:
            totals['files_changed'] += 1
            print(f"{'WOULD WRITE' if not args.apply else 'WRITING'} "
                  f"{blob.name}: +{stats['to_procurement']} procurement, "
                  f"+{stats['to_award']} award, {stats['source_filled']} source links")
            if args.apply:
                # Generation-guarded: the classify + source backfill is derived
                # purely from the frame, so re-running it on the freshest copy is
                # correct if a revision-check run rewrote this blob meanwhile.
                gcs_write.read_modify_write(
                    blob, lambda d: _process_frame(d)[0] if d is not None else new_df)

    print('\n-- summary --')
    print(f"  files scanned        {totals['files']}")
    print(f"  rows scanned         {totals['rows']}")
    print(f"  -> procurement       {totals['to_procurement']}")
    print(f"  -> award             {totals['to_award']}")
    print(f"  source links filled  {totals['source_filled']}")
    print(f"  files {'changed' if args.apply else 'to change'}         "
          f"{totals['files_changed']}")
    if not args.apply:
        print('\n  dry run — nothing written. Re-run with --apply to write.')


if __name__ == '__main__':
    main()
