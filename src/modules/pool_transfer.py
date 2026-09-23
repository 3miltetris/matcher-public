"""
Moving a company between pools
------------------------------
A targeted prospect that signs becomes a client. Nothing about the company
changes — the same contact rows, the same Deep Research output, the same
meeting digests, the same capability profile — only which pool holds it, so
promotion is a move of existing records rather than a re-import.

    contact rows   prospects/*.parquet  ->  clients/promoted_{date}_{hex6}.parquet
    profile row    prospect_profiles.parquet -> profiles.parquet

**Order matters: the destination is written first.** If the source rewrite
then fails, the company exists in both pools — visibly wrong and fixable by
deleting one side. The opposite order would lose the rows outright on the same
failure. This is why there is no pre-archive here, unlike client_delete.py:
nothing is destroyed until a copy is already stored somewhere else.

A key that already exists in the destination pool is reported as a conflict
and skipped, never merged: two rows for one company_key would make every
downstream join ambiguous.

Streamlit-free, like client_delete.py — the Client Records view drives it.
"""

import io
import uuid
from datetime import date, datetime, timezone

import pandas as pd

import src.modules.aspect_profile as ap
import src.modules.pools as pl
from src.modules.GoogleBucketManager.bucket_manager import BucketManager

BUCKET = ap.BUCKET


def _new_report(keys: list[str], source: str, dest: str) -> dict:
    return {
        'keys':             keys,
        'source_pool':      source,
        'dest_pool':        dest,
        'moved_keys':       [],
        'rows_moved':       0,
        'per_key':          {k: 0 for k in keys},
        'dest_blob':        '',
        'files_rewritten':  [],
        'files_deleted':    [],
        'profiles_moved':   [],
        'conflicts':        [],
        'notes':            [],
        'errors':           [],
    }


def move_companies(
    gcs_client,
    keys,
    *,
    source_pool: str,
    dest_pool: str,
    move_profile: bool = True,
    actor: str = '',
    bucket: str = BUCKET,
) -> dict:
    """Move companies from one pool to another. Returns a report dict;
    per-target failures land in `errors` rather than raising."""
    keys   = [k for k in dict.fromkeys(str(k) for k in keys) if k]
    report = _new_report(keys, source_pool, dest_pool)

    if source_pool == dest_pool:
        report['errors'].append('Source and destination pools are the same.')
        return report
    if not (pl.is_pool(source_pool) and pl.is_pool(dest_pool)):
        report['errors'].append('Unknown pool.')
        return report
    if not keys:
        report['errors'].append(f'No {pl.noun(source_pool)}s selected.')
        return report

    stamp      = datetime.now(timezone.utc).isoformat()
    bucket_obj = gcs_client.bucket(bucket)
    bm         = BucketManager(bucket, client=gcs_client)

    # ── Refuse keys the destination already holds ──────────────────────────
    dest_frames, dest_errors = pl.load_frames(gcs_client, dest_pool, bucket=bucket)
    report['errors'].extend(dest_errors)
    existing = set(pl.company_names(dest_frames))
    clash    = [k for k in keys if k in existing]
    if clash:
        report['conflicts'] = clash
        keys = [k for k in keys if k not in clash]
        if not keys:
            return report

    # ── Collect the rows to move ──────────────────────────────────────────
    plan: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []   # blob, kept, moved
    for blob in gcs_client.list_blobs(bucket, prefix=pl.contacts_prefix(source_pool)):
        if not blob.name.endswith('.parquet'):
            continue
        try:
            df = pl.normalize_company_columns(
                pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
            )
        except Exception as e:
            report['errors'].append(f'{blob.name}: unreadable, skipped ({e})')
            continue

        mask = pd.Series(False, index=df.index)
        for key in keys:
            m = pl.key_mask(df, key)
            n = int(m.sum())
            if n:
                report['per_key'][key] += n
                mask = mask | m
        if mask.any():
            plan.append((blob.name, df[~mask].copy(), df[mask].copy()))

    if not plan:
        report['notes'].append(
            f'No contact rows matched in {pl.contacts_prefix(source_pool)}.'
        )
    else:
        moved = pd.concat(
            [m.assign(pool_moved_from=source_pool, pool_moved_at=stamp,
                      pool_moved_by=actor or 'unknown')
             for _, _, m in plan],
            ignore_index=True,
        )
        dest_blob = (f'{pl.contacts_prefix(dest_pool)}promoted_'
                     f'{date.today().isoformat()}_{uuid.uuid4().hex[:6]}.parquet')
        try:
            bm.upload_file(dest_blob, moved)
        except Exception as e:
            # Nothing has been removed yet — stop before the source is touched.
            raise RuntimeError(
                f'Could not write the rows to {dest_blob} ({e}) — nothing was moved.'
            ) from e
        report['dest_blob']  = dest_blob
        report['rows_moved'] = len(moved)
        report['moved_keys'] = [k for k in keys if report['per_key'].get(k)]

        for blob_name, kept, moved_rows in plan:
            try:
                if kept.empty:
                    bucket_obj.blob(blob_name).delete()
                    report['files_deleted'].append(blob_name)
                else:
                    bm.upload_file(blob_name, kept.reset_index(drop=True))
                    report['files_rewritten'].append(blob_name)
            except Exception as e:
                # The rows are already in the destination, so this leaves a
                # duplicate rather than a hole. Say so loudly.
                report['errors'].append(
                    f'{blob_name}: rows copied to {dest_blob} but NOT removed '
                    f'from the source ({e}) — the company is now in both pools.'
                )

    # ── Profile row ───────────────────────────────────────────────────────
    if move_profile:
        try:
            src_profiles = ap.load_profiles(gcs_client, bucket=bucket, pool=source_pool)
            hits = []
            if not src_profiles.empty and 'company_key' in src_profiles.columns:
                hits = [
                    r for _, r in src_profiles.iterrows()
                    if str(r['company_key']) in set(keys)
                ]
            if hits:
                records = [
                    {c: r.get(c) for c in ap.PROFILE_COLUMNS} for r in hits
                ]
                dest_profiles = ap.load_profiles(gcs_client, bucket=bucket, pool=dest_pool)
                ap.save_profiles(
                    gcs_client, ap.upsert_profiles(dest_profiles, records, pool=dest_pool),
                    bucket=bucket, pool=dest_pool,
                )
                kept = src_profiles
                for r in hits:
                    kept = ap.delete_profile(kept, str(r['company_key']))
                ap.save_profiles(gcs_client, kept, bucket=bucket, pool=source_pool)
                report['profiles_moved'] = [str(r['company_key']) for r in hits]
        except Exception as e:
            report['errors'].append(f'{ap.profiles_blob(dest_pool)}: {e}')

    return report


def format_report(report: dict) -> str:
    """One-line-per-outcome markdown summary for the views."""
    src = pl.label(report['source_pool'])
    dst = pl.label(report['dest_pool'])
    lines = []
    if report['rows_moved']:
        lines.append(
            f"Moved **{report['rows_moved']}** contact row"
            f"{'s' if report['rows_moved'] != 1 else ''} "
            f"({len(report['moved_keys'])} compan"
            f"{'ies' if len(report['moved_keys']) != 1 else 'y'}) "
            f"from {src} to {dst}"
        )
    if report['profiles_moved']:
        lines.append(
            f"Moved **{len(report['profiles_moved'])}** capability profile(s)"
        )
    if report['dest_blob']:
        lines.append(f"New file: `{report['dest_blob']}`")
    if report['files_rewritten'] or report['files_deleted']:
        lines.append(
            f"{len(report['files_rewritten'])} source file(s) rewritten · "
            f"{len(report['files_deleted'])} emptied file(s) removed"
        )
    for key in report['conflicts']:
        lines.append(
            f"⚠️ `{key}` already exists in {dst} — skipped, nothing was moved for it"
        )
    return '\n\n'.join(f'- {line}' for line in lines) or '- Nothing was moved.'
