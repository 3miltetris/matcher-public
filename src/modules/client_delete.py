"""
Client deletion
---------------
Removes a company that is no longer a client from the pipeline:

  * its contact rows in `data/all-contacts/clients/` (parquets rewritten in
    place, or the blob deleted when it would be left empty),
  * its multi-aspect profile row in `data/client-profiles/profiles.parquet`,
  * its Drive Sync folder assignment(s), parked in `skipped` so the next scan
    neither syncs the folder nor proposes it as a brand-new client.

Deletion is irreversible in GCS, so every removed row is archived to
`data/deleted-clients/deleted_{date}_{hex6}.parquet` **before** anything is
rewritten — a mis-deleted client can be restored from that file. If the
archive write fails, nothing is deleted.

Streamlit-free. Shared by the Client Editor and Client Profiles views so both
delete paths behave identically; both gate it behind
`src/modules/access_control.py`.
"""

import io
import json
import uuid
from datetime import date, datetime, timezone

import pandas as pd

import src.modules.aspect_profile as ap
from src.modules.GoogleBucketManager.bucket_manager import BucketManager

# ── Constants ───────────────────────────────────────────────────────────────

BUCKET         = ap.BUCKET
CLIENTS_PREFIX = ap.CLIENTS_PREFIX
ARCHIVE_PREFIX = 'data/deleted-clients/'
ASSIGN_BLOB    = 'drive-sync-configs/assignments.json'


# ── Row matching ────────────────────────────────────────────────────────────

def key_mask(df: pd.DataFrame, key: str) -> pd.Series:
    """Rows of a clients parquet belonging to one `name||website` company key
    — the same identity used by aspect_profile.company_key()."""
    name, _, website = key.partition('||')
    names = (df.get('company_name', pd.Series('', index=df.index))
               .fillna('').astype(str).str.strip())
    if 'company_name' not in df.columns and 'companyName' in df.columns:
        names = df['companyName'].fillna('').astype(str).str.strip()
    sites = (df.get('companyWebsite', pd.Series('', index=df.index))
               .fillna('').astype(str).str.strip())
    return (names == name.strip()) & (sites == website.strip())


def count_rows(frames: dict[str, pd.DataFrame], keys) -> dict[str, int]:
    """{company_key: row count} across already-loaded frames — for the
    confirmation preview, so the UI does not re-download from GCS."""
    out = {k: 0 for k in keys}
    for df in frames.values():
        for key in keys:
            out[key] += int(key_mask(df, key).sum())
    return out


# ── Delete ──────────────────────────────────────────────────────────────────

def _new_report(keys: list[str]) -> dict:
    return {
        'keys':                  keys,
        'rows_deleted':          0,
        'per_key':               {k: 0 for k in keys},
        'files_rewritten':       [],
        'files_deleted':         [],
        'archive_blob':          '',
        'profiles_deleted':      [],
        'drive_folders_cleared': [],
        'notes':                 [],
        'errors':                [],
    }


def delete_clients(
    gcs_client,
    keys,
    *,
    delete_rows: bool = True,
    delete_profiles: bool = True,
    clear_drive_assignments: bool = True,
    actor: str = '',
    bucket: str = BUCKET,
) -> dict:
    """Delete one or more client companies. Returns a report dict; per-target
    failures are collected in `errors` rather than raised, except a failed
    archive write, which aborts before anything is destroyed.

    `delete_rows=False` removes only the profile (and assignment), leaving the
    contact rows in place."""
    keys   = [k for k in dict.fromkeys(str(k) for k in keys) if k]
    report = _new_report(keys)
    if not keys:
        report['errors'].append('No clients selected.')
        return report

    stamp      = datetime.now(timezone.utc).isoformat()
    bucket_obj = gcs_client.bucket(bucket)
    bm         = BucketManager(bucket, client=gcs_client)

    # ── Contact rows: plan → archive → apply ───────────────────────────────
    if delete_rows:
        plan: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []   # blob, kept, removed
        for blob in gcs_client.list_blobs(bucket, prefix=CLIENTS_PREFIX):
            if not blob.name.endswith('.parquet'):
                continue
            try:
                df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
            except Exception as e:
                report['errors'].append(f'{blob.name}: unreadable, skipped ({e})')
                continue

            mask = pd.Series(False, index=df.index)
            for key in keys:
                m = key_mask(df, key)
                n = int(m.sum())
                if n:
                    report['per_key'][key] += n
                    mask = mask | m
            if not mask.any():
                continue
            plan.append((blob.name, df[~mask].copy(), df[mask].copy()))

        if plan:
            removed = pd.concat(
                [r.assign(_deleted_from=b, _deleted_at=stamp, _deleted_by=actor or 'unknown')
                 for b, _, r in plan],
                ignore_index=True,
            )
            archive_blob = (f'{ARCHIVE_PREFIX}deleted_{date.today().isoformat()}'
                            f'_{uuid.uuid4().hex[:6]}.parquet')
            try:
                bm.upload_file(archive_blob, removed)
            except Exception as e:
                # Nothing has been written yet — fail loudly instead of
                # destroying rows with no way back.
                raise RuntimeError(
                    f'Could not archive the rows to be deleted ({e}) — nothing was deleted.'
                ) from e
            report['archive_blob'] = archive_blob

            for blob_name, kept, removed_rows in plan:
                try:
                    if kept.empty:
                        bucket_obj.blob(blob_name).delete()
                        report['files_deleted'].append(blob_name)
                    else:
                        bm.upload_file(blob_name, kept.reset_index(drop=True))
                        report['files_rewritten'].append(blob_name)
                    report['rows_deleted'] += len(removed_rows)
                except Exception as e:
                    report['errors'].append(f'{blob_name}: {e}')
        else:
            report['notes'].append(
                'No contact rows matched in data/all-contacts/clients/.'
            )

    # ── Profile store ──────────────────────────────────────────────────────
    if delete_profiles:
        try:
            profiles = ap.load_profiles(gcs_client, bucket=bucket)
            stored   = (set(profiles['company_key'].astype(str))
                        if not profiles.empty and 'company_key' in profiles.columns else set())
            hits = [k for k in keys if k in stored]
            if hits:
                merged = profiles
                for key in hits:
                    merged = ap.delete_profile(merged, key)
                ap.save_profiles(gcs_client, merged, bucket=bucket)
                report['profiles_deleted'] = hits
        except Exception as e:
            report['errors'].append(f'{ap.PROFILES_BLOB}: {e}')

    # ── Drive Sync assignments ─────────────────────────────────────────────
    if clear_drive_assignments:
        try:
            blob = bucket_obj.blob(ASSIGN_BLOB)
            if blob.exists():
                doc     = json.loads(blob.download_as_text())
                assigns = doc.get('assignments') or {}
                doc.setdefault('skipped', {})
                key_set = set(keys)
                cleared = []
                for fid, entry in list(assigns.items()):
                    entry = entry or {}
                    if str(entry.get('client_key') or '') in key_set:
                        # Parked as skipped, not dropped: an unknown folder gets
                        # re-proposed as a new client on the next Drive scan.
                        doc['skipped'][fid] = {
                            'folder_name': entry.get('folder_name', ''),
                            'section':     entry.get('section', ''),
                            'note':        f'client deleted {date.today().isoformat()}',
                        }
                        assigns.pop(fid, None)
                        cleared.append(fid)
                if cleared:
                    doc['assignments'] = assigns
                    doc['updated_at']  = stamp
                    blob.upload_from_string(json.dumps(doc), content_type='application/json')
                    report['drive_folders_cleared'] = cleared
        except Exception as e:
            report['errors'].append(f'{ASSIGN_BLOB}: {e}')

    return report


def format_report(report: dict) -> str:
    """One-line-per-outcome markdown summary for the views."""
    lines = []
    if report['rows_deleted'] or report['files_rewritten'] or report['files_deleted']:
        lines.append(
            f"Deleted **{report['rows_deleted']}** contact row"
            f"{'s' if report['rows_deleted'] != 1 else ''} · "
            f"{len(report['files_rewritten'])} file(s) rewritten · "
            f"{len(report['files_deleted'])} emptied file(s) removed"
        )
    if report['profiles_deleted']:
        lines.append(f"Deleted **{len(report['profiles_deleted'])}** aspect profile(s)")
    if report['drive_folders_cleared']:
        lines.append(
            f"Cleared **{len(report['drive_folders_cleared'])}** Drive Sync "
            f"folder assignment(s)"
        )
    if report['archive_blob']:
        lines.append(f"Backup of every removed row: `{report['archive_blob']}`")
    return '\n\n'.join(f'- {line}' for line in lines) or '- Nothing was deleted.'
