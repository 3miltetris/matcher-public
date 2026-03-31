"""
matching_job.py — Cloud Run Job entry point.

Reads a job config JSON from GCS, runs the full matching pipeline
(vector scoring → AI validation → email pre-write), saves 1 000-row
CSV segments to GCS, then writes a status.json when finished.

Invoked as:
    python jobs/matching_job.py <config_blob_path>
e.g.
    python jobs/matching_job.py job-configs/2026-03-31_14-00-00.json

Secrets come from Google Secret Manager — no st.secrets, no .env files.
The job's service account must have:
  - roles/storage.objectAdmin       (GCS read/write)
  - roles/secretmanager.secretAccessor
"""

import asyncio
import gc
import io
import json
import random
import sys
import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from anthropic import AsyncAnthropic
from google.cloud import secretmanager, storage
from openai import AsyncOpenAI

# ── src modules are on the path because the Dockerfile sets PYTHONPATH ────────
from src.modules.email_generator import async_generate_subject_line, async_josiah_copy

# ── Constants ────────────────────────────────────────────────────────────────

_BUCKET           = 'cc-matcher-bucket-jeg-v1'
_CONTACTS_PREFIX  = 'data/all-contacts/'
_TOPICS_PREFIX    = 'data/all-topics/processed/'
_RESULTS_PREFIX   = 'matching-results/'
_SEGMENT_SIZE     = 1000
_VALIDATION_BATCH = 10
_EMAIL_BATCH      = 5
_SCORE_CHUNK      = 2000
_MAX_RETRIES      = 5
_VALIDATION_SYSTEM = (
    'Tell me if this company summary and grant summary are aligned. '
    'Only give a one-word answer. Either "yes" or "no".'
)


# ── Secret Manager ────────────────────────────────────────────────────────────

def _get_secret(secret_id: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    # Project is inferred from the service account's ADC
    import google.auth
    _, project = google.auth.default()
    name = f'projects/{project}/secrets/{secret_id}/versions/latest'
    return client.access_secret_version(request={'name': name}).payload.data.decode()


# ── GCS helpers ───────────────────────────────────────────────────────────────

def _gcs() -> storage.Client:
    return storage.Client()   # uses ADC — the job's attached service account


def _load_topics(client: storage.Client, agencies: list[str]) -> pd.DataFrame:
    frames = []
    for agency in agencies:
        prefix = f'{_TOPICS_PREFIX}{agency}/'
        for blob in client.list_blobs(_BUCKET, prefix=prefix):
            if blob.name.endswith('.parquet'):
                df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
                df['broad_agency'] = agency
                frames.append(df)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return pd.concat(frames, ignore_index=True)


def _list_contact_blobs(client: storage.Client, sources: list[str]) -> list[tuple]:
    result = []
    for source in sources:
        for blob in client.list_blobs(_BUCKET, prefix=f'{_CONTACTS_PREFIX}{source}/'):
            if blob.name.endswith('.parquet'):
                result.append((source, blob))
    return result


def _upload_csv(client: storage.Client, df: pd.DataFrame, blob_path: str) -> None:
    client.bucket(_BUCKET).blob(blob_path).upload_from_string(
        df.to_csv(index=False).encode('utf-8'), content_type='text/csv'
    )


def _write_status(client: storage.Client, results_prefix: str, payload: dict) -> None:
    client.bucket(_BUCKET).blob(f'{results_prefix}status.json').upload_from_string(
        json.dumps(payload), content_type='application/json'
    )


# ── Async helpers ─────────────────────────────────────────────────────────────

async def _validate_rows(
    rows: list[tuple[int, dict]],
    anth_key: str,
) -> list[tuple[int, str]]:
    async with AsyncAnthropic(api_key=anth_key) as client:
        async def _one(idx: int, row: dict) -> tuple[int, str]:
            for attempt in range(_MAX_RETRIES):
                try:
                    msg = await client.messages.create(
                        model='claude-3-haiku-20240307',
                        max_tokens=15,
                        temperature=0,
                        system=_VALIDATION_SYSTEM,
                        messages=[{
                            'role': 'user',
                            'content': (
                                f"company summary: {row.get('company_summary', '')}\n\n"
                                f"grant summary: {row.get('grant_summary', '')}"
                            ),
                        }],
                    )
                    return idx, msg.content[0].text.strip().lower()
                except Exception as e:
                    err = str(e)
                    if any(x in err for x in ('529', '429', 'overloaded', 'rate_limit', 'rate limit')):
                        await asyncio.sleep((2 ** attempt) + random.random())
                    else:
                        raise
            return idx, 'no'

        return await asyncio.gather(*[_one(idx, row) for idx, row in rows])


async def _generate_email_batch(
    rows: list[tuple[int, dict]],
    openai_key: str,
    anth_key: str,
) -> list[tuple[int, str, str]]:
    async with AsyncOpenAI(api_key=openai_key) as oai, \
               AsyncAnthropic(api_key=anth_key) as anth:
        async def _one(idx: int, row: dict) -> tuple[int, str, str]:
            subject, body = await asyncio.gather(
                async_generate_subject_line(
                    company_summary=str(row.get('company_summary', '') or ''),
                    agency=str(row.get('agency', row.get('broad_agency', '')) or ''),
                    openai_client=oai,
                    anth_client=anth,
                ),
                async_josiah_copy(
                    company_summary=str(row.get('company_summary', '') or ''),
                    grant_summary=str(row.get('grant_summary', '') or ''),
                    word_limit=50,
                    anth_client=anth,
                ),
            )
            return idx, subject, body

        return await asyncio.gather(*[_one(idx, row) for idx, row in rows])


# ── Segment flush ─────────────────────────────────────────────────────────────

def _flush_segment(
    rows: list[dict],
    seg_num: int,
    results_prefix: str,
    gcs_client: storage.Client,
    ai_validation: bool,
    prewrite_email: bool,
    anth_key: str,
    openai_key: str,
    seen_websites: dict[str, str],
) -> int:
    """
    Build DataFrame from rows, validate, optionally generate emails,
    upload CSV to GCS. Returns number of rows saved (0 if all filtered).
    seen_websites is mutated in-place for cross-segment dedup.
    """
    segment = pd.DataFrame(rows)

    for col in ('scraped_at', 'open_date', 'close_date'):
        if col in segment.columns:
            segment[col] = segment[col].astype(str)

    # ── AI validation ──────────────────────────────────────────────────────
    if ai_validation:
        segment = segment.copy()
        segment['good_match'] = None

        website_map:  dict[str, list[int]] = {}
        unique_tasks: list[tuple[int, dict]] = []

        for idx, row in segment.iterrows():
            site = str(row.get('companyWebsite', '') or '').strip()
            if site and site in seen_websites:
                segment.at[idx, 'good_match'] = seen_websites[site]
                continue
            slim = {
                'company_summary': row.get('company_summary', ''),
                'grant_summary':   row.get('grant_summary',   ''),
            }
            if site:
                if site not in website_map:
                    website_map[site] = []
                    unique_tasks.append((idx, slim))
                website_map[site].append(idx)
            else:
                unique_tasks.append((idx, slim))

        total_tasks   = len(unique_tasks)
        idx_to_result: dict[int, str] = {}

        for i in range(0, total_tasks, _VALIDATION_BATCH):
            batch   = unique_tasks[i : i + _VALIDATION_BATCH]
            results = asyncio.run(_validate_rows(batch, anth_key))
            idx_to_result.update(results)
            done = i + len(batch)
            print(f'  segment {seg_num} · validated {done}/{total_tasks}', flush=True)

        for site, indices in website_map.items():
            result = idx_to_result.get(indices[0], 'no')
            seen_websites[site] = result
            for idx in indices:
                segment.at[idx, 'good_match'] = result
        for idx in segment.index:
            if segment.at[idx, 'good_match'] is None:
                segment.at[idx, 'good_match'] = idx_to_result.get(idx, 'no')

        del unique_tasks, website_map, idx_to_result
        gc.collect()

        segment = segment[
            segment['good_match'].str.contains('yes', na=False)
        ].reset_index(drop=True)

    if segment.empty:
        print(f'  segment {seg_num} · 0 rows after validation — skipping', flush=True)
        return 0

    # ── Email pre-write ────────────────────────────────────────────────────
    if prewrite_email:
        segment = segment.copy()
        segment['subject_line'] = None
        segment['ai_message']   = None

        email_rows = [(idx, row.to_dict()) for idx, row in segment.iterrows()]
        total      = len(email_rows)

        for i in range(0, total, _EMAIL_BATCH):
            batch   = email_rows[i : i + _EMAIL_BATCH]
            results = asyncio.run(_generate_email_batch(batch, openai_key, anth_key))
            for idx, subject, body in results:
                segment.at[idx, 'subject_line'] = subject
                segment.at[idx, 'ai_message']   = body
            done = i + len(batch)
            print(f'  segment {seg_num} · emails {done}/{total}', flush=True)

        del email_rows
        gc.collect()

    # ── Upload ─────────────────────────────────────────────────────────────
    blob_path = f'{results_prefix}segment_{seg_num:03d}.csv'
    _upload_csv(gcs_client, segment, blob_path)
    row_count = len(segment)
    print(f'  segment {seg_num} · saved {row_count} rows → {blob_path}', flush=True)

    del segment
    gc.collect()
    return row_count


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_blob_path: str) -> None:
    gcs_client = _gcs()

    # ── Load config ────────────────────────────────────────────────────────
    print(f'Loading config from {config_blob_path}', flush=True)
    config = json.loads(
        gcs_client.bucket(_BUCKET).blob(config_blob_path).download_as_text()
    )

    run_id          = config['run_id']
    threshold       = float(config['threshold'])
    top_k           = int(config['top_k'])
    sources         = config['sources']
    agencies        = config['agencies']
    ai_validation   = bool(config.get('ai_validation', True))
    prewrite_email  = bool(config.get('prewrite_email', False))
    results_prefix  = f'{_RESULTS_PREFIX}{run_id}/'

    # ── Secrets ────────────────────────────────────────────────────────────
    anth_key   = _get_secret('anthropic-api-key')
    openai_key = _get_secret('openai-api-key') if prewrite_email else ''

    # ── Load topics ────────────────────────────────────────────────────────
    print('Loading grant topics…', flush=True)
    topics_df = _load_topics(gcs_client, agencies)
    if topics_df.empty:
        raise RuntimeError('No topic parquet files found for selected agencies.')

    if 'grant_summary' not in topics_df.columns and 'description' in topics_df.columns:
        topics_df = topics_df.rename(columns={'description': 'grant_summary'})

    grant_embeddings = np.stack(topics_df['embeddings'].values).astype(np.float32)
    topics_df        = topics_df.drop(columns=['embeddings'])

    grant_cols = ['topic_number', 'title', 'agency', 'broad_agency', 'due_date', 'grant_summary']
    grant_meta = topics_df[[c for c in grant_cols if c in topics_df.columns]].reset_index(drop=True)
    del topics_df
    gc.collect()
    print(f'  {len(grant_meta):,} topics loaded', flush=True)

    # ── List contact blobs ─────────────────────────────────────────────────
    print('Listing contact files…', flush=True)
    blob_list = _list_contact_blobs(gcs_client, sources)
    if not blob_list:
        raise RuntimeError('No contact parquet files found for selected sources.')
    print(f'  {len(blob_list)} contact file(s)', flush=True)

    # ── Stream contacts → score in chunks → flush segments inline ─────────
    match_buffer:   list[dict]     = []
    seen_websites:  dict[str, str] = {}
    seg_num         = 0
    total_candidates = 0
    total_saved      = 0

    for file_i, (source, blob) in enumerate(blob_list):
        print(f'[{file_i+1}/{len(blob_list)}] {blob.name}', flush=True)
        df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        df = df[df['embeddings'].notna()].reset_index(drop=True)
        if df.empty:
            continue

        if 'company_summary' not in df.columns and 'summary' in df.columns:
            df = df.rename(columns={'summary': 'company_summary'})

        for chunk_start in range(0, len(df), _SCORE_CHUNK):
            chunk            = df.iloc[chunk_start : chunk_start + _SCORE_CHUNK]
            chunk_embeddings = np.stack(chunk['embeddings'].values).astype(np.float32)
            scores           = np.dot(chunk_embeddings, grant_embeddings.T)
            del chunk_embeddings

            for ci in range(len(chunk)):
                contact_scores = scores[ci]
                above = np.where(contact_scores >= threshold)[0]
                if len(above) == 0:
                    continue
                top_indices = above[np.argsort(contact_scores[above])[::-1][:top_k]]
                contact_row = chunk.iloc[ci]
                for gi in top_indices:
                    row = {
                        'companyName':     str(contact_row.get('companyName', '') or contact_row.get('company_name', '') or ''),
                        'companyWebsite':  str(contact_row.get('companyWebsite', '') or ''),
                        'firstName':       str(contact_row.get('firstName',      '') or ''),
                        'lastName':        str(contact_row.get('lastName',       '') or ''),
                        'email':           str(contact_row.get('email',          '') or ''),
                        'company_summary': str(contact_row.get('company_summary', '') or ''),
                        'source':          source,
                    }
                    for col in grant_meta.columns:
                        row[col] = grant_meta.iloc[gi].get(col, '')
                    match_buffer.append(row)
                    total_candidates += 1

                if len(match_buffer) >= _SEGMENT_SIZE:
                    seg_num += 1
                    print(f'Flushing segment {seg_num} ({_SEGMENT_SIZE} rows)…', flush=True)
                    saved = _flush_segment(
                        rows           = match_buffer[:_SEGMENT_SIZE],
                        seg_num        = seg_num,
                        results_prefix = results_prefix,
                        gcs_client     = gcs_client,
                        ai_validation  = ai_validation,
                        prewrite_email = prewrite_email,
                        anth_key       = anth_key,
                        openai_key     = openai_key,
                        seen_websites  = seen_websites,
                    )
                    total_saved  += saved
                    match_buffer  = match_buffer[_SEGMENT_SIZE:]

            del scores

        del df
        gc.collect()

    # ── Flush remainder ────────────────────────────────────────────────────
    if match_buffer:
        seg_num += 1
        print(f'Flushing final segment {seg_num} ({len(match_buffer)} rows)…', flush=True)
        saved = _flush_segment(
            rows           = match_buffer,
            seg_num        = seg_num,
            results_prefix = results_prefix,
            gcs_client     = gcs_client,
            ai_validation  = ai_validation,
            prewrite_email = prewrite_email,
            anth_key       = anth_key,
            openai_key     = openai_key,
            seen_websites  = seen_websites,
        )
        total_saved += saved

    del grant_embeddings, grant_meta, seen_websites
    gc.collect()

    print(
        f'\nDone. {total_candidates:,} candidates → {total_saved:,} rows saved '
        f'across {seg_num} segment(s).',
        flush=True,
    )
    _write_status(gcs_client, results_prefix, {
        'run_id':           run_id,
        'total_candidates': total_candidates,
        'total_saved':      total_saved,
        'segments':         seg_num,
        'error':            None,
    })


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python jobs/matching_job.py <config_blob_path>', file=sys.stderr)
        sys.exit(1)

    try:
        main(sys.argv[1])
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        # Best-effort status write so Streamlit can surface the error
        try:
            run_id = sys.argv[1].split('/')[-1].replace('.json', '')
            _write_status(
                _gcs(),
                f'{_RESULTS_PREFIX}{run_id}/',
                {'run_id': run_id, 'error': tb, 'total_saved': 0},
            )
        except Exception:
            pass
        sys.exit(1)
