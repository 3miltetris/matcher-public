"""
Run / job-status tools (Phase 5.1, Tier 1 read-only)
----------------------------------------------------
list_runs / get_run_status / get_match_results / whats_new. All read-only,
Streamlit-free, JSON-serialisable. They read the same `{prefix}-jobs/{run_id}/
status.json` payloads every view polls (via job_control.poll_status) plus the
matching-results CSV segments — nothing new is written anywhere.

`whats_new` is the "what did last night's scheduled runs produce" digest: the
latest status of each scheduled job plus the open attention counts, so a
consultant can see the state of the pipeline without opening Streamlit.
"""

from __future__ import annotations

import io

import pandas as pd

from src.modules import attention, job_control as jc
from src.modules.tools import _common as cm

# job name → the GCS prefix under which `{run_id}/status.json` is written. Note
# matching-job writes its status alongside its results (matching-results/), not a
# `matching-jobs/` prefix — every other job uses `{name}s/` or its config stem.
STATUS_PREFIXES = {
    'matching-job':        'matching-results/',
    'sam-gov-job':         'sam-gov-jobs/',
    'contact-import-job':  'contact-import-jobs/',
    'drive-sync-job':      'drive-sync-jobs/',
    'fathom-sync-job':     'fathom-jobs/',
    'client-profile-job':  'client-profile-jobs/',
    'deep-research-job':   'deep-research-jobs/',
    'drive-onboard-job':   'drive-onboard-jobs/',
}

# The scheduled jobs whats_new summarises, with the status fields worth surfacing.
_SCHEDULED = {
    'drive-sync-job':    ['clients_updated', 'clients_unchanged', 'clients_errored',
                          'attention_written', 'profile_run_id', 'stopped_early'],
    'fathom-sync-job':   ['meetings_ingested', 'clients_updated', 'stopped_early'],
    'sam-gov-job':       ['rows_saved', 'rows_revised', 'newsletter_good'],
    'deep-research-job': ['opportunities_new', 'opportunities_saved', 'sites_ok',
                          'cost_usd'],
}

KNOWN_JOBS = tuple(STATUS_PREFIXES)


def _status_prefix(job: str) -> str | None:
    return STATUS_PREFIXES.get(job)


def _run_ids(gcs_client, prefix: str, *, bucket: str) -> list[str]:
    """Run ids under a status prefix, newest-first. A run id is the path segment
    between the prefix and `/status.json`."""
    ids = set()
    for blob in gcs_client.list_blobs(bucket, prefix=prefix):
        name = blob.name
        if name.endswith('/status.json'):
            ids.add(name[len(prefix):-len('/status.json')])
    return sorted(ids, reverse=True)


def list_runs(gcs_client, job: str, *, limit: int = 10,
              bucket: str = jc.BUCKET) -> dict:
    """The most recent runs of a job, newest first: run_id + state + a couple of
    headline fields read from each status.json. `{error}` for an unknown job."""
    prefix = _status_prefix(job)
    if prefix is None:
        return {'job': job, 'error': f'unknown job; known: {", ".join(KNOWN_JOBS)}'}
    runs = []
    for run_id in _run_ids(gcs_client, prefix, bucket=bucket)[:max(1, int(limit))]:
        st = jc.poll_status(gcs_client, prefix, run_id, bucket=bucket) or {}
        runs.append({
            'run_id': run_id,
            'state': st.get('state', ''),
            'mode': st.get('mode'),
            'error': cm.cap_text(st.get('error'), 300) if st.get('error') else None,
        })
    return {'job': job, 'runs': runs}


def get_run_status(gcs_client, job: str, run_id: str, *,
                   bucket: str = jc.BUCKET) -> dict:
    """One run's full status.json, with long free-text fields capped and large
    embedded lists trimmed so the result stays within the tool-output budget.
    `{found: false}` if no status has been written yet."""
    prefix = _status_prefix(job)
    if prefix is None:
        return {'job': job, 'error': f'unknown job; known: {", ".join(KNOWN_JOBS)}'}
    st = jc.poll_status(gcs_client, prefix, run_id, bucket=bucket)
    if st is None:
        return {'job': job, 'run_id': run_id, 'found': False}
    return {'job': job, 'run_id': run_id, 'found': True, 'status': _trim_status(st)}


def _trim_status(st: dict, *, list_cap: int = 25, text_cap: int = 600) -> dict:
    """Shrink a status payload for a tool result: cap long strings, trim long
    lists (keeping a count), leave scalars alone. One level deep is enough — the
    job status payloads are flat dicts of scalars + a few lists of small dicts."""
    out = {}
    for k, v in st.items():
        if isinstance(v, str):
            out[k] = cm.cap_text(v, text_cap)
        elif isinstance(v, list):
            out[k] = v[:list_cap]
            if len(v) > list_cap:
                out[f'{k}_total'] = len(v)
        else:
            out[k] = v
    return out


def _match_segments(gcs_client, run_id: str, *, bucket: str) -> list[str]:
    prefix = f'matching-results/{run_id}/segment_'
    return sorted(b.name for b in gcs_client.list_blobs(bucket, prefix=prefix)
                  if b.name.endswith('.csv'))


_MATCH_COLS = ['companyName', 'companyWebsite', 'source', 'topic_number', 'agency',
               'broad_agency', 'title', 'due_date', 'good_match', 'subject_line']


def get_match_results(gcs_client, run_id: str, *, company_key: str | None = None,
                      good_only: bool = False, limit: int = 25, cursor=None,
                      bucket: str = jc.BUCKET) -> dict:
    """Page through a matching run's result rows. Optionally filter to one
    company (`company_key` = 'Name||https://site') or to rows the AI validator
    marked a good match. There is no per-row similarity score in the stored
    segments (the cosine is not written out), so filtering is by company and
    good_match, not a numeric threshold. Returns a `next_cursor` when more rows
    remain."""
    segments = _match_segments(gcs_client, run_id, bucket=bucket)
    if not segments:
        return {'run_id': run_id, 'found': False, 'rows': [], 'next_cursor': None}

    frames = []
    for name in segments:
        try:
            data = gcs_client.bucket(bucket).blob(name).download_as_bytes()
            frames.append(pd.read_csv(io.BytesIO(data), dtype=str))
        except Exception:
            continue
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return {'run_id': run_id, 'found': True, 'total': 0, 'rows': [],
                'next_cursor': None}

    if company_key:
        want = str(company_key).strip()
        keys = (df.get('companyName', '').fillna('') + '||'
                + df.get('companyWebsite', '').fillna(''))
        df = df[keys == want]
    if good_only and 'good_match' in df.columns:
        df = df[df['good_match'].fillna('').str.contains('yes', case=False)]

    total = len(df)
    off = cm.decode_cursor(cursor)
    page = df.iloc[off:off + max(1, int(limit))]
    cols = [c for c in _MATCH_COLS if c in df.columns]
    rows = [{c: cm.cap_text(r.get(c), 300) for c in cols}
            for r in page.to_dict('records')]
    nxt = cm.encode_cursor(off + len(page)) if off + len(page) < total else None
    return {'run_id': run_id, 'found': True, 'total': int(total),
            'returned': len(rows), 'rows': rows, 'next_cursor': nxt}


def whats_new(gcs_client, *, bucket: str = jc.BUCKET) -> dict:
    """A digest of what the scheduled pipeline produced: the latest run of each
    scheduled job with its headline counts, plus the open attention-queue counts
    (what needs a human). Read-only snapshot for 'what happened overnight'."""
    jobs = {}
    for job, fields in _SCHEDULED.items():
        prefix = STATUS_PREFIXES[job]
        ids = _run_ids(gcs_client, prefix, bucket=bucket)
        if not ids:
            jobs[job] = {'latest_run': None}
            continue
        run_id = ids[0]
        st = jc.poll_status(gcs_client, prefix, run_id, bucket=bucket) or {}
        summary = {f: st.get(f) for f in fields if st.get(f) is not None}
        jobs[job] = {'latest_run': run_id, 'state': st.get('state', ''), **summary}

    try:
        needs_attention = attention.open_counts(gcs_client, bucket=bucket)
    except Exception:
        needs_attention = {}
    return {'jobs': jobs, 'needs_attention': needs_attention}
