"""
SAM.gov Upload
--------------
Two input modes:
  - Upload CSV     : Upload SAM.gov contract opportunity CSVs (runs in Streamlit).
  - Fetch from API : Configure parameters and trigger the Cloud Run sam-gov-job.
                     Fetching, screening, summarising, and embedding all run in the
                     background job — Streamlit polls for completion.
"""

import io
import json
import secrets
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from anthropic import Anthropic
from google.cloud import run_v2, storage
from google.oauth2 import service_account

from src.modules.Embedding.text_embedder import TextProcessor
from src.modules.GoogleBucketManager.bucket_manager import BucketManager

# ── Constants ────────────────────────────────────────────────────────────────

_BUCKET          = 'cc-matcher-bucket-jeg-v1'
_SAM_PREFIX      = 'data/all-topics/processed/SAM-GOV/'
_SAM_CFG_PREFIX  = 'sam-gov-configs/'
_JOB_NAME        = 'projects/cc-matcher-v1/locations/us-central1/jobs/sam-gov-job'
_POLL_INTERVAL   = 10

_SCREEN_MODEL     = 'claude-haiku-4-5-20251001'
_SCREEN_WORKERS   = 8
_SCREEN_MAX_CHARS = 3000

_NOTICE_TYPE_OPTIONS = {
    'Presolicitation':                'p',
    'Solicitation':                   'o',
    'Combined Synopsis/Solicitation': 'k',
    'Sources Sought':                 'r',
    'Special Notice':                 's',
}

_DAILY_CONFIG_PATH = 'sam-gov-configs/daily_schedule.json'


# ── Screening prompt (CSV path) ───────────────────────────────────────────────

_SCREEN_SYSTEM = """\
You are a grant opportunity screening filter for a consultancy that serves innovative startups, R&D companies, and deep tech small businesses.

You will be given a federal opportunity that has already been pre-filtered to R&D and contract notices. Your only job is to decide: should this opportunity be imported into our matching system?

Answer YES if the opportunity is a realistic fit for startups, small businesses, or deep tech R&D companies.
Answer NO if it is not.

---

OUR IDEAL CLIENTS:
- Startups and small businesses doing technical R&D
- Deep tech ventures (biotech, defense tech, energy, AI/ML, hardware, advanced manufacturing, etc.)
- SBIR/STTR-eligible companies
- Commercialization-stage innovators

IMPORT (YES) if any of the following are true:
- The opportunity is explicitly open to or designed for small businesses or startups
- The NAICS descriptor suggests a field where small R&D companies commonly operate
- The work requires genuine technical innovation, applied research, or novel development
- It is a vehicle type our clients pursue (SBIR, STTR, BAA, IDIQ with small business tracks, etc.)

DO NOT IMPORT (NO) if any of the following are true:
- The opportunity is clearly intended for universities, nonprofits, state/local governments, or large prime contractors with no realistic small business angle
- It is a workforce development, training, community services, or infrastructure project
- It is a generic service procurement with no meaningful R&D or innovation component (e.g., janitorial, facilities, staffing, administrative support)
- The NAICS descriptor is in a sector where deep tech startups almost never compete (e.g., construction, food service, transportation logistics)
- The description makes clear that prior large-scale program experience or clearances beyond typical small business reach are required

When uncertain, only import if the opportunity is clearly relevant. Do not import on weak signals alone.

---

OUTPUT FORMAT:
Respond only with valid JSON. No preamble, no markdown, no explanation outside the JSON.
{"import": true, "confidence": "high", "reason": "One or two sentences explaining the decision."}\
"""


# ── Summarization prompt (CSV path) ──────────────────────────────────────────

_SUMMARY_SYSTEM = """\
You are preparing a federal contract opportunity for semantic matching against startup and R&D company profiles.

Summarize the opportunity in 3–5 sentences. Focus exclusively on:
- The specific technical problem, research area, or capability being sought
- Key deliverables or desired technical outcomes
- Relevant domain, technology, or sector (e.g., AI/ML, biotech, defense electronics, advanced manufacturing)

Strip out all procurement boilerplate: FAR clauses, set-aside language, submission deadlines, page limits, administrative instructions, and points of contact. Write in plain technical language. If the description is already short and technical, return it as-is.\
"""


# ── Daily schedule helpers ────────────────────────────────────────────────────

def _load_daily_config(client: storage.Client) -> dict | None:
    blob = client.bucket(_BUCKET).blob(_DAILY_CONFIG_PATH)
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def _save_daily_config(client: storage.Client, config: dict) -> None:
    client.bucket(_BUCKET).blob(_DAILY_CONFIG_PATH).upload_from_string(
        json.dumps(config), content_type='application/json'
    )


# ── GCS helpers ──────────────────────────────────────────────────────────────

def _get_credentials():
    return service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )


def _get_storage_client() -> storage.Client:
    return storage.Client(credentials=_get_credentials())


# ── Cloud Run helpers (API path) ──────────────────────────────────────────────

def _write_sam_config(client: storage.Client, config: dict) -> str:
    blob_path = f'{_SAM_CFG_PREFIX}{config["run_id"]}.json'
    client.bucket(_BUCKET).blob(blob_path).upload_from_string(
        json.dumps(config), content_type='application/json'
    )
    return blob_path


def _trigger_sam_job(config_blob_path: str) -> None:
    job_client = run_v2.JobsClient(credentials=_get_credentials())
    job_client.run_job(
        request=run_v2.RunJobRequest(
            name=_JOB_NAME,
            overrides=run_v2.RunJobRequest.Overrides(
                container_overrides=[
                    run_v2.RunJobRequest.Overrides.ContainerOverride(
                        args=[config_blob_path]
                    )
                ]
            ),
        )
    )


def _poll_sam_status(client: storage.Client, run_id: str) -> dict | None:
    blob = client.bucket(_BUCKET).blob(f'sam-gov-jobs/{run_id}/status.json')
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


# ── Column auto-detection (CSV path) ─────────────────────────────────────────

_CANDIDATES: dict[str, list[str]] = {
    'title':        ['title', 'opportunity title', 'solicitation title'],
    'description':  ['description', 'synopsis', 'description/synopsis'],
    'naics_desc':   ['naics desc', 'naics description', 'naics_desc', 'naics_description'],
    'notice_id':    ['notice id', 'notice_id', 'solicitation #', 'sol #', 'solicitation number'],
    'agency':       ['department/ind.agency', 'department', 'agency', 'department name'],
    'posted_date':  ['posted date', 'post date', 'posted_date'],
    'deadline':     ['response deadline', 'response_deadline', 'deadline', 'close date'],
    'source_url':   ['contract opportunity url', 'opportunity url', 'sam url', 'url', 'link'],
}


def _detect_col(columns: list[str], field: str) -> str | None:
    lower = {c.lower(): c for c in columns}
    for candidate in _CANDIDATES[field]:
        if candidate in lower:
            return lower[candidate]
    return None


# ── Screening (CSV path) ──────────────────────────────────────────────────────

def _screen_one(title: str, desc: str, naics: str, anth_key: str) -> dict:
    client   = Anthropic(api_key=anth_key)
    user_msg = (
        f"Title: {title}\n"
        f"Description: {str(desc)[:_SCREEN_MAX_CHARS]}\n"
        f"NAICS Descriptor: {naics}"
    )
    resp = client.messages.create(
        model=_SCREEN_MODEL,
        max_tokens=200,
        system=_SCREEN_SYSTEM,
        messages=[{'role': 'user', 'content': user_msg}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    return json.loads(raw)


def _run_screening(df: pd.DataFrame, col_map: dict, anth_key: str) -> pd.DataFrame:
    titles = df[col_map['title']].astype(str).tolist()
    descs  = df[col_map['description']].astype(str).tolist()
    naics  = df[col_map['naics_desc']].astype(str).tolist() if col_map.get('naics_desc') else [''] * len(df)

    results  = [None] * len(df)
    progress = st.progress(0, text='Screening rows…')

    with ThreadPoolExecutor(max_workers=_SCREEN_WORKERS) as pool:
        futures = {
            pool.submit(_screen_one, titles[i], descs[i], naics[i], anth_key): i
            for i in range(len(df))
        }
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                results[i] = {'import': False, 'confidence': 'low', 'reason': f'Screening error: {e}'}
            done += 1
            progress.progress(done / len(df), text=f'Screening rows… {done}/{len(df)}')

    progress.empty()

    out = df.copy()
    out['_import']     = [r['import']     for r in results]
    out['_confidence'] = [r['confidence'] for r in results]
    out['_reason']     = [r['reason']     for r in results]
    return out


# ── Summarization (CSV path) ──────────────────────────────────────────────────

def _summarize_one(title: str, desc: str, anth_key: str) -> str:
    client   = Anthropic(api_key=anth_key)
    user_msg = f"Title: {title}\n\nDescription:\n{str(desc)[:5000]}"
    resp = client.messages.create(
        model=_SCREEN_MODEL,
        max_tokens=400,
        system=_SUMMARY_SYSTEM,
        messages=[{'role': 'user', 'content': user_msg}],
    )
    return resp.content[0].text.strip()


def _summarize_descriptions(titles: list[str], descs: list[str], anth_key: str) -> list[str]:
    results  = [''] * len(descs)
    progress = st.progress(0, text='Summarizing descriptions…')

    with ThreadPoolExecutor(max_workers=_SCREEN_WORKERS) as pool:
        futures = {
            pool.submit(_summarize_one, titles[i], descs[i], anth_key): i
            for i in range(len(descs))
        }
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception:
                results[i] = descs[i]
            done += 1
            progress.progress(done / len(descs), text=f'Summarizing… {done}/{len(descs)}')

    progress.empty()
    return results


# ── Existing-record dedup (CSV path) ──────────────────────────────────────────

def _load_existing_keys(client: storage.Client) -> tuple[set[str], set[str]]:
    notice_ids: set[str] = set()
    titles:     set[str] = set()
    blobs = client.list_blobs(_BUCKET, prefix=_SAM_PREFIX)
    for blob in blobs:
        if not blob.name.endswith('.parquet'):
            continue
        try:
            df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()), columns=['topic_number', 'title'])
            notice_ids.update(df['topic_number'].dropna().astype(str).str.strip())
            titles.update(df['title'].dropna().astype(str).str.lower().str.strip())
        except Exception:
            pass
    return notice_ids, titles


# ── Embed + save (CSV path) ───────────────────────────────────────────────────

_SAM_RESERVED_COLS = frozenset({
    'topic_number', 'agency', 'title', 'description', 'open_date', 'due_date',
    'scraped_at', 'sam_confidence', 'sam_reason', 'grant_summary', 'embeddings', 'source',
})


def _embed_and_save(
    df: pd.DataFrame,
    col_map: dict,
    oai_key: str,
    anth_key: str,
    extra_cols: dict[str, str] | None = None,
) -> str:
    tp    = TextProcessor(api_key=oai_key)
    bm    = BucketManager(_BUCKET, client=_get_storage_client())
    today = datetime.today().strftime('%Y-%m-%d')

    out = pd.DataFrame()
    out['topic_number'] = df[col_map['notice_id']].astype(str)   if col_map.get('notice_id')   else ''
    out['agency']       = df[col_map['agency']].astype(str)       if col_map.get('agency')      else 'SAM-GOV'
    out['title']        = df[col_map['title']].astype(str)
    out['description']  = df[col_map['description']].astype(str)
    out['open_date']    = df[col_map['posted_date']].astype(str)  if col_map.get('posted_date') else ''
    out['due_date']     = df[col_map['deadline']].astype(str)     if col_map.get('deadline')    else ''
    out['source']       = df[col_map['source_url']].astype(str)  if col_map.get('source_url')  else ''
    out['scraped_at']   = today
    out['sam_confidence'] = df['_confidence'].values
    out['sam_reason']   = df['_reason'].values

    titles    = out['title'].tolist()
    descs     = out['description'].tolist()
    summaries = _summarize_descriptions(titles, descs, anth_key)
    out['grant_summary'] = summaries

    _THIN_PHRASES = ("don't have enough technical content", "not enough technical content", "i cannot summarize", "i'm unable to summarize")
    thin_mask = out['grant_summary'].apply(lambda s: any(p in s.lower() for p in _THIN_PHRASES))
    if thin_mask.any():
        n_thin = int(thin_mask.sum())
        st.warning(f'{n_thin} row(s) dropped — description had insufficient technical content to summarize.')
        out = out[~thin_mask].reset_index(drop=True)

    embed_texts = [s if s.strip() else d for s, d in zip(out['grant_summary'].tolist(), out['description'].tolist())]
    progress    = st.progress(0, text='Generating embeddings…')
    embeddings  = []
    for i, text in enumerate(embed_texts):
        embeddings.append(tp.get_embedding(text) if text.strip() else None)
        progress.progress((i + 1) / len(embed_texts), text=f'Embedding {i + 1}/{len(embed_texts)}…')
    progress.empty()

    out['embeddings'] = embeddings

    if extra_cols:
        for col_name, col_val in extra_cols.items():
            out[col_name] = col_val

    hex_suffix = secrets.token_hex(3)
    gcs_path   = f'{_SAM_PREFIX}sam_gov_{today}_{hex_suffix}.parquet'
    bm.upload_file(gcs_path, out)
    return gcs_path


# ── Session state ─────────────────────────────────────────────────────────────

for _k in ['sam_raw_df', 'sam_screened_df', 'sam_existing_keys', 'sam_col_map']:
    if _k not in st.session_state:
        st.session_state[_k] = None
if 'sam_active_run' not in st.session_state:
    st.session_state.sam_active_run = None
if 'sam_custom_cols' not in st.session_state:
    st.session_state.sam_custom_cols = []
if 'sam_api_custom_cols' not in st.session_state:
    st.session_state.sam_api_custom_cols = []
if 'sam_daily_config' not in st.session_state:
    st.session_state.sam_daily_config = 'unloaded'


# ── Page ──────────────────────────────────────────────────────────────────────

st.title('🏛️ SAM.gov Upload')
st.caption(
    'Screen and embed SAM.gov contract opportunities for the matching pipeline. '
    'Upload exported CSVs (processed in Streamlit) or fetch directly from the '
    'SAM.gov API (processed by Cloud Run).'
)


# ── Daily Schedule ─────────────────────────────────────────────────────────────

# Load the saved daily config once per session
if st.session_state.sam_daily_config == 'unloaded':
    try:
        st.session_state.sam_daily_config = _load_daily_config(_get_storage_client())
    except Exception:
        st.session_state.sam_daily_config = None

_daily_cfg     = st.session_state.sam_daily_config
_daily_params  = (_daily_cfg or {}).get('api_params', {})
_daily_exists  = _daily_cfg is not None

with st.expander('⏰ Daily API Parameters', expanded=not _daily_exists):
    st.caption(
        'Configure the filters for the automated 5 AM CST daily pull. '
        'Settings are saved to GCS and picked up by Cloud Scheduler each morning — '
        'no redeploy needed to change parameters.'
    )

    _sam_key = st.secrets.get('sam_gov_api_key')
    if not _sam_key:
        st.warning(
            'Add `sam_gov_api_key` to `.streamlit/secrets.toml` to configure the daily schedule.'
        )
    else:
        _d_lookback = st.number_input(
            'Lookback window (days)',
            min_value=1, max_value=30,
            value=int(_daily_params.get('lookback_days', 1)),
            help='Pull opportunities posted within the last N days relative to when the job fires.',
            key='daily_lookback',
        )

        _d_type_labels = st.multiselect(
            'Notice types',
            list(_NOTICE_TYPE_OPTIONS.keys()),
            default=[
                lbl for lbl, code in _NOTICE_TYPE_OPTIONS.items()
                if code in _daily_params.get('notice_types', ['p', 'o', 'k', 'r'])
            ],
            key='daily_notice_types',
        )

        _d_keyword = st.text_input(
            'Keyword filter (optional)',
            value=_daily_params.get('keyword', ''),
            placeholder='e.g. AI, biotech, cybersecurity',
            key='daily_keyword',
        )

        _dc_l, _dc_r = st.columns(2)
        with _dc_l:
            _d_max = st.number_input(
                'Max results (0 = no cap)',
                min_value=0,
                value=int(_daily_params.get('max_results', 500)),
                step=100,
                key='daily_max',
            )
        with _dc_r:
            _d_fetch_desc = st.checkbox(
                'Fetch full descriptions',
                value=bool(_daily_params.get('fetch_desc', True)),
                key='daily_fetch_desc',
            )

        if st.button('💾 Save Daily Schedule', key='save_daily_btn'):
            _new_daily = {
                'run_id':      'daily',
                'input_mode':  'api',
                'api_params':  {
                    'lookback_days':  int(_d_lookback),
                    'notice_types':   [_NOTICE_TYPE_OPTIONS[lbl] for lbl in _d_type_labels],
                    'keyword':        _d_keyword.strip(),
                    'max_results':    int(_d_max),
                    'fetch_desc':     bool(_d_fetch_desc),
                    'sam_gov_api_key': _sam_key,
                },
                'custom_cols': {},
            }
            try:
                _save_daily_config(_get_storage_client(), _new_daily)
                st.session_state.sam_daily_config = _new_daily
                st.success(f'Daily schedule saved to `{_DAILY_CONFIG_PATH}`.')
            except Exception as _e:
                st.error(f'Failed to save: {_e}')

    if _daily_exists:
        st.divider()
        st.caption(
            f'**Current schedule:** last {_daily_params.get("lookback_days", 1)} day(s) · '
            f'notice types: `{", ".join(_daily_params.get("notice_types", []))}`'
        )
        with st.expander('Cloud Scheduler setup (one-time)', expanded=False):
            st.caption(
                'Run this once in Cloud Shell to wire up the daily 5 AM CST trigger. '
                'After that, only the GCS config controls what gets fetched — no redeploy needed.'
            )
            st.code(
                'gcloud scheduler jobs create http sam-gov-daily \\\n'
                '  --schedule="0 11 * * *" \\\n'
                '  --uri="https://run.googleapis.com/v2/projects/cc-matcher-v1/locations/us-central1/jobs/sam-gov-job:run" \\\n'
                '  --message-body=\'{"overrides":{"containerOverrides":[{"args":["sam-gov-configs/daily_schedule.json"]}]}}\' \\\n'
                '  --oauth-service-account-email=matching-job@cc-matcher-v1.iam.gserviceaccount.com \\\n'
                '  --location=us-central1 \\\n'
                '  --time-zone="America/Chicago"',
                language='bash',
            )
            st.caption(
                'To update the trigger time later: '
                '`gcloud scheduler jobs update http sam-gov-daily --schedule="..." --location=us-central1`'
            )

# ── Revision Check ─────────────────────────────────────────────────────────────

with st.expander('🔁 Revision Check', expanded=False):
    st.caption(
        'Sweep every stored open SAM.gov notice (CSOs included) for revisions: '
        'each solicitation number is looked up on SAM.gov, notices with a new '
        'version get their description and attachments re-fetched, Claude diffs '
        'the content (topics added/removed), and the stored record is updated in '
        'place — new summary, embedding, deadline, and revision notes. Notices '
        'no longer on SAM.gov are marked archived and skipped by matching. '
        'Revisions that arrive in the daily pull are handled automatically; use '
        'this to catch anything the daily filters missed. '
        '**SAM.gov allows 1,000 API requests per key per day** (shared with the '
        'daily fetch), so each run checks a budgeted chunk of notices, oldest-'
        'checked first, and resumes where the last apply run left off — a full '
        'sweep of a large store completes over several days.'
    )

    _rc_key = st.secrets.get('sam_gov_api_key')
    if not _rc_key:
        st.warning('Add `sam_gov_api_key` to `.streamlit/secrets.toml` to run revision checks.')
    else:
        rc_l, rc_m, rc_r = st.columns(3)
        with rc_l:
            rc_dry = st.checkbox(
                'Report only (dry run)',
                value=True,
                key='rc_dry_run',
                help='Detect and diff revisions but write nothing — review the report before applying. '
                     'Dry runs do not advance the sweep cursor, so the next apply run '
                     'processes the same chunk of notices.',
            )
        with rc_m:
            rc_attach = st.checkbox(
                'Include attachments (PDF text)',
                value=True,
                key='rc_attachments',
                help='Download attached PDFs and include their text in the diff and updated summary. '
                     'CSO topics often live only in attachments.',
            )
        with rc_r:
            rc_budget = st.number_input(
                'API call budget',
                min_value=50,
                max_value=950,
                value=600,
                step=50,
                key='rc_budget',
                help='SAM.gov requests this run may spend. Your key allows 1,000/day total, '
                     'shared with the daily 5 AM fetch — leave headroom. The sweep stops '
                     'cleanly at the budget and resumes on the next run.',
            )

        if st.button('🔍 Check stored notices for revisions', type='primary', key='rc_run_btn'):
            rc_run_id = f"sam_gov_revcheck_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            rc_config = {
                'run_id':     rc_run_id,
                'input_mode': 'revision_check',
                'api_params': {
                    'sam_gov_api_key':     _rc_key,
                    'include_attachments': bool(rc_attach),
                    'max_api_calls':       int(rc_budget),
                },
                'dry_run':    bool(rc_dry),
            }
            try:
                rc_gcs = _get_storage_client()
                with st.spinner('Uploading job config…'):
                    rc_blob = _write_sam_config(rc_gcs, rc_config)
                with st.spinner('Triggering Cloud Run job…'):
                    _trigger_sam_job(rc_blob)
                st.session_state.sam_active_run = {
                    'run_id':      rc_run_id,
                    'config_blob': rc_blob,
                }
                st.rerun()
            except Exception as exc:
                st.error(f'Failed to trigger revision check: {exc}')
                st.code(traceback.format_exc())

st.divider()

# ── Active run: polling UI ────────────────────────────────────────────────────

if st.session_state.sam_active_run:
    active = st.session_state.sam_active_run
    run_id = active['run_id']

    st.subheader('🔄 Import in progress')
    st.caption(f'Run ID: `{run_id}`')

    try:
        gcs    = _get_storage_client()
        status = _poll_sam_status(gcs, run_id)

        if status is None:
            st.info(f'Job is running… checking again in {_POLL_INTERVAL}s.')
            if st.button('Cancel monitoring (job keeps running)'):
                st.session_state.sam_active_run = None
                st.rerun()
            time.sleep(_POLL_INTERVAL)
            st.rerun()

        elif status.get('error'):
            st.error('Import job failed.')
            st.code(status['error'])
            st.session_state.sam_active_run = None

        else:
            if status.get('mode') == 'revision_check':
                dr = ' (dry run — nothing was written)' if status.get('dry_run') else ''
                st.success(
                    f"Revision check complete{dr} — "
                    f"**{status.get('rows_checked', 0):,}** notices checked, "
                    f"**{status.get('revisions_found', 0):,}** revised, "
                    f"**{status.get('rows_archived', 0):,}** archived, "
                    f"**{status.get('rows_updated', 0):,}** updated in store."
                )
                if status.get('stopped_early'):
                    _why = (
                        'the SAM.gov **daily quota** ran out (resets midnight UTC)'
                        if status['stopped_early'] == 'quota'
                        else f"the run's API call budget ({status.get('api_call_budget', 0):,}) was reached"
                    )
                    st.warning(
                        f"Sweep stopped early because {_why} — "
                        f"**{status.get('rows_remaining', 0):,}** of "
                        f"{status.get('rows_candidates', 0):,} open notices still unchecked. "
                        f"Run again (tomorrow, if quota) to continue; apply runs resume "
                        f"from the least-recently-checked notices."
                    )
                if status.get('revisions_deferred'):
                    st.warning(
                        f"{status['revisions_deferred']} detected revision(s) could not be "
                        f"processed before the quota ran out — they will be re-detected "
                        f"on the next run."
                    )
                if status.get('api_calls_used'):
                    st.caption(
                        f"SAM.gov API calls used: {status['api_calls_used']:,} "
                        f"of {status.get('api_call_budget', 0):,} budgeted."
                    )
                if status.get('lookup_errors'):
                    st.warning(
                        f"{status['lookup_errors']} notice(s) could not be checked "
                        f"due to SAM.gov lookup errors — they will be retried on the next run."
                    )
            else:
                st.success(
                    f'Import complete — **{status.get("rows_saved", 0):,}** topics saved '
                    f'(fetched {status.get("rows_fetched", 0):,}, '
                    f'passed screening {status.get("rows_passed_screening", 0):,}, '
                    f'after dedup {status.get("rows_after_dedup", 0):,}).'
                )
                if status.get('rows_revised'):
                    st.info(
                        f"**{status['rows_revised']}** previously stored notice(s) had new "
                        f"SAM.gov versions and were updated in place."
                    )
                if status.get('gcs_path'):
                    st.caption(f'Saved to `{status["gcs_path"]}`')

            _revs = status.get('revisions') or []
            if _revs:
                with st.expander(f'📝 Revision details ({len(_revs)})', expanded=True):
                    st.dataframe(pd.DataFrame(_revs), hide_index=True, use_container_width=True)
            _arch = status.get('archived') or []
            if _arch:
                with st.expander(f'🗄️ Archived notices ({len(_arch)})', expanded=False):
                    st.dataframe(pd.DataFrame(_arch), hide_index=True, use_container_width=True)

            st.session_state.sam_active_run = None

    except Exception as e:
        st.error(f'Error polling status: {e}')
        st.code(traceback.format_exc())
        if st.button('Stop monitoring'):
            st.session_state.sam_active_run = None
            st.rerun()

    st.stop()


# ── Section 1 · Load ──────────────────────────────────────────────────────────

st.subheader('1 · Load opportunities')

tab_csv, tab_api = st.tabs(['📄 Upload CSV', '🔌 Fetch from API'])


# ── Tab: Upload CSV ───────────────────────────────────────────────────────────

with tab_csv:
    files = st.file_uploader(
        'SAM.gov export CSV(s)',
        type='csv',
        accept_multiple_files=True,
        label_visibility='collapsed',
    )
    if files:
        frames = []
        for f in files:
            try:
                try:
                    frames.append(pd.read_csv(f, dtype=str, encoding='utf-8'))
                except UnicodeDecodeError:
                    f.seek(0)
                    frames.append(pd.read_csv(f, dtype=str, encoding='latin-1'))
            except Exception as e:
                st.error(f'Could not read **{f.name}**: {e}')
        if frames:
            combined = pd.concat(frames, ignore_index=True).dropna(how='all')
            if (
                st.session_state.sam_raw_df is None
                or len(combined) != len(st.session_state.sam_raw_df)
            ):
                st.session_state.sam_raw_df        = combined
                st.session_state.sam_screened_df   = None
                st.session_state.sam_col_map       = None
                st.session_state.sam_existing_keys = None
                st.session_state.sam_custom_cols   = []


# ── Tab: Fetch from API ───────────────────────────────────────────────────────

with tab_api:
    sam_key = st.secrets.get('sam_gov_api_key')
    if not sam_key:
        st.warning(
            'Add `sam_gov_api_key` to `.streamlit/secrets.toml` to use API fetch. '
            'Register for a free key at **beta.sam.gov → Account Settings → API Keys**.'
        )
    else:
        st.caption(
            'Triggers the Cloud Run `sam-gov-job` — fetching, screening, summarising, '
            'and embedding all run in the background. Streamlit polls for completion.'
        )

        today_date   = datetime.today().date()
        default_from = today_date - timedelta(days=30)

        col_l, col_r = st.columns(2)
        with col_l:
            api_date_from = st.date_input('Posted from', value=default_from, key='sam_api_date_from')
        with col_r:
            api_date_to = st.date_input('Posted to', value=today_date, key='sam_api_date_to')

        selected_type_labels = st.multiselect(
            'Notice types',
            list(_NOTICE_TYPE_OPTIONS.keys()),
            default=['Solicitation', 'Presolicitation', 'Sources Sought'],
            key='sam_api_notice_types',
        )
        selected_type_codes = [_NOTICE_TYPE_OPTIONS[lbl] for lbl in selected_type_labels]

        api_keyword = st.text_input(
            'Keyword filter (optional)',
            placeholder='e.g. AI, biotech, cybersecurity',
            key='sam_api_keyword',
        )

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            api_max = st.number_input(
                'Max results (0 = no cap)',
                min_value=0,
                value=500,
                step=100,
                key='sam_api_max',
            )
        with col_r2:
            api_fetch_desc = st.checkbox(
                'Fetch full descriptions',
                value=True,
                key='sam_api_fetch_desc',
                help=(
                    'Makes one additional API call per opportunity to retrieve the full '
                    'synopsis text. Slower but significantly improves screening accuracy. '
                    'Disable for very large date ranges to stay within the daily API quota.'
                ),
            )

        with st.expander('Custom columns (optional)', expanded=bool(st.session_state.sam_api_custom_cols)):
            st.caption(
                'Add extra columns to tag every saved topic with campaign-specific metadata.'
            )

            for entry in list(st.session_state.sam_api_custom_cols):
                sc1, sc2, sc3 = st.columns([2, 4, 1])
                with sc1:
                    st.text(entry['name'])
                with sc2:
                    entry['value'] = st.text_input(
                        'Value', value=entry['value'],
                        key=f'sam_api_cfill_{entry["name"]}', label_visibility='collapsed',
                    )
                with sc3:
                    if st.button('✕', key=f'sam_api_crm_{entry["name"]}', use_container_width=True):
                        st.session_state.sam_api_custom_cols = [
                            e for e in st.session_state.sam_api_custom_cols if e['name'] != entry['name']
                        ]
                        st.rerun()

            if st.session_state.sam_api_custom_cols:
                st.divider()

            sna1, sna2, sna3 = st.columns([2, 4, 1])
            with sna1:
                new_api_sc_name = st.text_input(
                    'Column name', placeholder='e.g. campaign_name',
                    key='sam_api_new_sc_name', label_visibility='collapsed',
                )
            with sna2:
                new_api_sc_val = st.text_input(
                    'Value', placeholder='e.g. Spring 2026',
                    key='sam_api_new_sc_val', label_visibility='collapsed',
                )
            with sna3:
                if st.button('Add', key='sam_api_add_sc_btn', use_container_width=True):
                    clean = new_api_sc_name.strip().replace(' ', '_').lower()
                    existing_names = {e['name'] for e in st.session_state.sam_api_custom_cols}
                    if (
                        clean
                        and clean not in _SAM_RESERVED_COLS
                        and clean not in existing_names
                    ):
                        st.session_state.sam_api_custom_cols.append({'name': clean, 'value': new_api_sc_val.strip()})
                        st.rerun()

        if api_date_from > api_date_to:
            st.error('"Posted from" must be on or before "Posted to".')
        else:
            if st.button('▶ Run SAM.gov Import', type='primary', key='sam_api_run_btn'):
                run_id = f"sam_gov_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                config = {
                    'run_id':      run_id,
                    'input_mode':  'api',
                    'api_params':  {
                        'date_from':       api_date_from.strftime('%m/%d/%Y'),
                        'date_to':         api_date_to.strftime('%m/%d/%Y'),
                        'notice_types':    selected_type_codes,
                        'keyword':         api_keyword.strip(),
                        'max_results':     int(api_max),
                        'fetch_desc':      bool(api_fetch_desc),
                        'sam_gov_api_key': sam_key,
                    },
                    'custom_cols': {
                        e['name']: e['value']
                        for e in st.session_state.sam_api_custom_cols
                    },
                }
                try:
                    gcs_client = _get_storage_client()
                    with st.spinner('Uploading job config…'):
                        config_blob = _write_sam_config(gcs_client, config)
                    with st.spinner('Triggering Cloud Run job…'):
                        _trigger_sam_job(config_blob)
                    st.session_state.sam_active_run = {
                        'run_id':      run_id,
                        'config_blob': config_blob,
                    }
                    st.rerun()
                except Exception as exc:
                    st.error(f'Failed to trigger job: {exc}')
                    st.code(traceback.format_exc())


# ── CSV path guard ────────────────────────────────────────────────────────────

if st.session_state.sam_raw_df is None:
    st.stop()

df_raw   = st.session_state.sam_raw_df
src_label = f'{len(files)} file(s)' if files else '(previously loaded)'
st.caption(f'**{len(df_raw):,}** rows loaded from {src_label}.')
st.dataframe(df_raw.head(5), hide_index=True, use_container_width=True)


# ── Section 2 · Column mapping ────────────────────────────────────────────────

st.divider()
st.subheader('2 · Column mapping')
st.caption('Confirm which columns map to each field — auto-detected where possible.')

cols     = df_raw.columns.tolist()
none_opt = '— none —'
col_opts = [none_opt] + cols


def _sel(field: str, label: str, required: bool = False) -> str | None:
    detected = _detect_col(cols, field)
    idx      = col_opts.index(detected) if detected in col_opts else 0
    val      = st.selectbox(
        label + (' *' if required else ''),
        col_opts,
        index=idx,
        key=f'sam_map_{field}',
    )
    return val if val != none_opt else None


left_col, right_col = st.columns(2)
with left_col:
    m_title  = _sel('title',       'Title',              required=True)
    m_desc   = _sel('description', 'Description',        required=True)
    m_naics  = _sel('naics_desc',  'NAICS Descriptor')
    m_notice = _sel('notice_id',   'Notice ID')
with right_col:
    m_agency = _sel('agency',      'Agency / Department')
    m_posted = _sel('posted_date', 'Posted Date')
    m_dl     = _sel('deadline',    'Response Deadline')
    m_url    = _sel('source_url',  'Contract URL')

col_map = {
    'title':       m_title,
    'description': m_desc,
    'naics_desc':  m_naics,
    'notice_id':   m_notice,
    'agency':      m_agency,
    'posted_date': m_posted,
    'deadline':    m_dl,
    'source_url':  m_url,
}

if not m_title or not m_desc:
    st.warning('Title and Description columns are required before proceeding.')
    st.stop()


# ── Section 3 · Dedup against existing store ──────────────────────────────────

st.divider()
st.subheader('3 · Dedup against existing store')

if st.session_state.sam_existing_keys is None:
    with st.spinner('Checking existing records for duplicates…'):
        try:
            st.session_state.sam_existing_keys = _load_existing_keys(_get_storage_client())
        except Exception as e:
            st.warning(f'Could not load existing records for dedup check: {e}')
            st.session_state.sam_existing_keys = (set(), set())

existing_ids, existing_titles = st.session_state.sam_existing_keys


def _is_dup(row: pd.Series) -> bool:
    if m_notice and str(row.get(m_notice, '')).strip() in existing_ids:
        return True
    return str(row.get(m_title, '')).strip().lower() in existing_titles


dup_mask = df_raw.apply(_is_dup, axis=1)
dupes    = df_raw[dup_mask]
df_new   = df_raw[~dup_mask].reset_index(drop=True)

d1, d2, d3 = st.columns(3)
d1.metric('Uploaded rows',    f'{len(df_raw):,}')
d2.metric('Already in store', f'{len(dupes):,}')
d3.metric('New rows',         f'{len(df_new):,}')

if not dupes.empty:
    with st.expander(f'Skipped duplicates ({len(dupes)})', expanded=False):
        st.dataframe(
            dupes[[m_title] + ([m_notice] if m_notice else [])].reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
        )

if df_new.empty:
    st.success('All uploaded rows are already in the store — nothing new to screen.')
    st.stop()

# Invalidate stale screening results if the dedup output changed (e.g. mapping edited)
if (
    st.session_state.sam_screened_df is not None
    and len(st.session_state.sam_screened_df) != len(df_new)
):
    st.session_state.sam_screened_df = None


# ── Section 4 · Screening ─────────────────────────────────────────────────────

st.divider()
st.subheader('4 · Screen with Claude')

n_rows   = len(df_new)
est_mins = max(1, n_rows // 60)
screened = st.session_state.sam_screened_df

if screened is None:
    st.caption(
        f'Claude Haiku will screen **{n_rows:,}** new rows for relevance. '
        f'Estimated time: ~{est_mins} min at {_SCREEN_WORKERS} concurrent workers.'
    )
    if st.button('⚡ Run Screening', type='primary'):
        anth_key = st.secrets['anthropic_api_key']
        try:
            st.session_state.sam_screened_df = _run_screening(df_new, col_map, anth_key)
            st.rerun()
        except Exception as e:
            st.error(f'Screening failed: {e}')
    st.stop()

passing  = screened[screened['_import'] == True].copy()
failing  = screened[screened['_import'] == False].copy()
pass_pct = len(passing) / len(screened) * 100 if len(screened) > 0 else 0

m1, m2, m3 = st.columns(3)
m1.metric('Total rows',   f'{len(screened):,}')
m2.metric('Passing',      f'{len(passing):,}  ({pass_pct:.0f}%)')
m3.metric('Filtered out', f'{len(failing):,}')

if st.button('↺ Re-run screening'):
    st.session_state.sam_screened_df = None
    st.rerun()

_display = [c for c in [m_title, m_desc, '_confidence', '_reason'] if c]
_cfg = {
    m_title:       st.column_config.TextColumn('Title',       width='medium'),
    m_desc:        st.column_config.TextColumn('Description', width='large'),
    '_confidence': st.column_config.TextColumn('Confidence',  width='small'),
    '_reason':     st.column_config.TextColumn('Reason',      width='large'),
}

with st.expander(f'✅ Passing ({len(passing)})', expanded=True):
    if passing.empty:
        st.info('No rows passed screening.')
    else:
        st.dataframe(
            passing[_display].reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
            column_config=_cfg,
        )

with st.expander(f'❌ Filtered out ({len(failing)})', expanded=False):
    if failing.empty:
        st.info('Nothing was filtered out.')
    else:
        st.dataframe(
            failing[_display].reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
            column_config=_cfg,
        )


# ── Section 5 · Save ─────────────────────────────────────────────────────────

st.divider()
st.subheader('5 · Save to topic store')

if passing.empty:
    st.warning('No passing rows to save — nothing to embed.')
    st.stop()

new_rows = passing

with st.expander('➕ Custom columns', expanded=bool(st.session_state.sam_custom_cols)):
    st.caption(
        'Add extra columns to tag every saved topic with campaign-specific metadata. '
        'Columns are saved to the parquet and available as optional fields in HubSpot import.'
    )

    for entry in list(st.session_state.sam_custom_cols):
        sc1, sc2, sc3 = st.columns([2, 4, 1])
        with sc1:
            st.text(entry['name'])
        with sc2:
            entry['value'] = st.text_input(
                'Value', value=entry['value'],
                key=f'scfill_{entry["name"]}', label_visibility='collapsed',
            )
        with sc3:
            if st.button('✕', key=f'scrm_{entry["name"]}', use_container_width=True):
                st.session_state.sam_custom_cols = [
                    e for e in st.session_state.sam_custom_cols if e['name'] != entry['name']
                ]
                st.rerun()

    if st.session_state.sam_custom_cols:
        st.divider()

    sna1, sna2, sna3 = st.columns([2, 4, 1])
    with sna1:
        new_sc_name = st.text_input(
            'Column name', placeholder='e.g. campaign_name',
            key='new_sc_name', label_visibility='collapsed',
        )
    with sna2:
        new_sc_val = st.text_input(
            'Value', placeholder='e.g. Spring 2026',
            key='new_sc_val', label_visibility='collapsed',
        )
    with sna3:
        if st.button('Add', key='add_sc_btn', use_container_width=True):
            clean = new_sc_name.strip().replace(' ', '_').lower()
            existing_names = {e['name'] for e in st.session_state.sam_custom_cols}
            if (
                clean
                and clean not in _SAM_RESERVED_COLS
                and clean not in existing_names
            ):
                st.session_state.sam_custom_cols.append({'name': clean, 'value': new_sc_val.strip()})
                st.rerun()

st.caption(
    f'**{len(new_rows)}** new rows will be embedded (`text-embedding-ada-002`) and saved to '
    f'`{_SAM_PREFIX}sam_gov_{{date}}_{{hex}}.parquet`.'
)

if st.button('💾 Embed & Save', type='primary'):
    oai_key  = st.secrets['openai_api_key']
    anth_key = st.secrets['anthropic_api_key']
    extra_cols_dict = {e['name']: e['value'] for e in st.session_state.sam_custom_cols} or None
    try:
        path = _embed_and_save(new_rows.reset_index(drop=True), col_map, oai_key, anth_key, extra_cols=extra_cols_dict)
        st.success(f'Saved **{path}** — {len(new_rows)} topics ready for matching.')
        st.session_state.sam_raw_df        = None
        st.session_state.sam_screened_df   = None
        st.session_state.sam_existing_keys = None
        st.session_state.sam_col_map       = None
        st.session_state.sam_custom_cols   = []
        st.rerun()
    except Exception as e:
        st.error(f'Save failed: {e}')
