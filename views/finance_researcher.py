"""
Client Research
---------------
Run OpenAI Deep Research on client companies from data/all-contacts/clients/
and write the structured findings back onto their contact rows. Two research
focuses share the same launch/poll/apply flow:

  💰 Financials       → financial_data + financial_summary + financials_updated_at
                        (runs stored in finance-research-runs/, run IDs finres_*)
  🔬 Technology & R&D → technology_data + technology_summary + technology_updated_at
                        (runs stored in tech-research-runs/, run IDs techres_*)

Financial runs never touch summaries or embeddings. Technology runs can
optionally (checkbox at apply time, on by default) rewrite each company's
matching `summary` from the researched technology and re-embed it — this
intentionally changes grant matching for those companies.

Deep Research calls take 2-10 minutes each, so they are launched as
background tasks via the Responses API and polled — run state persists to
GCS at {runs_prefix}{run_id}/state.json so a refresh (or another session)
can resume monitoring with the Resume expander.
"""

import io
import json
import re
import time
import traceback
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account
from openai import OpenAI

import src.modules.finance_research as fr
import src.modules.tech_research as tr
from src.modules.Embedding.text_embedder import TextProcessor
from src.modules.GoogleBucketManager.bucket_manager import BucketManager

# ── Constants ──────────────────────────────────────────────────────────────

_BUCKET         = 'cc-matcher-bucket-jeg-v1'
_CLIENTS_PREFIX = 'data/all-contacts/clients/'
_POLL_INTERVAL  = 30   # seconds between Deep Research status checks

# Everything focus-specific lives here — the rest of the view is generic.
_FOCUS: dict[str, dict] = {
    'financials': {
        'title':        '💰 Financials',
        'badge':        '💰',
        'runs_prefix':  'finance-research-runs/',
        'id_prefix':    'finres_',
        'sections':     fr.FIELD_SECTIONS,
        'fields':       fr.ALL_FIELDS,
        'build_prompt': fr.build_research_prompt,
        'digest':       fr.build_financial_digest,
        'data_col':     'financial_data',
        'summary_col':  'financial_summary',
        'updated_col':  'financials_updated_at',
        'score_field':  'score_total',
        'score_label':  'Score',
        'results_cols': ['company_name', 'revenue_estimate', 'total_grant_funding',
                         'federal_awards_total_3yr', 'employee_count_current',
                         'score_total', 'recommendation'],
    },
    'technology': {
        'title':        '🔬 Technology & R&D',
        'badge':        '🔬',
        'runs_prefix':  'tech-research-runs/',
        'id_prefix':    'techres_',
        'sections':     tr.FIELD_SECTIONS,
        'fields':       tr.ALL_FIELDS,
        'build_prompt': tr.build_research_prompt,
        'digest':       tr.build_tech_digest,
        'data_col':     'technology_data',
        'summary_col':  'technology_summary',
        'updated_col':  'technology_updated_at',
        'score_field':  'confidence_score',
        'score_label':  'Confidence',
        'results_cols': ['company_name', 'technology_categories', 'flagship_products',
                         'trl_estimate', 'commercialization_stage', 'confidence_score'],
        # Tech findings describe what the company does — offer to rewrite the
        # matching summary + embedding from them at apply time.
        'matching_summary': tr.build_matching_summary,
    },
}


def _focus_for_run(run_id: str, state: dict | None = None) -> str:
    if state and state.get('focus') in _FOCUS:
        return state['focus']
    return 'technology' if run_id.startswith('techres_') else 'financials'


def _runs_prefix_for(run_id: str) -> str:
    return _FOCUS[_focus_for_run(run_id)]['runs_prefix']


# ── GCS ────────────────────────────────────────────────────────────────────

def _get_storage_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)


def _load_client_frames() -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Returns ({blob_name: df}, errors). Frames are kept per-blob so
    research columns can be written back to the exact file they came from."""
    client = _get_storage_client()
    blobs  = list(client.list_blobs(_BUCKET, prefix=_CLIENTS_PREFIX))

    frames: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for blob in blobs:
        if not blob.name.endswith('.parquet'):
            continue
        try:
            frames[blob.name] = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        except Exception as e:
            errors.append(f'{blob.name}: {e}')

    return frames, errors


def _company_key(row: pd.Series) -> str:
    name    = str(row.get('company_name') or '').strip()
    website = str(row.get('companyWebsite') or '').strip()
    return f'{name}||{website}'


def _group_mask(df: pd.DataFrame, key: str) -> pd.Series:
    name, website = key.split('||', 1)
    names    = df.get('company_name', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    websites = df.get('companyWebsite', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    return (names == name) & (websites == website)


# ── Run state (GCS-checkpointed) ───────────────────────────────────────────

def _state_blob(client: storage.Client, run_id: str):
    return client.bucket(_BUCKET).blob(f'{_runs_prefix_for(run_id)}{run_id}/state.json')


def _load_state(client: storage.Client, run_id: str) -> dict | None:
    blob = _state_blob(client, run_id)
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def _save_state(client: storage.Client, run_id: str, state: dict) -> None:
    _state_blob(client, run_id).upload_from_string(
        json.dumps(state), content_type='application/json'
    )


def _save_raw(client: storage.Client, run_id: str, idx: int, name: str, raw: str) -> None:
    slug = re.sub(r'[^A-Za-z0-9_-]+', '_', name)[:40] or 'company'
    client.bucket(_BUCKET).blob(
        f'{_runs_prefix_for(run_id)}{run_id}/raw/{idx:03d}_{slug}.txt'
    ).upload_from_string(raw, content_type='text/plain')


# ── Results helpers ────────────────────────────────────────────────────────

def _results_df(companies: list[dict]) -> pd.DataFrame:
    rows = []
    for c in companies:
        if c.get('output'):
            rows.append({'company_name': c['company_name'],
                         'website': c['website'], **c['output']})
    return pd.DataFrame(rows)


def _render_output(output: dict, sections) -> None:
    for section, fields in sections:
        st.markdown(f'**{section}**')
        st.dataframe(
            pd.DataFrame(
                [(f, output.get(f, '')) for f, _ in fields],
                columns=['Field', 'Value'],
            ),
            hide_index=True, use_container_width=True,
        )


def _apply_to_clients(state: dict, cfg: dict, update_summary: bool = False) -> None:
    """Write the focus's research columns onto every contact row of each
    researched company and rewrite the source parquets in place. With
    update_summary (technology focus only), also rewrite each company's
    matching `summary` from the researched technology and re-embed it."""
    completed = [c for c in state['companies'] if c.get('output')]

    with st.spinner('Loading client files from GCS…'):
        frames, load_errors = _load_client_frames()
    for err in load_errors:
        st.warning(err)

    tp = None
    if update_summary and cfg.get('matching_summary'):
        try:
            tp = TextProcessor(api_key=st.secrets['openai_api_key'])
        except Exception as e:
            st.error(f'Could not initialize embedder — summaries left unchanged: {e}')

    bm            = BucketManager(_BUCKET, client=_get_storage_client())
    today         = date.today().isoformat()
    touched_blobs = set()
    rows_updated  = 0
    summaries_updated = 0
    missing       = []

    for c in completed:
        new_summary, new_emb = None, None
        if tp is not None:
            text = cfg['matching_summary'](c['output']).strip()
            if text:
                try:
                    # float64 to match the dtype of existing rows — pyarrow
                    # cannot mix float32 and float64 ndarrays in one column
                    new_emb     = np.array(tp.get_embedding(text), dtype=np.float64)
                    new_summary = text
                except Exception as e:
                    st.warning(
                        f"{c['company_name']}: embedding failed — research columns "
                        f'saved, summary left unchanged ({e})'
                    )
            else:
                st.warning(
                    f"{c['company_name']}: no usable technology text — "
                    'summary left unchanged.'
                )

        found = False
        for blob_name, df in frames.items():
            mask = _group_mask(df, c['key'])
            if not mask.any():
                continue
            found = True
            df.loc[mask, cfg['data_col']]    = json.dumps(c['output'])
            df.loc[mask, cfg['summary_col']] = cfg['digest'](c['output'])
            df.loc[mask, cfg['updated_col']] = today
            if new_summary is not None:
                df.loc[mask, 'summary'] = new_summary
                for idx in df.index[mask]:
                    df.at[idx, 'embeddings'] = new_emb
            touched_blobs.add(blob_name)
            rows_updated += int(mask.sum())
        if not found:
            missing.append(c['company_name'])
        elif new_summary is not None:
            summaries_updated += 1

    for blob_name in sorted(touched_blobs):
        try:
            bm.upload_file(blob_name, frames[blob_name])
        except Exception as e:
            st.error(f'Failed to write {blob_name}: {e}')
            st.stop()

    if missing:
        st.warning('No client rows found for: ' + ', '.join(missing))

    state['applied'] = True
    _save_state(_get_storage_client(), state['run_id'], state)
    st.session_state.pop('fr_frames', None)   # force reload of client list
    msg = (
        f'Updated **{rows_updated}** contact row{"s" if rows_updated != 1 else ""} '
        f'across **{len(touched_blobs)}** file{"s" if len(touched_blobs) != 1 else ""} '
        f'for {len(completed) - len(missing)} compan'
        f'{"ies" if len(completed) - len(missing) != 1 else "y"}.'
    )
    if update_summary:
        msg += (
            f' Matching summaries re-embedded for **{summaries_updated}** '
            f'compan{"ies" if summaries_updated != 1 else "y"}.'
        )
    st.success(msg)


# ── Session state init ─────────────────────────────────────────────────────

if 'fr_active_run' not in st.session_state:
    st.session_state.fr_active_run = None


# ── Page ───────────────────────────────────────────────────────────────────

st.title('🧪 Client Research')
st.caption(
    'Run OpenAI Deep Research on client companies — financial diligence or '
    'technology/R&D profiling — then save the findings onto their client '
    'profiles. Financial runs never modify summaries or embeddings; '
    'technology runs can optionally rewrite the matching summary and re-embed.'
)

# ── Active run monitor (shown above everything else while a run exists) ────

if st.session_state.fr_active_run:
    run_id = st.session_state.fr_active_run
    st.subheader('🔄 Research run')

    try:
        gcs   = _get_storage_client()
        state = _load_state(gcs, run_id)
    except Exception as e:
        st.error(f'Error loading run state: {e}')
        st.code(traceback.format_exc())
        if st.button('Stop monitoring'):
            st.session_state.fr_active_run = None
            st.rerun()
        st.stop()

    if state is None:
        st.error(f'No state found for run `{run_id}`.')
        if st.button('Clear'):
            st.session_state.fr_active_run = None
            st.rerun()
        st.stop()

    cfg = _FOCUS[_focus_for_run(run_id, state)]
    st.caption(f"Run ID: `{run_id}`  ·  Focus: {cfg['title']}")

    companies = state['companies']
    pending   = [c for c in companies if c['status'] == 'pending']

    # Poll pending Deep Research tasks
    if pending:
        try:
            oai     = OpenAI(api_key=st.secrets['openai_api_key'])
            changed = False
            for c in pending:
                try:
                    resp = oai.responses.retrieve(c['response_id'])
                except Exception:
                    continue   # transient — retry next poll
                if resp.status in ('queued', 'in_progress'):
                    continue
                changed = True
                if resp.status == 'completed':
                    raw = resp.output_text or ''
                    try:
                        _save_raw(gcs, run_id, c['idx'], c['company_name'], raw)
                    except Exception:
                        pass
                    if getattr(resp, 'usage', None):
                        c['cost_usd'] = fr.response_cost_usd(state['model'], resp.usage)
                    parsed, err = fr.parse_research_output(oai, raw, fields=cfg['fields'])
                    if parsed:
                        c['status'] = 'completed'
                        c['output'] = parsed
                    else:
                        c['status'] = 'error'
                        c['error']  = err
                else:
                    c['status'] = 'error'
                    err = getattr(resp, 'error', None)
                    c['error'] = str(err) if err else f'Deep Research task {resp.status}'
            if changed:
                _save_state(gcs, run_id, state)
        except Exception as e:
            st.error(f'Error polling Deep Research tasks: {e}')
            st.code(traceback.format_exc())

    n_done  = sum(1 for c in companies if c['status'] == 'completed')
    n_err   = sum(1 for c in companies if c['status'] == 'error')
    n_pend  = len(companies) - n_done - n_err
    cost    = sum(c.get('cost_usd') or 0 for c in companies)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Companies', len(companies))
    m2.metric('Completed', n_done)
    m3.metric('Errors', n_err)
    m4.metric('Cost so far', f'${cost:,.2f}')
    st.progress((n_done + n_err) / max(len(companies), 1))

    st.dataframe(
        pd.DataFrame([{
            'Company':           c['company_name'],
            'Status':            c['status'],
            cfg['score_label']:  (c.get('output') or {}).get(cfg['score_field'], ''),
            'Cost':              f"${c.get('cost_usd') or 0:,.2f}",
            'Error':             c.get('error') or '',
        } for c in companies]),
        hide_index=True, use_container_width=True,
    )

    if n_pend:
        st.info(
            f'{n_pend} research task{"s" if n_pend != 1 else ""} still running — '
            f'Deep Research takes 2–10 minutes per company. '
            f'Checking again in {_POLL_INTERVAL}s.'
        )
        if st.button('Stop monitoring (tasks keep running — resume with the Run ID)'):
            st.session_state.fr_active_run = None
            st.rerun()
        time.sleep(_POLL_INTERVAL)
        st.rerun()

    # ── Run finished — review + apply ──────────────────────────────────────

    st.success('Run complete.')

    results = _results_df(companies)
    if not results.empty:
        st.subheader('Results')
        st.dataframe(
            results[[c for c in cfg['results_cols'] if c in results.columns]],
            hide_index=True, use_container_width=True,
        )
        for c in companies:
            if c.get('output'):
                with st.expander(f"{cfg['badge']} {c['company_name']}"):
                    st.caption(cfg['digest'](c['output']) or 'No headline findings.')
                    _render_output(c['output'], cfg['sections'])
        st.download_button(
            '⬇️ Download results CSV',
            results.to_csv(index=False).encode('utf-8'),
            file_name=f'{run_id}_results.csv',
            mime='text/csv',
        )

        st.divider()
        if state.get('applied'):
            st.info('These results have already been applied to the client profiles.')

        update_summary = False
        if cfg.get('matching_summary'):
            update_summary = st.checkbox(
                '🧬 Also rewrite the matching summary & re-embed from the '
                'researched technology',
                value=True,
                key='fr_update_summary',
                help='Replaces each company\'s `summary` with a description '
                     'assembled from the technology findings and re-embeds it. '
                     'This changes grant matching for these companies. Uncheck '
                     'to save the research columns only.',
            )

        apply_help = (
            f"Writes {cfg['data_col']}, {cfg['summary_col']}, and "
            f"{cfg['updated_col']} onto every contact row of each "
            'researched company.'
        )
        if cfg.get('matching_summary'):
            apply_help += (
                ' With the checkbox above, also rewrites summary + embeddings.'
            )
        else:
            apply_help += ' Summaries and embeddings are untouched.'

        if st.button('💾 Apply to client profiles', type='primary', help=apply_help):
            _apply_to_clients(state, cfg, update_summary=update_summary)
    else:
        st.warning('No companies produced usable results.')

    for c in companies:
        if c['status'] == 'error':
            st.error(f"{c['company_name']}: {c.get('error')}")

    if st.button('Close run'):
        st.session_state.fr_active_run = None
        st.rerun()

    st.stop()

# ── Resume a previous run ──────────────────────────────────────────────────

with st.expander('Resume monitoring a previous research run'):
    resume_id = st.text_input(
        'Run ID',
        key='fr_resume_run_id',
        placeholder='finres_2026-07-29_10-30-00 or techres_2026-07-29_10-30-00',
    )
    if st.button('Check status') and resume_id.strip():
        st.session_state.fr_active_run = resume_id.strip()
        st.rerun()

# ── Load clients ───────────────────────────────────────────────────────────

col_reload, col_count = st.columns([1, 5])
with col_reload:
    if st.button('↺ Reload', help='Refresh client data from GCS'):
        st.session_state.pop('fr_frames', None)
        st.rerun()

if 'fr_frames' not in st.session_state:
    with st.spinner('Loading clients from GCS…'):
        frames, load_errors = _load_client_frames()
    st.session_state.fr_frames = frames
    for err in load_errors:
        st.warning(err)

frames: dict[str, pd.DataFrame] = st.session_state.fr_frames

if not frames:
    st.warning(f'No parquet files found under {_CLIENTS_PREFIX} in GCS.')
    st.stop()

combined = pd.concat(frames.values(), ignore_index=True)
combined['_key'] = combined.apply(_company_key, axis=1)

_agg = {
    'company_name':   ('company_name', 'first'),
    'companyWebsite': ('companyWebsite', 'first'),
    'state': ('state', 'first') if 'state' in combined.columns else ('_key', lambda s: ''),
}
for _col in ('financials_updated_at', 'financial_data',
             'technology_updated_at', 'technology_data'):
    _agg[_col] = (_col, 'first') if _col in combined.columns else ('_key', lambda s: '')

groups = (
    combined.groupby('_key', sort=False)
    .agg(**_agg)
    .reset_index()
    .sort_values('company_name', key=lambda s: s.fillna('').str.lower())
)

with col_count:
    st.info(f'{len(groups):,} client companies loaded.')

# ── 1 · Select clients ─────────────────────────────────────────────────────

st.divider()
st.subheader('1 · Select clients to research')


def _label(row) -> str:
    label = f"{row['company_name'] or '—'}  ·  {row['companyWebsite'] or 'no website'}"
    for col, badge in (('financials_updated_at', '💰'), ('technology_updated_at', '🔬')):
        updated = str(row[col] or '').strip()
        if updated and updated.lower() != 'nan':
            label += f'  ·  {badge} {updated}'
    return label


labels = {row['_key']: _label(row) for _, row in groups.iterrows()}

select_all = st.checkbox('Select all clients')
selected_keys = st.multiselect(
    'Client companies',
    options=list(labels.keys()),
    default=list(labels.keys()) if select_all else [],
    format_func=lambda k: labels[k],
)

# Inspect what's already stored for a company
_has_any = groups[
    groups['financial_data'].fillna('').astype(str).str.strip().str.startswith('{')
    | groups['technology_data'].fillna('').astype(str).str.strip().str.startswith('{')
]
if not _has_any.empty:
    with st.expander('View stored research data'):
        view_key = st.selectbox(
            'Company',
            options=list(_has_any['_key']),
            format_func=lambda k: labels[k],
            key='fr_view_stored',
        )
        row = _has_any[_has_any['_key'] == view_key].iloc[0]
        for fkey, fcfg in _FOCUS.items():
            raw_stored = str(row[fcfg['data_col']] or '').strip()
            if not raw_stored.startswith('{'):
                continue
            st.markdown(f"### {fcfg['title']}")
            try:
                stored = json.loads(raw_stored)
                st.caption(fcfg['digest'](stored) or '')
                _render_output(stored, fcfg['sections'])
            except Exception as e:
                st.warning(f"Could not parse stored {fcfg['data_col']}: {e}")

if not selected_keys:
    st.caption('Select at least one client to continue.')
    st.stop()

# ── 2 · Configure & start ──────────────────────────────────────────────────

st.divider()
st.subheader('2 · Configure research')

focus_key = st.radio(
    'Research focus',
    options=list(_FOCUS.keys()),
    format_func=lambda k: _FOCUS[k]['title'],
    horizontal=True,
    help='Financials: revenue, funding, federal awards, proposal readiness. '
         'Technology & R&D: core technology, products, patents, TRL, and '
         'grant-alignment keywords.',
)
cfg = _FOCUS[focus_key]

model = st.radio(
    'Deep Research model',
    options=fr.DEEP_RESEARCH_MODELS,
    index=1,   # default to Terra — the recommended balance of cost and depth
    format_func=lambda m: f'{m}  ({fr.EST_COST_LABEL[m]})',
    help='Terra is recommended for most batches. Luna is the cheapest but '
         'noticeably shallower; Sol is the most thorough for high-value clients.',
)

est_total = fr.EST_COST_PER_COMPANY[model] * len(selected_keys)
st.metric(
    'Estimated cost',
    f'~${est_total:,.0f}',
    help='Rough estimate — actual cost is computed from token usage per company.',
)

confirmed = True
if est_total > fr.COST_CONFIRM_THRESHOLD_USD:
    confirmed = st.checkbox(
        f'I understand this run may cost roughly ${est_total:,.0f} '
        f'({len(selected_keys)} companies × {fr.EST_COST_LABEL[model]}).'
    )

start = st.button(
    f"🚀 Start {cfg['title']} research on {len(selected_keys)} compan"
    f'{"ies" if len(selected_keys) != 1 else "y"}',
    type='primary',
    disabled=not confirmed,
)

if not start:
    st.stop()

run_id = f"{cfg['id_prefix']}{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

try:
    oai       = OpenAI(api_key=st.secrets['openai_api_key'])
    companies = []
    bar       = st.progress(0.0, text='Launching Deep Research tasks…')

    for i, key in enumerate(selected_keys):
        row   = groups[groups['_key'] == key].iloc[0]
        entry = {
            'idx':          i,
            'key':          key,
            'company_name': str(row['company_name'] or ''),
            'website':      str(row['companyWebsite'] or ''),
            'response_id':  None,
            'status':       'pending',
            'error':        None,
            'cost_usd':     0.0,
            'output':       None,
        }
        try:
            resp = oai.responses.create(
                model=model,
                input=cfg['build_prompt']({
                    'company_name': entry['company_name'],
                    'website':      entry['website'],
                    'state':        str(row.get('state') or ''),
                }),
                background=True,
                tools=[{'type': 'web_search'}],
            )
            entry['response_id'] = resp.id
        except Exception as e:
            entry['status'] = 'error'
            entry['error']  = f'Failed to launch: {e}'
        companies.append(entry)
        bar.progress((i + 1) / len(selected_keys),
                     text=f'Launched {i + 1}/{len(selected_keys)}')

    state = {
        'run_id':     run_id,
        'focus':      focus_key,
        'model':      model,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'applied':    False,
        'companies':  companies,
    }
    _save_state(_get_storage_client(), run_id, state)
    st.session_state.fr_active_run = run_id
    st.rerun()

except Exception as e:
    st.error(f'Failed to start run: {e}')
    st.code(traceback.format_exc())
