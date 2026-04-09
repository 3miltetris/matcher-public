"""
Topic Importer
--------------
Upload a PDF or paste raw text from a grant solicitation.
Claude extracts every topic into an editable table.
Review, set the agency, then save with embeddings to the processed store.
"""

import json
import secrets
from datetime import datetime

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from anthropic import Anthropic
from google.oauth2 import service_account
from google.cloud import storage

from src.modules.Embedding.text_embedder import TextProcessor
from src.modules.GoogleBucketManager.bucket_manager import BucketManager

# ── GCS ────────────────────────────────────────────────────────────────────

_BUCKET        = 'cc-matcher-bucket-jeg-v1'
_TOPICS_PREFIX = 'data/all-topics/processed/'


def _get_storage_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)

# ── Extraction prompt ──────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You are a precise grant solicitation parser. Extract every distinct research \
topic or subtopic from the provided solicitation document.

For each topic, extract:
- topic_number: The topic or subtopic identifier (e.g., "A24-001", "N241-001", "Topic 3", etc.). Null if not present.
- title: The verbatim title of the topic. If no explicit title exists, create a concise 8-10 word snippet from the start of the description.
- description: The full verbatim description text for that topic. Include all technical requirements, objectives, and scope language. Do not truncate.
- due_date: Any due date, close date, or submission deadline associated with the topic. Null if not present. Use ISO format YYYY-MM-DD if possible.

Return ONLY a valid JSON array. No preamble, no markdown, no backticks.
CRITICAL: All string values must be properly JSON-escaped. Do NOT include literal newline characters inside JSON string values, use \\n instead.

Example:
[{"topic_number":"A24-001","title":"Autonomous UUV Navigation Systems","description":"Full verbatim description here.","due_date":"2024-09-15"}]

If the document has a single global due date, apply it to all topics. Extract ALL topics.\
"""

_EXTRACT_MODEL = 'claude-sonnet-4-6'
_EMBED_MODEL   = 'text-embedding-ada-002'
_COL_ORDER     = ['topic_number', 'title', 'agency', 'source', 'due_date', 'scraped_at', 'description']


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_broad_agencies() -> list[str]:
    try:
        client = _get_storage_client()
        blobs = client.list_blobs(_BUCKET, prefix=_TOPICS_PREFIX, delimiter='/')
        list(blobs)  # consume iterator to populate prefixes
        return sorted(
            p.replace(_TOPICS_PREFIX, '').strip('/')
            for p in blobs.prefixes
        )
    except Exception:
        return []


def _pdf_to_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype='pdf')
    text = '\n'.join(page.get_text() for page in doc)
    doc.close()
    return text


def _extract_topics(text: str, anth_key: str) -> list[dict]:
    text = text.strip()
    if not text:
        raise ValueError('Document text is empty — nothing to extract.')
    client = Anthropic(api_key=anth_key)
    with client.messages.stream(
        model=_EXTRACT_MODEL,
        max_tokens=64000,
        system=_EXTRACTION_PROMPT,
        messages=[{
            'role': 'user',
            'content': text,
        }],
    ) as stream:
        final = stream.get_final_message()
    if final.stop_reason == 'max_tokens':
        raise ValueError(
            'Claude hit the output token limit before finishing — the document '
            'is too large to process in one pass. Try splitting it into smaller '
            'sections (e.g. one agency or one batch of topics at a time).'
        )
    raw = final.content[0].text.strip()
    # Strip markdown code fences if Claude wrapped the output anyway
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[-1]
        raw = raw.rsplit('```', 1)[0].strip()
    return json.loads(raw)


def _build_df(topics: list[dict], sub_agency: str, source: str = '') -> pd.DataFrame:
    df = pd.DataFrame(topics)
    for col in ['topic_number', 'title', 'description', 'due_date']:
        if col not in df.columns:
            df[col] = None
    df['agency']     = sub_agency
    df['source']     = source
    df['scraped_at'] = datetime.today().strftime('%Y-%m-%d')
    present = [c for c in _COL_ORDER if c in df.columns]
    extra   = [c for c in df.columns if c not in _COL_ORDER]
    return df[present + extra].reset_index(drop=True)


def _embed_and_save(df: pd.DataFrame, broad_agency: str, oai_key: str) -> list[str]:
    tp     = TextProcessor(api_key=oai_key)
    bm     = BucketManager(_BUCKET, client=_get_storage_client())
    today  = datetime.today().strftime('%Y-%m-%d')
    saved  = []

    for sub_agency, group in df.groupby('agency'):
        group    = group.copy()
        progress = st.progress(0, text=f'Embedding topics for **{sub_agency}**…')
        embeddings = []
        for i, desc in enumerate(group['description'].astype(str)):
            embeddings.append(tp.get_embedding(desc) if desc.strip() else None)
            progress.progress((i + 1) / len(group), text=f'Embedding {i + 1}/{len(group)} — {sub_agency}')
        group['embeddings'] = embeddings
        progress.empty()

        hex_suffix = secrets.token_hex(3)  # 6-char hex, e.g. "a3f9c1"
        gcs_path = f'{_TOPICS_PREFIX}{broad_agency}/{sub_agency}_{today}_{hex_suffix}.parquet'
        bm.upload_file(gcs_path, group)
        saved.append(gcs_path)

    return saved


# ── Session state ──────────────────────────────────────────────────────────

if 'topics_df' not in st.session_state:
    st.session_state.topics_df = None
if 'save_results' not in st.session_state:
    st.session_state.save_results = []


# ── Page ───────────────────────────────────────────────────────────────────

st.title('📄 Topic Importer')
st.caption(
    'Extract grant topics from a solicitation PDF or pasted text. '
    'Review and edit, then save with embeddings to the processed store.'
)

# Show any post-save success banners
for msg in st.session_state.save_results:
    st.success(msg)

# ── Section 1 · Input document ─────────────────────────────────────────────

st.subheader('1 · Input document')

pdf_tab, text_tab = st.tabs(['📎 Upload PDF', '📋 Paste Text'])
with pdf_tab:
    uploaded_pdf = st.file_uploader('PDF solicitation file', type='pdf', label_visibility='collapsed')
with text_tab:
    pasted_text = st.text_area(
        'Paste text',
        height=220,
        placeholder='Paste the full solicitation text here…',
        label_visibility='collapsed',
    )

input_col1, input_col2 = st.columns(2)
with input_col1:
    sub_agency_input = st.text_input(
        'Sub-agency',
        placeholder='e.g. ARMY, USSOCOM, NCI — pre-fills the agency column for all extracted topics',
    )
with input_col2:
    source_input = st.text_input(
        'Source',
        placeholder='e.g. SAM.gov, Agency website — pre-fills the source column for all extracted topics',
    )

extract_btn = st.button('⚡ Extract Topics', type='primary')

if extract_btn:
    source_text = ''
    if uploaded_pdf is not None:
        with st.spinner('Reading PDF…'):
            source_text = _pdf_to_text(uploaded_pdf.read())
    elif pasted_text.strip():
        source_text = pasted_text.strip()
    else:
        st.error('Please upload a PDF or paste text first.')

    if source_text:
        char_count = len(source_text)
        if char_count > 400_000:
            st.warning(
                f'Document is very long ({char_count:,} chars). '
                'Consider trimming it to the relevant section to reduce cost and latency.'
            )

        anth_key = st.secrets['anthropic_api_key']

        with st.spinner('Calling Claude to extract topics — this may take a moment for long documents…'):
            try:
                topics = _extract_topics(source_text, anth_key)
            except json.JSONDecodeError as e:
                st.error(f'Claude returned unparseable JSON: {e}')
                st.stop()
            except Exception as e:
                st.error(f'Extraction failed: {e}')
                st.stop()

        if not topics:
            st.warning('No topics found in the document.')
        else:
            st.session_state.topics_df   = _build_df(topics, sub_agency_input.strip(), source_input.strip())
            st.session_state.save_results = []
            st.success(f'Extracted **{len(topics)}** topic(s). Review and edit below.')

# ── Section 2 · Review & edit ──────────────────────────────────────────────

if st.session_state.topics_df is not None:
    st.divider()
    st.subheader('2 · Review & edit')

    # Agency quick-fill bar
    fill_left, fill_right, _ = st.columns([2, 1, 3])
    with fill_left:
        fill_value = st.text_input(
            'Agency name',
            value=sub_agency_input.strip(),
            placeholder='Type agency name…',
            label_visibility='collapsed',
            key='agency_fill_input',
        )
    with fill_right:
        if st.button('Apply to all rows', key='apply_agency', width='stretch'):
            df = st.session_state.topics_df.copy()
            df['agency'] = fill_value.strip()
            st.session_state.topics_df = df
            st.rerun()

    # Source quick-fill bar
    src_left, src_right, _ = st.columns([2, 1, 3])
    with src_left:
        source_fill_value = st.text_input(
            'Source',
            value=source_input.strip(),
            placeholder='e.g. SAM.gov, Agency website…',
            label_visibility='collapsed',
            key='source_fill_input',
        )
    with src_right:
        if st.button('Apply to all rows', key='apply_source', width='stretch'):
            df = st.session_state.topics_df.copy()
            df['source'] = source_fill_value.strip()
            st.session_state.topics_df = df
            st.rerun()

    # Data editor — always sync edits back to session state so they survive reruns
    edited_df = st.data_editor(
        st.session_state.topics_df,
        width='stretch',
        num_rows='dynamic',
        column_config={
            'topic_number': st.column_config.TextColumn('Topic #',      width='small'),
            'title':        st.column_config.TextColumn('Title',        width='medium'),
            'agency':       st.column_config.TextColumn('Agency',       width='small'),
            'source':       st.column_config.TextColumn('Source',       width='small'),
            'due_date':     st.column_config.TextColumn('Due Date',     width='small'),
            'scraped_at':   st.column_config.TextColumn('Scraped At',   width='small'),
            'description':  st.column_config.TextColumn('Description',  width='large'),
        },
        hide_index=True,
        key='topics_editor',
    )
    # Persist edits so "Apply to all rows" and Save see the latest table state
    st.session_state.topics_df = edited_df

    # ── Section 3 · Save ──────────────────────────────────────────────────

    st.divider()
    st.subheader('3 · Save to processed store')

    existing_agencies = _get_broad_agencies()
    agency_options    = existing_agencies + ['+ New agency…']

    dest_col, new_col = st.columns([2, 2])
    with dest_col:
        selection = st.selectbox(
            'Broad agency (destination folder)',
            agency_options,
            help='Determines the subfolder under `processed/` the parquet is saved into.',
        )
    broad_agency = ''
    if selection == '+ New agency…':
        with new_col:
            broad_agency = st.text_input(
                'New broad agency name',
                placeholder='e.g. DOD, HHS, NASA',
            ).strip().upper()
    else:
        broad_agency = selection

    n_topics    = len(edited_df)
    n_agencies  = edited_df['agency'].nunique()
    agency_word = 'sub-agency' if n_agencies == 1 else 'sub-agencies'
    dest_label  = f'`processed/{broad_agency}/`' if broad_agency else '`processed/…/`'

    st.caption(
        f'**{n_topics}** topic(s) across **{n_agencies}** {agency_word} → {dest_label}  '
        f'— one parquet per unique sub-agency, named `{{sub_agency}}_{{date}}_{{hex}}.parquet`'
    )

    save_disabled = not broad_agency
    if st.button('💾 Save & Embed', type='primary', disabled=save_disabled):
        descriptions = edited_df['description'].astype(str).str.strip()
        if descriptions.eq('').all() or descriptions.eq('None').all():
            st.error('All descriptions are empty — nothing to embed.')
        else:
            oai_key = st.secrets['openai_api_key']

            try:
                out_paths = _embed_and_save(edited_df, broad_agency, oai_key)
                st.session_state.save_results = [
                    f'Saved **{p}**' for p in out_paths
                ]
                st.session_state.topics_df = None
                st.rerun()
            except Exception as e:
                st.error(f'Save failed: {e}')
