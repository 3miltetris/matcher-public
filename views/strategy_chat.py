"""
Funding Strategy
----------------
The roadmap conversation that used to happen in a Claude.ai Project, run here
against the match results the consultant just produced.

Flow: Grant Search or Aspect Match hands off a result set and a capability
profile -> pick a playbook (the general Funding Strategy one, or the NIH IC
portfolio one and its knowledge base) -> optionally attach documents from the
client's Google Drive folder -> work through the playbook's approval gates in
chat -> download the transcript as markdown.

Everything durable lives in GCS: playbooks under strategy-playbooks/, sessions
under strategy-sessions/{session_id}/session.json, rewritten after every turn.
A reload resumes rather than losing an hour of approvals, and a colleague can
open the same session by ID.

Two structural notes:

  * **This runs in the Streamlit process**, like Bulk Aspect Match — the page
    must stay open for the turn in flight. State is written after every turn,
    so the cost of a closed tab is the current answer, not the session.
  * **`st.stop()` is used freely below.** This is a top-level page, not one of
    the sub-views dispatched by a parent page, so unwinding the run is safe
    here (see the parent-page note in CLAUDE.md).
"""

import json
import time
import traceback
from datetime import datetime

import anthropic
import pandas as pd
import streamlit as st
from google.oauth2 import service_account

import src.modules.aspect_profile as ap
import src.modules.drive_client as dc
import src.modules.strategy_session as ss
import src.modules.ui_common as uc

_ASSIGN_BLOB = 'drive-sync-configs/assignments.json'

# Repaint the streaming answer at most this often (seconds). See on_text.
_PAINT_INTERVAL_S = 0.12

_DEFAULT_KICKOFF = (
    'Begin. The matched opportunities and the capability profile are attached '
    'above. Work through your stages in order and stop at each approval point.'
)

st.title('🗺️ Funding Strategy')
st.caption(
    'Turn a set of matched opportunities into an approved funding roadmap, '
    'without leaving the matcher.'
)

# "Continue on Haiku" after a partial refusal. It CONTINUES rather than
# re-generates: the partial stays in the history exactly as saved and one new
# consultant turn asks for the rest. Deleting the partial and regenerating
# would be a history edit, which newer models check against their own thinking
# blocks, and would throw away text the consultant already read. Appending
# keeps the history append-only and only pays for the missing part.
_FALLBACK_NAME = 'Haiku 4.5'
_CONTINUE_PROMPT = (
    'Your previous answer stopped part-way. Continue it from exactly where it '
    'stopped — do not repeat what you already wrote, and keep the same '
    'structure and approval gates.'
)

for _k, _v in {
    'sc_record':    None,     # the live session record
    'sc_pending':   False,    # an assistant turn is owed
    'sc_error':     None,
    'sc_drive_files': None,   # cached Drive listing for the selected company
    'sc_partial':   None,     # text streamed by a turn that then failed
    # One turn on a model other than the session's. Set only by the "continue
    # on Haiku" button after a partial refusal, and cleared as soon as that
    # turn runs — the session model is never changed, so the next turn goes
    # back to it and its cache stays warm.
    'sc_model_override': None,
}.items():
    st.session_state.setdefault(_k, _v)


# ── Clients ────────────────────────────────────────────────────────────────

def _gcs():
    return uc.get_storage_client()


def _anthropic():
    return anthropic.Anthropic(api_key=st.secrets['anthropic_api_key'])


def _drive_service():
    # The one place in this codebase that builds credentials WITH explicit
    # scopes — Drive rejects the default-scoped service-account token.
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account'], scopes=dc.DRIVE_SCOPES
    )
    return dc.build_drive_service(creds)


def _save(record: dict) -> None:
    try:
        ss.save_session(_gcs(), record)
    except Exception as e:
        # A failed checkpoint must not destroy the turn that was just paid for.
        st.warning(f'Session not saved to GCS ({e}). The conversation is still '
                   'on screen — download the transcript before reloading.')


# ── Resume ─────────────────────────────────────────────────────────────────

with st.expander('📂 Open an existing session', expanded=False):
    c1, c2 = st.columns([3, 1])
    resume_id = c1.text_input('Session ID', key='sc_resume_id',
                              placeholder='strategy_2026-09-28_10-30-00_acme')
    if c2.button('Open', key='sc_resume_go') and resume_id.strip():
        rec = ss.load_session(_gcs(), resume_id.strip())
        if rec is None:
            st.error(f'No session `{resume_id.strip()}` in GCS.')
        else:
            st.session_state.sc_record = rec
            st.session_state.sc_pending = False
            st.session_state.sc_error = None
            st.session_state.sc_drive_files = None
            st.rerun()

    if st.button('🔄 List recent sessions', key='sc_list'):
        try:
            rows = ss.list_sessions(_gcs())
            if rows:
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            else:
                st.caption('No sessions yet.')
        except Exception as e:
            st.error(f'Could not list sessions: {e}')


# ── Setup ──────────────────────────────────────────────────────────────────

def _handoff() -> dict | None:
    """The payload Grant Search / Aspect Match stashed before switching page."""
    return st.session_state.get('strategy_handoff')


def _render_setup() -> None:
    hand = _handoff()

    if hand:
        st.success(
            f"Handed off from **{hand.get('origin', 'a match run')}** — "
            f"{len(hand.get('results', [])):,} matched row(s)"
            + (f" for **{hand['company'].get('name', '')}**"
               if hand.get('company') else '')
        )
    else:
        st.info(
            'Start a session from **Grant Search** or **Aspect Match** — run a '
            'match, then press *🗺️ Start strategy session*. You can also begin '
            'here with pasted notes alone.'
        )

    # ── Playbook ───────────────────────────────────────────────────────────
    try:
        manifest = ss.load_manifest(_gcs())
    except Exception as e:
        st.error(f'Could not read the playbook registry: {e}')
        st.stop()

    books = manifest.get('playbooks', [])
    if not books:
        st.error(
            'No playbooks are installed. Upload them to '
            f'`{ss.PLAYBOOK_PREFIX}` in GCS and register them in '
            f'`{ss.MANIFEST_BLOB}`.'
        )
        st.stop()

    st.markdown('#### 1 · Playbook')
    labels = {p.get('label', p['id']): p['id'] for p in books}
    picked = st.radio('Playbook', list(labels), key='sc_playbook',
                      format_func=str)
    pb = ss.playbook(manifest, labels[picked])
    if pb.get('description'):
        st.caption(pb['description'])

    optional_keys: list[str] = []
    opts = pb.get('optional') or []
    if opts:
        with st.expander('Reference material to load', expanded=False):
            st.caption(
                'Each file is inlined into the system prompt and cached for the '
                'session, so it is paid for once rather than per turn — but it '
                'still counts against the context window. Load what the work '
                'needs.'
            )
            for asset in opts:
                hint  = asset.get('size_hint') or ''
                label = f"{asset['label']} · {hint}" if hint else asset['label']
                if st.checkbox(
                    label,
                    value=bool(asset.get('default')),
                    key=f"sc_opt_{pb['id']}_{asset['key']}",
                    help=asset.get('help'),
                ):
                    optional_keys.append(asset['key'])

    # ── Company + context ──────────────────────────────────────────────────
    st.markdown('#### 2 · Company and context')

    company = dict(hand.get('company') or {}) if hand else {}
    c1, c2 = st.columns(2)
    company['name'] = c1.text_input(
        'Company', value=company.get('name', ''), key='sc_company_name')
    company['website'] = c2.text_input(
        'Website', value=company.get('website', ''), key='sc_company_site')

    blocks: list[dict] = []
    if hand and hand.get('results') is not None:
        results = pd.DataFrame(hand['results'])
        if not results.empty:
            blocks.append(ss.results_block(results))
    if hand and hand.get('profile') is not None:
        try:
            blocks.append(ss.profile_block(dict(hand['profile'])))
        except Exception as e:
            st.warning(f'Capability profile could not be rendered: {e}')

    extra_notes = st.text_area(
        'Additional context (optional)', height=120, key='sc_setup_notes',
        placeholder='Anything the playbook should know up front — prior '
                    'awards, constraints, the client contact, target cycle…',
    )
    if extra_notes.strip():
        blocks.append(ss.notes_block(extra_notes))

    if blocks:
        st.caption('Attached at open: ' + ' · '.join(b['label'] for b in blocks))
    else:
        st.warning(
            'Nothing is attached. The playbook will have to ask for the '
            'opportunity list and company material before it can do useful work.'
        )

    # ── Run settings ───────────────────────────────────────────────────────
    st.markdown('#### 3 · Run settings')
    r1, r2 = st.columns(2)
    model = r1.selectbox(
        'Model', ss.MODELS, index=0, key='sc_model',
        format_func=lambda m: ss.MODEL_LABELS.get(m, m),
        help='Opus 5.5 is the default here — strongest on the portfolio and '
             '"why now" stages, and cheaper per token than Opus 5. Sonnet 4.6 '
             'matches every other Claude call in this app. On a client whose '
             'material trips the safety classifiers (dried or powdered '
             'biologics is the known case), pick Haiku 4.5 outright — it is '
             'the weakest model here but the one measured to complete that '
             'material; otherwise a refusal falls back to it automatically.',
    )
    web = r2.checkbox(
        '🌐 Live web search', value=True, key='sc_web',
        help='Lets Claude open solicitations, check receipt dates and find '
             'comparable awards during the conversation. Both playbooks assume '
             'it; without it, every "verify against the official source" step '
             'becomes a claim it cannot actually make. ~$0.01 per search.',
    )

    title = st.text_input(
        'Session title', key='sc_title',
        value=(f"{company.get('name', '')} funding strategy".strip()
               or 'Funding strategy session'),
    )

    if st.button('🚀 Start session', type='primary', key='sc_start'):
        try:
            with st.spinner('Loading the playbook…'):
                system_text, included = ss.assemble_system(
                    _gcs(), pb, optional_keys)
        except Exception as e:
            st.error(f'Could not assemble the playbook: {e}')
            st.code(traceback.format_exc())
            return

        record = {
            'session_id':     ss.new_session_id(company.get('name', '')),
            'title':          title.strip() or 'Funding strategy session',
            'created_at':     datetime.now().isoformat(timespec='seconds'),
            'created_by':     st.session_state.get('user_email', ''),
            'playbook':       pb['id'],
            'playbook_label': pb.get('label', pb['id']),
            'playbook_parts': included,
            'optional_keys':  optional_keys,
            'model':          model,
            'web_search':     bool(web),
            'company':        company,
            'context_blocks': blocks,
            'system_text':    system_text,
            'messages':       [],
            'cost_usd':       0.0,
        }
        _save(record)
        st.session_state.sc_record = record
        st.session_state.sc_error = None
        st.session_state.sc_drive_files = None
        st.session_state.pop('strategy_handoff', None)
        st.rerun()


# ── Drive attachments ──────────────────────────────────────────────────────

def _assigned_folders(company_key: str) -> tuple[list[str], str]:
    """(folder ids assigned to this company by Drive Sync, shared drive id).

    ([], '') when the company has no assignment — the normal state for a
    prospect or a profile built from pasted notes, not an error.
    """
    blob = _gcs().bucket(uc.BUCKET).blob(_ASSIGN_BLOB)
    if not blob.exists():
        return [], ''
    doc = json.loads(blob.download_as_text())
    folders = [
        fid for fid, a in (doc.get('assignments') or {}).items()
        if a.get('client_key') == company_key
    ]
    return folders, doc.get('drive_id', '')


def _render_drive(record: dict) -> None:
    company = record.get('company') or {}
    key = ap.company_key({
        'company_name':   company.get('name', ''),
        'companyWebsite': company.get('website', ''),
    })

    try:
        folders, drive_id = _assigned_folders(key)
    except Exception as e:
        st.warning(f'Could not read Drive assignments: {e}')
        return

    if not folders or not drive_id:
        st.info(
            'No Google Drive folder is assigned to this company. Assign one in '
            '**Client Sync → Google Drive**, or paste the material into the '
            'chat instead.'
        )
        return

    if st.button('📁 List documents', key='sc_drive_list'):
        try:
            svc = _drive_service()
            files = []
            for fid in folders:
                files.extend(dc.list_files_recursive(svc, drive_id, fid))
            files = [f for f in files if dc.is_extractable(f)]
            files.sort(key=lambda f: f.get('modifiedTime', ''), reverse=True)
            # Keyed by company: this cache is plain session state, so it
            # survives Close/Open. Unkeyed, opening another client's session
            # and pressing Attach would upload THIS client's internal
            # documents into that one and persist them there.
            st.session_state.sc_drive_files = {'key': key, 'files': files}
        except Exception as e:
            st.error(f'Drive listing failed: {e}')

    cached = st.session_state.get('sc_drive_files') or {}
    if cached.get('key') != key:
        return
    files = cached.get('files') or []
    if not files:
        return

    attached = ({b['label'] for b in record.get('context_blocks', [])}
                | set(record.get('attachments') or []))
    table = pd.DataFrame([{
        'Attach':   False,
        'Name':     f.get('name', ''),
        'Modified': (f.get('modifiedTime') or '')[:10],
        'id':       f['id'],
    } for f in files if f.get('name') not in attached])

    if table.empty:
        st.caption('Every extractable document in the folder is already attached.')
        return

    edited = st.data_editor(
        table, width='stretch', hide_index=True, key='sc_drive_table',
        column_config={'id': None,
                       'Attach': st.column_config.CheckboxColumn('Attach')},
    )
    chosen = edited[edited['Attach']]
    if chosen.empty:
        return

    if st.button(f'📎 Attach {len(chosen)} document(s)', key='sc_drive_attach'):
        by_id = {f['id']: f for f in files}
        svc   = _drive_service()
        added, failed = [], []
        prog = st.progress(0.0)
        for i, fid in enumerate(chosen['id'].tolist(), 1):
            f = by_id.get(fid)
            prog.progress(i / len(chosen), text=f.get('name', ''))
            text, note = dc.download_file_text(svc, f)
            if not text:
                failed.append(f"{f.get('name', '')} ({note or 'no text'})")
                continue
            added.append(ss.document_block(f.get('name', ''), text))
        prog.empty()

        for msg in failed:
            st.warning(f'Skipped {msg}')
        if not added:
            return

        # A document attached mid-session is a USER TURN, never an edit to the
        # context blocks: those are the second cached prefix, so appending to
        # them would re-bill the whole conversation AND send every document
        # twice (once in the prefix, once in this turn). Only the label is
        # recorded, for the sidebar and the transcript header.
        record.setdefault('attachments', []).extend(b['label'] for b in added)
        record['messages'].append({'role': 'user', 'content': [{
            'type': 'text',
            'text': ('I am attaching further material from the client\'s '
                     'Google Drive folder. Read it and tell me what it changes, '
                     'if anything, about the work so far.\n\n')
                    + '\n\n'.join(
                        f"========== {b['label']} ==========\n\n{b['text']}"
                        for b in added),
        }]})
        st.session_state.sc_pending = True
        st.session_state.sc_drive_files = None
        _save(record)
        st.rerun()


# ── Chat ───────────────────────────────────────────────────────────────────

def _render_chat(record: dict) -> None:
    company = (record.get('company') or {}).get('name', '')
    st.subheader(record.get('title', 'Session'))
    st.caption(
        f"`{record['session_id']}` · {record.get('playbook_label', '')} · "
        f"{record.get('model', '')}"
        + (' · 🌐 web search on' if record.get('web_search') else '')
        + (f" · {company}" if company else '')
    )

    with st.sidebar:
        st.markdown('### This session')
        st.metric('Cost so far', f"${float(record.get('cost_usd', 0.0)):.2f}")
        st.caption(f"{sum(1 for m in record['messages'] if m['role'] == 'user')} "
                   'consultant turn(s)')
        if record.get('playbook_parts'):
            with st.expander('Loaded reference'):
                for p in record['playbook_parts']:
                    st.caption(f'· {p}')
        if record.get('context_blocks') or record.get('attachments'):
            with st.expander('Attached context'):
                for b in record.get('context_blocks') or []:
                    st.caption(f"· {b['label']}")
                for label in record.get('attachments') or []:
                    st.caption(f"· 📎 {label}")
        st.download_button(
            '⬇ Transcript (.md)',
            ss.transcript_markdown(record).encode('utf-8'),
            file_name=f"{record['session_id']}.md",
            mime='text/markdown',
            width='stretch',
        )
        # Switchable mid-session: a turn refused by the chosen model may
        # complete on another, and starting a new session would throw away the
        # approvals already made.
        new_model = st.selectbox(
            'Model for the next turn', ss.MODELS,
            index=ss.MODELS.index(record['model'])
                  if record.get('model') in ss.MODELS else 0,
            format_func=lambda m: ss.MODEL_LABELS.get(m, m),
            key='sc_model_switch',
        )
        if new_model != record.get('model'):
            record['model'] = new_model
            _save(record)
            st.caption('Switched. The next turn re-writes the prompt cache once.')

        if st.button('← Close session', width='stretch'):
            st.session_state.sc_record = None
            st.session_state.sc_pending = False
            st.session_state.sc_error = None
            st.session_state.sc_drive_files = None
            st.rerun()

    with st.expander('📎 Attach Google Drive documents', expanded=False):
        _render_drive(record)

    # ── History ────────────────────────────────────────────────────────────
    for idx, msg in enumerate(record['messages']):
        text = ss.message_text(msg.get('content'))
        acts = ss.activity_lines(msg.get('content'))
        if not text and not acts:
            continue
        with st.chat_message(msg['role']):
            if msg.get('fell_back'):
                st.caption(f"⚠️ answered by {msg.get('model', '')} after a refusal")
            elif msg.get('continued_on'):
                st.caption(f"⚠️ continued by {msg.get('model', '')} after a "
                           'partial refusal')
            if msg.get('partial_refusal'):
                st.caption('✂️ stopped part-way by a safety classifier')
            for line in acts:
                st.caption(line)
            if text:
                st.markdown(text)
            if msg['role'] == 'assistant':
                _render_files(record, msg, idx)

    if st.session_state.sc_error:
        st.error(st.session_state.sc_error)
        # A failed turn is not stored, so anything it streamed exists only
        # here. Show it rather than discarding work the consultant watched
        # being written.
        if st.session_state.get('sc_partial'):
            with st.expander('Partial answer from the failed turn', expanded=True):
                st.markdown(st.session_state.sc_partial)
                st.download_button(
                    '⬇ Save partial (.md)',
                    st.session_state.sc_partial.encode('utf-8'),
                    file_name=f"{record['session_id']}_partial.md",
                    mime='text/markdown', key='sc_partial_dl',
                )

    # ── Kickoff ────────────────────────────────────────────────────────────
    if not record['messages'] and not st.session_state.sc_pending:
        st.info('Send the first message to begin. The playbook takes over from there.')
        if st.button('▶ Start the playbook', type='primary', key='sc_kickoff'):
            record['messages'].append(
                {'role': 'user', 'content': [{'type': 'text',
                                              'text': _DEFAULT_KICKOFF}]})
            st.session_state.sc_pending = True
            _save(record)
            st.rerun()

    # ── Owed assistant turn ────────────────────────────────────────────────
    if st.session_state.sc_pending:
        _run_turn(record)
        return

    if prompt := st.chat_input('Reply to Claude…'):
        record['messages'].append(
            {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]})
        st.session_state.sc_pending = True
        st.session_state.sc_error = None
        st.session_state.sc_partial = None
        _save(record)
        st.rerun()


def _render_files(record: dict, msg: dict, idx: int) -> None:
    """Download buttons for files Claude exported from its sandbox.

    The deck is the playbook's final deliverable and it only ever exists as a
    Files API `file_id` inside a tool result, so without this the consultant
    is told to download a file the page never offers. Archiving happens here
    rather than in `_run_turn` so sessions saved before this existed (Cell X)
    are filled in the first time they are opened.
    """
    if not ss.output_file_ids(msg.get('content')):
        return
    try:
        with st.spinner('Fetching the files Claude built…'):
            changed = ss.archive_message_files(
                _anthropic(), _gcs(), record['session_id'], msg)
        if changed:
            _save(record)
    except Exception as e:
        st.warning(f'Could not fetch the files from this turn: {e}')
        return

    cache = st.session_state.setdefault('sc_file_bytes', {})
    for j, f in enumerate(msg.get('files') or []):
        if not f.get('blob'):
            st.warning(f"A file from this turn could not be fetched "
                       f"(`{f.get('file_id')}`): {f.get('error', 'unknown error')}. "
                       'Reopen the session to retry.')
            continue
        data = cache.get(f['blob'])
        if data is None:
            try:
                data = cache[f['blob']] = ss.read_file_blob(_gcs(), f['blob'])
            except Exception as e:
                st.warning(f"Could not read {f['filename']} from GCS: {e}")
                continue
        st.download_button(
            f"⬇ {f['filename']} ({f.get('size', len(data)) / 1024:.0f} KB)",
            data, file_name=f['filename'], mime=f.get('mime'),
            key=f'sc_file_{idx}_{j}',
        )


def _log(msg: str) -> None:
    """stdout, which Cloud Run captures.

    The first real session appeared to freeze and left NOTHING in the logs —
    Streamlit logs no application detail, and this view logged none of its own,
    so diagnosing it needed a local reproduction of the exact request. These
    lines make the next one readable from `gcloud logging read`.
    """
    print(f'[strategy] {msg}', flush=True)


def _run_turn(record: dict) -> None:
    started = time.monotonic()
    override = st.session_state.sc_model_override
    st.session_state.sc_model_override = None      # one turn only, even on failure
    turn_model = override or record['model']
    request = ss.build_request(
        system_text  = record['system_text'],
        context_text = ss.render_context(record.get('context_blocks') or []),
        messages     = record['messages'],
        model        = turn_model,
        web_search   = bool(record.get('web_search')),
    )

    with st.chat_message('assistant'):
        act_box  = st.empty()
        text_box = st.empty()
        acts: list[str] = []
        buf: list[str] = []
        last_paint = [0.0]

        # Repaint on a timer, NOT on every delta.
        #
        # `st.empty().markdown(whole_answer)` re-sends the ENTIRE accumulated
        # text over the websocket each time it is called. Called once per token
        # that is O(n^2) traffic — by token 5,000 every keystroke re-sends
        # 5,000 tokens' worth — which saturates the connection, stops the
        # browser rendering, and kills the script run before it can save the
        # turn. Observed on the first real session: a long NIH answer streamed
        # correctly and then froze partway, leaving a record with the user
        # message, no assistant message and $0.00 cost.
        #
        # At _PAINT_HZ the cost is bounded by wall-clock instead of by answer
        # length, and the final flush below guarantees the complete text is
        # shown whatever the timing.
        def on_text(chunk: str) -> None:
            buf.append(chunk)
            now = time.monotonic()
            if now - last_paint[0] >= _PAINT_INTERVAL_S:
                last_paint[0] = now
                text_box.markdown(''.join(buf))

        def on_activity(line: str) -> None:
            acts.append(line)
            act_box.caption(' · '.join(acts[-4:]))

        _log(f"turn start session={record['session_id']} model={turn_model} "
             f"override={bool(override)} "
             f"playbook={record.get('playbook')} msgs={len(record['messages'])} "
             f"system_chars={len(record['system_text'])} web={record.get('web_search')}")
        try:
            result = ss.stream_turn(
                _anthropic(), request=request,
                on_text=on_text, on_activity=on_activity,
            )
        except Exception as e:
            _log(f"turn FAILED session={record['session_id']} "
                 f"after={time.monotonic() - started:.1f}s "
                 f"streamed={len(''.join(buf))}ch {type(e).__name__}: {e}")
            st.session_state.sc_pending = False
            st.session_state.sc_error = f'{type(e).__name__}: {e}'
            # Show whatever streamed before the failure — a partial answer is
            # worth more to the consultant than an empty bubble, and it is the
            # only copy, since a failed turn is not stored.
            if buf:
                text_box.markdown(''.join(buf))
                st.session_state.sc_partial = ''.join(buf)
            # The user turn stays in the record so Retry re-sends it rather than
            # making the consultant retype.
            st.rerun()
            return

        # Final flush: the throttle may have skipped the last deltas.
        if buf:
            text_box.markdown(''.join(buf))

    used = result.get('model_used', record['model'])
    _log(f"turn done session={record['session_id']} model={used} "
         f"stop={result['stop_reason']} fell_back={result.get('fell_back')} "
         f"partial_refusal={result.get('partial_refusal')} "
         f"blocks={len(result['content'])} "
         f"in={result['usage']['input_tokens']} "
         f"cache_r={result['usage']['cache_read_input_tokens']} "
         f"cache_w={result['usage']['cache_creation_input_tokens']} "
         f"out={result['usage']['output_tokens']} "
         f"searches={result['usage']['web_search_requests']} "
         f"cost=${result.get('cost_usd', 0):.4f} "
         f"elapsed={time.monotonic() - started:.1f}s")

    # A turn that was refused by every model comes back with ZERO content
    # blocks. Appending that as an assistant message would poison the session
    # permanently: the API rejects a message whose content is empty, so every
    # later turn in this conversation would 400 with nothing to point at. The
    # user turn is therefore left as the last message, which also makes Retry
    # the correct and available action. The cost is still counted — the
    # attempts were made.
    if result['content']:
        record['messages'].append({
            'role':       'assistant',
            'content':    result['content'],
            'model':      used,
            'fell_back':  bool(result.get('fell_back')),
            # Stored, not just shown, so the "continue on Haiku" button is
            # still offered after a reload or when a colleague opens the
            # session by ID. Metadata like `model`; build_request drops it.
            'partial_refusal': bool(result.get('partial_refusal')),
            'continued_on':    override,
            'usage':      result['usage'],
        })
    # stream_turn priced each attempt at its own model's rate; do not re-price
    # the merged usage here, or a fallback turn bills the refused model's
    # prefix at the fallback's cheaper rate.
    record['cost_usd'] = float(record.get('cost_usd', 0.0)) + float(
        result.get('cost_usd', 0.0))

    if result.get('partial_refusal'):
        st.session_state.sc_error = (
            'The answer stopped part-way: a safety classifier fired after '
            f'**{used}** had already written some of it. What you see above is '
            'real and has been saved — it is just incomplete. This fires on '
            'dried/powdered-biologic material and is a known false positive. '
            f'Use **↪ Continue on {_FALLBACK_NAME}** below to have it finish '
            'the answer — on this kind of material it is often the only model '
            'that does — or rephrase the turn yourself.'
        )
    elif result['stop_reason'] == 'refusal':
        st.session_state.sc_error = (
            f"Both **{turn_model}** and the fallback "
            f"**{ss.FALLBACK_MODEL}** declined this turn, so there is no "
            'answer to show. This is usually the client material rather than '
            'your question — dried/powdered biologics are a known false '
            'positive. Rephrasing the turn, or removing the triggering row '
            'from the attached results, normally clears it.'
        )
    elif result.get('fell_back'):
        st.session_state.sc_error = (
            f"**{record['model']}** declined this turn, so it was answered by "
            f"**{used}** instead — a weaker model. Judge the answer "
            'accordingly. Later turns go back to '
            f"**{record['model']}**."
        )
    elif result['stop_reason'] == 'max_tokens':
        st.session_state.sc_error = (
            'The answer hit the output ceiling and is cut off. Ask Claude to '
            'continue from where it stopped.'
        )

    st.session_state.sc_pending = False
    st.session_state.sc_partial = None
    _save(record)
    st.rerun()


# ── Entry ──────────────────────────────────────────────────────────────────

if st.session_state.sc_record is None:
    _render_setup()
else:
    _record = st.session_state.sc_record
    # Retry is only offered when the conversation ends on the CONSULTANT's
    # turn — i.e. the call failed before an answer was stored. Offering it
    # after a completed turn (a refusal, or a fallback answer) would re-send a
    # message list ending on an assistant message, which is a prefill and
    # returns a 400 on every model this view offers.
    _owed = (_record['messages']
             and _record['messages'][-1]['role'] == 'user'
             and not st.session_state.sc_pending)
    if st.session_state.sc_error and _owed:
        if st.button('↻ Retry the last turn', key='sc_retry'):
            st.session_state.sc_error = None
            st.session_state.sc_pending = True
            st.rerun()

    # Offered only when the conversation ENDS on a partially refused answer.
    # Once the consultant replies, the moment has passed: continuing would
    # answer a message they did not write.
    _last = _record['messages'][-1] if _record['messages'] else None
    _can_continue = (_last is not None
                     and _last['role'] == 'assistant'
                     and _last.get('partial_refusal')
                     and _last.get('model') != ss.FALLBACK_MODEL
                     and not st.session_state.sc_pending)
    if _can_continue:
        if st.button(f'↪ Continue on {_FALLBACK_NAME}', key='sc_continue_haiku',
                     help=f'Asks {_FALLBACK_NAME} to finish the answer from where '
                          'it stopped. The partial answer stays as it is; only '
                          'this one turn runs on the fallback model, so the next '
                          'turn goes back to the session model.'):
            _record['messages'].append(
                {'role': 'user', 'content': [{'type': 'text',
                                              'text': _CONTINUE_PROMPT}]})
            st.session_state.sc_model_override = ss.FALLBACK_MODEL
            st.session_state.sc_error = None
            st.session_state.sc_pending = True
            _save(_record)
            st.rerun()
    _render_chat(_record)
