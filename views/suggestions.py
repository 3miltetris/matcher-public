"""
Feature Suggestions
-------------------
Submit feature requests and upvote suggestions from the team.
Suggestions are stored as JSON blobs in GCS under suggestions/.
"""

import json
import uuid
from datetime import datetime

import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account

# ── GCS ────────────────────────────────────────────────────────────────────

_BUCKET = 'cc-matcher-bucket-jeg-v1'
_PREFIX = 'suggestions/'


def _get_storage_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)


def _load_suggestions(client: storage.Client) -> list[dict]:
    blobs = client.list_blobs(_BUCKET, prefix=_PREFIX)
    suggestions = []
    for blob in blobs:
        if blob.name.endswith('.json'):
            try:
                data = json.loads(blob.download_as_text())
                suggestions.append(data)
            except Exception:
                pass
    return sorted(suggestions, key=lambda s: s.get('votes', 0), reverse=True)


def _save_suggestion(client: storage.Client, suggestion: dict) -> None:
    blob = client.bucket(_BUCKET).blob(f"{_PREFIX}{suggestion['id']}.json")
    blob.upload_from_string(json.dumps(suggestion), content_type='application/json')


def _upvote(client: storage.Client, suggestion_id: str) -> int:
    blob = client.bucket(_BUCKET).blob(f"{_PREFIX}{suggestion_id}.json")
    data = json.loads(blob.download_as_text())
    data['votes'] = data.get('votes', 0) + 1
    blob.upload_from_string(json.dumps(data), content_type='application/json')
    return data['votes']


# ── Session state ──────────────────────────────────────────────────────────

if 'sug_voted' not in st.session_state:
    st.session_state.sug_voted = set()   # IDs voted on this session
if 'sug_list' not in st.session_state:
    st.session_state.sug_list = None     # cached list; None = needs refresh
if 'sug_submitted' not in st.session_state:
    st.session_state.sug_submitted = False


# ── Page ───────────────────────────────────────────────────────────────────

st.title('💡 Feature Suggestions')
st.caption('Have an idea? Submit it below and upvote the features you want to see built.')

# ── Submit form ────────────────────────────────────────────────────────────

with st.expander('Submit a new suggestion', expanded=st.session_state.sug_submitted is False):
    with st.form('suggestion_form', clear_on_submit=True):
        name = st.text_input('Your name', placeholder='e.g. Alex')
        suggestion = st.text_area(
            'Suggestion',
            height=100,
            placeholder='Describe the feature you'd like to see…',
        )
        submitted = st.form_submit_button('Submit', type='primary')

    if submitted:
        name = name.strip()
        suggestion = suggestion.strip()
        if not name or not suggestion:
            st.warning('Please fill in both fields before submitting.')
        else:
            new = {
                'id': str(uuid.uuid4()),
                'name': name,
                'suggestion': suggestion,
                'votes': 0,
                'created_at': datetime.utcnow().isoformat(),
            }
            try:
                client = _get_storage_client()
                _save_suggestion(client, new)
                st.session_state.sug_list = None   # invalidate cache
                st.session_state.sug_submitted = True
                st.success('Suggestion submitted — thanks!')
            except Exception as e:
                st.error(f'Failed to save suggestion: {e}')

# ── Load suggestions ────────────────────────────────────────────────────────

refresh_col, _ = st.columns([1, 5])
if refresh_col.button('Refresh', icon='🔄') or st.session_state.sug_list is None:
    with st.spinner('Loading suggestions…'):
        try:
            client = _get_storage_client()
            st.session_state.sug_list = _load_suggestions(client)
        except Exception as e:
            st.error(f'Failed to load suggestions: {e}')
            st.stop()

suggestions = st.session_state.sug_list

if not suggestions:
    st.info('No suggestions yet — be the first!')
    st.stop()

# ── Render suggestions ──────────────────────────────────────────────────────

st.subheader(f'{len(suggestions)} suggestion{"s" if len(suggestions) != 1 else ""}')

for sug in suggestions:
    sid = sug['id']
    already_voted = sid in st.session_state.sug_voted

    with st.container(border=True):
        left, right = st.columns([6, 1])
        with left:
            st.markdown(f"**{sug['suggestion']}**")
            submitted_by = sug.get('name', 'Anonymous')
            created = sug.get('created_at', '')[:10]   # YYYY-MM-DD
            st.caption(f"Submitted by {submitted_by}" + (f" · {created}" if created else ''))
        with right:
            vote_label = f"{'✓ ' if already_voted else ''}▲  {sug.get('votes', 0)}"
            if st.button(
                vote_label,
                key=f'vote_{sid}',
                disabled=already_voted,
                use_container_width=True,
                help='You already voted on this' if already_voted else 'Upvote this suggestion',
            ):
                try:
                    client = _get_storage_client()
                    new_count = _upvote(client, sid)
                    st.session_state.sug_voted.add(sid)
                    # Update in-memory list so the count reflects immediately
                    for s in st.session_state.sug_list:
                        if s['id'] == sid:
                            s['votes'] = new_count
                    # Re-sort by votes
                    st.session_state.sug_list.sort(key=lambda s: s.get('votes', 0), reverse=True)
                    st.rerun()
                except Exception as e:
                    st.error(f'Failed to record vote: {e}')
