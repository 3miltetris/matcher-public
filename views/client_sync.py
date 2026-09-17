"""
Client Sync
-----------
Parent page for the two sources of client material that are pulled in on a
schedule rather than uploaded: the client Google shared drive, and Fathom
call transcripts. Both feed the capability profiles in Stage 8.

Dispatches to exactly one sub-view per script run — see views/resumes.py for
why that single-branch shape is required rather than st.tabs.
"""

import importlib

import streamlit as st

_MODES = {
    '🗂️ Google Drive':    'views.drive_sync',
    '🎙️ Fathom Meetings': 'views.fathom_sync',
}

st.sidebar.divider()
mode = st.sidebar.radio('Client Sync', list(_MODES), key='csync_mode')

importlib.import_module(_MODES[mode]).render()
