"""
Resumes
-------
Parent page for the two resume steps. Import brings resumes in from HubSpot
and embeds them; Search queries those embeddings.

Dispatches to exactly one sub-view per script run. That single-branch shape
is load-bearing, not cosmetic: both sub-views use st.stop() to guard their
flow, and st.stop() unwinds the entire script run — so rendering both in one
pass (as st.tabs would) would silently blank whichever came second.
"""

import importlib

import streamlit as st

_MODES = {
    '📄 Import': 'views.resume_importer',
    '🔎 Search': 'views.resume_search',
}

st.sidebar.divider()
mode = st.sidebar.radio('Resumes', list(_MODES), key='res_mode')

importlib.import_module(_MODES[mode]).render()
