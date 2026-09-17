"""
Grant Sources
-------------
Parent page for every route by which a grant opportunity enters the store.
All four write the same canonical topic schema into
data/all-topics/processed/{BROAD_AGENCY}/ — they differ only in where the
opportunity comes from:

  Import Topics    a PDF or pasted solicitation, parsed by Claude
  SAM.gov          contract opportunities, by CSV upload or the API
  Grants.gov       the public search2 API
  Funding Sources  ~280 curated sites with no API, walked by a browser agent

Dispatches to exactly one sub-view per script run — see views/resumes.py for
why that single-branch shape is required rather than st.tabs. It also keeps
the page cheap: each sub-view does eager GCS work when it loads, and only the
selected one pays for it.
"""

import importlib

import streamlit as st

_MODES = {
    '📄 Import Topics':   'views.topic_importer',
    '🏛️ SAM.gov':         'views.sam_gov_upload',
    '🏦 Grants.gov':      'views.grants_gov_fetch',
    '🛰️ Funding Sources': 'views.funding_sources',
}

st.sidebar.divider()
mode = st.sidebar.radio('Grant source', list(_MODES), key='gsrc_mode')

importlib.import_module(_MODES[mode]).render()
