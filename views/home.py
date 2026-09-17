"""
Home
----
Landing page: what the pipeline does, in what order, and a link into each
step. The app has grown past the point where a flat page list explains
itself, so this page is the map.

Deliberately cheap to render — several views already do eager GCS work when
they load, and Home must not add another unconditional round-trip. The
"At a glance" counts are behind a button and cached in session state (the
repo caches by hand everywhere; there is no st.cache_data in views/).
"""

import streamlit as st

import src.modules.aspect_profile as ap
import src.modules.ui_common as uc

# ── Pipeline map ───────────────────────────────────────────────────────────
# (section, icon, page path, title, one-line description)

_STEPS = [
    ('Grants', [
        ('📥', 'views/grant_sources.py', 'Grant Sources',
         'Bring opportunities in — upload solicitations, pull SAM.gov and '
         'Grants.gov, or let the agent walk your watched funding sites.'),
        ('🔍', 'views/grant_search.py', 'Grant Search',
         'Ask which topics match a technology description, ad hoc.'),
    ]),
    ('Clients', [
        ('👤', 'views/contact_importer.py', 'Import Contacts',
         'Bring companies in from a spreadsheet or a HubSpot list, and '
         'profile them.'),
        ('✏️', 'views/client_editor.py', 'Client Records',
         'Edit a client summary and re-embed it.'),
        ('🧪', 'views/finance_researcher.py', 'Deep Research',
         'Run OpenAI deep research on a client — financials, or technology '
         'and R&D.'),
        ('🧩', 'views/client_profiler.py', 'Capability Profiles',
         'Split each client into independently searchable capabilities and '
         'the markets they serve.'),
        ('🔄', 'views/client_sync.py', 'Client Sync',
         'Pull client material in from Google Drive and Fathom call '
         'transcripts.'),
    ]),
    ('Matching', [
        ('⚙️', 'views/bulk_matching.py', 'Bulk Matching',
         'Score every client against every selected topic, whole-company.'),
        ('🎯', 'views/aspect_match.py', 'Aspect Match',
         'Score per capability or per market — finds what the blended '
         'company embedding averages away.'),
    ]),
    ('Talent', [
        ('📇', 'views/resumes.py', 'Resumes',
         'Import resumes from HubSpot and search them by expertise.'),
    ]),
    ('Export', [
        ('🔗', 'views/hubspot_import.py', 'HubSpot Import',
         'Push match results, research findings or capability profiles back '
         'into the CRM.'),
        ('💡', 'views/suggestions.py', 'Suggestions',
         'Ask for a feature, or upvote one.'),
    ]),
]


# ── At-a-glance counts ─────────────────────────────────────────────────────

def _gather_stats() -> dict:
    """One pass over GCS. Every lookup is independently guarded — a single
    missing prefix must degrade to a dash, not blank the whole panel."""
    gcs   = uc.get_storage_client()
    stats: dict[str, object] = {}

    try:
        clients = uc.load_parquets_from_prefix(gcs, uc.CLIENTS_PREFIX)
        if clients.empty:
            stats['clients'] = 0
        else:
            stats['clients'] = int(
                clients.apply(ap.company_key, axis=1).nunique()
            )
    except Exception as e:
        stats['clients'] = None
        stats['clients_err'] = str(e)

    try:
        profiles = ap.load_profiles(gcs)
        stats['profiles'] = 0 if profiles.empty else int(len(profiles))
    except Exception as e:
        stats['profiles'] = None
        stats['profiles_err'] = str(e)

    try:
        stats['agencies'] = uc.list_prefixes(gcs, uc.TOPICS_PREFIX)
    except Exception as e:
        stats['agencies'] = None
        stats['agencies_err'] = str(e)

    return stats


# ── Page ───────────────────────────────────────────────────────────────────

st.title('🏠 The Matcher')
st.caption(
    'Matches client companies to federal funding opportunities: opportunities '
    'and companies come in on the left, get embedded, matched and re-ranked, '
    'and go out to HubSpot.'
)

st.divider()

for section, entries in _STEPS:
    st.subheader(section)
    for icon, path, title, blurb in entries:
        link, desc = st.columns([1, 3])
        with link:
            st.page_link(path, label=title, icon=icon)
        with desc:
            st.caption(blurb)
    st.write('')

st.divider()

with st.expander('📊 At a glance', expanded=False):
    st.caption(
        'Reads several parquet stores from GCS — kept behind this button so '
        'opening Home stays instant.'
    )
    if st.button('Refresh counts', key='home_refresh'):
        with st.spinner('Reading GCS…'):
            st.session_state.home_stats = _gather_stats()

    stats = st.session_state.get('home_stats')
    if stats is None:
        st.info('Not loaded yet.')
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric('Client companies',
                  '—' if stats['clients'] is None else f"{stats['clients']:,}")
        c2.metric('Capability profiles',
                  '—' if stats['profiles'] is None else f"{stats['profiles']:,}")
        agencies = stats['agencies']
        c3.metric('Grant agency folders',
                  '—' if agencies is None else f'{len(agencies):,}')

        if agencies:
            st.caption('Agency folders: ' + ' · '.join(agencies))

        for key in ('clients_err', 'profiles_err', 'agencies_err'):
            if stats.get(key):
                st.warning(f'{key.replace("_err", "")}: {stats[key]}')
