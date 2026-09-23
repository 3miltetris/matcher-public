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
import src.modules.pools as pl
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
    ('Clients & prospects', [
        ('👤', 'views/contact_importer.py', 'Import Contacts',
         'Bring companies in from a spreadsheet or a HubSpot list — as leads, '
         'or straight into the prospect pool — and profile them.'),
        ('✏️', 'views/client_editor.py', 'Company Records',
         'Edit a client or prospect summary, re-embed it, or promote a '
         'prospect that signed.'),
        ('🧪', 'views/finance_researcher.py', 'Deep Research',
         'Run OpenAI deep research on a client or prospect — financials, or '
         'technology and R&D.'),
        ('🧩', 'views/client_profiler.py', 'Capability Profiles',
         'Split each client or prospect into independently searchable '
         'capabilities and the markets they serve.'),
        ('🔄', 'views/client_sync.py', 'Client Sync',
         'Pull client material in from Google Drive and Fathom call '
         'transcripts.'),
    ]),
    ('Matching', [
        ('⚙️', 'views/bulk_matching.py', 'Bulk Matching',
         'Score every contact source against every selected topic, '
         'whole-company.'),
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

    # Both pools, counted separately — a prospect is not a client and the two
    # numbers are read for different reasons.
    for pool_key in pl.POOL_KEYS:
        try:
            rows = uc.load_parquets_from_prefix(gcs, pl.contacts_prefix(pool_key))
            stats[pool_key] = 0 if rows.empty else int(
                rows.apply(ap.company_key, axis=1).nunique()
            )
        except Exception as e:
            stats[pool_key] = None
            stats[f'{pool_key}_err'] = str(e)

        try:
            profiles = ap.load_profiles(gcs, pool=pool_key)
            stats[f'{pool_key}_profiles'] = 0 if profiles.empty else int(len(profiles))
        except Exception as e:
            stats[f'{pool_key}_profiles'] = None
            stats[f'{pool_key}_profiles_err'] = str(e)

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
        cols = st.columns(len(pl.POOL_KEYS) * 2 + 1)
        for i, pool_key in enumerate(pl.POOL_KEYS):
            companies = stats.get(pool_key)
            profiles  = stats.get(f'{pool_key}_profiles')
            cols[i * 2].metric(
                f'{pl.label(pool_key)}',
                '—' if companies is None else f'{companies:,}',
            )
            cols[i * 2 + 1].metric(
                f'{pl.label(pool_key)} profiles',
                '—' if profiles is None else f'{profiles:,}',
            )
        agencies = stats['agencies']
        cols[-1].metric('Grant agency folders',
                        '—' if agencies is None else f'{len(agencies):,}')

        if agencies:
            st.caption('Agency folders: ' + ' · '.join(agencies))

        for key in [f'{k}_err' for k in pl.POOL_KEYS] + \
                   [f'{k}_profiles_err' for k in pl.POOL_KEYS] + ['agencies_err']:
            if stats.get(key):
                st.warning(f'{key.replace("_err", "")}: {stats[key]}')
