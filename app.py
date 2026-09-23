import subprocess
import sys

import streamlit as st

import src.modules.access_control as ac


@st.cache_resource(show_spinner=False)
def _install_playwright_browser():
    subprocess.run(
        [sys.executable, '-m', 'playwright', 'install', 'chromium'],
        capture_output=True,
    )


_install_playwright_browser()

st.set_page_config(
    page_title="The Matcher",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _iap_user_email() -> str | None:
    # Set by Google Identity-Aware Proxy on Cloud Run as
    # "accounts.google.com:<email>". Only trustworthy when the service sits
    # behind IAP with unauthenticated access blocked — IAP overwrites any
    # client-supplied value, so its presence proves a Google-authenticated,
    # access-granted user.
    try:
        header = st.context.headers.get("X-Goog-Authenticated-User-Email", "")
    except Exception:
        header = ""
    if ":" in header:
        return header.split(":", 1)[1] or None
    return None


_iap_email = _iap_user_email()
if _iap_email:
    st.session_state.authenticated = True
    st.session_state.user_email = _iap_email

if not st.session_state.get("authenticated"):
    # Local dev fallback — production auth is Google sign-in via IAP.
    st.title("The Matcher")
    pw = st.text_input("Password", type="password")
    if st.button("Enter"):
        if pw == st.secrets["app_password"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# ── Navigation ─────────────────────────────────────────────────────────────
# Grouped to mirror the pipeline: grant supply in, client demand in, match,
# export. Several entries are parent pages that dispatch to one sub-view per
# run (see views/grant_sources.py) — st.tabs cannot be used for those because
# st.stop() unwinds the whole script run and would blank the other tabs.
#
# url_path is pinned on every page whose title changed, so retitling does not
# break existing bookmarks.

pages = {
    "": [
        st.Page("views/home.py", title="Home", icon="🏠",
                url_path="home", default=True),
    ],
    "Grants": [
        st.Page("views/grant_sources.py", title="Grant Sources", icon="📥",
                url_path="grant_sources"),
        st.Page("views/grant_search.py", title="Grant Search", icon="🔍",
                url_path="grant_search"),
    ],
    "Clients & prospects": [
        st.Page("views/contact_importer.py", title="Import Contacts", icon="👤",
                url_path="contact_importer"),
        st.Page("views/client_editor.py", title="Company Records", icon="✏️",
                url_path="client_editor"),
        st.Page("views/finance_researcher.py", title="Deep Research", icon="🧪",
                url_path="finance_researcher"),
        st.Page("views/client_profiler.py", title="Capability Profiles", icon="🧩",
                url_path="client_profiler"),
        st.Page("views/client_sync.py", title="Client Sync", icon="🔄",
                url_path="client_sync"),
    ],
    "Matching": [
        st.Page("views/bulk_matching.py", title="Bulk Matching", icon="⚙️",
                url_path="bulk_matching"),
        st.Page("views/aspect_match.py", title="Aspect Match", icon="🎯",
                url_path="aspect_match"),
    ],
    "Talent": [
        st.Page("views/resumes.py", title="Resumes", icon="📇",
                url_path="resumes"),
    ],
    "Export & admin": [
        st.Page("views/hubspot_import.py", title="HubSpot Import", icon="🔗",
                url_path="hubspot_import"),
        st.Page("views/suggestions.py", title="Suggestions", icon="💡",
                url_path="suggestions"),
    ],
}

# Admin-only page — hidden from the navigation for everyone else. The page
# guards itself too, so hiding it here is convenience, not the control.
if ac.is_admin():
    pages["Export & admin"].append(
        st.Page("views/admin_portal.py", title="Admin Portal", icon="🛡️",
                url_path="admin_portal")
    )

if st.session_state.get("user_email"):
    st.sidebar.caption(
        f"Signed in as {st.session_state.user_email} · {ac.role_label()}"
    )

pg = st.navigation(pages)
pg.run()
