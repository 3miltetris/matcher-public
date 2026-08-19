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

pages = [
    st.Page("views/topic_importer.py", title="Topic Importer", icon="📄"),
    st.Page("views/grant_search.py",   title="Grant Search",   icon="🔍"),
    st.Page("views/bulk_matching.py",  title="Bulk Matching",  icon="⚙️"),

    st.Page("views/sam_gov_upload.py",    title="SAM.gov Upload",   icon="🏛️"),
    st.Page("views/grants_gov_fetch.py", title="Grants.gov Fetch", icon="🏦"),
    st.Page("views/hubspot_import.py", title="HubSpot Import", icon="🔗"),
    st.Page("views/suggestions.py",    title="Suggestions",    icon="💡"),
    st.Page("views/contact_importer.py", title="Contact Importer", icon="👤"),
    st.Page("views/client_editor.py",    title="Client Editor",    icon="✏️"),
    st.Page("views/finance_researcher.py", title="Client Research", icon="🧪"),
    st.Page("views/client_profiler.py",  title="Client Profiles",  icon="🧩"),
    st.Page("views/aspect_match.py",     title="Bulk Aspect Match", icon="🎯"),
    st.Page("views/drive_sync.py",       title="Drive Sync",       icon="🗂️"),
    st.Page("views/resume_importer.py", title="Resume Importer",  icon="📄"),
    st.Page("views/resume_search.py",   title="Resume Search",    icon="🔎"),
    # Uncomment as pages are built:
    # st.Page("views/matcher.py",          title="Matcher",          icon="🎯"),
    # st.Page("views/match_history.py",    title="Match History",    icon="📊"),
]

# Admin-only page — hidden from the navigation for everyone else. The page
# guards itself too, so hiding it here is convenience, not the control.
if ac.is_admin():
    pages.append(st.Page("views/admin_portal.py", title="Admin Portal", icon="🛡️"))

if st.session_state.get("user_email"):
    st.sidebar.caption(
        f"Signed in as {st.session_state.user_email} · {ac.role_label()}"
    )

pg = st.navigation(pages)
pg.run()
