import streamlit as st

st.set_page_config(
    page_title="The Matcher",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not st.session_state.get("authenticated"):
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
    # Uncomment as pages are built:
    # st.Page("views/contact_importer.py", title="Contact Importer", icon="👤"),
    # st.Page("views/matcher.py",          title="Matcher",          icon="🎯"),
    # st.Page("views/match_history.py",    title="Match History",    icon="📊"),
]

pg = st.navigation(pages)
pg.run()
