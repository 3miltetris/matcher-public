"""
Admin Portal
------------
Manages who may run the destructive actions in the app — deleting clients
from data/all-contacts/clients/ and deleting multi-aspect client profiles —
and which domains / addresses may open the public DD intake form
(admin-config/intake_allowlist.json, read by intake-web).

Visible to admins only (and hidden from the navigation for everyone else);
the admin list itself can only be changed by a super admin, which is a code
constant in src/modules/access_control.py.
"""

import re

import pandas as pd
import streamlit as st

import src.modules.access_control as ac
from src.modules.intake import allowlist as al
from src.modules.ui_common import get_storage_client

_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')

# ── Page ───────────────────────────────────────────────────────────────────

st.title('🛡️ Admin Portal')
ac.require_admin('The Admin Portal')

me       = ac.current_user_email()
is_super = ac.is_super_admin()

st.caption(
    'Admins can delete clients and client profiles, and decide who may open the '
    'DD intake form. Everything else in the app is open to the whole team.'
)

head = st.columns([2, 2, 1])
head[0].markdown(f"**Signed in as:** {me or 'local dev (no IAP header)'}")
head[1].markdown(f'**Role:** {ac.role_label()}')
with head[2]:
    if st.button('↺ Refresh', help='Re-read the admin list from GCS'):
        ac.load_admins(refresh=True)
        st.rerun()

if st.session_state.get('ap_flash'):
    # Set before an st.rerun() — a success message written just before a rerun
    # is discarded with the rest of the page.
    st.success(st.session_state.pop('ap_flash'))

doc = ac.load_admins()
if doc.get('error'):
    st.error(
        f'Could not read `{ac.ADMINS_BLOB}` ({doc["error"]}) — only the super '
        'admin(s) have access until this is resolved.'
    )

# ── DD intake access ───────────────────────────────────────────────────────
# Rendered above the admin sections because those end in st.stop() for
# non-super admins, and every admin may edit this list.

st.divider()
st.subheader('DD intake access')
st.caption(
    'Who can open the public due-diligence form. A founder enters their email '
    'and is sent a sign-in link only if their **company domain** or their '
    '**exact address** is listed here. Subdomains count (`acme.com` covers '
    '`jo@eng.acme.com`). Public email providers such as gmail.com cannot be '
    'approved as a domain — add the founder\'s exact address instead. Changes '
    'take effect within a minute, and removing an entry signs that founder out.'
)

_gcs = get_storage_client()
try:
    allow = al.load(_gcs)
except Exception as e:
    allow = None
    st.error(f'Could not read `{al.BLOB}` ({e}) — the intake form approves nobody until this is resolved.')

if allow is not None:
    st.caption(
        f'{len(allow["domains"])} domain(s) · {len(allow["emails"])} address(es) · '
        f'last changed {allow.get("updated_at") or "never"}'
        + (f' by {allow["updated_by"]}' if allow.get('updated_by') else '')
    )
    if not allow['domains'] and not allow['emails']:
        st.warning('The list is empty, so nobody can open the intake form.')

    with st.form('ap_intake_allow'):
        c1, c2 = st.columns(2)
        dom_text = c1.text_area(
            'Approved domains — one per line', value='\n'.join(allow['domains']),
            height=220, key='ap_intake_domains', placeholder='acme.com',
        )
        eml_text = c2.text_area(
            'Approved email addresses — one per line', value='\n'.join(allow['emails']),
            height=220, key='ap_intake_emails', placeholder='founder@gmail.com',
        )
        note = st.text_input('Note (optional)', key='ap_intake_note',
                             placeholder='e.g. added Acme ahead of kickoff call')
        save_clicked = st.form_submit_button('💾 Save intake access', type='primary')

    if save_clicked:
        domains = [l for l in dom_text.splitlines() if l.strip()]
        emails  = [l for l in eml_text.splitlines() if l.strip()]
        new_d, new_e, problems = al.check_entries(domains, emails)
        if problems:
            st.error('Nothing was saved:\n\n' + '\n'.join(f'- {p}' for p in problems))
        else:
            added   = sorted((set(new_d) | set(new_e)) - set(allow['domains']) - set(allow['emails']))
            removed = sorted((set(allow['domains']) | set(allow['emails'])) - set(new_d) - set(new_e))
            auto = '; '.join(filter(None, [
                f'added {", ".join(added)}' if added else '',
                f'removed {", ".join(removed)}' if removed else '',
            ])) or 'no change'
            try:
                al.save(_gcs, new_d, new_e, actor=me or 'local-dev',
                        note=f'{note.strip()} ({auto})' if note.strip() else auto)
                st.session_state.ap_flash = f'Intake access saved — {auto}.'
                st.rerun()
            except Exception as e:
                st.error(f'Could not save the intake allowlist: {e}')

    ahist = allow.get('history') or []
    if ahist:
        with st.expander(f'Intake access history ({len(ahist)})'):
            st.dataframe(pd.DataFrame(ahist[::-1]).reindex(columns=['at', 'by', 'note']),
                         hide_index=True, use_container_width=True)

# ── Super admins ───────────────────────────────────────────────────────────

st.divider()
st.subheader('Super admins')
st.caption(
    'Set in code (`SUPER_ADMINS` in `src/modules/access_control.py`). Cannot be '
    'added or removed here, and are the only role allowed to change the admin '
    'list below.'
)
st.dataframe(
    pd.DataFrame({'email': list(ac.SUPER_ADMINS)}),
    hide_index=True, use_container_width=True,
)

# ── Admins ─────────────────────────────────────────────────────────────────

st.divider()
st.subheader('Admins')

admins = list(doc['admins'])
st.caption(
    f'{len(admins)} admin{"s" if len(admins) != 1 else ""} · '
    f'last changed {doc.get("updated_at") or "never"}'
    + (f' by {doc["updated_by"]}' if doc.get('updated_by') else '')
)

if not is_super:
    st.info('Only a super admin can add or remove admins.')
    if admins:
        st.dataframe(pd.DataFrame({'email': admins}), hide_index=True,
                     use_container_width=True)
    else:
        st.caption('No additional admins — super admins only.')
    st.stop()

with st.form('ap_add_admin', clear_on_submit=True):
    new_email = st.text_input(
        'Add an admin', placeholder='name@bwcoconsulting.com',
        help='Must be the Google account email they sign in to the app with.',
    )
    add_clicked = st.form_submit_button('➕ Add admin', type='primary')

if add_clicked and new_email.strip():
    candidate = ac.norm_email(new_email)
    if not _EMAIL_RE.match(candidate):
        st.error(f'`{candidate}` is not a valid email address.')
    elif ac.is_super_admin(candidate):
        st.info(f'{candidate} is already a super admin.')
    elif candidate in admins:
        st.info(f'{candidate} is already an admin.')
    else:
        try:
            ac.save_admins([*admins, candidate], actor=me or 'local-dev',
                           note=f'added {candidate}')
            st.session_state.ap_flash = (
                f'{candidate} is now an admin — they will see admin controls '
                f'after reloading the app.'
            )
            st.rerun()
        except Exception as e:
            st.error(f'Could not save the admin list: {e}')

if not admins:
    st.caption('No additional admins yet — super admins only.')
else:
    st.markdown('**Current admins**')
    for email in admins:
        row = st.columns([4, 1])
        row[0].markdown(f'`{email}`')
        if row[1].button('🗑 Remove', key=f'ap_rm_{email}'):
            try:
                ac.save_admins([e for e in admins if e != email],
                               actor=me or 'local-dev', note=f'removed {email}')
                st.session_state.ap_flash = f'{email} is no longer an admin.'
                st.rerun()
            except Exception as e:
                st.error(f'Could not save the admin list: {e}')

# ── Change history ─────────────────────────────────────────────────────────

history = doc.get('history') or []
if history:
    with st.expander(f'Change history ({len(history)})'):
        hist_df = pd.DataFrame(history[::-1])
        for col in ('at', 'by', 'note'):
            if col not in hist_df.columns:
                hist_df[col] = ''
        st.dataframe(hist_df[['at', 'by', 'note']], hide_index=True,
                     use_container_width=True)
