"""
Newsletter
----------
The day's new grant topics that are worth a slot in the vertical newsletter.

Every ingest path (Topic Importer, SAM.gov, Grants.gov, the Funding Source
agent) tags a topic as it is stored — which newsletter verticals it earns, a
one-line hook, and the official title of the solicitation it came from — using
src/modules/newsletter.py. This page is where a consultant reviews that call:

  1. Load the topics scraped in a date window (today by default)
  2. Check any that were never tagged — topics stored before this existed, or
     whose classification failed at ingest
  3. Review: flip an item in or out, fix a vertical, a title or a hook, and
     save the review back onto the stored rows
  4. Export the draft, grouped by vertical, as Markdown or CSV

The test the tagger applies is "would someone in this field be glad they read
it", not "is it a fit" — and it would rather leave a vertical empty than fill
it with a weak item. Expect most of a SAM.gov day to be skipped.

Loads are behind a button and kept in session state (no st.cache_data — see
CLAUDE.md). Only parquets created on or after the window's start are read,
then rows are filtered on `scraped_at`; an overwritten blob gets a new creation
time, so that filter can include extra files but never miss one.
"""

import io
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

import src.modules.newsletter as nl
import src.modules.ui_common as uc

_BUCKET = uc.BUCKET
_PREFIX = uc.TOPICS_PREFIX

_SHOW_COLS = [
    'title', 'agency', 'broad_agency', 'topic_number', 'due_date', 'close_date',
    'funding_amount', 'award_ceiling', 'source', 'scraped_at', 'grant_summary',
    *nl.COLUMNS, *nl.REVIEW_COLUMNS,
]
_EDITABLE = [nl.COL_GOOD, nl.COL_VERTICALS, nl.COL_TITLE, nl.COL_HOOK]
_EST_COST_PER_TOPIC = 0.015     # Opus 5.5, medium effort, cached system prompt
_TITLE_IS_OFFICIAL  = {'SAM-GOV', 'GRANTS-GOV'}


def _s(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ''
    s = str(v).strip()
    return '' if s.lower() in ('nan', 'none', 'nat') else s


# ── Load ───────────────────────────────────────────────────────────────────

def _read_blob(gcs, name: str) -> pd.DataFrame:
    df = pd.read_parquet(io.BytesIO(gcs.bucket(_BUCKET).blob(name).download_as_bytes()))
    df = df.drop(columns=[c for c in ('embeddings',) if c in df.columns])
    df['_blob'] = name
    df['_row'] = range(len(df))
    # Same convention as every other reader: the folder is the broad agency.
    df['broad_agency'] = name[len(_PREFIX):].split('/', 1)[0]
    return df


def _load(start: date, end: date, agencies: list[str]) -> pd.DataFrame:
    gcs = uc.get_storage_client()
    start_dt = datetime.combine(start, datetime.min.time())
    names = []
    for agency in agencies:
        for b in gcs.list_blobs(_BUCKET, prefix=f'{_PREFIX}{agency}/'):
            if not b.name.endswith('.parquet'):
                continue
            created = b.time_created.replace(tzinfo=None) if b.time_created else None
            if created is None or created >= start_dt - timedelta(days=1):
                names.append(b.name)
    if not names:
        return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=16) as ex:
        frames = list(ex.map(lambda n: _read_blob(gcs, n), names))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    scraped = df.get('scraped_at', pd.Series([''] * len(df))).astype(str).str[:10]
    lo, hi = start.isoformat(), end.isoformat()
    df = df[(scraped >= lo) & (scraped <= hi)].reset_index(drop=True)
    if 'sam_status' in df.columns:
        df = df[df['sam_status'].fillna('active').astype(str) != 'archived'].reset_index(drop=True)

    for col in nl.COLUMNS + nl.REVIEW_COLUMNS:
        if col not in df.columns:
            df[col] = False if col == nl.COL_GOOD else ''
    df[nl.COL_GOOD] = df[nl.COL_GOOD].fillna(False).astype(bool)
    for col in (nl.COL_VERTICALS, nl.COL_TITLE, nl.COL_HOOK, nl.COL_REASON,
                nl.COL_MODEL, nl.COL_CHECKED, *nl.REVIEW_COLUMNS):
        df[col] = df[col].map(_s)
    return df


def _is_checked(df: pd.DataFrame) -> pd.Series:
    return df[nl.COL_CHECKED].astype(str).str.strip() != ''


def _deadline(row) -> str:
    return _s(row.get('due_date')) or _s(row.get('close_date'))


def _headline(row) -> str:
    """Solicitation title first — it is what a reader searches for — then the
    topic title when it adds something."""
    sol = nl.clean_title(row.get(nl.COL_TITLE))
    title = nl.clean_title(row.get('title'))
    if sol and title and sol.lower() != title.lower() and title.lower() not in sol.lower():
        return f'{sol} — {title}'
    return sol or title


# ── Tag unchecked rows ─────────────────────────────────────────────────────

def _tag(df: pd.DataFrame, mask: pd.Series) -> tuple[int, list[str]]:
    """Classify the masked rows, write them back to their parquets, and update
    the session frame in place."""
    idx = df.index[mask].tolist()
    subset = df.loc[idx].copy()
    # A title already on the row (from ingest or an earlier review) wins; for
    # the API stores the notice title IS the published solicitation title.
    hints = [
        _s(r[nl.COL_TITLE]) or (_s(r['title']) if r['broad_agency'] in _TITLE_IS_OFFICIAL else '')
        for _, r in subset.iterrows()
    ]
    bar = st.progress(0, text=f'Checking 0/{len(idx)}…')
    nl.tag_frame(subset, st.secrets['anthropic_api_key'], title_hints=hints,
                 progress=lambda d, t: bar.progress(d / t, text=f'Checking {d}/{t}…'))
    bar.empty()

    updates: dict = {}
    for i in idx:
        row = subset.loc[i]
        values = {c: row[c] for c in nl.COLUMNS}
        # A re-check discards an earlier human review of the same row; say so
        # by clearing it rather than leaving a stale "reviewed by".
        values.update({nl.COL_REVIEWED_BY: '', nl.COL_REVIEWED_AT: ''})
        updates.setdefault(row['_blob'], []).append((int(row['_row']), str(row['title']), values))
        for c, v in values.items():
            df.at[i, c] = v
    return nl.write_back(uc.get_storage_client(), _BUCKET, updates)


# ── Draft ──────────────────────────────────────────────────────────────────

def _draft_markdown(df: pd.DataFrame, start: date, end: date) -> str:
    window = start.isoformat() if start == end else f'{start.isoformat()} – {end.isoformat()}'
    out = [f'# Funding opportunities — {window}', '']
    placed = 0
    for vertical in nl.VERTICALS:
        # Each item appears once, under its first (most relevant) vertical —
        # a mass email that repeats an item in three sections reads as padding.
        items = [r for _, r in df.iterrows()
                 if nl.parse_verticals(r[nl.COL_VERTICALS])[0][:1] == [vertical]]
        if not items:
            continue
        out += [f'## {vertical}', '']
        for r in items:
            funding = _s(r.get('funding_amount')) or _s(r.get('award_ceiling'))
            if len(funding) > 80:
                funding = funding[:80].rsplit(' ', 1)[0] + '…'
            meta = [m for m in (_s(r.get('agency')), _deadline(r) and f'Due {_deadline(r)}',
                                funding)
                    if m]
            out.append(f'**{_headline(r)}**')
            if meta:
                out.append('_' + ' · '.join(meta) + '_')
            if r[nl.COL_HOOK]:
                out.append(r[nl.COL_HOOK])
            also = nl.parse_verticals(r[nl.COL_VERTICALS])[0][1:]
            if also:
                out.append(f'_Also relevant to: {", ".join(also)}_')
            if _s(r.get('source')).startswith('http'):
                out.append(f'[Read the solicitation]({_s(r.get("source"))})')
            out.append('')
            placed += 1
    if not placed:
        out.append('_Nothing in this window cleared the bar._')
    return '\n'.join(out)


# ── Page ───────────────────────────────────────────────────────────────────

for _k, _v in (('nl_df', None), ('nl_window', None), ('nl_msgs', [])):
    if _k not in st.session_state:
        st.session_state[_k] = _v

st.title('📰 Newsletter')
st.caption(
    'New grant topics worth a newsletter slot, by vertical. Every imported topic is '
    'screened on the way in with one question: would someone working in this field be '
    'glad they read about it? The bar is high — an empty vertical beats a weak item.'
)

for m in st.session_state.nl_msgs:
    st.info(m)
st.session_state.nl_msgs = []

# ── 1 · Window ─────────────────────────────────────────────────────────────

st.subheader('1 · Topics scraped')
gcs = uc.get_storage_client()
all_agencies = uc.list_prefixes(gcs, _PREFIX)

c1, c2 = st.columns([1, 2])
with c1:
    window = st.date_input('Scraped between', value=(date.today(), date.today()),
                           key='nl_dates', format='YYYY-MM-DD')
with c2:
    agencies = st.multiselect('Agency folders', all_agencies, default=all_agencies,
                              key='nl_agencies')

if isinstance(window, (tuple, list)):
    start, end = (window[0], window[-1]) if window else (date.today(), date.today())
else:
    start = end = window

if st.button('🔄 Load topics', type='primary', disabled=not agencies):
    with st.spinner('Reading topic parquets…'):
        try:
            st.session_state.nl_df = _load(start, end, agencies)
            st.session_state.nl_window = (start, end)
        except Exception as e:
            st.error(f'Load failed: {e}')
            st.stop()

df: pd.DataFrame | None = st.session_state.nl_df
if df is None:
    st.info('Pick a window and press **Load topics**.')
    st.stop()
if st.session_state.nl_window != (start, end):
    st.warning('The window changed since the last load — press **Load topics** to refresh.')
if df.empty:
    st.info('No topics were scraped in this window.')
    st.stop()

checked = _is_checked(df)
m1, m2, m3, m4 = st.columns(4)
m1.metric('Topics scraped', len(df))
m2.metric('Checked', int(checked.sum()))
m3.metric('📰 Newsletter-worthy', int((checked & df[nl.COL_GOOD]).sum()))
m4.metric('Unchecked', int((~checked).sum()))

# ── 2 · Check unchecked ────────────────────────────────────────────────────

n_unchecked = int((~checked).sum())
if n_unchecked:
    st.subheader('2 · Check unchecked topics')
    st.caption(
        f'**{n_unchecked}** topic(s) have no newsletter call — stored before screening '
        f'existed, or the check failed at import. Checking them costs about '
        f'**${n_unchecked * _EST_COST_PER_TOPIC:,.2f}** and writes the result onto the '
        'stored rows.'
    )
    failed = df.loc[~checked, nl.COL_REASON]
    failed = failed[failed.str.startswith('Not checked')]
    if not failed.empty:
        with st.expander(f'{len(failed)} failed at import — reasons'):
            for r in failed.unique()[:20]:
                st.code(r)
    if st.button(f'📰 Check {n_unchecked} topic(s)', key='nl_check_btn'):
        written, errors = _tag(df, ~checked)
        st.session_state.nl_df = df
        st.session_state.nl_msgs = [f'Checked and saved **{written}** topic(s).'] + errors[:10]
        st.rerun()

with st.expander('🔁 Re-check every loaded topic'):
    st.caption(
        f'Re-runs the screening on all {len(df)} topics in this window '
        f'(≈ ${len(df) * _EST_COST_PER_TOPIC:,.2f}) and **overwrites** any review saved '
        'below. Use it after the screening prompt changes.'
    )
    if st.checkbox('I understand this discards saved reviews', key='nl_recheck_ok'):
        if st.button('Re-check all', key='nl_recheck_btn'):
            written, errors = _tag(df, pd.Series(True, index=df.index))
            st.session_state.nl_df = df
            st.session_state.nl_msgs = [f'Re-checked and saved **{written}** topic(s).'] + errors[:10]
            st.rerun()

# ── 3 · Review ─────────────────────────────────────────────────────────────

st.subheader('3 · Review')

f1, f2, f3 = st.columns([1, 2, 1])
with f1:
    show = st.radio('Show', ['Newsletter-worthy', 'Skipped', 'All'], horizontal=False,
                    key='nl_show')
with f2:
    vfilter = st.multiselect('Verticals', nl.VERTICALS, key='nl_vfilter',
                             placeholder='All verticals')
with f3:
    hide_reviewed = st.checkbox('Hide reviewed', key='nl_hide_reviewed')

view = df[checked]
if show == 'Newsletter-worthy':
    view = view[view[nl.COL_GOOD]]
elif show == 'Skipped':
    view = view[~view[nl.COL_GOOD]]
if vfilter:
    view = view[view[nl.COL_VERTICALS].map(
        lambda s: bool(set(nl.parse_verticals(s)[0]) & set(vfilter)))]
if hide_reviewed:
    view = view[view[nl.COL_REVIEWED_AT] == '']

if view.empty:
    st.info('Nothing matches these filters.')
else:
    table = pd.DataFrame({
        nl.COL_GOOD:      view[nl.COL_GOOD],
        nl.COL_VERTICALS: view[nl.COL_VERTICALS],
        nl.COL_TITLE:     view[nl.COL_TITLE],
        'title':          view['title'].map(nl.clean_title),
        'agency':         view['agency'].map(_s) if 'agency' in view else '',
        'deadline':       view.apply(_deadline, axis=1),
        nl.COL_HOOK:      view[nl.COL_HOOK],
        nl.COL_REASON:    view[nl.COL_REASON],
        'source':         view['source'].map(_s) if 'source' in view else '',
        'folder':         view['broad_agency'],
        'reviewed':       view[nl.COL_REVIEWED_BY],
    }, index=view.index)

    st.caption(
        f'{len(table)} topic(s). Tick or untick **Include**, correct the verticals '
        f'(names separated by `|`, most relevant first — valid: {", ".join(nl.VERTICALS)}), '
        'the solicitation title or the hook, then save.'
    )
    edited = st.data_editor(
        table,
        key=f'nl_editor_{show}_{"-".join(vfilter)}_{hide_reviewed}',
        hide_index=True,
        width='stretch',
        disabled=[c for c in table.columns if c not in _EDITABLE],
        column_config={
            nl.COL_GOOD:      st.column_config.CheckboxColumn('Include', width='small'),
            nl.COL_VERTICALS: st.column_config.TextColumn('Verticals', width='medium'),
            nl.COL_TITLE:     st.column_config.TextColumn(
                'Solicitation title', width='medium',
                help='The official title the agency published — the headline readers search for.'),
            'title':          st.column_config.TextColumn('Topic title', width='medium'),
            'agency':         st.column_config.TextColumn('Agency', width='small'),
            'deadline':       st.column_config.TextColumn('Due', width='small'),
            nl.COL_HOOK:      st.column_config.TextColumn('Hook', width='large'),
            nl.COL_REASON:    st.column_config.TextColumn('Why', width='large'),
            'source':         st.column_config.LinkColumn('Link', width='small', display_text='open'),
            'folder':         st.column_config.TextColumn('Folder', width='small'),
            'reviewed':       st.column_config.TextColumn('Reviewed by', width='small'),
        },
    )

    changed = [i for i in edited.index
               if any(edited.at[i, c] != table.at[i, c] for c in _EDITABLE)]
    bad = {}
    for i in changed:
        _, unknown = nl.parse_verticals(edited.at[i, nl.COL_VERTICALS])
        if unknown:
            bad[i] = unknown
        elif edited.at[i, nl.COL_GOOD] and not nl.parse_verticals(edited.at[i, nl.COL_VERTICALS])[0]:
            bad[i] = ['(included but no vertical)']
    for i, unknown in list(bad.items())[:10]:
        st.error(f'“{table.at[i, "title"][:80]}”: not a newsletter vertical — {", ".join(unknown)}')

    b1, b2 = st.columns([1, 3])
    with b1:
        save = st.button(f'💾 Save review ({len(changed)} changed)', type='primary',
                         disabled=bool(bad))
    with b2:
        mark_all = st.checkbox('Also mark every row shown as reviewed', key='nl_mark_all',
                               help='Records you as the reviewer even on rows you left unchanged.')

    if save:
        who = st.session_state.get('user_email') or 'unknown'
        today = date.today().isoformat()
        targets = list(edited.index) if mark_all else changed
        updates: dict = {}
        for i in targets:
            verts, _ = nl.parse_verticals(edited.at[i, nl.COL_VERTICALS])
            good = bool(edited.at[i, nl.COL_GOOD]) and bool(verts)
            values = {
                nl.COL_GOOD:        good,
                nl.COL_VERTICALS:   nl.VERTICAL_SEP.join(verts),
                nl.COL_TITLE:       _s(edited.at[i, nl.COL_TITLE]),
                nl.COL_HOOK:        _s(edited.at[i, nl.COL_HOOK]),
                nl.COL_REVIEWED_BY: who,
                nl.COL_REVIEWED_AT: today,
            }
            row = df.loc[i]
            updates.setdefault(row['_blob'], []).append((int(row['_row']), str(row['title']), values))
            for c, v in values.items():
                df.at[i, c] = v
        if not targets:
            st.info('Nothing to save.')
        else:
            written, errors = nl.write_back(uc.get_storage_client(), _BUCKET, updates)
            st.session_state.nl_df = df
            st.session_state.nl_msgs = [f'Saved the review of **{written}** topic(s).'] + errors[:10]
            st.rerun()

# ── 4 · Draft ──────────────────────────────────────────────────────────────

st.subheader('4 · Draft')
chosen = df[checked & df[nl.COL_GOOD]]
if chosen.empty:
    st.info('No topic in this window is marked for the newsletter.')
else:
    by_vertical = {
        v: int(chosen[nl.COL_VERTICALS].map(lambda s: nl.parse_verticals(s)[0][:1] == [v]).sum())
        for v in nl.VERTICALS
    }
    st.caption(' · '.join(f'{v} **{n}**' for v, n in by_vertical.items() if n)
               + f' · empty: {sum(1 for n in by_vertical.values() if not n)} vertical(s)')
    md = _draft_markdown(chosen, start, end)
    with st.expander('Preview', expanded=True):
        st.markdown(md)

    export = pd.DataFrame({
        'vertical':           chosen[nl.COL_VERTICALS].map(lambda s: (nl.parse_verticals(s)[0] or [''])[0]),
        'also_relevant_to':   chosen[nl.COL_VERTICALS].map(lambda s: ', '.join(nl.parse_verticals(s)[0][1:])),
        'headline':           chosen.apply(_headline, axis=1),
        'solicitation_title': chosen[nl.COL_TITLE],
        'topic_title':        chosen['title'].map(nl.clean_title),
        'agency':             chosen['agency'].map(_s) if 'agency' in chosen else '',
        'deadline':           chosen.apply(_deadline, axis=1),
        'hook':               chosen[nl.COL_HOOK],
        'link':               chosen['source'].map(_s) if 'source' in chosen else '',
        'topic_number':       chosen['topic_number'].map(_s) if 'topic_number' in chosen else '',
        'reviewed_by':        chosen[nl.COL_REVIEWED_BY],
    })
    order = {v: i for i, v in enumerate(nl.VERTICALS)}
    export = export.sort_values('vertical', key=lambda s: s.map(order))
    stamp = start.isoformat() if start == end else f'{start.isoformat()}_{end.isoformat()}'
    d1, d2 = st.columns(2)
    d1.download_button('⬇️ Draft (.md)', md, file_name=f'newsletter_{stamp}.md',
                       mime='text/markdown')
    d2.download_button('⬇️ Items (.csv)', export.to_csv(index=False),
                       file_name=f'newsletter_{stamp}.csv', mime='text/csv')
