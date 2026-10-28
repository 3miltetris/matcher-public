"""Unit tests for the shared grant-topic date helpers and the expiry filter
folded into drop_dead_topics (Stage: rolling deadlines / connector fixes)."""

from datetime import date

import pandas as pd

from src.modules import grant_utils as gu

TODAY = date(2026, 10, 26)


# ── parse_deadline / clean_date_display ─────────────────────────────────────────

def test_clean_date_display_scrubs_junk():
    for junk in ('nan', 'NaT', 'None', 'null', '<NA>', '', '  '):
        assert gu.clean_date_display(junk) == ''
    assert gu.clean_date_display('2026-12-01') == '2026-12-01'
    assert gu.clean_date_display(None) == ''


def test_parse_deadline_formats_and_tokens():
    assert gu.parse_deadline('2026-12-01') == date(2026, 12, 1)
    assert gu.parse_deadline('12/01/2026') == date(2026, 12, 1)
    assert gu.parse_deadline('2026-12-01T17:00:00-05:00') == date(2026, 12, 1)
    for token in ('nan', 'NaT', '', 'Rolling', 'open until filled', 'TBD', 'N/A',
                  'see solicitation', 'garbage'):
        assert gu.parse_deadline(token) is None


# ── topic_open_status ───────────────────────────────────────────────────────────

def test_topic_open_status():
    f = lambda **r: gu.topic_open_status(r, TODAY)
    assert f(due_date='2030-01-01') == 'open'
    assert f(due_date='2001-01-01') == 'expired'
    assert f(due_date='', is_rolling=False) == 'unknown'
    assert f(due_date='nan') == 'unknown'
    assert f(due_date='2001-01-01', is_rolling=True) == 'rolling'    # rolling wins
    assert f(due_date='open until filled') == 'rolling'              # token → rolling
    assert f(close_date='2030-05-05') == 'open'                      # close_date fallback
    # A past-dated award is never 'expired'.
    assert f(due_date='2001-01-01', record_kind='award') == 'unknown'
    # is_rolling read back as a string from parquet/CSV.
    assert f(due_date='2001-01-01', is_rolling='True') == 'rolling'


# ── drop_dead_topics expiry fold ────────────────────────────────────────────────

def test_drop_dead_topics_expiry_and_legacy_filters():
    df = pd.DataFrame([
        {'topic_number': 'FUTURE', 'due_date': '2030-01-01', 'is_rolling': False,
         'sam_status': 'active', 'verify_status': 'active'},
        {'topic_number': 'PAST', 'due_date': '2001-01-01', 'is_rolling': False,
         'sam_status': 'active', 'verify_status': 'active'},
        {'topic_number': 'ROLLING_PAST', 'due_date': '2001-01-01', 'is_rolling': True,
         'sam_status': 'active', 'verify_status': 'active'},
        {'topic_number': 'BLANK', 'due_date': '', 'is_rolling': False,
         'sam_status': 'active', 'verify_status': 'active'},
        {'topic_number': 'ARCHIVED', 'due_date': '2030-01-01', 'is_rolling': False,
         'sam_status': 'archived', 'verify_status': 'active'},
        {'topic_number': 'INACTIVE', 'due_date': '2030-01-01', 'is_rolling': False,
         'sam_status': 'active', 'verify_status': 'inactive'},
    ])
    kept = set(gu.drop_dead_topics(df, TODAY)['topic_number'])
    assert kept == {'FUTURE', 'ROLLING_PAST', 'BLANK'}


def test_drop_dead_topics_nan_is_rolling_is_not_rolling():
    # A NaN `is_rolling` must read as False (not rolling), so a past-dated row is
    # still dropped as expired — matching the single-row topic_open_status path.
    df = pd.DataFrame([
        {'topic_number': 'PAST', 'due_date': '2001-01-01', 'is_rolling': float('nan')},
        {'topic_number': 'FUTURE', 'due_date': '2030-01-01', 'is_rolling': float('nan')},
    ])
    kept = set(gu.drop_dead_topics(df, TODAY)['topic_number'])
    assert kept == {'FUTURE'}


def test_drop_dead_topics_keeps_awards():
    df = pd.DataFrame([
        {'topic_number': 'AWD', 'due_date': '2001-01-01', 'record_kind': 'award'},
        {'topic_number': 'DEAD', 'due_date': '2001-01-01'},
    ])
    kept = set(gu.drop_dead_topics(df, TODAY)['topic_number'])
    assert kept == {'AWD'}


def test_drop_dead_topics_empty_and_no_date_cols():
    assert gu.drop_dead_topics(pd.DataFrame()).empty
    # No date columns at all → only archived/inactive filters apply (nothing dropped).
    df = pd.DataFrame([{'topic_number': 'X', 'sam_status': 'active'}])
    assert len(gu.drop_dead_topics(df, TODAY)) == 1


# ── normalize_grant_columns scrub ───────────────────────────────────────────────

def test_normalize_scrubs_nan_dates():
    df = pd.DataFrame([{'grant_summary': 'x', 'due_date': 'nan', 'close_date': 'NaT'}])
    out = gu.normalize_grant_columns(df)
    assert out.iloc[0]['due_date'] == '' and out.iloc[0]['close_date'] == ''


def test_normalize_guarantees_source_column():
    # Missing entirely → added as ''.
    out = gu.normalize_grant_columns(pd.DataFrame([{'grant_summary': 'x'}]))
    assert 'source' in out.columns and out.iloc[0]['source'] == ''
    # Junk tokens blanked; real URLs kept.
    out = gu.normalize_grant_columns(pd.DataFrame([
        {'grant_summary': 'x', 'source': 'nan'},
        {'grant_summary': 'y', 'source': 'https://sam.gov/opp/abc/view'},
    ]))
    assert list(out['source']) == ['', 'https://sam.gov/opp/abc/view']


# ── classify_record_kind / add_record_kind ──────────────────────────────────────

def test_classify_record_kind():
    f = gu.classify_record_kind
    # Fundable R&D → solicitation (the default).
    assert f({'title': 'NIAID Omnibus SBIR',
              'grant_summary': 'development of a novel vaccine platform'}) == 'solicitation'
    # VA/DLA supply buys title as a 4-digit PSC code then '--'.
    assert f({'title': '6505--Radiopharmaceuticals'}) == 'procurement'
    assert f({'title': '6515--442 NUCLEAR MEDICINE'}) == 'procurement'
    # Sole-source / intent-to-award / sources-sought.
    assert f({'title': 'Sole Source Notice'}) == 'procurement'
    assert f({'title': 'X', 'grant_summary': 'Notice of intent to award to Acme'}) == 'procurement'
    assert f({'title': 'Y', 'notice_type': 'Sources Sought'}) == 'procurement'
    # Completed awards.
    assert f({'title': 'Z', 'notice_type': 'Award Notice'}) == 'award'
    assert f({'title': 'Z', 'awardee_name': 'Acme Inc'}) == 'award'
    assert f({'title': 'Z', 'award_amount': '250000'}) == 'award'
    assert f({'title': 'Z', 'sam_status': 'awarded'}) == 'award'
    # An existing 'award' record_kind is preserved even without award fields.
    assert f({'title': 'Z', 'record_kind': 'award'}) == 'award'
    # A blank/zero award field is not an award.
    assert f({'title': 'Real R&D topic', 'award_amount': ''}) == 'solicitation'
    assert f({'title': 'Real R&D topic', 'awardee_name': 'nan'}) == 'solicitation'


def test_add_record_kind_frame_and_default():
    df = pd.DataFrame([
        {'title': '6505--Radiopharmaceuticals', 'grant_summary': ''},
        {'title': 'NIAID SBIR', 'grant_summary': 'novel vaccine R&D'},
        {'title': 'won', 'record_kind': 'award'},
    ])
    out = gu.add_record_kind(df)
    assert list(out['record_kind']) == ['procurement', 'solicitation', 'award']
    # The awards loader passes default='award' so unmarked rows still read as award.
    out2 = gu.add_record_kind(pd.DataFrame([{'title': 'anything'}]), default='award')
    assert out2.iloc[0]['record_kind'] == 'award'
