"""Deck helpers: house-style cleanup, narrative parse, deadlines, template."""

from datetime import date

import src.modules.roadmap.deck as deck


def test_clean_copy_strips_em_dashes():
    assert '—' not in deck.clean_copy('Phase I — then Phase II')
    assert deck.clean_copy('a  b\n c') == 'a b c'
    assert deck.clean_copy(None) == ''


def test_banned_hits():
    assert 'leverage' in deck.banned_hits('We leverage robust synergy')
    assert deck.banned_hits('A plain factual sentence.') == []


def test_parse_narrative_aligns_items_by_index():
    raw = ('garbage before {"why_now": "Now is good — really.", '
           '"items": [{"i": 0, "what_it_is": "A", "why_it_fits": "B"}, '
           '{"i": 2, "what_it_is": "C", "why_it_fits": "D"}]} trailing')
    out = deck._parse_narrative(raw)
    assert '—' not in out['why_now']
    assert set(out['items']) == {0, 2}
    assert out['items'][0] == {'what_it_is': 'A', 'why_it_fits': 'B'}


def test_parse_narrative_requires_json():
    import pytest
    with pytest.raises(ValueError):
        deck._parse_narrative('no json here')


def test_deadline_display_rolling_date_and_blank():
    assert deck.deadline_display({'is_rolling': True}) == ('Rolling', 'Open until filled')
    label, cd = deck.deadline_display({'due_date': '2026-01-14'}, today=date(2026, 1, 1))
    assert label == '14 Jan 2026' and cd == '13 days out'
    assert deck.deadline_display({'due_date': ''})[0] == 'Deadline to confirm'


def test_build_context_no_invented_amounts():
    opps = [{'title': 'DoD SBIR', 'agency': 'ARMY', 'grant_summary': 'Sensors.',
             'due_date': '2026-02-01', 'source': 'https://x', 'aspect_label': 'MEMS'}]
    ctx = deck.build_context(company_name='Acme', profile_summary='Sensors co.',
                             opportunities=opps,
                             narrative={'why_now': 'Soon.', 'items': {0: {
                                 'what_it_is': 'Funds prototypes.',
                                 'why_it_fits': 'Your MEMS fits.'}}},
                             presentation_date=date(2026, 1, 1))
    assert ctx['opportunities'][0]['number'] == 1
    assert ctx['opportunities'][0]['why_it_fits'] == 'Your MEMS fits.'
    # No dollar figure keys are ever emitted into the deck context.
    blob = str(ctx)
    assert '$' not in blob


def test_render_html_contains_company_and_opportunities():
    ctx = deck.build_context(company_name='Acme Robotics', profile_summary='Sensors.',
                             opportunities=[{'title': 'DoD SBIR', 'agency': 'ARMY',
                                             'grant_summary': 'x', 'source': 'https://x',
                                             'aspect_label': 'MEMS', 'due_date': '2026-02-01'}],
                             narrative={'why_now': 'Now.', 'items': {}},
                             presentation_date=date(2026, 1, 1))
    html = deck.render_html(ctx)
    assert 'Acme Robotics' in html and 'DoD SBIR' in html
    assert 'BW' in html  # wordmark


def test_render_html_empty_state():
    ctx = deck.build_context(company_name='Test Co', profile_summary='',
                             opportunities=[], narrative={'why_now': '', 'items': {}})
    html = deck.render_html(ctx)
    assert 'did not find a confident grant match' in html


# ── Verified ceiling display (Stage: verified ceilings) ──────────────────────

def test_fmt_usd():
    assert deck._fmt_usd(2095748) == '$2.1M'
    assert deck._fmt_usd(1000000) == '$1M'
    assert deck._fmt_usd(305000) == '$305K'
    assert deck._fmt_usd(0) == '' and deck._fmt_usd(None) == '' and deck._fmt_usd(float('nan')) == ''


def test_ceiling_display_exact_for_notice_sources():
    assert deck.ceiling_display(
        {'ceiling_source': 'grants_gov', 'award_ceiling_num': 500000.0}
    ) == 'Award ceiling: $500K'
    assert deck.ceiling_display(
        {'ceiling_source': 'stated_text', 'award_ceiling_num': 1800000.0}
    ) == 'Award ceiling: $1.8M'


def test_ceiling_display_hedged_for_table():
    out = deck.ceiling_display({'ceiling_source': 'agency_table',
                                'phase1_ceiling_num': 250000.0,
                                'phase2_ceiling_num': 1800000.0})
    assert 'Phase I up to ~$250K' in out and 'program typical' in out


def test_ceiling_display_nothing_when_no_source_or_suppressed():
    assert deck.ceiling_display({'ceiling_source': '', 'award_ceiling_num': None}) == ''
    assert deck.ceiling_display({'ceiling_source': 'agency_table',
                                 'phase1_ceiling_num': 250000.0},
                                show_table_ceilings=False) == ''


def test_build_context_sets_ceiling_and_prompt_still_forbids_inventing():
    ctx = deck.build_context(
        company_name='Acme', profile_summary='p',
        opportunities=[{'title': 'T', 'ceiling_source': 'grants_gov',
                        'award_ceiling_num': 500000.0}],
        narrative={'items': {}})
    assert ctx['opportunities'][0]['ceiling'] == 'Award ceiling: $500K'
    # The model is still told never to invent a dollar amount (we render verified
    # figures in a separate chip, so the guard stays).
    assert 'NEVER state a dollar amount' in deck._build_system()


# ── Funding totals summary (estimate of total available) ─────────────────────

def test_opportunity_value_by_source():
    assert deck.opportunity_value(
        {'ceiling_source': 'grants_gov', 'award_ceiling_num': 500000}) == 500000
    assert deck.opportunity_value(
        {'ceiling_source': 'stated_text', 'award_ceiling_num': 150000}) == 150000
    # SBIR table → full Phase I + Phase II path
    assert deck.opportunity_value(
        {'ceiling_source': 'agency_table', 'phase1_ceiling_num': 250000,
         'phase2_ceiling_num': 1800000}) == 2050000
    # table suppressed, or no sourced figure → None
    assert deck.opportunity_value(
        {'ceiling_source': 'agency_table', 'phase1_ceiling_num': 250000},
        show_table_ceilings=False) is None
    assert deck.opportunity_value({'ceiling_source': '', 'award_ceiling_num': None}) is None


def test_funding_totals_sums_and_counts():
    opps = [
        {'ceiling_source': 'grants_gov', 'award_ceiling_num': 500000},
        {'ceiling_source': 'agency_table', 'phase1_ceiling_num': 250000,
         'phase2_ceiling_num': 1800000},
        {'ceiling_source': '', 'award_ceiling_num': None},   # no figure
    ]
    t = deck.funding_totals(opps)
    assert t['amount'] == 2550000
    assert t['amount_display'] == '$2.5M'
    assert t['n_programs'] == 2 and t['n_total'] == 3
    assert t['all_counted'] is False


def test_funding_totals_none_when_no_figures():
    assert deck.funding_totals([{'ceiling_source': '', 'award_ceiling_num': None}]) is None
    assert deck.funding_totals([]) is None


def test_funding_totals_respects_suppression():
    opps = [{'ceiling_source': 'agency_table', 'phase1_ceiling_num': 250000,
             'phase2_ceiling_num': 1800000}]
    assert deck.funding_totals(opps, show_table_ceilings=False) is None
    assert deck.funding_totals(opps, show_table_ceilings=True)['amount'] == 2050000


def test_build_context_includes_totals_and_renders():
    ctx = deck.build_context(
        company_name='Acme', profile_summary='p',
        opportunities=[
            {'title': 'A', 'ceiling_source': 'grants_gov', 'award_ceiling_num': 500000},
            {'title': 'B', 'ceiling_source': 'agency_table',
             'phase1_ceiling_num': 250000, 'phase2_ceiling_num': 1800000},
        ],
        narrative={'items': {}})
    assert ctx['totals']['n_programs'] == 2 and ctx['totals']['all_counted'] is True
    html = deck.render_html(ctx)
    assert ctx['totals']['amount_display'] in html
    assert 'potential non-dilutive funding' in html
    assert 'across 2 programs' in html
