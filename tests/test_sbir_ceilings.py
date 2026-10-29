"""Unit tests for the SBIR/STTR ceiling reference table and layered resolver."""

import src.modules.sbir_ceilings as sc


# ── detect_program ──────────────────────────────────────────────────────────

def test_detect_program():
    assert sc.detect_program('A DoD SBIR Phase I topic') == 'SBIR'
    assert sc.detect_program('Army STTR solicitation') == 'STTR'
    assert sc.detect_program('Janitorial services contract') is None
    assert sc.detect_program('anything', explicit='STTR') == 'STTR'   # explicit wins
    assert sc.detect_program('sbir in body', explicit='nonsense') == 'SBIR'


# ── canonical_agency ────────────────────────────────────────────────────────

def test_canonical_agency_from_folder_and_string():
    assert sc.canonical_agency('DOD') == 'DOD'
    assert sc.canonical_agency('DARPA') == 'DOD'              # folds into DOD
    # Multi-agency folder → resolve from the row's own agency string
    assert sc.canonical_agency('GRANTS-GOV', 'National Institutes of Health') == 'HHS'
    assert sc.canonical_agency('SAM-GOV', 'DEPT OF DEFENSE.DEPT OF THE ARMY') == 'DOD'
    assert sc.canonical_agency('SAM-GOV', 'National Science Foundation') == 'NSF'
    assert sc.canonical_agency('GRANTS-GOV', 'Unknown Micro-Agency') is None


# ── resolution order: grants_gov > stated_text > agency_table ────────────────

def test_grants_gov_wins_and_is_exact():
    out = sc.resolve_ceilings(
        broad_agency='GRANTS-GOV', agency='NIH', program_text='SBIR',
        gg_synopsis={'awardCeiling': '500000', 'awardFloor': '50000'},
        funding_amount_text='Up to $9,000,000')
    assert out['ceiling_source'] == 'grants_gov'
    assert out['award_ceiling_num'] == 500000.0
    assert out['award_floor_num'] == 50000.0
    # SBIR phase caps still populated from the table alongside the exact figure
    assert out['phase1_ceiling_num'] and out['phase2_ceiling_num']


def test_stated_text_beats_table():
    out = sc.resolve_ceilings(
        broad_agency='SAM-GOV', agency='Army', program_text='SBIR',
        funding_amount_text='$50,000-$250,000')
    assert out['ceiling_source'] == 'stated_text'
    assert out['award_ceiling_num'] == 250000.0           # upper bound of the range


def test_agency_table_fallback_for_sbir():
    out = sc.resolve_ceilings(broad_agency='DOD', program_text='SBIR Phase I')
    assert out['ceiling_source'] == 'agency_table'
    assert out['ceiling_basis'] == 'dod_sbir_table'
    assert out['phase1_ceiling_num'] == 250000.0
    assert out['award_ceiling_num'] is None               # table-only: no notice figure


def test_non_sbir_gets_nothing_from_table():
    out = sc.resolve_ceilings(broad_agency='SAM-GOV', agency='Navy',
                              program_text='Janitorial services')
    assert out['ceiling_source'] == ''
    assert out['phase1_ceiling_num'] is None
    assert out['award_ceiling_num'] is None


def test_unknown_agency_gets_no_phase_caps():
    # A row that mentions SBIR but whose agency doesn't resolve to a known federal
    # SBIR agency (state match programs, accelerators, passing mentions) must NOT
    # get the federal statutory cap stamped on it.
    out = sc.resolve_ceilings(broad_agency='CONSORTIUM', agency='mietf.org',
                              program_text='SBIR/STTR State Match Program')
    assert out['ceiling_source'] == ''
    assert out['phase1_ceiling_num'] is None
    assert out['award_ceiling_num'] is None


def test_unknown_agency_still_takes_exact_notice_figure():
    # ...but an exact stated/Grants.gov figure on such a row is still captured.
    out = sc.resolve_ceilings(broad_agency='STATE', agency='dceo.illinois.gov',
                              program_text='SBIR Match', funding_amount_text='Up to $50,000')
    assert out['ceiling_source'] == 'stated_text'
    assert out['award_ceiling_num'] == 50000.0
    assert out['phase1_ceiling_num'] is None


def test_grants_gov_zero_ceiling_is_not_specified():
    # Grants.gov returns 0 for an unset ceiling — must read as "no figure", not $0,
    # and must not win the layer (falls through to the SBIR table here).
    out = sc.resolve_ceilings(broad_agency='DOD', program_text='SBIR',
                              gg_synopsis={'awardCeiling': '0', 'awardFloor': '0'})
    assert out['award_ceiling_num'] is None
    assert out['ceiling_source'] == 'agency_table'


def test_sttr_falls_back_to_sbir_row_when_absent():
    # USDA has only an SBIR row in the table; an STTR topic reuses it.
    out = sc.resolve_ceilings(broad_agency='USDA', program_text='USDA STTR')
    assert out['phase1_ceiling_num'] == float(sc.CEILINGS[('USDA', 'SBIR')]['phase1'])


# ── overrides ───────────────────────────────────────────────────────────────

def test_apply_overrides_wins_over_table():
    try:
        sc.apply_overrides({'DOD|SBIR': {'phase1': 999_999, 'phase2': 8_888_888}})
        out = sc.resolve_ceilings(broad_agency='DOD', program_text='SBIR')
        assert out['phase1_ceiling_num'] == 999_999.0
        assert out['phase2_ceiling_num'] == 8_888_888.0
    finally:
        sc._OVERRIDES.clear()


# ── CEILING_COLUMNS contract ────────────────────────────────────────────────

def test_resolve_returns_exactly_the_six_columns():
    out = sc.resolve_ceilings(broad_agency='DOD', program_text='SBIR')
    assert set(out) == set(sc.CEILING_COLUMNS)


def test_sub_floor_award_figure_is_dropped():
    # "50% of project fee, up to $15,000" mis-parses to 50 — a $50 award ceiling is
    # a parse artifact, so it drops to "no figure" rather than showing $50.
    out = sc.resolve_ceilings(broad_agency='CONSORTIUM', agency='x.org',
                              program_text='match',
                              funding_amount_text='50% of project fee, up to $15,000')
    assert out['award_ceiling_num'] is None
    assert out['ceiling_source'] == ''
    # A real figure above the floor is kept.
    out2 = sc.resolve_ceilings(broad_agency='STATE', agency='y.gov', program_text='m',
                               funding_amount_text='Up to $150,000')
    assert out2['award_ceiling_num'] == 150000.0
