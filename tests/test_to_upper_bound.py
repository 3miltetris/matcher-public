"""Unit tests for finance_research.to_upper_bound — the ceiling-correct parser
(the high end of a range), distinct from to_number's midpoint."""

import src.modules.finance_research as fr


def test_range_returns_high_end_not_midpoint():
    assert fr.to_upper_bound('$50,000-$250,000') == 250000.0
    assert fr.to_number('$50,000-$250,000') == 150000.0        # midpoint, for contrast
    assert fr.to_upper_bound('$250K-$1M') == 1000000.0
    assert fr.to_upper_bound('1-2 million') == 2000000.0        # suffix carries to high end


def test_single_and_up_to():
    assert fr.to_upper_bound('Up to $2,000,000') == 2000000.0
    assert fr.to_upper_bound('$150,000') == 150000.0
    assert fr.to_upper_bound('~$4.2M') == 4200000.0


def test_nullish_and_labels():
    assert fr.to_upper_bound('Not found') is None
    assert fr.to_upper_bound('') is None
    assert fr.to_upper_bound(None) is None
    assert fr.to_upper_bound('FY2024: $2.5M') == 2500000.0      # year is a label


def test_numbers_pass_through():
    assert fr.to_upper_bound(500000) == 500000.0
    assert fr.to_upper_bound(0) == 0.0
    assert fr.to_upper_bound(True) is None


def test_to_upper_bound_str():
    assert fr.to_upper_bound_str('$50,000-$250,000') == '250000'
    assert fr.to_upper_bound_str('Not found') == ''
