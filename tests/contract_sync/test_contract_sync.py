"""Unit tests for src/modules/contract_sync.py — pure matching + decision logic
and HubSpot quote→company resolution (Drive I/O is covered by the dry-run)."""

import pytest

from src.modules import contract_sync as cs


# ---------------------------------------------------------------- text
def test_fix_text_repairs_double_encoding():
    assert cs.fix_text('ProPlateÂ®') == 'ProPlate®'
    assert cs.fix_text('Clean Company') == 'Clean Company'


def test_safe_name_strips_slashes():
    assert cs.safe_name('A/B\\C') == 'A-B-C'
    assert cs.safe_name('') == 'Untitled'


def test_strip_hyperlink():
    assert cs.strip_hyperlink('=HYPERLINK("https://x.com/p", "HOLOCHIP CORP")') == 'HOLOCHIP CORP'
    assert cs.strip_hyperlink('Plain Company Inc') == 'Plain Company Inc'
    assert cs.strip_hyperlink('') == ''


@pytest.mark.parametrize('raw,expected', [
    ('https://www.Acme-Robotics.com/path', 'acme-robotics.com'),
    ('acme.com', 'acme.com'),
    ('sub.acme.co.uk', 'acme.co.uk'),
    ('880medical.com', '880medical.com'),
    ('', ''),
])
def test_bare_domain(raw, expected):
    assert cs.bare_domain(raw) == expected


def test_extract_term_ids_all_url_shapes_and_html_escaping():
    terms = (
        'See <a href="https://drive.google.com/file/d/1ce8JADdIOyO1x8AYnxCLysojpuERegzY/view">here</a> '
        'and https://docs.google.com/document/d/AAAAAAAAAAAAAAAAAAAAAAAA/edit '
        'and https://drive.google.com/open?id=BBBBBBBBBBBBBBBBBBBBBBBB&amp;usp=sharing'
    )
    ids = cs.extract_term_ids(terms)
    assert ids == ['1ce8JADdIOyO1x8AYnxCLysojpuERegzY',
                   'AAAAAAAAAAAAAAAAAAAAAAAA',
                   'BBBBBBBBBBBBBBBBBBBBBBBB']


def test_extract_term_ids_dedups_and_handles_empty():
    dup = 'x/file/d/AAAAAAAAAAAAAAAAAAAAAAAA/view y /file/d/AAAAAAAAAAAAAAAAAAAAAAAA/view'
    assert cs.extract_term_ids(dup) == ['AAAAAAAAAAAAAAAAAAAAAAAA']
    assert cs.extract_term_ids('') == []
    assert cs.extract_term_ids(None) == []


# ---------------------------------------------------------------- matching
@pytest.fixture
def index():
    assignments = {
        'f1': {'client_key': 'Acme||https://acme.com', 'folder_name': 'Acme Robotics_INTERNAL', 'assigned_at': '2026-01-01'},
        'f3': {'client_key': 'Acme||https://acme.com', 'folder_name': 'Acme Robotics Old_INTERNAL', 'assigned_at': '2026-02-01'},
        'f2': {'client_key': 'Beta||https://beta.io', 'folder_name': 'Beta_INTERNAL', 'assigned_at': '2026-01-01'},
        'f4': {'client_key': 'Dup1||https://dup.com', 'folder_name': 'Dup One_INTERNAL'},
        'f5': {'client_key': 'Dup2||https://dup.com', 'folder_name': 'Dup Two_INTERNAL'},
        'f6': {'client_key': 'NameX||https://nx1.com', 'folder_name': 'Shared Name_INTERNAL'},
        'f7': {'client_key': 'NameY||https://nx2.com', 'folder_name': 'Shared Name_INTERNAL'},
    }
    return cs.build_assignment_index(assignments)


def test_resolve_domain_exact(index):
    assert cs.resolve_client({'domain': 'acme.com'}, index) == ('Acme||https://acme.com', 'domain')
    # falls back to website when domain blank
    assert cs.resolve_client({'website': 'https://beta.io'}, index) == ('Beta||https://beta.io', 'domain')


def test_resolve_domain_ambiguous_goes_uncategorized(index):
    assert cs.resolve_client({'domain': 'dup.com'}, index) == (None, 'ambiguous_domain')


def test_resolve_name_fallback(index):
    # domain misses, exact normalized name hits exactly one client
    assert cs.resolve_client({'domain': 'nope.com', 'name': 'Beta'}, index) == ('Beta||https://beta.io', 'name')


def test_resolve_name_ambiguous(index):
    assert cs.resolve_client({'name': 'Shared Name'}, index) == (None, 'ambiguous_name')


def test_resolve_no_match(index):
    assert cs.resolve_client({'domain': 'nope.com', 'name': 'Nobody Co'}, index) == (None, 'no_match')
    assert cs.resolve_client({}, index) == (None, 'no_match')


def test_pick_folder_prefers_internal_then_earliest(index):
    folder = cs.pick_folder('Acme||https://acme.com', index)
    assert folder['folder_id'] == 'f1'  # earliest assigned_at among the two _INTERNAL folders
    assert cs.pick_folder('Nobody||x', index) is None


# ---------------------------------------------------------------- decision table
@pytest.mark.parametrize('existing_parent,dest,unc,matched,expected', [
    (None, 'dest', 'unc', True, 'create'),
    (None, 'unc', 'unc', False, 'create'),
    ('dest', 'dest', 'unc', True, 'skip'),
    ('unc', 'dest', 'unc', True, 'move'),          # now matched, was uncategorized
    ('unc', 'unc', 'unc', False, 'skip'),          # still unmatched, already parked
    ('unc', 'dest', 'unc', False, 'skip'),         # unmatched never moves
    ('other', 'dest', 'unc', True, 'skip'),        # already filed in a client folder
])
def test_decide(existing_parent, dest, unc, matched, expected):
    assert cs.decide(existing_parent, dest, unc, matched) == expected


def test_prefix_roundtrip():
    assert cs.prefixed('x.pdf', 'Acme Inc', matched=False) == '[Acme Inc] x.pdf'
    assert cs.prefixed('x.pdf', 'Acme Inc', matched=True) == 'x.pdf'
    assert cs.strip_prefix('[Acme Inc] 2026-10-13 - Deal (Signed).pdf') == '2026-10-13 - Deal (Signed).pdf'
    assert cs.strip_prefix('no prefix.pdf') == 'no prefix.pdf'


# ---------------------------------------------------------------- HubSpot resolution
def _fake_hs(routes):
    def hs(method, path, **kw):
        return routes.get((method, path), {'results': []})
    return hs


def test_company_for_quote_direct():
    hs = _fake_hs({
        ('GET', '/crm/v4/objects/quotes/Q1/associations/companies'):
            {'results': [{'toObjectId': 111, 'associationTypes': [{'label': 'Primary'}]}]},
        ('GET', '/crm/v3/objects/companies/111'):
            {'properties': {'name': 'Acme', 'domain': 'acme.com', 'website': 'https://acme.com'}},
    })
    assert cs.company_for_quote(hs, 'Q1') == {
        'id': '111', 'name': 'Acme', 'domain': 'acme.com', 'website': 'https://acme.com'}


def test_company_for_quote_deal_fallback():
    hs = _fake_hs({
        ('GET', '/crm/v4/objects/quotes/Q2/associations/companies'): {'results': []},
        ('GET', '/crm/v4/objects/quotes/Q2/associations/deals'): {'results': [{'toObjectId': 222}]},
        ('GET', '/crm/v4/objects/deals/222/associations/companies'): {'results': [{'toObjectId': 333}]},
        ('GET', '/crm/v3/objects/companies/333'):
            {'properties': {'name': 'Beta', 'domain': 'beta.io', 'website': ''}},
    })
    out = cs.company_for_quote(hs, 'Q2')
    assert out['id'] == '333' and out['name'] == 'Beta'


def test_company_for_quote_none():
    hs = _fake_hs({})  # every association empty
    assert cs.company_for_quote(hs, 'Q3') is None
