"""Onboarding proposal validation + drive_read domain helpers."""

import src.modules.onboarding as ob
from src.modules.tools import drive_read as dr

_CTX = dict(
    surfaced_file_ids={'fA', 'fB'},
    surfaced_domains={'acme.com'},
    folder_file_ids={'fA', 'fB'},
    known_keys_by_pool={'clients': {'Acme||https://acme.com'}, 'prospects': set()},
)


def test_valid_existing_client():
    p = {'decision': 'existing_client', 'company_key': 'Acme||https://acme.com',
         'confidence': 'high',
         'evidence': [{'claim': 'letterhead', 'source': 'file:fA'}],
         'documents': [{'file_id': 'fB', 'include': True}]}
    assert ob.validate_proposal(p, **_CTX) == []


def test_existing_client_unknown_key_rejected():
    p = {'decision': 'existing_client', 'company_key': 'Ghost||x', 'confidence': 'low',
         'evidence': []}
    errs = ob.validate_proposal(p, **_CTX)
    assert any('not found in clients pool' in e for e in errs)


def test_evidence_file_not_surfaced_rejected():
    p = {'decision': 'unsure', 'confidence': 'low',
         'evidence': [{'claim': 'x', 'source': 'file:fZ'}]}
    errs = ob.validate_proposal(p, **_CTX)
    assert any("file 'fZ' never returned" in e for e in errs)


def test_new_company_invented_website_rejected():
    p = {'decision': 'new_client', 'confidence': 'medium', 'evidence': [],
         'new_company': {'company_name': 'New Co', 'website': 'https://invented.io',
                         'website_source': 'document'}}
    errs = ob.validate_proposal(p, **_CTX)
    assert any('not among surfaced domains' in e for e in errs)


def test_new_company_surfaced_website_ok():
    p = {'decision': 'new_client', 'confidence': 'medium', 'evidence': [],
         'new_company': {'company_name': 'Acme', 'website': 'https://acme.com',
                         'website_source': 'domain_signal'}}
    assert ob.validate_proposal(p, **_CTX) == []


def test_document_not_in_folder_rejected():
    p = {'decision': 'unsure', 'confidence': 'low', 'evidence': [],
         'documents': [{'file_id': 'fGhost', 'include': True}]}
    errs = ob.validate_proposal(p, **_CTX)
    assert any("file 'fGhost' not in the folder listing" in e for e in errs)


# ── drive_read domain helpers ────────────────────────────────────────────────

def test_bare_domain():
    assert dr.bare_domain('jane@Acme-Robotics.com') == 'acme-robotics.com'
    assert dr.bare_domain('https://www.acme.com/path') == 'acme.com'
    assert dr.bare_domain('') == ''


def test_is_business_domain_filters():
    assert dr.is_business_domain('acme.com')
    assert not dr.is_business_domain('gmail.com')          # freemail
    assert not dr.is_business_domain('bwcoconsulting.com') # our own
    assert not dr.is_business_domain('army.mil')           # gov/mil
    assert not dr.is_business_domain('nih.gov')
