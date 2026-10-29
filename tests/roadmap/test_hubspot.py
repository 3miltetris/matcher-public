"""HubSpot read/write layer for the roadmap."""

import src.modules.roadmap.hubspot_roadmap as hr


def test_get_contact_by_id_and_form_inputs(hubspot):
    contact = hr.get_contact(hubspot, contact_id='c123')
    assert contact['_id'] == 'c123'
    inp = hr.form_inputs(contact)
    assert inp['contact_id'] == 'c123'
    assert inp['email'] == 'ada@acme.com'
    assert inp['company'] == 'Acme Robotics'
    assert inp['website'] == 'acme.com'
    assert 'MEMS gas sensors' in inp['description']


def test_get_contact_by_email(hubspot):
    contact = hr.get_contact(hubspot, email='Ada@Acme.com')
    assert contact['_id'] == 'c123'


def test_get_contact_missing_email_raises(hubspot):
    import pytest
    from src.modules.intake.hubspot_sync import HubSpotError
    with pytest.raises(HubSpotError):
        hr.get_contact(hubspot, email='nobody@nowhere.com')


def test_write_roadmap_creates_props_and_patches(hubspot):
    hr.write_roadmap(hubspot, 'c123', url='https://host/api/roadmap/tok.pdf',
                     status='complete', generated_at='2026-01-01')
    assert hr.PROP_URL in hubspot.created_props      # auto-created
    assert hubspot.patched[hr.PROP_URL] == 'https://host/api/roadmap/tok.pdf'
    assert hubspot.patched[hr.PROP_STATUS] == 'complete'
    assert hubspot.patched[hr.PROP_GENERATED_AT] == '2026-01-01'


def test_form_inputs_company_from_website_when_no_company_field():
    # The live form has no company field, only a website — derive the name.
    inp = hr.form_inputs({'_id': 'x', 'firstname': 'Grace', 'website': 'acme-robotics.com'})
    assert inp['company'] == 'Acme Robotics'


def test_form_inputs_falls_back_to_person_name_when_no_website():
    inp = hr.form_inputs({'_id': 'x', 'firstname': 'Grace', 'lastname': 'Hopper',
                          'email': 'g@h.com'})
    assert inp['company'] == 'Grace Hopper'


def test_company_from_website():
    assert hr.company_from_website('https://acme-robotics.com/about') == 'Acme Robotics'
    assert hr.company_from_website('') == ''


def test_contact_ref_from_webhook_shapes():
    assert hr.contact_ref_from_webhook({'objectId': 12345}) == ('12345', None)
    assert hr.contact_ref_from_webhook({'vid': 7}) == ('7', None)
    assert hr.contact_ref_from_webhook({'contact_id': 'c1', 'mode': 'x'})[0] == 'c1'
    cid, email = hr.contact_ref_from_webhook({'properties': {'email': {'value': 'A@B.com'}}})
    assert cid is None and email == 'a@b.com'
    assert hr.contact_ref_from_webhook({}) == (None, None)
    assert hr.contact_ref_from_webhook('nope') == (None, None)
