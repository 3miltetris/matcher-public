from src.modules.intake import schema as sc
from src.modules.intake import schema_export as sx
from src.modules.intake.validation import clean_answers, is_visible, validate


def test_schema_is_structurally_valid():
    assert sc.validate_schema() == []


def test_every_answer_maps_to_a_form_property():
    mapped = [f for f in sc.FIELDS if f.type != 'file']
    assert all(f.hubspot_property for f in mapped)
    assert len({f.hubspot_property for f in mapped}) == len(mapped)


def test_option_5_placeholder_is_dropped():
    assert 'Option 5' not in sc.FIELDS_BY_ID['who_pays'].options


def test_health_questions_visible_only_for_health_verticals():
    f = sc.FIELDS_BY_ID['patient_population']
    assert f.required
    assert not is_visible(f, {sc.VERTICALS: ['Cybersecurity']})
    assert is_visible(f, {sc.VERTICALS: ['Cybersecurity', 'Medtech']})


def test_hidden_required_question_is_not_required_and_not_stored(answers):
    _, errors = validate(answers)
    assert errors == {}                                  # patient population hidden → not required
    answers['patient_population'] = ['Adult']
    assert 'patient_population' not in clean_answers(answers)
    answers.pop('patient_population')
    _, errors = validate(dict(answers, verticals=['Biotech']))
    assert {'patient_population', 'study_types'} <= set(errors)


def test_investor_and_follow_up_conditions(answers):
    inv = sc.FIELDS_BY_ID['investor_types']
    assert not is_visible(inv, {'current_funding': ['Bootstrapped']})
    assert is_visible(inv, {'current_funding': ['Bootstrapped', 'Outside Investors']})
    adj = sc.FIELDS_BY_ID['adjacent_markets']
    assert is_visible(adj, {'open_to_adjacent': 'Possibly, depending on the project'})
    assert not is_visible(adj, {'open_to_adjacent': 'No'})
    fed = sc.FIELDS_BY_ID['federal_funding_detail']
    assert not is_visible(fed, {'federal_funding_received': ['No']})
    assert is_visible(fed, {'federal_funding_received': ['SBIR/STTR']})


def test_validate_required_options_and_email(answers):
    bad = dict(answers, stage='Almost done', contact_email='nope')
    bad.pop('company_legal_name')
    _, errors = validate(bad)
    assert set(errors) == {'stage', 'contact_email', 'company_legal_name'}


def test_multi_select_rejects_unknown_option(answers):
    _, errors = validate(dict(answers, verticals=['Biotech', 'Crypto']))
    assert 'verticals' in errors


def test_frontend_schema_shape():
    out = sx.frontend_schema()
    ids = [f['id'] for s in out['sections'] for f in s['fields']]
    assert sc.COMPANY_NAME in ids
    first = out['sections'][0]['fields'][0]
    assert 'profile_role' not in first and 'option_values' not in first


def test_profile_split_keeps_eligibility_and_intentions_out_of_capability(answers):
    a = dict(answers, next_milestone='Enter the defense market')
    ext = sx.profile_extracted(a)
    assert ext[sc.TECH_DESCRIPTION].startswith('MEMS')
    assert 'Enter the defense market' in ' '.join(ext['notable_updates'])
    assert 'us_owned_51' not in ext and 'next_milestone' not in ext
    assert 'owned by US citizens' not in sx.digest(a)


def test_hubspot_writes_coded_option_values_to_form_properties(answers):
    a = dict(answers, evidence=['Patents Filed or Granted', 'Paying Customers'])
    contact = sx.hubspot_properties(a, 'contact')
    assert contact['check_all_that_exist_'] == '5s_sLQmUGf1DkcPcrAEBA;jA22xRkizVSQWG9rmEuZe'
    assert contact['firstname'] == 'Ada'
    company = sx.hubspot_properties(a, 'company')
    assert company['verticals'] == 'Robotics & Autonomous Systems'
    assert company['company_name__legal_'] == 'Acme Robotics, Inc.'


def test_owned_properties_and_created_properties():
    owned = sx.owned_properties('company')
    assert {'dd_submitted_at', 'verticals'} <= owned
    assert 'name' not in owned and 'domain' not in owned
    assert [n for n, *_ in sx.hubspot_property_defs('company')] == [
        n for n, *_ in sx.DD_PROPERTIES]
    assert sx.hubspot_property_defs('contact') == []
