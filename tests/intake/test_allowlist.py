import pytest

from src.modules.intake import allowlist as al


def test_normalize_domain():
    assert al.normalize_domain('https://www.Acme.com/about') == 'acme.com'
    assert al.normalize_domain('@eng.acme.co.uk') == 'acme.co.uk'
    assert al.normalize_domain('jo@acme.io') == 'acme.io'
    assert al.normalize_domain('localhost') == ''


def test_save_refuses_freemail_and_junk_without_writing(storage):
    with pytest.raises(ValueError) as e:
        al.save(storage, ['acme.com', 'gmail.com', 'nope'], ['bad-address'], actor='t')
    assert 'gmail.com' in str(e.value) and 'nope' in str(e.value) and 'bad-address' in str(e.value)
    assert al.load(storage) == al.empty_doc()


def test_save_normalises_and_records_history(storage):
    al.save(storage, ['https://WWW.Acme.com', 'acme.com'], [' Founder@Gmail.com '], actor='a', note='first')
    doc = al.save(storage, ['acme.com', 'beta.io'], ['founder@gmail.com'], actor='b', note='second')
    assert doc['domains'] == ['acme.com', 'beta.io']
    assert doc['emails'] == ['founder@gmail.com']
    assert [h['note'] for h in al.load(storage)['history']] == ['first', 'second']


def test_is_approved():
    doc = {'domains': ['acme.com'], 'emails': ['founder@gmail.com']}
    assert al.is_approved(doc, 'Jo@Acme.com')
    assert al.is_approved(doc, 'jo@eng.acme.com')
    assert al.is_approved(doc, 'founder@gmail.com')
    assert not al.is_approved(doc, 'other@gmail.com')
    assert not al.is_approved(doc, 'jo@acme.com.evil.io')
    assert not al.is_approved(doc, 'jo@notacme.com')
    assert not al.is_approved(doc, 'not-an-email')
    # a freemail domain stored by hand is still never a blanket approval
    assert not al.is_approved({'domains': ['gmail.com'], 'emails': []}, 'x@gmail.com')
