"""search_companies / get_company against the in-memory GCS fake."""

import io
import json

import pandas as pd

from src.modules.tools import company_lookup as cl

BUCKET = 'cc-matcher-bucket-jeg-v1'
ACME = 'Acme Robotics||https://acme.com'


def _put_parquet(storage, name, df):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    storage.bucket(BUCKET).blob(name).upload_from_string(buf.getvalue())


def _seed(storage):
    _put_parquet(storage, 'data/all-contacts/clients/c.parquet', pd.DataFrame({
        'company_name': ['Acme Robotics'], 'companyWebsite': ['https://acme.com'],
        'summary': ['We build autonomous robots.']}))
    _put_parquet(storage, 'data/all-contacts/prospects/p.parquet', pd.DataFrame({
        'company_name': ['Beta Systems'], 'companyWebsite': ['https://beta.com'],
        'summary': ['Beta does sensors.']}))
    storage.bucket(BUCKET).blob('drive-sync-configs/assignments.json') \
        .upload_from_string(json.dumps({'assignments': {
            'f1': {'client_key': ACME, 'folder_name': 'Acme Robotics_INTERNAL',
                   'match_type': 'manual'}}}))


def test_search_finds_client_with_assignment(storage):
    _seed(storage)
    res = cl.search_companies(storage, 'Acme Robotics')['results']
    top = res[0]
    assert top['company_key'] == ACME
    assert top['pool'] == 'clients'
    assert top['has_drive_assignment'] is True
    assert top['score'] >= 0.9


def test_hide_assignment_for_shadow_eval(storage):
    _seed(storage)
    res = cl.search_companies(storage, 'Acme', hide_assignment_for={ACME})['results']
    assert res[0]['has_drive_assignment'] is False   # hidden from the eval


def test_search_finds_prospect(storage):
    _seed(storage)
    res = cl.search_companies(storage, 'Beta Systems')['results']
    assert res[0]['pool'] == 'prospects'
    assert res[0]['has_drive_assignment'] is False


def test_get_company(storage):
    _seed(storage)
    c = cl.get_company(storage, ACME)
    assert c['found'] is True
    assert c['summary'].startswith('We build')
    assert c['pool'] == 'clients'
    assert c['folder_assignments'][0]['folder_id'] == 'f1'


def test_get_company_unknown(storage):
    _seed(storage)
    assert cl.get_company(storage, 'Nobody||x')['found'] is False
