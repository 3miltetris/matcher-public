"""GCS fixtures for the MCP tool-layer tests (shared fake in tests/gcs_fake.py)."""

import io
import json

import pandas as pd
import pytest

from tests.gcs_fake import FakeStorage

BUCKET = 'cc-matcher-bucket-jeg-v1'


@pytest.fixture
def storage():
    return FakeStorage()


@pytest.fixture(autouse=True)
def _clear_contact_index_cache():
    """The contact index is cached in-process; never let one test see another's."""
    from src.modules.tools import contact_index
    contact_index.clear_cache()
    yield
    contact_index.clear_cache()


def put_parquet(storage, path: str, df: pd.DataFrame, *, bucket: str = BUCKET):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    storage.bucket(bucket).blob(path).upload_from_string(buf.getvalue())


def put_json(storage, path: str, obj, *, bucket: str = BUCKET):
    storage.bucket(bucket).blob(path).upload_from_string(
        json.dumps(obj), content_type='application/json')


def put_csv(storage, path: str, df: pd.DataFrame, *, bucket: str = BUCKET):
    storage.bucket(bucket).blob(path).upload_from_string(
        df.to_csv(index=False), content_type='text/csv')
