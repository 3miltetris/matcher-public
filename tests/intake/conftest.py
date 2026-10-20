"""GCS fixtures for the intake tests. The in-memory fake itself lives in
tests/gcs_fake.py so tests/sync/ can reuse it."""

import pytest

from tests.gcs_fake import FakeBlob, FakeBucket, FakeStorage  # noqa: F401


@pytest.fixture
def storage():
    return FakeStorage()


@pytest.fixture
def bm(storage):
    from src.modules.GoogleBucketManager.bucket_manager import BucketManager
    return BucketManager('test-bucket', client=storage)


def complete_answers(**overrides) -> dict:
    """A valid submission: every required, always-visible question answered
    (first option for selects), plus a realistic identity block."""
    from src.modules.intake.schema import FIELDS
    out = {}
    for f in FIELDS:
        if not f.required or f.condition or f.type == 'file':
            continue
        if f.type == 'multi_select':
            out[f.id] = [f.options[0]]
        elif f.type == 'single_select':
            out[f.id] = f.options[0]
        else:
            out[f.id] = f'answer to {f.id}'
    out.update({
        'company_legal_name': 'Acme Robotics, Inc.',
        'website': 'acme-robotics.com',
        'contact_first_name': 'Ada', 'contact_last_name': 'Lovelace',
        'contact_email': 'Ada@Acme-Robotics.com',
        'technology_description': 'MEMS gas sensors for confined-space monitoring.',
        'verticals': ['Robotics & Autonomous Systems'],
    })
    out.update(overrides)
    return out


@pytest.fixture
def answers():
    return complete_answers()
