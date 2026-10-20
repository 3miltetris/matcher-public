"""GCS fixtures for the sync / concurrency tests (shared fake in tests/gcs_fake.py)."""

import pytest

from tests.gcs_fake import FakeStorage


@pytest.fixture
def storage():
    return FakeStorage()
