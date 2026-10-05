"""In-memory stand-ins for GCS, shared by the intake tests."""

import pytest
from google.api_core.exceptions import NotFound, PreconditionFailed


class FakeBlob:
    def __init__(self, store, name):
        self.store, self.name = store, name

    @property
    def generation(self):
        rec = self.store.get(self.name)
        return rec['gen'] if rec else None

    @property
    def size(self):
        rec = self.store.get(self.name)
        return len(rec['data']) if rec else None

    def upload_from_string(self, data, content_type=None, if_generation_match=None):
        if isinstance(data, str):
            data = data.encode('utf-8')
        cur = self.store.get(self.name)
        if if_generation_match is not None:
            if (if_generation_match == 0 and cur) or \
               (if_generation_match and (not cur or cur['gen'] != if_generation_match)):
                raise PreconditionFailed('generation mismatch')
        self.store[self.name] = {'data': data, 'gen': (cur['gen'] + 1 if cur else 1)}

    def download_as_bytes(self, start=None, end=None):
        if self.name not in self.store:
            raise NotFound(self.name)
        data = self.store[self.name]['data']
        return data if start is None else data[start:end + 1]

    def exists(self):
        return self.name in self.store

    def delete(self):
        self.store.pop(self.name, None)


class FakeBucket:
    def __init__(self, store, name):
        self.store, self.name = store, name

    def blob(self, path):
        return FakeBlob(self.store, path)

    def get_blob(self, path):
        return FakeBlob(self.store, path) if path in self.store else None


class FakeStorage:
    def __init__(self):
        self.store: dict[str, dict] = {}

    def bucket(self, name):
        return FakeBucket(self.store, name)

    def list_blobs(self, bucket, prefix=''):
        return [FakeBlob(self.store, n) for n in sorted(self.store) if n.startswith(prefix)]


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
