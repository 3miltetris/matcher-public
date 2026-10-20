"""In-memory stand-ins for Google Cloud Storage, shared across test suites.

Models the one thing the generation-guarded write paths depend on: a per-blob
integer generation that increments on every write, `if_generation_match`
enforcement (including `=0` create-only) raising PreconditionFailed, and
download raising NotFound when absent. Extracted from tests/intake/conftest.py
so tests/sync/ can reuse it; both conftests import from here.
"""

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
