"""Fixtures + in-memory fakes for the roadmap tests."""

import pytest

from tests.gcs_fake import FakeStorage  # noqa: F401


@pytest.fixture
def storage():
    return FakeStorage()


@pytest.fixture
def bm(storage):
    from src.modules.GoogleBucketManager.bucket_manager import BucketManager
    return BucketManager('test-bucket', client=storage)


def _present(props: dict, key: str) -> bool:
    return props.get(key) not in (None, '')


class _FakeResp:
    def __init__(self, ok=True, data=None, status=200, text=''):
        self.ok, self._data, self.status_code, self.text = ok, data or {}, status, text

    def json(self):
        return self._data


class _FakeHttp:
    """Just enough of a requests.Session for HubSpot Files uploads."""

    def __init__(self, outer):
        self.outer = outer

    def post(self, url, headers=None, files=None, timeout=None):
        self.outer.file_uploads.append({'url': url, 'files': files})
        if self.outer.files_fail:
            return _FakeResp(ok=False, status=500, text='boom')
        return _FakeResp(data={'url': self.outer.files_url, 'id': 'file-123'})


class FakeHubSpot:
    """Duck-typed stand-in for intake.hubspot_sync.HubSpot — the methods the
    roadmap code calls, backed by an in-memory set of contacts (keyed by id)."""

    def __init__(self, contact_id: str | None = None, properties: dict | None = None):
        self.token = 'test-token'
        self.contacts: dict[str, dict] = {}
        self.existing_props: set[str] = set()
        self.created_props: list[str] = []
        self.patched: dict = {}                 # last PATCH's properties (back-compat)
        self.patched_by_id: dict[str, dict] = {}
        # HubSpot Files upload surface
        self.http = _FakeHttp(self)
        self.file_uploads: list[dict] = []
        self.files_fail = False
        self.files_url = 'https://files.hubspotusercontent.net/hubfs/roadmap.pdf'
        if contact_id is not None:
            self.add(contact_id, properties or {})

    def add(self, contact_id: str, properties: dict) -> 'FakeHubSpot':
        self.contacts[contact_id] = dict(properties)
        self.existing_props |= set(properties)
        return self

    # back-compat accessors used by single-contact tests
    @property
    def _id(self):
        return next(iter(self.contacts), None)

    @property
    def _props(self):
        return self.contacts.get(self._id, {})

    def _req(self, method: str, path: str, **kw):
        if method == 'GET' and path == '/crm/v3/properties/contacts':
            return {'results': [{'name': n} for n in self.existing_props]}
        if method == 'POST' and path == '/crm/v3/properties/contacts':
            name = kw['json']['name']
            self.created_props.append(name)
            self.existing_props.add(name)
            return {}
        if method == 'POST' and path == '/crm/v3/objects/contacts/search':
            return {'results': self._run_search(kw['json'])}
        if method == 'GET' and path.startswith('/crm/v3/objects/contacts/'):
            cid = path.rsplit('/', 1)[1]
            return {'id': cid, 'properties': dict(self.contacts.get(cid, {}))}
        if method == 'PATCH' and path.startswith('/crm/v3/objects/contacts/'):
            cid = path.rsplit('/', 1)[1]
            props = dict(kw['json']['properties'])
            self.contacts.setdefault(cid, {}).update(props)
            self.patched = props
            self.patched_by_id.setdefault(cid, {}).update(props)
            return {}
        raise AssertionError(f'unexpected HubSpot call: {method} {path}')

    def _search(self, plural, prop, value, props):
        if plural == 'contacts' and prop == 'email':
            for cid, p in self.contacts.items():
                if str(p.get('email', '')).lower() == str(value).lower():
                    return {'id': cid, 'properties': dict(p)}
        return None

    def _run_search(self, body: dict) -> list[dict]:
        def match(p, f):
            op, key = f['operator'], f['propertyName']
            if op == 'HAS_PROPERTY':
                return _present(p, key)
            if op == 'NOT_HAS_PROPERTY':
                return not _present(p, key)
            if op == 'EQ':
                return str(p.get(key, '')) == str(f.get('value'))
            return True
        groups = body.get('filterGroups') or []
        out = []
        for cid, p in self.contacts.items():
            ok = (any(all(match(p, f) for f in g.get('filters', [])) for g in groups)
                  if groups else True)
            if ok:
                out.append({'id': cid, 'properties': dict(p)})
        return out[:body.get('limit', 100)]


@pytest.fixture
def hubspot():
    return FakeHubSpot('c123', {
        'email': 'ada@acme.com', 'firstname': 'Ada', 'lastname': 'Lovelace',
        'company': 'Acme Robotics', 'website': 'acme.com',
        'event_company_description': 'MEMS gas sensors for confined-space monitoring.',
    })
