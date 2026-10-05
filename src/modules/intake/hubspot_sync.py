"""
hubspot_sync.py — upsert the intake's HubSpot company + contact and associate
them. Streamlit-free, plain requests (the repo's HubSpot code is otherwise
inline in views and import-API only).

Ownership rules (DD_INTAKE_PLAN §8):
  * A record is created with every mapped property.
  * On an EXISTING record only the intake's own custom properties
    (schema_export.owned_properties) are overwritten. Standard properties —
    name, domain, firstname, phone, … — are staff-maintained and are only
    filled where currently blank.

Writing through the CRM API means no form-submission event fires; workflows
that used the old form trigger on `dd_submitted_at is known` instead.

Token scopes: crm.objects.companies.read/write, crm.objects.contacts.read/
write, crm.schemas.companies.write, crm.schemas.contacts.write.
"""

import os
import time

import requests

from src.modules.intake import schema as sc
from src.modules.intake import schema_export as sx

API = 'https://api.hubapi.com'


class HubSpotError(RuntimeError):
    pass


class HubSpot:
    def __init__(self, token: str | None = None, session=None):
        self.token = token or os.environ['HUBSPOT_API_KEY']
        self.http  = session or requests.Session()
        self._ensured: set[str] = set()

    # ── transport ────────────────────────────────────────────────────────

    def _req(self, method: str, path: str, retries: int = 4, **kw) -> dict:
        for attempt in range(retries + 1):
            r = self.http.request(
                method, API + path, timeout=30,
                headers={'Authorization': f'Bearer {self.token}',
                         'Content-Type': 'application/json'}, **kw)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(float(r.headers.get('Retry-After') or 2 ** attempt))
                continue
            if not r.ok:
                raise HubSpotError(f'{method} {path} → {r.status_code}: {r.text[:300]}')
            return r.json() if r.content else {}
        raise HubSpotError(f'{method} {path}: retries exhausted')

    # ── properties ───────────────────────────────────────────────────────

    def ensure_properties(self, obj: str) -> list[str]:
        """Create any missing intake custom properties on companies/contacts.
        Cached per instance. Returns the names created."""
        if obj in self._ensured:
            return []
        plural  = 'companies' if obj == 'company' else 'contacts'
        group   = 'companyinformation' if obj == 'company' else 'contactinformation'
        have    = {p['name'] for p in self._req('GET', f'/crm/v3/properties/{plural}')
                   .get('results', [])}
        created = []
        for name, label, ptype, field_type in sx.hubspot_property_defs(obj):
            if name in have:
                continue
            self._req('POST', f'/crm/v3/properties/{plural}', json={
                'name': name, 'label': label, 'type': ptype,
                'fieldType': field_type, 'groupName': group})
            created.append(name)
        self._ensured.add(obj)
        return created

    # ── objects ──────────────────────────────────────────────────────────

    def _search(self, plural: str, prop: str, value: str, props: list[str]) -> dict | None:
        res = self._req('POST', f'/crm/v3/objects/{plural}/search', json={
            'filterGroups': [{'filters': [
                {'propertyName': prop, 'operator': 'EQ', 'value': value}]}],
            'properties': props, 'limit': 2})
        results = res.get('results', [])
        return results[0] if results else None

    def _upsert(self, plural: str, existing: dict | None, wanted: dict[str, str],
                owned: set[str], record_id: str | None = None) -> tuple[str, bool]:
        wanted = {k: v for k, v in wanted.items() if v not in (None, '')}
        if record_id is None and existing is None:
            res = self._req('POST', f'/crm/v3/objects/{plural}', json={'properties': wanted})
            return res['id'], True
        rid     = record_id or existing['id']
        current = (existing or {}).get('properties') or {}
        patch   = {k: v for k, v in wanted.items()
                   if k in owned or not str(current.get(k) or '').strip()}
        if patch:
            self._req('PATCH', f'/crm/v3/objects/{plural}/{rid}', json={'properties': patch})
        return rid, False

    def upsert_company(self, answers: dict, domain: str, extra: dict,
                       known_id: str | None = None) -> tuple[str, bool]:
        wanted = {'name': answers.get(sc.COMPANY_NAME, ''), 'domain': domain,
                  **sx.hubspot_properties(answers, 'company'), **extra}
        existing = None
        if known_id:   # the id a previous intake stored — may since have been merged/deleted
            try:
                existing = self._req('GET', f'/crm/v3/objects/companies/{known_id}',
                                     params={'properties': ','.join(wanted)})
            except HubSpotError:
                existing = None
        if existing is None:
            existing = self._search('companies', 'domain', domain, list(wanted))
        return self._upsert('companies', existing, wanted,
                             sx.owned_properties('company'))

    def upsert_contact(self, answers: dict, extra: dict | None = None) -> tuple[str, bool]:
        email  = str(answers.get(sc.CONTACT_EMAIL, '')).strip().lower()
        wanted = {**sx.hubspot_properties(answers, 'contact'), 'email': email,
                  **(extra or {})}
        existing = self._search('contacts', 'email', email, list(wanted))
        return self._upsert('contacts', existing, wanted, sx.owned_properties('contact'))

    def associate(self, company_id: str, contact_id: str) -> None:
        self._req('PUT', f'/crm/v4/objects/companies/{company_id}'
                         f'/associations/default/contacts/{contact_id}')


def sync(answers: dict, domain: str, extra_company: dict, known_company_id=None,
         client: HubSpot | None = None) -> dict:
    """Full upsert. Idempotent: a retry finds the records it created by
    domain/email (or by the stored id) and re-applies the same values."""
    hs = client or HubSpot()
    hs.ensure_properties('company')
    hs.ensure_properties('contact')
    company_id, c_new = hs.upsert_company(answers, domain, extra_company, known_company_id)
    contact_id, p_new = hs.upsert_contact(answers)
    hs.associate(company_id, contact_id)
    return {'hubspot_company_id': company_id, 'hubspot_contact_id': contact_id,
            'company_created': c_new, 'contact_created': p_new}
