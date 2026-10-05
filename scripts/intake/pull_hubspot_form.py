"""
Dump the HubSpot due-diligence form definition so schema.py can be
transcribed verbatim (labels, help text, options, required flags, conditional
logic, and the property each field writes).

Needs the `forms` scope on the private-app token in .streamlit/secrets.toml.

    python scripts/intake/pull_hubspot_form.py                 # list forms
    python scripts/intake/pull_hubspot_form.py <form_id> <out> # dump one form

Write <out> OUTSIDE the repo (the scratchpad): the dump is reference material,
not source.
"""

import json
import sys
import tomllib

import requests

API = 'https://api.hubapi.com'


def _token() -> str:
    with open('.streamlit/secrets.toml', 'rb') as f:
        return tomllib.load(f)['hubspot_api_key']


def _get(path: str, **params) -> dict:
    r = requests.get(API + path, params=params, timeout=30,
                     headers={'Authorization': f'Bearer {_token()}'})
    if not r.ok:
        sys.exit(f'{r.status_code}: {r.text[:400]}')
    return r.json()


def list_forms() -> None:
    after = None
    while True:
        res = _get('/marketing/v3/forms', limit=100, **({'after': after} if after else {}))
        for f in res.get('results', []):
            print(f"{f['id']}  {f.get('formType', ''):<10} {'(archived) ' if f.get('archived') else ''}{f.get('name')}")
        after = (res.get('paging') or {}).get('next', {}).get('after')
        if not after:
            return


def dump(form_id: str, out: str) -> None:
    form = _get(f'/marketing/v3/forms/{form_id}')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(form, f, indent=2, ensure_ascii=False)
    n = sum(len(g.get('fields', [])) for g in form.get('fieldGroups', []))
    print(f'{form.get("name")}: {n} fields → {out}')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        list_forms()
    elif len(sys.argv) == 3:
        dump(sys.argv[1], sys.argv[2])
    else:
        sys.exit(__doc__)
