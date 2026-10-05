"""
schema_export.py — every shape derived from schema.FIELDS.

  frontend_schema()      → GET /api/schema
  answer_sections()      → Google Doc + digest layout (labels by section)
  digest()               → intake_summary on the prospect rows
  profile_extracted()    → intake_data['extracted'] (capability + notable_updates)
  embedding_text()       → text embedded into the prospect row's `embeddings`
  hubspot_properties()   → {property: value} for one HubSpot object
  hubspot_property_defs()→ custom properties to ensure exist
"""

from src.modules.intake.schema import (FIELDS, SECTIONS, TECH_DESCRIPTION,
                                       Field)
from src.modules.intake.validation import is_visible

_FRONTEND_KEYS = ('id', 'label', 'help_text', 'type', 'options', 'required',
                  'condition')

# HubSpot properties every intake writes, beyond the schema's own map.
DD_PROPERTIES = [
    # (name, label, type, fieldType)
    ('dd_submitted_at',       'DD intake submitted at',  'datetime', 'date'),
    ('dd_drive_folder_url',   'DD Drive folder',         'string',   'text'),
    ('dd_matcher_profile_id', 'Matcher company key',     'string',   'text'),
    ('dd_intake_source',      'DD intake source',        'string',   'text'),
]
INTAKE_SOURCE = 'custom_intake_v1'

def frontend_schema(fields: list[Field] = FIELDS) -> dict:
    by_section: dict[str, list[dict]] = {}
    for f in fields:
        d = f.to_dict()
        by_section.setdefault(f.section, []).append({k: d[k] for k in _FRONTEND_KEYS})
    return {'sections': [
        {'id': sid, 'title': title, 'fields': by_section[sid]}
        for sid, title in SECTIONS if sid in by_section
    ]}


def answer_text(value) -> str:
    if isinstance(value, (list, tuple)):
        return ', '.join(str(v) for v in value)
    return '' if value is None else str(value)


def answer_sections(answers: dict, fields: list[Field] = FIELDS,
                    roles: tuple[str, ...] | None = None
                    ) -> list[tuple[str, list[tuple[str, str]]]]:
    """[(section title, [(label, answer text)])] for answered, visible fields,
    in schema order. `roles` restricts to those profile_roles."""
    out = []
    for sid, title in SECTIONS:
        rows = []
        for f in fields:
            if f.section != sid or f.type == 'file' or not is_visible(f, answers):
                continue
            if roles is not None and f.profile_role not in roles:
                continue
            text = answer_text(answers.get(f.id))
            if text:
                rows.append((f.label, text))
        if rows:
            out.append((title, rows))
    return out


def digest(answers: dict, fields: list[Field] = FIELDS) -> str:
    """Plain-text digest of the profile-relevant answers (capability +
    intention), by section. Eligibility/personal answers are left out: this
    column is shown in the Matcher and fed to the aspect builder."""
    blocks = []
    for title, rows in answer_sections(answers, fields, roles=('capability', 'intention')):
        blocks.append(title + ':\n' + '\n'.join(f'- {label}: {text}' for label, text in rows))
    return '\n\n'.join(blocks)


def profile_extracted(answers: dict, fields: list[Field] = FIELDS) -> dict:
    """intake_data['extracted']: capability answers keyed by field id, and
    intention answers as `label: answer` lines under notable_updates."""
    extracted: dict = {}
    updates: list[str] = []
    for f in fields:
        if f.type == 'file' or not is_visible(f, answers):
            continue
        text = answer_text(answers.get(f.id))
        if not text:
            continue
        if f.profile_role == 'capability':
            extracted[f.id] = answers[f.id]
        elif f.profile_role == 'intention':
            updates.append(f'{f.label}: {text}')
    if updates:
        extracted['notable_updates'] = updates
    return extracted


def embedding_text(answers: dict, fields: list[Field] = FIELDS) -> str:
    """The prospect row's `summary` — what Bulk Matching embeds. Technology
    description first, then the other capability answers."""
    parts = [answer_text(answers.get(TECH_DESCRIPTION))]
    for _, rows in answer_sections(answers, fields, roles=('capability',)):
        for label, text in rows:
            if text != parts[0]:
                parts.append(f'{label}: {text}')
    return '\n'.join(p for p in parts if p)


def hubspot_properties(answers: dict, obj: str = 'company',
                       fields: list[Field] = FIELDS) -> dict[str, str]:
    """{property: value} for the mapped, answered, visible fields of one
    object type, in HubSpot's own option values (the form's coded values, not
    the labels). Multi-selects are `;`-joined, HubSpot's checkbox format."""
    out: dict[str, str] = {}
    for f in fields:
        if not f.hubspot_property or f.hubspot_object != obj:
            continue
        if not is_visible(f, answers) or f.id not in answers:
            continue
        val = answers[f.id]
        if isinstance(val, (list, tuple)):
            out[f.hubspot_property] = ';'.join(f.hubspot_value(v) for v in val)
        else:
            out[f.hubspot_property] = f.hubspot_value(str(val))
    return out


def hubspot_property_defs(obj: str = 'company', fields: list[Field] = FIELDS
                          ) -> list[tuple[str, str, str, str]]:
    """Custom properties the intake CREATES: only its own dd_* tracking
    properties. Answers go to the properties the HubSpot form already writes,
    which exist and are never created or redefined from here."""
    return list(DD_PROPERTIES) if obj == 'company' else []


def owned_properties(obj: str = 'company', fields: list[Field] = FIELDS) -> set[str]:
    """Properties the intake may OVERWRITE on an existing record: its dd_*
    properties plus every property the DD form itself writes (a form
    submission overwrote them too). Anything else — including the company's
    standard name/domain — is staff-maintained: hubspot_sync sets it on create
    and only fills it where blank."""
    props = {name for name, *_ in hubspot_property_defs(obj, fields)}
    props |= {f.hubspot_property for f in fields
              if f.hubspot_property and f.hubspot_object == obj}
    return props
