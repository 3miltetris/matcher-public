"""
validation.py — visibility conditions and answer validation for the intake
form. The frontend (static/app.js) evaluates the same condition structure, so
keep is_visible() and its JS twin in step.
"""

import re

from src.modules.intake.schema import FIELDS, Field

_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
MAX_TEXT  = 500
MAX_LONG  = 8000


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def is_visible(f: Field, answers: dict) -> bool:
    """A field with no condition is always visible; otherwise any clause whose
    referenced answer intersects its `any_of` makes it visible."""
    if not f.condition:
        return True
    for clause in f.condition:
        if set(_as_list(answers.get(clause['field']))) & set(clause['any_of']):
            return True
    return False


def clean_answers(answers: dict, fields: list[Field] = FIELDS) -> dict:
    """Keep only known, visible, non-file fields; strip strings; multi-selects
    become lists. Hidden answers are dropped so a founder who changes their
    sector cannot leave stale 2A answers behind."""
    out: dict = {}
    for f in fields:
        if f.type == 'file' or f.id not in answers or not is_visible(f, answers):
            continue
        val = answers[f.id]
        if f.type == 'multi_select':
            vals = _as_list(val)
            if vals:
                out[f.id] = vals
        else:
            text = '' if val is None else str(val).strip()
            if text:
                out[f.id] = text
    return out


def validate(answers: dict, fields: list[Field] = FIELDS) -> tuple[dict, dict[str, str]]:
    """(clean answers, {field_id: error}). Empty errors = valid."""
    clean  = clean_answers(answers, fields)
    errors: dict[str, str] = {}
    for f in fields:
        if f.type == 'file' or not is_visible(f, answers):
            continue
        val = clean.get(f.id)
        if val in (None, '', []):
            if f.required:
                errors[f.id] = 'Required'
            continue
        if f.type == 'single_select' and val not in f.options:
            errors[f.id] = 'Choose one of the listed options'
        elif f.type == 'multi_select':
            bad = [v for v in val if v not in f.options]
            if bad:
                errors[f.id] = f'Unknown option(s): {", ".join(bad)}'
        elif f.type == 'email' and not _EMAIL_RE.match(val):
            errors[f.id] = 'Enter a valid email address'
        elif f.type == 'url' and ('.' not in val or ' ' in val):
            errors[f.id] = 'Enter a valid web address'
        elif f.type == 'textarea' and len(val) > MAX_LONG:
            errors[f.id] = f'Keep this under {MAX_LONG} characters'
        elif f.type in ('text', 'email', 'phone', 'url') and len(val) > MAX_TEXT:
            errors[f.id] = f'Keep this under {MAX_TEXT} characters'
    return clean, errors
