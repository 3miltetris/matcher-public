"""
notify.py — email alerts from the intake service (the SMTP pattern of
views/bulk_matching.py, reading env vars instead of st.secrets).

Messages carry session ids, company names and step names only — never answer
contents or document text.
"""

import logging
import os
import smtplib
from email.mime.text import MIMEText

log = logging.getLogger('intake')


def recipients() -> list[str]:
    raw = os.environ.get('INTAKE_NOTIFY_TO', 'john@bwcoconsulting.com')
    return [r.strip() for r in raw.split(',') if r.strip()]


def send(subject: str, body: str) -> bool:
    """Best-effort: a failed alert is logged, never raised."""
    user = os.environ.get('SMTP_USER')
    pw   = os.environ.get('SMTP_PASSWORD')
    to   = recipients()
    if not (user and pw and to):
        log.warning('notify: SMTP not configured; dropped %r', subject)
        return False
    msg = MIMEText(body, 'plain')
    msg['From'], msg['To'], msg['Subject'] = user, ', '.join(to), subject
    try:
        with smtplib.SMTP(os.environ.get('SMTP_HOST', 'smtp.gmail.com'),
                          int(os.environ.get('SMTP_PORT', 587)), timeout=30) as s:
            s.ehlo()
            s.starttls()
            s.login(user, pw)
            s.sendmail(user, to, msg.as_string())
        return True
    except Exception as e:
        log.error('notify: send failed for %r: %s', subject, e)
        return False


def submission_received(session_id: str, company: str, links: dict) -> bool:
    lines = [f'New due-diligence intake: {company}', '', f'Session: {session_id}']
    for label, url in links.items():
        if url:
            lines.append(f'{label}: {url}')
    return send(f'DD intake submitted: {company}', '\n'.join(lines))


def step_failed(session_id: str, company: str, step: str, error: str) -> bool:
    return send(
        f'DD intake pipeline failed: {company} ({step})',
        f'Session: {session_id}\nStep: {step}\nError: {error[:500]}\n\n'
        f'Resume with:\n  python -m src.modules.intake.submit_pipeline resume {session_id}\n',
    )
