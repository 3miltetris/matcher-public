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


def smtp_configured() -> bool:
    return bool(os.environ.get('SMTP_USER') and os.environ.get('SMTP_PASSWORD'))


def send(subject: str, body: str) -> bool:
    """Alert to the team. Best-effort: a failure is logged, never raised."""
    return send_to(recipients(), subject, body)


def send_to(to: list[str], subject: str, body: str) -> bool:
    """Best-effort: a failed send is logged (subject only), never raised."""
    user = os.environ.get('SMTP_USER')
    pw   = os.environ.get('SMTP_PASSWORD')
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


def sign_in_link(email: str, link: str, ttl_min: int) -> bool:
    """The founder's magic link. Carries no answer data."""
    return send_to([email], 'Your BW&CO due diligence form link', (
        'Use the link below to open the BW&CO due diligence form. Your answers '
        'save as you go, so you can come back to it with a new link at any time.\n\n'
        f'{link}\n\n'
        f'The link works once and expires in {ttl_min} minutes. If you did not '
        'ask for it, you can ignore this email.\n'))


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
