"""
gdoc_writer.py — the founder's responses as HTML, laid out by schema section,
for drive_folder.write_doc() to upload as a Google Doc. Every answered,
visible question is included (eligibility too — this is the staff record).
"""

import html

from src.modules.intake import schema as sc
from src.modules.intake import schema_export as sx


def doc_title(answers: dict, submitted_at: str) -> str:
    return (f'Due Diligence Responses — {answers.get(sc.COMPANY_NAME, "").strip()} '
            f'— {submitted_at[:10]}')


def render_html(answers: dict, submitted_at: str, uploads: list[dict] = (),
                ai_accepted: set[str] = frozenset()) -> str:
    """`ai_accepted` (Phase 2) names labels whose value came from the founder's
    documents and was accepted unchanged; they are annotated."""
    e = html.escape
    parts = [f'<h1>{e(doc_title(answers, submitted_at))}</h1>',
             f'<p><i>Submitted {e(submitted_at)} via the BW&amp;CO intake form.</i></p>']
    for title, rows in sx.answer_sections(answers):
        parts.append(f'<h2>{e(title)}</h2>')
        for label, text in rows:
            note = ' <i>(suggested from documents)</i>' if label in ai_accepted else ''
            body = '<br>'.join(e(line) for line in text.splitlines())
            parts.append(f'<p><b>{e(label)}</b>{note}<br>{body}</p>')
    files = [u['filename'] for u in uploads if not u.get('rejected')]
    if files:
        parts.append('<h2>Documents</h2><ul>'
                     + ''.join(f'<li>{e(f)}</li>' for f in files) + '</ul>')
    return '<html><body>' + '\n'.join(parts) + '</body></html>'
