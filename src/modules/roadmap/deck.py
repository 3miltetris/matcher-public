"""
roadmap.deck — turn matched opportunities into a branded roadmap PDF.

Two steps:
  1. generate_narrative() — one Claude call writes the short "why now" intro and a
     grounded what-it-is / why-it-fits line per opportunity. House style follows
     the BW&CO funding-deck skill: no em dashes, no hype words, and nothing
     invented — in particular NO dollar amounts, because the topic store does not
     carry verified Phase I/II ceilings and a roadmap must not claim figures it
     cannot source.
  2. render_pdf() — a jinja HTML template + brand CSS, rendered by weasyprint.
     weasyprint is imported lazily so the narrative step (and its tests) need no
     native dependency.

Brand tokens (INK/NAVY/…) and the text wordmark are lifted from the skill's
build_deck.py, so the PDF reads as BW&CO even though it is a lighter artifact
than the consultant deck.
"""

import json
import re
from datetime import date, datetime
from pathlib import Path

import src.modules.profile_build as pb

TEMPLATE_DIR = Path(__file__).parent / 'templates'

DECK_MODEL = 'claude-sonnet-4-6'

# Hype words the deck must not use (a subset of the skill's DEFAULT_BANNED).
_BANNED = ('well-positioned', 'well positioned', 'leverage', 'robust', 'seamless',
           'cutting-edge', 'cutting edge', 'exciting opportunity', 'best-in-class',
           'synergy', 'game-changing', 'world-class', 'revolutionary')

_SUMMARY_CAP = 600          # per-opportunity grant text handed to Claude

_DEADLINE_COLS = ('due_date', 'close_date', 'deadline')
_DATE_FORMATS  = ('%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%m-%d-%Y')


# ── House-style cleanup ──────────────────────────────────────────────────────

def clean_copy(text) -> str:
    """Drop em/en dashes (the skill forbids them) and collapse whitespace. Not a
    rewrite of banned phrases — the prompt handles those; this only fixes the
    one thing a model reliably slips in."""
    s = str(text or '').strip()
    s = s.replace(' — ', ', ').replace('—', ', ').replace(' – ', ', ').replace('–', '-')
    return re.sub(r'\s+', ' ', s).strip()


def banned_hits(text: str) -> list[str]:
    low = text.lower()
    return [w for w in _BANNED if w in low]


# ── Verified ceiling display ─────────────────────────────────────────────────
# The roadmap shows a dollar figure ONLY from our own verified data, rendered in a
# chip separate from the LLM copy — the model is never asked to state an amount, so
# "nothing invented" still holds. A notice-specific figure (Grants.gov structured
# or stated in the solicitation) is shown exactly; a program/agency table figure is
# shown hedged as "typical", and only when `show_table_ceilings` is on.

def _fmt_usd(value) -> str:
    """A compact USD string: 250000 -> "$250K", 1_800_000 -> "$1.8M". '' for a
    blank / NaN / non-positive value."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ''
    if v != v or v <= 0:                 # NaN or non-positive
        return ''
    if v >= 1_000_000:
        return f'${v / 1_000_000:.1f}M'.replace('.0M', 'M')
    if v >= 1_000:
        return f'${round(v / 1_000)}K'
    return f'${int(round(v))}'


def ceiling_display(opp: dict, *, show_table_ceilings: bool = True) -> str:
    """The verified ceiling chip text for one opportunity, or '' to show nothing.

    - 'grants_gov' / 'stated_text' → the exact notice figure ("Award ceiling: $500K")
    - 'agency_table' → the hedged program caps ("Phase I up to ~$250K, Phase II up
      to ~$1.8M (program typical)"), gated behind `show_table_ceilings`
    - anything else / no figure → ''
    """
    source = str(opp.get('ceiling_source') or '')
    if source in ('grants_gov', 'stated_text'):
        amount = _fmt_usd(opp.get('award_ceiling_num'))
        if amount:
            return f'Award ceiling: {amount}'
    if source == 'agency_table' and show_table_ceilings:
        parts = []
        p1 = _fmt_usd(opp.get('phase1_ceiling_num'))
        p2 = _fmt_usd(opp.get('phase2_ceiling_num'))
        if p1:
            parts.append(f'Phase I up to ~{p1}')
        if p2:
            parts.append(f'Phase II up to ~{p2}')
        if parts:
            return ', '.join(parts) + ' (program typical)'
    return ''


# ── Narrative generation ─────────────────────────────────────────────────────

def _build_system() -> str:
    return (
        'You are a non-dilutive funding strategist at BW&CO. You are given one '
        "company's capability profile and a shortlist of federal funding "
        'opportunities a matching engine scored as fits. Write concise, factual '
        'copy for a one-page funding roadmap the founder will read.\n\n'
        'Rules:\n'
        '1. Never invent facts. In particular NEVER state a dollar amount, award '
        'ceiling, or deadline — those are shown separately from verified data.\n'
        '2. No em dashes. Use commas, colons or parentheses.\n'
        '3. No hype words (well-positioned, leverage, robust, seamless, '
        'cutting-edge, exciting opportunity, best-in-class, synergy). State a fact '
        'instead.\n'
        '4. Every "why_it_fits" must name something SPECIFIC about this company '
        '(a named capability, technology or domain from its profile), not a '
        'generic claim.\n'
        '5. "what_it_is" is one plain sentence on what the program funds.\n\n'
        'Return ONLY a JSON object of this exact shape:\n'
        '{"why_now": "<=2 sentences on why pursuing non-dilutive funding now makes '
        'sense for this company, or an empty string>", '
        '"items": [{"i": <the opportunity number as given>, '
        '"what_it_is": "<=1 sentence", "why_it_fits": "<=1 sentence>"}]}'
    )


def _build_user(company_name: str, profile_summary: str, aspect_labels: str,
                opportunities: list[dict]) -> str:
    lines = [f'COMPANY: {company_name}', '', f'PROFILE: {profile_summary}']
    if aspect_labels:
        lines += ['', f'CAPABILITIES: {aspect_labels}']
    lines += ['', 'OPPORTUNITIES:']
    for n, o in enumerate(opportunities):
        title   = str(o.get('title') or o.get('topic_number') or 'Untitled')
        agency  = str(o.get('agency') or o.get('broad_agency') or '')
        summary = str(o.get('grant_summary') or '')[:_SUMMARY_CAP]
        matched = str(o.get('aspect_label') or '')
        lines.append(f'[{n}] {title} ({agency})')
        if matched:
            lines.append(f'    matched capability: {matched}')
        if summary:
            lines.append(f'    about: {summary}')
    return '\n'.join(lines)


def _parse_narrative(raw: str) -> dict:
    """Extract and validate the narrative JSON. Returns {why_now, items} where
    items is {i: {what_it_is, why_it_fits}}."""
    m = re.search(r'\{.*\}', raw, re.S)
    if not m:
        raise ValueError('no JSON object in response')
    obj = json.loads(m.group(0))
    items = {}
    for it in obj.get('items') or []:
        if not isinstance(it, dict) or 'i' not in it:
            continue
        try:
            idx = int(it['i'])
        except (TypeError, ValueError):
            continue
        items[idx] = {
            'what_it_is':  clean_copy(it.get('what_it_is')),
            'why_it_fits': clean_copy(it.get('why_it_fits')),
        }
    return {'why_now': clean_copy(obj.get('why_now')), 'items': items}


def generate_narrative(anth, *, company_name: str, profile_summary: str,
                       aspect_labels: str, opportunities: list[dict],
                       model: str = DECK_MODEL) -> tuple[dict, list[str]]:
    """({why_now, items}, warnings). One Claude call, refusal-falls-back to the
    other model via the shared claude_json. Never raises for an empty shortlist —
    returns empty narrative so the deck can still render the raw opportunities."""
    warnings: list[str] = []
    if not opportunities:
        return {'why_now': '', 'items': {}}, warnings
    try:
        parsed, built_by = pb.claude_json(
            anth, model, _build_system(),
            _build_user(company_name, profile_summary, aspect_labels, opportunities),
            _parse_narrative,
        )
    except ValueError as e:
        # The narrative is a nice-to-have; a failure must not lose the roadmap.
        return {'why_now': '', 'items': {}}, [f'narrative generation failed: {e}']
    if built_by != model:
        warnings.append(f'{model} declined; narrative written by {built_by}.')
    hits = banned_hits(parsed['why_now']) + [
        w for v in parsed['items'].values()
        for w in banned_hits(v['what_it_is'] + ' ' + v['why_it_fits'])
    ]
    if hits:
        warnings.append('narrative used discouraged phrases: ' + ', '.join(sorted(set(hits))))
    return parsed, warnings


# ── Deadline display ─────────────────────────────────────────────────────────

def _parse_date(value) -> date | None:
    s = str(value or '').strip()
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s[:10], fmt).date()
        except ValueError:
            pass
    return None


def deadline_display(opp: dict, today: date | None = None) -> tuple[str, str]:
    """(label, countdown) for one opportunity. Honest: a rolling/blank deadline
    reads 'Rolling' / 'Open until filled', never a fabricated date."""
    if str(opp.get('is_rolling')).lower() in ('true', '1', 'yes'):
        return 'Rolling', 'Open until filled'
    for col in _DEADLINE_COLS:
        d = _parse_date(opp.get(col))
        if d:
            today = today or date.today()
            days  = (d - today).days
            cd = f'{days} days out' if days >= 0 else 'Closed'
            return d.strftime('%d %b %Y'), cd
    return 'Deadline to confirm', ''


# ── Context assembly + render ────────────────────────────────────────────────

def build_context(*, company_name: str, profile_summary: str,
                  opportunities: list[dict], narrative: dict,
                  presentation_date: date | None = None,
                  show_table_ceilings: bool = True) -> dict:
    """The template context. One item per opportunity, narrative merged in by
    index, deadlines resolved honestly, and a verified ceiling chip from our own
    data (never from the LLM). `show_table_ceilings` gates the program-typical
    (agency-table) figures so they can be suppressed per client/event; exact
    notice figures (Grants.gov / stated) always show."""
    today = presentation_date or date.today()
    items = narrative.get('items') or {}
    opps = []
    for n, o in enumerate(opportunities):
        label, countdown = deadline_display(o, today)
        nar = items.get(n, {})
        opps.append({
            'number':      n + 1,
            'title':       clean_copy(o.get('title') or o.get('topic_number') or 'Opportunity'),
            'agency':      clean_copy(o.get('agency') or o.get('broad_agency') or ''),
            'topic_number': str(o.get('topic_number') or ''),
            'deadline':    label,
            'countdown':   countdown,
            'ceiling':     ceiling_display(o, show_table_ceilings=show_table_ceilings),
            'what_it_is':  nar.get('what_it_is') or clean_copy(str(o.get('grant_summary') or '')[:280]),
            'why_it_fits': nar.get('why_it_fits') or '',
            'matched':     clean_copy(o.get('aspect_label') or ''),
            'url':         str(o.get('source') or ''),
        })
    return {
        'company_name':    company_name,
        'profile_summary': clean_copy(profile_summary),
        'why_now':         narrative.get('why_now') or '',
        'opportunities':   opps,
        'generated':       today.strftime('%B %Y'),
        'firm_url':        'www.bwcoconsulting.com',
    }


def render_html(context: dict) -> str:
    """Render the roadmap template to an HTML string (jinja only, no native
    deps). Split out so the template is testable and previewable without
    weasyprint's GTK/pango libraries."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(['html']),
    )
    return env.get_template('roadmap.html.j2').render(**context)


def render_pdf(context: dict) -> bytes:
    """Render the roadmap to PDF bytes. weasyprint is imported here so the rest
    of this module loads without its native dependencies."""
    from weasyprint import HTML
    return HTML(string=render_html(context), base_url=str(TEMPLATE_DIR)).write_pdf()
