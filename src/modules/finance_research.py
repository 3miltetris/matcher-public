"""
Shared helpers for the Client Financials view — OpenAI Deep Research
prompt construction, response parsing/repair, digest generation, and
cost accounting.

Adapted from the findata-deep-researcher spec: one Deep Research call
per company via the OpenAI Responses API (background mode), returning
a strict-JSON object that follows FIELD_SECTIONS below. If API calls
fail with a model error, check https://platform.openai.com/docs/models
for current model identifiers — OpenAI renames these frequently.
"""

import json
import math
import re

# ── Models & pricing ────────────────────────────────────────────────────────
# The dedicated deep-research models (o3-deep-research / o4-mini-deep-research)
# were shut down 2026-07-23; OpenAI's deprecations page names gpt-5.6-sol as
# the replacement. Terra is the cheaper family member for larger batches.
# These models take the 'web_search' tool (not the old 'web_search_preview').

DEEP_RESEARCH_MODELS = ['gpt-5.6-luna', 'gpt-5.6-terra', 'gpt-5.6-sol']

# (USD per 1M input tokens, USD per 1M output tokens)
_PRICING = {
    'gpt-5.6-sol':   (5.0, 30.0),
    'gpt-5.6-terra': (2.5, 15.0),
    'gpt-5.6-luna':  (1.0, 6.0),
    # retired — kept so old run states still price correctly
    'o3-deep-research':      (10.0, 40.0),
    'o4-mini-deep-research': (2.0, 8.0),
}

# Rough pre-run estimate per company, used for the cost gate. Actual cost
# is computed from response.usage after each call completes.
EST_COST_PER_COMPANY = {
    'gpt-5.6-sol':   3.0,
    'gpt-5.6-terra': 1.5,
    'gpt-5.6-luna':  0.6,
}
EST_COST_LABEL = {
    'gpt-5.6-sol':   '~$1.50–$5 per company',
    'gpt-5.6-terra': '~$0.75–$2.50 per company',
    'gpt-5.6-luna':  '~$0.30–$1 per company',
}

# Runs whose estimated total exceeds this require an explicit confirmation
# checkbox before the Start button is enabled.
COST_CONFIRM_THRESHOLD_USD = 50.0


def response_cost_usd(model: str, usage) -> float:
    """Cost of one Deep Research response from its usage object."""
    in_rate, out_rate = _PRICING.get(model, (5.0, 30.0))
    try:
        return round(
            usage.input_tokens / 1e6 * in_rate
            + usage.output_tokens / 1e6 * out_rate,
            4,
        )
    except Exception:
        return 0.0


# ── Output schema ───────────────────────────────────────────────────────────
# Flat JSON object, one key per field. All values are strings; where a
# confidence label applies it is appended in parentheses, e.g.
# "$4.2M (Estimated)".

FIELD_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    ('Identity', [
        ('company_name_resolved', 'Resolved legal/operating company name'),
        ('uei', 'SAM.gov Unique Entity Identifier, or "Not found"'),
        ('sam_status', 'SAM.gov registration: Active / Inactive / Not found'),
        ('website_resolved', 'Primary company website URL'),
        ('naics_codes', 'Comma-separated NAICS codes'),
        ('entity_type', 'e.g. LLC, C-Corp, S-Corp, nonprofit'),
    ]),
    ('Financial Findings', [
        ('revenue_estimate', 'Annual revenue estimate (USD, bounded range OK)'),
        ('revenue_method', 'How the revenue figure was derived'),
        ('revenue_year', 'Year the revenue figure applies to'),
        ('total_venture_funding', 'Total venture/equity funding raised (USD)'),
        ('total_grant_funding', 'Total grant funding received (USD)'),
        ('estimated_valuation', 'Estimated company valuation (USD)'),
        ('valuation_method', 'How the valuation was derived'),
    ]),
    ('Federal Funding (last 3 years)', [
        ('federal_awards_count_3yr', 'Count of federal awards in the last 3 years'),
        ('federal_awards_total_3yr', 'Total federal award dollars in the last 3 years'),
        ('federal_agencies_3yr', 'Comma-separated awarding agencies'),
        ('sbir_sttr_count_3yr', 'Count of SBIR/STTR awards in the last 3 years'),
        ('sbir_sttr_total_3yr', 'Total SBIR/STTR dollars in the last 3 years'),
        ('latest_award_date', 'Date of most recent federal award'),
        ('award_detail_json', 'JSON array string of notable awards: [{"agency","program","phase","amount","date","title"}]'),
    ]),
    ('Headcount', [
        ('employee_count_current', 'Current employee count or range'),
        ('employee_count_source', 'Source of the headcount figure'),
        ('headcount_trend', 'Growing / Stable / Shrinking, with brief evidence'),
    ]),
    ('Financial Health', [
        ('customer_concentration_signal', 'Evidence of dependence on few customers/agencies'),
        ('hiring_trend_signal', 'Current job postings / hiring activity'),
        ('leadership_stability_signal', 'Recent executive turnover or stability'),
        ('litigation_bankruptcy_signal', 'Litigation, liens, or bankruptcy findings'),
        ('sam_certifications', 'SAM certifications, e.g. 8(a), WOSB, HUBZone, SDVOSB'),
    ]),
    ('Grant Activity', [
        ('grant_activity_classification', 'Novice / Active / Experienced / Dormant'),
        ('phase_progression_detected', 'Evidence of SBIR Phase I → II → III progression'),
        ('multi_agency_activity', 'Whether awards span multiple agencies'),
        ('grant_momentum', 'Accelerating / Steady / Decelerating'),
        ('estimated_proposals_per_year', 'Estimated proposals submitted per year'),
    ]),
    ('Discretionary Budget Signals', [
        ('signal_recent_grant_inflows', 'Recent grant money arriving (last 12 months)'),
        ('signal_hiring_activity', 'Active hiring as a budget signal'),
        ('signal_new_products', 'New product/service launches'),
        ('signal_conference_participation', 'Conference/trade-show participation'),
        ('signal_rd_programs', 'Active R&D programs'),
        ('signal_facility_expansion', 'Facility expansion or new locations'),
    ]),
    ('Proposal Readiness Score', [
        ('score_total', 'Total score 0–100 (sum of the five sub-scores)'),
        ('score_funding_activity', 'Sub-score /30 — federal funding activity'),
        ('score_momentum', 'Sub-score /20 — grant momentum'),
        ('score_org_capacity', 'Sub-score /20 — organizational capacity'),
        ('score_budget_signals', 'Sub-score /20 — discretionary budget signals'),
        ('score_fit_external_support', 'Sub-score /10 — fit for external proposal support'),
    ]),
    ('Qualification', [
        ('internal_proposal_capability', 'Evidence of in-house grant-writing capability'),
        ('proposal_budget_estimate', 'Estimated budget available for proposal support'),
        ('recommendation', 'Pursue / Nurture / Deprioritize, with one-line reason'),
        ('outreach_angle', 'Suggested outreach angle for this company'),
        ('outreach_triggers', 'Timely events that justify outreach now'),
        ('risks_red_flags', 'Risks or red flags found'),
        ('confidence_score', 'Overall research confidence 0–100'),
        ('confidence_notes', 'What drove the confidence score'),
    ]),
    ('Meta', [
        ('sources_used', 'Every URL or source consulted, comma-separated'),
    ]),
]

ALL_FIELDS: list[str] = [f for _, fields in FIELD_SECTIONS for f, _ in fields]


# ── Numeric normalization ───────────────────────────────────────────────────
# Research values are prose-y strings ("$0–250,000 (Estimated)", "48/100",
# "~$4.2M", "50-100 employees"). For lead scoring in HubSpot those need to be
# plain numbers, so every field below also gets a "{field}_num" companion
# holding a single integer — the midpoint of a range, or the lone figure.

NUMERIC_FIELDS: list[str] = [
    'revenue_estimate',
    'total_venture_funding',
    'total_grant_funding',
    'estimated_valuation',
    'federal_awards_count_3yr',
    'federal_awards_total_3yr',
    'sbir_sttr_count_3yr',
    'sbir_sttr_total_3yr',
    'employee_count_current',
    'estimated_proposals_per_year',
    'score_total',
    'score_funding_activity',
    'score_momentum',
    'score_org_capacity',
    'score_budget_signals',
    'score_fit_external_support',
    'proposal_budget_estimate',
    'confidence_score',
]

NUMERIC_SUFFIX = '_num'

_NULLISH = {
    '', '-', '—', '–', 'not found', 'notfound', 'unknown', 'n/a', 'na',
    'none', 'nan', 'tbd', 'not applicable', 'not disclosed', 'undisclosed',
    'no data', 'not available',
}

_MULTIPLIERS = {
    'k': 1e3, 'thousand': 1e3,
    'm': 1e6, 'mm': 1e6, 'million': 1e6,
    'b': 1e9, 'bn': 1e9, 'billion': 1e9,
}

# A figure plus an optional scale suffix. The \b after the suffix stops the
# "b" of "board members" or the "m" of "months" being read as a multiplier.
_NUM_RE = re.compile(
    r'(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)'
    r'\s*(?P<suffix>(?:thousand|million|billion|mm|bn|[kmb])\b)?',
    re.I,
)

# Text allowed between two figures for them to count as a range. The trailing
# currency/approximation characters matter: "$0 - $250,000" and "$250K-$1M"
# both put a "$" between the two figures.
_RANGE_SEP_RE = re.compile(
    r'^\s*(?:-|–|—|~|to|and|through)\s*[~$€£]?\s*$', re.I
)

# Research prose routinely labels a figure with a fiscal year or a quarter
# ("FY2024: $2.5M", "Q1 2025 revenue of $3M"). Those tokens are context, not
# the value, and reading one as the answer writes a garbage number into a
# HubSpot number property used for lead scoring. They are dropped whenever a
# real figure survives alongside them — a bare year on its own is still
# returned, since there it may genuinely be the answer.
_YEAR_RE            = re.compile(r'^(?:19|20)\d{2}$')
_PERIOD_BEFORE_RE   = re.compile(r'(?:^|[^a-z0-9])(?:q|fy|cy)\s*$', re.I)
_CURRENCY_BEFORE_RE = re.compile(r'[$€£¥]\s*$')


def _is_period_label(m, text: str) -> bool:
    """True when a match reads as a fiscal-period label rather than a figure."""
    raw = m.group('num')
    if m.group('suffix') or ',' in raw or '.' in raw:
        return False
    before = text[:m.start()]
    if _CURRENCY_BEFORE_RE.search(before):     # "$2024" is money, not a year
        return False
    if _PERIOD_BEFORE_RE.search(before):       # Q1, FY24, CY2025
        return True
    return bool(_YEAR_RE.match(raw))


def to_number(value) -> float | None:
    """Best-effort single number out of a research field value.

    "$0-250,000 (Estimated)" → 125000.0   (midpoint of the range)
    "~$4.2M"                 → 4200000.0
    "1-2 million"            → 1500000.0  (suffix carries to the low end)
    "48/100"                 → 48.0       (leading figure, not a range)
    "50-100 employees"       → 75.0
    "FY2024: $2.5M"          → 2500000.0  (the year is a label, not the value)
    "Not found"              → None
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = re.sub(r'\([^)]*\)', ' ', str(value))   # drop "(Estimated)" labels
    text = text.replace('%', ' ')
    if text.strip().lower() in _NULLISH:
        return None

    matches = list(_NUM_RE.finditer(text))
    if not matches:
        return None
    figures = [m for m in matches if not _is_period_label(m, text)]
    if figures:
        matches = figures

    def figure(m, fallback_suffix: str | None = None) -> float:
        n = float(m.group('num').replace(',', ''))
        suffix = (m.group('suffix') or fallback_suffix or '').strip().lower()
        return n * _MULTIPLIERS.get(suffix, 1.0)

    lo = matches[0]
    if len(matches) > 1:
        hi = matches[1]
        between = text[lo.end():hi.start()]
        if _RANGE_SEP_RE.match(between):
            # "1-2 million": the low end inherits the high end's scale
            return (figure(lo, hi.group('suffix')) + figure(hi)) / 2

    return figure(lo)


def to_number_str(value) -> str:
    """to_number() rendered as an integer string, '' when unparseable.

    Empty is deliberate — a HubSpot number property left blank reads as
    "no data", whereas a 0 would drag scoring models down.
    """
    n = to_number(value)
    if n is None:
        return ''
    # Round half away from zero — round() is banker's rounding, which would
    # turn the midpoint of "2-3 proposals" into 2.
    return str(math.floor(n + 0.5) if n >= 0 else -math.floor(-n + 0.5))


def numeric_columns(d: dict) -> dict[str, str]:
    """{field}_num integer-string companions for every NUMERIC_FIELDS entry."""
    return {
        f'{f}{NUMERIC_SUFFIX}': to_number_str(d.get(f))
        for f in NUMERIC_FIELDS
    }


# ── Prompt construction ─────────────────────────────────────────────────────

_INSTRUCTIONS = """You are a financial diligence analyst working for a proposal writing and grant support services firm. Research the company described below using public sources. Return ONLY a valid JSON object — no preamble, no markdown, no code fences, no citations outside the JSON — that strictly follows the schema provided.

Rules:
- Every schema key must be present. All values are strings.
- Label each value's confidence in parentheses where meaningful: Exact | Reported | Third-party | Estimated | Not found
- Do not invent funding amounts, revenue, or valuations
- If a value is unavailable, use a bounded estimate with method noted, or "Not found"
- Prioritize sources: company website > SAM.gov > USAspending > SBIR.gov > NIH RePORTER > NSF > state registries > news
- Prioritize recency; weight the last 12 months more heavily
- Apply the scoring rubric exactly as specified
- For sources_used, list every URL or source you consulted"""

_QUESTIONS = """Research questions to answer:
1. Resolve the company's identity: legal name, UEI, SAM.gov registration status, website, NAICS codes, entity type.
2. Estimate annual revenue, noting the method and year.
3. Total venture/equity funding raised and any valuation signals.
4. Total grant funding received across federal, state, and foundation sources.
5. Federal awards over the last 3 years: count, total dollars, awarding agencies, SBIR/STTR breakdown, most recent award date, and details of notable awards.
6. Current headcount and its trend over the last 2 years.
7. Financial health signals: customer concentration, hiring activity, leadership stability, litigation or bankruptcy history, SAM certifications.
8. Grant activity profile: classification, SBIR phase progression, multi-agency breadth, momentum, and estimated proposals submitted per year.
9. Discretionary budget signals from the last 12 months: grant inflows, hiring, new products, conference participation, R&D programs, facility expansion.
10. Score the company 0-100 using the rubric: funding activity /30, grant momentum /20, organizational capacity /20, budget signals /20, fit for external proposal support /10. score_total must equal the sum of the five sub-scores.
11. Qualify the company as a prospect for paid proposal support: internal capability, likely budget, recommendation, outreach angle and triggers, risks, and your confidence in this research."""


def _schema_block() -> str:
    lines = ['Return a JSON object with exactly these keys (all string values):', '{']
    for section, fields in FIELD_SECTIONS:
        lines.append(f'  // {section}')
        for field, desc in fields:
            lines.append(f'  "{field}": "<{desc}>",')
    lines[-1] = lines[-1].rstrip(',')
    lines.append('}')
    return '\n'.join(lines)


def build_research_prompt(company: dict) -> str:
    """company keys: company_name (required), website, state (all optional)."""
    context = '\n'.join([
        'COMPANY TO RESEARCH:',
        f"Name: {company.get('company_name') or 'Unknown'}",
        f"Website: {company.get('website') or 'Unknown'}",
        f"State: {company.get('state') or 'Unknown'}",
    ])
    return '\n\n'.join([_INSTRUCTIONS, context, _QUESTIONS, _schema_block()])


# ── Parsing ─────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    start, end = cleaned.find('{'), cleaned.rfind('}')
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(cleaned[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _repair_json(oai_client, raw_text: str) -> dict | None:
    """Send malformed output to a cheap model to fix — never back to o3."""
    try:
        resp = oai_client.chat.completions.create(
            model='gpt-4o-mini',
            max_tokens=8000,
            messages=[{
                'role': 'user',
                'content': (
                    'The text below was supposed to be a single valid JSON object '
                    'but is malformed. Return ONLY the corrected JSON object — no '
                    'commentary, no code fences. Preserve all content.\n\n'
                    + raw_text[:60000]
                ),
            }],
        )
        return _extract_json(resp.choices[0].message.content or '')
    except Exception:
        return None


def parse_research_output(
    oai_client, raw_text: str, fields: list[str] | None = None
) -> tuple[dict | None, str | None]:
    """Returns (normalized_dict, None) on success or (None, error) on failure.

    `fields` defaults to the financial schema (ALL_FIELDS); pass
    tech_research.ALL_FIELDS to normalize a technology research response.
    """
    parsed = _extract_json(raw_text)
    if parsed is None:
        parsed = _repair_json(oai_client, raw_text)
    if parsed is None:
        return None, 'Could not parse JSON from response (repair attempt failed)'

    out = {}
    for field in (fields if fields is not None else ALL_FIELDS):
        val = parsed.get(field, 'Not found')
        if isinstance(val, (dict, list)):
            val = json.dumps(val)
        out[field] = str(val) if val is not None else 'Not found'
    return out, None


# ── Digest ──────────────────────────────────────────────────────────────────

def build_financial_digest(d: dict) -> str:
    """Short human-readable summary of the key findings, for the
    financial_summary column. No AI call — assembled from parsed fields."""
    def val(key):
        v = str(d.get(key) or '').strip()
        return v if v and v.lower() not in ('not found', 'unknown', 'n/a', 'none') else None

    parts = []
    if val('revenue_estimate'):
        rev = f"Revenue: {val('revenue_estimate')}"
        if val('revenue_year'):
            rev += f" ({val('revenue_year')})"
        parts.append(rev)
    if val('total_venture_funding'):
        parts.append(f"Venture funding: {val('total_venture_funding')}")
    if val('total_grant_funding'):
        parts.append(f"Grant funding: {val('total_grant_funding')}")
    if val('federal_awards_count_3yr'):
        fed = f"Federal awards (3yr): {val('federal_awards_count_3yr')}"
        if val('federal_awards_total_3yr'):
            fed += f" totaling {val('federal_awards_total_3yr')}"
        parts.append(fed)
    if val('employee_count_current'):
        hc = f"Headcount: {val('employee_count_current')}"
        if val('headcount_trend'):
            hc += f" ({val('headcount_trend')})"
        parts.append(hc)
    if val('score_total'):
        parts.append(f"Proposal readiness: {val('score_total')}/100")
    if val('recommendation'):
        parts.append(f"Recommendation: {val('recommendation')}")
    return ' · '.join(parts)
