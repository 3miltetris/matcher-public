"""
Shared helpers for the Technology & R&D research focus of the Client
Research view — same Deep Research strategy as finance_research.py but
aimed at a company's technology, products, and R&D activity instead of
its financials.

Models, pricing, cost estimation, and JSON parse/repair are shared with
finance_research.py — this module only defines the technology output
schema, prompt, and digest. Parse a response with
finance_research.parse_research_output(oai_client, raw, fields=ALL_FIELDS).
"""

import re


# ── Output schema ───────────────────────────────────────────────────────────
# Flat JSON object, one key per field. All values are strings; where a
# confidence label applies it is appended in parentheses, e.g.
# "TRL 6 (Estimated)".

FIELD_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    ('Identity', [
        ('company_name_resolved', 'Resolved legal/operating company name'),
        ('website_resolved', 'Primary company website URL'),
        ('industry_vertical', 'Primary industry / vertical'),
        ('year_founded', 'Year the company was founded'),
        ('hq_location', 'Headquarters city and state'),
    ]),
    ('Core Technology', [
        ('core_technology_summary', '3-6 sentence plain-language summary of the core technology'),
        ('technology_categories', 'Comma-separated technology categories, e.g. AI/ML, photonics, synthetic biology'),
        ('technical_approach', 'How the technology works — methods, architectures, materials, algorithms'),
        ('key_capabilities', 'Concrete technical capabilities the company demonstrates'),
        ('scientific_disciplines', 'Underlying scientific/engineering disciplines'),
        ('hardware_software_mix', 'Hardware / Software / Both, with brief detail'),
    ]),
    ('Products & Services', [
        ('flagship_products', 'Named flagship products or platforms, comma-separated'),
        ('product_detail_json', 'JSON array string of products: [{"name","description","status","customers"}]'),
        ('services_offered', 'Services offered (contract R&D, testing, integration, consulting, etc.)'),
        ('target_customers', 'Who buys — market segments, agencies, industries'),
        ('use_cases', 'Primary use cases / applications'),
        ('delivery_model', 'e.g. SaaS, licensed IP, manufactured hardware, contract R&D'),
    ]),
    ('R&D Activity', [
        ('active_rd_programs', 'Active R&D programs or projects with brief descriptions'),
        ('rd_focus_areas', 'Current R&D focus areas / problem spaces'),
        ('research_partnerships', 'Universities, national labs, or corporate research partners'),
        ('publications_signal', 'Peer-reviewed publications, whitepapers, or conference talks found'),
        ('emerging_directions', 'Where the technology roadmap appears to be heading'),
    ]),
    ('Intellectual Property', [
        ('patent_count', 'Number of granted patents / published applications found'),
        ('notable_patents', 'Notable patents: number, title, year'),
        ('patent_areas', 'Technical areas covered by the patent portfolio'),
        ('proprietary_platforms', 'Proprietary platforms, datasets, or processes referenced publicly'),
    ]),
    ('Technology Maturity', [
        ('trl_estimate', 'Estimated Technology Readiness Level 1-9 with one-line justification'),
        ('commercialization_stage', 'Concept / Prototype / Pilot / Commercial / Scaling'),
        ('production_capacity', 'Manufacturing or deployment capacity evidence'),
        ('regulatory_certifications', 'Regulatory clearances or certifications (FDA, FCC, ISO, ITAR, CMMC, etc.)'),
    ]),
    ('Differentiation', [
        ('key_differentiators', 'What sets the technology apart from alternatives'),
        ('competitive_landscape', 'Known competitors or competing technical approaches'),
        ('dual_use_potential', 'Defense/civilian dual-use potential'),
        ('barriers_to_entry', 'Moats: IP, data, expertise, certifications, relationships'),
    ]),
    ('Grant Alignment', [
        ('suggested_grant_keywords', 'Comma-separated technical keywords for matching this company to grant topics'),
        ('agency_fit', 'Federal agencies whose R&D missions align (DOD, NIH, NSF, DOE, NASA, …) and why'),
        ('sbir_topic_alignment', 'SBIR/STTR topic areas this technology plausibly fits'),
        ('technology_gaps', 'Capability gaps that R&D funding could plausibly close'),
    ]),
    ('Meta', [
        ('confidence_score', 'Overall research confidence 0-100'),
        ('confidence_notes', 'What drove the confidence score'),
        ('sources_used', 'Every URL or source consulted, comma-separated'),
    ]),
]

ALL_FIELDS: list[str] = [f for _, fields in FIELD_SECTIONS for f, _ in fields]


# ── Prompt construction ─────────────────────────────────────────────────────

_INSTRUCTIONS = """You are a technology analyst working for a proposal writing and grant support services firm. Research the company described below using public sources, focusing on its technology, products, and R&D activity. Return ONLY a valid JSON object — no preamble, no markdown, no code fences, no citations outside the JSON — that strictly follows the schema provided.

Rules:
- Every schema key must be present. All values are strings.
- Label each value's confidence in parentheses where meaningful: Exact | Reported | Third-party | Estimated | Not found
- Do not invent capabilities, patents, products, or certifications
- If a value is unavailable, use a bounded estimate with method noted, or "Not found"
- Prefer technical specificity over marketing language — describe what the technology actually does and how
- Prioritize sources: company website and product/technical documentation > patent databases (USPTO, Google Patents) > SBIR.gov award abstracts > peer-reviewed publications and conference proceedings > press releases > news
- Prioritize recency; weight the last 24 months more heavily for R&D activity and roadmap signals
- For sources_used, list every URL or source you consulted"""

_QUESTIONS = """Research questions to answer:
1. Resolve the company's identity: name, website, industry vertical, founding year, headquarters location.
2. Characterize the core technology: what it is, how it works, technology categories, demonstrated capabilities, underlying scientific disciplines, and the hardware/software mix.
3. Catalog products and services: flagship products with per-product detail, services offered, target customers, primary use cases, and delivery model.
4. Map R&D activity: active programs, current focus areas, research partnerships (universities, national labs, corporates), publication activity, and where the roadmap appears to be heading.
5. Assess intellectual property: patent count, notable patents, technical areas covered, and any proprietary platforms, datasets, or processes referenced publicly.
6. Judge technology maturity: estimated TRL with justification, commercialization stage, production/deployment capacity, and regulatory clearances or certifications.
7. Analyze differentiation: key differentiators, competitive landscape, dual-use potential, and barriers to entry.
8. Assess grant alignment: technical keywords that would match this company to federal R&D grant topics, which agencies' missions align and why, plausible SBIR/STTR topic areas, and capability gaps R&D funding could close.
9. State your overall confidence in this research and every source used."""


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


# ── Digest & matching summary ───────────────────────────────────────────────

# Confidence labels the prompt asks for, e.g. "(Estimated)" — stripped when
# composing text that will be embedded.
_CONF_RE = re.compile(
    r'\s*\((?:Exact|Reported|Third-party|Estimated|Not found)[^)]*\)', re.I
)


def _val(d: dict, key: str, strip_conf: bool = False) -> str | None:
    v = str(d.get(key) or '').strip()
    if strip_conf:
        v = _CONF_RE.sub('', v).strip()
    return v if v and v.lower() not in ('not found', 'unknown', 'n/a', 'none') else None


def build_matching_summary(d: dict) -> str:
    """Embedding-ready company description assembled from the researched
    technology fields. Used when the user opts to rewrite the client's
    matching summary (and re-embed) from a technology research run.
    Confidence labels are stripped; no AI call."""
    def val(key):
        return _val(d, key, strip_conf=True)

    parts = []
    if val('core_technology_summary'):
        parts.append(val('core_technology_summary'))
    if val('technical_approach'):
        parts.append(f"Technical approach: {val('technical_approach')}")
    if val('key_capabilities'):
        parts.append(f"Key capabilities: {val('key_capabilities')}")
    if val('flagship_products'):
        parts.append(f"Products: {val('flagship_products')}")
    if val('services_offered'):
        parts.append(f"Services: {val('services_offered')}")
    if val('use_cases'):
        parts.append(f"Use cases: {val('use_cases')}")
    if val('rd_focus_areas'):
        parts.append(f"R&D focus areas: {val('rd_focus_areas')}")
    if val('suggested_grant_keywords'):
        parts.append(f"Technology keywords: {val('suggested_grant_keywords')}")
    return '\n'.join(parts)


def build_tech_digest(d: dict) -> str:
    """Short human-readable summary of the key findings, for the
    technology_summary column. No AI call — assembled from parsed fields."""
    def val(key):
        v = str(d.get(key) or '').strip()
        return v if v and v.lower() not in ('not found', 'unknown', 'n/a', 'none') else None

    parts = []
    core = val('core_technology_summary')
    if core:
        parts.append(core if len(core) <= 400 else core[:397] + '…')
    if val('technology_categories'):
        parts.append(f"Categories: {val('technology_categories')}")
    if val('flagship_products'):
        parts.append(f"Products: {val('flagship_products')}")
    if val('trl_estimate'):
        parts.append(f"TRL: {val('trl_estimate')}")
    if val('patent_count'):
        parts.append(f"Patents: {val('patent_count')}")
    if val('suggested_grant_keywords'):
        parts.append(f"Grant keywords: {val('suggested_grant_keywords')}")
    if val('agency_fit'):
        parts.append(f"Agency fit: {val('agency_fit')}")
    return ' · '.join(parts)
