import asyncio
import hashlib
import random
import re
import time

from anthropic import Anthropic, AsyncAnthropic, InternalServerError
from openai import AsyncOpenAI, OpenAI

import src.modules.anthropic_utils as au


# ── Default system prompt templates (use {word_limit} as a placeholder) ───────

DEFAULT_SUBJECT_SYSTEM = (
    "Generate a subject line for a cold email that is no more than {word_limit} words "
    "that summarizes what this company's area of research, products, etc. is. "
    "Just focus on the technologies and research areas. Ignore investing, finance, etc. "
    'The format should be like this: "Grant for {summary} - {agency}"'
    " Use the agency name exactly as it is given to you — never invent, guess or "
    "substitute a different agency, and never emit a placeholder in its place."
)

DEFAULT_JOSIAH_SYSTEM = (
    "You're a human grants consultant writing a personalized cold email. "
    "Write ONE natural sentence ({word_limit} words max) using this structure:\n\n"
    '"They are looking for [specific grant focus] and [natural connection to company]"\n\n'
    "CRITICAL: The first part states what they're looking for (can be a list). "
    "The second part connects to the company naturally WITHOUT parallel structure.\n\n"
    "Human writing rules for the second part:\n"
    "- Don't mirror the list structure from the first part\n"
    "- Use specific product/tech names, not generic categories\n"
    '- Add natural language: "looks like", "seems like", "from what I can tell", "it appears"\n'
    "- Use dashes or parentheses to break up rhythm\n"
    "- Focus on the strongest overlap, not everything\n\n"
    'Bad (robotic parallel): "They are looking for X, Y, and Z and it seems you are doing A, B, and C"\n'
    'Good (natural): "They are looking for autonomous UAS operations, payload capabilities, and powertrain '
    'enhancements and it looks like your heavy-lift platform with the modular engine system is exactly that"\n'
    'Good (natural): "They are looking for COTS UAS modifications including ruggedization and secure software '
    "and from what I can tell, you're doing custom UAS builds - particularly the ruggedized variants for "
    'defense applications"'
)


# ── Agency handling ──────────────────────────────────────────────────────────
# The subject line ends in the grant's agency, and the model will happily invent
# one ("- NIH", "- NASA", "- {agency}") when the value handed to it is blank or a
# stringified NaN. Normalize the value, tell the model explicitly what to do when
# it is missing, and scrub any placeholder that still comes back.

_NULLISH_AGENCIES = {'', 'nan', 'none', 'null', 'n/a', 'na', '<na>', 'nat'}

_TRAILING_JUNK_RE = re.compile(
    r"""[\s\-\u2013\u2014:]+(?:\{agency\}|\{summary\}|\[[^\]]*\]|nan|none|null|n/?a)"""
    r"""[\s"\u201d\u2019']*$""",
    re.IGNORECASE,
)

_QUOTE_OPEN = '"\u201c\u2018\''
_QUOTE_CLOSE = '"\u201d\u2019\''


def clean_agency(agency) -> str:
    """Return a usable agency label, or '' for blank / NaN / 'nan' / 'None'."""
    a = '' if agency is None else str(agency).strip().strip('"\'')
    return '' if a.lower() in _NULLISH_AGENCIES else a


def _subject_user_text(company_summary: str, agency: str) -> str:
    """User message for the subject-line call. The agency rule lives here (not only
    in the system prompt) so it still applies when the caller passes a custom one."""
    if agency:
        return (
            f"company summary:{company_summary}, agency: {agency}\n\n"
            f'The agency is exactly "{agency}" — use that name verbatim in the subject '
            f"line. Do not substitute, expand or guess a different agency."
        )
    return (
        f"company summary:{company_summary}, agency: (not provided)\n\n"
        "The agency is unknown. Omit it entirely: end the subject line after the "
        "research summary, with no trailing ' - agency', no placeholder and no guess."
    )


def _sanitize_subject(text: str, agency: str) -> str:
    """Strip wrapping quotes and repair/remove leaked agency placeholders."""
    s = str(text or '').strip()
    while len(s) >= 2 and s[0] in _QUOTE_OPEN and s[-1] in _QUOTE_CLOSE:
        s = s[1:-1].strip()
    if agency:
        s = re.sub(r'\{agency\}|\[agency[^\]]*\]', agency, s, flags=re.IGNORECASE)
    s = _TRAILING_JUNK_RE.sub('', s).strip()
    return s.strip(' -\u2013\u2014:')


# ── In-memory subject line cache (persists for the duration of the session) ──
# Key: md5(company_summary + agency), Value: generated subject line string
_subject_line_cache: dict[str, str] = {}


def generate_subject_line(
    company_summary: str,
    agency: str,
    openai_client: OpenAI,
    anth_client: Anthropic,
    word_limit: int = 15,
    max_retries: int = 3,
) -> str:
    """
    Generate a cold-email subject line for a company × agency pair.

    Tries OpenAI (gpt-4o-mini) first. On a 429 rate-limit it falls back
    immediately to Anthropic (claude-haiku). Results are cached in-memory
    by (company_summary, agency) so repeated calls for the same company
    never hit the API twice in a single session.
    """

    # ── cache lookup ──────────────────────────────────────────────────────
    agency    = clean_agency(agency)
    cache_key = hashlib.md5(f"{company_summary}||{agency}".encode()).hexdigest()
    if cache_key in _subject_line_cache:
        return _subject_line_cache[cache_key]

    system = DEFAULT_SUBJECT_SYSTEM.replace('{word_limit}', str(word_limit))
    text   = _subject_user_text(company_summary, agency)

    # ── try OpenAI first ──────────────────────────────────────────────────
    use_fallback = False
    for attempt in range(max_retries):
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": text},
                ],
            )
            result = _sanitize_subject(completion.choices[0].message.content, agency)
            _subject_line_cache[cache_key] = result
            return result

        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                print(f"  [generate_subject_line] OpenAI rate limit hit — switching to Anthropic fallback.")
                use_fallback = True
                break
            wait = (2 ** attempt) + random.random()
            print(f"  [generate_subject_line] OpenAI error (attempt {attempt + 1}): {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)

    if not use_fallback:
        # exhausted retries without a rate-limit — still try fallback
        use_fallback = True

    # ── Anthropic fallback (claude-haiku) ─────────────────────────────────
    for attempt in range(max_retries):
        try:
            message = anth_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=60,
                system=system,
                messages=[{"role": "user", "content": text}],
            )
            result = _sanitize_subject(au.response_text(message), agency)
            _subject_line_cache[cache_key] = result
            print(f"  [generate_subject_line] Anthropic fallback succeeded.")
            return result

        except Exception as e:
            wait = (2 ** attempt) + random.random()
            print(f"  [generate_subject_line] Anthropic error (attempt {attempt + 1}): {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)

    raise RuntimeError(
        f"generate_subject_line failed after all retries "
        f"(OpenAI + Anthropic) for agency={agency}"
    )


def generate_body(
    company_summary: str,
    grant_summary: str,
    agency: str,
    word_limit: int,
    anth_client: Anthropic,
    max_tokens: int = 50,
    model: str = "claude-3-7-sonnet-20250219",
) -> str:
    system = (
        f"Generate a one sentence summary of no more than {word_limit} words on why this "
        f"company's description and this agency's grant description are well aligned. "
        f"Keep it very simple and general. "
        f'Insert it into this sentence: "You may be a good fit because {{summary}}."'
    )
    text = f"company:{company_summary}\n\n, grant:{grant_summary}\n\n, grant agency: {agency}"

    message = anth_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
    )
    return au.response_text(message)


def generate_tech_summary(
    company_summary: str,
    word_limit: int,
    anth_client: Anthropic,
    max_tokens: int = 50,
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    system = (
        f"Generate a summary of no more than {str(word_limit)} words about what this company "
        f"is developing. Be sure to focus on specific technologies and research areas. "
        f'Insert it into this sentence: "It looks like you guys are working in {{summary}}"'
    )
    text = f"company: {company_summary}"

    message = anth_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
    )
    return au.response_text(message)


async def async_generate_subject_line(
    company_summary: str,
    agency: str,
    openai_client: AsyncOpenAI,
    anth_client: AsyncAnthropic,
    word_limit: int = 15,
    max_retries: int = 3,
    system_override: str | None = None,
) -> str:
    """Async version of generate_subject_line — OpenAI first, Anthropic fallback."""
    agency    = clean_agency(agency)
    cache_key = hashlib.md5(f"{company_summary}||{agency}".encode()).hexdigest()
    if cache_key in _subject_line_cache:
        return _subject_line_cache[cache_key]

    template = system_override if system_override else DEFAULT_SUBJECT_SYSTEM
    system = template.replace('{word_limit}', str(word_limit))
    text = _subject_user_text(company_summary, agency)

    for attempt in range(max_retries):
        try:
            completion = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": text},
                ],
            )
            result = _sanitize_subject(completion.choices[0].message.content, agency)
            _subject_line_cache[cache_key] = result
            return result
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                break
            await asyncio.sleep((2 ** attempt) + random.random())

    for attempt in range(max_retries):
        try:
            message = await anth_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=60,
                system=system,
                messages=[{"role": "user", "content": text}],
            )
            result = _sanitize_subject(au.response_text(message), agency)
            _subject_line_cache[cache_key] = result
            return result
        except Exception as e:
            await asyncio.sleep((2 ** attempt) + random.random())

    raise RuntimeError(f"async_generate_subject_line failed for agency={agency}")


async def async_josiah_copy(
    company_summary: str,
    grant_summary: str,
    word_limit: int,
    anth_client: AsyncAnthropic,
    model: str = "claude-haiku-4-5-20251001",
    system_override: str | None = None,
) -> str:
    """Async version of josiah_copy."""
    template = system_override if system_override else DEFAULT_JOSIAH_SYSTEM
    system = template.replace('{word_limit}', str(word_limit))
    for attempt in range(5):
        try:
            message = await anth_client.messages.create(
                model=model,
                max_tokens=500,
                system=system,
                messages=[{"role": "user", "content": [{"type": "text", "text": f"Company: {company_summary}\nGrant: {grant_summary}"}]}],
            )
            return au.response_text(message)
        except Exception as e:
            err = str(e)
            if any(x in err for x in ('529', '429', 'overloaded', 'rate_limit', 'rate limit')):
                await asyncio.sleep((2 ** attempt) + random.random())
            else:
                raise
    raise RuntimeError('async_josiah_copy failed after all retries')


async def async_custom_prompt(
    text: str,
    system: str,
    anth_client: AsyncAnthropic,
    model: str = 'claude-haiku-4-5-20251001',
    max_tokens: int = 500,
) -> str:
    """Run a custom system prompt against assembled column text."""
    for attempt in range(5):
        try:
            message = await anth_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{'role': 'user', 'content': [{'type': 'text', 'text': text}]}],
            )
            return au.response_text(message)
        except Exception as e:
            err = str(e)
            if any(x in err for x in ('529', '429', 'overloaded', 'rate_limit', 'rate limit')):
                await asyncio.sleep((2 ** attempt) + random.random())
            else:
                raise
    raise RuntimeError('async_custom_prompt failed after all retries')


def josiah_copy(
    company_summary: str,
    grant_summary: str,
    word_limit: int,
    anth_client: Anthropic,
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    system = (
        f"You're a human grants consultant writing a personalized cold email. "
        f"Write ONE natural sentence ({word_limit} words max) using this structure:\n\n"
        f'"They are looking for [specific grant focus] and [natural connection to company]"\n\n'
        f"CRITICAL: The first part states what they're looking for (can be a list). "
        f"The second part connects to the company naturally WITHOUT parallel structure.\n\n"
        f"Human writing rules for the second part:\n"
        f"- Don't mirror the list structure from the first part\n"
        f"- Use specific product/tech names, not generic categories\n"
        f'- Add natural language: "looks like", "seems like", "from what I can tell", "it appears"\n'
        f"- Use dashes or parentheses to break up rhythm\n"
        f"- Focus on the strongest overlap, not everything\n\n"
        f'Bad (robotic parallel): "They are looking for X, Y, and Z and it seems you are doing A, B, and C"\n'
        f'Good (natural): "They are looking for autonomous UAS operations, payload capabilities, and powertrain '
        f"enhancements and it looks like your heavy-lift platform with the modular engine system is exactly that\"\n"
        f'Good (natural): "They are looking for COTS UAS modifications including ruggedization and secure software '
        f"and from what I can tell, you're doing custom UAS builds - particularly the ruggedized variants for "
        f'defense applications"'
    )
    text = f"Company: {company_summary}\nGrant: {grant_summary}"

    message = anth_client.messages.create(
        model=model,
        max_tokens=500,
        system=system,
        messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
    )
    return au.response_text(message)
