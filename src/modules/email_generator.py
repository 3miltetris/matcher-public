import hashlib
import time
import random

from anthropic import Anthropic, InternalServerError
from openai import OpenAI


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
    cache_key = hashlib.md5(f"{company_summary}||{agency}".encode()).hexdigest()
    if cache_key in _subject_line_cache:
        return _subject_line_cache[cache_key]

    system = (
        f"Generate a subject line for a cold email that is no more than {word_limit} words "
        f"that summarizes what this company's area of research, products, etc. is. "
        f"Just focus on the technologies and research areas. Ignore investing, finance, etc. "
        f'The format should be like this: "Grant for {{summary}} - {{agency}}"'
    )
    text = f"company summary:{company_summary}, agency: {agency}"

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
            result = completion.choices[0].message.content
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
            result = message.content[0].text
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
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
    )
    return message.content[0].text


def generate_tech_summary(
    company_summary: str,
    word_limit: int,
    anth_client: Anthropic,
    max_tokens: int = 50,
    model: str = "claude-3-haiku-20240307",
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
    return message.content[0].text


def josiah_copy(
    company_summary: str,
    grant_summary: str,
    word_limit: int,
    anth_client: Anthropic,
    model: str = "claude-sonnet-4-20250514",
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
        temperature=0.7,
        system=system,
        messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
    )
    return message.content[0].text
