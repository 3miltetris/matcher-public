"""
ai_analyzer.py — LLM-based match analysis (yes/no alignment calls).

All functions take explicit client args rather than relying on globals,
so they work cleanly when imported from a notebook or script.

Usage
-----
from src.modules.ai_analyzer import analyze_matches_anthropic, analyze_matches_async

matches = analyze_matches_anthropic(matches, anth_client)
matches = await analyze_matches_async(matches, anth_client_async)
"""

import asyncio
import random

import pandas as pd
from anthropic import Anthropic, AsyncAnthropic, RateLimitError, InternalServerError
from openai import OpenAI, AsyncOpenAI

# ---------------------------------------------------------------------------
# Shared prompt builders
# ---------------------------------------------------------------------------

def _build_prompt(row: pd.Series, source: str = 'grant') -> tuple[str, str]:
    """Return (system, user) strings for a given match row."""
    if source == 'grant':
        system = (
            'Tell me if this company summary and grant summary are aligned. '
            'Only give a one-word answer. Either "yes" or "no".'
        )
        text = (
            f"company summary: {row['company_summary']}\n\n"
            f"grant summary: {row['grant_summary']}"
        )
    elif source == 'abstract':
        system = (
            'Tell me if this company summary and awardee abstract are aligned. '
            'Only give a one-word answer. Either "yes" or "no". '
            'If they are the same company, answer "no".'
        )
        text = (
            f"company summary: {row['company_summary']}\n\n"
            f"Abstract: {row['grant_summary']}"
        )
    else:
        raise ValueError(f'Unknown source: {source!r}. Use "grant" or "abstract".')

    return system, text


# ---------------------------------------------------------------------------
# Synchronous — Anthropic only
# ---------------------------------------------------------------------------

def analyze_matches_anthropic(
    matches: pd.DataFrame,
    anth_client: Anthropic,
    model: str = 'claude-3-haiku-20240307',
    source: str = 'grant',
) -> pd.DataFrame:
    """
    Sync yes/no alignment pass using Anthropic only.

    Skips subsequent rows for a company website that already got a 'yes',
    saving API calls when one company appears against many grants.
    """
    yes_websites: list[str] = []
    matches = matches.copy()
    matches['good_match'] = None

    for index, row in matches.iterrows():
        website = row.get('companyWebsite', '')

        if website in yes_websites:
            print(f'{website} already matched — skipping.')
            matches.at[index, 'good_match'] = 'yes'
            continue

        system, text = _build_prompt(row, source)

        message = anth_client.messages.create(
            model=model,
            max_tokens=15,
            temperature=0,
            system=system,
            messages=[{'role': 'user', 'content': text}],
        )

        result = message.content[0].text
        matches.at[index, 'good_match'] = result

        if 'yes' in result.lower():
            yes_websites.append(website)

        remaining = len(matches) - (index + 1)
        print(f'[{index}] {result}  |  {remaining} remaining')

    return matches


# ---------------------------------------------------------------------------
# Synchronous — OpenAI only
# ---------------------------------------------------------------------------

def analyze_matches_openai(
    matches: pd.DataFrame,
    openai_client: OpenAI,
    model: str = 'gpt-4o-2024-05-13',
    source: str = 'grant',
) -> pd.DataFrame:
    """Sync yes/no alignment pass using OpenAI only."""
    matches = matches.copy()
    matches['good_match'] = None

    for index, row in matches.iterrows():
        system, text = _build_prompt(row, source)

        completion = openai_client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user',   'content': text},
            ],
        )

        result = completion.choices[0].message.content
        matches.at[index, 'good_match'] = result
        print(f'[{index}] {result}')

    return matches


# ---------------------------------------------------------------------------
# Synchronous — Anthropic → OpenAI double-check
# ---------------------------------------------------------------------------

def analyze_matches_dual(
    matches: pd.DataFrame,
    anth_client: Anthropic,
    openai_client: OpenAI,
    anthropic_model: str = 'claude-3-haiku-20240307',
    openai_model: str = 'gpt-4o-2024-05-13',
    source: str = 'grant',
) -> pd.DataFrame:
    """
    Sync two-stage analysis: Anthropic screens first, OpenAI confirms 'yes' answers.
    Reduces OpenAI costs by only calling it for Anthropic positives.
    """
    matches = matches.copy()
    matches['good_match'] = None

    for index, row in matches.iterrows():
        system, text = _build_prompt(row, source)

        anth_msg = anth_client.messages.create(
            model=anthropic_model,
            max_tokens=15,
            temperature=0,
            system=system,
            messages=[{'role': 'user', 'content': text}],
        )
        anth_result = anth_msg.content[0].text

        if 'yes' in anth_result.lower():
            print(f'[{index}] Anthropic said yes — confirming with OpenAI...')
            completion = openai_client.chat.completions.create(
                model=openai_model,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user',   'content': text},
                ],
            )
            result = completion.choices[0].message.content
        else:
            result = anth_result

        matches.at[index, 'good_match'] = result
        print(f'[{index}] {result}')

    return matches


# ---------------------------------------------------------------------------
# Async — Anthropic (batched, with exponential backoff)
# ---------------------------------------------------------------------------

async def analyze_matches_async(
    matches: pd.DataFrame,
    anth_client_async: AsyncAnthropic,
    model: str = 'claude-3-haiku-20240307',
    source: str = 'grant',
    batch_size: int = 100,
    max_retries: int = 10_000,
    inter_batch_delay: float = 2.0,
) -> pd.DataFrame:
    """
    Async yes/no alignment pass using Anthropic, with:
      - batched concurrency (default 100 at a time)
      - exponential backoff on RateLimitError / InternalServerError
      - same per-website deduplication as the sync version

    Run from a notebook with:
        matches = await analyze_matches_async(matches, anth_client_async)
    """
    yes_websites: list[str] = []
    matches = matches.copy()
    matches['good_match'] = None

    async def process_match(index: int, row: pd.Series) -> None:
        website = row.get('companyWebsite', '')

        if website in yes_websites:
            print(f'{website} already matched — skipping.')
            matches.at[index, 'good_match'] = 'yes'
            return

        system, text = _build_prompt(row, source)
        attempt = 0

        while True:
            try:
                message = await anth_client_async.messages.create(
                    model=model,
                    max_tokens=15,
                    temperature=0,
                    system=system,
                    messages=[{'role': 'user', 'content': text}],
                )

                result = message.content[0].text
                matches.at[index, 'good_match'] = result

                if 'yes' in result.lower():
                    yes_websites.append(website)

                remaining = len(matches) - (index + 1)
                print(f'[{index}] {result}  |  {remaining} remaining')
                return

            except (RateLimitError, InternalServerError) as e:
                attempt += 1
                wait = min(600, (2 ** attempt) + random.random())
                print(
                    f'[{index}] {type(e).__name__} on attempt {attempt}. '
                    f'Retrying in {wait:.1f}s...'
                )
                await asyncio.sleep(wait)

                if attempt >= max_retries:
                    print(f'[{index}] Max retries exceeded. Moving on.')
                    return

    for i in range(0, len(matches), batch_size):
        batch = matches.iloc[i : i + batch_size]
        await asyncio.gather(*(process_match(idx, row) for idx, row in batch.iterrows()))
        await asyncio.sleep(inter_batch_delay)

    return matches


# ---------------------------------------------------------------------------
# Async — OpenAI
# ---------------------------------------------------------------------------

async def analyze_matches_openai_async(
    matches: pd.DataFrame,
    openai_client_async: AsyncOpenAI,
    model: str = 'gpt-4o-mini',
    source: str = 'grant',
) -> pd.DataFrame:
    """Fully concurrent async yes/no pass using OpenAI."""
    matches = matches.copy()
    matches['good_match'] = None

    async def fetch(index: int, row: pd.Series) -> None:
        system, text = _build_prompt(row, source)
        completion = await openai_client_async.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user',   'content': text},
            ],
        )
        result = completion.choices[0].message.content
        matches.at[index, 'good_match'] = result
        print(f'[{index}] {result}')

    await asyncio.gather(*(fetch(idx, row) for idx, row in matches.iterrows()))
    return matches


# ---------------------------------------------------------------------------
# Pivot analysis — re-screen "no" matches for adaptability
# ---------------------------------------------------------------------------

PIVOT_CHECK_SYSTEM = (
    'You are evaluating whether a company could realistically adapt or pivot its existing '
    'technology to meet the requirements of a grant program — even if it is not a direct fit today. '
    'Consider whether the core competencies, platform, or underlying science could be '
    'redirected with relative ease. '
    'Only answer "yes" or "no".'
)

PIVOT_NOTE_SYSTEM = (
    'You are a grants consultant. A company\'s technology is not a direct fit for a grant, '
    'but could plausibly be adapted or pivoted to qualify. '
    'In 1-2 sentences, describe specifically what that pivot or application shift would look like. '
    'Be concrete — name the technology and the grant requirement. '
    'Do not say "the company could pivot"; just describe the pivot itself.'
)


async def analyze_pivot_async(
    matches: pd.DataFrame,
    anth_client_async: AsyncAnthropic,
    model: str = 'claude-3-haiku-20240307',
    batch_size: int = 100,
    max_retries: int = 10_000,
    inter_batch_delay: float = 2.0,
) -> pd.DataFrame:
    """
    Re-screen rows that got a 'no' from the primary alignment check,
    asking whether the technology could be adapted or pivoted to meet
    the grant requirements with relative ease.

    Expects a DataFrame that already has a 'good_match' column (from
    analyze_matches_async). Only rows where good_match contains 'no'
    are re-evaluated; all others are passed through unchanged.

    Adds a 'pivot_possible' column ('yes' / 'no').
    """
    matches = matches.copy()
    matches['pivot_possible'] = None

    no_mask = matches['good_match'].fillna('').str.contains('no', case=False)
    no_rows  = matches[no_mask]

    print(f'Pivot check: {len(no_rows)} "no" rows to re-evaluate...')

    async def process_row(index: int, row: pd.Series) -> None:
        text = (
            f"Company summary: {row['company_summary']}\n\n"
            f"Grant summary: {row['grant_summary']}"
        )
        attempt = 0

        while True:
            try:
                message = await anth_client_async.messages.create(
                    model=model,
                    max_tokens=15,
                    temperature=0,
                    system=PIVOT_CHECK_SYSTEM,
                    messages=[{'role': 'user', 'content': text}],
                )
                result = message.content[0].text
                matches.at[index, 'pivot_possible'] = result
                remaining = len(no_rows) - list(no_rows.index).index(index) - 1
                print(f'  [{index}] pivot={result}  |  {remaining} remaining')
                return

            except (RateLimitError, InternalServerError) as e:
                attempt += 1
                wait = min(600, (2 ** attempt) + random.random())
                print(f'  [{index}] {type(e).__name__} attempt {attempt}, retrying in {wait:.1f}s...')
                await asyncio.sleep(wait)
                if attempt >= max_retries:
                    print(f'  [{index}] Max retries exceeded.')
                    return

    for i in range(0, len(no_rows), batch_size):
        batch = no_rows.iloc[i : i + batch_size]
        await asyncio.gather(*(process_row(idx, row) for idx, row in batch.iterrows()))
        await asyncio.sleep(inter_batch_delay)

    return matches


async def generate_pivot_notes(
    matches: pd.DataFrame,
    anth_client_async: AsyncAnthropic,
    model: str = 'claude-3-5-sonnet-20241022',
    batch_size: int = 50,
    max_retries: int = 1_000,
    inter_batch_delay: float = 1.0,
) -> pd.DataFrame:
    """
    For rows where pivot_possible='yes', generate a short concrete note
    describing what the application pivot would look like.

    Adds a 'pivot_note' column.
    """
    matches = matches.copy()
    matches['pivot_note'] = None

    pivot_mask = matches['pivot_possible'].fillna('').str.contains('yes', case=False)
    pivot_rows  = matches[pivot_mask]

    print(f'Generating pivot notes for {len(pivot_rows)} rows...')

    async def process_row(index: int, row: pd.Series) -> None:
        text = (
            f"Company summary: {row['company_summary']}\n\n"
            f"Grant summary: {row['grant_summary']}"
        )
        attempt = 0

        while True:
            try:
                message = await anth_client_async.messages.create(
                    model=model,
                    max_tokens=150,
                    temperature=0.4,
                    system=PIVOT_NOTE_SYSTEM,
                    messages=[{'role': 'user', 'content': text}],
                )
                note = message.content[0].text
                matches.at[index, 'pivot_note'] = note
                print(f'  [{index}] {note[:80]}...')
                return

            except (RateLimitError, InternalServerError) as e:
                attempt += 1
                wait = min(600, (2 ** attempt) + random.random())
                print(f'  [{index}] {type(e).__name__} attempt {attempt}, retrying in {wait:.1f}s...')
                await asyncio.sleep(wait)
                if attempt >= max_retries:
                    print(f'  [{index}] Max retries exceeded.')
                    return

    for i in range(0, len(pivot_rows), batch_size):
        batch = pivot_rows.iloc[i : i + batch_size]
        await asyncio.gather(*(process_row(idx, row) for idx, row in batch.iterrows()))
        await asyncio.sleep(inter_batch_delay)

    return matches
