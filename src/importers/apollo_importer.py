"""
apollo_importer.py – Apollo CSV → parquet pipeline.

Converts raw Apollo CSV exports into embedding-enriched parquet files
suitable for the matcher pipeline.

Usage
-----
    # from Streamlit or another script:
    import asyncio
    from src.importers.apollo_importer import run
    out_path = asyncio.run(run())

    # CLI:
    python -m src.importers.apollo_importer
"""

import asyncio
import os
from datetime import datetime

import pandas as pd
from openai import AsyncOpenAI

from src.modules.Embedding.text_embedder import TextProcessor
from src.modules.lead_importer import (
    load_csvs,
    normalize_columns,
    filter_non_commercial,
    dedup_against_existing,
    scrape_pages,
    summarize_companies,
    embed_summaries,
    export_contacts,
)


# ---------------------------------------------------------------------------
# Apollo-specific column mapping
# ---------------------------------------------------------------------------

_APOLLO_RENAMES = {
    'company':              'company_name',
    'industry':             'segment',
    'website':              'companyWebsite',
    'company_linkedin_url': 'linkedin',
}

_KEEP_COLS = [
    'company_name', 'company_name_for_emails', 'segment',
    'companyWebsite', 'linkedin', 'short_description',
    'firstName', 'lastName', 'email', 'phone',
]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run(
    input_glob: str = 'data/unprocessed/apollo/*',
    export_path: str = 'data/all-contacts/apollo',
    existing_contacts_glob: str = 'data/all-contacts/**/*.parquet',
    year_filter: str | None = None,
    openai_api_key: str | None = None,
) -> str:
    """
    Full Apollo import pipeline.

    Args:
        input_glob:             Glob matching raw Apollo CSV files.
        export_path:            Directory to write the output parquet.
        existing_contacts_glob: Glob used to dedup against already-processed contacts.
        year_filter:            Optional filename substring filter (e.g. '2025').
        openai_api_key:         OpenAI key (falls back to OPENAI_API_KEY env var).

    Returns:
        Path to the exported parquet file, or '' if nothing was processed.
    """
    api_key       = openai_api_key or os.environ['OPENAI_API_KEY']
    openai_client = AsyncOpenAI(api_key=api_key)
    tp            = TextProcessor(api_key=api_key)
    today         = datetime.today().strftime('%Y-%m-%d')

    # ── 1. Load & normalise ────────────────────────────────────────────────
    raw = load_csvs(input_glob, year_filter=year_filter)
    raw = normalize_columns(raw)
    raw = raw.rename(columns={k: v for k, v in _APOLLO_RENAMES.items() if k in raw.columns})
    print(f'Loaded: {len(raw)} rows')

    df = (
        raw
        .assign(company_name=lambda x: x['company_name'].astype(str))
        .sort_values('company_name', ascending=False)
        .drop_duplicates(subset='companyWebsite', keep='first')
        .reset_index(drop=True)
        [[c for c in _KEEP_COLS if c in raw.columns]]
    )

    runoff = df[df['companyWebsite'].isna()].reset_index(drop=True)
    df     = df[df['companyWebsite'].notna()].reset_index(drop=True)
    df     = filter_non_commercial(df)
    print(f'After normalisation: {len(df)} rows | Runoff (no website): {len(runoff)}')

    # ── 2. Dedup against existing parquets ────────────────────────────────
    df = dedup_against_existing(df, existing_contacts_glob)
    print(f'After dedup: {len(df)} new contacts')

    if len(df) == 0:
        print('Nothing new to process.')
        return ''

    # ── 3. Scrape ─────────────────────────────────────────────────────────
    page_texts  = await scrape_pages(df)

    failed_urls = page_texts[page_texts['page_text'] == 'FAILED']['companyWebsite'].tolist()
    if failed_urls:
        retry = await scrape_pages(pd.DataFrame({'companyWebsite': failed_urls}))
        page_texts.update(retry.set_index('companyWebsite'), overwrite=True)
    print(f'Scraped {len(page_texts)} pages | {len(failed_urls)} retried')

    # ── 4. Merge Apollo short_description + scraped text ─────────────────
    df_merged = pd.merge(df, page_texts, on='companyWebsite', how='left')
    short_desc = df_merged.get('short_description', pd.Series('', index=df_merged.index))
    df_merged['page_text'] = (
        short_desc.fillna('').astype(str) + '\n' + df_merged['page_text'].fillna('')
    )

    bad_mask  = short_desc.isna() & df_merged['page_text'].str.contains('ERROR', na=False)
    runoff    = pd.concat([runoff, df_merged[bad_mask]]).reset_index(drop=True)
    df_merged = df_merged[~bad_mask].reset_index(drop=True)
    print(f'Ready for summarisation: {len(df_merged)} | Runoff: {len(runoff)}')

    # ── 5. Summarise ──────────────────────────────────────────────────────
    df_merged = await summarize_companies(df_merged, openai_client, text_col='page_text')

    # ── 6. Embed ──────────────────────────────────────────────────────────
    df_merged = await embed_summaries(df_merged, tp, text_col='company_summary')

    # ── 7. Export ─────────────────────────────────────────────────────────
    return export_contacts(df_merged, export_path, source_name='apollo', today=today)


if __name__ == '__main__':
    asyncio.run(run())
