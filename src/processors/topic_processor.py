"""
topic_processor.py – Grant topic spreadsheet → parquet pipeline.

Loads grant topic Excel/CSV files, optionally summarises long descriptions
via GPT-4o, generates OpenAI embeddings, and saves the result as parquet.

Usage
-----
    # from Streamlit or another script:
    from src.processors.topic_processor import run
    out_paths = run(['CUSTOM/USSOCOM_2026-03-18.xlsx'], summarize=False)

    # CLI (pass relative file paths as arguments):
    python -m src.processors.topic_processor CUSTOM/USSOCOM_2026-03-18.xlsx
"""

import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd
from openai import OpenAI

from src.modules.Embedding.text_embedder import TextProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_file(path: str) -> pd.DataFrame:
    """Load CSV, XLSX, JSON, or Parquet by file extension."""
    ext = os.path.splitext(path)[1].lower()
    loaders = {
        '.csv':     pd.read_csv,
        '.xlsx':    pd.read_excel,
        '.xls':     pd.read_excel,
        '.json':    pd.read_json,
        '.parquet': pd.read_parquet,
    }
    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(f'Unsupported file type: {ext}')
    return loader(path)


def _summarize_description(
    description: str,
    openai_client: OpenAI,
    model: str = 'gpt-4o',
) -> str:
    """Summarise a grant description (≤5000 chars, tech-keyword focused)."""
    system = (
        'Using 5000 characters or less, summarise this grant description '
        'with a heavy focus on the technology and research terminology and keywords.'
    )
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user',   'content': description},
        ],
    )
    return resp.choices[0].message.content


def _fix_phone_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce mixed-type phone columns to str so parquet serialises cleanly."""
    for col in ['phone', 'phone.1']:
        if col in df.columns and df[col].dtype == 'object':
            if pd.to_numeric(df[col], errors='coerce').isna().any():
                df[col] = df[col].astype(str)
    return df


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def process_file(
    input_path: str,
    output_dir: str,
    tp: TextProcessor,
    openai_client: Optional[OpenAI] = None,
    summarize: bool = False,
    char_limit: int = 5000,
    description_col: str = 'description',
) -> str:
    """
    Process a single grant topic file.

    Steps:
      1. Load spreadsheet.
      2. Optionally summarise descriptions longer than *char_limit*.
      3. Generate embeddings for each topic.
      4. Save as parquet in *output_dir*.

    Args:
        input_path:      Path to the input file (CSV, XLSX, etc.).
        output_dir:      Directory to write the output parquet.
        tp:              Initialised TextProcessor instance.
        openai_client:   Required when summarize=True.
        summarize:       Whether to summarise long descriptions via GPT-4o.
        char_limit:      Character threshold above which to summarise.
        description_col: Column name containing the grant description text.

    Returns:
        Path to the output parquet file.
    """
    print(f'Processing: {input_path}')
    topics = _load_file(input_path)

    if description_col not in topics.columns:
        raise ValueError(
            f"Column '{description_col}' not found in {input_path}. "
            f"Available: {topics.columns.tolist()}"
        )

    topics['embeddings'] = None

    for idx, row in topics.iterrows():
        description = str(row[description_col])

        if summarize and len(description) > char_limit:
            if openai_client is None:
                raise ValueError('openai_client is required when summarize=True')
            print(f'  [{idx}] Summarising ({len(description)} chars)…')
            description = _summarize_description(description, openai_client)
            topics.at[idx, description_col] = description

        topics.at[idx, 'embeddings'] = tp.get_embedding(description)
        print(f'  Embedded {idx + 1}/{len(topics)}')

    topics = _fix_phone_columns(topics)
    topics['scraped_at'] = datetime.today().strftime('%Y-%m-%d')

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_path  = os.path.join(output_dir, f'{base_name}.parquet')
    os.makedirs(output_dir, exist_ok=True)
    topics.to_parquet(out_path, index=False)
    print(f'Exported → {out_path}')
    return out_path


def run(
    file_names: list[str],
    unprocessed_dir: str = 'data/all-topics/unprocessed',
    processed_dir: str   = 'data/all-topics/processed',
    summarize: bool      = False,
    char_limit: int      = 5000,
    openai_api_key: Optional[str] = None,
) -> list[str]:
    """
    Batch-process a list of relative file paths.

    Args:
        file_names:      Paths relative to *unprocessed_dir*, e.g.
                         ``['HHS/topics.xlsx', 'CUSTOM/foo.xlsx']``.
        unprocessed_dir: Root directory for input files.
        processed_dir:   Root directory for output parquets.
        summarize:       Whether to summarise long descriptions.
        char_limit:      Character threshold for summarisation.
        openai_api_key:  OpenAI key (falls back to OPENAI_API_KEY env var).

    Returns:
        List of output parquet paths.
    """
    api_key       = openai_api_key or os.environ['OPENAI_API_KEY']
    tp            = TextProcessor(api_key=api_key)
    openai_client = OpenAI(api_key=api_key) if summarize else None

    out_paths = []
    for file_name in file_names:
        agency     = file_name.split('/')[0]
        input_path = os.path.join(unprocessed_dir, file_name)
        output_dir = os.path.join(processed_dir, agency)

        out_paths.append(
            process_file(
                input_path=input_path,
                output_dir=output_dir,
                tp=tp,
                openai_client=openai_client,
                summarize=summarize,
                char_limit=char_limit,
            )
        )

    return out_paths


if __name__ == '__main__':
    files = sys.argv[1:] or ['CUSTOM/USSOCOM_2026-03-18.xlsx']
    run(files)
