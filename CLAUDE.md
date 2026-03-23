# CLAUDE.md — The Matcher

## Project Overview

**The Matcher** is a RAG (Retrieval-Augmented Generation) pipeline that matches companies/contacts to government grant programs (SBIR/STTR and similar). It ingests leads from sources like Apollo and SBA, processes grant topics from federal agencies, and uses OpenAI embeddings + cosine similarity + Claude/GPT LLM verification to identify strong company–grant alignments. Results are exported as Excel files for outreach campaigns.

The project is being **migrated from Google Colab notebooks + Google Drive** to a **Streamlit application** with consolidated Python modules. All `.ipynb` files are being converted to `.py` files.

---

## Current Architecture

```
src/
  modules/
    Embedding/
      text_embedder.py      # TextProcessor class — embeddings, chunking, normalization
    Storage/
      bucket_manager.py     # BucketManager class — GCS upload/download
    Scraping/
      web_scraper.py        # WebScraper class — Selenium-based scraper

notebooks/pipelines/
  leads-import/             # Lead importers (→ all-contacts/)
  data/all-topics/          # Grant topic processor (→ processed/)
  matching-contacts-grants/ # Core matcher
```

### Pipeline Stages

**Stage 1 — Lead Import** (contacts → embeddings)
- Raw CSVs (Apollo exports, SBA data, free alert lists) are loaded, normalized, and deduplicated
- Company websites are scraped (aiohttp fast pass → Playwright fallback) to get `page_text`
- OpenAI (`text-embedding-ada-002`) generates a `summary` and `embeddings` vector per company
- Output: `.parquet` files saved to `data/all-contacts/{source}/` with filename `{source}_{YYYY-MM-DD}.parquet`

**Stage 2 — Grant Topic Processing** (grants → embeddings)
- Grant topic spreadsheets (`.xlsx`) are loaded by agency folder (e.g., `all-topics/unprocessed/HHS/`)
- Long descriptions optionally summarized via Claude/GPT (5000 char limit)
- Embeddings generated per topic
- Output: `.parquet` files saved to `data/all-topics/processed/{AGENCY}/`

**Stage 3 — Matching**
- Loads grant topics and contacts from their respective parquet stores
- Computes cosine similarity (dot product on normalized embeddings) to find candidates above `starting_threshold` (typically `0.79`–`0.82`)
- Claude (Haiku or Sonnet) performs a binary yes/no alignment check on top candidates
- De-duplicates against recent matches (last 30 days) pulled from `all-matches/`
- Output: `.xlsx` files saved to `matches/matches/tier1/{YYYY-MM-DD}/`

---

## Source Files & Their Roles

### Existing `.py` modules (keep and extend)

| File | Class | Purpose |
|------|-------|---------|
| `text_embedder.py` | `TextProcessor` | OpenAI embeddings, text chunking, token reduction, normalization, LLM summarization. Constructor takes `api_key: str` directly — NOT a file path. |
| `bucket_manager.py` | `BucketManager` | Google Cloud Storage upload/download (parquet, CSV). Constructor signature: `BucketManager(bucket_path: str, client=None)` — always pass a `storage.Client` built from `get_storage_client()`. |
| `web_scraper.py` | `WebScraper` | Selenium-based website scraper — legacy, being phased out in favor of Playwright. |

### Notebooks being converted to `.py`

| Notebook | Target module | Purpose |
|----------|--------------|---------|
| `apollo_importer__1_.ipynb` | `importers/apollo_importer.py` | Ingest Apollo CSV exports, normalize columns, scrape websites, generate embeddings |
| `SBA_importer.ipynb` | `importers/sba_importer.py` | Ingest SBA CSV exports, normalize, scrape, embed |
| `fwee_alluts_impoatah.ipynb` | `importers/free_alert_importer.py` | Ingest "free alert" lead CSVs; uses async aiohttp + Playwright two-stage scraping |
| `topic_processor_v2__1_.ipynb` | `processors/topic_processor.py` | Process grant topic spreadsheets per agency, optionally summarize, generate embeddings |
| `contacts_grants_matcher_v3.ipynb` | `matcher/matcher.py` | Core matching engine: load data, similarity scoring, LLM verification, export results |

> **Note on filename:** `fwee_alluts_impoatah.ipynb` = "free alerts importer" (phonetic/scrambled). Canonical name going forward: `free_alert_importer.py`.

---

## Data Schemas

### Contact record (parquet, `all-contacts/`)
| Field | Type | Notes |
|-------|------|-------|
| `companyName` | str | Company name |
| `companyWebsite` | str | Full URL with protocol |
| `firstName` | str | |
| `lastName` | str | |
| `email` | str | |
| `phone` | str | |
| `segment` / `industry` | str | Industry/vertical |
| `summary` / `company_summary` | str | LLM-generated description from scraped page text |
| `embeddings` | list[float] | `text-embedding-ada-002` vector |
| `scraped_at` | str | ISO date of processing |
| `uuid` | str | Unique record ID |

### Grant topic record (parquet, `all-topics/processed/`)
| Field | Type | Notes |
|-------|------|-------|
| `topic_number` | str | Agency topic/solicitation ID |
| `agency` | str | e.g. `HHS`, `DOD`, `ARPA` |
| `title` | str | |
| `description` | str | Raw grant description text |
| `grant_summary` | str | LLM-summarized description (if summarized) |
| `embeddings` | list[float] | `text-embedding-ada-002` vector |
| `open_date` / `close_date` | datetime | |
| `scraped_at` | str | ISO date of processing |

### Match output (`.xlsx`, `all-matches/`)
Includes merged fields from both contact and grant records plus:
- `similarity_score` — cosine similarity value
- `good_match` — `"yes"` / `"no"` from LLM verification
- `alignment_reason` — LLM explanation (when present)

---

## GCS Bucket Structure

Bucket name: `cc-matcher-bucket-jeg-v1` (single-region, us-central1). All pipeline data lives here — no local filesystem writes in production.

```
cc-matcher-bucket-jeg-v1/
  all-topics/
    processed/
      DOD/
        ARMY_2026-03-01.parquet
        USSOCOM_2026-03-01.parquet
      HHS/
        NCI_2026-03-01.parquet
      ARPA/
        ...
  all-contacts/
    apollo/
      apollo_2026-03-01.parquet
    sba/
      sba_2026-03-01.parquet
    free_alert/
      free_alert_2026-03-01.parquet
  all-matches/
    campaign_name_2026-03-01.xlsx
```

### BucketManager usage pattern

```python
# Always instantiate with a client from get_storage_client()
bm = BucketManager("matcher-data", client=get_storage_client())

# Write
bm.upload_file("all-topics/processed/DOD/ARMY_2026-03-01.parquet", df)

# Read
df = bm.download_file("all-topics/processed/DOD/ARMY_2026-03-01.parquet")
```

### Listing GCS prefixes (replaces os.listdir for agency dropdowns)

```python
def list_broad_agencies(client) -> list[str]:
    bucket = client.bucket("matcher-data")
    blobs  = client.list_blobs("matcher-data", prefix="all-topics/processed/", delimiter="/")
    list(blobs)  # must consume iterator to populate prefixes
    return sorted(p.replace("all-topics/processed/", "").strip("/") for p in blobs.prefixes)
```

---



Agencies are referenced by short-code keys throughout the codebase:

`DOD`, `HHS`, `ARPA`, `DOE`, `NOAA`, `DOC`, `DOT`, `NAVAIR`, `SERDP`, `CPRIT`, `DHS`, `AFOSR`, `MTEC`, `EU-GRANTS`, `ED`, `EIC`, `GRANTS-GOV`, `SBA`, `CUSTOM`

Each agency entry in the matcher's `grants` dict has this structure:
```python
grants['DOD'] = {
    'status': True,       # Whether to include this agency in the current run
    'priority': 1,        # Processing order (lower = higher priority)
    'standard_topics': True,
    'custom_topics': False,
    'topics': pd.DataFrame(...)  # Loaded at runtime
}
```

---

## Key Dependencies

| Package | Use |
|---------|-----|
| `openai` | Embeddings (`text-embedding-ada-002`), GPT-3.5/4 summarization |
| `anthropic` | Claude (Haiku, Sonnet) for match verification — async preferred |
| `tiktoken` | Token counting before embedding (7500 token limit) |
| `playwright` | Async JS-rendered page scraping (fallback when aiohttp fails) |
| `aiohttp` + `BeautifulSoup` | Fast async scraping (first-pass) |
| `selenium` | Legacy scraper in `web_scraper.py` (being phased out in favor of Playwright) |
| `google-cloud-storage` | GCS bucket I/O via `BucketManager` |
| `duckdb` | In-notebook data querying (used in importers) |
| `pandas`, `numpy` | Data manipulation throughout |
| `tldextract` | Domain normalization |
| `streamlit` | **Target UI framework** |

---

## Secrets / API Keys

All secrets are loaded via `st.secrets` — never from `.txt` files, never hardcoded. The old `_load_key(filename)` helper pattern must not be used in any new code.

### `.streamlit/secrets.toml` (local dev — gitignored)

```toml
openai_api_key    = "sk-..."
anthropic_api_key = "sk-ant-..."

[gcp_service_account]
type                        = "service_account"
project_id                  = "your-project-id"
private_key_id              = "..."
private_key                 = "-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----\n"
client_email                = "matcher-app@your-project.iam.gserviceaccount.com"
client_id                   = "..."
auth_uri                    = "https://accounts.google.com/o/oauth2/auth"
token_uri                   = "https://oauth2.googleapis.com/token"
token_uri                   = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url        = "..."
```

The `[gcp_service_account]` block is the contents of `ServiceKey_GoogleCloud.json` reformatted as TOML. When deploying to Streamlit Cloud, paste the same values into **App Settings → Secrets** in the UI.

### Accessing secrets in code

```python
import streamlit as st

oai_key  = st.secrets["openai_api_key"]
anth_key = st.secrets["anthropic_api_key"]
```

### GCS client (always build this way)

```python
from google.oauth2 import service_account
from google.cloud import storage

def get_storage_client():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return storage.Client(credentials=creds)
```

Pass this client into `BucketManager` — never rely on the `GOOGLE_APPLICATION_CREDENTIALS` env var in Streamlit code.

### Secret migration reference

| Secret | Old pattern | New pattern |
|--------|------------|-------------|
| OpenAI API key | `keys/openai_key.txt` | `st.secrets["openai_api_key"]` |
| Anthropic API key | `keys/anthropic.txt` | `st.secrets["anthropic_api_key"]` |
| GCP service account | `ServiceKey_GoogleCloud.json` file | `st.secrets["gcp_service_account"]` dict |

---

## Migration Goals (Colab → Streamlit)

1. **Remove all `google.colab` imports** — no `drive.mount()`, no `!pip install`, no `%cd` magic
2. **Convert each notebook to a standalone `.py` module** with a clear class or set of functions (no top-level execution code outside `if __name__ == "__main__":`)
3. **Centralize config** — paths, thresholds, model names, and agency lists should come from a single config file (e.g., `config.py` or `config.yaml`), not be hardcoded inline
4. **Use `st.secrets` for all API keys and GCP credentials** — no reading from `.txt` files, no `GOOGLE_APPLICATION_CREDENTIALS` env var in Streamlit code
5. **All data I/O goes through `BucketManager` against GCS** — no local filesystem reads or writes in production; no `glob` over local paths
6. **`TextProcessor` takes `api_key: str` directly** — not a file path; update constructor and all call sites
7. **`BucketManager` takes an optional `client` param** — always pass a `storage.Client` built from `get_storage_client()`; never rely on ambient env var auth in Streamlit code
8. **Preserve the two-stage scraping pattern** — aiohttp fast pass → Playwright fallback is the standard; do not revert to Selenium or collapse to a single scraper
9. **Async-first for LLM calls** — use `AsyncAnthropic` / `AsyncOpenAI` wherever batch processing occurs
10. **Streamlit UI** should expose:
    - Lead importer runner (select source, upload CSVs, trigger pipeline)
    - Grant topic processor (select agency, upload files, trigger processing)
    - Matcher runner (configure grants/contacts/threshold, run, preview results, download Excel)
    - Match history browser

---

## Naming Conventions

- **Files:** `snake_case.py`
- **Classes:** `PascalCase` (e.g., `TextProcessor`, `BucketManager`)
- **Contact fields:** camelCase for legacy compatibility (`companyWebsite`, `companyName`, `firstName`, `lastName`) — preserve these names to avoid breaking downstream column references
- **Grant fields:** snake_case (`grant_summary`, `open_date`, `close_date`, `scraped_at`)
- **Parquet output filenames:** `{source_or_agency}_{YYYY-MM-DD}.parquet`
- **Match output filenames:** `{campaign_name}_{YYYY-MM-DD}.xlsx`
- **Agency keys:** UPPERCASE short-codes (e.g., `DOD`, `HHS`, `ARPA`)

---

## Common Patterns

### Loading parquet files from GCS prefix

```python
import io, pandas as pd

def load_parquets_from_prefix(bm: BucketManager, prefix: str) -> pd.DataFrame:
    blobs = bm.storage_client.list_blobs(bm.bucket.name, prefix=prefix)
    frames = [pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
              for blob in blobs if blob.name.endswith('.parquet')]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
```

### Cosine similarity matching (vectorized)
```python
# grant_topics['embeddings'] and contacts['embeddings'] are lists of floats
import numpy as np

contact_matrix = np.stack(contacts['embeddings'])  # shape: (n_contacts, 1536)
grant_topics['similarity_scores'] = grant_topics['embeddings'].apply(
    lambda x: np.dot(contact_matrix, x)
)
```

### LLM match verification (async Claude)
```python
system = "Tell me if this company summary and grant summary are aligned. Only give a one-word answer. Either \"yes\" or \"no\"."
response = await anth_client_async.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=10,
    messages=[{"role": "user", "content": f"Company: {company_summary}\n\nGrant: {grant_summary}"}],
    system=system
)
result = response.content[0].text.strip().lower()
```

### De-duplicating against recent matches (from GCS)

```python
from datetime import datetime, timedelta
import re, io
import pandas as pd

def load_recent_matches(bm: BucketManager, days: int = 30) -> pd.DataFrame:
    cutoff = datetime.now() - timedelta(days=days)
    blobs  = bm.storage_client.list_blobs(bm.bucket.name, prefix="all-matches/")
    frames = []
    for blob in blobs:
        m = re.search(r'(\d{4}-\d{2}-\d{2})', blob.name)
        if m and datetime.strptime(m.group(1), '%Y-%m-%d') >= cutoff:
            frames.append(pd.read_parquet(io.BytesIO(blob.download_as_bytes())))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
```
