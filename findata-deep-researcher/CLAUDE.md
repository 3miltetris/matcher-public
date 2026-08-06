# CLAUDE.md — Bulk Company Research Pipeline (ChatGPT Deep Research)

## Project Overview

A Streamlit web application that accepts a CSV of companies and uses the **OpenAI Responses API with Deep Research** to autonomously research each company and produce a structured CSV output — one row per company — suitable for sales qualification and outreach prioritization for a proposal writing and grant support services firm.

This is the Deep Research variant of the pipeline. Unlike the scraping-based version, this delegates all data gathering to OpenAI's Deep Research model, which browses the web autonomously. The tradeoff: simpler infrastructure, higher cost per company, slower throughput, less control over sources.

The report structure is based on the detailed financial diligence prompt in `prompt_template.md`.

---

## When to Use This Version vs. the Scraping Version

| Factor | This version (Deep Research) | Scraping version |
|---|---|---|
| Infrastructure complexity | Low | High |
| Cost per company | ~$5–$20 | ~$0.02–$0.05 |
| Speed per company | 2–10 minutes | 30–90 seconds |
| Batch throughput | Low (5–10 concurrently max) | High (50+ concurrently) |
| Source coverage | Broad (GPT browses freely) | Structured (fixed API targets) |
| Consistency | Variable | Consistent |
| Best for | Smaller batches, hard-to-find companies | Large batches, companies with federal award history |

---

## Target User

Non-technical consulting staff who will upload a CSV, click Run, monitor progress, and download results.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| AI Research | OpenAI Responses API (`o3` or `o4-mini` with web search) |
| Async HTTP | `aiohttp` + `asyncio` |
| Checkpoint store | SQLite via `sqlite3` (stdlib) |
| Output | CSV via `pandas`, optional Word docs via `python-docx` |
| Config | `python-dotenv` for local dev; Streamlit secrets for deployment |

---

## Environment Variables

```
OPENAI_API_KEY=
```

Store in `.env` for local dev. Document in `.env.example`. No other API keys required — Deep Research handles all sourcing.

---

## Project File Structure

```
/
├── CLAUDE.md                  # This file
├── .env.example
├── .gitignore
├── requirements.txt
├── prompt_template.md         # Full diligence prompt — source of truth for output schema
├── app.py                     # Streamlit entry point
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py        # Async batch runner; manages concurrency + checkpointing
│   ├── researcher.py          # OpenAI Deep Research API wrapper (one call per company)
│   ├── parser.py              # Parses Deep Research response into structured output row
│   ├── checkpoint.py          # SQLite read/write helpers
│   └── schema.py              # Pydantic models for output row
├── output/
│   └── .gitkeep
└── tests/
    └── test_parser.py
```

---

## Input CSV Format

Required column: `company_name`

Optional columns (include if available — improves research quality by giving the model more anchors):

| Column | Description |
|---|---|
| `company_name` | **Required** |
| `website` | Company domain or full URL |
| `uei` | SAM.gov Unique Entity Identifier |
| `address` | Street address |
| `city` | City |
| `state` | State (2-letter) |
| `country` | Country (default: USA) |
| `alt_names` | Comma-separated alternate/DBA names |

---

## Output CSV Schema

Identical to the scraping pipeline version. One row per company, ~45 columns.

### Identity
- `company_name_resolved`
- `uei`
- `sam_status` — Active / Inactive / Not found
- `website_resolved`
- `naics_codes`
- `entity_type`

### Financial Findings
- `revenue_estimate`
- `revenue_method`
- `revenue_year`
- `total_venture_funding`
- `total_grant_funding`
- `estimated_valuation`
- `valuation_method`

### Federal Funding (last 3 years)
- `federal_awards_count_3yr`
- `federal_awards_total_3yr`
- `federal_agencies_3yr`
- `sbir_sttr_count_3yr`
- `sbir_sttr_total_3yr`
- `latest_award_date`
- `award_detail_json`

### Headcount
- `employee_count_current`
- `employee_count_source`
- `headcount_trend`

### Financial Health
- `customer_concentration_signal`
- `hiring_trend_signal`
- `leadership_stability_signal`
- `litigation_bankruptcy_signal`
- `sam_certifications`

### Grant Activity
- `grant_activity_classification`
- `phase_progression_detected`
- `multi_agency_activity`
- `grant_momentum`
- `estimated_proposals_per_year`

### Discretionary Budget Signals
- `signal_recent_grant_inflows`
- `signal_hiring_activity`
- `signal_new_products`
- `signal_conference_participation`
- `signal_rd_programs`
- `signal_facility_expansion`

### Proposal Readiness Score (0–100)
- `score_total`
- `score_funding_activity` — /30
- `score_momentum` — /20
- `score_org_capacity` — /20
- `score_budget_signals` — /20
- `score_fit_external_support` — /10

### Qualification
- `internal_proposal_capability`
- `proposal_budget_estimate`
- `recommendation`
- `outreach_angle`
- `outreach_triggers`
- `risks_red_flags`
- `confidence_score`
- `confidence_notes`

### Meta
- `sources_used`
- `processing_timestamp`
- `processing_duration_seconds`
- `estimated_cost_usd`

---

## Core Component: `researcher.py`

This is the heart of the pipeline. Each company gets one Deep Research API call.

### API Details

Use the **OpenAI Responses API** with web search enabled:

```python
import openai

client = openai.OpenAI(api_key=OPENAI_API_KEY)

response = client.responses.create(
    model="o3",                          # or "o4-mini" for lower cost
    tools=[{"type": "web_search_preview"}],
    input=build_research_prompt(company)
)
```

> **Note:** Deep Research via the Responses API uses the `o3` model with `web_search_preview`. If OpenAI updates model names or tool names, check `https://platform.openai.com/docs/guides/deep-research` for the current identifiers. Do not assume model strings — verify against live OpenAI docs if there are any API errors.

### Response Handling

The response object contains the full research output as text. Extract it:

```python
output_text = response.output_text  # Full research response as a string
```

Check `response.usage` for token counts and estimate cost:
```python
input_tokens = response.usage.input_tokens
output_tokens = response.usage.output_tokens
```

### Prompt Design (`build_research_prompt` in `researcher.py`)

Build a single prompt per company that:
1. Identifies the company with all available input fields
2. Instructs the model to research all required questions
3. Instructs the model to return a JSON object matching the output schema
4. Embeds the full schema definition inline

**System-style instruction (embed at top of prompt):**
```
You are a financial diligence analyst. Research the company described below using 
public sources. Return ONLY a valid JSON object — no preamble, no markdown, no 
code fences — that strictly follows the schema provided.

Rules:
- Label each value's confidence as: Exact | Reported | Third-party | Estimated | Not found
- Do not invent funding amounts, revenue, or valuations
- If a value is unavailable, use a bounded estimate with method noted, or "Not found"
- Prioritize sources: company website > SAM.gov > USAspending > SBIR.gov > NIH RePORTER > NSF > state registries > news
- Prioritize recency; weight last 12 months more heavily
- Apply the scoring rubric exactly as specified in the schema
- For sources_used, list every URL or source you consulted
```

**Company context block:**
```
COMPANY TO RESEARCH:
Name: {company_name}
Website: {website or "Unknown"}
UEI: {uei or "Unknown"}
Address: {address, city, state or "Unknown"}
Alternate names: {alt_names or "None"}
```

**Schema block:** Paste the full output schema column definitions inline so the model knows exactly what JSON keys and value formats to return.

**Research questions block:** Paste the 11 core questions from `prompt_template.md` so the model knows what to look for.

---

## Parser (`parser.py`)

After `researcher.py` returns the raw text response:

1. Strip any accidental markdown fences (` ```json `, ` ``` `)
2. `json.loads()` the cleaned text
3. Validate against the Pydantic output schema
4. On validation failure:
   - Log the raw response
   - Attempt a repair call: send the malformed JSON back to `gpt-4o` (not o3 — cheaper) with instruction to fix and return valid JSON only
   - If repair fails, write an error row with `status = error` and store raw response for manual review

---

## Orchestrator (`orchestrator.py`)

### Flow (per company)
1. Check SQLite checkpoint — if `status = complete`, skip
2. Call `researcher.py` → get raw response text
3. Call `parser.py` → get validated output dict
4. Write completed row to SQLite
5. Emit progress update to Streamlit callback

### Concurrency
- Default: **3 companies in parallel** (Deep Research is slow and expensive — do not set this high)
- Configurable via Streamlit sidebar: 1–5 max
- Use `asyncio.Semaphore` to cap concurrent requests
- Each company call can take 2–10 minutes; set timeout at 15 minutes before marking as error

### Checkpointing
- SQLite table: `results(run_id, company_name, status, output_json, raw_response, error, cost_usd, timestamp)`
- `status` values: `pending` / `running` / `complete` / `error`
- On rerun with same CSV: skip rows with `status = complete`
- Store `raw_response` always — allows manual inspection and re-parsing without re-running the expensive API call
- User can force full re-run via UI toggle

### Error Handling
- Per-company try/except — one failure never blocks the batch
- Timeout errors get `status = timeout`; user can retry individual rows
- Rate limit / quota errors pause the queue and surface a warning in the UI

---

## Cost Management

Deep Research is expensive. Build cost controls into the UI:

**Before run:**
- Display estimated cost range based on company count and selected model
  - `o3`: ~$10–$20 per company (estimate; verify current pricing at platform.openai.com)
  - `o4-mini`: ~$2–$5 per company (estimate)
- Require user to confirm before starting runs over $50 estimated total

**During run:**
- Track actual token usage from each `response.usage`
- Display running cost total in sidebar
- Provide a "Pause" button that stops queuing new companies after current in-flight calls complete

**Model selector in sidebar:**
- `o3` — Higher quality, higher cost (recommended for small batches / high-value prospects)
- `o4-mini` — Lower cost, slightly less thorough (recommended for large batches / initial screening)

---

## Streamlit App (`app.py`)

### Layout
```
Title + subtitle
---
Sidebar:
  - OpenAI API key input (pre-filled from env if set)
  - Model selector (o3 / o4-mini)
  - Concurrency slider (1–5)
  - Word doc output toggle
  - Force re-run toggle
  - Estimated cost display
  - Running cost display (updates live)
  - Pause button (appears during run)

Main area:
  Tab 1: Upload
    - CSV uploader
    - Column mapping preview (auto-detect or manual map)
    - Validation warnings
    - Estimated cost + confirmation checkbox for large batches
    - Run button

  Tab 2: Progress
    - Live table: company | status | score | recommendation | cost | duration
    - Progress bar
    - Error count + expandable error list with "Retry" per row

  Tab 3: Results
    - Preview of output dataframe (paginated)
    - Download CSV button
    - Download Word docs ZIP button (if enabled)
    - Total cost summary
```

### State Management
Use `st.session_state` for:
- `run_id` — UUID for current job
- `results_df` — accumulated output rows
- `is_running` — flag
- `is_paused` — flag
- `cost_tracker` — cumulative API cost (dollars)

---

## Word Doc Output (optional)

If enabled, generate one `.docx` per company using `python-docx`. Use the parsed output dict — no additional AI call needed, since Deep Research already produced all the content.

Format: one-pager with sections A through P from `prompt_template.md`. Zip all docs for bulk download.

---

## `requirements.txt`

```
streamlit
openai
aiohttp
pandas
python-dotenv
pydantic
python-docx
```

No Playwright, BeautifulSoup, or SerpAPI needed — Deep Research handles all sourcing.

---

## `prompt_template.md`

Create this file alongside `CLAUDE.md`. It should contain the full original diligence prompt verbatim (provided by the user). This is the source of truth for:
- What questions to answer (sections 1–11)
- How to classify and score
- Output format definitions

The `researcher.py` prompt builder references this file directly.

---

## Build Order

Build and test in this sequence:

1. **Schema + models** (`schema.py`) — define all Pydantic output models first; keep identical to scraping version
2. **Checkpoint store** (`checkpoint.py`) — SQLite helpers; keep identical to scraping version
3. **Researcher** (`researcher.py`) — build prompt constructor + API call wrapper; test with one known company
4. **Parser** (`parser.py`) — build JSON extractor + validator + repair logic; test with real API responses
5. **Orchestrator** (`orchestrator.py`) — wire together with checkpointing; test with 3-company batch
6. **Streamlit app** (`app.py`) — build UI with cost controls; connect to orchestrator
7. **Word doc generator** — optional, build last

---

## Testing Notes

- Test with a company that has a known, verifiable SBIR history (look one up on sbir.gov first) — this gives ground truth to validate output quality
- Test parser against intentionally malformed JSON to validate repair logic
- Run a 3-company pilot and manually audit every output field before using on full batches
- Compare a few results against the scraping pipeline version to calibrate quality differences
- Log all raw responses during testing — Deep Research output is verbose and inspecting it helps tune the prompt

---

## Known Constraints / Gotchas

- **Model name drift** — OpenAI changes model identifiers frequently. If API calls fail with a model error, check `https://platform.openai.com/docs/models` for current names. Do not hardcode without verifying.
- **Response API vs. Chat Completions** — Deep Research uses the Responses API (`client.responses.create`), not the standard Chat Completions API (`client.chat.completions.create`). These are different endpoints with different response shapes.
- **JSON compliance** — Even with strict instructions, o3 sometimes wraps output in markdown. Always strip fences before parsing.
- **Timeout risk** — Deep Research calls for complex companies can run 5–10 minutes. Set `asyncio` timeout at 15 minutes and surface a clear timeout error in the UI rather than hanging.
- **Cost overruns** — Without the confirmation gate and pause button, a 100-company batch could cost $500–$2,000 unattended. These controls are not optional.
- **Source citation quality** — Deep Research will cite sources, but they may not always be stable URLs. Store raw responses in SQLite so citations can be reviewed even after the run.
- **Rate limits** — OpenAI imposes rate limits on Deep Research / o3 requests. Implement exponential backoff on 429 errors and surface a clear message if the user hits their quota.
