# CLAUDE.md — The Matcher

## Project Overview

**The Matcher** is a RAG (Retrieval-Augmented Generation) pipeline that matches companies/contacts to government grant programs (SBIR/STTR and similar). It ingests leads from sources like Apollo and SBA, processes grant topics from federal agencies, and uses OpenAI embeddings + cosine similarity + Claude/GPT LLM verification to identify strong company–grant alignments. Results are exported as CSV files for outreach campaigns.

The project has been **migrated from Google Colab notebooks + Google Drive** to a **Streamlit application** with consolidated Python modules.

---

## Current Architecture

```
app.py                        # Streamlit entry point — auth gate, playwright install, navigation
packages.txt                  # Streamlit Cloud apt packages (Chromium system libs for Playwright)
views/
  contact_importer.py         # Upload a lead spreadsheet OR pull a HubSpot company list → map columns → dedup (per-source or all-sources) → pick profiling method (scrape or Deep Research tech focus) → trigger contact-import-job → poll status
  client_editor.py            # Edit a client company summary → re-embed → rewrite source parquet in data/all-contacts/clients/; admin-only delete of clients (rows + profile + Drive assignment)
  finance_researcher.py       # Client Research — OpenAI Deep Research on client companies, two focuses: financials or technology/R&D, written back onto client rows
  topic_importer.py           # Upload PDF/text solicitations → extract topics → embed → save
  grant_search.py             # Cosine-similarity search across processed grant topics
  bulk_matching.py            # Configure + trigger Cloud Run matching job, poll status
  sam_gov_upload.py           # Upload SAM.gov CSVs or fetch from API → dedup vs store → Claude screening → embed → save
  grants_gov_fetch.py         # Query Grants.gov public search2 API → Claude Haiku screening → embed → save to GRANTS-GOV/
  hubspot_import.py           # Import companies to HubSpot via Imports API — from a matching run or a financial research run (per-field mapping)
  resume_importer.py          # Upload HubSpot contacts CSV → fetch resumes by URL → extract text → GPT expertise summary → embed → save
  resume_search.py            # Natural-language search across resume embeddings → ranked candidate list
  drive_sync.py               # Scan the client Google shared drive → auto-assign folders to clients (fuzzy match, saved assignments) → trigger drive-sync-job → results + new-client review queue
  client_profiler.py          # Pick clients + sources → trigger client-profile-job (multi-aspect capability profiles from website summary / Drive docs / Deep Research) → poll status; review/edit aspects in-process → data/client-profiles/profiles.parquet
  aspect_match.py             # Bulk multi-aspect match: every client aspect × every selected grant topic → threshold/coverage filter → Claude LLM re-rank → CSV
  suggestions.py              # Team feature-request board with upvoting
  admin_portal.py             # Admin-only: view/add/remove admins (admin-config/admins.json); super admins are a code constant

src/
  modules/
    Embedding/
      text_embedder.py        # TextProcessor class — embeddings, chunking, normalization
    GoogleBucketManager/
      bucket_manager.py       # BucketManager class — GCS upload/download
    Scraping/
      web_scraper.py          # WebScraper class — Selenium-based scraper (legacy)
    email_generator.py        # async_generate_subject_line, async_josiah_copy
    grant_utils.py            # normalize_grant_columns() — coalesces description → grant_summary at load time
    finance_research.py       # Deep Research prompt/schema, JSON parse+repair, digest, cost accounting (Client Research view — financial focus + shared plumbing)
    tech_research.py          # Technology & R&D research schema/prompt/digest (Client Research view — tech focus; reuses finance_research parse/pricing)
    aspect_profile.py         # Multi-aspect client profile schema/prompt/parse + per-company material merge + flat-packed aspect embeddings + profiles.parquet store I/O — shared by Client Profiles + Bulk Aspect Match views and client-profile-job
    access_control.py         # Admin gating for destructive actions — SUPER_ADMINS code constant + admin-config/admins.json; is_admin()/require_admin() (Streamlit-only)
    client_delete.py          # Streamlit-free client deletion — archive removed rows → rewrite/delete clients parquets → drop profile → park Drive assignment; shared by Client Editor + Client Profiles
    doc_extract.py            # Streamlit-free document text extraction (PDF/DOCX/XLSX/TXT/CSV) — shared by Drive Sync view + drive-sync-job
    drive_client.py           # Streamlit-free Google Drive v3 helpers (shared-drive listing, export/download, backoff) — shared by Drive Sync view + drive-sync-job
    lead_importer.py          # Shared scraping/summarization/embedding helpers (reference; not imported by jobs)

jobs/
  matching_job.py             # Cloud Run Job — scoring, AI validation, email pre-write
  sam_gov_job.py              # Cloud Run Job — SAM.gov fetch metadata → screen (title+NAICS) → dedup → fetch descriptions for survivors → summarize → embed → save
  contact_import_job.py       # Cloud Run Job — download staged file → map cols → dedup → profile (scrape+GPT or Deep Research tech focus) → embed → save contacts
  drive_sync_job.py           # Cloud Run Job — Drive scan per assigned client → diff vs sync_state → extract changed docs → Claude profile merge → write docs columns + re-embed summary
  client_profile_job.py       # Cloud Run Job — merge each client's material rows → Claude aspect split (concurrent) → embed each aspect → upsert data/client-profiles/profiles.parquet
  Dockerfile                  # Image for matching-job (uses requirements.job.txt)
  Dockerfile.sam_gov          # Image for sam-gov-job (uses requirements.sam_gov_job.txt)
  Dockerfile.contact_import   # Image for contact-import-job — python:3.11-slim + Chromium system libs + playwright install chromium; also COPYs src/modules/{finance,tech}_research.py (shared Deep Research schema)
  Dockerfile.drive_sync       # Image for drive-sync-job — python:3.11-slim, no Chromium; COPYs src/modules/{doc_extract,drive_client}.py
  Dockerfile.client_profile   # Image for client-profile-job — python:3.11-slim; COPYs src/modules/{aspect_profile,finance_research,tech_research}.py + GoogleBucketManager/bucket_manager.py

requirements.txt                    # Streamlit app dependencies (includes aiohttp, tldextract, playwright, google-api-python-client, openpyxl)
requirements.job.txt                # matching-job dependencies (lean — no Streamlit)
requirements.sam_gov_job.txt        # sam-gov-job dependencies (adds requests, beautifulsoup4, tiktoken, pymupdf)
requirements.contact_import_job.txt # contact-import-job dependencies (aiohttp, playwright, tldextract, openpyxl, tiktoken, openai, pandas/numpy/pyarrow, GCS)
requirements.drive_sync_job.txt     # drive-sync-job dependencies (anthropic, openai, google-api-python-client, pymupdf, python-docx, openpyxl, tiktoken)
requirements.client_profile_job.txt # client-profile-job dependencies (anthropic, openai, tiktoken, pandas/numpy/pyarrow, GCS)
```

### Pipeline Stages

**Stage 1 — Lead Import** (contacts → embeddings)
- **Via Contact Importer view:** Upload any CSV/Excel spreadsheet **or pull a HubSpot company list** (lists search → memberships → batch company read; fetched companies are staged as a standard-columns CSV so the rest of the flow is identical), map columns to standard fields (URL, company name, state, name, email, phone, industry), dedup against existing GCS records by bare domain (per-source folder, or all sources via checkbox — default ON for HubSpot pulls), then trigger the `contact-import-job` Cloud Run Job. The job builds the company profile per the selected **profiling method** — `scrape` (default: aiohttp → Playwright scrape, GPT-3.5-turbo summary) or `deep_research` (one background OpenAI Deep Research task per unique company domain, technology focus, using the shared `tech_research.py` schema; the matching summary is `build_matching_summary()` of the findings and the full output is stored in `technology_data`/`technology_summary`/`technology_updated_at` columns; failed/deadline-exceeded companies are skipped and re-importable later) — then embeds with `text-embedding-ada-002` and saves to GCS. Deep Research mode shows a per-company cost estimate in the UI (unique domains × model rate) with a >$50 confirmation checkbox, and reports actual cost in the status payload. Streamlit polls `contact-import-jobs/{run_id}/status.json` for completion.
- Supported sources (selectable in UI): `apollo`, `sba`, `free_alert`, or any custom label
- Excel HYPERLINK formulas (e.g. `=HYPERLINK("url", "COMPANY NAME")` from SBA exports) are automatically stripped to plain text in all mapped columns.
- Output: `.parquet` files saved to `data/all-contacts/{source}/` with filename `{source}_{YYYY-MM-DD}_{hex6}.parquet`

**Stage 2 — Grant Topic Processing** (grants → embeddings)
- **Via Topic Importer view:** PDF/text solicitations are parsed by Claude, reviewed/edited in the UI, then embedded and saved to `data/all-topics/processed/{BROAD_AGENCY}/`
- **Via Grants.gov Fetch view:** query the Grants.gov public `search2` API (no API key required) with keyword, posted-date range, opportunity status, funding instrument, and agency filters → Claude Haiku screens each row for relevance → passing rows embedded and saved to `data/all-topics/processed/GRANTS-GOV/`
- **Via SAM.gov Upload view:** Three modes — (1) CSV uploads deduped against the existing store (notice ID + title, before any Claude calls), then screened, summarized, and embedded in Streamlit; (2) manual API fetch that writes a config to GCS and triggers `sam-gov-job`, with Streamlit polling `sam-gov-jobs/{run_id}/status.json`; (3) **daily schedule** — configure `lookback_days` + filters in the "Daily API Parameters" section, save to `sam-gov-configs/daily_schedule.json`, and Cloud Scheduler fires `sam-gov-job` daily at 5 AM CST. Results saved to `data/all-topics/processed/SAM-GOV/`.
- **`sam-gov-job` pipeline order (API mode):** Fetch all record metadata (no descriptions) → Claude screens on title + NAICS → dedup vs existing store → **then** fetch full descriptions from SAM.gov only for rows that passed screening and dedup (typically 20–40% of total, avoiding 3–5× unnecessary API calls). A global 8 req/s rate limiter (proactive token bucket, shared across all workers) prevents thundering-herd 429s. CSV mode is unaffected — descriptions come from the uploaded file.
- **Revision handling (amended notices, e.g. revised CSOs):** SAM.gov amendments publish a new version with a new `noticeId` but the same `solicitationNumber` (our `topic_number`). Two mechanisms keep the store current:
  - **Ingest-time (API mode, automatic — including the daily run):** dup rows whose fetched `noticeId` differs from the stored `notice_version_id` are routed to an update path instead of being dropped — new description + attachment PDF text fetched, Claude diffs old vs new content (topics added/removed → `sam_revision_notes`), content re-summarized/re-embedded only when substantively changed (deadline-only amendments keep the old embedding), and the stored parquet rows are rewritten in place.
  - **`revision_check` job mode (manual budgeted sweep):** triggered from the "🔁 Revision Check" expander in the SAM.gov Upload view. Looks up stored open notices by `solnum` (walking back one-year `postedFrom`/`postedTo` windows — the API requires them and only returns the latest active version), updates revised notices, and marks notices no longer on SAM.gov with `sam_status='archived'` (rows kept; `matching_job` and Grant Search filter them out). Supports `dry_run` (report only — the UI default) so the report can be reviewed before applying. **SAM.gov enforces a hard 1,000 requests/day quota per API key** (shared with the daily fetch; the API exposes no `updatedDate` field, so each notice costs ≥1 lookup call), so each run spends at most `max_api_calls` (UI default 600) and sweeps **least-recently-checked notices first** using a cursor persisted at `sam-gov-configs/revcheck_state.json`. The cursor advances only on apply (non-dry) runs — a dry run and the apply run that follows cover the same chunk. A quota-exhaustion 429 ("exceeded your quota", Retry-After at midnight UTC) aborts the sweep immediately via `QuotaExhaustedError` instead of retrying (throttle 429s still back off); revisions detected but not yet content-fetched when quota dies are deferred (no partial writes) and re-detected next run. A full sweep of a large store completes over several daily runs.

**Stage 3 — Matching**
- Streamlit's Bulk Matching view writes a job config JSON to `job-configs/` in GCS and triggers the Cloud Run job via the `google-cloud-run` API
- The job loads grant topics from `data/all-topics/processed/` and contacts from `data/all-contacts/`
- Computes cosine similarity (vectorized numpy dot product) to find candidates above threshold (minimum `0.82`)
- Claude Haiku performs a binary yes/no alignment check on top candidates (async, batched)
- Optionally pre-writes subject lines and email copy via `email_generator.py`
- Output: CSV segments saved to `matching-results/{run_id}/segment_NNN.csv`; completion signalled by `matching-results/{run_id}/status.json`

**Stage 4 — Resume Pipeline** (individual contacts → expertise embeddings)
- **Via Resume Importer view:** Upload a HubSpot contacts CSV that includes a resume URL column → fetch each file (PDF or DOCX) from HubSpot using the private app token as a Bearer header → extract text → GPT-3.5-turbo expertise summary → embed → save to `data/resumes/`
- Dedup by email (lowercase) against existing parquets in `data/resumes/`
- Minimum 400 chars of extracted text required before summarizing — prevents hallucination from header-only extractions
- **Via Resume Search view:** Embed a natural-language query → cosine similarity against all resume parquets → ranked result cards with download. Optional include keyword (single term) and comma-separated exclude keywords pre-filter the pool before scoring.
- Embeddings are per-person (not per-company); join key is `email`

**Stage 5 — HubSpot Import** (match results, financial research, or client profiles → CRM)
- The HubSpot Import view has three source modes selected by a radio at the top:
  - **Matching run** — loads all segment CSVs for a completed matching run and imports the standard `matcher_*` property set (original flow)
  - **Financial research run** — loads a completed Client Research financial-focus run (`finres_*`) from `finance-research-runs/{run_id}/state.json` (technology runs in `tech-research-runs/` are not listed here) (one row per researched company: identity + `financial_summary` digest + all 54 research fields). A `st.data_editor` mapping table lets users toggle each financial field on/off and point it at either an **existing** writable HubSpot company property (fetched live from the portal) or a new auto-created `matcher_fin_<field>` property (textarea for long-form fields). Headline fields are pre-checked by default; duplicate property targets are rejected before submit.
  - **Client profiles** — loads `data/client-profiles/profiles.parquet` (Stage 8) and flattens each company's aspect array into importable columns: `profile_summary`, `aspect_labels` (` | `-joined), `aspects_full` (numbered `label (kind)` / text / `Keywords:` block), `aspect_keywords` (deduped across aspects, first spelling wins), `aspect_kinds`, `n_aspects`, `sources_used`, `profile_model`, `profile_built_at`, plus `aspect_{i}_label` / `aspect_{i}_text` per aspect (off by default; companies with fewer aspects get empty strings). A client multiselect (All / None buttons, everything selected by default) picks which profiles go over, then the same mapping table as financial mode targets existing properties or auto-creates `matcher_profile_summary`, `matcher_aspect_labels`, `matcher_aspects_full`, `matcher_aspect_keywords`, `matcher_aspect_count`, `matcher_aspect_{i}_*`, etc. Multi-line values are quoted CSV fields — HubSpot preserves the newlines.
- All modes submit via the CRM Imports API as **Company** objects, deduplicating by `domain` (`companyWebsite`); rows without a website are skipped. The mapping table refuses a target of `name` or `domain` (already mapped from the company name/website columns) and rejects duplicate targets before submit.
- Standard properties: `domain`, `name`, `description` (from `company_summary`)
- Custom properties auto-created on first run (prefixed `matcher_`): `matcher_source`, `matcher_topic_number`, `matcher_grant_title`, `matcher_agency`, `matcher_broad_agency`, `matcher_due_date`, `matcher_grant_summary`, `matcher_good_match`, `matcher_subject_line`, `matcher_ai_message`
- Requires `hubspot_api_key` secret and Private App scopes: `crm.import`, `crm.schemas.companies.write` (full scope list for the shared key is in the Secret reference table)

**Stage 6 — Client Research** (clients → financial diligence or technology/R&D profile data)
- **Via Client Research view** (`views/finance_researcher.py`): select client companies from `data/all-contacts/clients/`, pick a **research focus** — 💰 Financials or 🔬 Technology & R&D — then launch one **deep-research-style OpenAI** call per company (`gpt-5.6-sol`, `gpt-5.6-terra`, or `gpt-5.6-luna` via the Responses API with `background=True` + the `web_search` tool), poll until complete, review results, then apply. The dedicated `o3-deep-research`/`o4-mini-deep-research` models were shut down 2026-07-23; `gpt-5.6-sol` is OpenAI's named replacement.
- Research calls take minutes and cost real money (~$0.30–$1/company on Luna, ~$0.75–$2.50 on Terra, ~$1.50–$5 on Sol; Terra is the default). The UI shows a pre-run cost estimate and requires a confirmation checkbox above $50 estimated total; actual cost is computed from `response.usage` per call.
- Background mode means no long-held Streamlit connection: run state (response IDs, per-company status/output/cost) is checkpointed to GCS — financial runs at `finance-research-runs/{run_id}/state.json` (run IDs `finres_*`), technology runs at `tech-research-runs/{run_id}/state.json` (run IDs `techres_*`); raw responses saved under `{runs_prefix}/{run_id}/raw/` for manual inspection. A "Resume monitoring" expander re-attaches to a run by ID after a refresh or from another session (the `finres_`/`techres_` prefix selects the GCS prefix; `state.json` also stores `focus`).
- **Financial focus output** is a strict-JSON object of 54 fields (identity, revenue/funding, federal awards 3yr, headcount, health signals, grant activity, budget signals, 0–100 proposal-readiness score, qualification, sources) defined in `src/modules/finance_research.py::FIELD_SECTIONS`. **Technology focus output** is a strict-JSON object of ~40 fields (identity, core technology, products & services, R&D activity, IP/patents, TRL/maturity, differentiation, grant-alignment keywords/agency fit, sources) defined in `src/modules/tech_research.py::FIELD_SECTIONS`. Malformed JSON gets one repair attempt via `gpt-4o-mini` before the row is marked error (`finance_research.parse_research_output` is shared — pass `fields=tech_research.ALL_FIELDS` for tech runs).
- **Apply** writes three focus-specific columns onto every contact row of each researched company in the clients parquets (rewritten in place): financial focus → `financial_data` (full JSON string), `financial_summary` (digest, no AI call), `financials_updated_at` (ISO date); technology focus → `technology_data`, `technology_summary`, `technology_updated_at`. **Financial runs never modify `summary` or `embeddings`.** Technology runs show a checkbox at apply time (on by default): rewrite each company's matching `summary` from `tech_research.build_matching_summary()` (core tech + approach + capabilities + products/services + use cases + R&D focus + keywords, confidence labels stripped, no AI call) and re-embed it (`text-embedding-ada-002`, float64 to match stored dtype) — this intentionally changes grant matching; uncheck to save research columns only.
- HubSpot Import's "Financial research run" mode only lists `finance-research-runs/` — technology runs are not currently importable to HubSpot.

**Stage 7 — Drive Sync** (client Google Drive documents → profile updates)
- **Via Drive Sync view** (`views/drive_sync.py`): the client shared drive is structured root → section folders (`a-h`, `i-p`, …, `nonprofit & nonR&D business`, `internal projects`) → `{Client Name}_INTERNAL` folders (one per client, docs inside recursively). The `internal projects` section is excluded by default.
- **Setup (one-time):** save the shared drive ID; the drive must have both `matcher-app@` and `matching-job@` service accounts added as **Viewer members** (no domain-wide delegation). Drive access uses `google-api-python-client` with scope `drive.readonly` — the only place in the codebase where credentials are built **with explicit scopes**.
- **Scan & auto-assign:** list sections → scan chosen sections → each client folder is fuzzy-matched against `company_name` in `data/all-contacts/clients/` (normalize: strip trailing `_INTERNAL`, lowercase, drop punctuation + legal suffixes; tiers: exact → containment (≥5 chars) → `difflib` ratio ≥ 0.87 with ≥ 0.05 margin). Matches are saved to `drive-sync-configs/assignments.json` (`folder_id → {client_key, match_type: auto|manual}`); unmatched folders go to a review table (`st.data_editor` — assign to any client, mark "new client", or skip). Rescans never overwrite existing assignments.
- **Sync:** pick exactly which assigned clients to process (multiselect showing each client's last-synced date + folder count, quick-select buttons for All / None / Never synced / Stale > 30 days, plus a read-only status table of every assigned client), pick which **unassigned folders** should produce new-client proposals (defaults to never-proposed folders; All / None / Never proposed buttons), choose a **time budget** (1 h → 24 h, default 4 h) and optionally raise the per-client document caps → `drive-sync-job` Cloud Run Job lists each client's folder recursively, diffs file `modifiedTime` against `drive-sync-configs/sync_state.json` (unchanged clients cost zero downloads/LLM calls; "Full re-scan" checkbox bypasses), extracts changed docs (Google Docs/Sheets/Slides exported; PDF/DOCX/XLSX/TXT/CSV binaries ≤15 MB via `src/modules/doc_extract.py`; caps: 40 docs / 150k chars per client, overflow deferred to next run), then one Claude (`claude-sonnet-4-6`) merge call per changed client: current `summary` + prior `client_docs_data` + new doc texts → `{no_meaningful_change, updated_summary, docs_digest, extracted}`. Writes `client_docs_data` (JSON) / `client_docs_summary` (digest) / `docs_updated_at` (ISO date) onto every contact row of the client; rewrites `summary` + re-embeds (`text-embedding-ada-002`, float64) **only when the change is meaningful**. Touched parquets + sync_state + interim status checkpoint every 10 clients (re-trigger resumes after timeout). Dry-run mode reports without writing.
- **New clients:** unassigned folders produce proposals (name/summary/digest) in the status payload. **Website extraction:** business email/URL domains are harvested around each proposal folder — Drive share permissions (`permissions.list` on the folder, weight ×3), file `lastModifyingUser` emails (×2), and emails/URLs in doc text (×1) — with freemail/our-own/gov domains filtered out. The top candidates are passed to Claude as `candidate_domains` (it may pick one that clearly belongs to the company); if Claude leaves the website empty, a deterministic fallback fills it when a domain stem matches the company/folder name (exact → containment → acronym → difflib ≥ 0.8, frequency tie-break) — `website_source` records `claude` vs `domain_match`, and `candidate_domains` ships in the proposal for UI hints. Websites are never invented beyond these signals. The view's review queue requires a website per approval (**proposals with a pre-filled website are pre-checked for approval**), embeds in Streamlit, writes rows (clients convention: `company_name`/`summary`) to a new `data/all-contacts/clients/drive_sync_{date}_{hex6}.parquet`, and converts the folder into a normal assignment.

**Stage 8 — Multi-aspect profiling & matching** (clients → per-aspect embeddings → re-ranked client×topic matches)

A client's single blended `summary` embedding averages away everything except its dominant theme, so a company with three unrelated capability areas matches poorly on all three. Stage 8 splits each client into a handful of independently embedded aspects and matches per aspect.

- **Via Client Profiles view** (`views/client_profiler.py`): reads only material that already exists on the client rows — website summary/scrape (`summary`, `full_text`/`page_text`), Drive extractions (`client_docs_summary` + `client_docs_data.extracted`), and Deep Research output (`technology_data`, `financial_data`) — per-source include checkboxes (financials off by default, since it describes money not capability). Material is read per company as the **first non-empty value across its contact rows** (a partially-updated file can leave some rows blank; `aspect_profile.merge_company_row()`) and capped per source (~8k–14k chars). The view itself only builds the directory (material available + profile status) and triggers the job — **building runs in `client-profile-job`** (config → `client-profile-configs/{run_id}.json`, status polled at `client-profile-jobs/{run_id}/status.json`, resumable by run ID), so a large batch survives a closed tab. One Claude call per client (`claude-sonnet-4-6` default, Haiku optional, one strict-JSON retry, 4 clients concurrently) returns `{profile_summary, aspects:[{label, kind, text, keywords, evidence}]}` — 2–8 aspects, each an independently searchable capability/technology/product/domain/market, grounded only in the supplied material. Each aspect's `label + text + keywords` is embedded (`text-embedding-ada-002`) and the profile is upserted as **one row per company** into `data/client-profiles/profiles.parquet`. Aspects are editable in the view (`st.data_editor`) with a re-embed-and-save button (still in-process — one company), plus delete.
- **Staleness:** each profile stores a `source_fingerprint` (sha256 over *all* available source texts, not just the ones used). The view recomputes it live and flags profiles ⚠️ stale when the client's website/Drive/research material has changed since the build; the picker pre-selects stale + unprofiled clients. Clients with no material at all are listed separately, not offered.
- **Nothing is written to the client parquets** — profiles live in their own store, so Client Editor / Client Research / Drive Sync keep rewriting client rows freely.
- **Via Bulk Aspect Match view** (`views/aspect_match.py`): select profiled clients + grant agencies (same keyword/date filter UI as Grant Search), then one numpy matmul per client scores its `(n_aspects, 1536)` matrix against the topic matrix. A topic qualifies for a client when its **best** aspect score clears the threshold and at least `min_hits` aspects clear it (default 1 — a client's aspects are alternative capabilities, not conjunctive requirements of one query, unlike Grant Search's multi-aspect mode; raise it to demand topics spanning several capabilities). Top-K topics per client are kept with the winning aspect attributed (`aspect_label`, `aspect_score`, `aspects_hit`, and a per-aspect `aspect_scores` JSON). Survivors are re-ranked 1–5 by Claude (async `AsyncAnthropic`, 15 concurrent, exponential backoff on 429/529) with the matched aspect text + profile summary + topic as context; unparseable/failed pairs get score 0 and are reported, never silently promoted. Rows ≥ the minimum LLM score are shown, downloadable as CSV, and saved to `aspect-match-results/{run_id}/results.csv`. Scoring and re-ranking run **in the Streamlit process** — the page must stay open; above 2,500 re-rank calls a confirmation checkbox is required.

---

## Streamlit Views

| File | Title | Purpose |
|------|-------|---------|
| `views/contact_importer.py` | Contact Importer | Upload any lead spreadsheet **or** pull a HubSpot company list (lists search → memberships → batch company read) → map columns → dedup preview vs GCS (per-source or all-sources scope) → choose profiling method (🌐 scrape+GPT summary or 🔬 Deep Research technology focus, with model picker + cost estimate + >$50 confirmation) → stage file + trigger `contact-import-job` Cloud Run Job → poll `contact-import-jobs/{run_id}/status.json` |
| `views/client_editor.py` | Client Editor | Select a company from `data/all-contacts/clients/` → edit its `summary` → re-embed (float64, matching stored dtype) → apply to all contact rows of that company → rewrite the source parquet in place. **Admins only:** a "🗑 Delete clients" section (multiselect + type-DELETE confirmation) removes companies that are no longer clients — every contact row, their aspect profile, and their Drive Sync assignment |
| `views/finance_researcher.py` | Client Research | Select clients from `data/all-contacts/clients/` → pick research focus (💰 Financials / 🔬 Technology & R&D) → launch background Deep Research tasks (one per company) → poll `finance-research-runs/` or `tech-research-runs/` `{run_id}/state.json` → review parsed results → apply `financial_data`/`financial_summary`/`financials_updated_at` or `technology_data`/`technology_summary`/`technology_updated_at` back onto client rows |
| `views/topic_importer.py` | Topic Importer | Upload PDF or paste text → Claude extracts topics → editable table → embed + save to `processed/` |
| `views/grant_search.py` | Grant Search | Select agencies, apply keyword filters, embed a tech description, find matching topics by cosine similarity |
| `views/bulk_matching.py` | Bulk Matching | Select contact sources + grant agencies, configure threshold/top-k/AI validation, trigger Cloud Run job, poll status |
| `views/sam_gov_upload.py` | SAM.gov Upload | Upload SAM.gov CSVs (processed in Streamlit: map columns → dedup vs existing store by notice ID + title **before** screening → Claude screening → summarize → embed → save) **or** configure an API fetch that triggers the `sam-gov-job` Cloud Run Job. Also exposes a **Daily API Parameters** section (daily 5 AM CST schedule config saved to `sam-gov-configs/daily_schedule.json`) and a **Revision Check** expander that triggers the `revision_check` job mode (dry-run by default) to sweep stored open notices for SAM.gov amendments. Streamlit polls `sam-gov-jobs/{run_id}/status.json` for manual run completion and renders revision/archived tables from the status payload. |
| `views/grants_gov_fetch.py` | Grants.gov Fetch | Query the Grants.gov public `search2` API (keyword, date range, status, funding instrument, agency — no API key) → Claude Haiku relevance screening → embed → save to `data/all-topics/processed/GRANTS-GOV/` |
| `views/hubspot_import.py` | HubSpot Import | Three source modes: **Matching run** (concatenate segment CSVs → standard `matcher_*` properties), **Financial research run** (load `finance-research-runs/{run_id}/state.json` → per-field mapping table: each financial field → existing HubSpot property or auto-created `matcher_fin_*`), or **Client profiles** (load `data/client-profiles/profiles.parquet` → pick clients → flattened aspect fields → existing property or auto-created `matcher_profile_*`/`matcher_aspect_*`). All submit as company imports via `/crm/v3/imports` (dedup by `domain`) and poll for completion |
| `views/resume_importer.py` | Resume Importer | Upload HubSpot contacts CSV with resume URL column → dedup by email → fetch files (PDF/DOCX) via HubSpot Bearer auth → extract text → GPT expertise summary → embed → save to `data/resumes/` |
| `views/resume_search.py` | Resume Search | Natural-language query → embed → cosine similarity against resume parquets → ranked candidate cards + CSV export. Supports an optional include keyword (single term) and comma-separated exclude keywords to pre-filter the resume pool before scoring. |
| `views/drive_sync.py` | Drive Sync | Scan the client Google shared drive (sections → `{Client}_INTERNAL` folders) → fuzzy auto-assign folders to clients with persistent assignments (`drive-sync-configs/assignments.json`) + review table → select the exact clients to sync (last-synced dates + All/None/Never/Stale quick-picks) and the exact unassigned folders to propose as new clients, set the time budget (1–24 h) and per-client doc caps → trigger `drive-sync-job` (incremental via `sync_state.json`) → poll `drive-sync-jobs/{run_id}/status.json` → results + new-client review queue (approve with website → rows created in `data/all-contacts/clients/`) |
| `views/client_profiler.py` | Client Profiles | Directory of client companies with the source material available per client (website / Drive / technology / financials) and profile status (none / current / ⚠️ stale by `source_fingerprint`) → pick clients + sources + target aspect count + model → trigger `client-profile-job` → poll `client-profile-jobs/{run_id}/status.json` (one Claude call per client → embed each aspect → upsert `data/client-profiles/profiles.parquet`); an expander resumes monitoring by run ID. Second section reviews/edits a profile's aspects (`st.data_editor`) and re-embeds in-process, or deletes it (delete is admin-only). **Admins only:** a third section bulk-deletes profiles, with an opt-in checkbox to delete the clients' contact rows too |
| `views/aspect_match.py` | Bulk Aspect Match | Select profiled clients + grant agencies + filters → per-client matmul of aspect vectors against topic vectors → keep topics clearing the threshold on ≥ `min_hits` aspects, top-K per client → async Claude re-rank 1–5 with the matched aspect as context → results table + CSV download + `aspect-match-results/{run_id}/results.csv`. Runs in-process (keep the page open) |
| `views/suggestions.py` | Suggestions | Team feature-request board — submit by name, upvote once per session; stored as JSON blobs in `suggestions/` |
| `views/admin_portal.py` | Admin Portal | **Admins only** (and hidden from the navigation for everyone else) — lists the code-constant super admins, then lets a **super admin** add/remove admins in `admin-config/admins.json` with an append-only change history. Non-super admins see the list read-only |

---

## Source Files & Their Roles

### `src/modules/` (keep and extend)

| File | Class / Export | Purpose |
|------|---------------|---------|
| `text_embedder.py` | `TextProcessor` | OpenAI embeddings, text chunking, token reduction, normalization, LLM summarization. Constructor takes `api_key: str` directly — NOT a file path. |
| `bucket_manager.py` | `BucketManager` | Google Cloud Storage upload/download (parquet, CSV). Constructor: `BucketManager(bucket_path: str, client=None)` — always pass a `storage.Client` from `get_storage_client()`. |
| `web_scraper.py` | `WebScraper` | Selenium-based website scraper — legacy, being phased out in favour of Playwright. |
| `email_generator.py` | `async_generate_subject_line`, `async_josiah_copy` | Async email copy generation. Subject line tries GPT-4o-mini first, falls back to Claude Haiku on 429. Both functions accept async client objects passed in from the caller. |
| `grant_utils.py` | `normalize_grant_columns` | Call this whenever a topics DataFrame is loaded. Ensures `grant_summary` is always present: renames `description` → `grant_summary` if the column is absent, or fills empty `grant_summary` values from `description` if both exist. |
| `finance_research.py` | `FIELD_SECTIONS`, `build_research_prompt`, `parse_research_output`, `build_financial_digest`, `response_cost_usd` | Deep Research helpers for the Client Research view (financial focus + shared plumbing) — 54-field output schema, prompt builder, JSON extract + `gpt-4o-mini` repair (`parse_research_output` takes an optional `fields=` list so it can normalize either focus's schema), headline digest (no AI call), and per-response cost from `usage`. Model IDs (`gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`) drift — verify against developers.openai.com/api/docs/models on API errors. |
| `aspect_profile.py` | `SOURCES`, `MATERIAL_COLS`, `merge_company_row`, `assemble_source_texts`, `source_fingerprint`, `build_aspect_system`, `build_aspect_user_message`, `parse_aspect_response`, `aspect_embed_text`, `pack_embeddings`/`unpack_embeddings`, `build_profile_record`, `load_profiles`/`save_profiles`/`upsert_profiles`/`delete_profile`, `company_key` | Multi-aspect client profiles (Stage 8). Streamlit-free (shared by the two views **and `client_profile_job.py`**): merges a company's contact rows into one material row, pulls source material off it per source, fingerprints it for staleness, builds the aspect-generation prompt, parses/normalizes Claude's JSON (unknown `kind` → `capability`, duplicate labels dropped, ≤ `MAX_ASPECTS`), and owns the `data/client-profiles/profiles.parquet` store. **Aspect vectors are stored flat** (`aspect_embeddings` = `n_aspects × embedding_dim` float64 in one list column) — a flat double list round-trips through parquet without nested-list dtype ambiguity; always read them back via `unpack_embeddings()`. |
| `tech_research.py` | `FIELD_SECTIONS`, `ALL_FIELDS`, `build_research_prompt`, `build_tech_digest`, `build_matching_summary` | Technology & R&D research schema/prompt/digest for the Client Research view's tech focus — ~40-field output (core technology, products, R&D activity, patents, TRL/maturity, differentiation, grant-alignment keywords). `build_matching_summary()` assembles embedding-ready text (confidence labels stripped) for the optional summary-rewrite at apply time. Reuses finance_research's models, pricing, and JSON parse/repair. |
| `access_control.py` | `SUPER_ADMINS`, `current_user_email`, `is_admin`, `is_super_admin`, `role_label`, `require_admin`, `admin_only_notice`, `load_admins`/`save_admins`/`admin_emails` | Admin gating for the delete actions and the Admin Portal. Identity is the IAP email `app.py` puts in `st.session_state.user_email`; the admin list lives in `admin-config/admins.json` (cached per session — a newly added admin must reload the page). `SUPER_ADMINS` is a code constant: not editable from the UI, never stored in the JSON, and the only role allowed to change the list. A GCS read failure grants nothing beyond the super admins. Streamlit-only. |
| `client_delete.py` | `delete_clients`, `count_rows`, `key_mask`, `format_report`, `ARCHIVE_PREFIX` | Streamlit-free client deletion shared by Client Editor and Client Profiles. Re-reads the clients parquets from GCS (never the view's session copy), **archives every removed row to `data/deleted-clients/` before writing anything** — a failed archive aborts the whole delete — then rewrites each touched parquet (deleting the blob outright when no rows remain), drops the company's `profiles.parquet` row, and moves its Drive Sync folder assignment to `skipped` so the next scan neither syncs nor re-proposes it. Per-target failures land in `report['errors']`, not exceptions. |

### Resume Importer — 5-step UI flow

The `views/resume_importer.py` view ingests individual-level resumes from HubSpot:

1. **Upload** — HubSpot contacts CSV or Excel (UTF-8 / Latin-1 fallback)
2. **Column mapping** — email (required, join key) and resume URL (required); firstName, lastName, phone, company optional; auto-detects column names
3. **Dedup** — loads existing parquets from `data/resumes/`, deduplicates by lowercase email
4. **Fetch + Extract + Summarise** — fetches each file from its HubSpot URL using `Authorization: Bearer {hubspot_api_key}` on the first request (no retry cycle needed); rejects `text/html` responses immediately; extracts text via three-stage waterfall:
   - **PDF**: `fitz.open(stream, filetype='pdf')` page by page
   - **DOCX primary**: parse `word/document.xml` directly as a ZIP to pull every `<w:t>` text run — the only method that captures text boxes (used by most resume templates for sidebar/column layout)
   - **DOCX fallback**: python-docx paragraphs + table cells, then fitz
   - Minimum **400 chars** of extracted text required; shorter extractions are discarded before GPT is called to prevent hallucination from header-only content
   - GPT-3.5-turbo expertise summary (skills, domain, years of experience, project types); responds with `-` if text is too thin — stored as empty string, not embedded
5. **Embed & Save** — `text-embedding-ada-002` on the `expertise_summary`; rows with empty summary get an empty embedding list and are skipped by the search; saves to `data/resumes/resumes_{YYYY-MM-DD}_{hex6}.parquet`

**Key implementation notes:**
- HubSpot's `hubspot_api_key` requires the **Files** scope in addition to CRM scopes — missing this causes `fetch_failed` for all URLs
- HubSpot sometimes returns `200 OK` with an HTML redirect page instead of a 401/403 — the fetch function explicitly rejects `Content-Type: text/html` responses
- Parquet embeddings column reads back from GCS as `numpy.ndarray`, not `list` — always filter with `isinstance(e, (list, np.ndarray))` not `isinstance(e, list)`
- Do NOT use `@st.cache_data` wrapping GCS calls in this codebase — use `st.session_state` or load fresh; the grant_search pattern (no cache) is the reference
- A raw text preview expander appears after fetch so the team can verify extraction quality before committing to the summarization API calls

### Contact Importer — 4-step UI flow

The `views/contact_importer.py` view handles any lead source generically. Steps 1–3 run in Streamlit; the heavy work runs in a Cloud Run Job.

1. **Input source** — radio between two modes:
   - **📄 Upload spreadsheet** — CSV or Excel (UTF-8 / Latin-1 fallback); file bytes stored in `st.session_state.ci_file_bytes` for staging
   - **🟠 HubSpot company list** — Load lists (`POST /crm/v3/lists/search`, objectTypeId `0-2`, paged) → pick a list → fetch members (`GET /crm/v3/lists/{id}/memberships`, paged) → batch-read companies (`POST /crm/v3/objects/companies/batch/read`; properties name/domain/website/state/industry/phone). The fetched companies become a DataFrame with the **standard column names** (contact-level fields empty — company lists carry companies only), serialized to CSV bytes and staged exactly like an upload, so steps 2–4 and the job are identical. `companyWebsite` = HubSpot `domain`, falling back to `website`. Requires `hubspot_api_key` scopes `crm.lists.read` + `crm.objects.companies.read`.
2. **Source & column mapping** — select `apollo`, `sba`, `free_alert`, `hubspot`, or a custom label; auto-detect column names (HubSpot pulls auto-map fully since the columns are already standard); URL column is required, all others optional. Excel HYPERLINK formulas are stripped via `_strip_hyperlink()` on every mapped column.
3. **Dedup** (preview only) — loads existing parquets from GCS, compares bare domains via `tldextract`, shows already-stored vs new counts. A **"check against all sources"** checkbox (default ON for HubSpot pulls, OFF for uploads) widens the scope from `data/all-contacts/{source}/` to all of `data/all-contacts/` — a HubSpot list can contain companies already imported under any source. The scope is passed to the job as `dedup_all_sources` so the runtime re-dedup uses the same scope. Invalidates if source, URL column, or scope changes.
4. **Start import job** — choose the **company profiling method**: 🌐 `scrape` (default — scrape + GPT-3.5-turbo summary, near-free) or 🔬 `deep_research` (technology-focus Deep Research per unique company domain; shows a model picker (Luna/Terra/Sol), an estimated-cost metric of unique domains × `fr.EST_COST_PER_COMPANY`, and a confirmation checkbox above $50). Uploads raw file bytes to `contact-import-uploads/{run_id}{ext}`, writes config JSON (including `profile_method` + `research_model`) to `contact-import-configs/{run_id}.json`, triggers `contact-import-job` via `run_v2.JobsClient`; stores `run_id` in `st.session_state.ci_active_run` and polls `contact-import-jobs/{run_id}/status.json` every 10s until complete. The completion screen relabels "Scraped OK" → "Researched OK" and shows companies-researched counts + actual research cost for deep-research runs.

URL normalization (adds `https://` if missing) happens at mapping time. The job re-deduplicates at runtime as a safety check — the Streamlit dedup is for preview only.

**Resume monitoring after refresh:** an expander at the top of the page accepts a `run_id` string to resume polling a job from a previous session.

> The legacy notebooks (`apollo_importer__1_.ipynb`, `SBA_importer.ipynb`, `fwee_alluts_impoatah.ipynb`) are superseded by this view and no longer need conversion.

---

## Data Schemas

### Contact record (parquet, `data/all-contacts/`)
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

Rows may additionally carry Deep Research columns — financial focus (Client Research view, `data/all-contacts/clients/` only): `financial_data` (JSON string of the full 54-field Deep Research output), `financial_summary` (human-readable digest), `financials_updated_at` (ISO date); technology focus (Client Research view on clients, or any source imported via the Contact Importer's `deep_research` profiling method): `technology_data` (JSON string of the ~40-field tech research output), `technology_summary` (digest), `technology_updated_at` (ISO date). For deep-research imports, `company_summary` holds `tech_research.build_matching_summary()` output (not a scraped-page GPT summary) and `embeddings` is its vector. Client rows updated by Drive Sync additionally carry `client_docs_data` (JSON: extracted fields + source_files + last_run), `client_docs_summary` (plain-text digest), and `docs_updated_at` (ISO date).

### Multi-aspect client profile (parquet, `data/client-profiles/profiles.parquet`)
One row per client company — written by `client-profile-job` (and by single-profile edits in the Client Profiles view), read by Bulk Aspect Match and by HubSpot Import's **Client profiles** mode. Never written to the client contact parquets.

| Field | Type | Notes |
|-------|------|-------|
| `company_key` | str | **Join key** — `{company_name}\|\|{companyWebsite}`, the same identity used by Client Editor / Client Research / Drive Sync (`aspect_profile.company_key()`) |
| `company_name` / `companyWebsite` | str | Copied from the client rows at build time |
| `profile_summary` | str | 2–4 sentence company summary. Context for the LLM re-ranker — **not embedded** |
| `aspects` | str | JSON array of `{label, kind, text, keywords, evidence}`; `kind` ∈ technology/capability/product/domain/market. Read with `profile_aspects()` |
| `aspect_labels` | str | `' \| '`-joined labels, for display without parsing the JSON |
| `n_aspects` / `embedding_dim` | int | Shape of the packed vectors (dim is 1536 for `text-embedding-ada-002`) |
| `aspect_embeddings` | list[float] | **Flat** `n_aspects × embedding_dim` float64 vectors of each aspect's `label + text + keywords`. Read with `unpack_embeddings()` — never index this directly |
| `sources_used` | str | Comma-separated source keys actually included in the build (`website`, `drive`, `technology`, `financials`) |
| `source_fingerprint` | str | sha256[:16] over **all** available source texts at build time — mismatch against a live recompute means the profile is stale |
| `model` | str | Aspect-generation model; manually edited profiles get ` + manual edit` appended once |
| `built_at` | str | ISO date of the build or last edit |

### Grant topic record (parquet, `data/all-topics/processed/`)
| Field | Type | Notes |
|-------|------|-------|
| `topic_number` | str | Agency topic/solicitation ID |
| `agency` | str | Sub-agency (e.g. `ARMY`, `NCI`) |
| `broad_agency` | str | Folder-level agency key (e.g. `DOD`, `HHS`) — added at load time |
| `title` | str | |
| `grant_summary` | str | **Canonical text field** — always present after `normalize_grant_columns()`. Topic Importer and SAM.gov sources write this directly. |
| `description` | str | Raw source text — present in SAM.gov parquets as a backup alongside `grant_summary`; absent from Topic Importer parquets. Never use this directly; always call `normalize_grant_columns()` after loading. |
| `embeddings` | list[float] | `text-embedding-ada-002` vector |
| `open_date` / `close_date` | str | |
| `source` | str | Origin URL or label |
| `scraped_at` | str | ISO date of processing |

### SAM.gov topic record (parquet, `data/all-topics/processed/SAM-GOV/`)
Same base schema as grant topic, plus extra columns written by `sam_gov_job.py`:

| Field | Type | Notes |
|-------|------|-------|
| `source` | str | SAM.gov opportunity URL — `https://sam.gov/opp/{noticeId}/view`. Auto-populated in API mode; mapped from a CSV column in CSV mode (optional). Updated to the new noticeId when a revision is applied. |
| `sam_confidence` | str | `"high"` / `"medium"` / `"low"` — Claude screening confidence |
| `sam_reason` | str | One-sentence explanation of the screening decision |
| `notice_version_id` | str | Version-specific SAM.gov `noticeId` — revision detection compares this against the latest version. Backfilled from the `source` URL for parquets written before this column existed. |
| `sam_status` | str | `"active"` / `"archived"` — set to archived by the revision check when the notice is no longer on SAM.gov. Archived rows are kept but filtered out by `matching_job` and Grant Search. |
| `revised_at` | str | ISO date the last revision was applied; empty if never revised |
| `sam_revision_notes` | str | Claude-written diff of the last revision — what changed, topics added/removed |

### Match output (CSV, `matching-results/{run_id}/`)
Includes merged fields from both contact and grant records plus:
- `good_match` — `"yes"` / `"no"` from Claude Haiku AI validation
- `subject_line` — generated subject line (when `prewrite_email` is enabled)
- `ai_message` — generated email body copy (when `prewrite_email` is enabled)

Results are written in 1 000-row segments (`segment_001.csv`, `segment_002.csv`, …).
A `status.json` file is written on completion (or failure) and polled by the Streamlit UI.

### Resume record (parquet, `data/resumes/`)
| Field | Type | Notes |
|-------|------|-------|
| `uuid` | str | Unique record ID |
| `email` | str | **Join key** — lowercase; used for dedup |
| `firstName` | str | |
| `lastName` | str | |
| `phone` | str | |
| `company` | str | Employer name if present in HubSpot export |
| `resume_url` | str | Original HubSpot file URL |
| `file_type` | str | `pdf`, `docx`, `unknown`, `fetch_failed`, `missing` |
| `expertise_summary` | str | GPT-3.5-turbo 3-5 sentence summary of skills, domain, experience, and project types. Empty string if extraction yielded < 400 chars or GPT returned insufficient text. |
| `embeddings` | list[float] | `text-embedding-ada-002` vector of `expertise_summary`. Empty list `[]` when `expertise_summary` is blank — these rows are skipped by Resume Search. |
| `processed_at` | str | ISO date of processing |

### Suggestion record (JSON, `suggestions/`)
| Field | Type | Notes |
|-------|------|-------|
| `id` | str | UUID4 |
| `name` | str | Submitter name |
| `suggestion` | str | Feature request text |
| `votes` | int | Upvote count |
| `created_at` | str | ISO datetime (UTC) |

---

## GCS Bucket Structure

Bucket name: `cc-matcher-bucket-jeg-v1` (single-region, us-central1). All pipeline data lives here — no local filesystem writes in production.

```
cc-matcher-bucket-jeg-v1/
  data/
    all-topics/
      processed/
        DOD/
          ARMY_2026-03-01_a3f9c1.parquet
          USSOCOM_2026-03-01_b2e4d7.parquet
        HHS/
          NCI_2026-03-01_c1d8a2.parquet
        ARPA/
          ...
        SAM-GOV/
          sam_gov_2026-04-16_f3a1b9.parquet
        GRANTS-GOV/
          grants_gov_2026-07-01_d4c2e8.parquet
    all-contacts/
      apollo/
        apollo_2026-03-01_a3f9c1.parquet
      sba/
        sba_2026-03-01_b2e4d7.parquet
      free_alert/
        free_alert_2026-03-01_c1d8a2.parquet
    resumes/
      resumes_2026-06-24_a3f9c1.parquet   # individual resume records, deduped by email
    client-profiles/
      profiles.parquet                  # multi-aspect client profiles, one row per company (overwritten in place on every build/edit/delete — not versioned)
    deleted-clients/                    # backups written before any client deletion — restore source if a delete was a mistake
      deleted_2026-08-19_a3f9c1.parquet # the removed rows + _deleted_from / _deleted_at / _deleted_by columns
  admin-config/
    admins.json                         # {admins: [email], updated_at, updated_by, history: [{at, by, note}]} — super admins are NOT in here (code constant)
  aspect-match-results/                 # Bulk Aspect Match runs (written from Streamlit, no job)
    aspect_match_2026-08-18_15-30-00/
      results.csv
  job-configs/                          # matching-job configs
    2026-04-16_10-30-00_ag-DOD-HHS_src-apollo.json
  matching-results/
    2026-04-16_10-30-00_ag-DOD-HHS_src-apollo/
      segment_001.csv
      segment_002.csv
      status.json
  sam-gov-configs/                      # sam-gov-job configs
    sam_gov_2026-06-01_10-30-00.json
    daily_schedule.json               # persistent daily schedule config (overwritten on save, not versioned)
    revcheck_state.json               # revision-check sweep cursor: topic_number → last-checked date (advanced by apply runs only)
  sam-gov-uploads/                      # staging: CSV blobs uploaded by Streamlit (CSV mode)
    sam_gov_2026-06-01_10-30-00.csv
  sam-gov-jobs/                         # sam-gov-job completion status
    sam_gov_2026-06-01_10-30-00/
      status.json
  contact-import-uploads/               # staging: raw file bytes uploaded by Streamlit
    contact_import_2026-06-25_10-30-00_apollo.csv
  contact-import-configs/               # contact-import-job trigger configs
    contact_import_2026-06-25_10-30-00_apollo.json
  contact-import-jobs/                  # contact-import-job completion status
    contact_import_2026-06-25_10-30-00_apollo/
      status.json
  drive-sync-configs/                   # Drive Sync state + job configs
    assignments.json                  # drive_id + folder_id → client_key assignments (+ unassigned/skipped) — overwritten on save
    sync_state.json                   # files: file_id → modifiedTime last synced (incremental diffing) + proposed: folder_id → last-proposed date (proposal rotation cursor); advanced only by non-dry runs
    drive_sync_2026-08-11_15-30-00.json  # per-run job config
  drive-sync-jobs/                      # drive-sync-job completion status
    drive_sync_2026-08-11_15-30-00/
      status.json
  client-profile-configs/               # client-profile-job trigger configs
    client_profile_2026-08-19_10-30-00.json
  client-profile-jobs/                  # client-profile-job progress + completion status
    client_profile_2026-08-19_10-30-00/
      status.json
  finance-research-runs/                # Client Research run checkpoints — financial focus
    finres_2026-07-29_10-30-00/
      state.json                        # per-company response IDs, status, parsed output, cost (+ focus)
      raw/
        000_Acme_Robotics.txt           # raw Deep Research response text (for manual review)
  tech-research-runs/                   # Client Research run checkpoints — technology focus (same layout)
    techres_2026-07-31_10-30-00/
      state.json
      raw/
  suggestions/
    <uuid>.json
```

### BucketManager usage pattern

```python
# Always instantiate with a client from get_storage_client()
bm = BucketManager('cc-matcher-bucket-jeg-v1', client=get_storage_client())

# Write
bm.upload_file('data/all-topics/processed/DOD/ARMY_2026-03-01.parquet', df)

# Read
df = bm.download_file('data/all-topics/processed/DOD/ARMY_2026-03-01.parquet')
```

### Listing GCS prefixes (replaces os.listdir for agency dropdowns)

```python
def list_broad_agencies(client) -> list[str]:
    blobs = client.list_blobs(
        'cc-matcher-bucket-jeg-v1',
        prefix='data/all-topics/processed/',
        delimiter='/'
    )
    list(blobs)  # must consume iterator to populate prefixes
    return sorted(
        p.replace('data/all-topics/processed/', '').strip('/')
        for p in blobs.prefixes
    )
```

---

## Agency Short-Codes

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
| `streamlit` | UI framework |
| `anthropic` | Claude (Haiku, Sonnet) — match verification, topic extraction, SAM.gov screening, email copy |
| `openai` | Embeddings (`text-embedding-ada-002`), GPT summarization, subject line generation, deep-research-style research (`gpt-5.6-sol`/`terra`/`luna` via Responses API background mode) for Client Research (financial + technology focuses) and the Contact Importer's `deep_research` profiling mode |
| `tiktoken` | Token counting before embedding (7500 token limit) |
| `google-cloud-storage` | GCS bucket I/O via `BucketManager` |
| `google-cloud-run` | Programmatic Cloud Run job triggering from Bulk Matching view (`run_v2.JobsClient`) |
| `requests` | HubSpot API calls in HubSpot Import view |
| `pymupdf` (`fitz`) | PDF text extraction in Topic Importer, Resume Importer (also used as DOCX fallback), and sam-gov-job (attachment text for revision diffs) |
| `python-docx` | DOCX text extraction in Resume Importer (paragraphs + table cells; XML parse is primary) |
| `pandas`, `numpy` | Data manipulation throughout |
| `pyarrow` | Parquet read/write |
| `playwright` | Async JS-rendered page scraping (fallback when aiohttp fails). Browser binary downloaded at server start via `@st.cache_resource` in `app.py`. System libs declared in `packages.txt`. |
| `aiohttp` + `BeautifulSoup` | Fast async scraping (first-pass in Contact Importer) |
| `tldextract` | Domain normalization (dedup in Contact Importer) |
| `selenium` | Legacy scraper in `web_scraper.py` (being phased out) |
| `duckdb` | In-notebook data querying (used in legacy importers) |

---

## Secrets / API Keys

All secrets are loaded via `st.secrets` in Streamlit code — never from `.txt` files, never hardcoded. The Cloud Run job reads secrets from environment variables injected by Cloud Run at startup (no `st.secrets` available outside Streamlit).

### `.streamlit/secrets.toml` (local dev — gitignored)

```toml
app_password      = "..."
openai_api_key    = "sk-..."
anthropic_api_key = "sk-ant-..."

[gcp_service_account]
type                        = "service_account"
project_id                  = "cc-matcher-v1"
private_key_id              = "..."
private_key                 = "-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----\n"
client_email                = "matcher-app@cc-matcher-v1.iam.gserviceaccount.com"
client_id                   = "..."
auth_uri                    = "https://accounts.google.com/o/oauth2/auth"
token_uri                   = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url        = "..."
```

The `[gcp_service_account]` block is the contents of `ServiceKey_GoogleCloud.json` reformatted as TOML. When deploying to Streamlit Cloud, paste the same values into **App Settings → Secrets** in the UI.

### Accessing secrets in Streamlit views

```python
import streamlit as st

oai_key    = st.secrets['openai_api_key']
anth_key   = st.secrets['anthropic_api_key']
```

### GCS client (always build this way in Streamlit)

```python
from google.oauth2 import service_account
from google.cloud import storage

def get_storage_client():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets['gcp_service_account']
    )
    return storage.Client(credentials=creds)
```

Pass this client into `BucketManager` — never rely on the `GOOGLE_APPLICATION_CREDENTIALS` env var in Streamlit code.

### Cloud Run job secrets

The matching job uses `storage.Client()` with no arguments (ADC via the attached service account) and reads API keys from environment variables:

```python
import os
anth_key   = os.environ['ANTHROPIC_API_KEY']   # injected from Secret Manager by Cloud Run
openai_key = os.environ['OPENAI_API_KEY']
```

### Secret reference

| Secret | Where used | How accessed |
|--------|-----------|--------------|
| `app_password` | Login gate in `app.py` | `st.secrets['app_password']` |
| `openai_api_key` | All views that embed + Client Research (Deep Research calls + JSON repair) | `st.secrets['openai_api_key']` |
| `anthropic_api_key` | Topic Importer, SAM.gov Upload, Bulk Matching | `st.secrets['anthropic_api_key']` |
| `gcp_service_account` | All views that touch GCS | `st.secrets['gcp_service_account']` dict |
| `hubspot_api_key` | HubSpot Import view + Resume Importer + Contact Importer (HubSpot list pulls) | `st.secrets['hubspot_api_key']` — must be placed **above** `[gcp_service_account]` in secrets.toml. Private App requires scopes: `crm.import`, `crm.schemas.companies.write`, **`files`** (files scope required for Resume Importer to download attachments), **`crm.lists.read`** + **`crm.objects.companies.read`** (Contact Importer HubSpot list pulls). All five scopes are granted on the current Private App (verified 2026-08) — a 401/403 from HubSpot means an expired/rotated token, not a missing scope. |
| `sam_gov_api_key` | SAM.gov Upload view (API fetch tab) | `st.secrets['sam_gov_api_key']` — free key from beta.sam.gov → Account Settings → API Keys. Passed into the sam-gov-job config JSON (not a Secret Manager secret). |
| `anthropic-api-key` (Secret Manager) | Cloud Run matching job + sam-gov-job | `os.environ['ANTHROPIC_API_KEY']` |
| `openai-api-key` (Secret Manager) | Cloud Run matching job + sam-gov-job + contact-import-job | `os.environ['OPENAI_API_KEY']` |

---

## Access Control (admins & destructive actions)

Everything in the app is open to the whole team except the **destructive** actions, which are admin-gated by `src/modules/access_control.py`:

| Action | Where | Required role |
|--------|-------|---------------|
| Delete clients (contact rows + profile + Drive assignment) | Client Editor → "🗑 Delete clients" | admin |
| Delete a single aspect profile | Client Profiles → section 2 | admin |
| Bulk-delete profiles (± their contact rows) | Client Profiles → section 3 | admin |
| Add / remove admins | Admin Portal | **super admin** |
| Everything else (imports, matching, research, Drive Sync, edits) | — | any signed-in user |

- **Identity** is the IAP-verified email that `app.py` writes to `st.session_state.user_email`. No email (the local-dev `app_password` fallback) counts as super admin — that path is unreachable behind IAP.
- **Super admins** are the `SUPER_ADMINS` tuple in `access_control.py` (currently `john@bwcoconsulting.com`). To change them, edit the constant and redeploy — they are deliberately not editable from the UI and are never written to `admins.json`.
- **Admins** live in `admin-config/admins.json`, managed in the Admin Portal (append-only `history` records who changed what). The list is cached in session state (the navigation checks it on every rerun), so a newly added admin sees their new rights after a page reload.
- A failed read of `admins.json` grants nothing beyond the super admins.
- Every deletion archives the removed rows to `data/deleted-clients/deleted_{date}_{hex6}.parquet` **before** anything is rewritten, and a failed archive write aborts the delete — that file is the only way back, since parquets are rewritten in place.
- Both delete paths call `client_delete.delete_clients()`, which re-reads the clients parquets from GCS rather than trusting the view's session copy.

---

## Naming Conventions

- **Files:** `snake_case.py`
- **Classes:** `PascalCase` (e.g., `TextProcessor`, `BucketManager`)
- **Contact fields:** camelCase for legacy compatibility (`companyWebsite`, `companyName`, `firstName`, `lastName`) — preserve these names to avoid breaking downstream column references
- **Grant fields:** snake_case (`grant_summary`, `open_date`, `close_date`, `scraped_at`)
- **Parquet output filenames:** `{source_or_agency}_{YYYY-MM-DD}_{hex6}.parquet` (hex suffix avoids collisions on same-day re-runs)
- **Match output filenames:** `segment_{NNN}.csv` under a run-ID prefix
- **Agency keys:** UPPERCASE short-codes (e.g., `DOD`, `HHS`, `ARPA`)

---

## Streamlit App on Cloud Run (`matcher-app` service)

The Streamlit UI runs as a Cloud Run **service** (not job) named `matcher-app`, replacing Streamlit Cloud. Auth is Google sign-in via **IAP (Identity-Aware Proxy)** enabled directly on the service — access is granted to the `team@bwcoconsulting.com` Google group with `roles/iap.httpsResourceAccessor`. The old password gate in `app.py` remains only as a local-dev fallback: when the `X-Goog-Authenticated-User-Email` header is present (set by IAP, spoof-proof because unauthenticated access is blocked), the session is authenticated automatically and the email shown in the sidebar.

- **Image:** `us-central1-docker.pkg.dev/cc-matcher-v1/matcher/matcher-app:latest` — built from `Dockerfile.app` (repo root) via `cloudbuild.app.yaml`. Installs the `packages.txt` Chromium libs and Playwright's Chromium at build time (`PLAYWRIGHT_BROWSERS_PATH=/ms-playwright`).
- **Secrets:** the full local `.streamlit/secrets.toml` is stored in Secret Manager as `streamlit-secrets` and volume-mounted at `/app/.streamlit/secrets.toml`, so `st.secrets` works unchanged. To rotate/edit secrets: `gcloud secrets versions add streamlit-secrets --data-file=.streamlit/secrets.toml --project cc-matcher-v1`, then redeploy (or restart) the service.
- **`.gcloudignore` / `.dockerignore`** exclude `.streamlit/`, `*.json` (service-account keys), and `notebooks/` from the build context — never remove those entries.
- **Config:** 4 GiB / 2 CPU, `--timeout 3600`, `--session-affinity`, `--max-instances 1` (Streamlit session state is per-instance; do not scale out without sticky sessions verified), runs as `matcher-app@cc-matcher-v1.iam.gserviceaccount.com`.

### Build and deploy (run every time app/view code changes)

```bash
gcloud builds submit --config cloudbuild.app.yaml --project cc-matcher-v1 .

gcloud run services update matcher-app \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/matcher-app:latest \
  --region us-central1 --project cc-matcher-v1
```

### One-time setup (already done — reference only)

```bash
gcloud services enable iap.googleapis.com --project cc-matcher-v1

# Secrets file into Secret Manager
gcloud secrets create streamlit-secrets --data-file=.streamlit/secrets.toml --project cc-matcher-v1
gcloud secrets add-iam-policy-binding streamlit-secrets --project cc-matcher-v1 \
  --member="serviceAccount:matcher-app@cc-matcher-v1.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Deploy with IAP enabled, no public access
gcloud run deploy matcher-app \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/matcher-app:latest \
  --region us-central1 --project cc-matcher-v1 \
  --memory 4Gi --cpu 2 --timeout 3600 \
  --session-affinity --max-instances 1 \
  --service-account matcher-app@cc-matcher-v1.iam.gserviceaccount.com \
  --update-secrets=/app/.streamlit/secrets.toml=streamlit-secrets:latest \
  --no-allow-unauthenticated \
  --iap

# Grant the workspace group access through IAP
gcloud iap web add-iam-policy-binding --project cc-matcher-v1 \
  --resource-type=cloud-run --service=matcher-app --region=us-central1 \
  --member="group:team@bwcoconsulting.com" \
  --role="roles/iap.httpsResourceAccessor"
```

If sign-in loops or 403s for a team member, confirm they're in the `team@bwcoconsulting.com` group and that the IAP service agent (`service-{PROJECT_NUMBER}@gcp-sa-iap.iam.gserviceaccount.com`) has `roles/run.invoker` on the service (the `--iap` flag normally wires this automatically).

---

## Cloud Run Job Setup

Two Cloud Run Jobs run the heavy pipeline work so Streamlit never hits memory or timeout limits. Both share the same service account and Secret Manager secrets. Streamlit writes a config to GCS, triggers the job, then polls `status.json` for completion.

### GCP details
- **Project:** `cc-matcher-v1`
- **Region:** `us-central1`
- **Artifact Registry repo:** `matcher`
- **Job service account (shared):** `matching-job@cc-matcher-v1.iam.gserviceaccount.com`
- **Secret Manager secrets (shared):** `anthropic-api-key`, `openai-api-key`

| Job name | Entry point | Dockerfile | Requirements | Triggered from |
|----------|------------|------------|--------------|----------------|
| `matching-job` | `jobs/matching_job.py` | `jobs/Dockerfile` | `requirements.job.txt` | Bulk Matching view |
| `sam-gov-job` | `jobs/sam_gov_job.py` | `jobs/Dockerfile.sam_gov` | `requirements.sam_gov_job.txt` | SAM.gov Upload view (manual) + Cloud Scheduler (daily) |
| `contact-import-job` | `jobs/contact_import_job.py` | `jobs/Dockerfile.contact_import` | `requirements.contact_import_job.txt` | Contact Importer view |
| `drive-sync-job` | `jobs/drive_sync_job.py` | `jobs/Dockerfile.drive_sync` | `requirements.drive_sync_job.txt` | Drive Sync view |
| `client-profile-job` | `jobs/client_profile_job.py` | `jobs/Dockerfile.client_profile` | `requirements.client_profile_job.txt` | Client Profiles view |

### `matching-job` config schema (`job-configs/{run_id}.json`)

```json
{
  "run_id":         "2026-04-16_10-30-00_ag-DOD_src-apollo",
  "threshold":      0.82,
  "top_k":          5,
  "sources":        ["apollo", "sba"],
  "agencies":       ["DOD", "HHS"],
  "topic_filters":  [
    {"column": "title", "type": "keyword", "keyword": "cyber", "operator": "AND"},
    {"column": "open_date", "type": "date_range", "date_from": "2026-06-01", "date_to": "2026-06-30", "operator": "AND"}
  ],
  "ai_validation":  true,
  "prewrite_email": false
}
```

### One-time setup (already done — skip if job exists)

```bash
# Enable APIs
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  cloudbuild.googleapis.com

# Create Artifact Registry repo
gcloud artifacts repositories create matcher \
  --repository-format=docker \
  --location=us-central1

# Create job service account
gcloud iam service-accounts create matching-job \
  --display-name="Matching Job Runner"

# Grant GCS access
gcloud projects add-iam-policy-binding cc-matcher-v1 \
  --member="serviceAccount:matching-job@cc-matcher-v1.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# Grant Secret Manager access
gcloud projects add-iam-policy-binding cc-matcher-v1 \
  --member="serviceAccount:matching-job@cc-matcher-v1.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Build and deploy (run this every time job code changes)

Run from the repo root (`~/matcher-public` in Cloud Shell):

```bash
# Step 1 — write cloudbuild.yaml (only needed once or after deleting it)
printf 'steps:\n- name: "gcr.io/cloud-builders/docker"\n  args: ["build", "-t", "us-central1-docker.pkg.dev/cc-matcher-v1/matcher/matching-job:latest", "-f", "jobs/Dockerfile", "."]\nimages:\n- "us-central1-docker.pkg.dev/cc-matcher-v1/matcher/matching-job:latest"\n' > cloudbuild.yaml

# Step 2 — build image and push to Artifact Registry
gcloud builds submit --config cloudbuild.yaml .

# Step 3 — first deploy (only needed once)
gcloud run jobs create matching-job \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/matching-job:latest \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --task-timeout 3600 \
  --max-retries 0 \
  --service-account matching-job@cc-matcher-v1.iam.gserviceaccount.com

# Step 3 (subsequent deploys — use update instead of create)
gcloud run jobs update matching-job \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/matching-job:latest \
  --region us-central1
```

### Verify the job exists

```bash
gcloud run jobs list --region us-central1
```

### Trigger a test run manually

```bash
gcloud run jobs execute matching-job --region us-central1
```

### View logs

```bash
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=matching-job" \
  --limit 50 --format "value(textPayload)"
```

---

### `sam-gov-job` — build, deploy, and manage

#### `sam-gov-job` config schema (`sam-gov-configs/{run_id}.json`)

Manual one-off run (date range explicit):
```json
{
  "run_id":      "sam_gov_2026-06-01_10-30-00",
  "input_mode":  "api",

  // CSV mode only:
  "csv_blob_path": "sam-gov-uploads/sam_gov_2026-06-01_10-30-00.csv",
  "col_map": {"title": "Opportunity Title", "description": "Synopsis", "notice_id": "Notice ID", "agency": "Department", "posted_date": "Posted Date", "deadline": "Response Deadline", "source_url": "Contract Opportunity URL"},

  // API mode only:
  "api_params": {
    "date_from":       "01/01/2026",
    "date_to":         "06/01/2026",
    "notice_types":    ["p", "o", "k", "r"],
    "keyword":         "",
    "max_results":     500,
    "fetch_desc":      true,
    "sam_gov_api_key": "..."
  },

  "custom_cols": {"campaign_name": "Spring 2026"}
}
```

Revision check run (written by the "🔁 Revision Check" expander in the SAM.gov Upload view):
```json
{
  "run_id":      "sam_gov_revcheck_2026-07-13_10-30-00",
  "input_mode":  "revision_check",
  "api_params":  {
    "sam_gov_api_key":     "...",
    "include_attachments": true,
    "max_api_calls":       600
  },
  "dry_run":     true
}
```
Its status payload uses a different shape: `{run_id, mode: "revision_check", dry_run, rows_candidates, rows_checked, rows_remaining, revisions_found, revisions_deferred, rows_archived, rows_updated, lookup_errors, api_calls_used, api_call_budget, stopped_early ("quota" | "budget" | null), revisions: [{topic_number, title, changed, notes}], archived: [{topic_number, title}], error}` (revisions/archived lists capped at 200 entries).

Daily schedule config (`sam-gov-configs/daily_schedule.json`) — written by the SAM.gov Upload UI, read by Cloud Scheduler each morning. `run_id: "daily"` is a sentinel; the job replaces it with a timestamped ID at runtime. `lookback_days` replaces explicit `date_from`/`date_to` — the job computes `date_from = today - N days` at execution time:
```json
{
  "run_id":      "daily",
  "input_mode":  "api",
  "api_params": {
    "lookback_days":   1,
    "notice_types":    ["p", "o", "k", "r"],
    "keyword":         "",
    "max_results":     500,
    "fetch_desc":      true,
    "sam_gov_api_key": "..."
  },
  "custom_cols": {}
}
```

#### Status schema (`sam-gov-jobs/{run_id}/status.json`)

```json
{
  "run_id":                "sam_gov_2026-06-01_10-30-00",
  "rows_fetched":          1000,
  "rows_passed_screening": 300,
  "rows_after_dedup":      280,
  "rows_saved":            280,
  "rows_revised":          4,
  "revisions":             [{"topic_number": "...", "title": "...", "changed": true, "notes": "..."}],
  "gcs_path":              "data/all-topics/processed/SAM-GOV/sam_gov_2026-06-01_abc123.parquet",
  "error":                 null
}
```

`rows_revised`/`revisions` report stored notices that were updated in place because the daily/manual pull carried a new version of them (see Revision handling above). Revision-check runs write the different payload documented under the config schema.

#### One-time setup (run once — skip if job already exists)

The service account and Secret Manager secrets are already configured for `matching-job` and are reused here.

```bash
gcloud run jobs create sam-gov-job \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/sam-gov-job:latest \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --task-timeout 14400 \
  --max-retries 0 \
  --service-account matching-job@cc-matcher-v1.iam.gserviceaccount.com \
  --set-secrets=ANTHROPIC_API_KEY=anthropic-api-key:latest \
  --set-secrets=OPENAI_API_KEY=openai-api-key:latest
```

#### Build and deploy (run every time `sam_gov_job.py` changes)

Run from the repo root after `git pull origin <branch>`:

```bash
# Step 1 — write the build config (use heredoc, not printf — avoids YAML parse errors)
cat > cloudbuild.sam_gov.yaml << 'EOF'
steps:
- name: "gcr.io/cloud-builders/docker"
  args:
  - "build"
  - "-t"
  - "us-central1-docker.pkg.dev/cc-matcher-v1/matcher/sam-gov-job:latest"
  - "-f"
  - "jobs/Dockerfile.sam_gov"
  - "."
images:
- "us-central1-docker.pkg.dev/cc-matcher-v1/matcher/sam-gov-job:latest"
EOF

# Step 2 — build and push
gcloud builds submit \
  --config cloudbuild.sam_gov.yaml \
  .

# Step 3 — update the existing job (image + 4-hour timeout)
gcloud run jobs update sam-gov-job \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/sam-gov-job:latest \
  --task-timeout 14400 \
  --region us-central1
```

#### View logs

```bash
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=sam-gov-job" \
  --limit 50 --format "value(textPayload)"
```

---

### Daily SAM.gov Schedule (Cloud Scheduler)

Cloud Scheduler triggers `sam-gov-job` every morning at 5 AM CST by POSTing to the Cloud Run Jobs API. The job reads `sam-gov-configs/daily_schedule.json` from GCS — all parameter changes are made through the SAM.gov Upload UI and saved to that config; no redeploy is needed.

#### One-time Cloud Scheduler setup (run once in Cloud Shell)

The OAuth service account the Scheduler impersonates (`matching-job@`) must be able to run `sam-gov-job` **with overrides** — the trigger body carries `containerOverrides`, so `roles/run.invoker` is NOT enough (`run.jobs.runWithOverrides` lives in `roles/run.admin`, same as the contact-import-job pattern). Granting the wrong role fails silently: the scheduler reports `status.code: 7` (PERMISSION_DENIED) on every attempt and no Cloud Run execution is ever created, so nothing shows up in the job's execution history.

```bash
# Allow the scheduler's OAuth service account to run the job with overrides
gcloud run jobs add-iam-policy-binding sam-gov-job \
  --region us-central1 \
  --member="serviceAccount:matching-job@cc-matcher-v1.iam.gserviceaccount.com" \
  --role="roles/run.admin"

# Create the daily trigger. NOTE: the cron expression is interpreted in the
# job's --time-zone, so "0 5 * * *" + America/Chicago = 5 AM Central year-round.
gcloud scheduler jobs create http sam-gov-daily \
  --schedule="0 5 * * *" \
  --uri="https://run.googleapis.com/v2/projects/cc-matcher-v1/locations/us-central1/jobs/sam-gov-job:run" \
  --message-body='{"overrides":{"containerOverrides":[{"args":["sam-gov-configs/daily_schedule.json"]}]}}' \
  --oauth-service-account-email=matching-job@cc-matcher-v1.iam.gserviceaccount.com \
  --location=us-central1 \
  --time-zone="America/Chicago"
```

#### Update the schedule time

```bash
gcloud scheduler jobs update http sam-gov-daily \
  --schedule="0 5 * * *" \
  --location=us-central1
```

#### Check whether the daily trigger is working

Scheduler-triggered executions appear in the same history as manual ones — Cloud Run → Jobs → sam-gov-job → Executions, or `gcloud run jobs executions list --job sam-gov-job --region us-central1`. The **RUN BY** column distinguishes them: `matching-job@` = Cloud Scheduler, `matcher-app@` = triggered from Streamlit. If no `matching-job@` executions exist, check the scheduler itself: `gcloud scheduler jobs describe sam-gov-daily --location=us-central1` — a non-empty `status.code` (7 = PERMISSION_DENIED) means the trigger is failing before any execution is created.

#### Trigger a manual test run of the daily schedule

```bash
gcloud scheduler jobs run sam-gov-daily --location=us-central1
```

#### How the daily run_id works

The daily config uses `"run_id": "daily"` as a sentinel. When `sam_gov_job.py` sees this, it generates a real timestamped ID (`sam_gov_YYYY-MM-DD_HH-MM-SS`) at startup. Status is written to `sam-gov-jobs/sam_gov_{date}/status.json` — each day gets its own status file; no file is overwritten.

---

### `contact-import-job` — build, deploy, and manage

#### `contact-import-job` config schema (`contact-import-configs/{run_id}.json`)

```json
{
  "run_id":        "contact_import_2026-06-25_10-30-00_apollo",
  "source":        "apollo",
  "file_ext":      ".csv",
  "csv_blob_path": "contact-import-uploads/contact_import_2026-06-25_10-30-00_apollo.csv",
  "col_map": {
    "companyWebsite": "Website URL",
    "companyName":    "Company Name",
    "state":          null,
    "segment":        "Industry",
    "firstName":      "First Name",
    "lastName":       "Last Name",
    "email":          "Email",
    "phone":          "Phone Number"
  },
  "profile_method": "scrape",
  "research_model": "gpt-5.6-terra",
  "dedup_all_sources": false
}
```

`col_map` values are actual column names from the uploaded file; `null` = unmapped optional field. `file_ext` is `.csv`, `.xlsx`, or `.xls`. `profile_method` is `"scrape"` (default) or `"deep_research"`; `research_model` (deep_research only) is one of the `DEEP_RESEARCH_MODELS`. `dedup_all_sources: true` makes the job's runtime dedup compare against all of `data/all-contacts/` instead of only the source's folder (set automatically by the UI's dedup-scope checkbox; default ON for HubSpot-list pulls).

#### Status schema (`contact-import-jobs/{run_id}/status.json`)

```json
{
  "run_id":          "contact_import_2026-06-25_10-30-00_apollo",
  "rows_fetched":    500,
  "rows_after_dedup": 450,
  "rows_scraped_ok": 400,
  "rows_saved":      400,
  "gcs_path":        "data/all-contacts/apollo/apollo_2026-06-25_abc123.parquet",
  "error":           null,
  "profile_method":  "scrape"
}
```

Deep-research runs (`profile_method: "deep_research"`) add: `research_model`, `companies_researched`, `companies_research_ok`, `research_cost_usd` (actual, from `response.usage`). `rows_scraped_ok` then counts rows whose company was successfully researched.

#### Job pipeline (`contact_import_job.py`)

Mostly self-contained, but imports the streamlit-free shared research modules `src/modules/finance_research.py` + `src/modules/tech_research.py` (copied into the image by `Dockerfile.contact_import` along with the empty `src/`/`src/modules/` `__init__.py` files — keep those COPY lines when editing the Dockerfile):

1. Download staged file from `contact-import-uploads/` → parse CSV/Excel
2. Apply `col_map` → standard fields; strip Excel HYPERLINK formulas via `_strip_hyperlink()`; normalize URLs
3. Load existing bare domains from `data/all-contacts/{source}/` parquets → filter duplicates
4. Build company profiles per `profile_method`:
   - **`scrape`** (default) — async scrape: aiohttp (8 concurrent semaphore) → Playwright fallback with `--no-sandbox --disable-dev-shm-usage` (required in Docker; these args are NOT in `lead_importer._playwright_scrape` — always inline Playwright in the job); then summarize: `ThreadPoolExecutor(max_workers=10)` → `gpt-3.5-turbo`
   - **`deep_research`** — one background Responses-API Deep Research task per unique bare domain (`research_model` from config, `web_search` tool), polled every 30s with a 6000s deadline (leaves headroom in the 7200s task timeout; deadline-exceeded tasks are cancelled best-effort and their rows skipped). Output parsed via `fr.parse_research_output(fields=tr.ALL_FIELDS)`; per-domain results fan out to all contact rows of that domain — `company_summary` = `tr.build_matching_summary()`, plus `technology_data`/`technology_summary`/`technology_updated_at` columns. Actual cost accumulated from `response.usage`.
5. Free `raw_df`, `mapped_df`, `new_df` (and scrape-path page text) before embedding (prevents OOM on large imports)
6. Embed: `ThreadPoolExecutor(max_workers=8)` → `text-embedding-ada-002` with 7500-token tiktoken guard
7. Save parquet to `data/all-contacts/{source}/{source}_{date}_{hex6}.parquet`
8. Write `contact-import-jobs/{run_id}/status.json` — deep-research runs add `profile_method`, `research_model`, `companies_researched`, `companies_research_ok`, `research_cost_usd`

**Resource config:** 8 GiB RAM, 2 CPU, 7200s timeout. Playwright runs up to 8 concurrent Chromium processes (~200 MB each); large imports (30k+ rows) require the explicit `del` of page text and summaries before embedding to stay within memory.

#### One-time setup (run once — skip if job already exists)

```bash
gcloud run jobs create contact-import-job \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/contact-import-job:latest \
  --region us-central1 \
  --memory 8Gi \
  --cpu 2 \
  --task-timeout 7200 \
  --max-retries 0 \
  --service-account matching-job@cc-matcher-v1.iam.gserviceaccount.com \
  --set-secrets=OPENAI_API_KEY=openai-api-key:latest
```

The Streamlit app's service account (`matcher-app@`) needs `roles/run.admin` on the job (not just `run.invoker` — `runWithOverrides` requires the higher role):

```bash
gcloud run jobs add-iam-policy-binding contact-import-job \
  --region us-central1 \
  --member="serviceAccount:matcher-app@cc-matcher-v1.iam.gserviceaccount.com" \
  --role="roles/run.admin"
```

#### Build and deploy (run every time `contact_import_job.py` changes)

```bash
# cloudbuild.contact_import.yaml is checked into the repo root — recreate only if missing:
cat > cloudbuild.contact_import.yaml << 'EOF'
steps:
- name: "gcr.io/cloud-builders/docker"
  args:
  - "build"
  - "-t"
  - "us-central1-docker.pkg.dev/cc-matcher-v1/matcher/contact-import-job:latest"
  - "-f"
  - "jobs/Dockerfile.contact_import"
  - "."
images:
- "us-central1-docker.pkg.dev/cc-matcher-v1/matcher/contact-import-job:latest"
EOF

gcloud builds submit --config cloudbuild.contact_import.yaml .

gcloud run jobs update contact-import-job \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/contact-import-job:latest \
  --region us-central1
```

Note: Playwright + Chromium installation adds ~5–8 minutes to the build vs. other jobs.

#### View logs

```bash
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=contact-import-job" \
  --limit 50 --format "value(textPayload)"
```

---

### `drive-sync-job` — build, deploy, and manage

#### `drive-sync-job` config schema (`drive-sync-configs/{run_id}.json`)

```json
{
  "run_id":                "drive_sync_2026-08-11_15-30-00",
  "drive_id":              "0ABc...",
  "folder_ids":            ["<assigned client folder ids to sync>"],
  "new_client_folder_ids": ["<unassigned folder ids to propose>"],
  "full_resync":           false,
  "dry_run":               false,
  "max_docs_per_client":   40,
  "per_client_char_cap":   150000,
  "max_proposals":         40,
  "task_timeout_s":        14400,
  "model":                 "claude-sonnet-4-6"
}
```

The job resolves `folder_ids → client_key` through `drive-sync-configs/assignments.json` and groups multi-folder clients into one unit of work. `full_resync` bypasses the `sync_state.json` modifiedTime diff; `dry_run` reports outcomes without touching parquets or sync_state. **Time budget:** `task_timeout_s` (chosen in the view's "Time budget" selector — 1 h to 24 h, default 4 h; clamped to 900–86 400 s in the job) sets the window, and the job stops gracefully ~10 min before it (`stopped_early: "timeout"`), deferring remaining clients/proposals to the next run — a hard timeout kill would lose un-checkpointed work and leave no status file. The Cloud Run job is deployed with `--task-timeout 86400` (Cloud Run's 24 h maximum), so any selectable budget is safe; **never set `task_timeout_s` above the deployed task timeout**. When proposal candidates exist, the client phase gets a tighter deadline (90s reserved per planned proposal, capped at half the budget) so a heavy client sweep can't starve the proposals phase. **Proposal cap + rotation:** at most `max_proposals` new-client proposals per run (each ≈1 min: folder download + Claude call) — the view now sets it to the number of folders the user explicitly picked, so only the time budget can defer them; when it does bite, candidates are ordered never-proposed-first then least-recently-proposed via the `proposed` cursor in `sync_state.json` (advanced on non-dry runs only), so a large unassigned backlog drains across successive runs instead of re-proposing the same chunk.

#### Status schema (`drive-sync-jobs/{run_id}/status.json`)

```json
{
  "run_id": "...", "state": "complete", "dry_run": false, "stopped_early": null,
  "clients_total": 42, "clients_updated": 17, "clients_unchanged": 22, "clients_errored": 3,
  "files_scanned": 812, "files_changed": 63,
  "files_skipped": [{"name": "big.pdf", "reason": ">15MB"}],
  "results": [{"client_key": "...", "folder_ids": ["..."], "outcome": "updated|unchanged|error|deferred",
               "files_processed": 4, "summary_changed": true, "note": ""}],
  "new_client_proposals": [{"folder_id": "...", "folder_name": "...", "proposed_name": "...",
                             "proposed_website": "", "website_source": "claude|domain_match|",
                             "candidate_domains": ["acme.com"], "proposed_summary": "...",
                             "docs_summary": "...", "docs_data": "{...}", "error": null}],
  "proposals_deferred": 0, "task_timeout_s": 14400, "max_proposals": 12,
  "error": null
}
```

`state: "running"` payloads are written at every 10-client checkpoint (`clients_done`/`clients_total`) and after **every** proposal (`proposals_done`/`proposals_total` + accumulated `new_client_proposals`) so the UI shows progress and a mid-run kill never loses completed proposal work; `files_skipped` is capped at 200 entries. `stopped_early: "timeout"` + `outcome: "deferred"` rows / `proposals_deferred` mean the time budget or proposal cap hit — re-running the same selection continues where it left off (already-synced files are skipped via `sync_state.json`). **Permanently unextractable files** (image-only PDFs, oversized exports, 404 shortcut targets — reasons matching `_PERMANENT_SKIP_MARKERS`) are marked synced with a `skip_reason` so they stop burning budget on every run; a Full re-scan retries them.

#### One-time setup (run once — skip if job already exists)

```bash
# Drive API must be enabled on the project (done 2026-08-11)
gcloud services enable drive.googleapis.com --project cc-matcher-v1

gcloud run jobs create drive-sync-job \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/drive-sync-job:latest \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --task-timeout 86400 \
  --max-retries 0 \
  --service-account matching-job@cc-matcher-v1.iam.gserviceaccount.com \
  --set-secrets=ANTHROPIC_API_KEY=anthropic-api-key:latest \
  --set-secrets=OPENAI_API_KEY=openai-api-key:latest \
  --project cc-matcher-v1

# Streamlit's service account needs run.admin for runWithOverrides (same as contact-import-job)
gcloud run jobs add-iam-policy-binding drive-sync-job \
  --region us-central1 \
  --member="serviceAccount:matcher-app@cc-matcher-v1.iam.gserviceaccount.com" \
  --role="roles/run.admin" \
  --project cc-matcher-v1
```

**Manual Drive step (Google Drive UI, not gcloud):** add both `matcher-app@cc-matcher-v1.iam.gserviceaccount.com` and `matching-job@cc-matcher-v1.iam.gserviceaccount.com` as **Viewer members of the client shared drive**. Without this the view's "List sections" returns nothing and the job 404s on every folder.

#### Build and deploy (run every time `drive_sync_job.py`, `doc_extract.py`, or `drive_client.py` changes)

```bash
gcloud builds submit --config cloudbuild.drive_sync.yaml --project cc-matcher-v1 .

gcloud run jobs update drive-sync-job \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/drive-sync-job:latest \
  --task-timeout 86400 \
  --region us-central1 --project cc-matcher-v1
```

#### View logs

```bash
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=drive-sync-job" \
  --limit 50 --format "value(textPayload)"
```

---

### `client-profile-job` — build, deploy, and manage

Builds the multi-aspect client profiles of Stage 8. Everything the job needs is already on the client rows, so there is no staging upload — the config just names the companies.

#### `client-profile-job` config schema (`client-profile-configs/{run_id}.json`)

```json
{
  "run_id":         "client_profile_2026-08-19_10-30-00",
  "company_keys":   ["Acme Robotics||https://acme.com"],
  "sources":        ["website", "drive", "technology"],
  "target_aspects": 4,
  "model":          "claude-sonnet-4-6",
  "concurrency":    4,
  "dry_run":        false
}
```

`company_keys` are `{company_name}||{companyWebsite}` identities (`aspect_profile.company_key()`). `sources` is any subset of `website`, `drive`, `technology`, `financials`; invalid keys are dropped and an empty result is a hard error. `concurrency` is clamped to 1–8 (each unit is one Claude call plus its aspect embeddings). `dry_run` runs the Claude calls and embeddings but never writes `profiles.parquet`.

#### Status schema (`client-profile-jobs/{run_id}/status.json`)

```json
{
  "run_id": "...", "state": "running|complete|error", "dry_run": false,
  "model": "claude-sonnet-4-6", "sources": ["website", "drive"], "target_aspects": 4,
  "clients_total": 40, "clients_done": 17,
  "built": [{"company_key": "...", "company_name": "...", "n_aspects": 4,
             "sources_used": "website,drive", "aspect_labels": "A | B | C | D"}],
  "errors": ["Acme Robotics: invalid response twice: ..."],
  "deferred": ["Beta Systems"],
  "stopped_early": null, "profiles_blob": "data/client-profiles/profiles.parquet",
  "error": null
}
```

A `running` payload is written at start and every 5 completed clients; the view renders `clients_done / clients_total` as a progress bar and resumes monitoring by run ID.

**Concurrency safety:** the profile store is a single blob. Before every save the job **re-reads `profiles.parquet` from GCS** and upserts the records built so far into that fresh copy, so a profile edited in the view mid-run is not clobbered wholesale (the run still wins for the companies it rebuilt). Same reason the checkpoints re-upsert everything rather than appending.

**Time budget:** the job stops handing out work ~5 min before the 7200s task timeout; unstarted clients come back as `deferred` with `stopped_early: "timeout"` and the view tells the user to build them again. Profiles already built are safe — they were checkpointed.

#### One-time setup (run once — skip if job already exists)

```bash
gcloud run jobs create client-profile-job \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/client-profile-job:latest \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --task-timeout 7200 \
  --max-retries 0 \
  --service-account matching-job@cc-matcher-v1.iam.gserviceaccount.com \
  --set-secrets=ANTHROPIC_API_KEY=anthropic-api-key:latest \
  --set-secrets=OPENAI_API_KEY=openai-api-key:latest \
  --project cc-matcher-v1

# Streamlit's service account needs run.admin for runWithOverrides
gcloud run jobs add-iam-policy-binding client-profile-job \
  --region us-central1 \
  --member="serviceAccount:matcher-app@cc-matcher-v1.iam.gserviceaccount.com" \
  --role="roles/run.admin" \
  --project cc-matcher-v1
```

#### Build and deploy (run every time `client_profile_job.py` or `aspect_profile.py` changes)

```bash
gcloud builds submit --config cloudbuild.client_profile.yaml --project cc-matcher-v1 .

gcloud run jobs update client-profile-job \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/client-profile-job:latest \
  --region us-central1 --project cc-matcher-v1
```

#### View logs

```bash
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=client-profile-job" \
  --limit 50 --format "value(textPayload)"
```

---

## Common Patterns

### Loading parquet files from GCS prefix

```python
import io, pandas as pd
from src.modules.grant_utils import normalize_grant_columns

def load_parquets_from_prefix(client, bucket: str, prefix: str) -> pd.DataFrame:
    blobs = client.list_blobs(bucket, prefix=prefix)
    frames = [pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
              for blob in blobs if blob.name.endswith('.parquet')]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return normalize_grant_columns(df)  # ensures grant_summary is always present
```

### Cosine similarity matching (vectorized)

```python
import numpy as np

# Both columns contain list[float] vectors of length 1536
contact_matrix  = np.stack(contacts['embeddings']).astype(np.float32)   # (n_contacts, 1536)
grant_embeddings = np.stack(topics['embeddings']).astype(np.float32)    # (n_topics, 1536)
scores = np.dot(contact_matrix, grant_embeddings.T)                     # (n_contacts, n_topics)
```

### LLM match verification (async Claude)

```python
system = (
    'You are evaluating whether a company could potentially benefit from or be relevant to a government grant. '
    'Answer "yes" if there is any reasonable connection, even if indirect. '
    'Answer "no" only if there is clearly no connection. '
    'Only respond with a single word: yes or no.'
)
response = await anth_client_async.messages.create(
    model='claude-haiku-4-5-20251001',
    max_tokens=15,
    system=system,
    messages=[{'role': 'user', 'content': f'Company: {company_summary}\n\nGrant: {grant_summary}'}],
)
result = response.content[0].text.strip().lower()
```

### Reading SAM.gov CSVs safely (Windows-1252 encoding)

```python
try:
    df = pd.read_csv(f, dtype=str, encoding='utf-8')
except UnicodeDecodeError:
    f.seek(0)
    df = pd.read_csv(f, dtype=str, encoding='latin-1')
```
