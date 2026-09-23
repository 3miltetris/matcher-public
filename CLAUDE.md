# CLAUDE.md — The Matcher

## Project Overview

**The Matcher** is a RAG (Retrieval-Augmented Generation) pipeline that matches companies/contacts to government grant programs (SBIR/STTR and similar). It ingests leads from sources like Apollo and SBA, processes grant topics from federal agencies, and uses OpenAI embeddings + cosine similarity + Claude/GPT LLM verification to identify strong company–grant alignments. Results are exported as CSV files for outreach campaigns.

The project has been **migrated from Google Colab notebooks + Google Drive** to a **Streamlit application** with consolidated Python modules.

---

## Current Architecture

```
app.py                        # Streamlit entry point — auth gate, playwright install, navigation
                              #   Nav is a SECTIONS DICT (Grants / Clients / Matching / Talent /
                              #   Export & admin). Renamed pages pin url_path so bookmarks survive.
packages.txt                  # Streamlit Cloud apt packages (Chromium system libs for Playwright)
views/
  home.py                     # Landing page — pipeline map + st.page_link into every page + opt-in "At a glance" counts
  grant_sources.py            # PARENT PAGE → Import Topics / SAM.gov / Grants.gov / Funding Sources
  client_sync.py              # PARENT PAGE → Google Drive / Fathom Meetings
  resumes.py                  # PARENT PAGE → Import / Search
  contact_importer.py         # Upload a lead spreadsheet OR pull a HubSpot company list → map columns → dedup (per-source or all-sources) → pick profiling method (scrape or Deep Research tech focus) → trigger contact-import-job → poll status
  client_editor.py            # Company Records — pick a pool (Clients / Prospects) → edit a company summary → re-embed → rewrite its source parquet; promote a prospect to client; admin-only delete (rows + profile + Drive assignment)
  finance_researcher.py       # Deep Research — OpenAI Deep Research on one pool's companies, two focuses: financials or technology/R&D, written back onto their rows (the run records its pool)
  topic_importer.py           # Upload PDF/text solicitations → extract topics → embed → save
  grant_search.py             # Search grant topics two ways: a typed technology description
                              #   (one embedding), or ONE company's capability profile -- picked
                              #   from the client store or built here from pasted notes (and
                              #   optionally saved into profiles.parquet). Replaced multi-aspect search
  bulk_matching.py            # Configure + trigger Cloud Run matching job, poll status
  sam_gov_upload.py           # Upload SAM.gov CSVs or fetch from API → dedup vs store → Claude screening → embed → save
  grants_gov_fetch.py         # Query Grants.gov public search2 API → Claude Haiku screening → embed → save to GRANTS-GOV/
  hubspot_import.py           # Import companies to HubSpot via Imports API — from a matching run or a financial research run (per-field mapping)
  resume_importer.py          # Upload HubSpot contacts CSV → fetch resumes by URL → extract text → GPT expertise summary → embed → save
  resume_search.py            # Natural-language search across resume embeddings → ranked candidate list
  drive_sync.py               # Scan the client Google shared drive → auto-assign folders to clients (fuzzy match, saved assignments) → trigger drive-sync-job → results + new-client review queue
  fathom_sync.py              # Fathom notetaker: connection test → metadata scan → external-invitee-domain → client assignments (review table) → pick clients → trigger fathom-sync-job → results + stored transcript browser
  client_profiler.py          # Pick a pool + companies + sources → trigger client-profile-job (multi-aspect capability profiles + their markets, from website summary / Drive docs / Deep Research) → poll status; review/edit aspects + markets in-process → data/client-profiles/profiles.parquet
  aspect_match.py             # Bulk multi-aspect match: whole company or per market (tier + category filters) × every selected grant topic → threshold/coverage filter → deduped Claude LLM re-rank → CSV
  funding_sources.py          # Master list of funding-source websites (cadence + per-site navigation instructions) → edit/import → trigger deep-research-job → poll status → API / login-wall / auto-paused flags
  suggestions.py              # Team feature-request board with upvoting
  admin_portal.py             # Admin-only: view/add/remove admins (admin-config/admins.json); super admins are a code constant

src/
  modules/
    Embedding/
      text_embedder.py        # TextProcessor class — embeddings, chunking, normalization
    GoogleBucketManager/
      bucket_manager.py       # BucketManager class — GCS upload/download
    fathom_client.py        # Streamlit-free Fathom REST client — global rate-limit gate, retry/backoff, iter_meetings/get_transcript, domain + field helpers (shared by the Fathom Meetings view and fathom-sync-job)
    WebScraper/
      web_scraper.py          # WebScraper class — Selenium-based scraper (legacy, unused; goto_url hardcodes headless=False so it cannot run in a container)
    email_generator.py        # async_generate_subject_line, async_josiah_copy
    grant_utils.py            # normalize_grant_columns() — coalesces description → grant_summary at load time
    finance_research.py       # Deep Research prompt/schema, JSON parse+repair, digest, cost accounting (Client Research view — financial focus + shared plumbing)
    tech_research.py          # Technology & R&D research schema/prompt/digest (Client Research view — tech focus; reuses finance_research parse/pricing)
    aspect_matching.py        # Streamlit-free scoring + LLM re-rank core (Stage 8) -- unit planning,
                              #   per-unit matmul, the confirmed/unexplored re-rank prompt split and
                              #   the re-rank dedup. Shared by Bulk Aspect Match and Grant Search
    aspect_profile.py         # Multi-aspect client profile schema/prompt/parse + market vocabulary, normalisation and near-duplicate merging + per-company material merge + flat-packed aspect/market embeddings + profiles.parquet store I/O — shared by Client Profiles + Bulk Aspect Match views and client-profile-job
    pools.py                  # Company pools (Stage 11) — the clients/prospects registry: contacts
                              #   prefix + profile blob + capabilities per pool, the per-blob frame
                              #   loader, and the company-column normalisation that lets lead-import
                              #   parquets read like clients ones. Streamlit-free (3 jobs import it)
    pool_transfer.py          # Streamlit-free pool-to-pool move (Stage 11) — promote a prospect to
                              #   client: contact rows then profile row, destination written FIRST
    source_registry.py        # Funding-source master list (Stage 10) — sources.parquet schema/IO, cadence due logic, agency-routing inference, CSV import, per-source seen indexes; shared by Funding Sources view + deep-research-job
    browser_agent.py          # Claude-drives-Playwright tool loop (Stage 10) — open_page/click/go_back/find_on_page/report_findings, manual agentic loop, per-site tool/page/time budgets, container-safe Chromium args
    ui_common.py              # Shared Streamlit-side plumbing — BUCKET + path prefixes, get_credentials/
                              #   get_storage_client, list_prefixes, load_parquets_from_prefix, and the
                              #   Cloud Run write_job_config/trigger_job/poll_status trio. Streamlit-only;
                              #   deliberately NOT imported by aspect_profile.py (which the jobs import)
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
  fathom_sync_job.py          # Cloud Run Job — one cheap /meetings sweep → attribute by external invitee domain → fetch transcripts for new calls → store raw JSON + index parquet → Claude per-client digest → write meeting columns (never touches summary/embeddings)
  client_profile_job.py       # Cloud Run Job — merge each client's material rows → Claude aspect + market split (concurrent) → embed each aspect and market narrative → merge near-identical markets → upsert data/client-profiles/profiles.parquet
  deep_research_job.py        # Cloud Run Job — pick due sites → Claude drives headless Chromium per site → dedup vs seen index → embed → write topic parquets per destination agency folder → update the registry
  Dockerfile                  # Image for matching-job (uses requirements.job.txt)
  Dockerfile.sam_gov          # Image for sam-gov-job (uses requirements.sam_gov_job.txt)
  Dockerfile.contact_import   # Image for contact-import-job — python:3.11-slim + Chromium system libs + playwright install chromium; also COPYs src/modules/{finance,tech}_research.py (shared Deep Research schema) + {pools,aspect_profile}.py + bucket_manager.py (pool routing)
  Dockerfile.drive_sync       # Image for drive-sync-job — python:3.11-slim, no Chromium; COPYs src/modules/{doc_extract,drive_client}.py
  Dockerfile.fathom_sync      # Image for fathom-sync-job — python:3.11-slim; COPYs src/modules/fathom_client.py + {pools,aspect_profile,finance_research,tech_research}.py + bucket_manager.py (attribution across both pools)
  Dockerfile.client_profile   # Image for client-profile-job — python:3.11-slim; COPYs src/modules/{aspect_profile,pools,finance_research,tech_research}.py + GoogleBucketManager/bucket_manager.py
  Dockerfile.deep_research    # Image for deep-research-job — python:3.11-slim + Chromium system libs + playwright install chromium; COPYs src/modules/{source_registry,browser_agent}.py

requirements.txt                    # Streamlit app dependencies (includes aiohttp, tldextract, playwright, google-api-python-client, openpyxl)
requirements.job.txt                # matching-job dependencies (lean — no Streamlit)
requirements.sam_gov_job.txt        # sam-gov-job dependencies (adds requests, beautifulsoup4, tiktoken, pymupdf)
requirements.contact_import_job.txt # contact-import-job dependencies (aiohttp, playwright, tldextract, openpyxl, tiktoken, openai, pandas/numpy/pyarrow, GCS)
requirements.drive_sync_job.txt     # drive-sync-job dependencies (anthropic, openai, google-api-python-client, pymupdf, python-docx, openpyxl, tiktoken)
requirements.fathom_sync_job.txt    # fathom-sync-job dependencies (anthropic, requests, pandas/numpy/pyarrow, GCS — no openai: this job never embeds)
requirements.client_profile_job.txt # client-profile-job dependencies (anthropic, openai, tiktoken, pandas/numpy/pyarrow, GCS)
requirements.deep_research_job.txt  # deep-research-job dependencies (anthropic, openai, playwright, beautifulsoup4, tiktoken, pandas/numpy/pyarrow, GCS)
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
- **Two search parameters were wrong from the start, and both failed silently (fixed 2026-09-15).** They are documented here because each produced *plausible* output — no error, no empty result, just less data than anyone thought.
  - **`offset` is a PAGE INDEX, not a record offset.** The fetch loop advanced it by the number of records returned (`offset += len(page_items)`), so page 2 of a 1,000-row page size asked SAM.gov for page **1,000**. The API answers an out-of-range page with `200 OK` and an empty `opportunitiesData` array, which the loop read as "no more results" and exited. Every API run therefore stopped at exactly 1,000 rows. Measured on the 2026-09-15 daily run: `SAM.gov total records: 1,749`, `fetched 1,000 items` — **~43–55% of each day's notices were never fetched, never screened, never stored**, for as long as the job has existed. Verified directly against the API: `limit=1000&offset=1000` → 200, `totalRecords: 2254`, **0 records**; likewise `limit=500` at offsets 1000 and 1500. The loop now increments the page by 1 and stops on `len(items) >= totalRecords`.
  - **The notice-type filter had never been applied.** The job sent `ntype=p,o,k,r,s`; the documented parameter is **`ptype`** ([GSA docs](https://open.gsa.gov/api/get-opportunities-public-api/)). SAM.gov ignores unknown query parameters, so every run silently pulled *all* notice types — award notices, justifications, surplus-property sales, intent-to-bundle. Confirmed by `ntype=p` returning the identical `totalRecords` (2,254) as `ntype=p,o,k,r,s`. This was not excluding anything, but it spent the (then 1,000-row) ceiling on notice types nobody asked for, which is why Claude screened out ~979 of every 1,000 rows daily.
  - **Consequence: the notice-type selection is now load-bearing, and the UI defaults had to change.** With `ntype` ignored, every type arrived regardless of what was ticked, so the manual-fetch default (`Solicitation, Presolicitation, Sources Sought`) was harmless. With `ptype` working it decides coverage — and **23 of the 25 Commercial Solutions Openings posted in a sampled 30-day window were `Special Notice`**. Special Notice is therefore now in the manual-fetch default and in the daily-config fallback; dropping it would silently cut CSO intake to near zero.
  - **A short read is now loud.** `_sam_search_all` compares `retrieved` against `totalRecords` and reports `truncated: true` per stream in `fetch_reports`, which the view renders as a red banner naming the shortfall. The old failure mode was total silence; anything that caps coverage must now say so.
- **Past awards are a separate store (`data/all-topics/awards/SAM-GOV/`), fetched by their own query.** `include_awards` in `api_params` adds one `ptype=a` query stream. Two things make this its own query rather than a filter over a combined pull: `active=Yes` is *wrong* for awards (an award notice auto-archives ~15 days after posting, so the live filter would hide nearly all of them, and the awards query deliberately omits `active`), and a narrower query keeps each stream's record count below any per-query ceiling. Awards run the full pipeline — screened with `_AWARD_SCREEN_SYSTEM` ("is this award worth tracking?", not "should we bid?"), summarized with `_AWARD_SUMMARY_SYSTEM`, embedded — and carry `awardee_name`, `award_amount` (+ an `award_amount_num` float companion, `None` never `0.0` when unparseable), `award_date`, `set_aside`, `base_type` and NAICS, all of which ride along free in the same search response. No revision handling: an award notice is final. A notice returned under the wrong `ptype` is re-routed by its own `type` field, so an award can never land in the solicitation store by accident.
  - **Why `awards/` sits outside `processed/`:** Grant Search, Bulk Matching and Bulk Aspect Match all enumerate the prefixes under `data/all-topics/processed/` as selectable "agencies", and **Bulk Matching pre-checks every one of them** (`value=True`). An `SAM-GOV-AWARDS` folder in there would have been matched against clients on every run by default — generating outreach for work that has already been awarded. Awards are reachable only by explicit path, which is why Grant Search needs its own opt-in checkbox to see them.
- **Revision handling (amended notices, e.g. revised CSOs):** SAM.gov amendments publish a new version with a new `noticeId` but the same `solicitationNumber` (our `topic_number`). Two mechanisms keep the store current:
  - **Ingest-time (API mode, automatic — including the daily run):** dup rows whose fetched `noticeId` differs from the stored `notice_version_id` are routed to an update path instead of being dropped — new description + attachment PDF text fetched, Claude diffs old vs new content (topics added/removed → `sam_revision_notes`), content re-summarized/re-embedded only when substantively changed (deadline-only amendments keep the old embedding), and the stored parquet rows are rewritten in place.
  - **`revision_check` job mode (manual budgeted sweep):** triggered from the "🔁 Revision Check" expander in the SAM.gov Upload view. Looks up stored open notices by `solnum` (walking back one-year `postedFrom`/`postedTo` windows — the API requires them and only returns the latest active version), updates revised notices, and marks notices no longer on SAM.gov with `sam_status='archived'` (rows kept; `matching_job` and Grant Search filter them out). Supports `dry_run` (report only — the UI default) so the report can be reviewed before applying. **SAM.gov enforces a hard daily request quota per API key** (shared with the daily fetch; the API exposes no `updatedDate` field, so each notice costs ≥1 lookup call), so each run spends at most `max_api_calls` (UI default 600) and sweeps **least-recently-checked notices first** using a cursor persisted at `sam-gov-configs/revcheck_state.json`. The cursor advances only on apply (non-dry) runs — a dry run and the apply run that follows cover the same chunk. A quota-exhaustion 429 ("exceeded your quota", Retry-After at midnight UTC) aborts the sweep immediately via `QuotaExhaustedError` instead of retrying (throttle 429s still back off); revisions detected but not yet content-fetched when quota dies are deferred (no partial writes) and re-detected next run. A full sweep of a large store completes over several daily runs.

- **The project now runs on a non-federal System Account key (approved 2026-09-21); the individual key it replaced died at ~35-40 requests/day.** On 2026-09-15 that individual key returned `429 {"code":"900804" … "You have exceeded your quota"}` after roughly **35–40 requests** (that day's daily run, ~20, plus ~15 diagnostic calls). Per [GSA's own guidance](https://open.gsa.gov/api/get-opportunities-public-api/), the tiers are: individual/personal key ≈ 10/day, **non-federal system account = 1,000/day**, federal system account up to 10,000/day — and only federal system accounts may request an increase. The system account was requested 2026-09-16 and **approved 2026-09-21**; the key was rotated into both places that hold it the same day (see "SAM.gov static egress IP" for the rotation checklist and the 90-day expiry).
  - **The 1,000/day ceiling is documented, not yet measured on this account.** The verification run of 2026-09-21 (`sam_gov_2026-09-21_14-19-54`) proved the key works **from the Cloud NAT egress IPs** — the one thing that could not be tested any other way, since a laptop request does not traverse the allowlisted addresses — but a Sunday `lookback_days: 1` window spent only **3 API calls**, an order of magnitude below where the old key failed. Nothing has yet pushed this key past ~40 requests. Treat the tier as unverified until a wide-window run or a `revision_check` sweep reports a high `api_calls_used` with `rows_deferred_quota: 0`.
  - `revision_check` mode has **still never executed** — no `mode: revision_check` status file exists in `sam-gov-jobs/`. It was previously guaranteed to abort on `QuotaExhaustedError` within a minute; with the system account it should now complete, and its `max_api_calls: 600` default (which always silently assumed this tier) is the natural way to find out where the real ceiling is. Note it shares the daily pool with the fetch.
  - `include_awards` remains **off** on the daily schedule. It roughly doubles the query count — affordable at 1,000/day, and the reason it was never enabled is now gone, so this is a preference rather than a constraint.
- **Quota exhaustion mid-run now defers instead of destroying the run.** Fixing the pagination roughly doubles the rows reaching the description step, which makes running out of quota mid-fetch far more likely. `_fetch_descriptions_batch` returns `(descriptions, quota_exhausted)` and marks unfetched rows `None`; those rows are **dropped from the run, not saved with an empty description** — an un-stored row is still "new" next time, whereas a row saved title-only would be deduped away forever. The count surfaces as `rows_deferred_quota` / `awards_deferred`. Previously `QuotaExhaustedError` propagated out of `main()` and discarded everything, including one Claude screening call per fetched row.

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
    - **Auto-assignment (`_auto_assign_properties`):** with the "🪄 Auto-assign properties" checkbox on (default), the mapping table is pre-filled from the portal's live property list instead of defaulting every row to *create new*. Tiers, most confident first: (1) the field's own `matcher_fin_<field>` already exists — a previous import created it, so reuse it; (2) a **custom** property whose name *or* label normalizes to the field name (`[^a-z0-9]` stripped, lowercased); (3) a curated standard-property alias from `_FIN_STANDARD_ALIASES` (`revenue_estimate_num`→`annualrevenue`, `employee_count_current_num`→`numberofemployees`, `total_venture_funding`→`total_money_raised`, `website_resolved`→`website`, …), reached **only** when the second checkbox ("Include standard HubSpot properties") is ticked — off by default because those hold CRM-owned data the import overwrites for every company in the run; (4) `difflib` ≥ 0.90 against custom property names/labels. Every tier requires a **compatible property type** (`_kind_fits`: a `_num` column only lands in a `number` property, prose only in a `string` one — so enumeration/date targets are never auto-picked and prose can never be written into `annualrevenue`), and each property is claimed by at most one field, so auto-assignment can't produce the duplicate-target error. Auto-assigned rows are also **checked for import**, and an "Auto-match" column plus a summary expander name the property and the tier that picked it so every guess is reviewable/overridable. The data_editor key includes a hash of the seed, so toggling either checkbox re-renders the table instead of showing stale edits.
    - **Numeric companions (for lead scoring):** research values are prose (`"21 (estimated)"`, `"0-250,000 (estimated)"`, `"~$4.2M"`, `"11-50"`), which HubSpot cannot sort, range-filter, or score on. Every money/count/score field listed in `finance_research.NUMERIC_FIELDS` (18 fields: revenue/funding/valuation, federal + SBIR award counts and totals, headcount, proposals per year, the 6 proposal-readiness scores, proposal budget, confidence) therefore also carries a `<field>_num` column holding **one integer** — the midpoint of a range, or the lone figure (`"0-250,000 (estimated)"` → `125000`, `"21 (estimated)"` → `21`, `"$250K-$1M"` → `625000`). Each `_num` row sits directly beneath its base field in the mapping table and auto-creates as a HubSpot **`number`** property (`matcher_fin_<field>_num`); a companion is pre-checked whenever its base field is. Unparseable values import as **blank, never 0**, so they read as "no data" instead of dragging a score down. A "🔢 Numeric companion columns" expander shows the parsed-vs-total count and an example conversion per field for spot-checking before import.
  - **Client profiles** — loads `data/client-profiles/profiles.parquet` (Stage 8) and flattens each company's aspect array into importable columns: `profile_summary`, `aspect_labels` (` | `-joined), `aspects_full` (numbered `label (kind)` / text / `Keywords:` block), `aspect_keywords` (deduped across aspects, first spelling wins), `aspect_kinds`, `n_aspects`, the market fields (`market_labels`, `markets_full` — a `1st · Market — subtitle` + narrative block per market, tier rendered as an ordinal via `ap.tier_ordinal(ap.market_tier_rank(m))` so legacy string tiers still export, `market_categories`, `n_markets`, `defense_market` yes/no, `defense_use_case` — the Defense narrative, or the recorded reason there is none), the unexplored-market fields (`unexplored_labels`, `unexplored_full` — narrative + `Gap remaining:` + `Draws on:` per market, `n_unexplored`), clearly named so nobody reads them as current business, `sources_used`, `profile_model`, `profile_built_at`, plus `aspect_{i}_label` / `aspect_{i}_text` per aspect (off by default; companies with fewer aspects get empty strings). A client multiselect (All / None buttons, everything selected by default) picks which profiles go over, then the same mapping table as financial mode targets existing properties or auto-creates `matcher_profile_summary`, `matcher_aspect_labels`, `matcher_aspects_full`, `matcher_aspect_keywords`, `matcher_aspect_count`, `matcher_market_labels`, `matcher_markets_full`, `matcher_market_count`, `matcher_defense_market`, `matcher_defense_use_case`, `matcher_aspect_{i}_*`, etc. Multi-line values are quoted CSV fields — HubSpot preserves the newlines.
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

**Stage 9 — Fathom meeting ingestion** (client calls → capability material for Stage 8)

Client calls are where clients actually describe their technology, their R&D, and the federal awards they have already won — detail that never makes it onto a website or into a Drive folder. Stage 9 pulls those calls from the [Fathom](https://developers.fathom.ai) notetaker, stores them durably, and distils them per client into a digest the Stage 8 aspect builder reads as its `meetings` source.

- **Attribution is by external calendar-invitee domain.** A meeting belongs to the client whose `companyWebsite` domain matches one of its external invitees; anything else lands in a review table and is remembered in `fathom-configs/assignments.json` (`domain → {client_key, match_type: auto|manual}`, plus `unassigned` / `skipped` buckets). Fathom's own HubSpot company match (`include_crm_matches`) is shown in the review table as a **hint only** and never auto-assigns. **Measured on the real store (2026-09-09): ~46% of swept meetings auto-attribute, leaving ~200 distinct unmatched domains.** Most of those are prospects and one-off calls that should be **skipped**, not clients that were mismatched — so triaging the review table once is the actual first-run workflow, not an edge case. Skipped domains are never offered again. Unattributed meetings are deliberately **not** marked synced, so assigning a domain later ingests its whole backlog on the next run without a full re-sync. Only meetings with at least one external invitee are swept — an internal-only call has no external domain and could never be attributed. **A meeting whose only external attendee is on a freemail domain is also unattributable:** it passes the API's `one_or_more_external` filter, but `fathom_client.bare_domain()` drops `GENERIC_DOMAINS` (gmail, outlook, …), so no domain survives to match on (19 of 400 meetings in a sample sweep). Those calls are reported as swept but never ingested, and there is no domain to assign in the review table.
- **Via Fathom Meetings view** (`views/fathom_sync.py`): connection test (lists the newest meetings *and their recorders*, so the key's real visibility is verifiable) → metadata-only scan over a chosen window (no transcripts, no LLM calls) that auto-matches domains and files the rest for review → domain review `st.data_editor` → client multiselect (All / None / Never synced / Has calls quick-picks, labels showing last-synced date and calls seen vs ingested) → look-back window, time budget (1–24 h), dry run, full re-sync, and an Advanced expander for the per-client caps → trigger `fathom-sync-job` → poll `fathom-jobs/{run_id}/status.json`. A final section browses `data/fathom/meetings_index.parquet` and loads any stored transcript.
- **`fathom-sync-job` pipeline:** one paginated `GET /meetings` sweep with `include_summary` + `include_action_items` + `include_crm_matches` (the AI summary rides along on each page, so only transcripts cost a per-meeting call) → attribute → skip `recording_id`s already in `fathom-configs/sync_state.json` → fetch transcripts for new attributed calls of the selected clients → store each call verbatim at `data/fathom/meetings/{recording_id}.json` and one row per call in `data/fathom/meetings_index.parquet` → one Claude (`claude-sonnet-4-6`) call per client merging prior digest + prior extraction + the new calls → write `client_meetings_data` / `client_meetings_summary` / `meetings_updated_at` onto every contact row of the client. Touched parquets + sync_state + interim status checkpoint every 10 clients; calls over `max_meetings_per_client` (default 12) or `per_client_char_cap` (default 120k) are **deferred, not marked synced**, so the next run picks them up. Active clients routinely exceed the per-client cap — the heaviest have 20–30 calls in a 90-day window against a default of 12 — so catching up a long-standing client takes 2–3 successive runs rather than one.
- **This job never touches `summary` or `embeddings`** — by construction, the merge prompt has no `updated_summary` field. Meeting material influences grant matching only through the multi-aspect profiles, so Bulk Matching results are unaffected until a profile is rebuilt.
- **Confirmed vs aspirational** is the prompt's central rule: transcripts are full of "we're thinking about pivoting to…". Only capability the company demonstrably has today goes in the main `extracted` arrays; anything hypothetical, planned, or belonging to a third party is pushed into `notable_updates` with its status made explicit, so the aspect builder can never read speculation as fact. This is why `aspect_profile._RULES` needed no change.
- **The digest is scoped to the company, not the engagement.** These are consulting calls, so most of what is said is about our own work: deliverables in preparation, submission deadlines, portal mechanics, who owes whom what. A first pass let that through and it flowed straight into the profile source text, because `_meetings_text()` feeds the digest verbatim. The prompt now excludes the project-management wrapper while explicitly keeping the substance underneath it — which agencies and programs the company is pursuing still lands in `federal_programs_agencies`, and the technology being demonstrated still lands in `technologies`. Measured effect on one client: the digest went from opening on grant-calendar admin to opening on its sensor stack, and `technologies` fell 14 → 6 as pursuit items dropped out.
- **`aspect_profile._meetings_text()` deliberately excludes the meeting list** (titles/dates live in `client_meetings_data['meetings']` for provenance and are shown in the view). Folding them into the source text would change `source_fingerprint` — and so flag every profile ⚠️ stale — every time a purely administrative call was ingested. When Claude reports `no_meaningful_change`, the prior digest and extraction are kept for the same reason.
- **Sweep size and duration are predictable:** `GET /meetings` returns **10 meetings per page**, so a 90-day window over current call volume is ~103 pages ≈ 1,000 meetings, and at the heavy-tier 2.5 s gate the sweep alone takes ~4–5 minutes before the first transcript is fetched. Budget the time window accordingly — the sweep is the floor on run duration, not the transcripts.
- **Rate limits are the binding constraint.** Fathom allows 60 requests/60 s in general but only **30/60 s for "heavy" calls** (anything carrying a summary or transcript), and its own docs warn this can collapse to 5/60 s under load. Every request therefore passes through one process-global pacing gate in `fathom_client` (1.05 s standard, 2.50 s heavy) with `Retry-After`-aware backoff; exhausted 429 retries raise `RateLimitStalledError`, which aborts the sweep with `stopped_early: "rate_limit"` rather than burning the task timeout on doomed waits.
- **API keys are per user, never per org.** A key sees meetings its owner recorded plus meetings shared with their team — never other people's private unshared calls. Whichever admin's key is configured therefore defines the ceiling on what the matcher can ever ingest; the connection test exists to make that visible before a big run.
- Webhooks (`new-meeting-content-ready`) are not used: they need a public HTTPS endpoint with HMAC verification, and the app is behind IAP.

**Stage 8 — Multi-aspect profiling & matching** (clients → per-aspect embeddings → re-ranked client×topic matches)

A client's single blended `summary` embedding averages away everything except its dominant theme, so a company with three unrelated capability areas matches poorly on all three. Stage 8 splits each client into a handful of independently embedded aspects and matches per aspect.

Aspects are additionally grouped into the **markets** they serve, so a client's defense story can be matched on its own instead of being averaged in with its commercial work. Market names come from the fixed `aspect_profile.MARKET_CATEGORIES` vocabulary (Defense, Aerospace & Space, Health & Life Sciences, …, Other) so one market means the same thing for every client and can be picked from a single dropdown across the directory; everything client-specific lives in the market's free-form `subtitle` and `narrative`. Every aspect is earmarked to at least one market.

**Market tiers are integer ranks, not buckets.** `tier` is `1` for the market most core to the business today, then `2`, `3`… — stored as an **int** and rendered `1st`/`2nd`/`3rd` only at display time (`aspect_profile.tier_ordinal()`); a stored `"1st"` would repeat the prose-instead-of-number mistake the HubSpot `_num` companions exist to undo. Profiles written before ranks existed carry the old `'primary'`/`'secondary'` strings, and **`aspect_profile.market_tier_rank()` coerces them on read** (`primary`→1, `secondary`→2, digit strings, ints), so nothing had to be rebuilt — the 282 live profiles (921 markets) all loaded unchanged, collapsing into ranks 1st and 2nd. Every read of a market's `tier` must go through that helper. `normalize_markets()` renumbers survivors densely 1…N after capping and merging, so a hand-edit or a dropped market can never leave `1st, 4th` with nothing between.

**Defense earns its rank** rather than being pinned first as it used to be (`normalize_markets` previously sorted on `market != DEFENSE_MARKET` before anything else). A defense-first company gets `1st Defense`; a commercial company with a loose dual-use angle gets Defense at a later rank, behind its actual core market — which is what makes a "tier 1 only" match run mean "each client's most important market". Defense keeps everything else that made it special: its own `_DEFENSE_RULES` assessment, exemption from the `max_markets` cap (so it can never be ranked out), survivor priority at equal rank in `merge_similar_markets`, and the `dod_assessment` fallback when there genuinely is none.

**Unexplored markets** answer the other question: not where the client sells today, but which customer world its capabilities *combine* into that it is **not** selling to yet. A **second** Claude call per client does this, deliberately not folded into the first — the aspect prompt's entire discipline is "ground everything, never invent", and asking that same call to also speculate would contradict its central rule and contaminate the grounded output. Pass 2 reads pass 1's **structured output** (aspects + confirmed markets + the client's own `notable_updates`) rather than the raw material again, since the question is what the aspects combine into. Its output lives in its own columns, its own embedding block and its own re-rank prompt — never a `status` flag on the confirmed `markets` array that every consumer would then have to remember to filter.

- **Via Client Profiles view** (`views/client_profiler.py`): reads only material that already exists on the client rows — website summary/scrape (`summary`, `full_text`/`page_text`), Fathom meeting material (`client_meetings_summary` + `client_meetings_data.extracted`, Stage 9), Drive extractions (`client_docs_summary` + `client_docs_data.extracted`), and Deep Research output (`technology_data`, `financial_data`) — per-source include checkboxes (financials off by default, since it describes money not capability). Material is read per company as the **first non-empty value across its contact rows** (a partially-updated file can leave some rows blank; `aspect_profile.merge_company_row()`) and capped per source (~8k–14k chars). The view itself only builds the directory (material available + profile status) and triggers the job — **building runs in `client-profile-job`** (config → `client-profile-configs/{run_id}.json`, status polled at `client-profile-jobs/{run_id}/status.json`, resumable by run ID), so a large batch survives a closed tab. **Pass 1** (`claude-sonnet-4-6` default, Haiku optional, one strict-JSON retry, 4 clients concurrently) returns `{profile_summary, aspects:[{label, kind, text, keywords, evidence, markets}], markets:[{market, tier, subtitle, narrative, keywords, aspects}], dod_assessment}` — 2–`MAX_ASPECTS` aspects (**ceiling 12**; the prompt asks for `target_aspects ± ~2`, and the view's target still defaults to 4, so raising the ceiling alone changes nothing — the **target slider is the lever**), each an independently searchable capability/technology/product/domain/market, plus up to `max_markets` (default 4) markets and a **Defense** market **only when the material evidences an actual DoD relationship or an active pursuit of one** (funding or award incl. a defense-agency SBIR/STTR, a named solicitation/contract/program being pursued, a defense agency or prime named as customer or partner, or a product the company itself describes as built for defense use). A merely plausible dual-use angle is explicitly **not** enough: it goes into `dod_assessment` as the reason there is no confirmed Defense market, and pass 2 raises it as an *unexplored* Defense market instead. **Why the bar exists:** the earlier wording ("include Defense whenever the connection is real even if it is loose") predated pass 2, when a loose angle had nowhere else to go. Measured across the live store, it put a confirmed Defense market on **115 of 282 profiles (41%) that contain zero defense tokens anywhere in their source material** — a music-therapy company carried Defense as its second-most-core market. Ranked tiers alone did not fix this (on four zero-evidence clients: one dropped Defense correctly, two demoted it to last, one was unchanged), because the prompt was still asking for it. Everything is grounded only in the supplied material; `notable_updates` lines are explicitly excluded from confirmed aspects and markets by the prompt, because they are what the client *said it plans*, not what it can do. Each aspect's `label + text + keywords` **and each market's `market + subtitle + narrative + keywords`** are embedded (`text-embedding-ada-002`).
  - **Near-duplicate aspects are merged, not dropped**: two aspect vectors of one company scoring ≥ `aspect_merge_threshold` (default **0.94**) are one capability written twice, so the longer text survives and absorbs the other's keywords, market membership and label in `merged_from` (`merge_similar_aspects()`). Dropping instead of merging would delete a capability and could leave a market pointing at nothing. Merging runs **before** market membership is re-derived, so no market can end up pointing at a folded-away aspect.
    - **The threshold is measured, and cosine is a weak signal here.** An initial 0.96 — reasoned from ada-002's compressed range rather than measured — was **unreachable and fired zero times**, which is strictly worse than not having the feature. Across three real clients rebuilt at 10–11 aspects the entire within-company pairwise distribution topped out at 0.890 / 0.902 / 0.915 (median ≈ 0.81). Worse, the ranking is not semantically meaningful: the clearest duplicates in that sample (`Single-Fiber CLE Optical Platform` ~ `Life Science and Research CLE Platform`, 0.894; `In-Service AST Floor Inspection Capability` ~ `Turnkey AST Inspection Service Delivery`, 0.890) scored **below** more debatable pairs (a device vs its clinical programme, 0.915). A 0.90 setting was then measured across the whole book (277 clients): it fired **33 times, of which roughly 12–14 were wrong in a systematic way** — it absorbs a platform into its own application (`Electronic Polymer Radiation Dosimeter Patch` ← `Occupational Radiation Safety Monitoring`; `TetherNet Aerostat Edge Platform` ← `Rural Broadband Aerostat ISP Delivery`) or merges two genuinely distinct products (`FOS Myopia Control Contact Lenses` → `FOS Myopia Control Spectacle Lenses`; two separate therapeutic assets at Asylia; a stenting platform into an IP portfolio). **A platform and its applications should stay separate aspects** — they match different grant topics, which is the whole point of multi-aspect profiling. The 383 reported near-misses clustered tightly (median 0.867, p90 0.888, max 0.900), so 0.90 sat *inside* a dense band rather than above it; there is no clean threshold to find. The default is therefore **0.94**, above the band, where it fires rarely and does little harm. Practical duplicate detection is `nearest_aspect_pairs()`, which reports every pair ≥ `ASPECT_NEAR_FLOOR` (0.85) that did **not** merge, per client, in `built[].aspect_near_pairs` and in an expander in the view — a reviewed list beats a silent deletion. Merges are named in `warnings` **with their cosine**, so the next adjustment is data-driven; the 0.90 round could not be, because only the near-misses carried scores. **Measured at 0.94 across the same 276 clients:** 3 auto-merges (down from 33), all defensible, against 747 reported near-misses (max 0.940, p90 0.911, median 0.883) — the threshold now sits at the top of the band rather than inside it.
  - Markets of one company whose narrative vectors score ≥ `market_merge_threshold` (default 0.93) are one market described twice and are folded together; `normalize_markets()` canonicalises names, caps the non-defense count, dense-renumbers ranks, and puts any aspect the model left unassigned into the **lowest-ranked** market.
  - **Pass 2** (optional, `assess_unexplored`, on by default) then returns `{unexplored_markets:[{market, tier, subtitle, narrative, keywords, rationale, aspects}]}` — up to `max_unexplored` (default 3) markets the client does **not** serve, each required to draw on ≥ `MIN_LINKED_ASPECTS` (2) named existing aspects and to carry a `rationale` stating the gap that would remain. Defense is assessed here too, so a client with no confirmed Defense market can still surface an unexplored one. `normalize_unexplored()` drops anything canonicalising to `Other`, anything the client **already serves** (a contradiction that would also duplicate topic retrieval), anything without a narrative (the narrative vector is the only thing an unexplored market is ever scored on), and filters `aspect_labels` against the real aspects so a hallucinated label cannot reach the re-rank prompt. **When that filter leaves nothing it backfills from the prose** (`_labels_in_text()`): observed on a real build, the model named the aspects it was combining in the `rationale` ("The Single-Fiber CLE Optical Platform and Real-Time Edge AI Inference Engine combine directly") while returning an **empty `aspects` array** for all three of one client's markets. Those labels are the only evidence the re-ranker judges an unexplored market against, so without recovery it is asked whether a company could extend into a topic without being told what the company can do — and correctly scores it low. Recovery restored 1,000–1,500 chars of capability evidence per market on that client. **An empty result is a legitimate answer** — a single-product client may support no market beyond the ones it serves. Pass 2 is wrapped in **its own error handling**: it is additive, and losing it must never discard the pass-1 work, which cost a Claude call and every embedding; a failure is reported in `warnings` and the profile saves with zero unexplored markets. Enabling it roughly **doubles per-client latency**, so the job's graceful time budget is spent twice as fast.
  - The profile is upserted as **one row per company** into `data/client-profiles/profiles.parquet`. Aspects, markets **and unexplored markets** are all editable in the view (three `st.data_editor`s — aspect membership is edited as a comma-separated `Markets` column and re-derived on save; market tier is a `NumberColumn`) with a re-embed-and-save button (still in-process — one company), plus delete. A market listed in **both** the served and unexplored tables is rejected at save with both names, rather than being silently dropped by `normalize_unexplored`.
- **Profiles built before markets existed** carry `n_markets` 0 and show as ⚠️ **no markets** in the directory: they still work in the whole-company match mode but are skipped (with a warning naming them) by market-scoped runs until rebuilt. They are pre-selected for rebuild alongside stale profiles. Profiles built before **unexplored markets** existed simply carry `n_unexplored` 0 — they are fully usable, and an Unexplored-kind run names them as needing a rebuild rather than failing. `_STATUS_NOMARKET` stays keyed on the **confirmed** `n_markets`, since an unexplored-only profile is still not usable for a confirmed run.
- **Staleness:** each profile stores a `source_fingerprint` (sha256 over *all* available source texts, not just the ones used). The view recomputes it live and flags profiles ⚠️ stale when the client's website/Drive/research material has changed since the build; the picker pre-selects stale + unprofiled clients. Clients with no material at all are listed separately, not offered.
- **Nothing is written to the client parquets** — profiles live in their own store, so Client Editor / Client Research / Drive Sync keep rewriting client rows freely.
- **Via Bulk Aspect Match view** (`views/aspect_match.py`): select profiled clients + grant agencies (same keyword/date filter UI as Grant Search), then pick a **match mode**. `All aspects (whole company)` is the original behaviour — one scoring unit per client holding every aspect vector. `By market` makes one unit per (client, market), holding only that market's earmarked aspect vectors **plus the market's own narrative vector** (labelled `Market: <name>` when it wins a topic — for Defense that narrative is the only place the DoD framing exists, so aspect-only scoring would miss topics it should hit). `By market` units come in two **kinds**, and three controls filter them: **market kind** (Confirmed / Unexplored / Both — default **Confirmed**, so an ordinary run behaves exactly as before and the unit count doesn't quietly balloon past the 2,500-call confirmation), a **tier-rank multiselect** populated only with the ranks the selected clients actually have (empty = every tier), and **market category** (All, or one canonical market with the count of selected clients that have it). The old `Defense only` scope is **gone** — `category = Defense` always did the same thing, and the scope existed only to bypass the category filter. Each unit is scored, capped and ranked as if it were its own company — one numpy matmul per unit against the topic matrix, and `top_k` applies **per unit**. A topic qualifies for a unit when its **best** vector score clears the threshold and at least `min_hits` of its **aspect** vectors clear it — the market narrative is scored and can win a topic, but it is one more description of the same market rather than an extra capability, so it counts towards `min_hits` only for a market no aspect was earmarked to (default 1 — a client's aspects are alternative capabilities, not conjunctive requirements of one query; raise it to demand topics spanning several capabilities. Grant Search's old multi-aspect mode, which decomposed one description into dimensions that ALL had to clear the threshold, was the opposite and no longer exists). An **unexplored** market unit holds its **narrative vector and nothing else**: scoring the confirmed aspect vectors under it would mostly return the topics its confirmed markets already returned, and the re-rank dedup keys on (client, agency, topic, aspect) — so one aspect winning the same topic under both kinds would share a single score written from whichever framing came first. Because such a unit has no aspect vectors, `min_hits` is forced to **1** for it explicitly (it previously only worked by the accident of `0` being falsy, and a `min_hits > 1` run would have dropped every unexplored unit with no explanation). Top-K topics per unit are kept with the winning aspect attributed (`market`, `market_kind`, `market_tier` as an ordinal, `market_subtitle`, `market_rationale`, `aspect_label`, `aspect_score`, `aspects_hit`, and a per-vector `aspect_scores` JSON). Survivors are re-ranked 1–5 by Claude (async `AsyncAnthropic`, 15 concurrent, exponential backoff on 429/529). **There are two re-rank prompts, and the split is load-bearing.** `_RERANK_SYSTEM` asks whether the client could propose *today* and says "never assume capabilities that are not stated" — pointed at an unexplored market, which is a hypothesis by construction, it correctly returns 1s and 2s, the default `Keep LLM score ≥ 3` drops every row, and the run shows an **empty table with no error at all** (the same silent-zero class as the anthropic-1.3.0 incident under SDK version pins). Unexplored units therefore get `_RERANK_UNEXPLORED_SYSTEM`, which asks how plausibly the client could **extend** into the topic from the capabilities it demonstrably has, and is shown the market hypothesis, the `rationale` gap, and the text of the confirmed aspects the market says it would draw on — so a hypothesis can't be scored on its own optimism. Every result row carries `market_kind`, so a 4 on a hypothesis is never read as a 4 on a real capability. Unparseable/failed pairs get score 0 and are reported, never silently promoted. **Re-rank calls are deduped** by (client, agency, topic, matched aspect, **market kind**): the same aspect can win the same topic under two markets, and the prompt carries no market context, so the pair is scored once and shared — but never across kinds, since the two prompts answer different questions. When a run ends with **no rows at all**, the results panel distinguishes the two very different causes: it reports how many candidates similarity scoring found, how many of them Claude could not score, and the distinct failure reasons verbatim — an unscored pair is stored as 0, which is below every selectable minimum, so a re-ranker outage silently filters away an entire run (see SDK version pins for the incident that motivated this). Rows ≥ the minimum LLM score are shown, downloadable as CSV, and saved to `aspect-match-results/{run_id}/results.csv`. Scoring and re-ranking run **in the Streamlit process** — the page must stay open; above 2,500 re-rank calls a confirmation checkbox is required.


- **Via Grant Search** (`views/grant_search.py`): the same match, for **one** company at a time, run from the topic-browsing view rather than over the directory. The profile comes either from `profiles.parquet` (a search box over name / website / market + aspect labels, so 280-odd profiles stay pickable) or is **built on the spot from pasted notes** — call notes, a deck, a capability statement — for a company we may hold no records for. That build is the same one `client-profile-job` runs: `ap.build_aspect_system` / `parse_aspect_response`, the aspect and market merges, `normalize_markets`, the optional unexplored pass, and `build_profile_record`, so the result is a **normal profile record** and a button writes it straight into `profiles.parquet` where Capability Profiles, Bulk Aspect Match and HubSpot Import all see it. Pasted material rides in as a source key of its own (`ap.NOTES_SOURCE_KEY`, capped at `NOTES_CAP`) rather than being disguised as website text, so `sources_used` says `notes` and the stored `source_fingerprint` is over what was actually pasted. Saving needs **both** a name and a website — together they are the `company_key` every other view joins on — and an existing key is called out as an overwrite before the button is offered. Scope is whole-company or a multiselect of that profile's markets (confirmed and unexplored, labelled), and everything downstream — thresholds, `min_hits`, top-K, the two re-rank prompts, the dedup — is `src/modules/aspect_matching.py`, so the same (client, topic, aspect) pair ranks identically here and in a directory-wide run. Saving a profile does **not** create client contact rows; it only adds the profile.
- **This replaced Grant Search's "multi-aspect search"**, which asked Claude to decompose the typed description into 2–4 required dimensions and kept only topics clearing the threshold on **all** of them. That is a narrowing operation on one description and the opposite of what a company profile wants: a company's aspects are alternative capabilities, and a topic matching any one of them is a real hit. Keeping both would have meant two different meanings of the word "aspect" one page apart. The plain single-embedding description search is untouched and is still the default mode.

**Stage 10 — Funding Source Watch** (curated websites → scheduled agentic browsing → grant topics)

Roughly 280 funding sources have no API and no feed: OTA consortium sites (NSTXL, S2MARTS, MTEC, the Manufacturing USA institutes, the `*werx` network), agency portals, and state economic-development programs. They were checked by hand from a Google Sheet on a Daily / Weekly / Monthly cadence, and the sheet's `Last Checked` column showed what that really meant — most rows months stale. Stage 10 automates the sweep.

- **The master list is a store, not a spreadsheet.** `deep-research-configs/sources.parquet`, one row per site, owned by `src/modules/source_registry.py` and edited in the **Funding Sources** view. Each row carries a `cadence`, free-form `instructions` for the agent, and the `broad_agency` folder its opportunities are filed under. `source_registry.import_rows()` seeds it from the exported sheet once; re-importing an updated export adds only what is new (dedup by normalized URL, tracking params stripped).
- **Claude drives a real browser, not a fetch tool.** `src/modules/browser_agent.py` gives Claude five tools over a headless Chromium — `open_page`, `click` (by numbered link, or by text for JS-only buttons), `go_back`, `find_on_page`, and the terminal `report_findings` — so it can follow instructions like "click through each challenge and read the full description". Pages are rendered from the live DOM (`page.content()` → BeautifulSoup), which is what makes JS-rendered listings readable and also produces the numbered link list the agent navigates by.
- **The loop is manual, deliberately.** `while stop_reason == "tool_use"`, not `client.beta.messages.tool_runner` — the tool runner is a beta surface and the 2026-09-02 SDK-pin incident is exactly the silent-failure class to stay away from. Same reason nothing here passes `temperature`/`top_p`/`top_k`.
- **Every site is budgeted three ways** — tool calls (default 25), pages (per-row, default 12), and wall clock (default 240 s). Blowing a budget is not an exception: the loop switches to `tool_choice={'type':'tool','name':'report_findings'}` and forces the report, so the work already done is still returned with a `stopped_early` reason. Without that, a site that ran long would lose everything the agent had found.
- **The seen index is what keeps this cheap.** `deep-research-configs/seen/{source_id}.json` maps an opportunity key (its own URL where it has one, else source + normalized title) to `{title, first_seen}`. It does double duty: it drops re-listings of still-open solicitations before they reach the store, and its most recent titles are pasted into the agent's prompt so the agent skips them instead of paying to re-extract them. A second, cheaper check compares titles against those already in the destination folder, which catches anything that arrived via Topic Importer or Grants.gov.
- **No separate screening call.** Unlike SAM.gov and Grants.gov, this is a *curated* list of approved programs, so relevance is assumed and the extraction call does the filtering: the prompt names what counts (solicitations, RFPs, project calls, BAAs, challenges, state matching funds) and what never does (news, already-made awards, events, closed items, "how to join" pages). An opportunity whose reported description is under 80 characters is dropped in code — that is the signature of a listing row the agent never actually opened, and it would embed to noise.
- **Routing is per-source and reviewable.** `infer_broad_agency()` suggests a destination folder from the host (`*.mil` → `DOD`, `nih.gov` → `HHS`, `<agency>.<state>.gov` → `STATE`, …); everything unrecognised defaults to `CONSORTIUM`. The suggestion is shown in the import preview and stored in an editable column — it is never a silent decision. Two traps the first version fell into: the `.us` rule must be narrow (`americamakes.us` and `nextflex.us` are Manufacturing USA institutes, not state agencies, so only the `<agency>.<state>.us` form counts), and **the rules must be checked against the folders that actually exist** — the store already separates `DARPA`, `DEVCOM`, `CIRM` and `TEXAS` from the broad `DOD`/`STATE` buckets, and routing `darpa.mil` to `DOD` would have split one source's opportunities across two folders. List `data/all-topics/processed/` before adding a rule.
- **`CONSORTIUM/` and `STATE/` sit under `processed/`, unlike `awards/`.** There is no agency constants list anywhere in this codebase — the set of agencies *is* the set of folder prefixes under `data/all-topics/processed/`, and `broad_agency` is injected from the folder name at load time. Creating the folder is therefore the whole wiring job, and **Bulk Matching pre-checks every folder it finds** (`value=True`). For past awards that was wrong and they were kept outside `processed/`; for these it is exactly right — they are live opportunities and should reach Bulk Matching and Bulk Aspect Match without anyone opting in. Do **not** write a `broad_agency` column into these parquets; every reader overwrites it.
- **Failure is a first-class outcome.** `last_checked` advances even on failure, so a broken site cannot monopolise every subsequent run, and `consecutive_failures` auto-sets `cadence: paused` at 3. The LinkedIn rows and `marketplace.gocolosseum.org` will hit this immediately — gocolosseum's example text in the sheet came from a logged-in session, and credentialed browsing is out of scope. Those sites are reported as `requires_login` and surfaced in the view's flags panel rather than retried forever.
- **Sites that expose an API are flagged, not scraped.** The agent reports `has_api` with evidence, backed by a deterministic regex over the page text, and the Funding Sources view banners them for the team to integrate directly (as Grants.gov Fetch already does). There is deliberately no email: `views/bulk_matching.py:176-213` has a working SMTP sender, but it reads `st.secrets` and lives in the view's poll loop, so it sends nothing during an unattended scheduled run. Moving it into the job means Secret Manager entries for the SMTP pair — a small, separate change.
- **Output is the canonical grant-topic schema** plus `record_kind='solicitation'`, `source_site`, `source_id`, `source_name`, `first_seen` and `deep_research_run_id`, written to `data/all-topics/processed/{BROAD_AGENCY}/deep_research_{date}_{hex6}.parquet`. Both `close_date` and `due_date` carry the deadline: `due_date` is what `matching_job` exports and Bulk Matching filters on, `close_date` is what Grants.gov writes, and both stores exist.
- **Buffers flush every 10 sites**, not once at the end. Embedding and the parquet write happen on a worker thread so a flush never stalls the browsers still running, and a timeout keeps the extraction already paid for. Anything found for a site is written before that site is marked checked.
- **Cost is driven by pages visited, not sites.** Measured on the first real run (4 sites, dry, `claude-sonnet-4-6`): `launchtn.org` 7 pages **$0.10** (1 opportunity), `marketplace.gocolosseum.org` 12 pages **$0.39** (7), `nstxl.org` 12 pages **$0.40** (8), `darpa.mil` 12 pages **$0.97** (26) — **$1.85 for 42 opportunities**, averaging **$0.46/site**. A full 278-site sweep at that rate is roughly **$130**.
  The nonlinearity is structural: every tool result appends a full ~12k-char page render to the conversation, and the whole conversation is re-sent on every turn, so a 12-page site pays for its early pages a dozen times over — spend grows with the *square* of pages visited. Actual spend is accumulated from `response.usage` and reported as `cost_usd` in `status.json` and in the view.
- **Prompt caching is what keeps that affordable — measured, keep it.** `roll_cache_breakpoint()` moves a single `cache_control` marker onto the newest tool result before every request, and `_SYSTEM_BLOCKS` carries a second one that caches tools + system (identical for every site, so it is reused across sites inside the TTL). A single rolling marker rather than one per turn, because the API allows at most 4 per request. Measured on `darpa.mil`, same site and same 12-page cap as the baselines above: **$0.971 → $0.590, a 39% cut, with no loss of extraction** (26 vs 32 opportunities — run-to-run variance, not an effect of caching, which cannot change what the model sees). Of ~238k input tokens, 129,560 were cache reads, 108,459 cache writes and only **416** were billed at full rate.
  Two things follow. **Caching fails silently** — a varying prefix or a misplaced marker just means you keep paying full price with no error — so `cache_read_tokens` / `cache_write_tokens` ride on every site result and are printed per site as `[cache r=… w=… in=…]`. **If `in=` is large, caching has broken.** And `_usage_cost()` must keep billing reads at 0.10x and writes at 1.25x; the original version billed cached reads at full input price, which would have hidden the entire saving. Note the write premium dominates (it was $0.41 of the $0.45 input spend), so the saving is smaller than the naive 10x-on-cache-reads arithmetic suggests.
- **Trimming the conversation was tried, measured, and rejected — do not re-add it.** The obvious fix for the above is to stub out older page renders (keep the first and the last two, replace the middle with a placeholder). A/B on `darpa.mil`, same site, same 12-page budget, only the trimming changed:

  | | opportunities | cost | cost each |
  |---|---:|---:|---:|
  | full history | **26** | $0.971 | $0.037 |
  | trimmed | 11 | $0.525 | $0.048 |

  Cost fell 46% but extraction fell 58% — strictly worse value per dollar. The cause is the report shape: the agent accumulates findings and emits them in **one** terminal `report_findings` call at the end, so a page whose text was stubbed out five turns earlier can no longer be reported. Trimming here does not discard redundancy, it discards the payload. (Single-trial comparison, but the mechanism is deterministic.) The way to get the saving properly is to make the agent record each opportunity **as it finds it** — an incremental `record_opportunity` tool — which moves the content out of the conversation entirely and makes trimming safe. That also needs `max_tool_calls` raised, since DARPA alone would spend 26 calls recording.
- **The first full sweep (2026-09-16) is the reference figure: 277 sites, 1,194 opportunities found, 856 new, saved into 66 parquets, zero errors, zero deferred, $56.40** — well under the $105-130 projected before caching existed. Caching carried it: **72.6% of 34.5M input tokens were served from cache and 0.32% were paid at full rate.** A full sweep takes just under two hours at concurrency 4.
- **12 pages was not enough, and the sweep says exactly for whom.** 74 sites used their entire page budget; they cost $25.86 (avg $0.349) and produced 539 of the 1,194 finds, while the 205 sites that finished early cost $30.54 (avg $0.149) for 721. So the capped sites are the productive ones, and they were being truncated mid-listing — DARPA's own note at 12 pages read *"the listing page was truncated before I could capture all 42 entries"*, and at a 25-page cap it used **17** pages, reported *"all 42 listed opportunities were loaded"*, and cost 29% more ($0.590 -> $0.762). Those 74 sites (plus 3 that ran out of *time*) are now set to `max_pages: 25`, which prices at **+$7.54 per sweep, $56 -> $64**.
- **Raising `max_pages` without raising `max_tool_calls` does nothing.** Every page costs at least one tool call, so a 25-page site needs far more than the 25-call default — DARPA spent 45 calls reaching 17 pages. At a 12-page cap **no site hit the tool budget**, which is exactly why it is easy to forget: the tool budget silently becomes the new page cap. The daily config and the view default are both 45, and per-site time went 240s -> 480s after 3 sites ran out.
- **Read the notes, not just the counts.** `stopped_early: page_budget` is appended to each site's `note`, which is how the 74 were identified. The same field surfaced a wrong URL (NSTXL's listing) and DARPA's RSS feed. Filtering `results[]` on that string is the standing way to decide which sites need more budget.

**Stage 11 — Company pools** (clients and targeted prospects, side by side)

The pipeline holds two directories of companies that get the *same* treatment — Deep Research, Fathom meetings, capability profiles, aspect matching, HubSpot export — and differ only in where they live and what may feed them:

| Pool | Contacts | Profiles | Fed by | Drive Sync |
|---|---|---|---|---|
| 🏢 `clients` | `data/all-contacts/clients/` | `data/client-profiles/profiles.parquet` | website scrape, Drive, Fathom, Deep Research | yes |
| 🎯 `prospects` | `data/all-contacts/prospects/` | `data/client-profiles/prospect_profiles.parquet` | website scrape, Fathom, Deep Research | no — a prospect has no shared-drive folder |

- **Separate prefixes and separate profile blobs, not one store with a `pool` column.** Every consumer enumerates the store it loads, so a shared store would make filtering something each reader has to remember, and the reader that forgets silently mixes prospects into client work — the same trap documented for `data/all-topics/awards/`. Instead `pool` is a keyword argument defaulting to `'clients'` on `aspect_profile.load_profiles/save_profiles/upsert_profiles`, `client_delete.delete_clients` and the job configs, so **every pre-pool call site keeps its exact previous behaviour** and a new call site has to name the pool it means. `load_profiles` stamps a `pool` column on the frame from the blob it read (derived, never stored — `save_profiles` drops it), which is what makes "both pools" a legitimate concat in the read-only views.
- **The registry is `src/modules/pools.py`**, and it is Streamlit-free because `client-profile-job`, `contact-import-job` and `fathom-sync-job` all import it. It owns the prefixes, labels and per-pool capabilities; `aspect_profile` owns the profile blob names (`ap.profiles_blob(pool)`) and `pools` imports *it*, not the other way round — the reverse would be circular.
- **One column convention.** `clients/` parquets use `company_name` / `summary`; the Contact Importer's lead parquets use `companyName` / `company_summary`. A prospect import is written in the **clients** convention by `contact_import_job`, and `pools.normalize_company_columns()` fixes anything that is not at load time. It **renames rather than copies**, deliberately: a frame carrying both spellings would let an edit to `summary` leave a stale `company_summary` behind, and `matching_job` reads whichever it finds.
- **The UI is a selector, not a second set of pages.** Company Records, Deep Research and Capability Profiles each render `ui_common.pool_selector()` at the top and everything below follows it — material read, profile store written, job config, delete section. The selector's `clears=` names the session keys holding the previous pool's data and drops them the run the selection changes, so one pool's companies can never appear under the other's heading. The read-only views (Aspect Match, Grant Search, HubSpot Import) use `ui_common.pool_scope_selector()`, which adds a **Both** option and returns a list — a run across both pools is a real question, and every result row carries its `pool`. The write-side views never offer Both: an edit, a build or a delete must land in exactly one store.
- **Fathom is the deliberate exception and has no selector.** A meeting is attributed by external invitee domain, which may belong to a client or a prospect, and the sweep costs one paginated pass either way — so both the view and `fathom-sync-job` always resolve **both** pools into one `{blob: frame}` dict. Blob names are unique per prefix, so the merged dict still writes each row back to the file it came from and a prospect's digest lands in the prospect parquet. Companies are labelled with their pool icon in the picker.
- **Promotion, not re-import.** `pool_transfer.move_companies()` moves a signed prospect into the client pool: contact rows to `clients/promoted_{date}_{hex6}.parquet`, then the profile row from `prospect_profiles.parquet` into `profiles.parquet`. **The destination is written first** — if the source rewrite then fails, the company is in both pools (visible, fixable) rather than in neither, which is why there is no pre-archive here as there is in `client_delete.py`. A key the destination already holds is reported as a conflict and skipped, never merged. Exposed in Company Records when the Prospects pool is selected, and available to any signed-in user: it is an everyday workflow action, not a destructive one.
- **Bulk Matching lists `prospects/` but leaves it unchecked.** It sits under `data/all-contacts/`, so the source enumeration finds it automatically and would have pre-checked it like every other source (`value=True`) — quietly changing what every existing bulk run means. Prospects get their per-capability treatment in Aspect Match; tick the box to include them here as well.
- **Deletion is per pool.** `client_delete.delete_clients(..., pool=...)` reads that pool's contacts prefix and profile store, stamps `_deleted_pool` on the archived rows, and skips the Drive Sync assignment step entirely for pools where `supports_drive` is false.

---

## Streamlit Views

### Parent pages and the `render()` dispatch pattern

Four nav entries are **parent pages** that hold no UI of their own: `grant_sources.py`,
`client_sync.py`, `resumes.py` (and `home.py`, which is just links). A parent renders a
sidebar selector and then calls `render()` on **exactly one** sub-view module per script run.

The eight sub-views (`topic_importer`, `sam_gov_upload`, `grants_gov_fetch`,
`funding_sources`, `drive_sync`, `fathom_sync`, `resume_importer`, `resume_search`) keep
their module-level imports, constants and helpers where they were, but everything from
`st.title(...)` down lives inside `def render():`. They are imported, never registered as
`st.Page` targets.

Two rules follow, and both are load-bearing:

- **Never use `st.tabs` to combine sub-views.** `st.tabs` renders every tab body in the same
  script run, and these views contain **96 `st.stop()` calls** between them — all of them the
  "abort the rest of the page" idiom rather than a defensive stop. `st.stop()` raises
  `StopException`, which unwinds the *whole* script run, so the first tab to hit one would
  silently blank every tab after it, with no error anywhere. The single-branch selector is
  what keeps those 96 calls correct. (`sam_gov_upload.py` does use `st.tabs` internally, but
  safely: both tab bodies finish rendering before its `st.stop()`s are reached, so they only
  guard the CSV pipeline *below* the tab widget. The hazard is specifically an `st.stop()`
  *inside* a tab body, which would prevent every later tab from rendering at all.)
- **Session-state init must sit inside `render()`, not at module level.** Module-level code
  runs once per *process*; a second browser session would import nothing and never initialise
  its keys. Every wrapped view has its `for _k in (...)` block inside `render()` for this reason.

Sub-view session/widget keys are namespaced per view (`ti_`, `sam_`, `ggov_`, `fsrc_`, `ds_`,
`fs_`, `ri_`, `rs_`) because merging puts them in one namespace — `topic_importer`'s formerly
bare `topics_df`/`save_results` and `sam_gov_upload`'s bare `daily_*`/`rc_*` keys were
prefixed when they were merged. Any new widget in a sub-view needs a prefixed `key=`.


| File | Title | Purpose |
|------|-------|---------|
| `views/home.py` | Home | Landing page — the pipeline map (grants in → clients in → match → export) with an `st.page_link` into every page, plus a collapsed "At a glance" panel (client/profile/agency counts) that only reads GCS when its Refresh button is pressed |
| `views/grant_sources.py` | Grant Sources | **Parent page.** Sidebar selector dispatching to exactly one of Import Topics / SAM.gov / Grants.gov / Funding Sources per script run |
| `views/client_sync.py` | Client Sync | **Parent page.** Sidebar selector dispatching to Google Drive or Fathom Meetings |
| `views/resumes.py` | Resumes | **Parent page.** Sidebar selector dispatching to Import or Search |
| `views/contact_importer.py` | Import Contacts | Pick a **destination** (📁 lead source folder, or the 🎯 prospect pool) → upload any lead spreadsheet **or** pull a HubSpot company list (lists search → memberships → batch company read) → map columns → dedup preview vs GCS (per-source or all-sources scope) → choose profiling method (🌐 scrape+GPT summary or 🔬 Deep Research technology focus, with model picker + cost estimate + >$50 confirmation) → stage file + trigger `contact-import-job` Cloud Run Job → poll `contact-import-jobs/{run_id}/status.json` |
| `views/client_editor.py` | Company Records | Pick a pool (🏢 Clients / 🎯 Prospects) → select a company from that pool → edit its `summary` → re-embed (float64, matching stored dtype) → apply to all contact rows of that company → rewrite the source parquet in place. On the Prospects pool an **⬆️ Promote to client** section moves the selected companies (and their capability profiles) into the client pool — available to everyone, deliberately placed above the admin gate. **Admins only:** a "🗑 Delete" section (multiselect + type-DELETE confirmation) removes companies that are no longer clients — every contact row, their aspect profile, and their Drive Sync assignment |
| `views/finance_researcher.py` | Deep Research | Pick a pool → select its companies → pick research focus (💰 Financials / 🔬 Technology & R&D) → launch background Deep Research tasks (one per company) → poll `finance-research-runs/` or `tech-research-runs/` `{run_id}/state.json` → review parsed results → apply `financial_data`/`financial_summary`/`financials_updated_at` or `technology_data`/`technology_summary`/`technology_updated_at` back onto the rows of **the pool the run was launched for** (recorded in `state.json` as `pool`, so a run resumed by ID in another session cannot write into the wrong pool) |
| `views/topic_importer.py` | Import Topics *(mode of Grant Sources)* | Upload PDF or paste text → Claude extracts topics → editable table → embed + save to `processed/` |
| `views/grant_search.py` | Grant Search | Select agencies, apply keyword filters, then search **one of two ways**. 📝 *Technology description* embeds a paragraph and ranks topics against it (unchanged). 🧬 *Capability profile* runs a **single-company aspect match**: pick a stored profile from the client store, the prospect store or both (searchable over name / website / market + aspect labels, each labelled with its pool icon) **or** paste notes / source material and have Claude build a capability profile on the spot — same prompt, aspects, markets and embeddings as the Capability Profiles job, optionally saved into **one** chosen pool's store (Prospects by default: a company profiled from pasted notes is usually one we hold no records for). Scope it to the whole company or to chosen markets (confirmed and unexplored), then score, re-rank and download — all through the shared `src/modules/aspect_matching.py`, so a pair ranks identically here and in Bulk Aspect Match. An opt-in **"Include past awards"** checkbox (default off, shown only when an awards store exists) adds `data/all-topics/awards/*` to the pool for "who has won work like this?" questions; every row carries a `record_kind` column (`solicitation` / `award`) promoted to the front of the results table so an awarded contract is never read as something to bid on |
| `views/bulk_matching.py` | Bulk Matching | Select contact sources (every folder pre-checked **except** `prospects/` — see Stage 11) + grant agencies, configure threshold/top-k/AI validation, trigger Cloud Run job, poll status |
| `views/sam_gov_upload.py` | SAM.gov *(mode of Grant Sources)* | Upload SAM.gov CSVs (processed in Streamlit: map columns → dedup vs existing store by notice ID + title **before** screening → Claude screening → summarize → embed → save) **or** configure an API fetch that triggers the `sam-gov-job` Cloud Run Job. Both the manual fetch and the daily schedule expose an **"Also fetch past awards"** checkbox (default off) writing `include_awards` into the config. Also exposes a **Daily API Parameters** section (daily 5 AM CST schedule config saved to `sam-gov-configs/daily_schedule.json`) and a **Revision Check** expander that triggers the `revision_check` job mode (dry-run by default) to sweep stored open notices for SAM.gov amendments. Streamlit polls `sam-gov-jobs/{run_id}/status.json` for manual run completion and renders revision/archived tables from the status payload. |
| `views/grants_gov_fetch.py` | Grants.gov *(mode of Grant Sources)* | Query the Grants.gov public `search2` API (keyword, date range, status, funding instrument, agency — no API key) → Claude Haiku relevance screening → embed → save to `data/all-topics/processed/GRANTS-GOV/` |
| `views/funding_sources.py` | Funding Sources *(mode of Grant Sources)* | The master list of watched funding-source websites (Stage 10) and the control panel for the agent that walks them. **1 · Source list** — `st.data_editor` over `deep-research-configs/sources.parquet` (cadence, per-site navigation instructions, destination agency folder, page cap, enable/pause), filters (Due now / Never checked / Paused / Failing / Has API / Needs login), plus a one-time spreadsheet import with a routing preview. **2 · Run** — pick Due / All / specific sites, set the time budget (1–24 h), concurrency and dry-run, with a live per-site cost estimate → write `deep-research-configs/{run_id}.json` → trigger `deep-research-job` → poll `deep-research-jobs/{run_id}/status.json`; a "Resume monitoring" expander re-attaches by run ID, and a "Daily schedule" expander saves `daily_schedule.json` and prints the one-time Cloud Scheduler commands. **3 · Flags** — sites found to expose an API (with evidence), login-walled sites, and sites auto-paused after repeated failures |
| `views/hubspot_import.py` | HubSpot Import | Three source modes (the **Client profiles** mode takes a Clients / Prospects / Both store selector, and every exported row carries a `pool` column so the two stay distinguishable in the CRM): **Matching run** (concatenate segment CSVs → standard `matcher_*` properties), **Financial research run** (load `finance-research-runs/{run_id}/state.json` → per-field mapping table, **pre-filled by auto-assignment** (existing `matcher_fin_*` → name/label match → opt-in standard-property alias → close match, all type-checked): each financial field → existing HubSpot property or auto-created `matcher_fin_*`, with a `<field>_num` integer companion per money/count/score field created as a HubSpot `number` property for lead scoring), or **Client profiles** (load `data/client-profiles/profiles.parquet` → pick clients → flattened aspect + market + unexplored-market fields → existing property or auto-created `matcher_profile_*`/`matcher_aspect_*`/`matcher_market_*`/`matcher_defense_*`/`matcher_unexplored_*`). All submit as company imports via `/crm/v3/imports` (dedup by `domain`) and poll for completion |
| `views/resume_importer.py` | Import *(mode of Resumes)* | Upload HubSpot contacts CSV with resume URL column → dedup by email → fetch files (PDF/DOCX) via HubSpot Bearer auth → extract text → GPT expertise summary → embed → save to `data/resumes/` |
| `views/resume_search.py` | Search *(mode of Resumes)* | Natural-language query → embed → cosine similarity against resume parquets → ranked candidate cards + CSV export. Supports an optional include keyword (single term) and comma-separated exclude keywords to pre-filter the resume pool before scoring. |
| `views/fathom_sync.py` | Fathom Meetings *(mode of Client Sync)* | Test the Fathom connection (verifies which recorders the key can see) → metadata-only scan of a date window → auto-match external invitee domains to **clients and prospects alike** by `companyWebsite` (no pool selector — one sweep covers both, and each company is labelled with its pool icon), everything else to a review `st.data_editor` (`fathom-configs/assignments.json`) → pick clients (All / None / Never synced / Has calls) + look-back + time budget + dry run/full re-sync + per-client caps → trigger `fathom-sync-job` → poll `fathom-jobs/{run_id}/status.json` → results (updated/unchanged/errored, calls ingested, unmatched domains) + a browser over `data/fathom/meetings_index.parquet` that loads any stored transcript |
| `views/drive_sync.py` | Google Drive *(mode of Client Sync)* | Scan the client Google shared drive (sections → `{Client}_INTERNAL` folders) → fuzzy auto-assign folders to clients with persistent assignments (`drive-sync-configs/assignments.json`) + review table → select the exact clients to sync (last-synced dates + All/None/Never/Stale quick-picks) and the exact unassigned folders to propose as new clients, set the time budget (1–24 h) and per-client doc caps → trigger `drive-sync-job` (incremental via `sync_state.json`) → poll `drive-sync-jobs/{run_id}/status.json` → results + new-client review queue (approve with website → rows created in `data/all-contacts/clients/`) |
| `views/client_profiler.py` | Capability Profiles | Pool selector, then a directory of that pool's companies with the source material available per company (website / Drive / meetings / technology / financials), their markets, their unexplored markets, and profile status (none / current / ⚠️ stale by `source_fingerprint` / ⚠️ no markets) → pick clients + sources + target aspect count + max markets + Defense assessment + unexplored assessment + max unexplored + model (Advanced: market and aspect merge thresholds) → trigger `client-profile-job` → poll `client-profile-jobs/{run_id}/status.json` (one or two Claude calls per client → embed each aspect, market and unexplored market → upsert `data/client-profiles/profiles.parquet`); an expander resumes monitoring by run ID, and a "build note(s)" expander lists aspect merges and any failed unexplored pass. Second section reviews/edits a profile's aspects, **markets** (tier as a `NumberColumn` rank) **and unexplored markets** (three `st.data_editor`s) and re-embeds all three vector blocks in-process, or deletes it (delete is admin-only); a market listed in both the served and unexplored tables is rejected by name. **Admins only:** a third section bulk-deletes profiles, with an opt-in checkbox to delete the clients' contact rows too |
| `views/aspect_match.py` | Aspect Match | Pick the profile store(s) — Clients / Prospects / Both — then select profiled companies + grant agencies + filters → pick a match mode (whole company, or by market filtered on three axes: **kind** (Confirmed / Unexplored / Both, default Confirmed) × **tier ranks** × category) → per-unit matmul of that unit's vectors against topic vectors — a confirmed unit holds its earmarked aspect vectors plus the market narrative, an **unexplored unit holds its narrative alone** (with `min_hits` forced to 1) → keep topics clearing the threshold on ≥ `min_hits` vectors, top-K per unit → async Claude re-rank 1–5 using **one of two prompts** chosen per row from `market_kind` ("could they propose today" for confirmed, "could they plausibly extend into this" for unexplored, the latter shown the gap `rationale` and the linked aspects as evidence), deduped by (client, agency, topic, aspect, **kind**) → results table carrying `market_kind` **and `pool`** + CSV download + `aspect-match-results/{run_id}/results.csv`. Runs in-process (keep the page open) |
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
| `finance_research.py` | `FIELD_SECTIONS`, `build_research_prompt`, `parse_research_output`, `build_financial_digest`, `response_cost_usd`, `NUMERIC_FIELDS`/`NUMERIC_SUFFIX`, `to_number`/`to_number_str`/`numeric_columns` | Deep Research helpers for the Client Research view (financial focus + shared plumbing) — 54-field output schema, prompt builder, JSON extract + `gpt-4o-mini` repair (`parse_research_output` takes an optional `fields=` list so it can normalize either focus's schema), headline digest (no AI call), and per-response cost from `usage`. `to_number()` pulls one figure out of a prose research value (range → midpoint, `$4.2M`/`250K` scale suffixes, `(Estimated)` labels and `%` stripped, `48/100` → `48`, nullish → `None`; a bare `k`/`m`/`b` only counts as a scale suffix at a word boundary, so "5 board members" stays 5; a fiscal-period label is skipped when a real figure follows it, so `"FY2024: $2.5M"` → `2500000` and `"Q1 2025 revenue of $3M"` → `3000000`, while a bare `"2024"` with nothing else in the string is still returned). `to_number_str()` renders it as a round-half-up integer string (`''` when unparseable — deliberately blank, not 0, so HubSpot reads "no data"), and `numeric_columns()` builds the `<field>_num` set consumed by HubSpot Import. Model IDs (`gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`) drift — verify against developers.openai.com/api/docs/models on API errors. |
| `fathom_client.py` | `API_BASE`, `INCLUDABLE`, `GENERIC_DOMAINS`, `fathom_get`, `iter_meetings`, `get_transcript`, `bare_domain`, `external_domains`, `meeting_title`/`meeting_date`/`duration_minutes`/`summary_markdown`/`action_item_lines`/`attendee_lines`/`crm_company_names`, `transcript_text`, `meeting_block`, `call_count`, `FathomError`/`FathomAuthError`/`RateLimitStalledError` | Fathom notetaker REST client (Stage 9). Streamlit-free, shared by the Fathom Meetings view and `fathom_sync_job.py`. Plain `requests`, not the official `fathom-python` SDK (still 0.0.30 with documented breaking changes). `fathom_get` owns a **process-global pacing gate** — one shared timestamp at 1.05 s (standard) / 2.50 s (heavy: any summary or transcript payload), because a heavy call spends standard quota too — plus `Retry-After`-aware backoff, `FathomAuthError` on 401/403 (never retried), and `RateLimitStalledError` when 429s outlast the retries. `iter_meetings` follows `next_cursor`; `transcript_text` caps by dropping the **middle**, since meetings front-load what the company builds and back-load what it will do next. |
| `aspect_matching.py` | `Unit`, `TOPIC_COLS`, `ROW_CONFIRMED`/`ROW_UNEXPLORED`, `RERANK_MODELS`/`CONCURRENCY`, `stack_topic_embeddings`, `unit_markets`/`market_counts`/`tier_counts`/`plan_units`, `match_units`, `RERANK_SYSTEM`/`RERANK_UNEXPLORED_SYSTEM`, `rerank_user_message`/`parse_rerank`/`rerank_async`/`rerank_groups`/`run_rerank`, `display_frame` | The Stage 8 scoring and re-rank core, extracted from the Bulk Aspect Match view so the Grant Search view's single-company match is the **same** implementation rather than a second one. Holds the three things that are easy to get subtly wrong: `min_hits` counts only *aspect* vectors (the market narrative counts only for a market no aspect was earmarked to, and an unexplored unit is forced to 1); confirmed and unexplored units are scored by **different prompts**; re-rank calls are deduped by (client, agency, topic, aspect, kind) and never across kinds. Streamlit-free — progress is reported through plain callables (`progress(fraction, text)` for `match_units`, `progress(done, total)` for `run_rerank`), so the caller owns the widget. |
| `aspect_profile.py` | `SOURCES`, `NOTES_SOURCE_KEY`/`NOTES_CAP`/`notes_source_texts`/`source_label`, `MATERIAL_COLS`, `merge_company_row`, `assemble_source_texts`, `stated_intentions`, `source_fingerprint`, `build_aspect_system`, `build_aspect_user_message`, `parse_aspect_response`, `aspect_embed_text`, `MAX_ASPECTS`/`ASPECT_MERGE_THRESHOLD`, `merge_similar_aspects`, `MARKET_CATEGORIES`/`DEFENSE_MARKET`/`MAX_MARKETS`/`MAX_MARKET_TIER`/`MARKET_MERGE_THRESHOLD`, `market_tier_rank`/`tier_ordinal`, `canonical_market`, `normalize_markets`, `merge_similar_markets`, `market_embed_text`, `market_label`, `profile_markets`, `market_aspect_indices`, `unpack_market_embeddings`, `MAX_UNEXPLORED`/`MIN_LINKED_ASPECTS`, `build_unexplored_system`/`build_unexplored_user_message`/`parse_unexplored_response`, `normalize_unexplored`, `profile_unexplored`, `unpack_unexplored_embeddings`, `unexplored_aspect_indices`, `pack_embeddings`/`unpack_embeddings`, `build_profile_record`, `load_profiles`/`save_profiles`/`upsert_profiles`/`delete_profile`, `company_key` | Multi-aspect client profiles (Stage 8). Streamlit-free (shared by the two views **and `client_profile_job.py`**): merges a company's contact rows into one material row, pulls source material off it per source, fingerprints it for staleness, builds the aspect-generation prompt, parses/normalizes Claude's JSON (unknown `kind` → `capability`, duplicate labels dropped, ≤ `MAX_ASPECTS`), and owns the `data/client-profiles/profiles.parquet` store. **Caller-supplied material:** `build_aspect_user_message` also emits any source key that is not one of `SOURCES` — today that is `notes`, the text pasted into Grant Search, which no extractor could ever pull off a contact row. It is capped, fingerprinted and embedded like any other source, so a profile built from notes is a normal profile and `sources_used` says honestly where it came from. **Markets:** `canonical_market()` snaps a free-form name onto `MARKET_CATEGORIES` (exact → punctuation-squashed → `difflib` ≥ 0.85 → leading-word containment → `Other`); `normalize_markets()` collapses repeats of one category, caps the non-defense count, resolves membership in both directions (the model may state it on the aspect, the market, or both), puts orphan aspects in the **lowest-ranked** market, dense-renumbers the surviving ranks 1…N, and re-derives each market's `aspect_labels` — **the aspects own membership**, so editing or deleting an aspect row can never leave a stale pointer. A market **no aspect claims is kept as long as it has a narrative** (it is then scored on that narrative vector alone) — dropping it would silently delete a Defense market, and with it the only DoD framing in the profile, whenever the model's `aspects` list did not string-match an aspect label; only a market with neither aspects nor a narrative is discarded; `merge_similar_markets()` folds markets whose narrative vectors score ≥ the threshold into one. **Tiers:** `market_tier_rank()` is the single coercion point for every historical `tier` shape (int, `"2"`, `"2nd"`, `'primary'`→1, `'secondary'`→2, anything else last) and `tier_ordinal()` renders it — **never read `tier` directly**. **Aspects:** `merge_similar_aspects()` folds near-identical aspects at `ASPECT_MERGE_THRESHOLD` (0.96, higher than the market threshold on purpose), the longer text surviving and absorbing the other's keywords + market membership, and returns the merge list so the job can report it. **Unexplored markets (pass 2):** `stated_intentions()` pulls the client's own `notable_updates` off `client_meetings_data`/`client_docs_data` — the one place aspirational statements are useful, and the reason pass 1's prompt now explicitly excludes them; `normalize_unexplored()` drops `Other`, anything already served, and anything without a narrative, and filters `aspect_labels` to real aspects. `MIN_LINKED_ASPECTS` is asked for in the prompt but **not enforced** in code — a market whose labels merely failed to string-match would otherwise be deleted along with its narrative, the same trap `normalize_markets` documents for Defense. **Aspect, market and unexplored vectors are stored flat** (`aspect_embeddings` = `n_aspects × embedding_dim`, `market_embeddings` = `n_markets × embedding_dim`, `unexplored_embeddings` = `n_unexplored × embedding_dim`, float64 in one list column each) — a flat double list round-trips through parquet without nested-list dtype ambiguity; always read them back via `unpack_embeddings()` / `unpack_market_embeddings()` / `unpack_unexplored_embeddings()`. |
| `tech_research.py` | `FIELD_SECTIONS`, `ALL_FIELDS`, `build_research_prompt`, `build_tech_digest`, `build_matching_summary` | Technology & R&D research schema/prompt/digest for the Client Research view's tech focus — ~40-field output (core technology, products, R&D activity, patents, TRL/maturity, differentiation, grant-alignment keywords). `build_matching_summary()` assembles embedding-ready text (confidence labels stripped) for the optional summary-rewrite at apply time. Reuses finance_research's models, pricing, and JSON parse/repair. |
| `access_control.py` | `SUPER_ADMINS`, `current_user_email`, `is_admin`, `is_super_admin`, `role_label`, `require_admin`, `admin_only_notice`, `load_admins`/`save_admins`/`admin_emails` | Admin gating for the delete actions and the Admin Portal. Identity is the IAP email `app.py` puts in `st.session_state.user_email`; the admin list lives in `admin-config/admins.json` (cached per session — a newly added admin must reload the page). `SUPER_ADMINS` is a code constant: not editable from the UI, never stored in the JSON, and the only role allowed to change the list. A GCS read failure grants nothing beyond the super admins. Streamlit-only. |
| `source_registry.py` | `BUCKET`/`SOURCES_BLOB`/`SEEN_PREFIX`, `COLUMNS`/`COLUMN_DEFAULTS`, `CADENCES`/`CADENCE_DAYS`/`MAX_CONSECUTIVE_FAILURES`, `normalize_url`/`url_key`/`host_of`, `infer_broad_agency`, `blank_row`/`ensure_columns`, `load_sources`/`save_sources`/`upsert_sources`/`delete_sources`, `is_due`/`due_sources`/`days_since_checked`, `import_rows`/`merge_import`, `seen_key`/`load_seen`/`save_seen`/`known_titles` | The Stage 10 master list. Streamlit-free and client-injected (every function takes a `storage.Client`), so the view passes a service-account client and the job passes an ADC one. `upsert_sources` re-reads from GCS before merging — a long run must not clobber an edit made in the view while it was running. `infer_broad_agency` only ever *suggests* a destination folder; the stored column is what the job routes on. `ensure_columns` runs on both read and write, so a parquet written by an older version of the module keeps loading. |
| `browser_agent.py` | `TOOLS`, `MODEL`, `DEFAULT_MAX_TOOL_CALLS`/`DEFAULT_MAX_PAGES`/`DEFAULT_SITE_TIMEOUT_S`, `PageSession`, `research_site`, `launch_browser` | Claude driving headless Chromium over one site (Stage 10). `research_site()` **never raises** — a failure comes back as `ok: False` with an error string, because one bad site must not abort a 280-site run. Budgets are enforced by forcing the terminal `report_findings` tool rather than by aborting, so partial findings survive. `launch_browser` passes `--no-sandbox --disable-dev-shm-usage` (required under Cloud Run; `lead_importer._playwright_scrape` omits both and must not be reused in a job). Page text is capped by dropping the **middle**, since listings front-load opportunities and detail pages back-load deadlines. |
| `pools.py` | `CLIENTS`/`PROSPECTS`/`DEFAULT`, `POOLS`/`POOL_KEYS`, `pool`/`is_pool`/`label`/`noun`/`display`/`icon`, `contacts_prefix`/`profiles_blob`/`supports_drive`/`pool_of_prefix`, `COLUMN_ALIASES`/`normalize_company_columns`, `company_key`/`key_mask`, `load_frames`/`load_all_frames`/`company_names`/`combined_frame` | The clients/prospects registry (Stage 11). Streamlit-free — `client-profile-job`, `contact-import-job` and `fathom-sync-job` import it. Imports `aspect_profile` for the profile blob names; `aspect_profile` must never import this, or the two are circular. `normalize_company_columns` **renames** the lead-import spellings (`companyName`, `company_summary`) onto the clients convention rather than copying them, so no frame ever carries two spellings of one value. `load_frames` keeps one frame per blob because every write path rewrites the exact file a row came from. |
| `pool_transfer.py` | `move_companies`, `format_report` | Promoting a prospect to a client (Stage 11) — contact rows then profile row, **destination written before the source is touched**, so a mid-way failure duplicates rather than deletes. A company key the destination already holds is a reported conflict, never a merge. Streamlit-free, like `client_delete.py`. |
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

Rows may additionally carry Deep Research columns — financial focus (Client Research view, `data/all-contacts/clients/` only): `financial_data` (JSON string of the full 54-field Deep Research output), `financial_summary` (human-readable digest), `financials_updated_at` (ISO date); technology focus (Client Research view on clients, or any source imported via the Contact Importer's `deep_research` profiling method): `technology_data` (JSON string of the ~40-field tech research output), `technology_summary` (digest), `technology_updated_at` (ISO date). For deep-research imports, `company_summary` holds `tech_research.build_matching_summary()` output (not a scraped-page GPT summary) and `embeddings` is its vector. Client rows updated by Drive Sync additionally carry `client_docs_data` (JSON: extracted fields + source_files + last_run), `client_docs_summary` (plain-text digest), and `docs_updated_at` (ISO date). Client rows updated by `fathom-sync-job` (Stage 9) carry `client_meetings_data` (JSON: `extracted` fields + `meetings` provenance list + `last_run`), `client_meetings_summary` (plain-text digest), and `meetings_updated_at` (ISO date) — that job never modifies `summary` or `embeddings`.

### Multi-aspect capability profile (parquet, `data/client-profiles/profiles.parquet` · `prospect_profiles.parquet`)
The same schema in two blobs, one per pool (Stage 11) — `aspect_profile.profiles_blob(pool)` resolves which. A `pool` column is stamped on read and dropped on write, so it is never stored. One row per company — written by `client-profile-job` (and by single-profile edits in the Client Profiles view), read by Bulk Aspect Match and by HubSpot Import's **Client profiles** mode. Never written to the client contact parquets.

| Field | Type | Notes |
|-------|------|-------|
| `company_key` | str | **Join key** — `{company_name}\|\|{companyWebsite}`, the same identity used by Client Editor / Client Research / Drive Sync (`aspect_profile.company_key()`) |
| `company_name` / `companyWebsite` | str | Copied from the client rows at build time |
| `profile_summary` | str | 2–4 sentence company summary. Context for the LLM re-ranker — **not embedded** |
| `aspects` | str | JSON array of `{label, kind, text, keywords, evidence, markets}`; `kind` ∈ technology/capability/product/domain/market; `markets` is the list of canonical market names this aspect serves (**the source of truth for membership**). Read with `profile_aspects()` |
| `aspect_labels` | str | `' \| '`-joined labels, for display without parsing the JSON |
| `n_aspects` / `embedding_dim` | int | Shape of the packed vectors (dim is 1536 for `text-embedding-ada-002`) |
| `aspect_embeddings` | list[float] | **Flat** `n_aspects × embedding_dim` float64 vectors of each aspect's `label + text + keywords`. Read with `unpack_embeddings()` — never index this directly |
| `markets` | str | JSON array of `{market, tier, subtitle, narrative, keywords, aspect_labels, merged_from}` — `market` ∈ `MARKET_CATEGORIES`, `tier` an **integer rank** (1 = most core; legacy rows carry `'primary'`/`'secondary'` strings, coerced on read by `market_tier_rank()`), `aspect_labels` derived from the aspects, `merged_from` naming any market folded into this one. Empty for profiles built before markets existed. Read with `profile_markets()` |
| `market_labels` | str | `' \| '`-joined `Market (1st)` display labels. A **stored display string** — rows written before ranks existed still read `(primary)`/`(secondary)` until rebuilt |
| `n_markets` | int | Number of confirmed markets; `0` flags the profile ⚠️ no markets in the Client Profiles directory and excludes it from market-scoped match runs |
| `market_embeddings` | list[float] | **Flat** `n_markets × embedding_dim` float64 vectors of each market's `market + subtitle + narrative + keywords`. Read with `unpack_market_embeddings()` |
| `unexplored_markets` | str | JSON array of `{market, tier, subtitle, narrative, keywords, rationale, aspect_labels}` — markets the client does **not** serve, inferred by linking its aspects (pass 2). `tier` ranks promise (1 = most promising), `rationale` states the gap that would remain and is shown to the re-ranker but **not embedded**, `aspect_labels` are the existing aspects it would draw on. Empty for profiles built before pass 2 existed, and legitimately empty for a client whose aspects support nothing new. Read with `profile_unexplored()` |
| `unexplored_labels` | str | `' \| '`-joined `Market (1st)` display labels for the unexplored block |
| `n_unexplored` | int | Number of unexplored markets. `0` is normal and does **not** flag the profile — it only means an Unexplored-kind match run will skip this client |
| `unexplored_embeddings` | list[float] | **Flat** `n_unexplored × embedding_dim` float64 vectors of each unexplored market's `market + subtitle + narrative + keywords` (same embed text as a confirmed market). The **only** thing an unexplored market is scored on. Read with `unpack_unexplored_embeddings()` |
| `dod_assessment` | str | One sentence on why the client has **no** Defense market; empty when a Defense market was created |
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

### SAM.gov award record (parquet, `data/all-topics/awards/SAM-GOV/`)
Completed contracts, kept as competitive intelligence — **never** open opportunities. Same base columns as a grant topic record (so `normalize_grant_columns()` and Grant Search render it unchanged) with `sam_status` fixed at `'awarded'` and `due_date` empty, plus:

| Field | Type | Notes |
|-------|------|-------|
| `award_date` / `award_number` | str | From the search response's `award` block — no extra API call |
| `award_amount` | str | Raw string as SAM.gov returns it (e.g. `"24500000.00"`) |
| `award_amount_num` | float \| None | Parsed for sorting/filtering. **`None`, never `0.0`, when unparseable** — same reasoning as the HubSpot `_num` companions: missing must read as "no data", not as a $0 award |
| `awardee_name` / `awardee_city` / `awardee_state` | str | The winning company and where it sits |
| `set_aside` | str | `typeOfSetAsideDescription` — the small-business/8(a)/SDVOSB signal |
| `notice_type` / `base_type` | str | `Award Notice`, and what it was awarded against (`Special Notice` for a CSO award, `Solicitation`, …) |
| `agency_path` | str | `fullParentPathName`, e.g. `DEPT OF DEFENSE.DEPT OF THE ARMY.AMC.ACC…` |
| `naics_code` | str | |

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
      awards/                       # past awards — deliberately NOT under processed/
        SAM-GOV/                    # (everything enumerating processed/ would match them)
          sam_awards_2026-09-16_a3f9c1.parquet
    all-contacts/
      clients/                          # the client pool (Stage 11) — company_name / summary convention
        drive_sync_2026-08-11_a3f9c1.parquet
      prospects/                        # the prospect pool — same convention, same treatment, no Drive
        prospects_2026-09-22_b2e4d7.parquet
        promoted_2026-09-22_c1d8a2.parquet   # written by pool_transfer when a prospect is promoted
      apollo/
        apollo_2026-03-01_a3f9c1.parquet
      sba/
        sba_2026-03-01_b2e4d7.parquet
      free_alert/
        free_alert_2026-03-01_c1d8a2.parquet
    resumes/
      resumes_2026-06-24_a3f9c1.parquet   # individual resume records, deduped by email
    fathom/
      meetings/
        123456789.json                # one raw Fathom call: metadata + invitees + AI summary + action items + full transcript turns
      meetings_index.parquet          # one row per ingested call (recording_id, client_key, title, date, duration, transcript lines, blob_path)
    client-profiles/
      profiles.parquet                  # multi-aspect CLIENT profiles, one row per company (overwritten in place on every build/edit/delete — not versioned)
      prospect_profiles.parquet         # the same schema for the prospect pool — a separate blob, not a `pool` column (see Stage 11)
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
  fathom-configs/                       # Fathom Meetings state + job configs
    assignments.json                  # domain → client_key assignments (+ unassigned/skipped review buckets, settings.lookback_days) — merged, never clobbered
    sync_state.json                   # meetings: recording_id → {client_key, created_at, synced_at} (incremental skip) + clients: client_key → last-synced date
    fathom_2026-09-09_10-30-00.json   # per-run job config
  fathom-jobs/                          # fathom-sync-job progress + completion status
    fathom_2026-09-09_10-30-00/
      status.json
  drive-sync-configs/                   # Drive Sync state + job configs
    assignments.json                  # drive_id + folder_id → client_key assignments (+ unassigned/skipped) — overwritten on save
    sync_state.json                   # files: file_id → modifiedTime last synced (incremental diffing) + proposed: folder_id → last-proposed date (proposal rotation cursor); advanced only by non-dry runs
    drive_sync_2026-08-11_15-30-00.json  # per-run job config
  drive-sync-jobs/                      # drive-sync-job completion status
    drive_sync_2026-08-11_15-30-00/
      status.json
  deep-research-configs/                # Funding Source Watch state + job configs
    sources.parquet                   # the master list — one row per site (overwritten in place, not versioned)
    daily_schedule.json               # persistent daily-run config (`run_id: "daily"` sentinel)
    seen/
      a3f9c1.json                     # per-source index of opportunities already extracted: key → {title, first_seen}
    deep_research_2026-09-16_06-00-00.json  # per-run job config
  deep-research-jobs/                   # deep-research-job progress + completion status
    deep_research_2026-09-16_06-00-00/
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
| `anthropic` | Claude (Haiku, Sonnet) — match verification, topic extraction, SAM.gov screening, email copy. **Pinned `>=1.3.0,<2`** — see SDK version pins below before changing it |
| `openai` | **Pinned `>=3.7.0,<4`** (see SDK version pins below). Embeddings (`text-embedding-ada-002`), GPT summarization, subject line generation, deep-research-style research (`gpt-5.6-sol`/`terra`/`luna` via Responses API background mode) for Client Research (financial + technology focuses) and the Contact Importer's `deep_research` profiling mode |
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

### SDK version pins (read this before editing any `requirements*.txt`)

Every requirements file — the app's and all five jobs' — pins **`anthropic>=1.3.0,<2`**
and **`openai>=3.7.0,<4`**. The upper bounds are deliberate and were added after an
unpinned major broke production silently.

**What happened (2026-09-02):** the files previously said `anthropic>=0.86.0`. A routine
image rebuild pulled **anthropic 1.3.0**, which **removed `temperature`, `top_p`, and
`top_k` from `messages.create()`**. Because the SDK drops the keyword rather than the API
rejecting it, the failure is a local `TypeError: AsyncMessages.create() got an unexpected
keyword argument 'temperature'` — not an HTTP 400. In Bulk Aspect Match every re-rank call
raised it, `TypeError` is not in the retryable list, so every pair was stored with
`llm_score = 0` and then filtered out by the minimum-score control (whose floor is 1). The
symptom was "no topics cleared" at *every* threshold and *every* minimum score, with no
error shown anywhere.

**Rules that follow from it:**

- **Never pass `temperature` / `top_p` / `top_k` to `client.messages.create()`** anywhere in
  this codebase. All six live Anthropic call sites were stripped (`views/aspect_match.py`,
  `jobs/matching_job.py`, and four in `src/modules/email_generator.py`). Anthropic is also
  removing these parameters model-by-model, so this is the direction of travel regardless.
- Parameters anthropic 1.3.0 **does** accept on `messages.create`: `model`, `messages`,
  `max_tokens`, `system`, `stop_sequences`, `tools`, `tool_choice`, `thinking`,
  `output_config`, `metadata`, `service_tier`, `container`, `cache_control`, `inference_geo`,
  `stream`, `timeout`, plus the `extra_*` escape hatches.
- **Before raising either upper bound, check the signatures rather than deploying and
  hoping** — this catches the exact failure class above for free, with no API calls:

  ```python
  import inspect
  from anthropic.resources.messages import Messages
  from openai.resources.responses import Responses
  from openai.resources.chat.completions import Completions
  print(sorted(inspect.signature(Messages.create).parameters))
  print(sorted(inspect.signature(Responses.create).parameters))   # needs background, tools, reasoning
  print(sorted(inspect.signature(Completions.create).parameters))
  ```

- Two **unimported** legacy modules still pass `temperature` and would raise immediately if
  revived: `src/modules/ai_analyzer.py` and `src/llm_functions.py`. Nothing in `views/`,
  `app.py`, or `jobs/` imports them, so they ship in no image.
- **Email copy runs at the model default now.** `email_generator.py` used `temperature=0.7`
  for body copy; measured against the real `DEFAULT_JOSIAH_SYSTEM` prompt, removing it left
  the opening, structure, and claims intact and only widened run-to-run phrasing variety
  slightly (pairwise word overlap across three samples 0.67 → 0.60; longest sample 44 → 51
  words). **Subject lines are unaffected** — they generate on `gpt-4o-mini`, and neither that
  call nor its Haiku fallback ever set `temperature`.
- `openai 3.7.0` was verified against this codebase's usage: `embeddings.create`,
  `chat.completions.create` (still accepts `temperature`), and `responses.create` with
  `background=True` + the `web_search` tool. The deep-research paths were signature-checked,
  not billed end-to-end — watch Client Research and the Contact Importer's `deep_research`
  mode on first use after any bump.

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
| `sam_gov_api_key` | SAM.gov Upload view (API fetch tab) | `st.secrets['sam_gov_api_key']` — passed into the sam-gov-job config JSON (**not** a Secret Manager secret; it therefore sits in plaintext inside the `sam-gov-configs/daily_schedule.json` GCS blob, unlike the Fathom key which is deliberately Secret Manager-only). **Two tiers exist and they are not interchangeable.** The project ran on the *individual* key (Account Details → API Key) until 2026-09-21; it dies after a few dozen requests/day. **It now runs on a non-federal *system account* key** (approved 2026-09-21), created under SAM.gov Workspace → System Accounts, which gets 1,000/day, requires the Non-Federal System Administrator role from your Entity Administrator, **expires every 90 days — next rotation due ~2026-12-20** (replacement auto-generated 15 days ahead, both valid during the overlap), and is bound to the IP allowlist documented under "SAM.gov static egress IP". **Rotation touches three places, and the second is the one that fails silently:** `streamlit-secrets`, the plaintext copy in the `daily_schedule.json` blob (the only one the daily Cloud Scheduler run reads), and a `matcher-app` redeploy so the UI remounts the secret. |
| `fathom_api_key` | Fathom Meetings view (connection test + metadata scan) | `st.secrets.get('fathom_api_key')` — created at fathom.video → Settings → API Keys. Must sit **above** `[gcp_service_account]` in secrets.toml. Keys are **per user, not per org**: a key sees only meetings its owner recorded plus meetings shared with their team. The view degrades gracefully without it (a sync can still be triggered — the job reads its own copy from Secret Manager). |
| `fathom-api-key` (Secret Manager) | Cloud Run fathom-sync-job | `os.environ['FATHOM_API_KEY']` — deliberately Secret Manager rather than the job config JSON, so the token never lands in a GCS blob (unlike the SAM.gov key) |
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
- **`.gcloudignore` / `.dockerignore`** exclude `.streamlit/`, `*.json` (service-account keys), `notebooks/`, and `*_api_key.txt` / `*_api_key` from the build context — never remove those entries. The key-file patterns matter specifically because `Dockerfile.app` ends in `COPY . .`: an API key dropped at the repo root that is not excluded gets **baked into the published `matcher-app` image** in Artifact Registry, which no amount of not-committing prevents. The same three patterns are in `.gitignore`. Verify an exclusion with `gcloud meta list-files-for-upload | grep -i <name>` before building.
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
| `fathom-sync-job` | `jobs/fathom_sync_job.py` | `jobs/Dockerfile.fathom_sync` | `requirements.fathom_sync_job.txt` | Fathom Meetings view |
| `client-profile-job` | `jobs/client_profile_job.py` | `jobs/Dockerfile.client_profile` | `requirements.client_profile_job.txt` | Client Profiles view |
| `deep-research-job` | `jobs/deep_research_job.py` | `jobs/Dockerfile.deep_research` | `requirements.deep_research_job.txt` | Funding Sources view + Cloud Scheduler (daily) |

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

`cloudbuild.yaml` is checked into the repo root (recreate it only if it goes missing —
it was absent until 2026-09-08, which is why older notes here built it inline with `printf`).

```bash
# Step 1 — build image and push to Artifact Registry
gcloud builds submit --config cloudbuild.yaml --project cc-matcher-v1 .

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
    "include_awards":  false,
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
    "include_awards":  false,
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
  "rows_deferred_quota":   0,
  "fetch_reports":         [{"label": "solicitations", "total": 2254, "retrieved": 2254,
                             "pages": 3, "truncated": false, "capped": false}],
  "api_calls_used":        24,
  "awards_fetched":        40,
  "awards_passed_screening": 9,
  "awards_after_dedup":    7,
  "awards_saved":          7,
  "awards_deferred":       0,
  "awards_gcs_path":       "data/all-topics/awards/SAM-GOV/sam_awards_2026-09-16_a3f9c1.parquet",
  "error":                 null
}
```

`fetch_reports` carries one entry per query stream. **`truncated: true` means SAM.gov returned fewer records than it said matched** — check it before trusting a run's coverage; the view renders it as a red banner. `api_calls_used` is the run's total SAM.gov HTTP request count, which is the number to watch against the daily quota. Award counts are written even when the solicitation stream exits early, because the awards phase runs first for exactly that reason.

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

### SAM.gov static egress IP (Cloud NAT) — `sam-gov-job` only

> ✅ **APPROVED AND LIVE (2026-09-21).** The non-federal system account was approved and its key is in production. **Both copies were rotated on 2026-09-21** — `streamlit-secrets` (version 5) and the plaintext `sam_gov_api_key` inside the `sam-gov-configs/daily_schedule.json` GCS blob — and `matcher-app` was redeployed (revision `matcher-app-00042-844`) so the UI remounts the new secret. Verified end to end by run `sam_gov_2026-09-21_14-19-54`: 174/174 records retrieved, `truncated: false`, `rows_deferred_quota: 0`, `error: null`.
>
> **The NAT is load-bearing from now on.** The individual key this replaced was not IP-restricted, so the static egress below changed nothing operationally until today. A system-account key **is** bound to the allowlist, so releasing either address now breaks every SAM.gov call from the job.
>
> **Rotating the key (repeat this before ~2026-12-20 — system-account keys expire every 90 days; the replacement is auto-generated 15 days ahead and both are valid during the overlap):**
> 1. Retrieve the key from SAM.gov **Workspace → System Accounts** (not Account Details — that is the individual key).
> 2. Put it in `.streamlit/secrets.toml`, then push it to **both** places: `gcloud secrets versions add streamlit-secrets --data-file=.streamlit/secrets.toml`, **and** the plaintext copy inside the `daily_schedule.json` blob. **Missing the second is a silent failure** — the daily Cloud Scheduler run reads only the blob, never Secret Manager.
> 3. Redeploy/restart `matcher-app`, or the UI keeps serving the old secret from its mounted volume until the next revision.
> 4. Confirm the next run's `api_calls_used` in `sam-gov-jobs/{run_id}/status.json` is not truncated by quota and `rows_deferred_quota` is 0.
>
> **A 200 from a laptop proves nothing about the allowlist.** General search requests from an unlisted address were observed succeeding on 2026-09-21; only a run through the NAT exercises the allowlisted path. Verify with the job, not with curl.

SAM.gov **system accounts enforce an IP allowlist**: the request form states *"All system-to-system requests must come from an IP address listed here. Addresses may include Classless Inter-Domain Routing (CIDR) addresses."* Cloud Run by default "connects to external endpoints on the internet using a **dynamic IP address pool**" ([Google docs](https://cloud.google.com/run/docs/configuring/static-outbound-ip)), so there was no address to put on the form until this was built (2026-09-16).

**The addresses submitted to SAM.gov are `34.60.178.41` and `35.223.63.211`.** Both were listed on the system account request of 2026-09-16, **approved 2026-09-21** and now serving live traffic; a redundant pair means NAT can fail over and more gateways can be added later without re-editing the account. **Never delete or re-create these addresses** — releasing one silently breaks every SAM.gov call from the job until the account's IP list is edited and re-reviewed by GSA.

| Resource | Name | Notes |
|----------|------|-------|
| VPC | `default` | Auto-created when Compute API was first enabled on the project |
| Subnet | `default` (`10.128.0.0/20`, us-central1) | |
| Static IPs | `sam-egress-ip-1`, `sam-egress-ip-2` | The two addresses above. `MANUAL_ONLY` allocation, so egress can come from **only** these |
| Cloud Router | `matcher-router` | us-central1, on `default` |
| Cloud NAT | `matcher-nat` | `--nat-all-subnet-ip-ranges`, pool = both static IPs |
| Job egress | `sam-gov-job` | `--network default --subnet default --vpc-egress all-traffic` |

**Only `sam-gov-job` is routed through the NAT.** `jobs/sam_gov_job.py` is the only file in the codebase that touches `api.sam.gov` — the Streamlit view just writes a config blob and triggers the job — so the other five jobs and `matcher-app` keep the default dynamic egress and are unaffected.

**`all-traffic` is required, not optional.** `private-ranges-only` would leave public destinations (SAM.gov included) on the default dynamic pool, which is the exact thing the allowlist rejects. The side effect is that the job's GCS, Anthropic and OpenAI traffic also exits via the NAT and incurs NAT data-processing charges; at this job's volume (~30 MB/day of parquet) that is cents per month. Verified working after the change — a throwaway Cloud Run job on the same network settings reached `storage.googleapis.com`, `api.anthropic.com`, `api.openai.com` and `api.sam.gov`, listed real GCS blobs with the `matching-job@` service account, and reported `EGRESS_IP=34.60.178.41`.

**Cost:** Cloud NAT is billed per gateway-hour plus data processed, and the reserved IPs are billed whether or not traffic flows — roughly **$35/month standing**, independent of the job running for two minutes a day. This is the first always-on infrastructure in an otherwise fully serverless project. The cheaper alternative considered and rejected was an `e2-micro` egress proxy (~$4/month, since `_sam_get()` is the single chokepoint for every SAM.gov request), rejected because it adds a VM to patch and a new silent-failure mode for an unattended daily job.

#### Recreating it (one-time setup, already done — reference only)

```bash
gcloud services enable compute.googleapis.com --project cc-matcher-v1

gcloud compute addresses create sam-egress-ip-1 --region us-central1 --project cc-matcher-v1
gcloud compute addresses create sam-egress-ip-2 --region us-central1 --project cc-matcher-v1

gcloud compute routers create matcher-router \
  --network default --region us-central1 --project cc-matcher-v1

gcloud compute routers nats create matcher-nat \
  --router matcher-router --region us-central1 --project cc-matcher-v1 \
  --nat-external-ip-pool sam-egress-ip-1,sam-egress-ip-2 \
  --nat-all-subnet-ip-ranges

gcloud run jobs update sam-gov-job --region us-central1 --project cc-matcher-v1 \
  --network default --subnet default --vpc-egress all-traffic
```

#### Verify the egress IP after any network change

```bash
gcloud compute addresses list --regions us-central1 --project cc-matcher-v1
gcloud run jobs describe sam-gov-job --region us-central1 --project cc-matcher-v1 | grep -A3 "VPC access"
```

To re-confirm the address the job actually leaves from, create a throwaway job on the **same** `--network/--subnet/--vpc-egress` flags running
`python -c "import requests; print(requests.get('https://api.ipify.org').text)"`
off the `sam-gov-job` image, then delete it. Do not use the real job for this — it would spend SAM.gov quota.

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
  "dedup_all_sources": false,
  "pool":           null
}
```

`col_map` values are actual column names from the uploaded file; `null` = unmapped optional field. `file_ext` is `.csv`, `.xlsx`, or `.xls`. `profile_method` is `"scrape"` (default) or `"deep_research"`; `research_model` (deep_research only) is one of the `DEEP_RESEARCH_MODELS`. `dedup_all_sources: true` makes the job's runtime dedup compare against all of `data/all-contacts/` instead of only the source's folder (set automatically by the UI's dedup-scope checkbox; default ON for HubSpot-list pulls). `pool` (Stage 11) is `null` for an ordinary lead import, or `"prospects"` to write straight into the prospect pool: the output parquet lands in `data/all-contacts/prospects/`, is written in the **clients column convention** (`company_name` / `summary`, via `pools.normalize_company_columns`), and is deduplicated against the **prospect and client pools** rather than the source folder — a company we already work for must not re-enter as a prospect, while a company sitting in an old apollo lead list still can.

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

### `fathom-sync-job` — build, deploy, and manage

Ingests Fathom meeting transcripts and notes (Stage 9). No staging upload — the
config just names the clients; everything else comes from the Fathom API.

#### `fathom-sync-job` config schema (`fathom-configs/{run_id}.json`)

```json
{
  "run_id":                  "fathom_2026-09-09_10-30-00",
  "client_keys":             ["Acme Robotics||https://acme.com"],
  "lookback_days":           90,
  "created_after":           null,
  "full_resync":             false,
  "dry_run":                 false,
  "max_meetings":            400,
  "max_meetings_per_client": 12,
  "per_client_char_cap":     120000,
  "transcript_char_cap":     40000,
  "task_timeout_s":          14400,
  "model":                   "claude-sonnet-4-6"
}
```

`client_keys` are `{company_name}||{companyWebsite}` identities — the same
`client_key` Drive Sync and Client Profiles use; an empty list means every
client with matched meetings. `created_after` (ISO-8601) overrides
`lookback_days` when set. `max_meetings` caps how many **new attributed** calls
one sweep will collect; `max_meetings_per_client` / `per_client_char_cap` cap
what one Claude call sees, and the overflow is deferred rather than marked
synced. `full_resync` re-ingests every call in the window; `dry_run` performs
the whole pipeline (Fathom calls and the Claude merge included) but writes
nothing except `status.json`.

#### Status schema (`fathom-jobs/{run_id}/status.json`)

```json
{
  "run_id": "...", "state": "running|complete|error", "dry_run": false,
  "stopped_early": null, "note": "", "created_after": "2026-06-11T00:00:00Z",
  "meetings_swept": 412, "meetings_new": 37, "meetings_ingested": 33,
  "meetings_deferred": 4,
  "clients_total": 12, "clients_done": 12,
  "clients_updated": 9, "clients_unchanged": 2, "clients_errored": 1,
  "api_calls_used": 51,
  "results": [{"client_key": "...", "company_name": "...",
               "outcome": "updated|unchanged|error|deferred",
               "meetings_processed": 3, "note": ""}],
  "unmatched_domains": [{"domain": "unknown-co.io", "meeting_count": 4,
                          "sample_titles": ["..."], "crm_companies": ["..."]}],
  "model": "claude-sonnet-4-6",
  "dry_run_preview": [{"company_name": "...", "meetings": ["..."], "digest": "- bullet
- bullet",
                        "extracted_counts": {"technologies": 12, "capabilities": 11},
                        "notable_updates": ["plans to..."]}],
  "error": null
}
```

**`dry_run_preview` exists because a dry run writes nothing** — without it the digest the run just paid Claude to produce is unobservable, and "the pipeline works" and "the extraction is any good" would need two separate runs to answer. Populated on dry runs only, for the first `_MAX_DRY_PREVIEWS` (3) clients, digest truncated to 4,000 chars; the view renders it under the results panel.

`state: "running"` payloads are written at start and every 10 completed clients
so the view can render a progress bar. `stopped_early` is `"timeout"` (time
budget), `"rate_limit"` (429s outlasted the retries) or `"max_meetings"` (sweep
cap) — in every case re-running the same selection continues where it left off,
because only successfully written calls are recorded in `sync_state.json`.
`unmatched_domains` is also merged into `fathom-configs/assignments.json`, so
the view's review table sees domains the job discovered.

#### One-time setup (run once — skip if job already exists)

```bash
# The Fathom key goes in Secret Manager, NOT the job config JSON
gcloud secrets create fathom-api-key --data-file=- --project cc-matcher-v1   # paste the key, then Ctrl-D
# matching-job@ already holds project-level roles/secretmanager.secretAccessor

gcloud run jobs create fathom-sync-job \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/fathom-sync-job:latest \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --task-timeout 86400 \
  --max-retries 0 \
  --service-account matching-job@cc-matcher-v1.iam.gserviceaccount.com \
  --set-secrets=ANTHROPIC_API_KEY=anthropic-api-key:latest \
  --set-secrets=FATHOM_API_KEY=fathom-api-key:latest \
  --project cc-matcher-v1

# Streamlit's service account needs run.admin for runWithOverrides
gcloud run jobs add-iam-policy-binding fathom-sync-job \
  --region us-central1 \
  --member="serviceAccount:matcher-app@cc-matcher-v1.iam.gserviceaccount.com" \
  --role="roles/run.admin" \
  --project cc-matcher-v1
```

To rotate the key later: `gcloud secrets versions add fathom-api-key --data-file=- --project cc-matcher-v1`
(the job resolves `:latest` at each execution, so no redeploy is needed), and
update `fathom_api_key` in the `streamlit-secrets` secret for the view.

#### Build and deploy (run every time `fathom_sync_job.py` or `fathom_client.py` changes)

```bash
gcloud builds submit --config cloudbuild.fathom_sync.yaml --project cc-matcher-v1 .

gcloud run jobs update fathom-sync-job \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/fathom-sync-job:latest \
  --task-timeout 86400 \
  --region us-central1 --project cc-matcher-v1
```

#### View logs

```bash
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=fathom-sync-job" \
  --limit 50 --format "value(textPayload)"
```

---

### `client-profile-job` — build, deploy, and manage

Builds the multi-aspect client profiles of Stage 8. Everything the job needs is already on the client rows, so there is no staging upload — the config just names the companies.

#### `client-profile-job` config schema (`client-profile-configs/{run_id}.json`)

```json
{
  "run_id":         "client_profile_2026-08-19_10-30-00",
  "pool":           "clients",
  "company_keys":   ["Acme Robotics||https://acme.com"],
  "sources":        ["website", "drive", "technology"],
  "target_aspects": 4,
  "max_markets":    4,
  "assess_defense": true,
  "assess_unexplored": true,
  "max_unexplored": 3,
  "market_merge_threshold": 0.93,
  "aspect_merge_threshold": 0.96,
  "model":          "claude-sonnet-4-6",
  "concurrency":    4,
  "dry_run":        false
}
```

`pool` (Stage 11) selects which contacts prefix is read and which profile store is written — `clients` (the default, and what a config written before pools existed means) or `prospects`; an unknown value is a hard error rather than a silent fallback, since writing a whole run into the wrong store is unrecoverable without a re-run. `company_keys` are `{company_name}||{companyWebsite}` identities (`aspect_profile.company_key()`). `sources` is any subset of `website`, `drive`, `technology`, `financials`; invalid keys are dropped and an empty result is a hard error. `concurrency` is clamped to 1–8 (each unit is one or two Claude calls plus its aspect, market and unexplored embeddings). `max_markets` (1–`MAX_MARKETS`) caps the non-defense markets; `assess_defense` toggles the Defense assessment in the prompt; `market_merge_threshold` is the cosine above which two of a company's market narratives are folded into one, and `aspect_merge_threshold` (default 0.96) the cosine above which two of its aspects are. `assess_unexplored` turns on the **second Claude call** and `max_unexplored` (1–`MAX_UNEXPLORED`) caps what it may return — enabling it roughly **doubles per-client latency**, so the graceful time budget below is spent twice as fast. `dry_run` runs the Claude calls and embeddings but never writes `profiles.parquet`.

#### Status schema (`client-profile-jobs/{run_id}/status.json`)

```json
{
  "run_id": "...", "state": "running|complete|error", "dry_run": false,
  "model": "claude-sonnet-4-6", "sources": ["website", "drive"], "target_aspects": 4,
  "max_markets": 4, "assess_defense": true,
  "assess_unexplored": true, "max_unexplored": 3,
  "clients_total": 40, "clients_done": 17,
  "built": [{"company_key": "...", "company_name": "...", "n_aspects": 4,
             "sources_used": "website,drive", "aspect_labels": "A | B | C | D",
             "n_markets": 3, "market_labels": "Defense (1st) | Energy & Power (3rd)",
             "has_defense": true,
             "n_unexplored": 2, "unexplored_labels": "Maritime (1st) | Agriculture & Food (2nd)",
             "aspect_merges": ["Edge inference -> Onboard compute"]}],
  "errors": ["Acme Robotics: invalid response twice: ..."],
  "warnings": ["Acme Robotics: merged near-identical aspects Edge inference -> Onboard compute",
               "Inaedis Inc.: claude-sonnet-4-6 declined to answer (refusal) - the profile was built with claude-haiku-4-5-20251001 instead",
               "Beta Systems: unexplored-market pass failed (...) - the profile was saved without unexplored markets"],
  "deferred": ["Beta Systems"],
  "stopped_early": null, "profiles_blob": "data/client-profiles/profiles.parquet",
  "error": null
}
```

A `running` payload is written at start and every 5 completed clients; the view renders `clients_done / clients_total` as a progress bar and resumes monitoring by run ID.

**A refusal falls back to the other supported model.** Some entirely legitimate clients trip a false-positive refusal: measured on Inaedis Inc. (aerosolised thermostable vaccine powders, DoD CBD SBIR on an MVA smallpox/mpox vaccine), `claude-sonnet-4-6` returned `stop_reason='refusal'` with **zero content blocks** on the website text and on the Drive text *independently*, while `claude-haiku-4-5-20251001` built the profile normally (8 aspects, 3 markets). A refusal is deterministic for the same model and material, so the plain strict-JSON retry can only burn a second call — `_claude_json` therefore retries a refusal on the other entry in `ap.ASPECT_MODELS`, and only on a refusal (bad JSON twice is a prompt/material problem another model would not fix). The model that actually answered is stored as the profile's `model` and the swap is reported in `warnings`, so a profile never claims a model that declined it.

**Two bugs this path had, both of which hid their own cause — do not reintroduce either.**
- `resp.content[0].text` raises **IndexError** on an empty content list, and `IndexError` is not the `ValueError` the retry loop catches, so a refusal escaped on the first attempt and was reported as `list index out of range`. `_response_text()` now joins every text block and raises a `ValueError` carrying `stop_reason` — which is the only field that explains a refusal.
- The job's pool config variable is named **`pool_key`, not `pool`**, because `with ThreadPoolExecutor(...) as pool` is the house idiom in 16 places in this repo and one of them is further down the same function. `_status()` and `_save_records()` are defined above it and called below it, so a local named `pool` silently rebound the name for both: the run died on `json.dumps` of a `ThreadPoolExecutor`, and every save in it had resolved `ap.profiles_blob(<executor>)` — which fell through to the **client** store. `ap.profiles_blob()` now raises on a non-string pool for exactly this reason; an unknown *string* still falls back, since that is a stale UI selection rather than a shadowed variable.

`warnings` is deliberately separate from `errors`: an aspect merge and a failed pass 2 both leave a **saved, usable profile**, so they must not read as a build failure. The view shows them in a collapsed "build note(s)" expander. Reporting merges there is the only way to tell whether `aspect_merge_threshold` is collapsing genuinely distinct capabilities — check them before lowering it.

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


### `deep-research-job` — build, deploy, and manage

Walks the Stage 10 master list. No staging upload — the config just says which sites to check; everything else comes from `deep-research-configs/sources.parquet`.

#### `deep-research-job` config schema (`deep-research-configs/{run_id}.json`)

```json
{
  "run_id":         "deep_research_2026-09-16_06-00-00",
  "select":         "due",
  "source_ids":     [],
  "model":          "claude-sonnet-4-6",
  "concurrency":    3,
  "max_tool_calls": 25,
  "site_timeout_s": 240,
  "max_pages":      null,
  "task_timeout_s": 14400,
  "dry_run":        false
}
```

`select` is `"due"` (cadence-driven — daily always, weekly after 6 days, monthly after 27), `"all"` (every enabled site), or `"ids"` (exactly `source_ids`, which runs even if a site is paused — an explicit pick is an operator override). `concurrency` is clamped 1–6; each concurrent site holds its own Chromium context at roughly 200 MB. `max_pages` overrides every row's own cap when set. `task_timeout_s` is clamped 900–86 400 and the job stops handing out sites 10 minutes before it, reporting the rest as `deferred`. `dry_run` browses, extracts and reports a preview but writes no parquets and marks no site checked.

The daily Cloud Scheduler config lives at `deep-research-configs/daily_schedule.json` with `"run_id": "daily"` as a sentinel the job replaces with a timestamped ID at runtime — same pattern as `sam-gov-daily`, so no day's status file overwrites another's.

#### Status schema (`deep-research-jobs/{run_id}/status.json`)

```json
{
  "run_id": "...", "state": "running|complete|error", "dry_run": false,
  "select": "due", "model": "claude-sonnet-4-6",
  "sites_total": 142, "sites_done": 142, "sites_ok": 131,
  "sites_errored": 9, "sites_deferred": 2,
  "opportunities_found": 88, "opportunities_new": 41, "opportunities_saved": 41,
  "gcs_paths": ["data/all-topics/processed/CONSORTIUM/deep_research_2026-09-16_a3f9c1.parquet"],
  "api_sites":    [{"source_id": "a3f9c1", "name": "nstxl.org", "url": "...", "evidence": "..."}],
  "login_walled": [{"source_id": "9662fa", "name": "marketplace.gocolosseum.org", "url": "..."}],
  "results": [{"source_id": "...", "name": "...", "status": "ok|no_opportunities|needs_login|error|deferred",
               "found": 3, "new": 2, "note": ""}],
  "dry_run_preview": [{"name": "...", "url": "...", "new": 2, "opportunities": [{"title": "...", "description": "..."}]}],
  "cost_usd": 12.84, "stopped_early": null, "error": null
}
```

A `running` payload is written at start and at every 10-site checkpoint. `results` is capped at 400 rows. `dry_run_preview` is populated on dry runs only, for the first few sites — a dry run writes nothing, so without it the extraction the run just paid for would be unobservable (the `fathom-sync-job` precedent).

#### One-time setup (run once — skip if the job already exists)

```bash
gcloud run jobs create deep-research-job \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/deep-research-job:latest \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --task-timeout 14400 \
  --max-retries 0 \
  --service-account matching-job@cc-matcher-v1.iam.gserviceaccount.com \
  --set-secrets=ANTHROPIC_API_KEY=anthropic-api-key:latest \
  --set-secrets=OPENAI_API_KEY=openai-api-key:latest \
  --project cc-matcher-v1

# Streamlit's service account needs run.admin for runWithOverrides
gcloud run jobs add-iam-policy-binding deep-research-job \
  --region us-central1 \
  --member="serviceAccount:matcher-app@cc-matcher-v1.iam.gserviceaccount.com" \
  --role="roles/run.admin" \
  --project cc-matcher-v1

# Cloud Scheduler's OAuth SA needs it too — the body carries containerOverrides
gcloud run jobs add-iam-policy-binding deep-research-job \
  --region us-central1 \
  --member="serviceAccount:matching-job@cc-matcher-v1.iam.gserviceaccount.com" \
  --role="roles/run.admin" \
  --project cc-matcher-v1

gcloud scheduler jobs create http deep-research-daily \
  --schedule="0 6 * * *" \
  --uri="https://run.googleapis.com/v2/projects/cc-matcher-v1/locations/us-central1/jobs/deep-research-job:run" \
  --message-body='{"overrides":{"containerOverrides":[{"args":["deep-research-configs/daily_schedule.json"]}]}}' \
  --oauth-service-account-email=matching-job@cc-matcher-v1.iam.gserviceaccount.com \
  --location=us-central1 \
  --time-zone="America/Chicago"
```

#### Build and deploy (run every time `deep_research_job.py`, `source_registry.py`, or `browser_agent.py` changes)

```bash
gcloud builds submit --config cloudbuild.deep_research.yaml --project cc-matcher-v1 .

gcloud run jobs update deep-research-job \
  --image us-central1-docker.pkg.dev/cc-matcher-v1/matcher/deep-research-job:latest \
  --task-timeout 14400 \
  --region us-central1 --project cc-matcher-v1
```

The Chromium install adds several minutes to the build, as it does for contact-import-job.

#### View logs

```bash
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=deep-research-job" \
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
