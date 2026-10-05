# Due Diligence Intake Service — Implementation Plan

**Audience:** Claude Code, working in the Matcher repo (GCP project `cc-matcher-v1`).
**Owner:** John (BW&CO).

## 0. Rules for Claude Code (read first)

1. Read `CLAUDE.md`, then the existing storage layer (`BucketManager`, `get_storage_client()`), the existing prospect-profile code and directory, and any existing HubSpot / Google Drive client code **before writing anything**.
2. Complete **Phase 0 (Discovery)** and report findings to John before starting Phase 1. Do not guess the prospect-profile format.
3. GCS via `BucketManager` is the canonical storage layer. No local filesystem patterns (`glob`, `open()` on local paths) in service code.
4. Reuse existing clients and conventions. If an existing pattern doesn't fit, stop and ask rather than introducing a parallel one.
5. Ask targeted questions when a decision in Section 11 (Open Decisions) blocks you. Don't silently pick.
6. When this work is done, add the durable architectural rules from this plan to `CLAUDE.md`.

---

## 1. Goal

Replace the HubSpot due diligence form with a custom, public intake flow:

1. The founder enters company basics and **optionally uploads** documents (pitch deck, one-pager, technical lead CV).
2. An LLM **pre-fills the descriptive fields** from the uploads. Eligibility fields are never AI-filled.
3. Federal databases are queried to **suggest** SAM/UEI status and prior awards, which the founder confirms.
4. The founder reviews, edits, confirms each section, and submits.
5. On submit:
   - write a prospect profile into the Matcher's prospect profiles directory (GCS)
   - create (or reuse) a `{Company_Name}_INTERNAL` folder in the BW&CO shared drive
   - copy uploaded documents into that folder
   - create a Google Doc of the form responses in that folder
   - upsert a HubSpot **company + contact** and associate them

## 2. Architecture

```
Founder browser (plain HTML/JS wizard)
        │  HTTPS
        ▼
intake-web  (FastAPI, Cloud Run, PUBLIC, separate from Matcher UI and Agent Hub)
        │
        ├── GCS (via BucketManager)
        │     ├── intake/sessions/{session_id}.json     draft state + AI draft
        │     ├── intake/uploads/{session_id}/...       raw uploads (signed-URL PUT)
        │     └── <prospect profiles dir>/...           final profile (Matcher reads this)
        ├── Anthropic API                               extraction (no tools)
        ├── SAM.gov Entity API, USAspending, SBIR.gov   enrichment suggestions
        ├── Google Drive / Docs API (service account)   _INTERNAL folder + Google Doc
        └── HubSpot CRM API (private app token)         company + contact upsert
```

The Matcher Streamlit app is unchanged except that new profiles appear in its prospect profiles directory.

### Repo layout (proposed; adjust to existing conventions found in Phase 0)

```
intake/                      # shared, importable package (no web framework deps)
  schema.py                  # single source of truth for the form
  schema_export.py           # schema -> frontend JSON, -> LLM JSON schema, -> mappings
  extraction.py              # document text extraction + Claude structured extraction
  enrichment.py              # SAM entity / USAspending / SBIR.gov lookups
  sessions.py                # session state read/write in GCS
  profile_writer.py          # final answers -> Matcher prospect profile
  drive_folder.py            # find-or-create {Company_Name}_INTERNAL, copy uploads
  gdoc_writer.py             # responses -> Google Doc in the folder
  hubspot_sync.py            # company/contact upsert + association
  submit_pipeline.py         # idempotent, resumable orchestration of submit steps
services/intake_web/
  main.py                    # FastAPI app, routes only, thin
  static/                    # index.html, app.js, styles.css
  Dockerfile
tests/intake/
evals/intake/                # extraction eval harness (Section 9)
```

Keep `intake/` free of FastAPI imports so the Matcher (or a Cloud Run job) can reuse it.

## 3. Form schema (single source of truth)

Define every question once in `intake/schema.py` (Pydantic models or a typed dict list). The frontend, LLM extraction schema, prospect profile mapping, Google Doc layout, and HubSpot mapping are all **generated from this file**. Editing a question should mean editing one place.

Per-field attributes:

| Attribute | Purpose |
|---|---|
| `id` | stable key (snake_case), never reused |
| `section` | `1_company`, `2_technology`, `2a_nih`, `3_evidence`, `4_team`, `5_funding`, `5_company_funding`, `5a_federal`, `6_commercialization`, `7_uploads` |
| `label`, `help_text` | display text, copied verbatim from the current form |
| `type` | `text`, `textarea`, `single_select`, `multi_select`, `email`, `phone`, `url`, `file` |
| `options` | for selects; exact option strings from the current form |
| `required` | bool |
| `condition` | visibility rule, e.g. section 2A shown only if primary or secondary sector ∈ {Healthtech, Medtech, Biotech}; investor-type questions shown only if funding includes "Outside investors" |
| `ai_fillable` | bool (see below) |
| `enrichable` | bool, for fields suggested by federal lookups |
| `profile_key` | key in the Matcher prospect profile (set in Phase 0) |
| `hubspot_property` | HubSpot property internal name, or null |

### `ai_fillable` defaults (to be confirmed by the eval in Section 9)

**Never AI-fillable (founder must answer):** company legal name, entity type, state of incorporation, contact name/title/email/phone, employee count incl. affiliates, US citizen/PR ownership ≥51%, VC/PE/hedge fund >50%, PI employed >50% (2A), human/animal studies (2A), openness to university/lab partnering, openness to CRO/subcontractor, target funding amount, earliest start date, ability to reach milestone with current funding, openness to adjacent projects, federal traction level, conversations with PMs/COs.

**Enrichable (suggested from lookups, founder confirms):** SAM.gov registration/UEI status, prior SBIR/STTR and federal awards.

**AI-fillable candidates:** website, technology description, primary/secondary sectors, stage, dual-use potential, visible failure/cost, urgency/window, competitors and differentiation, patient population (2A), unmet need vs improvement (2A), evidence checkboxes, publication/press links, technical lead name/title/degree, in-house capabilities, non-dilutive funding goals, identified solicitation, current funding sources, investor types, primary investors, next milestone and timing, adjacent markets, who pays, regulatory pathway, short-term business goal.

The sector list must match the Matcher's internal sector taxonomy. If the database taxonomy differs from the 13 website sectors, **use the database version** (per BW&CO's instruction). Find it in Phase 0.

## 4. User flow and API

### Flow

1. **Start:** legal company name, website, contact email, Turnstile CAPTCHA → session created.
2. **Optional upload:** files go directly to GCS via signed URLs. A "Skip" option is always available.
3. **Background jobs start** (if uploads exist): extraction; enrichment runs regardless, keyed on company name + website domain.
4. **Section 1 eligibility** is answered by the founder while jobs run.
5. **Review sections 2–6:** AI-filled fields are prefilled, badged "Suggested from your documents", show source (file + page), and stay editable. Enrichment results appear as confirm prompts ("We found UEI XXXX in SAM.gov, active. Is this your company?"). Each section requires a "Looks right" confirmation before advancing.
6. **Submit** → confirmation screen.

The form must be fully usable with **no uploads and no AI**. Extraction failure, timeout, or empty results fall back silently to a blank form.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/schema` | frontend form definition (generated) |
| POST | `/api/sessions` | create session; verifies Turnstile; returns `session_id` |
| POST | `/api/sessions/{id}/upload-urls` | returns V4 signed PUT URLs (content-type + size restricted) |
| POST | `/api/sessions/{id}/extract` | start extraction + enrichment |
| GET | `/api/sessions/{id}/extract` | poll status/results |
| PUT | `/api/sessions/{id}/answers` | autosave partial answers |
| POST | `/api/sessions/{id}/submit` | validate + run submit pipeline |

Session IDs are unguessable (≥128-bit random). Sessions expire after 14 days.

Extraction can run as a FastAPI background task for the MVP. Note that Cloud Run may throttle CPU after the response returns: set `--no-cpu-throttling` (CPU always allocated) on the service, or move extraction to a Cloud Run job / Cloud Tasks later.

## 5. Extraction

1. **Text extraction:** PDF via pypdf/pdfplumber with page numbers preserved; PPTX via python-pptx (slide = page); DOCX via python-docx. Alternatively, send PDFs directly to Claude as document blocks (preserves page context and handles image-heavy decks better). Decide after checking eval results on 3–4 real decks.
2. **Limits:** max 3 files, 25 MB each, PDF/PPTX/DOCX only, max ~60 pages total sent to the model. Reject other types at signed-URL issuance and re-check on read.
3. **Model call:** one Claude call with a JSON schema generated from `ai_fillable` fields only. Model name comes from config, not hard-coded. **No tools enabled.** Treat document content as untrusted data.
4. **Per-field output:**
   ```json
   {"value": ..., "source_file": "deck.pdf", "source_page": 4, "evidence": "<short excerpt, <25 words>", "confidence": "high|medium|low"}
   ```
   `value` must be `null` unless the documents state it explicitly. The prompt must say: do not infer, do not estimate, prefer null. Select values must match schema options exactly; validate and null out anything that doesn't.
5. **Post-validation:** drop low-confidence values for select fields; verify `evidence` actually appears in the extracted text for that page (cheap hallucination check, reuse Agent Hub's audit pattern if applicable).
6. Store the full AI draft in the session JSON. It's retained internally for draft-vs-final comparison (Section 7, step 7), not placed in Drive.

## 6. Enrichment

- **SAM.gov:** the Matcher currently uses the *Get Opportunities* API. UEI/registration status requires the **Entity Management API** (different endpoint, API key with entity access). Confirm access in Phase 0. Search by legal name and refine by website domain or state.
- **Prior awards:** USAspending recipient/award search by UEI if found, otherwise by name. SBIR.gov awards API for SBIR/STTR history. SBIR.gov's API has had reliability issues, so treat it as best-effort with short timeouts.
- Results are **suggestions only**. Never auto-submit an enriched value without founder confirmation. Ambiguous matches (multiple entities) are shown as a pick-list with "None of these."

## 7. Submit pipeline (idempotent and resumable)

`submit_pipeline.py` runs these steps in order, recording each step's status and output IDs in the session JSON so a retry resumes where it stopped:

1. **Validate** answers against the schema (required fields, option values, conditions).
2. **Write prospect profile** to the Matcher prospect profiles directory in the existing format (Phase 0). This is the canonical record. Once it succeeds, the founder sees the success screen; remaining steps continue and retry server-side.
3. **Find or create** `{Company_Name}_INTERNAL` in the configured shared drive (Section 8).
4. **Copy uploads** from GCS into the folder (original filenames, prefixed with date if collisions occur).
5. **Create Google Doc** "Due Diligence Responses — {Company_Name} — {YYYY-MM-DD}" in the folder, laid out by section using schema labels, with AI-suggested-and-accepted fields annotated. Simplest approach: render HTML and upload via Drive API with target mimeType `application/vnd.google-apps.document`.
6. **HubSpot upsert** (Section 8).
7. **Store draft-vs-final record** (AI draft, enrichment suggestions, final answers, per-field accepted/edited/rejected) in GCS at `intake/submissions/{session_id}.json`. Internal only.
8. Write Drive folder ID/URL and HubSpot IDs back into the prospect profile; mark the session `complete`.

On any step failure after step 2: log with session ID, retry with backoff (max 3), then alert (email or Slack, whichever exists). Never surface internal errors to the founder.

## 8. Integration details

### Google Drive

- The service account must be a **member of the shared drive** (Content Manager). John grants this; it cannot be done from code.
- Every Drive call uses `supportsAllDrives=True` (and `includeItemsFromAllDrives=True` / `corpora="drive"` + `driveId` on lists).
- Parent location: configurable `INTAKE_SHARED_DRIVE_ID` and optional `INTAKE_PARENT_FOLDER_ID`.
- **Folder naming:** `{Company_Name}_INTERNAL` using the founder's legal company name. Strip characters problematic for Drive and downstream scripts (`/ \ : * ? " < > |`), collapse whitespace, trim. Exact spaces-vs-underscores rule: see Section 11.
- **Dedup:** before creating, search the parent for an existing `*_INTERNAL` folder matching the normalized name (case-insensitive, punctuation and suffixes like Inc/LLC/Corp ignored). If the prospect profile already stores a `drive_folder_id` for this company (matched by website domain), use that. On ambiguity, create a new folder and flag the profile for manual review rather than writing into the wrong client's folder.

### HubSpot

- Private app token with scopes for contacts and companies (read + write).
- **Company:** search by website domain first; create if not found. Update only intake-owned properties on existing companies. Don't overwrite fields BW&CO staff maintain.
- **Contact:** upsert by email; associate to the company.
- **Custom properties** to create (John or Claude Code via API, after confirming names): `dd_submitted_at`, `dd_drive_folder_url`, `dd_matcher_profile_id`, `dd_intake_source` (`custom_intake_v1`), plus mapped fields per `hubspot_property` in the schema. Map only what staff actually use in HubSpot; the full record lives in the profile and Google Doc.
- **Trade-off to confirm:** creating records via the CRM API (not the Forms API) means no HubSpot form-submission event, so anything triggered by the old form (workflows, lists, attribution, notifications) **will not fire**. Inventory those in Phase 0. If any matter, either re-trigger them from a property change (e.g. `dd_submitted_at` is known) or submit through the Forms Submission API instead.

## 9. Extraction evaluation (gates which fields are AI-fillable)

- John supplies 10–20 past clients' decks/one-pagers plus their actual HubSpot form responses.
- `evals/intake/run_eval.py` runs extraction and scores per field: accuracy when filled, null rate, and **wrong-when-filled rate** (the metric that matters most).
- Any field with a meaningful wrong-when-filled rate gets `ai_fillable=False`, regardless of how useful it would be.
- Rerun after prompt or model changes. Store results in GCS or the repo (no client documents committed to git).

## 10. Security and data handling

- Turnstile on session creation; per-IP and per-session rate limits; max sessions per IP per day.
- Signed URLs: short expiry (15 min), content-type and `x-goog-content-length-range` enforced.
- Uploads bucket private; no public objects. CORS limited to the intake domain.
- Per-session cap on extraction calls (1 run + 1 retry) and pages, to bound API cost.
- Upload screen notice: what happens to files, that they're used to pre-fill the form, and "**Please do not upload export-controlled (ITAR/EAR) or CUI material.**"
- Raw uploads in `intake/uploads/` deleted by GCS lifecycle rule after 30 days (Drive copy is the retained version). Abandoned sessions deleted after 14 days.
- Secrets in Secret Manager, mounted as env vars. No keys in code or images.
- Log session IDs and step statuses, not answer contents or document text.

## 11. Open decisions (ask John when you reach them)

1. Folder naming: keep spaces from the legal name (`Acme Robotics_INTERNAL`) or convert to underscores (`Acme_Robotics_INTERNAL`)?
2. Which shared drive and parent folder ID?
3. Which HubSpot workflows/lists/notifications currently depend on the old form submission?
4. Who gets alerted on submit-pipeline failures, and how (email vs Slack)?
5. Public domain for the service (e.g. `intake.bwcoconsulting.com`)?
6. Should consultants get notified on each new submission, and where?

## 12. Phases

**Phase 0 — Discovery (report before coding).** Document: prospect-profile format and directory path; how the Matcher reads profiles; sector taxonomy source; existing HubSpot/Drive clients and auth; SAM Entity API access; old-form HubSpot dependencies.

**Phase 1 — Schema + manual form + submit pipeline (no AI).** `schema.py`, frontend wizard rendered from `/api/schema`, sessions, validation, full submit pipeline (profile, Drive folder, uploads copy, Google Doc, HubSpot). Deploy to Cloud Run. *This replaces the HubSpot form even if the AI parts never ship.*

**Phase 2 — Extraction + review UI.** Extraction module, eval harness, AI badges/sources, per-section confirmation, draft-vs-final record.

**Phase 3 — Enrichment.** SAM entity, USAspending, SBIR.gov suggestions with confirm UI.

**Phase 4 — Hardening and polish.** Magic-link save/resume, alerting, lifecycle rules, consultant notification, load and abuse testing.

## 13. Testing

- Unit: schema → frontend JSON / LLM schema / profile mapping; condition logic (2A visibility, investor questions); folder name normalization and dedup matching; option validation of extraction output.
- Integration (mocked external APIs): full submit pipeline, including resume after failure at each step and no duplicate folders/records on retry.
- Manual end-to-end against a test shared drive and a HubSpot sandbox or test records before go-live.
