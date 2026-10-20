# Plan: Signed-Contract Sync (HubSpot → Client Drive folders) inside The Matcher

## Context

`sync_signed_quotes.py` (in the repo root; move it into the Matcher's package as part of Phase 1) is a working standalone script. It pulls signed HubSpot quotes and their Terms & Conditions PDFs into Google Drive, one folder per company, under a single parent folder.

This plan moves the script into The Matcher (GCP project `cc-matcher-v1`). The Matcher already maps clients to their Drive folders in Firestore. Contracts should be filed into each client's real folder in the shared drive, not into a separate parent folder.

### Verified facts (checked against the live HubSpot portal 23128098, 2026-10-14)

| Item | Value |
|---|---|
| Signed filter | `hs_quote_esign_status = SIGNED` (366 quotes right now) |
| Signing date | `hs_esign_date` (ISO datetime) |
| Signed PDF | `hs_pdf_download_link`, a public URL. **Not yet verified to return the signed/audit-trail version. See Phase 0.** |
| T&C link | A Drive URL embedded in the `hs_terms` HTML, e.g. `https://drive.google.com/file/d/<id>/view` |
| Title encoding | Some `hs_title` values are double-encoded (`ProPlateÂ®`). The existing `fix_text()` handles this. |

### Target behavior

1. A matched client gets its files in `{client folder}/Contracts/`. "Client folder" means the `{company}_INTERNAL` folder the Matcher has on record. Create `Contracts` if it doesn't exist.
2. An unmatched client gets its files in `/uncategorized-contracts/` at the **top level of the clients shared drive**. Create that folder if it doesn't exist.
3. When a previously unmatched client later becomes matched, its files are **moved** from uncategorized into the client's `Contracts` folder, not duplicated.
4. Two files per signed quote: `{date} - {title} (Signed).pdf` and `{date} - {title} - Terms and Conditions.pdf`. In uncategorized, prefix the names with `[{HubSpot company name}]`.
5. The sync is idempotent across the whole shared drive, using Drive `appProperties` (`hubspot_quote_id`, `kind`).

---

## Phase 0: Discovery (do this first, then STOP and report)

Do not write implementation code until this phase is reported back and confirmed. Everything after it depends on what the Matcher actually stores. The Firestore assumption is unconfirmed.

1. **Find the client → folder mapping.** Search the Matcher codebase for Firestore collections that hold Drive folder or file IDs. Report:
   - the collection name(s) and document schema
   - the document key (HubSpot company ID? company name? domain? Drive folder ID?)
   - whether it stores the `{company}_INTERNAL` folder ID specifically, or a different level (the parent client folder, or the shared-with-client subfolder)
   - how and how often it is refreshed (the sync job's name and schedule)
2. **Determine the matching key.** Can a HubSpot **company ID** be resolved to a mapping document directly? If the mapping is keyed only by name, say so explicitly. That changes the matching strategy (see Guardrails).
3. **Check auth.** Identify the service account the Matcher's Cloud Run jobs use, and confirm:
   - it can write to the clients shared drive (it can already read it)
   - it can read the T&C source files in Drive
   - whether the existing HubSpot token or secret includes `crm.objects.quotes.read`. If not, list the scope that needs adding.
4. **Find the shared drive.** Get the clients shared drive ID, and check whether `uncategorized-contracts` already exists at its top level.
5. **Check the signed PDF.** Download `hs_pdf_download_link` for quote `43009909001` (Airwave, signed 2026-10-13) to a local temp file. Report whether it contains the signatures and the Dropbox Sign audit trail page. If it doesn't, the signed PDF must come from somewhere else (check for file attachments on the quote's associated deal), and the download step needs redesigning.
6. **Check the shared drive's membership.** List the members of the clients shared drive and flag any external (non-BW&CO) accounts. Anyone who is a member would see everything in `uncategorized-contracts`, including other clients' contracts.

**Report format:** a short summary of findings for each item, plus any assumption in this plan that turned out to be wrong.

---

## Phase 1: Restructure the script into the Matcher

- Move the script into the Matcher's package as `contract_sync/` (or whatever matches the repo's existing layout for jobs).
- Reuse the Matcher's existing HubSpot client, Drive client, Firestore client, config loading, and logging. Don't keep a parallel set from the standalone script.
- Remove the `DRIVE_PARENT_FOLDER_ID` concept. Replace it with `CLIENTS_SHARED_DRIVE_ID` and a folder resolver (Phase 2).
- Keep these from the existing script: quote search and pagination, the 429/5xx retry with backoff, `fix_text`, T&C link extraction (`DRIVE_ID_RE`), the Google Doc → PDF export path, the `--backfill`, `--dry-run`, and `--limit` flags, and a non-zero exit code on any failure.

## Phase 2: Folder resolution

Implement `resolve_destination(hubspot_company_id, company_name) -> (folder_id, matched: bool)`:

1. Look up the Firestore mapping using the key found in Phase 0.
2. If it matches, take the mapped `_INTERNAL` folder ID, then find or create its `Contracts` child folder. Match `Contracts` case-insensitively so an existing `contracts` or `CONTRACTS` folder is reused. Tag the folder with `appProperties.contract_sync_role = "contracts"`.
3. If it doesn't match, find or create `uncategorized-contracts` at the shared drive root, tagged with `appProperties.contract_sync_role = "uncategorized"`.
4. Cache resolutions per run.

Always resolve folders by **stored Drive ID**, never by searching for the `{company}_INTERNAL` name. That naming convention isn't consistent across clients.

## Phase 3: Idempotency and reconcile

- Before uploading, search the **entire shared drive** for files with `appProperties has {key='hubspot_quote_id' and value='<id>'}`. Use `corpora=drive` with `driveId`. This replaces the old search scoped to a single folder.
- **If the file exists in the correct destination:** skip it.
- **If it exists in `uncategorized-contracts` but the client is now matched:** move it with `files.update` and `addParents`/`removeParents`, strip the `[company]` name prefix, and log it as `reconciled`.
- **If it doesn't exist:** upload it, or copy it in the T&C case.
- Add a `--reconcile-only` flag. It scans `uncategorized-contracts`, re-resolves each file's company (store `hubspot_company_id` in each file's appProperties so this needs no HubSpot calls), and moves any file that now has a match. Run it at the end of every scheduled run.

## Phase 4: Tests

- Unit tests (mock HubSpot, Drive, and Firestore):
  - `fix_text`
  - T&C ID extraction (from `/file/d/`, `/document/d/`, and `?id=` URLs, with HTML-escaped input)
  - quote → company resolution, including the deal fallback and the no-company case
  - destination resolution, matched and unmatched
  - the idempotency decision table: skip, move, or upload
- Integration dry-run: `--backfill --dry-run --limit 10` against real services. It should print the planned destination for each quote, marked **MATCHED** or **UNCATEGORIZED**.

## Phase 5: Deploy

- A new Cloud Run **job** `contract-sync` in `cc-matcher-v1`, using the Matcher's existing service account.
- Secrets come from Secret Manager, the same pattern the Matcher already uses.
- Cloud Scheduler runs it every 30–60 min (pick one) with `LOOKBACK_DAYS=3`. The overlap is deliberate; idempotency handles it.
- Do the one-time backfill as a manual job execution with `--backfill`, **only after** the dry-run output has been reviewed.
- Send failures and the run summary (`synced / skipped / reconciled / uncategorized / failed`) to Cloud Logging. Add a log-based alert on `failed > 0`.

---

## Guardrails (non-negotiable)

- **No fuzzy matching of companies to folders.** A contract filed in the wrong client's folder is a confidentiality breach. Accept only an exact HubSpot company ID match, or, if Phase 0 shows the mapping is name-keyed, an exact match on normalized names (case and whitespace normalization only). Anything ambiguous goes to `uncategorized-contracts` and is logged.
- **Never write into the shared-with-client subfolder.** Contracts go only into `{company}_INTERNAL/Contracts/`.
- **Never delete Drive files.** The only mutations allowed are create, copy, and move.
- **Nothing irreversible on the first runs.** The backfill must be preceded by a reviewed `--dry-run`.

## Acceptance criteria

- [ ] Phase 0 report reviewed and confirmed
- [ ] Dry-run of all signed quotes shows a destination for each, with the matched vs. uncategorized counts reported
- [ ] Backfill completes; re-running it produces 0 new uploads
- [ ] A test quote for an unmatched company lands in uncategorized, then moves to `Contracts` after the company is added to the mapping
- [ ] Scheduled job is running, with an alert configured
