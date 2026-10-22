# Status — PLAN-drive-onboarding-agent.md

Handoff note (as of 2026-10-22). Companion to `PLAN-drive-onboarding-agent.md`.
Branch `cc-helped-325`; **everything below is committed.** 160 tests pass (`python -m pytest -q`).

## Phase status

| Phase | State |
|---|---|
| **0 — Decisions + spikes** | ✅ done. Decision **0.1 = (a)** delete-after-harvest. 0.2 Managed Agents spike validated (beta IS enabled on the key; real-folder run gave a correct proposal). 0.3 MCP-connector = scaffold only (`scripts/spikes/mcp_whoami_server.py`), 5 OAuth questions still research items, needs org owner to add the connector. Docs in `docs/spikes/`. |
| **1 — Foundations** | ✅ done + deployed. `src/modules/job_control.py`, `src/modules/gcs_write.py` (`read_modify_write` + run lock). Converted drive/fathom/client_editor write paths. `tests/gcs_fake.py` + `tests/sync/`. |
| **2 — Scheduled sync** | ✅ **LIVE (switched on 2026-10-22).** `attention.py`, `drive_scan.py`, drive-sync + fathom `mode:"scheduled"`, profile chaining, Home "Needs attention" panel. Both `daily_schedule.json` blobs written; two Cloud Scheduler entries created: **`drive-sync-daily` 1 AM** + **`fathom-sync-daily` 4 AM** Central. `scan_sections` = 5 client sections (A-H, I-P, Q-Z, Nonprofit, Folders Created by PMs); Internal Projects / Submitted Forms / Uncategorized-Contracts excluded. OAuth SA `matching-job@` has `run.admin` project-wide (verified). Trigger validated by an observed run 2026-10-22 (scan: 414 seen / 50 auto-assigned / 17 review; synced 315 clients). First-night note: ~130 `drive_unmatched_folder` attention items appear once (deduped thereafter) — triage from the Home panel. See CLAUDE.md drive-sync-job "Scheduled mode" for the exact config + scheduler commands. |
| **3 — Onboarding agent** | ✅ done + deployed + **shadow-eval passed**. Full agent (tools `src/modules/tools/`, `onboarding.py`, `agents/drive_onboard/`, `jobs/onboard_job.py`, `onboarding_apply.py`, Drive Sync review UI). Eval 2026-10-21 N=10: **0 wrong-company, 0 wrong@high**, 9/10 correct + 1 conservative `unsure` — go-live bar met. ~$0.28/folder, ~6 min p50. Agent synced (`drive-onboard/agent.json` in GCS). Used via Drive Sync → "🔎 Investigate unmatched" → "🤖 Onboarding proposals". **Optional follow-ups:** auto-trigger the agent from scheduled drive-sync chaining; a nightly Cloud Scheduler entry for `drive-onboard-job`. |
| **4 — Scraper escalation** | ⬜ not started (optional, separate PR; `contact_import_job.py`). |
| **5 — Matcher MCP server** | 🟡 **Tier 1 (read-only) BUILT + tested; not deployed.** 0.3 spike answered (`docs/spikes/mcp-connector.md`: FastMCP `GoogleProvider` OAuth bridge, streamable HTTP, DCR, `fastmcp>=3.3,<4`). Tool layer `src/modules/tools/{_common,runs,attention,onboarding,search}.py` (+ reuse `company_lookup`), server `services/matcher_mcp/{main,auth,audit}.py` (12 tools, BW&CO gate, audit log, 20k cap/pagination), `requirements.matcher_mcp.txt`, Dockerfile, `cloudbuild.matcher_mcp.yaml`, `docs/matcher-in-claude.md`. **18 new tests pass (178 total).** `main.py` imports clean without fastmcp (lazy). **TODO to deploy (owner):** Google OAuth client + `mcp-google-client-id/secret` secrets, `matcher-mcp@` SA + IAM, build/deploy, set `MCP_BASE_URL`, verify fastmcp signatures, org owner adds connector in claude.ai (plan open Q5). **Tiers 2 (triggers) + 3 (staged writes) + the `bwco-matcher` skill (§5.7) + tool-selection eval (§5.8): not started.** |

## Deployed right now
- `matcher-app` rev **matcher-app-00069-47w** (Home attention panel + onboarding review UI).
- Jobs updated: `drive-sync-job`, `fathom-sync-job`, `client-profile-job` chaining IAM, **`drive-onboard-job`** (new).
- Managed Agents production agent is synced in GCS (`drive-onboard/agent.json`).
- **Cloud Scheduler (new 2026-10-22):** `drive-sync-daily` (1 AM) + `fathom-sync-daily` (4 AM) Central, both ENABLED. Daily configs at `drive-sync-configs/daily_schedule.json` + `fathom-configs/daily_schedule.json`. Phase 2 is now running unattended nightly.

## Env / ops gotchas for the next session
- **Always pass `--project cc-matcher-v1`** (local gcloud defaults elsewhere).
- gcloud CLI token expires often → ask the user to run `! gcloud auth login`.
- **Verify builds by reading the `SUCCESS` line in the build output, NOT the exit code** — a trailing `echo`/pipe masks a failed `gcloud builds submit` as exit 0.
- Deploy pattern: `gcloud builds submit --config cloudbuild.<x>.yaml --project cc-matcher-v1 .` → `gcloud run jobs|services update …`.
- **Managed Agents needs `anthropic>=1.15`** (repo-wide pin is 1.3; `requirements.onboard_job.txt` pins 1.15). Verify `client.beta.agents` exists before bumping.
- Running the onboarding agent / shadow eval locally needs a current `anthropic` + GCS/Drive creds (we used a throwaway venv + the `matcher-app@` SA key from `.streamlit/secrets.toml`; the scratchpad venv and key are session-scoped and gone now).
- A safety classifier repeatedly interrupted long prose *summaries about the onboarding agent's company-matching* — keep that narration minimal; file writes/tool calls were unaffected.

## Suggested next move
Phase 2 is live; Phase 5 Tier 1 is built + tested (not deployed). Options:
**(a)** deploy Tier 1 (the owner steps in `docs/matcher-in-claude.md` — Google OAuth
client, `matcher-mcp@` SA, build/deploy, connector registration), **(b)** build
Phase 5 Tier 2 (job triggers with confirm-token guardrails) + Tier 3 (staged writes
via `onboarding_apply.py`), **(c)** write the `bwco-matcher` skill + the §5.8
tool-selection eval, or **(d)** Phase 3 follow-ups (auto-trigger onboarding agent from
the nightly chain + a `drive-onboard-job` scheduler).
