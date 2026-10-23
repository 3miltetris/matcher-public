---
name: bwco-matcher
description: >-
  Use when working with the BW&CO Matcher from claude.ai through the matcher-mcp
  connector: finding grant/SBIR topics that fit a client, checking what the
  scheduled pipeline produced overnight (Drive sync, Fathom sync, SAM.gov,
  deep-research), looking up a company's capability profile or its match results,
  reviewing Drive onboarding proposals, triaging the attention queue, or searching
  résumés. It can also TRIGGER jobs (run a match, rebuild profiles, sync Drive or
  Fathom, run the onboarding agent) behind a two-step confirmation that always
  shows a cost estimate first. Explains which tool to reach for in each workflow,
  how to read the results, and what still has to be done in the Streamlit app.
---

# Using the Matcher from claude.ai

The **matcher-mcp** connector exposes the BW&CO Matcher to claude.ai. Most tools
are **read-only**; a handful **trigger jobs** behind a two-step confirmation. It
reads and acts on **live** state (the same GCS store the Streamlit app uses), is
gated to verified **@bwcoconsulting.com** accounts, and every call is audited.

Each tool's own description carries its exact parameters; this skill is the map of
which tool to use for which job.

## The tools at a glance

- **Find a company:** `search_companies` → then `get_company` (summary, website,
  pool, Drive assignment, intake) or `get_profile` (capability aspects, confirmed
  and unexplored markets).
- **Find grants:** `search_grants` — ranks open grant topics against a typed
  technology description.
- **Find a person:** `search_resumes`.
- **Check the pipeline:** `whats_new` (overnight digest) · `list_runs` /
  `get_run_status` (any job) · `get_match_results` (a matching run's rows).
- **Work the queues:** `list_attention` · `list_onboarding_proposals` /
  `get_onboarding_proposal`.
- **Trigger a job (two-step):** `run_matching` · `build_profiles` · `sync_drive` ·
  `sync_fathom` · `run_onboarding_agent`. See **Triggering jobs** below.

## Workflows

**"What grants fit <client>?"**
1. `search_companies(query="<name>")` to get the company's `company_key`.
2. `get_profile(company_key)` to see what the company actually does (its aspects
   and markets).
3. `search_grants(query="<a sentence or two describing that capability>")` — the
   search matches a **typed technology description**, so summarise the client's
   capability in the query. Narrow with `agency="DOD"` (or HHS, etc.) when the
   client is agency-specific. The `score` is 0–1 cosine similarity (relative,
   higher = closer) — not a probability or a decision.

*(A one-click "profile → topics" company match and bulk match runs live in the
Streamlit app's Grant Search / Bulk Aspect Match; the connector's `search_grants`
is the typed-description version.)*

**"What happened overnight?"** → `whats_new`. Gives the latest run of each
scheduled job (Drive sync, Fathom sync, SAM.gov, deep-research) with its headline
counts, plus what's sitting in the attention queue.

**"Did last night's sync run / what did run X do?"** →
`list_runs(job="drive-sync-job")` for recent run ids and state, then
`get_run_status(job, run_id)` for the counts and any error. Job names are
`drive-sync-job`, `fathom-sync-job`, `matching-job`, `sam-gov-job`,
`deep-research-job`, `client-profile-job`, `contact-import-job`,
`drive-onboard-job`.

**"Show the matches for <client> in run <id>"** →
`get_match_results(run_id, company_key="Name||https://site", good_only=true)`.
`good_match` is the AI validator's yes/no call; there is **no numeric per-row
score** in the results. Large runs page — pass the returned `next_cursor` to
continue.

**"What needs attention?"** → `list_attention` (optionally filter by `kind`, e.g.
`drive_unmatched_folder`, `fathom_unmatched_domain`). For folders the onboarding
agent investigated, `list_onboarding_proposals` then
`get_onboarding_proposal(folder_id)` for the decision, confidence, evidence and
the document include/exclude list.

**"Who do we know who works on <topic>?"** → `search_resumes(query=...)`.

## Triggering jobs (two-step — always show the estimate first)

The five trigger tools start a Cloud Run job. **Every one is two calls:**

1. Call the tool **without** `confirm_token`. It validates the arguments and
   returns an **estimate** (`units`, `est_cost_usd`, `est_minutes`), the caller's
   `spend_today_usd` against the `daily_cap_usd`, and a `confirm_token`. Nothing
   runs. **Show the estimate to the person and get their go-ahead.**
2. Call the tool **again with the same arguments plus `confirm_token`** to start
   the run. It returns `run_id`; follow it with `get_run_status(job, run_id)`.

The tools:
- `run_matching(agencies, sources, …)` — a bulk match run (contacts × topics).
- `build_profiles(company_keys, pool?, …)` — (re)build capability profiles, **≤25
  companies** per call.
- `sync_drive(company_keys, …)` — sync specific clients' Drive folders (only
  companies that already have an assigned folder).
- `sync_fathom(company_keys, lookback_days?, …)` — ingest recent Fathom meetings.
- `run_onboarding_agent(folder_ids, …)` — investigate **≤10** unmatched Drive
  folders (ids come from `list_attention` `drive_unmatched_folder` items).

Rules to relay honestly:
- The cost/time figures are **rough** — they bound spend and set expectations; the
  job reports actual cost in its status.
- There is a **per-user daily spend cap** (default $25). A confirm that would
  exceed it is refused; an admin raises it in `admin-config/mcp_limits.json`.
- If the job is already running, the tool says so (`already_running` + the live
  `run_id`) instead of starting a duplicate — report that, don't retry.
- A `confirm_token` lasts ~10 minutes and is bound to the exact arguments and
  caller. If it expires or the arguments change, just re-estimate.

## Conventions that matter

- **`company_key` is `"Name||https://website"`.** Always get it from
  `search_companies` first — never hand-construct or guess it.
- **Pools:** companies live in two pools, **clients** and **prospects**; lookups
  span both by default. `get_company`/`get_profile`/results say which pool a
  company is in.
- **Results are capped and paged.** Long text is truncated (middle-out) and big
  sets paginate with an opaque `next_cursor`; ask for the next page rather than a
  bigger limit.
- **It's live.** What you see is the current store. The read tools change nothing;
  the trigger tools start real jobs **only** after the confirm step.

## What this connector can't do — use the Streamlit app

- **Approving an onboarding proposal** (assigning a folder, promoting a prospect)
  — review it here, approve it in Drive Sync.
- **Running a deep-research / funding-source sweep** (`deep-research-job`,
  `contact-import-job`) — not exposed as triggers; use the Streamlit app.
- **Writing to HubSpot, or deleting companies/profiles** — admin/write actions
  deliberately stay out of the connector.

If someone asks for one of these, point them to the relevant Streamlit page
rather than implying the connector can do it.
