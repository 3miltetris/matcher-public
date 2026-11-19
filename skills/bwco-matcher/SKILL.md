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

- **Find a company by name:** `search_companies` → then `get_company` (summary,
  website, pool, Drive assignment, intake) or `get_profile` (capability aspects,
  confirmed and unexplored markets).
- **Find companies by what they do:** `search_contacts` — semantic/embedding
  search across the whole contact store (lead sources + clients/prospects) for
  companies whose summary fits a technology description (e.g. "autonomous
  underwater drones"). Returns one row per company with the folder it is in and a
  0-1 score. Use this when you have a capability in mind but not a name;
  `search_companies` is for when you know the name.
- **Find grants:** `search_grants` — ranks open grant topics against a typed
  technology description.
- **Find a person:** `search_resumes`.
- **Check the pipeline:** `whats_new` (overnight digest) · `list_runs` /
  `get_run_status` (any job) · `get_match_results` (a matching run's rows).
- **Work the queues:** `list_attention` · `list_onboarding_proposals` /
  `get_onboarding_proposal`.
- **Trigger a job (two-step):** `run_matching` · `build_profiles` · `sync_drive` ·
  `sync_fathom` · `run_onboarding_agent`. See **Triggering jobs** below.
- **Set up a bulk match:** `list_match_options` lists the real contact sources
  (apollo / sba / clients / prospects…) and agency folders (DOD / HHS / SAM-GOV…)
  to choose from. Call it before `run_matching`. See **Running a bulk match**.

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

*(`search_grants` is the typed-description search for one capability. To score a
whole contact pool against an agency's topics, run a **bulk match** — see below.
The one-click "profile → topics" per-company aspect match still lives in the
Streamlit app's Grant Search / Bulk Aspect Match.)*

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

### Running a bulk match

When someone asks to run a bulk match, gather the choices before triggering —
don't guess codes:

1. **Call `list_match_options` first.** It returns the real `contact_sources`
   (the company pools/lead folders) and `agencies` (grant-topic folders).
2. **Ask the person to choose**, one short question at a time (in claude.ai these
   render as selectable options). Offer only what `list_match_options` returned:
   - **Company source(s)** — one or more of `contact_sources`. Note **prospects**
     is opt-in (`default_selected` is false); include it only if they say so.
   - **Agency/agencies** — one or more of `agencies` (e.g. DOD, HHS, SAM-GOV).
   - **Grant date filter (optional)** — offer to limit by `due_date` or
     `scraped_at`, with a from/to range (either side alone is fine, e.g. "due
     before 2026-12-31"), and whether to keep undated/rolling grants
     (`include_undated`, default yes). Skip it if they don't want one.
   - **Custom messages** — if they haven't said, ask whether they'd like a
     custom message written for each match (see below). Skip it if not.
3. **Then the two-step:** call `run_matching` with the chosen `sources`,
   `agencies`, any `date_column`/`date_from`/`date_to`/`include_undated`, and
   any confirmed `custom_messages`, **without** `confirm_token` → show the cost
   estimate (it echoes the custom messages back) → call again with the
   **same** arguments plus the `confirm_token` → follow with
   `get_run_status("matching-job", run_id)` and read the rows with
   `get_match_results`.

#### Custom messages

Custom messages are the connector's version of Bulk Matching's "custom
prompts": each one adds an AI-written column to every matched row (an intro
line, a "why you fit" sentence, a LinkedIn note…), written from that row's
company and grant. Never invent one or skip the confirmation:

1. **Ask** whether they want a custom message, unless they already said.
2. **Ask what it should say and for an example** of the exact output they want
   (or the format to follow). Also ask for a column name if they care, and a
   rough length.
3. **Confirm with a different example.** Write one yourself in the same format,
   tone and length but about a *different* made-up company and grant, show it,
   and ask "Is this exactly the output you want?" Revise until they say yes —
   do not call `run_matching` before that.
4. Pass it as `custom_messages`, a JSON array (up to 5):
   ```json
   [{"name": "intro_line",
     "instructions": "One friendly sentence telling the founder why this topic fits their technology.",
     "example": "Your low-SWaP lidar lines up well with the Army's GPS-denied navigation topic.",
     "columns": ["company_summary", "grant_summary", "title", "agency"],
     "word_limit": 40}]
   ```
   `name` is a lowercase column name (it can't reuse an existing column like
   `subject_line`). `example` is used as a format/tone reference — the job
   never copies its specifics. `columns` default to `company_summary` +
   `grant_summary`; add `firstName`, `companyName`, `title`, `agency`,
   `due_date`, `funding_amount`… if the message needs them (a message that
   greets by name needs `firstName`). Setting `custom_messages` turns on
   email pre-write, so the run also writes the default `subject_line` and
   `ai_message` columns.
5. The messages appear as extra columns in `get_match_results`.

Remember a contact source (apollo, clients…) is **what gets scored**; an agency
(DOD, SAM-GOV…) is **what it is scored against** — SAM-GOV is an agency, not a
contact source.

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
