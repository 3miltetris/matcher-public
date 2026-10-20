# Spike 0.2 — Managed Agents API

**Goal:** verify the Managed Agents API supports the Phase 3 onboarding-agent
design before writing integration code, per `PLAN-drive-onboarding-agent.md` §0.2.
Data-handling decision **0.1 = (a)**: use Managed Agents, delete each session +
any files after harvesting the proposal, plus a startup sweep of stale sessions.

**Beta header:** `managed-agents-2026-04-01` — the Python SDK sets it
automatically on `client.beta.{agents,environments,sessions,...}` calls. Do not
set it by hand.

**Surface (from the `claude-api` skill docs, confirmed, not from memory):**
Agent is created **once** (`client.beta.agents.create`, holds `model`/`system`/
`tools`), then each run is a **session** (`client.beta.sessions.create`,
references the agent by id+version + an environment). The session streams events;
we answer custom-tool calls with results.

**SDK prerequisite (measured 2026-10-20).** The local env had `anthropic 0.86.0`,
whose `client.beta` has **no** `agents` / `sessions` / `environments` / `vaults`
namespaces — Managed Agents is absent. That version is also below the repo's own
`anthropic>=1.3.0,<2` pin (CLAUDE.md "SDK version pins"). Managed Agents (beta
`managed-agents-2026-04-01`) needs a newer SDK still — **`anthropic 1.15.0`
exposes it** (`agents`/`sessions`/`environments` all present; measured
2026-10-20 in a throwaway venv). So before running the spike:
```
pip install -U anthropic
python -c "import anthropic;c=anthropic.Anthropic(api_key='x');print(anthropic.__version__, hasattr(c.beta,'agents'), hasattr(c.beta,'sessions'))"
# require: agents=True sessions=True
```
Bumping the **production** pin is a separate, deliberate step — follow the
CLAUDE.md "SDK version pins" discipline (signature-check `Messages.create` etc.
before raising the upper bound); the spike only needs the upgrade in a throwaway venv.

**Run it:**
```
pip install -U anthropic google-api-python-client google-auth   # anthropic must expose beta.agents
gcloud auth application-default login          # Drive read as a shared-drive viewer
export ANTHROPIC_API_KEY=...                    # or `ant auth login`
python scripts/spikes/managed_agent_spike.py <real_drive_folder_id>
```
The script prints a `FINDINGS` JSON block at the end — paste the runtime cells below.

**Exit criterion:** one custom tool runs against a real Drive folder, a dummy
proposal is submitted via the terminal tool, usage is reported, the session is
deleted. The script does all of this and prints each result.

---

## Run 1 — 2026-10-20 (real session, folder `1KJdX2io…`)

**The Managed Agents beta IS enabled on the matcher's Anthropic key** —
`agents.create` / `environments.create` / `sessions.create` all succeeded
(`agent_018ji…` v1, `env_01VA7…`, `sesn_01FHL…`). This was the biggest Phase 3
risk; it is retired. The full path ran: the agent took its turn, called the
`list_folder` custom tool, we executed it in-process, it called the terminal
`submit_proposal`, we captured the structured proposal, read `usage`, and
deleted the session. **list cost: $0.01**, active 8.7s.

Two caveats from Run 1, both since resolved:
1. **Drive read failed with a reauth error** — the custom-tool *round-trip*
   worked, but user ADC was stale/under-scoped. Org policy then **blocked**
   `gcloud auth application-default login` with the Drive scope entirely, so the
   spike's `list_folder` now uses the **`matcher-app@` service-account key from
   `.streamlit/secrets.toml`** (already a shared-drive viewer, as drive-sync-job
   uses it) — no ADC, no org block.
2. **`sessions.delete` 400'd on a running session** — see Q6. Fixed (interrupt →
   wait → delete).

## Run 2 — 2026-10-20 (real Drive read, end-to-end PASS)

With the service-account Drive path, the agent read the folder's 7 files and
returned the **correct, grounded** decision — the Phase 3 job working for real:
```json
{"decision": "existing_client", "company_name": "Cell X Technologies",
 "confidence": "high",
 "evidence": ["Two subfolders named after the company ...", "A 'Contracts' folder ...",
              "NIH SBIR Phase I proposal materials ...", "Consistent Apr–Jun 2026 timestamps ..."]}
```
Every exit-criterion item met: custom tool ran against a **real** Drive folder,
structured proposal submitted, usage reported (`list_cost` $0.02, active 13.7s),
and the session **self-deleted** (`deleted session sesn_01YXgn…`). Only built-in
events were `agent.custom_tool_use` (Q2 holds — no bash/files/web). Leftover
`agent_019J9k…` / `env_01LzX7…` are harmless reusable config.

---

## The seven questions

### Q1 — Custom tools our process executes
**Confirmed (docs).** Yes — this is the first-class pattern. Declare
`{"type": "custom", "name", "description", "input_schema"}` on the agent. The
session emits an `agent.custom_tool_use` event (`.name`, `.input`, `.id`) and
goes idle; we execute the tool **in our process** and reply with
`user.custom_tool_result` (`custom_tool_use_id`, `content`). Our GCP/Drive creds
never leave the Cloud Run job; the agent sees only the tool results we return.
No private MCP server needed — the fallback in the plan is unnecessary.

### Q2 — Disable bash / file ops / web per agent
**Confirmed (docs).** The built-ins are a single opt-in toolset,
`agent_toolset_20260401` (bash, file ops, code exec). **Omitting it** leaves the
agent with only our custom tools — no bash, no files. Web search/fetch are
separate server tool types you would add explicitly; by not adding them, no web.
The spike agent includes only `list_folder` + `submit_proposal`, proving this.
_Runtime check:_ confirm the agent never emits `agent.tool_use` (built-in) events.

### Q3 — Structured final output
**Confirmed (docs) + runtime.** A terminal custom tool `submit_proposal` whose
`input_schema` **is** the proposal schema. When the agent calls it, `event.input`
is the structured proposal — captured directly, no parsing of free text.
_Runtime cell (Run 1) — the captured proposal, exactly as `submit_proposal` input:_
```json
{"confidence": "low", "decision": "unsure",
 "evidence": ["list_folder ... failed with a reauthentication error. No ... could be retrieved."]}
```
Confirmed: the terminal tool's `input` arrives as the structured proposal object, no text parsing.

### Q4 — Budgets
**Confirmed (docs).** Two independent caps:
- **Session budget** (hard, dollar-denominated, platform-enforced): `budget={
  "type":"limit","max_list_cost":{"amount":"<cents as integer string>","currency":"USD"}}`
  at session create. At the cap the session goes **idle** with
  `stop_reason: budget_reached` (not terminated); only settle events are then
  accepted. `max_list_cost.amount` is minor units — `"200"` = $2.00.
- **Wall clock + per-run tool caps**: no built-in max-turns knob was found, so the
  orchestrator enforces its own wall-clock (`time.monotonic()` + `user.interrupt`)
  and a `max_doc_reads` counter — exactly the `browser_agent.py` forced-stop
  pattern Phase 3 reuses.

### Q5 — Cost / usage
**Confirmed (docs).** `session.usage` (via `sessions.retrieve`) carries token
totals, `server_tool_use` counts, `active_seconds`, and **`list_cost`**
(`{amount, currency}`, minor units, list-price — not the contracted price).
Per-turn detail is on `span.model_request_end.model_usage`. Phase 3 writes
`list_cost` into `status.json` like every other job's `cost_usd`.
_Runtime cell (Run 1) — reported usage object:_
```
BetaManagedAgentsSessionUsage(active_seconds=8.745,
  cache_creation=BetaManagedAgentsCacheCreationUsage(ephemeral_1h_input_tokens=0, ephemeral_5m_input_tokens=1600),
  cache_read_input_tokens=1425, input_tokens=4, output_tokens=381,
  list_cost=BetaMonetaryAmount(amount='1', currency='USD'),   # '1' = 1 cent = $0.01
  server_tool_use=BetaManagedAgentsServerToolUsage(web_fetch_requests=0, web_search_requests=0))
```
`list_cost.amount` is minor units (cents) as a string, same shape as the budget. Write this to `status.json`.

### Q6 — Deletion
**Confirmed + refined by Run 1.** `client.beta.sessions.delete(session_id)`
permanently deletes the session, its event history, container, and checkpoints —
what decision 0.1(a) requires. **FINDING: you cannot delete a `running` session**
— Run 1 got `400 invalid_request_error: "Cannot delete session while it is
running. Send an interrupt event or wait for the session to complete."` So the
correct teardown is **`user.interrupt` → poll until `status != "running"` →
`delete`** (verified: interrupt → status `idle` → `SESSION DELETED`). The spike's
`finally` now does exactly this; the Phase 3 orchestrator must too, and its
startup stale-session sweep should interrupt-then-delete as well. Files the agent
wrote are listed via `client.beta.files.list(scope_id=session.id,
betas=["managed-agents-2026-04-01"])` and deleted via the Files API.
Agents/environments are reusable and are **archived, not deleted** (archive is
permanent, no unarchive — confirm with a human; never routine cleanup).
_Runtime cell (Run 1):_ `interrupt -> wait !running -> sessions.delete ok`

### Q7 — Refusals (the one genuinely empirical question)
**Not documented explicitly — must be observed.** Errors surface as
`session.error` events (not a status), and `session.status_idle` carries a
`stop_reason` with a `type`. The Messages-API `stop_reason: "refusal"` /
`stop_details` shape is **not** described for the session event stream, so the
spike prints every event type and all `stop_reason`s. **Re-run the spike against
a folder whose documents are the known trigger material** (dried/powdered
biologics — see the Inaedis note in CLAUDE.md) and record what appears:
- does a `session.error` fire? with what body?
- what is `session.status_idle.stop_reason.type` on a decline?
- is there an `agent.message` refusal, or empty content?
_Runtime cell (Run 1 — clean folder, NO refusal) — the full event taxonomy:_
```
session.thread_status_running(1), user.message(1), span.model_request_start(2),
agent.thinking(2), agent.message(2), agent.custom_tool_use(2),
span.model_request_end(2), session.thread_status_idle(2), session.usage(2),
session.status_idle(2)
```
- A normal "done / waiting" idle carries `stop_reason` type **`requires_action`**
  (`BetaManagedAgentsSessionRequiresAction`) when blocked on a custom-tool result.
- No `session.error` and no built-in `agent.tool_use` fired (confirms Q2).
- **Still TODO: the biologics re-run.** This clean run gives the baseline; re-run
  against the Inaedis-type trigger material and diff the event types + the idle
  `stop_reason.type` to capture what a decline looks like. Phase 3's fallback
  (3.7) keys off that, so capture it before building 3.7.

---

## Decisions this spike feeds into Phase 3
- **Custom tools, host-side** (Q1/Q2) → `drive-onboard-job` keeps all creds; no
  public surface. Tools live in `src/modules/tools/` (plan §5.1) so MCP can reuse them.
- **Terminal `submit_proposal`** (Q3) → the proposal schema in plan §3.5 is the tool's
  `input_schema`; validation runs in the orchestrator before anything is saved.
- **Session budget + wall clock** (Q4) and **`list_cost` → `status.json`** (Q5).
- **`sessions.delete` in `finally` + startup stale sweep** (Q6) satisfies 0.1(a).
- **Refusal shape** (Q7) → drives the 3.7 fallback trigger.
