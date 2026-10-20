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
`managed-agents-2026-04-01`) needs a newer SDK still. So before running the spike:
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
_Runtime cell — the captured proposal:_
```
<paste findings.questions.Q3_structured_output>
```

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
_Runtime cell — reported usage:_
```
<paste findings.questions.Q5_usage>
```

### Q6 — Deletion
**Confirmed (docs).** `client.beta.sessions.delete(session_id)` permanently
deletes the session, its event history, container, and checkpoints — this is what
decision 0.1(a) requires. Files the agent wrote are listed via
`client.beta.files.list(scope_id=session.id, betas=["managed-agents-2026-04-01"])`
and deleted via the Files API. Agents/environments are reusable and are
**archived, not deleted** (archive is permanent, no unarchive — confirm with a
human; never as routine cleanup). The spike deletes the session in a `finally`.
_Runtime cell:_ `<paste findings.questions.Q6_delete>`

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
_Runtime cell — event types seen (and on the biologics re-run):_
```
<paste findings.questions.Q7_event_types_seen for a normal folder AND a biologics folder>
```
Phase 3's fallback (3.7) — decline → end session → one-shot Haiku path — keys off
whatever this run shows, so fill it in before building 3.7.

---

## Decisions this spike feeds into Phase 3
- **Custom tools, host-side** (Q1/Q2) → `drive-onboard-job` keeps all creds; no
  public surface. Tools live in `src/modules/tools/` (plan §5.1) so MCP can reuse them.
- **Terminal `submit_proposal`** (Q3) → the proposal schema in plan §3.5 is the tool's
  `input_schema`; validation runs in the orchestrator before anything is saved.
- **Session budget + wall clock** (Q4) and **`list_cost` → `status.json`** (Q5).
- **`sessions.delete` in `finally` + startup stale sweep** (Q6) satisfies 0.1(a).
- **Refusal shape** (Q7) → drives the 3.7 fallback trigger.
