"""Transport-agnostic tool implementations (Phase 3 / Phase 5.1).

Plain functions, Streamlit-free, JSON-serialisable in/out, shared by the Drive
onboarding agent (jobs/onboard_job.py) and — later — the Matcher MCP server.
Written and tested once here; the agent's tools.py holds only the schemas the
model sees.
"""
