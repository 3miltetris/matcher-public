# matcher-mcp — the PUBLIC MCP server for the claude.ai custom connector.
# Explicit COPY list, never `COPY . .`: nothing from the repo root (secrets.toml,
# service-account key files) ships in a public image. src/modules/ holds only
# code (no secrets), and the server reaches a broad slice of it transitively, so
# it is copied wholesale rather than enumerated file by file.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.matcher_mcp.txt .
RUN pip install --no-cache-dir -r requirements.matcher_mcp.txt

COPY src/__init__.py src/__init__.py
COPY src/modules src/modules
COPY services/__init__.py services/__init__.py
COPY services/matcher_mcp services/matcher_mcp

ENV PYTHONPATH=/app PYTHONUNBUFFERED=1

# Fail the build if any import is missing from the COPY list above. fastmcp and
# openai are imported lazily at request time, so this proves the module graph
# resolves without standing up the server.
RUN python -c "import services.matcher_mcp.main"

CMD ["python", "-m", "services.matcher_mcp.main"]
