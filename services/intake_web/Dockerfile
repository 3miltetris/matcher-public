# intake-web — the PUBLIC DD intake service. Explicit COPY list, never
# `COPY . .`: nothing from the IAP-protected Matcher UI ships in a public image.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.intake_web.txt .
RUN pip install --no-cache-dir -r requirements.intake_web.txt

COPY services/__init__.py services/__init__.py
COPY services/intake_web/__init__.py services/intake_web/__init__.py
COPY services/intake_web/main.py services/intake_web/main.py
COPY services/intake_web/static services/intake_web/static

# src/modules/intake + what it imports: pools/aspect_profile (+ the research
# field lists and BucketManager they pull in), folder_match + drive_client
# (+ doc_extract),
# and TextProcessor for the prospect-row embedding.
COPY src/__init__.py src/__init__.py
COPY src/modules/__init__.py src/modules/__init__.py
COPY src/modules/intake src/modules/intake
COPY src/modules/pools.py src/modules/pools.py
COPY src/modules/aspect_profile.py src/modules/aspect_profile.py
COPY src/modules/finance_research.py src/modules/finance_research.py
COPY src/modules/tech_research.py src/modules/tech_research.py
COPY src/modules/folder_match.py src/modules/folder_match.py
COPY src/modules/drive_client.py src/modules/drive_client.py
COPY src/modules/doc_extract.py src/modules/doc_extract.py
COPY src/modules/Embedding/__init__.py src/modules/Embedding/__init__.py
COPY src/modules/Embedding/text_embedder.py src/modules/Embedding/text_embedder.py
COPY src/modules/GoogleBucketManager/bucket_manager.py src/modules/GoogleBucketManager/bucket_manager.py

ENV PYTHONPATH=/app PYTHONUNBUFFERED=1

# Fail the build if any import is missing from the COPY list above.
RUN python -c "import services.intake_web.main"

CMD ["sh", "-c", "uvicorn services.intake_web.main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips='*'"]
