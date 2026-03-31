FROM python:3.11-slim

# Install only the packages the job actually needs — no streamlit, no selenium
WORKDIR /app

COPY requirements.job.txt .
RUN pip install --no-cache-dir -r requirements.job.txt

# Copy application code
COPY src/ src/
COPY jobs/matching_job.py jobs/matching_job.py

# src/ modules are importable as 'src.modules.*'
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "jobs/matching_job.py"]
