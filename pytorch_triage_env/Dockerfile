# Build context: ./pytorch_triage_env
# docker build -f pytorch_triage_env/Dockerfile -t pytorch-triage-env ./pytorch_triage_env
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/env/

ENV PYTHONPATH=/app/env
ENV PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
