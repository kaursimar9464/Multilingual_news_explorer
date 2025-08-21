FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

# install system essentials (certs + tz)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata && \
    rm -rf /var/lib/apt/lists/*

# install Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade "pip<24" && \
    pip install --no-cache-dir -r requirements.txt && \
    pip check

# app code
COPY . .

# fly.io expects something listening on 8080
ENV PORT=8080
CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:8080", "multilingualnews:app"]
