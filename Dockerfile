FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/share/nltk_data \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install --upgrade pip

RUN pip install "numpy==1.26.4"

RUN pip install "torch==2.2.0" --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python - <<'PY'

import nltk
for p in ['punkt','wordnet','omw-1.4','stopwords']:
    nltk.download(p, download_dir='/usr/share/nltk_data')
print("nltk ready")
PY

RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("sbert cached")
PY

COPY multilingualnews.py .


EXPOSE 8080
CMD ["gunicorn", "multilingualnews:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "2", "--timeout", "120"]
