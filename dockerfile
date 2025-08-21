FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/share/nltk_data \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_MAX_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preload NLTK corpora at build-time (no runtime downloads)
RUN python - <<'PY'
import nltk
for pkg in ["wordnet", "omw-1.4", "punkt", "stopwords"]:
    nltk.download(pkg, download_dir="/usr/share/nltk_data")
PY

# Prefetch the SBERT model into the image (no runtime HF download)
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
PY

COPY . .
EXPOSE 8080

# SINGLE small worker to avoid OOM
CMD ["gunicorn","multilingualnews:app","--bind","0.0.0.0:8080","--workers","1","--threads","2","--timeout","120"]
