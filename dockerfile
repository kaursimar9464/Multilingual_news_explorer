# Python + NLTK corpora without importing nltk at build
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/share/nltk_data \
    PORT=8080 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Fetch NLTK corpora (no nltk import)
RUN mkdir -p $NLTK_DATA/corpora $NLTK_DATA/tokenizers && \
    curl -L -o $NLTK_DATA/corpora/wordnet.zip   https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip && \
    curl -L -o $NLTK_DATA/corpora/omw-1.4.zip   https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/omw-1.4.zip && \
    curl -L -o $NLTK_DATA/tokenizers/punkt.zip  https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip && \
    curl -L -o $NLTK_DATA/corpora/stopwords.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip

COPY . /app

# Use gunicorn in production
RUN pip install gunicorn
CMD exec gunicorn multilingualnews:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 120
