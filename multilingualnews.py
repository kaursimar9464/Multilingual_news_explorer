# -*- coding: utf-8 -*-
# Run locally:
#   pip install -U flask flask-cors feedparser langdetect sentence-transformers torch
#   python app.py
# In your index.html set: const ENDPOINT = "http://127.0.0.1:5000";

import json
import time
import hashlib
import re, html
import numpy as np
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from lexrank import LexRank
import nltk
import os
nltk.download("punkt")      
nltk.download("punkt_tab")  
from nltk.tokenize import sent_tokenize
import requests
import feedparser
from langdetect import detect, LangDetectException

from sentence_transformers import SentenceTransformer, util

from flask import Flask, request, jsonify, send_from_directory

from flask_cors import CORS
NLTK_DIR = os.environ.get("NLTK_DATA", "/opt/render/nltk_data")
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)

# Download only if missing (keeps boots fast after first run)
for pkg in ["wordnet", "omw-1.4", "punkt", "stopwords"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_DIR)

URL = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
languages = ["en","fr","es","de","it","pt", "hi", "ja"]
import numpy as np
app = Flask(__name__, static_folder = 'static')
def degree_centrality_scores_min(sim_matrix: np.ndarray, threshold: float | None = 0.1) -> np.ndarray:
    """
    Compute degree centrality on a similarity graph:
    - zero the diagonal
    - threshold edges
    - centrality = number of neighbors above threshold
    """
    S = sim_matrix.copy().astype(float)
    np.fill_diagonal(S, 0.0)
    if threshold is None:
        nz = S[S > 0]
        threshold = float(nz.mean()) if nz.size else 0.0
    A = (S >= threshold).astype(float)
    return A.sum(axis=1)

# Locales you want (lang, region)
LOCALES = [
    ("en","US"),
    ("fr","FR"),
    ("es","ES"),
    ("de","DE"),
    ("it","IT"),
    ("pt","BR"),
    ("hi", "IN"), 
    ("ja", "JP")
    # add more: ("ar","SA"), ("ru","RU"), ("hi","IN"), ("ja","JP"), ("ko","KR"), ("zh","TW")
]

# Optional sections (empty list = only top stories)
TOPICS = ["WORLD","BUSINESS","TECHNOLOGY","SCIENCE","HEALTH","SPORTS","ENTERTAINMENT","NATION"]
# Or do only top stories by setting: TOPICS = []

def gnews_top_rss(lang, region, topic=None):
    # hl wants lang-region, ceid wants COUNTRY:lang
    hl = f"{lang}-{region}"
    ceid = f"{region}:{lang}"
    base = f"https://news.google.com/rss?hl={hl}&gl={region}&ceid={ceid}"
    if topic:
        base += f"&topic={topic}"
    return base

def safe_lang(text):
    try:
        return detect(text) if text and text.strip() else None
    except LangDetectException:
        return None

def fetch_feed(url, tag):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    feed = feedparser.parse(r.content)
    rows = []
    for e in feed.entries:
        rows.append({
            "Title": e.get("title", "") or "",
            "Description": e.get("summary", "") or "",
            "Url": e.get("link"),
            "PublishedAt": e.get("published", None),
            "Source": tag
        })
    return rows

def dedupe(rows):
    seen, out = set(), []
    for a in rows:
        key = a["Url"] or hashlib.md5((a["Title"] + (a["Description"] or "")).lower().encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out

# -------- Build articles (your original flow) --------
articles = []
for lang, region in LOCALES:
    # top stories (no topic)
    url = gnews_top_rss(lang, region, topic=None)
    articles += fetch_feed(url, tag=f"Top[{lang}-{region}]")
    # add sections
    for t in TOPICS:
        url = gnews_top_rss(lang, region, topic=t)
        articles += fetch_feed(url, tag=f"{t}[{lang}-{region}]")
        time.sleep(0.2)  # polite delay

articles = dedupe(articles)

# annotate locale + detected language (optional)
for a in articles:
    a["Locale"] = a["Source"].split("[")[-1].rstrip("]") if "[" in a["Source"] else ""
    a["LangDetect"] = safe_lang(f"{a['Title']}. {a['Description']}".strip())
    a["IngestedAt"] = datetime.now(timezone.utc).isoformat()

print("Collected:", len(articles))
print(json.dumps(articles[:3], indent=2, ensure_ascii=False))


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

texts = [(a["Title"] + " " + (a.get("Description") or "")).strip() for a in articles]

corpus_embeddings = model.encode(texts, convert_to_tensor=True)
def get_articles(query, lang=None, top_k=10):
    query_embedding = model.encode(query, convert_to_tensor=True)

    
    if lang:
        idxs = [
            i for i, a in enumerate(articles)
            if (a.get("LangDetect") or a.get("Locale", "")).lower().startswith(lang)
        ]
        if not idxs:  
            return []
        sub_articles = [articles[i] for i in idxs]
        sub_embeddings = corpus_embeddings[idxs]
    else:
        idxs = list(range(len(articles)))
        sub_articles = articles
        sub_embeddings = corpus_embeddings

    # 🔹 Step 2: Run semantic search only on this subset
    hits = util.semantic_search(query_embedding, sub_embeddings, top_k=top_k)[0]

    # 🔹 Step 3: Map hits back to articles
    output = []
    for hit in hits:
        i = idxs[hit["corpus_id"]]
        a = articles[i]
        lang_code = (a.get("LangDetect") or a.get("Locale", "").split("-")[0]).lower()
        short = summarize_text(a.get("Description") or a.get("Title") or "")
        output.append({
            "title": a["Title"],
            "summary": a["Description"],
            "summary_short": short,        # 👈 NEW
            "url": a["Url"],
            "published_at": a["PublishedAt"],
            "source": a["Source"],
            "lang": lang_code,
            "score": float(hit["score"])
        })
    return output


def summarize_text(text: str, max_sentences: int = 5) -> str:
   
    txt = (text or "").strip()
    if not txt:
        return ""

    # Sentence split (works for many langs with punkt/punkt_tab)
    sents = sent_tokenize(txt)
    if len(sents) <= max_sentences:
        return " ".join(sents)

    # Embed sentences
    sent_emb = model.encode(sents, convert_to_tensor=True)

    # Cosine similarity matrix (NxN)
    sim = util.cos_sim(sent_emb, sent_emb).cpu().numpy()

    # Remove self-similarity from centrality
    np.fill_diagonal(sim, 0.0)

    # Degree centrality = sum of similarities per sentence
    scores = degree_centrality_scores_min(sim, threshold=0.1)

    top_idx = np.argsort(-scores)[:max_sentences]
    top_idx_sorted = sorted(top_idx)
    return " ".join(sents[i].strip() for i in top_idx_sorted)


import re, html

def clean_news(rows):
    out = []
    for r in rows:
        title = r.get('title')
        summary = r.get('summary') or ""
        url = r.get('url')
        published = r.get('published_at')
        source = r.get('source')
        short = r.get('summary_short') or ""
        lang = (r.get('lang') or "").lower()           # 👈 NEW
        score = r.get('score')                          # (optional)

        summary = re.sub(r'<[^>]+>', ' ', summary)
        summary = html.unescape(summary)
        summary = re.sub(r'[\u00A0\u202F]+', ' ', summary)
        summary = re.sub(r'\s+', ' ', summary).strip()

        title_clean = re.sub(r'\s+-\s+.+$', '', title or "").strip()
        out.append({
            'title': title_clean,
            'summary': summary,
            'summary_short': short,                    
            'url': url,
            'published_at': published,
            'source': source,
            'lang': lang,                              
            'score': score,                            
        })
    return out


# ---------------- Flask API (minimal) ----------------
app = Flask(__name__)
# Keep CORS open for localhost:5500 → localhost:5000 during dev
CORS(app, resources={r"/*": {"origins": "*"}})

@app.get("/ping")
def ping():
    return "ok", 200

@app.get("/")
def root():
    return send_from_directory(app.static_folder, "index.html")

@app.get("/health")
def health():
    try:
        n_articles = len(articles)
    except Exception:
        n_articles = None
    emb_shape = getattr(corpus_embeddings, "shape", None)
    return jsonify({"articles": n_articles, "embeddings_shape": emb_shape is not None})

@app.get("/search")
def search():
    q = (request.args.get("q") or "").strip()
    k = int(request.args.get("k", 10))
    lang = (request.args.get("lang") or "").lower()   # ⬅ read dropdown value

    if not q:
        return jsonify([])

    try:
        results = get_articles(q, lang=lang, top_k=k)
        return jsonify(clean_news(results))
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
