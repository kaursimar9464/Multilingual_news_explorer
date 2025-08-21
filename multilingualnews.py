

import os, json, re, html, time, threading, hashlib
NLTK_DIR = os.environ.get("NLTK_DATA", "/usr/share/nltk_data")
os.environ["NLTK_DATA"] = NLTK_DIR
os.makedirs(NLTK_DIR, exist_ok=True)
import nltk
nltk.data.path = [NLTK_DIR] + nltk.data.path


_needed = ["punkt", "wordnet", "omw-1.4", "stopwords"]
for pkg in _needed:
    try:
        subdir = "tokenizers" if pkg == "punkt" else "corpora"
        nltk.data.find(f"{subdir}/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_DIR)
from datetime import datetime, timezone

import numpy as np
import requests, feedparser, nltk
from nltk.tokenize import sent_tokenize
from langdetect import detect, LangDetectException
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
from flask_cors import CORS




app = Flask(__name__)
app.url_map.strict_slashes = False
CORS(app, resources={r"/*": {"origins": [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://kaursimar9464.github.io",
]}})


LOCALES = [
    ("en","US"),
    ("fr","FR"),
    ("es","ES"),
    ("de","DE"),
    ("it","IT"),
    ("pt","BR"),
    ("hi","IN"),
    ("ja","JP"),
]


LANGUAGES = sorted({lang for (lang, _) in LOCALES})
TOPICS  = [] 

BOOT = {"state": "warming_up", "error": None}
model = None
articles = []
corpus_embeddings = None

def degree_centrality_scores_min(S: np.ndarray, threshold: float|None=0.1) -> np.ndarray:
    S = S.copy().astype(float)
    np.fill_diagonal(S, 0.0)
    if threshold is None:
        nz = S[S > 0]
        threshold = float(nz.mean()) if nz.size else 0.0
    return (S >= threshold).astype(float).sum(axis=1)

def gnews_top_rss(lang, region, topic=None):
    hl, ceid = f"{lang}-{region}", f"{region}:{lang}"
    base = f"https://news.google.com/rss?hl={hl}&gl={region}&ceid={ceid}"
    return base + (f"&topic={topic}" if topic else "")

def safe_lang(text):
    try:
        return detect(text) if text and text.strip() else None
    except LangDetectException:
        return None

def fetch_feed(url, tag):
    try:
        r = requests.get(url, timeout=12); r.raise_for_status()
    except Exception:
        return []
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
        key = a["Url"] or hashlib.md5((a["Title"] + (a.get("Description") or "")).lower().encode()).hexdigest()
        if key in seen: continue
        seen.add(key); out.append(a)
    return out

def summarize_text(text: str, max_sentences: int = 4) -> str:
    if not text: return ""
    s = sent_tokenize(text)
    if len(s) <= max_sentences: return " ".join(s)
    emb = model.encode(s, convert_to_tensor=True)
    sim = util.cos_sim(emb, emb).cpu().numpy()
    np.fill_diagonal(sim, 0.0)
    idx = np.argsort(-degree_centrality_scores_min(sim, 0.1))[:max_sentences]
    return " ".join(s[i].strip() for i in sorted(idx))

def clean_news(rows):
    out=[]
    for r in rows:
        t = r.get('title'); s = r.get('summary') or ""
        s = re.sub(r'<[^>]+>', ' ', s); s = html.unescape(s)
        s = re.sub(r'[\u00A0\u202F]+', ' ', s); s = re.sub(r'\s+', ' ', s).strip()
        out.append({
            'title': re.sub(r'\s+-\s+.+$', '', t or "").strip(),
            'summary': s,
            'summary_short': r.get('summary_short') or "",
            'url': r.get('url'),
            'published_at': r.get('published_at'),
            'source': r.get('source'),
            'lang': (r.get('lang') or "").lower(),
            'score': r.get('score'),
        })
    return out

def bootstrap():
    global model, articles, corpus_embeddings, BOOT
    try:
        BOOT = {"state": "warming_up", "error": None}
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        rows=[]
        for lang, region in LOCALES:
            rows += fetch_feed(gnews_top_rss(lang, region, None), f"Top[{lang}-{region}]")
            for t in TOPICS:
                rows += fetch_feed(gnews_top_rss(lang, region, t), f"{t}[{lang}-{region}]")
                time.sleep(0.1)

        rows = dedupe(rows)
        for a in rows:
            a["Locale"] = a["Source"].split("[")[-1].rstrip("]") if "[" in a["Source"] else ""
            a["LangDetect"] = safe_lang(f"{a['Title']}. {a['Description']}".strip())
            a["IngestedAt"] = datetime.now(timezone.utc).isoformat()
        articles = rows

        texts = [(a["Title"] + " " + (a.get("Description") or "")).strip() for a in articles]
        if texts:
            emb = model.encode(texts, convert_to_tensor=True)
        else:
            emb = None
        globals()["corpus_embeddings"] = emb
        BOOT = {"state": "ready", "error": None}
        print(f"[bootstrap] ready: {len(articles)} articles")
    except Exception as e:
        BOOT = {"state": "error", "error": str(e)}
        print("[bootstrap] ERROR:", e)

threading.Thread(target=bootstrap, daemon=True).start()

@app.get("/")
def root():
    return jsonify({"service":"Multilingual News Explorer API","status":"ok"}), 200

@app.get("/health")
def health():
    emb_ready = bool(getattr(corpus_embeddings, "shape", None)) if corpus_embeddings is not None else False
    return jsonify({"status": BOOT["state"], "error": BOOT["error"], "articles": len(articles), "embeddings_ready": emb_ready}), 200

def get_articles(query, lang=None, top_k=10):
    if not articles or corpus_embeddings is None:
        return []
    q = model.encode(query, convert_to_tensor=True)
    if lang:
        idxs = [i for i,a in enumerate(articles) if (a.get("LangDetect") or a.get("Locale","")).lower().startswith(lang)]
        if not idxs: return []
        sub = corpus_embeddings[idxs]; index_map = idxs
    else:
        sub = corpus_embeddings; index_map = list(range(len(articles)))
    hits = util.semantic_search(q, sub, top_k=top_k)[0]
    out=[]
    for h in hits:
        i = index_map[h["corpus_id"]]; a = articles[i]
        lang_code = (a.get("LangDetect") or a.get("Locale","").split("-")[0]).lower()
        short = summarize_text(a.get("Description") or a.get("Title") or "")
        out.append({
            "title": a["Title"], "summary": a["Description"], "summary_short": short,
            "url": a["Url"], "published_at": a["PublishedAt"], "source": a["Source"],
            "lang": lang_code, "score": float(h["score"])
        })
    return out

@app.get("/search")
def search():
    if BOOT["state"] != "ready":
        return jsonify({"error":"warming_up","status":BOOT["state"]}), 503
    q = (request.args.get("q") or "").strip()
    k = int(request.args.get("k", 10))
    lang = (request.args.get("lang") or "").lower()
    if not q: return jsonify([])
    try:
        return jsonify(clean_news(get_articles(q, lang=lang, top_k=k)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
