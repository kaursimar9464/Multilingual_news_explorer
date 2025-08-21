# 🌍 Multilingual News Explorer

Cross-lingual semantic search + concise summaries over global Google News feeds.

**Try it online: https://kaursimar9464.github.io/Multilingual_news_explorer/**  
API (auto-used by the UI): https://multilingualnews.fly.dev

---

## What it does

- **Search in your language, read in any language.**  
  Type a query (e.g., “AI transparency rules”) and the app finds relevant news across locales.
- **Short, readable summaries.**  
  Each result includes a 2–5 sentence extractive brief.
- **Language filter.**  
  Narrow results to a specific language when you want.

**Supported languages:** EN, FR, ES, DE, IT, PT, HI, JA

---

## How it works (in brief)

- **Semantic retrieval:** Uses multilingual SBERT (`paraphrase-multilingual-MiniLM-L12-v2`) to embed headlines/descriptions and rank results by cosine similarity.
- **Summarization:** Builds a sentence-similarity graph and selects central sentences (TextRank-style) for a compact summary.
- **Feeds:** Pulls public Google News RSS for multiple locales and topics, dedupes near-duplicates, and caches embeddings for fast queries.

---

## Why it’s useful

- Cut through duplicate headlines across outlets/countries.
- Discover coverage outside your default locale.
- Skim summaries to decide what’s worth a full read.

---

## Notes & Limits

- **Cold start:** If the API was idle, your first search may take a few seconds while it “warms up.” Try again if needed.
- **Source links:** Click a card’s title to read the full article on the publisher’s site.
- **Privacy:** No accounts. Queries are sent to the hosted API only to compute results; no personal data is stored.

---

## Feedback & Issues

Bug report or feature idea? Open an issue in this repo—include your query, selected language, and a screenshot if possible.

---

## For developers (optional)

- UI: static site (Tailwind) on GitHub Pages, points to the Fly.io Flask API.
- API endpoints:
  - `GET /health` → `{ articles, embeddings_ready }`
  - `GET /search?q=<query>&k=<int>&lang=<code>` → ranked results with `summary_short`

> You don’t need any of this to use the app—just visit the live link above.
