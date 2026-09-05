# Incerro RAG Chatbot

A simple, end-to-end Retrieval-Augmented Generation (RAG) chatbot that answers questions about [Incerro](https://www.incerro.ai/) — their products, services, and company information — grounded in scraped website content.

---

## Architecture

```
Incerro Website
      ↓
  scrape.py       → Fetches pages, strips boilerplate HTML, saves .txt files to data/
      ↓
  ingest.py       → Chunks text, embeds with Gemini, stores in ChromaDB
      ↓
  chat.py         → Retrieves top-4 chunks, sends to Gemini LLM with guardrails
      ↓
  app.py          → Streamlit UI with chat history + source attribution
```

---

## Data Sources

The following Incerro pages are scraped:

| File | URL |
|------|-----|
| home.txt | https://www.incerro.ai/ |
| services.txt | https://www.incerro.ai/services |
| about-us.txt | https://www.incerro.ai/about-us |
| geographic-intelligence.txt | https://www.incerro.ai/products/geographic-intelligence |
| 4sight.txt | https://www.incerro.ai/products/4sight |
| document-intelligence.txt | https://www.incerro.ai/products/document-intelligence |
| data-intelligence.txt | https://www.incerro.ai/products/data-intelligence |
| financial-intelligence.txt | https://www.incerro.ai/products/financial-intelligence |
| mvp-insight.txt | https://www.incerro.ai/insights/... |
| contact-us.txt | https://www.incerro.ai/contact-us |

---

## Scraping Approach

- Uses `requests` with a realistic browser User-Agent
- Parses HTML with `BeautifulSoup`
- Removes `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>` tags
- Extracts text from `<main>`, `<article>`, or `<body>`
- Cleans excessive whitespace
- Warns on low-content pages (possible JS-rendered pages)
- Logs and skips pages that fail; continues with the rest

---

## Chunking Approach

- Splits on paragraph boundaries (`\n\n`) first (semantic-aware)
- Target chunk size: ~1000 characters
- Overlap: ~120 characters (for context continuity)
- Filters out tiny fragments (< 50 characters)
- Falls back to sentence splitting for very long paragraphs

---

## Embeddings

- **Primary**: Google Gemini `models/text-embedding-004` via `chromadb`'s `GoogleGenerativeAiEmbeddingFunction`
- **Fallback**: ChromaDB default local embedding function (if no API key)

---

## ChromaDB

- Local persistent vector database stored in `./chroma_db/`
- Collection name: `incerro_docs`
- Each chunk stored with metadata: `source_url`, `page_title`, `chunk_index`
- Re-ingestion clears and recreates the collection

---

## Retrieval

- Top-4 most relevant chunks retrieved per query
- Simple distance-based relevance threshold (`MAX_DISTANCE = 1.5`)
- If all retrieved chunks exceed the threshold → fallback response returned

---

## LLM

- **Model**: Google Gemini `gemini-2.0-flash`
- System prompt enforces Incerro-only grounding
- User question + retrieved context sent together
- Returns grounded answer

---

## Guardrails

The system prompt instructs the model to:
- Answer ONLY from retrieved context
- Not use outside knowledge about Incerro
- Return a fallback if context is insufficient
- Not answer off-topic questions
- Resist prompt injection attempts

---

## Source Attribution

- Source URLs from retrieved chunks are displayed after every answer
- URLs come directly from chunk metadata (never invented)

---

## Setup

### 1. Clone / navigate to the project

```bash
cd incerro-rag-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Gemini API key

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
# Edit .env and set: GEMINI_API_KEY=your_key_here
```

Get a free key at: https://aistudio.google.com/apikey

### 4. Scrape Incerro pages

```bash
python scrape.py
```

### 5. Ingest into ChromaDB

```bash
python ingest.py
```

### 6. Run the chatbot

**Streamlit UI (recommended):**
```bash
streamlit run app.py
```

**Terminal mode:**
```bash
python chat.py
```

---

## Project Structure

```
incerro-rag-chatbot/
├── app.py              # Streamlit UI
├── scrape.py           # Web scraper
├── ingest.py           # Chunker + ChromaDB ingestion
├── chat.py             # RAG retrieval + Gemini LLM
├── requirements.txt
├── .env.example
├── README.md
├── data/               # Scraped .txt files
│   ├── home.txt
│   ├── services.txt
│   └── ...
└── chroma_db/          # Local persistent vector DB
```

---

## Possible Future Improvements

- **Sitemap-based incremental crawling** — auto-discover and re-scrape updated pages
- **Better chunking** — semantic chunking using sentence transformers or sliding window
- **Hybrid retrieval** — combine dense vector search with BM25 keyword search
- **Cross-encoder reranking** — rerank retrieved chunks for higher relevance
- **Metadata filtering** — filter by product/service category before retrieval
- **Evaluation framework** — RAGAS or similar for measuring answer faithfulness
- **Document versioning** — track page changes over time
- **Production vector database** — Pinecone, Weaviate, or Qdrant for scale
- **Improved hallucination detection** — confidence scoring or claim verification
- **Playwright-based JS rendering** — for fully dynamic React/Next.js pages
