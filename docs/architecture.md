# Architecture & System Design
## CS4241 RAG Chatbot — [YOUR NAME] | [YOUR INDEX NUMBER]

---

## System Overview

The system follows a classic RAG (Retrieval-Augmented Generation) architecture with custom implementations at every layer. No end-to-end framework is used.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                            │
│  Ghana_Election_Result.csv     2025_Budget_Statement.pdf       │
└────────────┬───────────────────────────────┬────────────────────┘
             │                               │
             ▼                               ▼
     ┌───────────────┐             ┌──────────────────┐
     │  load_csv.py  │             │  load_pdf.py     │
     │  (pandas)     │             │  (PyMuPDF)       │
     └───────┬───────┘             └────────┬─────────┘
             │                              │
             ▼                              ▼
     ┌───────────────┐             ┌──────────────────┐
     │  clean_csv.py │             │  clean_pdf.py    │
     │  row → text   │             │  normalize text  │
     └───────┬───────┘             └────────┬─────────┘
             │                              │
             │                    ┌─────────▼──────────┐
             │                    │   chunking.py       │
             │                    │  A: fixed-size      │
             │                    │  B: paragraph-aware │
             │                    └─────────┬──────────┘
             │                              │
             └──────────────┬───────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │     All Chunks          │
              │  (chunk_id, source,     │
              │   text, keywords, year) │
              └────────────┬────────────┘
                           │
             ┌─────────────▼─────────────┐
             │       embedder.py         │
             │  sentence-transformers    │
             │  all-MiniLM-L6-v2 (384d) │
             └──────┬────────────────────┘
                    │
        ┌───────────▼──────────┐
        │    vector_store.py   │
        │  FAISS IndexFlatIP   │
        │  (cosine similarity) │
        └──────────────────────┘

        ┌──────────────────────┐
        │  bm25_retriever.py   │
        │  BM25Okapi index     │
        │  (keyword matching)  │
        └──────────────────────┘
```

---

## Query Pipeline

```
User Query (Streamlit)
        │
        ▼
hybrid_retriever.py
  ├── classify_query()
  │     └── Keyword rules → election / budget / mixed
  │
  ├── VectorStore.search()    ← top-15 semantic candidates
  │
  └── BM25Retriever.search()  ← top-15 keyword candidates
        │
        ▼
   scoring.py
   domain_score()
   final = 0.50×vec + 0.30×bm25 + 0.10×src + 0.07×kw + 0.03×yr
        │
        ▼
   Top-4 chunks (filtered, deduplicated)
        │
        ▼
prompt_builder.py
   V1 / V2 / V3 prompt templates
   (context window managed: ≤1800 words)
        │
        ▼
llm_client.py
   OpenAI or Ollama Cloud (OpenAI-compatible client; `https://ollama.com/v1` + `OLLAMA_API_KEY`)
   temperature=0.2
        │
        ▼
   Response + logger.py
   (saved to outputs/logs.json)
        │
        ▼
Streamlit UI display
```

---

## Component Descriptions

### 1. Ingestion
- **load_csv.py** — Uses `pandas.read_csv()`. Returns a DataFrame.
- **load_pdf.py** — Uses `fitz.open()` (PyMuPDF). Returns list of `{page, text}` dicts.

### 2. Preprocessing
- **clean_csv.py** — Drops null rows, strips whitespace, converts each row to readable `Field: Value | Field: Value` text.
- **clean_pdf.py** — Removes short noise lines, isolated page numbers, normalizes whitespace.

### 3. Chunking
- **Fixed-size** (Strategy A): 400-word windows, 80-word overlap. Simple, predictable, ensures no context is lost across boundaries.
- **Paragraph-aware** (Strategy B): Groups paragraphs to 300–500 words. Respects logical document structure. Better for budget sections that correspond to policy topics.

**Justification:** 400 words ≈ 512 tokens, which fits all-MiniLM-L6-v2's 256-token limit with truncation fallback. The 80-word overlap (20%) prevents meaningful sentences split across boundaries.

### 4. Embeddings
- Model: `all-MiniLM-L6-v2` (384-dimensional vectors)
- L2-normalized before FAISS indexing to convert inner product to cosine similarity

### 5. FAISS Vector Store
- Index type: `IndexFlatIP` (exact inner-product search after L2 normalization)
- Chosen over approximate indexes for exam reproducibility and small dataset size

### 6. BM25 Keyword Retriever
- Uses `rank-bm25`'s `BM25Okapi`
- Handles cases where dense embeddings miss exact entity names (e.g., constituency names, precise dollar figures)

### 7. Hybrid Retrieval & Domain-Specific Scoring
- Merges candidates from FAISS and BM25
- Normalizes each score to [0,1] using min-max normalization
- Applies weighted combination (Innovation feature — see Part G)
- Source-match bonus prevents cross-contamination (election query → budget chunk penalty)

### 8. Prompt Builder
- V1: Basic context injection
- V2: Explicit "do not hallucinate" instruction
- V3 (production): Chunk IDs cited, strict grounding, concise answer guidance
- Context window: Chunks truncated at 1800 words, weak chunks (score < 0.15) dropped

### 9. LLM Response Generation
- **OpenAI** (`gpt-4o-mini` by default) or **Ollama Cloud** (any cloud model via `OLLAMA_MODEL`, default `llama3.2`)
- Temperature: 0.2 for consistency
- System prompt reinforces strict context adherence

### 10. Streamlit UI
- Sidebar controls: top-K, prompt version, chunking method, debug toggle
- Main panel: query → chunks (with scores) → score table → prompt → answer
- Evaluation mode button to run the full test suite
- Pure LLM comparison expander

### 11. Logging
- Every query event saved to `outputs/logs.json`
- Logged fields: timestamp, query, query_type, chunks, scores, prompt, response

### 12. Evaluation
- Factual, numeric, and adversarial test sets
- RAG vs pure LLM comparison
- Heuristic scoring: hallucination indicators, appropriate refusal detection
- Results saved to `outputs/evaluation_results.json`
