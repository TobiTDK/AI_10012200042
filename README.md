# SourceGround RAG — CS4241 (Academic City)

**Student name:** _—_  
**Index number:** _—_  
**Institution:** Academic City University College  

*(Add your name and index in this section before official submission.)*

---

## Overview

**SourceGround RAG** is a custom **Retrieval-Augmented Generation (RAG)** system for the CS4241 project. It answers questions from two local sources:

1. `data/Ghana_Election_Result.csv` — structured election data  
2. `data/2025_budget.pdf` (or `data/2025_Budget_Statement.pdf`) — national budget text  

The pipeline is implemented in `src/` **without** LangChain or LlamaIndex: ingestion, chunking, embeddings, FAISS, BM25, hybrid scoring, prompts, and generation are hand-wired. The main entry point is `app.py`: a **SourceGround**-branded, **ChatGPT-style** chat UI on top of `RAGPipeline` (Streamlit with `st.chat_input` / `st.chat_message`).

The structure and feature ideas in this README are informed by **typical CS4241 RAG briefs** (hybrid search, multi-mode generation, strong evaluation story). The **“What this repository actually implements”** table below is what the code in `src/` does today; use the **roadmap** section for planned or future extensions.

---

## What this repository actually implements

These features are **present in the current code**:

| Area | What it does |
|------|----------------|
| **Ingestion** | CSV via pandas; PDF text via PyMuPDF (`src/ingestion/`). |
| **Chunking** | Row-based election chunks; PDF **fixed** or **paragraph-aware** windows (`src/preprocessing/`). |
| **Retrieval** | `sentence-transformers` embeddings, **FAISS** (cosine), **BM25** (`rank-bm25`), merged candidate pool. |
| **Scoring** | **Domain-aware fusion**: `vector` + `BM25` + **source / keyword / year** bonuses; rule-based `election` / `budget` / `mixed` query type (`src/retrieval/hybrid_retriever.py`, `scoring.py`). |
| **Generation** | Prompt versions **v1 / v2 / v3** (`src/generation/prompt_builder.py`). **OpenAI** or **Ollama** via OpenAI-compatible clients (`src/generation/llm_client.py` — ollama.com/v1, optional local base via env). |
| **UI** | **Chat-style** Streamlit: `st.chat_message` + `st.chat_input`, **multi-turn** session, **suggested-question** chips (six topics in the main area; first three duplicated in the sidebar), **New chat** to clear the thread. Under each assistant turn: expanders for **Sources & evidence** (chunks, scores, table), optional **full prompt** and **debug JSON** (toggles in the sidebar), and **no-RAG** model comparison. Footer expander: **Run evaluation suite**. |
| **Logging** | Append events to `outputs/logs.json` (`src/utils/logger.py`). |
| **Evaluation** | `src/evaluation/run_evaluation.py` and adversarial / factual cases → `outputs/evaluation_results.json`. |
| **Tests** | `tests/test_chunking.py`, `tests/test_retrieval.py`. |

**Scoring (implemented)** — weighted combination, including normalized dense + lexical + domain heuristics (see `src/retrieval/scoring.py` for the exact `WEIGHTS`).

---

## Roadmap (not all implemented here)

Course-style **target architectures** may include: query expansion, cross-encoder rerank, JSONL logging with `request_id`, strict offline / extractive mode, chat export, `build_index.py`, CI benchmarks. The items below are **candidates** for a design report or viva; implement and document them in-repo before claiming them in writing.

- Query expansion / alias normalization before retrieval  
- Cross-encoder or second-stage rerank on a larger candidate pool  
- Dedicated **offline** path (no API key): extractive or template answer from top chunks only  
- JSONL append-only logging with per-stage timings and `request_id`  
- Thread **export** to file, long-term **history** across sessions, confidence scoring from numeric agreement (the app keeps **in-session** chat only)  
- `build_index.py` to precompute embeddings and cold-start cache  

---

## Architecture (simplified, as implemented)

```text
User message (chat input, sidebar chip, or main suggested chip)
  -> RAGPipeline.query
  -> query classification (election / budget / mixed)   [rule-based keywords]
  -> FAISS + BM25 retrieval (overlapping candidate pool)
  -> domain-specific score fusion
  -> prompt build (v1 / v2 / v3) + context selection
  -> LLM: OpenAI or Ollama (OpenAI-compatible API)
  -> log to outputs/logs.json; optional evaluation JSON
  -> Streamlit: append to in-memory chat history (session only)
```

### Suggested questions (in `app.py`)

The `SUGGESTED_Q` list defines **six** full questions (with short button labels) covering NDC/NPP, regional results, budget revenue, deficit/debt, and parliament. The **first three** are repeated in the **sidebar** for quick access. Edit the list in code to tune demos for your viva.

---

## Data files

Place files under `data/`:

- `Ghana_Election_Result.csv`  
- Budget PDF as **`2025_budget.pdf`** *or* **`2025_Budget_Statement.pdf`** (pipeline auto-picks the one that exists).

`data/*.csv` and `data/*.pdf` are gitignored in this project—**copy them in** before running (they are not always bundled in archives).

---

## Setup

```bash
python -m venv .venv
```

Windows: `.venv\Scripts\activate`  
macOS/Linux: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` (see **Generation modes** below and `.env.example` for variable names this codebase reads).

```bash
streamlit run app.py
```

**Tests (no pytest in requirements by default):**

```bash
python tests/test_chunking.py
python tests/test_retrieval.py
```

**Evaluation:** from the app expander, or by importing `run_evaluation` once a pipeline is initialised — output: `outputs/evaluation_results.json`  

There is no `build_index.py` in this tree yet; the Streamlit app builds indexes on first load (cached for the session).

---

## Generation modes (this repo)

| Mode | Configuration |
|------|----------------|
| **OpenAI** | `OPENAI_API_KEY` set; optional `OPENAI_MODEL`. |
| **Ollama Cloud** | `OLLAMA_API_KEY` + `OLLAMA_MODEL` (must be a **cloud** model tag, e.g. from [ollama.com search — cloud](https://ollama.com/search?c=cloud) or `GET https://ollama.com/api/tags`). Default in code: `gpt-oss:120b` if `OLLAMA_MODEL` is unset. `OLLAMA_BASE_URL` / `OLLAMA_CLOUD_BASE_URL` default to `https://ollama.com/v1`. |
| **Both keys** | `USE_OLLAMA=0` / `false` / `no` → prefer **OpenAI**; otherwise Ollama wins when both keys are set. `LLM_PROVIDER` overrides. |

**Local Ollama** (e.g. `ollama run llama3.1:8b`): use an OpenAI-compatible base such as `http://127.0.0.1:11434` in `OLLAMA_BASE_URL` / `OLLAMA_CLOUD_BASE_URL` and set `OLLAMA_MODEL` to a tag you have pulled **locally**. **Cloud** and **local** use different name lists — a 404 on ollama.com often means the tag is local-only. See Ollama docs for your version and auth.

A fully **key-free offline** default is **not** implemented in the current `llm_client` (no keys → `EnvironmentError`). You can add an `OFFLINE_MODE` path in a future change.

---

## Streamlit UI (as implemented)

- **Layout:** `layout="centered"`, page title **SourceGround**, first load shows a **status** while the pipeline indexes data (cached; changing **chunking** in the sidebar rebuilds the index).  
- **Sidebar:** **SourceGround** + retrieval sliders (**Top-K**, **Prompt** v1–v3, **Chunking** fixed vs paragraph), toggles (**Debug JSON**, **Show full prompt**), the **first three** suggested-question buttons (same list as the main list), **New chat**, and backend / data captions.  
- **Main (chat):** a **suggested-questions** grid: **two rows of three** chips; when the thread is not empty, the same set is available in a **collapsed expander** so the conversation stays in view. The user can also type in **`st.chat_input`** at the bottom.  
- **Per assistant turn:** the reply uses **`st.chat_message`**; expanders for **Full prompt** (if enabled), **Sources & evidence** (per-chunk scores and table), **Debug** (if enabled), and a **no-document** comparison (button to call the raw model for that user question).  
- **Bottom:** **Run evaluation suite** expander (writes `outputs/evaluation_results.json`).  
- **Not in scope:** account login, server-side history, streaming tokens, or export to `.md`/`.json` (you can add these later).

---

## Streamlit Community Cloud (notes)

- The app is memory-heavy (embeddings + FAISS). Reduce default `top_k` if the host is tight on RAM.  
- For share.streamlit.io, set **Secrets** like your `.env`. **Local Ollama** does not run on cloud hosts; use a remote API.  
- Add `runtime.txt` / `packages.txt` if you deploy, matching your Python version.

---

## Limitations

- Answers are constrained by the CSV/PDF and chunking.  
- **No** full query rewrite; classification is **keyword**-based.  
- **No** key-free “offline LLM” in the default `llm_client` — configure an API for generation or add extractive mode.  
- `outputs/logs.json` is a **single JSON array** file, not JSONL, unless you extend the logger.

---

## Submission

For CS4241 at **Academic City**: fill in **name** and **index** at the top of this README, keep the write-up aligned with the **“What this repository actually implements”** table, and cite your own repository URL on the cover sheet if your instructor requires it.

---

## Design reference

A strong **course-level RAG write-up** often includes hybrid retrieval, clear evaluation, and ablations. This README’s **roadmap** section is a place to list goals that go beyond the current `src/` tree, so you can show examiners both **what runs** and **what you would add next**—without overstating the code’s current scope.
