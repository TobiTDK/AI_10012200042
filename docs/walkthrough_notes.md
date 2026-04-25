# Video Walkthrough Notes
## CS4241 RAG Chatbot — [YOUR NAME] | [YOUR INDEX NUMBER]

---

## Suggested Video Structure (~8-10 minutes)

---

### 1. Introduction (0:00 – 0:45)
- State your name and index number
- Briefly describe the project: a custom RAG chatbot for Academic City
- Mention the two data sources: Ghana Election Results CSV, 2025 Budget Statement PDF
- State: "No LangChain or LlamaIndex — every component is implemented manually"

---

### 2. Code Walkthrough (0:45 – 3:30)

Walk through the folder structure live in VS Code or terminal:

```
ai_index_number/
├── app.py               ← Streamlit entry point
├── src/
│   ├── ingestion/       ← Data loading (CSV + PDF)
│   ├── preprocessing/   ← Cleaning + chunking
│   ├── retrieval/       ← Embedder, FAISS, BM25, hybrid scoring
│   ├── generation/      ← Prompt builder, LLM client
│   ├── pipeline/        ← Full RAG orchestrator
│   └── evaluation/      ← Test suite
```

**Key files to highlight:**
- `src/preprocessing/chunking.py` — Show both fixed-size and paragraph-aware strategies. Explain the overlap logic.
- `src/retrieval/scoring.py` — Show the `domain_score()` function. Explain each weight.
- `src/generation/prompt_builder.py` — Show V1, V2, V3 prompt templates side by side. Highlight the differences.
- `src/pipeline/rag_pipeline.py` — Show the full pipeline flow: ingest → embed → retrieve → score → prompt → generate → log.

---

### 3. Live Demo (3:30 – 6:30)

Open the Streamlit app:
```bash
streamlit run app.py
```

**Demo queries (run these in order):**

1. ✅ **Factual – Election:**
   > "How many votes did the NDC win in the 2024 presidential election?"
   - Show: retrieved chunks, scores table, V3 prompt, answer.

2. ✅ **Factual – Budget:**
   > "What percentage of Ghana's 2025 budget is allocated to education?"
   - Show: top chunk from PDF, final_score breakdown.

3. ⚠️ **Adversarial – Out of scope:**
   > "What is the capital of France?"
   - Show: low scores, system says "I do not have enough information."

4. ⚠️ **Adversarial – Ambiguous:**
   > "Who won?"
   - Show: confused retrieval, V3 prompt's fallback instruction kicks in.

5. 📊 **Prompt comparison:**
   - Switch sidebar from V3 to V1 on the France question.
   - Show V1 answers "Paris" from LLM training — demonstrate why V3 is better.

6. ⚖️ **RAG vs. Pure LLM:**
   - For the NDC votes question, expand the "Compare: Pure LLM answer" panel.
   - Show LLM making up a number vs RAG citing the CSV.

---

### 4. Evaluation Suite (6:30 – 8:00)

- Click "Run Evaluation" button in the app (or show terminal: `python src/evaluation/run_evaluation.py`)
- Open `outputs/evaluation_results.json` — show structure
- Summarize: RAG accuracy ~100% for factual, LLM hallucination ~78%
- Mention the score metrics: `refused_appropriately`, `contains_numbers`, `possible_hallucination`

---

### 5. Innovation Feature (8:00 – 8:45)

- Open `src/retrieval/scoring.py`
- Walk through the weights:
  - 0.50 vector (primary semantic relevance)
  - 0.30 BM25 (exact keyword matching)
  - 0.10 source match (prevents election query → budget chunk)
  - 0.07 keyword overlap (reinforces domain signal)
  - 0.03 year bonus (anchors date-specific queries)
- Demonstrate: show score table in Streamlit for an election query — election CSV chunks score higher than budget chunks.

---

### 6. Wrap Up (8:45 – 9:30)
- Summarise what was built
- Mention possible improvements: BERT re-ranker, Pinecone production vector store, streaming responses
- State: "All evaluation results are in `docs/evaluation_report.md` and `outputs/evaluation_results.json`"
- Sign off with name and index number

---

## Commands to Demo

```bash
# Run app
streamlit run app.py

# Run evaluation script
python src/evaluation/run_evaluation.py

# Run tests
python tests/test_chunking.py
python tests/test_retrieval.py

# Inspect logs
cat outputs/logs.json | python -m json.tool | head -80
```
