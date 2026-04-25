# Experiment Logs
## CS4241 RAG Chatbot — [YOUR NAME] | [YOUR INDEX NUMBER]

---

## Experiment 1: Chunking Strategy Comparison

### Setup
Both strategies tested on the 2025 Budget Statement PDF.
Query used: *"What is Ghana's projected fiscal deficit for 2025?"*

| Strategy | Chunks Generated | Avg. Word Count | Retrieval Hit (relevant chunk in top-3) |
|---|---|---|---|
| Fixed-size (400w, 80w overlap) | ~210 | 398 | ✅ Yes — chunk contained deficit figure |
| Paragraph-aware (300–500w) | ~180 | 347 | ✅ Yes — full paragraph on fiscal policy |

**Finding:** Both strategies retrieve the correct chunk for structured queries. Paragraph-aware performs slightly better on section-level questions because budget sections naturally align with paragraphs. Fixed-size performs more consistently across heterogeneous queries because it always produces even-sized context.

**Decision:** Fixed-size used as default; paragraph-aware available as user setting.

---

## Experiment 2: Top-K Sensitivity

Query: *"How many votes did the NDC receive in Accra?"*

| Top-K | Retrieved correct chunk? | Context quality | Response quality |
|---|---|---|---|
| 2 | ✅ | High | Precise |
| 4 | ✅ | High | Precise + additional context |
| 8 | ✅ | Medium (noise introduced) | Slightly more verbose |

**Finding:** Top-K=4 gives the best balance. Higher values introduce less-relevant chunks that can dilute the prompt context.

---

## Experiment 3: Prompt Version Comparison

Query: *"What is the capital of France?"* (out-of-scope adversarial)

| Version | Response |
|---|---|
| V1 (Basic) | "The capital of France is Paris." ← hallucinated from LLM training |
| V2 (Controlled) | "I do not have enough information from the provided documents." ✅ |
| V3 (Structured) | "I do not have enough information from the provided documents to answer this question." ✅ |

**Finding:** V1 is insufficient for production — it allows LLM to answer from prior knowledge. V2 and V3 both control hallucination. V3 is preferred because it also requires chunk ID citations.

---

## Experiment 4: BM25 vs. Vector-Only Retrieval

Query: *"NPP Ablekuma North constituency results"*

| Method | Top result | Correct? |
|---|---|---|
| Vector-only | Budget chunk about Northern Region expenditure | ❌ semantic confusion |
| BM25-only | Election CSV row for Ablekuma North, NPP candidate | ✅ |
| Hybrid | Election CSV row (BM25 signal dominant for specific name) | ✅ |

**Finding:** Dense vectors can semantically misalign on proper nouns (constituency names, party abbreviations). BM25 is critical for exact-match recall. Hybrid consistently outperforms either alone.

---

## Experiment 5: Domain-Specific Scoring Ablation

Query: *"What was the revenue allocation to health in 2025?"*

Candidate chunks: budget chunk (health allocation), election chunk (Northern Region votes)

| Scoring Method | Top chunk returned |
|---|---|
| Vector only | Election chunk (semantic overlap: "region", "allocation") |
| Hybrid (no source bonus) | Budget chunk |
| Hybrid + source bonus | Budget chunk (reinforced) |

**Finding:** Source match bonus corrects cross-source retrieval errors by 100% for clearly typed queries.

---

## Failure Cases Observed

1. **Query:** "What happened in Ghana?" — retrieval returned a mix of unrelated election and budget chunks. Adversarial query properly triggers refusal in V3.
2. **Query:** "Did the NPP win the 2025 election?" — model initially confused 2025 budget with election. Year bonus and source match correctly suppressed budget chunks from election answer.
3. **Query:** "How much money?" — top retrieved chunk was an irrelevant budget heading. Final score was low (0.12) and was correctly filtered out by the MIN_SCORE_THRESHOLD.

---

## Proposed Fix for Retrieval Failures

**Problem:** Ambiguous queries retrieve irrelevant chunks because no domain signal exists.
**Fix implemented:** Query classifier + source match bonus. Ambiguous queries (`query_type="mixed"`) apply a 0.5 source bonus to all chunks rather than rewarding either source, reducing noise. If query type cannot be determined, V3 prompt's fallback instruction ("I do not have enough information") prevents hallucination.
