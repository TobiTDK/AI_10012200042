# generation/prompt_builder.py
# Author: [Your Name] | Index: [Your Index Number]
#
# Three prompt versions as required by Part C:
#   V1 - Basic prompt (just injects context)
#   V2 - Hallucination-controlled prompt (explicit grounding instructions)
#   V3 - Structured prompt with chunk IDs, strict citation grounding
#
# Context window management:
#   - Chunks are pre-ranked by final_score (best first)
#   - We truncate to MAX_CONTEXT_WORDS to avoid overflowing the LLM context
#   - Weak chunks (score < MIN_SCORE_THRESHOLD) are filtered out

from typing import List, Tuple, Dict

MAX_CONTEXT_WORDS = 1800   # keeps us well inside typical 4k-token windows
MIN_SCORE_THRESHOLD = 0.15  # drop very low-scoring chunks


def _build_context_block(
    scored_chunks: List[Tuple[Dict, float, float, float]],
    max_words: int = MAX_CONTEXT_WORDS,
    min_score: float = MIN_SCORE_THRESHOLD,
) -> Tuple[str, List[Dict]]:
    """
    Filter, deduplicate, and truncate chunks.
    Returns the context string and the list of selected chunk dicts.
    """
    selected = []
    word_count = 0

    for chunk, v_score, b_score, final_score in scored_chunks:
        if final_score < min_score:
            continue
        chunk_words = len(chunk["text"].split())
        if word_count + chunk_words > max_words:
            break
        selected.append((chunk, final_score))
        word_count += chunk_words

    # Build readable context block
    parts = []
    for chunk, score in selected:
        source_label = f"[{chunk['chunk_id']} | Source: {chunk['source']} | Score: {score:.3f}]"
        parts.append(f"{source_label}\n{chunk['text']}")

    context_str = "\n\n---\n\n".join(parts)
    return context_str, [c for c, _ in selected]


# ─────────────────────────────────────────────────────
# V1 – Basic prompt
# ─────────────────────────────────────────────────────

def build_prompt_v1(query: str, scored_chunks: List[Tuple[Dict, float, float, float]]) -> Tuple[str, List[Dict]]:
    """Basic: inject context with minimal instruction."""
    context, selected = _build_context_block(scored_chunks)
    prompt = f"""You are a helpful assistant for Academic City University.

Context:
{context}

Question: {query}

Answer:"""
    return prompt, selected


# ─────────────────────────────────────────────────────
# V2 – Hallucination-controlled prompt
# ─────────────────────────────────────────────────────

def build_prompt_v2(query: str, scored_chunks: List[Tuple[Dict, float, float, float]]) -> Tuple[str, List[Dict]]:
    """Hallucination-controlled: tells the model to stick to provided context."""
    context, selected = _build_context_block(scored_chunks)
    prompt = f"""You are an AI assistant for Academic City University.
Your answers must be based ONLY on the context provided below.
Do NOT use any prior knowledge. If the answer is not in the context, say:
"I do not have enough information from the provided documents."

Context:
{context}

Question: {query}

Answer (based only on the above context):"""
    return prompt, selected


# ─────────────────────────────────────────────────────
# V3 – Structured final production prompt
# ─────────────────────────────────────────────────────

def build_prompt_v3(query: str, scored_chunks: List[Tuple[Dict, float, float, float]]) -> Tuple[str, List[Dict]]:
    """
    Structured prompt with chunk IDs, strict grounding, concise answer guidance.
    This is the default production prompt.
    """
    context, selected = _build_context_block(scored_chunks)
    chunk_id_list = ", ".join(c["chunk_id"] for c in selected) if selected else "none"

    prompt = f"""You are a precise AI assistant for Academic City University College.

INSTRUCTIONS:
1. Answer ONLY using the document excerpts below.
2. Do NOT fabricate or infer facts not present in the context.
3. If the provided context does not contain the answer, respond with:
   "I do not have enough information from the provided documents to answer this question."
4. Prefer concise, factual answers. Cite the chunk ID in brackets when possible, e.g. [pdf_fixed_3].
5. For numeric or statistical questions, quote exact figures if available.

Referenced chunk IDs: {chunk_id_list}

DOCUMENT EXCERPTS:
{context}

QUESTION: {query}

ANSWER:"""
    return prompt, selected


# ─────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────

PROMPT_BUILDERS = {
    "v1": build_prompt_v1,
    "v2": build_prompt_v2,
    "v3": build_prompt_v3,
}


def build_prompt(
    query: str,
    scored_chunks: List[Tuple[Dict, float, float, float]],
    version: str = "v3",
) -> Tuple[str, List[Dict]]:
    """Build a prompt using the specified version (v1, v2, or v3)."""
    builder = PROMPT_BUILDERS.get(version, build_prompt_v3)
    return builder(query, scored_chunks)
