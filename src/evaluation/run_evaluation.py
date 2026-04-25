# evaluation/run_evaluation.py
# Author: [Your Name] | Index: [Your Index Number]
#
# Runs a structured evaluation of the RAG system vs pure LLM.
# Saves results to outputs/evaluation_results.json

import json
import os
import sys

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from src.pipeline.rag_pipeline import RAGPipeline

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "../../outputs/evaluation_results.json")

FACTUAL_QUERIES = [
    "How many votes did the NDC win in the 2024 presidential election?",
    "What was Ghana's GDP growth target in the 2025 budget?",
    "Which party won the most parliamentary seats?",
    "What is the total revenue allocation in the 2025 budget statement?",
    "Who won the Ablekuma North constituency?",
]

NUMERIC_QUERIES = [
    "What percentage of the budget was allocated to education?",
    "How many total valid votes were cast in the election?",
    "What is the projected fiscal deficit for 2025?",
]

ADVERSARIAL_QUERIES = [
    # Ambiguous
    "Who won?",
    "What happened in Ghana?",
    # Misleading
    "Did the NPP win the 2025 election?",  # Should note 2025 budget, not 2025 election
    "What did the finance minister say about taxes in 1999?",  # Wrong year
    # Incomplete
    "How much money?",
    "What are the results?",
    # Out-of-scope
    "What is the capital of France?",
    "Who is the CEO of OpenAI?",
]


def score_response(response: str, query: str) -> dict:
    """
    Heuristic scoring — not ground-truth, but useful for comparison.
    Returns dict with qualitative notes.
    """
    no_info_flag = "do not have enough information" in response.lower()
    has_numbers = any(char.isdigit() for char in response)
    is_short = len(response.split()) < 20
    hallucination_risk = (
        not no_info_flag
        and not has_numbers
        and any(k in query.lower() for k in ["how many", "percentage", "total", "amount"])
    )

    return {
        "refused_appropriately": no_info_flag,
        "contains_numbers": has_numbers,
        "is_very_short": is_short,
        "possible_hallucination": hallucination_risk,
    }


def run_evaluation(pipeline: RAGPipeline) -> None:
    results = {
        "factual": [],
        "numeric": [],
        "adversarial": [],
    }

    print("\n=== FACTUAL QUERIES ===")
    for query in FACTUAL_QUERIES:
        print(f"\nQ: {query}")
        rag_result = pipeline.query(query, top_k=4, prompt_version="v3")
        pure_llm = pipeline.query_pure_llm(query)
        scoring = score_response(rag_result["response"], query)

        entry = {
            "query": query,
            "rag_response": rag_result["response"],
            "pure_llm_response": pure_llm,
            "query_type": rag_result["query_type"],
            "final_scores": rag_result["final_scores"],
            "scoring": scoring,
        }
        results["factual"].append(entry)
        print(f"  RAG: {rag_result['response'][:200]}")
        print(f"  LLM: {pure_llm[:200]}")

    print("\n=== NUMERIC QUERIES ===")
    for query in NUMERIC_QUERIES:
        print(f"\nQ: {query}")
        rag_result = pipeline.query(query, top_k=4, prompt_version="v3")
        pure_llm = pipeline.query_pure_llm(query)
        scoring = score_response(rag_result["response"], query)

        entry = {
            "query": query,
            "rag_response": rag_result["response"],
            "pure_llm_response": pure_llm,
            "scoring": scoring,
        }
        results["numeric"].append(entry)

    print("\n=== ADVERSARIAL QUERIES ===")
    for query in ADVERSARIAL_QUERIES:
        print(f"\nQ: {query}")
        rag_result = pipeline.query(query, top_k=4, prompt_version="v3")
        pure_llm = pipeline.query_pure_llm(query)
        scoring = score_response(rag_result["response"], query)

        entry = {
            "query": query,
            "rag_response": rag_result["response"],
            "pure_llm_response": pure_llm,
            "scoring": scoring,
            "notes": "adversarial",
        }
        results["adversarial"].append(entry)
        print(f"  RAG: {rag_result['response'][:200]}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Evaluation] Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    p = RAGPipeline()
    p.initialize(chunking_method="fixed")
    run_evaluation(p)
