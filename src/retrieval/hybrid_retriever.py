# retrieval/hybrid_retriever.py
# Author: [Your Name] | Index: [Your Index Number]
# Hybrid retriever: merges FAISS (semantic) and BM25 (keyword) results,
# then applies domain-specific scoring.

import re
from typing import List, Tuple, Dict

from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.scoring import compute_final_scores
from src.retrieval.embedder import embed_query
from src.utils.helpers import deduplicate_chunks

ELECTION_KEYWORDS = {"election", "vote", "votes", "constituency", "candidate", "party", "npp", "ndc", "results", "polling", "parliamentary", "presidential", "region"}
BUDGET_KEYWORDS = {"budget", "revenue", "expenditure", "gdp", "fiscal", "tax", "allocation", "ministry", "spending", "deficit", "surplus", "appropriation", "2025"}


def classify_query(query: str) -> str:
    """
    Simple rule-based query classifier.
    Returns: 'election', 'budget', or 'mixed'
    """
    q_lower = query.lower()
    q_tokens = set(re.findall(r"[a-zA-Z]+", q_lower))

    election_hits = len(q_tokens & ELECTION_KEYWORDS)
    budget_hits = len(q_tokens & BUDGET_KEYWORDS)

    if election_hits > budget_hits:
        return "election"
    elif budget_hits > election_hits:
        return "budget"
    else:
        return "mixed"


class HybridRetriever:
    """
    Combines VectorStore and BM25Retriever with domain-specific scoring.
    """

    def __init__(self, vector_store: VectorStore, bm25: BM25Retriever):
        self.vector_store = vector_store
        self.bm25 = bm25

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 15,
    ) -> Tuple[str, List[Tuple[Dict, float, float, float]]]:
        """
        Full hybrid retrieval.

        Steps:
        1. Classify the query domain
        2. Retrieve top fetch_k from FAISS (vector search)
        3. Retrieve top fetch_k from BM25 (keyword search)
        4. Merge candidate pools
        5. Apply domain-specific scoring
        6. Return top_k results

        Returns:
            query_type (str),
            list of (chunk, vector_score, bm25_score, final_score)
        """
        query_type = classify_query(query)

        # Dense retrieval
        q_vec = embed_query(query)
        vector_results = self.vector_store.search(q_vec, top_k=fetch_k)

        # Keyword retrieval
        bm25_results = self.bm25.search(query, top_k=fetch_k)

        # Merge, score, rank
        scored = compute_final_scores(query, query_type, vector_results, bm25_results)

        # Return top_k
        return query_type, scored[:top_k]
