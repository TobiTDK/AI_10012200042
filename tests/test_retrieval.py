# tests/test_retrieval.py
# Author: [Your Name] | Index: [Your Index Number]

import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.scoring import compute_final_scores

SAMPLE_CHUNKS = [
    {"chunk_id": "c1", "source": "election.csv", "text": "NDC won the Accra constituency with 12000 votes.", "keywords": ["ndc", "accra", "votes"], "year": 2024},
    {"chunk_id": "c2", "source": "budget.pdf", "text": "The 2025 budget allocates 3.2% of GDP to education.", "keywords": ["budget", "education", "gdp"], "year": 2025},
    {"chunk_id": "c3", "source": "election.csv", "text": "NPP secured 8000 votes in Kumasi North constituency.", "keywords": ["npp", "kumasi", "votes"], "year": 2024},
]


def test_vector_store_search():
    from src.retrieval.embedder import embed_texts, embed_query
    embeddings = embed_texts([c["text"] for c in SAMPLE_CHUNKS])
    vs = VectorStore()
    vs.build(embeddings, SAMPLE_CHUNKS)
    results = vs.search(embed_query("NDC election votes"), top_k=2)
    assert len(results) == 2
    assert results[0][1] >= results[1][1], "Results should be sorted by score descending"


def test_bm25_search():
    bm25 = BM25Retriever()
    bm25.build(SAMPLE_CHUNKS)
    results = bm25.search("NDC votes constituency", top_k=2)
    assert len(results) == 2
    assert results[0][1] >= results[1][1], "BM25 results should be ranked descending"


def test_scoring_merge():
    vector_results = [(SAMPLE_CHUNKS[0], 0.9), (SAMPLE_CHUNKS[2], 0.6)]
    bm25_results = [(SAMPLE_CHUNKS[0], 5.0), (SAMPLE_CHUNKS[1], 2.0)]
    scored = compute_final_scores("NDC votes", "election", vector_results, bm25_results)
    assert len(scored) > 0
    final_scores = [s[3] for s in scored]
    assert final_scores == sorted(final_scores, reverse=True), "Final scores should be sorted"


if __name__ == "__main__":
    test_bm25_search()
    test_scoring_merge()
    test_vector_store_search()
    print("All retrieval tests passed.")
