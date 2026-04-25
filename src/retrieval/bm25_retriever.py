# retrieval/bm25_retriever.py
# Author: [Your Name] | Index: [Your Index Number]
# BM25 keyword-based retriever using the rank-bm25 library

from rank_bm25 import BM25Okapi
from typing import List, Tuple, Dict
import re


def _tokenize(text: str) -> List[str]:
    """Simple lowercase tokenizer."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class BM25Retriever:
    """
    Keyword-based retriever. Complements FAISS by catching exact keyword matches
    that dense embeddings sometimes miss (e.g., specific candidate names, numbers).
    """

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: List[Dict] = []

    def build(self, chunks: List[Dict]) -> None:
        """Build BM25 index from chunk texts."""
        self.chunks = chunks
        tokenized = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"[BM25] Built index over {len(chunks)} chunks")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Return top_k chunks ranked by BM25 score.
        Scores are raw BM25 values (not normalized here — normalization happens in hybrid retriever).
        """
        if self.bm25 is None:
            raise RuntimeError("BM25Retriever not built. Call build() first.")

        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Pair each chunk with its score and sort
        scored = list(zip(self.chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
