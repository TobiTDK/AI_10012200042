# retrieval/vector_store.py
# Author: [Your Name] | Index: [Your Index Number]
# Builds and queries a FAISS index for dense vector similarity search

import faiss
import numpy as np
from typing import List, Tuple, Dict


class VectorStore:
    """
    Wraps a FAISS flat L2 index.
    Stores chunk metadata alongside vectors for retrieval.
    """

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None  # Inner-product (cosine after normalizing)
        self.chunks: List[Dict] = []
        self.dim: int = 0

    def build(self, embeddings: np.ndarray, chunks: List[Dict]) -> None:
        """
        Build the FAISS index from a batch of embedding vectors.
        Uses IndexFlatIP (inner product) after L2-normalising vectors
        so that inner product == cosine similarity.
        """
        if embeddings.size == 0 or embeddings.ndim != 2:
            raise ValueError(
                "Embeddings are empty or invalid. Ensure documents were loaded and chunked before indexing."
            )
        assert len(embeddings) == len(chunks), "Embedding count must match chunk count"
        self.dim = embeddings.shape[1]
        self.chunks = chunks

        # Normalize vectors so cosine similarity = dot product
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        print(f"[VectorStore] Built FAISS index with {self.index.ntotal} vectors (dim={self.dim})")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for top_k nearest chunks.
        Returns list of (chunk_dict, cosine_score) tuples, sorted by score descending.
        """
        if self.index is None:
            raise RuntimeError("VectorStore is not built. Call build() first.")

        query = query_vector.copy().reshape(1, -1).astype("float32")
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results
