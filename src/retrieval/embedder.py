# retrieval/embedder.py
# Author: [Your Name] | Index: [Your Index Number]
# Generates dense vector embeddings using sentence-transformers (all-MiniLM-L6-v2)

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        print(f"[Embedder] Loading model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of text strings.
    Returns a 2D numpy array of shape (n, embedding_dim).
    """
    model = get_model()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return np.array(embeddings, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.
    Returns a 1D numpy array.
    """
    model = get_model()
    embedding = model.encode([query], show_progress_bar=False)
    return np.array(embedding, dtype="float32")
