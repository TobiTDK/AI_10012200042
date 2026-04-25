# pipeline/rag_pipeline.py
# Author: [Your Name] | Index: [Your Index Number]
#
# Full RAG Pipeline:
# User Query → Classify → Retrieve (Hybrid) → Score → Select Context
#            → Build Prompt → Generate (LLM) → Log → Return Response
#
# This module initialises and wires all components together.
# It is used by app.py (Streamlit UI) and the evaluation scripts.

import json
import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from src.ingestion.load_csv import load_csv
from src.ingestion.load_pdf import load_pdf
from src.preprocessing.clean_csv import csv_to_chunks
from src.preprocessing.clean_pdf import clean_pages
from src.preprocessing.chunking import fixed_size_chunks, paragraph_aware_chunks
from src.retrieval.embedder import embed_texts
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.prompt_builder import build_prompt
from src.generation.llm_client import generate_response, generate_pure_llm
from src.utils.logger import log_query_event
from src.utils.helpers import deduplicate_chunks

# Paths (relative to project root)
_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
CSV_PATH = os.path.join(_DATA_DIR, "Ghana_Election_Result.csv")
# Accept common filenames (course handout vs. local download)
_BUDGET_PDF_NAMES = ("2025_Budget_Statement.pdf", "2025_budget.pdf")


def _resolve_budget_pdf_path() -> str:
    for name in _BUDGET_PDF_NAMES:
        p = os.path.join(_DATA_DIR, name)
        if os.path.isfile(p):
            return p
    return os.path.join(_DATA_DIR, _BUDGET_PDF_NAMES[0])


PDF_PATH = _resolve_budget_pdf_path()
CHUNKS_OUTPUT = os.path.join(os.path.dirname(__file__), "../../outputs/chunks.json")


class RAGPipeline:
    """
    Singleton-style pipeline class.
    Call .initialize() once at startup, then .query() for each user question.
    """

    def __init__(self):
        self.chunks: List[Dict] = []
        self.vector_store = VectorStore()
        self.bm25 = BM25Retriever()
        self.hybrid: Optional[HybridRetriever] = None
        self.ready = False
        self.chunking_method = "fixed"   # "fixed" or "paragraph"

    # ─────────────────────────────────────────────
    # Stage 1: Ingest and chunk documents
    # ─────────────────────────────────────────────

    def _load_and_chunk(self, chunking_method: str = "fixed") -> List[Dict]:
        all_chunks: List[Dict] = []

        # CSV → chunks
        df = load_csv(CSV_PATH)
        if df is not None:
            csv_chunks = csv_to_chunks(df)
            all_chunks.extend(csv_chunks)

        # PDF → pages → cleaned → chunks
        pages = load_pdf(PDF_PATH)
        if pages is not None:
            cleaned = clean_pages(pages)
            pdf_source_label = os.path.basename(PDF_PATH)
            if chunking_method == "paragraph":
                pdf_chunks = paragraph_aware_chunks(cleaned, source=pdf_source_label)
            else:
                pdf_chunks = fixed_size_chunks(cleaned, source=pdf_source_label)
            all_chunks.extend(pdf_chunks)

        # Deduplicate
        all_chunks = deduplicate_chunks(all_chunks)
        print(f"[Pipeline] Total chunks after dedup: {len(all_chunks)}")
        return all_chunks

    # ─────────────────────────────────────────────
    # Stage 2: Embed and index
    # ─────────────────────────────────────────────

    def _embed_and_index(self) -> None:
        texts = [c["text"] for c in self.chunks]
        if not texts:
            raise RuntimeError(
                "No document chunks available to index. "
                f"Expected data files under `{os.path.abspath(_DATA_DIR)}`. "
                "Add `Ghana_Election_Result.csv` and a budget PDF "
                "(`2025_budget.pdf` or `2025_Budget_Statement.pdf`) before running."
            )
        print(f"[Pipeline] Embedding {len(texts)} chunks...")
        embeddings = embed_texts(texts)

        self.vector_store.build(embeddings, self.chunks)
        self.bm25.build(self.chunks)
        self.hybrid = HybridRetriever(self.vector_store, self.bm25)
        print("[Pipeline] FAISS and BM25 indexes ready.")

    # ─────────────────────────────────────────────
    # Public: Initialize
    # ─────────────────────────────────────────────

    def initialize(self, chunking_method: str = "fixed") -> None:
        """
        Load data, chunk, embed, and build indexes.
        Call once before running queries.
        """
        self.chunking_method = chunking_method
        self.chunks = self._load_and_chunk(chunking_method)

        # Save chunks for inspection
        os.makedirs(os.path.dirname(CHUNKS_OUTPUT), exist_ok=True)
        with open(CHUNKS_OUTPUT, "w") as f:
            json.dump(self.chunks, f, indent=2, default=str)

        self._embed_and_index()
        self.ready = True
        print("[Pipeline] Initialization complete. Ready to answer queries.")

    # ─────────────────────────────────────────────
    # Public: Query
    # ─────────────────────────────────────────────

    def query(
        self,
        user_query: str,
        top_k: int = 4,
        prompt_version: str = "v3",
    ) -> Dict:
        """
        Run the full RAG pipeline for a user query.

        Returns a dict with all pipeline artifacts:
            - query_type
            - retrieved_chunks (list of chunk dicts)
            - vector_scores, bm25_scores, final_scores
            - selected_chunks (after context filtering)
            - final_prompt
            - response
        """
        if not self.ready:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # Step 1: Hybrid retrieval
        query_type, scored = self.hybrid.retrieve(user_query, top_k=top_k, fetch_k=top_k * 3)

        # Step 2: Build prompt (context selection happens inside)
        prompt, selected_chunks = build_prompt(user_query, scored, version=prompt_version)

        # Step 3: Generate response
        response = generate_response(prompt)

        # Step 4: Extract score arrays for logging/display
        retrieved_chunks = [s[0] for s in scored]
        vector_scores = [round(s[1], 4) for s in scored]
        bm25_scores = [round(s[2], 4) for s in scored]
        final_scores = [round(s[3], 4) for s in scored]

        selected_context = "\n\n".join(c["text"][:300] for c in selected_chunks)

        # Step 5: Log
        log_query_event(
            query=user_query,
            query_type=query_type,
            retrieved_chunks=[{"chunk_id": c["chunk_id"], "source": c["source"]} for c in retrieved_chunks],
            vector_scores=vector_scores,
            bm25_scores=bm25_scores,
            final_scores=final_scores,
            selected_context=selected_context,
            final_prompt=prompt,
            response=response,
        )

        return {
            "query": user_query,
            "query_type": query_type,
            "retrieved_chunks": retrieved_chunks,
            "scored": scored,
            "vector_scores": vector_scores,
            "bm25_scores": bm25_scores,
            "final_scores": final_scores,
            "selected_chunks": selected_chunks,
            "final_prompt": prompt,
            "response": response,
        }

    def query_pure_llm(self, user_query: str) -> str:
        """Generate answer without RAG context (for comparison)."""
        return generate_pure_llm(user_query)
