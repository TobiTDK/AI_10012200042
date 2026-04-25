# retrieval/scoring.py
# Author: [Your Name] | Index: [Your Index Number]
#
# INNOVATION: Domain-specific weighted scoring function (Part G)
#
# Instead of relying on a single similarity measure, this function combines:
#   1. vector_score  — cosine similarity from FAISS (semantic relevance)
#   2. bm25_score    — BM25 keyword match score (lexical relevance)
#   3. source_match  — bonus if the query topic matches the chunk's data source
#   4. keyword_bonus — bonus for overlap between query keywords and chunk keywords
#   5. year_bonus    — bonus for chunks containing years mentioned in the query
#
# Weighting rationale:
#   - Vector similarity is the primary signal (0.50)
#   - BM25 helps with precise terms like candidate names and numeric figures (0.30)
#   - Source-match corrects cross-contamination (election query → budget chunk) (0.10)
#   - Keyword overlap reinforces domain relevance (0.07)
#   - Year bonus helps anchor date-specific facts (0.03)

import re
from typing import List, Dict, Tuple


WEIGHTS = {
    "vector": 0.50,
    "bm25": 0.30,
    "source_match": 0.10,
    "keyword_overlap": 0.07,
    "year_bonus": 0.03,
}

ELECTION_KEYWORDS = {"election", "vote", "votes", "constituency", "candidate", "party", "npp", "ndc", "results", "polling", "electoral", "parliament", "presidential"}
BUDGET_KEYWORDS = {"budget", "revenue", "expenditure", "gdp", "fiscal", "tax", "allocation", "ministry", "government", "spending", "deficit", "surplus", "ghana", "2025"}


def _normalize(scores: List[float]) -> List[float]:
    """Min-max normalize a list of scores to [0, 1]."""
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def _extract_years_from_query(query: str) -> List[int]:
    return [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", query)]


def _source_match_bonus(query_type: str, chunk_source: str) -> float:
    """Return 1.0 if chunk source aligns with detected query domain, else 0.0."""
    is_election_source = "election" in chunk_source.lower()
    is_budget_source = "budget" in chunk_source.lower()

    if query_type == "election" and is_election_source:
        return 1.0
    if query_type == "budget" and is_budget_source:
        return 1.0
    if query_type == "mixed":
        return 0.5  # no penalty for mixed queries
    return 0.0


def _keyword_overlap_bonus(query: str, chunk_keywords: List[str]) -> float:
    """
    Fraction of chunk keywords that also appear in the query.
    Capped at 1.0.
    """
    query_words = set(re.findall(r"[a-zA-Z]{3,}", query.lower()))
    if not chunk_keywords:
        return 0.0
    overlap = sum(1 for kw in chunk_keywords if kw in query_words)
    return min(overlap / max(len(chunk_keywords), 1), 1.0)


def _year_bonus(query: str, chunk_text: str) -> float:
    """1.0 if a year from the query also appears in the chunk."""
    query_years = _extract_years_from_query(query)
    if not query_years:
        return 0.0
    for year in query_years:
        if str(year) in chunk_text:
            return 1.0
    return 0.0


def domain_score(
    query: str,
    query_type: str,
    chunk: Dict,
    vector_score_norm: float,
    bm25_score_norm: float,
) -> float:
    """
    Compute the final weighted domain score for a single chunk.
    All individual components are in [0, 1] range.
    """
    source_bonus = _source_match_bonus(query_type, chunk.get("source", ""))
    kw_bonus = _keyword_overlap_bonus(query, chunk.get("keywords", []))
    yr_bonus = _year_bonus(query, chunk.get("text", ""))

    score = (
        WEIGHTS["vector"] * vector_score_norm
        + WEIGHTS["bm25"] * bm25_score_norm
        + WEIGHTS["source_match"] * source_bonus
        + WEIGHTS["keyword_overlap"] * kw_bonus
        + WEIGHTS["year_bonus"] * yr_bonus
    )
    return round(score, 4)


def compute_final_scores(
    query: str,
    query_type: str,
    vector_results: List[Tuple[Dict, float]],
    bm25_results: List[Tuple[Dict, float]],
) -> List[Tuple[Dict, float, float, float]]:
    """
    Merge vector and BM25 results, compute domain scores, and return
    a ranked list of (chunk, vector_score, bm25_score, final_score).
    """
    # Collect all unique chunks
    chunk_map: Dict[str, Dict] = {}
    vector_map: Dict[str, float] = {}
    bm25_map: Dict[str, float] = {}

    for chunk, score in vector_results:
        cid = chunk["chunk_id"]
        chunk_map[cid] = chunk
        vector_map[cid] = score

    for chunk, score in bm25_results:
        cid = chunk["chunk_id"]
        chunk_map[cid] = chunk
        bm25_map[cid] = score

    all_ids = list(chunk_map.keys())

    # Normalize raw scores independently
    raw_vec = [vector_map.get(cid, 0.0) for cid in all_ids]
    raw_bm25 = [bm25_map.get(cid, 0.0) for cid in all_ids]

    norm_vec = _normalize(raw_vec)
    norm_bm25 = _normalize(raw_bm25)

    results = []
    for i, cid in enumerate(all_ids):
        chunk = chunk_map[cid]
        v_norm = norm_vec[i]
        b_norm = norm_bm25[i]
        final = domain_score(query, query_type, chunk, v_norm, b_norm)
        results.append((chunk, raw_vec[i], raw_bm25[i], final))

    # Sort by final score descending
    results.sort(key=lambda x: x[3], reverse=True)
    return results
