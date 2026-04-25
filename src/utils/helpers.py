# utils/helpers.py
# Author: [Your Name] | Index: [Your Index Number]
# Shared utility functions used across the pipeline

import re
from typing import List


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces and newlines into clean readable text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Simple keyword extractor: lowercase alphanum tokens, remove stopwords.
    Returns top_n most frequent meaningful words.
    """
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "was", "are", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall", "that",
        "this", "these", "those", "it", "its", "as", "if", "not", "no",
        "so", "we", "our", "their", "they", "he", "she", "his", "her",
        "which", "who", "whom", "than", "then", "also", "all", "any",
        "each", "more", "other", "such", "into", "through", "during",
        "before", "after", "above", "below", "up", "down", "out", "off",
        "over", "under", "again", "further",
    }
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    freq: dict = {}
    for tok in tokens:
        if tok not in stopwords:
            freq[tok] = freq.get(tok, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]


def count_words(text: str) -> int:
    return len(text.split())


def truncate_to_word_limit(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def deduplicate_chunks(chunks: List[dict]) -> List[dict]:
    """Remove duplicate chunks by text content."""
    seen = set()
    unique = []
    for chunk in chunks:
        key = chunk["text"][:200]
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique
