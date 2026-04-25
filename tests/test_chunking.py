# tests/test_chunking.py
# Author: [Your Name] | Index: [Your Index Number]

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.chunking import fixed_size_chunks, paragraph_aware_chunks

SAMPLE_PAGES = [
    {"page": 1, "text": "The government of Ghana presented the 2025 budget statement. " * 60},
    {"page": 2, "text": "Revenue projections for 2025 include tax receipts. " * 40},
]


def test_fixed_size_chunks_count():
    chunks = fixed_size_chunks(SAMPLE_PAGES, chunk_size=400, overlap=80)
    assert len(chunks) > 0, "Should produce at least one chunk"


def test_fixed_size_chunk_metadata():
    chunks = fixed_size_chunks(SAMPLE_PAGES, chunk_size=400, overlap=80)
    for c in chunks:
        assert "chunk_id" in c
        assert "source" in c
        assert "text" in c
        assert "keywords" in c


def test_paragraph_chunks_count():
    chunks = paragraph_aware_chunks(SAMPLE_PAGES, min_words=50, max_words=200)
    assert len(chunks) > 0


def test_chunk_word_count():
    chunks = fixed_size_chunks(SAMPLE_PAGES, chunk_size=400, overlap=80)
    for c in chunks[:-1]:  # last chunk may be shorter
        assert c["word_count"] <= 400


if __name__ == "__main__":
    test_fixed_size_chunks_count()
    test_fixed_size_chunk_metadata()
    test_paragraph_chunks_count()
    test_chunk_word_count()
    print("All chunking tests passed.")
