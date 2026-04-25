# preprocessing/chunking.py
# Author: [Your Name] | Index: [Your Index Number]
#
# Two chunking strategies for the PDF budget document:
# A. Fixed-size chunking (400 words, 80-word overlap)
# B. Paragraph-aware chunking (300-500 words, light sentence overlap)
#
# Design justification:
# - 400-word chunks are large enough to contain meaningful budget context
#   (a complete policy paragraph or table explanation) but small enough
#   to fit comfortably inside an embedding model's context window.
# - 80-word overlap prevents losing information that spans a chunk boundary.
# - Paragraph-aware chunks respect the document's logical structure, which
#   improves relevance scoring since paragraphs in budget documents often
#   correspond to single policy topics.

import re
from typing import List, Dict
from src.utils.helpers import extract_keywords, count_words
from src.preprocessing.clean_pdf import detect_section_title


# ─────────────────────────────────────────────
# Strategy A: Fixed-size chunking
# ─────────────────────────────────────────────

def fixed_size_chunks(
    pages: List[Dict],
    chunk_size: int = 400,
    overlap: int = 80,
    source: str = "2025_Budget_Statement.pdf",
) -> List[Dict]:
    """
    Concatenate all page text, then slice into fixed-size word windows
    with an overlap of `overlap` words between consecutive chunks.
    """
    # Combine all pages into one text stream, tracking page numbers
    full_text = " ".join(p["text"] for p in pages)
    words = full_text.split()

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        text = " ".join(chunk_words)

        chunk = {
            "chunk_id": f"pdf_fixed_{chunk_index}",
            "source": source,
            "chunk_type": "fixed_size",
            "text": text,
            "section_title": detect_section_title(text),
            "year": 2025,
            "keywords": extract_keywords(text),
            "word_count": len(chunk_words),
        }
        chunks.append(chunk)
        chunk_index += 1

        if end == len(words):
            break
        start += chunk_size - overlap  # slide window forward

    print(f"[Chunker] Fixed-size: {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


# ─────────────────────────────────────────────
# Strategy B: Paragraph-aware chunking
# ─────────────────────────────────────────────

def _split_into_paragraphs(text: str) -> List[str]:
    """Split text on double-newlines or blank lines."""
    paragraphs = re.split(r"\n{2,}|\r\n{2,}", text)
    return [p.strip() for p in paragraphs if len(p.strip()) > 30]


def paragraph_aware_chunks(
    pages: List[Dict],
    min_words: int = 300,
    max_words: int = 500,
    source: str = "2025_Budget_Statement.pdf",
) -> List[Dict]:
    """
    Group paragraphs until the chunk reaches the target word range.
    If a single paragraph exceeds max_words it is included as-is.
    Light overlap: the last sentence of a chunk is prepended to the next.
    """
    all_paragraphs: List[str] = []
    for page in pages:
        all_paragraphs.extend(_split_into_paragraphs(page["text"]))

    chunks = []
    current: List[str] = []
    current_words = 0
    chunk_index = 0
    carry_sentence = ""  # last sentence from previous chunk for overlap

    for para in all_paragraphs:
        para_words = count_words(para)

        # If adding this paragraph exceeds max_words and we already have content, flush
        if current_words + para_words > max_words and current_words >= min_words:
            text = (carry_sentence + " " + " ".join(current)).strip()
            chunks.append(_make_pdf_chunk(text, chunk_index, source, "paragraph_aware"))
            # Carry last sentence into next chunk
            sentences = re.split(r"(?<=[.!?])\s+", text)
            carry_sentence = sentences[-1] if sentences else ""
            chunk_index += 1
            current = []
            current_words = 0

        current.append(para)
        current_words += para_words

    # Flush remaining
    if current:
        text = (carry_sentence + " " + " ".join(current)).strip()
        chunks.append(_make_pdf_chunk(text, chunk_index, source, "paragraph_aware"))

    print(f"[Chunker] Paragraph-aware: {len(chunks)} chunks")
    return chunks


def _make_pdf_chunk(text: str, index: int, source: str, chunk_type: str) -> Dict:
    return {
        "chunk_id": f"pdf_para_{index}",
        "source": source,
        "chunk_type": chunk_type,
        "text": text,
        "section_title": detect_section_title(text),
        "year": 2025,
        "keywords": extract_keywords(text),
        "word_count": count_words(text),
    }
