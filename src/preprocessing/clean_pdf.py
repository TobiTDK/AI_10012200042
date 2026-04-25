# preprocessing/clean_pdf.py
# Author: [Your Name] | Index: [Your Index Number]
# Cleans raw PDF page text: removes junk, normalizes spacing

import re
from typing import List, Dict
from src.utils.helpers import normalize_whitespace


def clean_page_text(text: str) -> str:
    """
    Clean a single PDF page's extracted text:
    - Remove excessive line breaks
    - Remove page numbers standing alone
    - Normalize whitespace
    """
    # Remove isolated page numbers (e.g., lines that are just a number)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    # Remove header/footer artifacts (short repeated lines)
    lines = text.split("\n")
    lines = [l for l in lines if len(l.strip()) > 3]
    text = "\n".join(lines)
    return normalize_whitespace(text)


def clean_pages(pages: List[Dict]) -> List[Dict]:
    """
    Apply cleaning to all pages returned by load_pdf.
    Returns the same structure with cleaned text.
    """
    cleaned = []
    for page in pages:
        cleaned_text = clean_page_text(page["text"])
        if cleaned_text:
            cleaned.append({
                "page": page["page"],
                "text": cleaned_text,
            })
    print(f"[PDF Cleaner] Cleaned {len(cleaned)} non-empty pages")
    return cleaned


def detect_section_title(text: str) -> str | None:
    """
    Heuristic: if a line is short (< 80 chars), all-caps or title-cased, treat it
    as a section heading.
    """
    lines = text.split()[:12]  # look at start of chunk
    candidate = " ".join(lines)
    if len(candidate) < 100 and (candidate.isupper() or candidate.istitle()):
        return candidate
    return None
