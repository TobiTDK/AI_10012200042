# ingestion/load_pdf.py
# Author: [Your Name] | Index: [Your Index Number]
# Loads raw text from a PDF using PyMuPDF (fitz)

from typing import List, Dict, Optional
import fitz  # PyMuPDF


def load_pdf(filepath: str) -> Optional[List[Dict]]:
    """
    Load a PDF and return a list of page dicts with page number and text.
    Each page is: {"page": int, "text": str}
    Returns None on failure.
    """
    try:
        doc = fitz.open(filepath)
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            pages.append({
                "page": page_num + 1,
                "text": text,
            })
        doc.close()
        print(f"[PDF Loader] Loaded {len(pages)} pages from {filepath}")
        return pages
    except FileNotFoundError:
        print(f"[PDF Loader] ERROR: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"[PDF Loader] ERROR: {e}")
        return None
