# preprocessing/clean_csv.py
# Author: [Your Name] | Index: [Your Index Number]
# Cleans election CSV data and converts rows to natural-language text chunks

import pandas as pd
from typing import List, Dict
from src.utils.helpers import normalize_whitespace, extract_keywords


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the election results DataFrame:
    - Drop rows where all values are null
    - Strip whitespace from string columns
    - Fill missing numeric values with 0
    """
    df = df.dropna(how="all")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(0)
    df = df.reset_index(drop=True)
    return df


def row_to_text(row: pd.Series, columns: List[str]) -> str:
    """
    Convert a single DataFrame row to a readable natural-language sentence.
    E.g.: "Constituency: Ablekuma North | Candidate: John Doe | Party: NPP | Votes: 12345"
    """
    parts = []
    for col in columns:
        val = row.get(col, "")
        if str(val).strip() and str(val).lower() not in ("nan", "none", ""):
            parts.append(f"{col}: {val}")
    return " | ".join(parts)


def csv_to_chunks(df: pd.DataFrame, source: str = "Ghana_Election_Result.csv") -> List[Dict]:
    """
    Convert each cleaned CSV row into a chunk dict with metadata.
    Returns a list of chunk dicts.
    """
    df = clean_dataframe(df)
    columns = list(df.columns)
    chunks = []

    for idx, row in df.iterrows():
        text = row_to_text(row, columns)
        text = normalize_whitespace(text)
        if not text:
            continue

        # Try to extract year if present
        year = None
        for col in df.columns:
            val = str(row.get(col, ""))
            if val.isdigit() and 1990 <= int(val) <= 2030:
                year = int(val)
                break

        chunk = {
            "chunk_id": f"csv_{idx}",
            "source": source,
            "chunk_type": "csv_row",
            "text": text,
            "section_title": None,
            "year": year,
            "keywords": extract_keywords(text),
            "row_index": idx,
        }
        chunks.append(chunk)

    print(f"[CSV Cleaner] Generated {len(chunks)} chunks from CSV")
    return chunks
