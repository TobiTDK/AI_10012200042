# ingestion/load_csv.py
# Author: [Your Name] | Index: [Your Index Number]
# Loads the Ghana Election Results CSV using pandas

import pandas as pd
from typing import Optional


def load_csv(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load the Ghana Election Results CSV.
    Returns a DataFrame or None on failure.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[CSV Loader] Loaded {len(df)} rows from {filepath}")
        return df
    except FileNotFoundError:
        print(f"[CSV Loader] ERROR: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"[CSV Loader] ERROR: {e}")
        return None
