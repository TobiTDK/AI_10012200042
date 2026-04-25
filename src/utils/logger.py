# utils/logger.py
# Author: [Your Name] | Index: [Your Index Number]
# Logs every stage of the RAG pipeline to outputs/logs.json

import json
import os
from datetime import datetime
from typing import Any, Dict, List

LOG_PATH = os.path.join(os.path.dirname(__file__), "../../outputs/logs.json")


def _load_logs() -> List[Dict]:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def _save_logs(logs: List[Dict]) -> None:
    with open(LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2, default=str)


def log_query_event(
    query: str,
    query_type: str,
    retrieved_chunks: List[Dict],
    vector_scores: List[float],
    bm25_scores: List[float],
    final_scores: List[float],
    selected_context: str,
    final_prompt: str,
    response: str,
) -> None:
    """Append a full pipeline event to the log file."""
    logs = _load_logs()
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "query_type": query_type,
        "retrieved_chunks": retrieved_chunks,
        "vector_scores": vector_scores,
        "bm25_scores": bm25_scores,
        "final_scores": final_scores,
        "selected_context_preview": selected_context[:500],
        "final_prompt_preview": final_prompt[:1000],
        "response": response,
    }
    logs.append(event)
    _save_logs(logs)


def get_all_logs() -> List[Dict]:
    return _load_logs()
