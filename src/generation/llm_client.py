# generation/llm_client.py
# Author: [Your Name] | Index: [Your Index Number]
# LLM calls for RAG and pure-LLM comparison.
# Supports OpenAI and Ollama Cloud. Cloud uses the OpenAI-compatible base
# https://ollama.com/v1 (see https://docs.ollama.com/cloud). Cloud model names
# are listed at https://ollama.com/search?c=cloud — use the exact tag in OLLAMA_MODEL.

import os
from typing import Literal, Tuple

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client: OpenAI | None = None
_client_key: Tuple[str, str, str] | None = None  # (provider, base_url, api_key_snippet)

# OpenAI-compatible base for ollama.com (chat/completions)
DEFAULT_OLLAMA_CLOUD_BASE = "https://ollama.com/v1"
# Ollama *Cloud* API model tag (see https://ollama.com/search?c=cloud). Local tags like
# llama3.1:8b work only with local Ollama + OLLAMA_BASE_URL=http://127.0.0.1:11434 — not on ollama.com.
DEFAULT_OLLAMA_CLOUD_MODEL = "gpt-oss:120b"


def _ollama_api_base() -> str:
    """
    API base for Ollama Cloud (must end in /v1 for the OpenAI client).
    Precedence: OLLAMA_CLOUD_BASE_URL > OLLAMA_BASE_URL > default.
    """
    raw = (
        (os.getenv("OLLAMA_CLOUD_BASE_URL") or "").strip()
        or (os.getenv("OLLAMA_BASE_URL") or "").strip()
        or DEFAULT_OLLAMA_CLOUD_BASE
    )
    base = raw.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _resolve_provider() -> Literal["openai", "ollama_cloud"]:
    explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("openai", "ollama_cloud"):
        return explicit  # type: ignore[return-value]
    ollama_key = (os.getenv("OLLAMA_API_KEY") or "").strip()
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    # If both keys exist: prefer Ollama unless USE_OLLAMA=0|no|false (prefer OpenAI)
    if ollama_key and openai_key:
        uo = (os.getenv("USE_OLLAMA") or "1").strip().lower()
        if uo in ("0", "false", "no"):
            return "openai"
        return "ollama_cloud"
    if ollama_key:
        return "ollama_cloud"
    if openai_key:
        return "openai"
    raise EnvironmentError(
        "No LLM credentials found. Set one of:\n"
        "  • OLLAMA_API_KEY — Ollama Cloud (https://ollama.com/settings/keys), optional LLM_PROVIDER=ollama_cloud\n"
        "  • OPENAI_API_KEY — OpenAI, optional LLM_PROVIDER=openai\n"
        "Or set LLM_PROVIDER explicitly together with the matching key."
    )


def _default_model(provider: Literal["openai", "ollama_cloud"]) -> str:
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_CLOUD_MODEL).strip()


def _get_client() -> OpenAI:
    global _client, _client_key
    provider = _resolve_provider()

    if provider == "ollama_cloud":
        api_key = (os.getenv("OLLAMA_API_KEY") or "").strip()
        if not api_key:
            raise EnvironmentError(
                "LLM_PROVIDER is ollama_cloud (or OLLAMA_API_KEY is set) but OLLAMA_API_KEY is empty."
            )
        base = _ollama_api_base()
        key_sig = (provider, base, (api_key[:8] if len(api_key) >= 8 else api_key) + "...")
    else:
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set. Check your .env file.")
        base = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
        key_sig = (provider, base or "default", (api_key[:8] if len(api_key) >= 8 else api_key) + "...")

    if _client is not None and _client_key == key_sig:
        return _client

    if provider == "ollama_cloud":
        _client = OpenAI(base_url=f"{base}/", api_key=api_key)
    else:
        kwargs = {"api_key": api_key}
        if base:
            kwargs["base_url"] = base
        _client = OpenAI(**kwargs)
    _client_key = key_sig
    return _client


def get_llm_provider_label() -> str:
    """For UI / logging: human-readable active backend."""
    try:
        p = _resolve_provider()
    except EnvironmentError:
        return "not configured"
    if p == "ollama_cloud":
        m = _default_model("ollama_cloud")
        return f"Ollama Cloud ({m})"
    return f"OpenAI ({_default_model('openai')})"


def generate_response(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:
    """
    Send a constructed RAG prompt to the configured API and return the response text.
    Low temperature (0.2) for factual, consistent answers.
    """
    provider = _resolve_provider()
    model = model or _default_model(provider)
    client = _get_client()
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise academic assistant. "
                        "Follow all instructions in the user prompt strictly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = completion.choices[0].message.content
        return (content or "").strip()
    except Exception as e:
        err = str(e)
        hint = ""
        if "not found" in err.lower() or "404" in err:
            hint = (
                " Set OLLAMA_MODEL to a cloud model from https://ollama.com/search?c=cloud "
                "or run `curl https://ollama.com/api/tags`. "
                "Local-only names (e.g. llama3.1:8b) need local Ollama, not ollama.com."
            )
        return f"[LLM ERROR] {err}{hint}"


def generate_pure_llm(query: str, model: str | None = None) -> str:
    """
    Generate an answer WITHOUT retrieval context.
    Used in evaluation to compare RAG vs. pure LLM.
    """
    provider = _resolve_provider()
    model = model or _default_model(provider)
    client = _get_client()
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.3,
            max_tokens=600,
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledgeable assistant about Ghana politics and economics.",
                },
                {"role": "user", "content": query},
            ],
        )
        content = completion.choices[0].message.content
        return (content or "").strip()
    except Exception as e:
        err = str(e)
        hint = ""
        if "not found" in err.lower() or "404" in err:
            hint = (
                " Set OLLAMA_MODEL to a cloud model from https://ollama.com/search?c=cloud "
                "or run `curl https://ollama.com/api/tags`. "
                "Local-only names (e.g. llama3.1:8b) need local Ollama, not ollama.com."
            )
        return f"[LLM ERROR] {err}{hint}"
