"""Lightweight OpenAI client for LLM-powered document ranking.

Auth priority: OPENAI_API_KEY env var > OAuth client credentials > None.
When no credentials are available, get_llm_client() returns None so callers
can gracefully fall back to heuristic-only ranking.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.logger import get_logger
from src.infra.security import configure_rbc_security_certs

logger = get_logger(__name__)


def _fetch_token() -> tuple[Optional[str], Optional[str]]:
    """Return (token, base_url) or (None, None) when no credentials exist."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        base_url = os.getenv("AZURE_BASE_URL") or "https://api.openai.com/v1"
        logger.info("LLM auth: using OPENAI_API_KEY (base_url=%s)", base_url)
        return api_key, base_url

    oauth_url = os.getenv("OAUTH_URL", "")
    client_id = os.getenv("CLIENT_ID", "")
    client_secret = os.getenv("CLIENT_SECRET", "")

    if not all([oauth_url, client_id, client_secret]):
        logger.info("LLM auth: no credentials available (OPENAI_API_KEY / OAuth)")
        return None, None

    import requests

    logger.info("LLM auth: using OAuth2 client credentials flow")
    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    try:
        resp = requests.post(oauth_url, data=payload, timeout=30)
        resp.raise_for_status()
        token = resp.json().get("access_token")
        if not token:
            logger.warning("OAuth response missing access_token")
            return None, None
        base_url = os.getenv("AZURE_BASE_URL") or "https://api.openai.com/v1"
        return str(token), base_url
    except Exception as exc:
        logger.warning("OAuth token fetch failed: %s", exc)
        return None, None


class LLMClient:
    """Thin OpenAI chat-completions wrapper with JSON mode support."""

    def __init__(self, token: str, base_url: str):
        configure_rbc_security_certs()
        from openai import OpenAI

        self._client = OpenAI(api_key=token, base_url=base_url)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        max_tokens: int = 2000,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {"model": model, "messages": messages}

        # Reasoning / GPT-5 family: skip explicit temperature
        _no_temp = ("o1", "o3", "o4", "gpt-5")
        if not any(model.startswith(p) for p in _no_temp):
            kwargs["temperature"] = temperature

        # Newer models require max_completion_tokens
        if any(model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4", "gpt-4.1")):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        text = self._extract_text(response)
        if not text.strip():
            logger.warning("LLM returned empty content (model=%s)", model)
        return text

    @staticmethod
    def _extract_text(response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except Exception:
            return ""
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if isinstance(content, list):
            parts = []
            for part in content:
                t = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                if t:
                    parts.append(str(t))
            return "\n".join(parts).strip()
        return str(content)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_client: Optional[LLMClient] = None
_client_lock = threading.Lock()


def get_llm_client() -> Optional[LLMClient]:
    """Return a shared LLMClient, or None when credentials are unavailable."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                token, base_url = _fetch_token()
                if token is None:
                    return None
                try:
                    _client = LLMClient(token, base_url)
                except Exception as exc:
                    logger.warning("Failed to initialize LLM client: %s", exc)
                    return None
    return _client
