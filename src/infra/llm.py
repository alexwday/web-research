"""
LLM Client Module for Deep Research Agent
Uses OpenAI API with support for direct API key or OAuth2 token auth.
"""
import json
import os
import threading
import time
from typing import Optional, List, Dict, Any

from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import get_config, get_env_settings
from src.infra.oauth import fetch_oauth_token
from src.infra.security import configure_rbc_security_certs
from src.config.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# MODEL PRICING (per 1M tokens: input / output USD)
# =============================================================================

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini":   {"input": 0.15,  "output": 0.60},
    "gpt-4o":        {"input": 2.50,  "output": 10.00},
    "gpt-4.1-nano":  {"input": 0.10,  "output": 0.40},
    "gpt-4.1-mini":  {"input": 0.40,  "output": 1.60},
    "gpt-4.1":       {"input": 2.00,  "output": 8.00},
    "gpt-5-mini":    {"input": 0.25,  "output": 2.00},
    "gpt-5.1-mini":  {"input": 0.80,  "output": 3.20},
    "gpt-5":         {"input": 10.00, "output": 30.00},
    "gpt-5.1":       {"input": 3.00,  "output": 12.00},
    "gpt-5.2":       {"input": 10.00, "output": 30.00},
    "o3-mini":       {"input": 1.10,  "output": 4.40},
    "o3":            {"input": 10.00, "output": 40.00},
    "o4-mini":       {"input": 1.10,  "output": 4.40},
}


# =============================================================================
# TOKEN TRACKER
# =============================================================================

def _build_pricing_map() -> Dict[str, Dict[str, float]]:
    """Build model→pricing map from env vars, falling back to hardcoded defaults."""
    pricing = dict(MODEL_PRICING)
    try:
        settings = get_env_settings()
        config = get_config()
        for role in ("planner", "researcher", "writer", "editor", "analyzer"):
            model_name = getattr(config.llm.models, role)
            input_cost = getattr(settings, f"{role}_model_input_cost")
            output_cost = getattr(settings, f"{role}_model_output_cost")
            if input_cost is not None or output_cost is not None:
                pricing[model_name] = {
                    "input": input_cost or 0.0,
                    "output": output_cost or 0.0,
                }
    except Exception:
        pass
    return pricing


class TokenTracker:
    """Thread-safe, in-memory token usage and cost tracker."""

    def __init__(self):
        self._lock = threading.Lock()
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost = 0.0
        self._calls = 0
        self._pricing: Optional[Dict[str, Dict[str, float]]] = None

    def _get_pricing(self) -> Dict[str, Dict[str, float]]:
        if self._pricing is None:
            self._pricing = _build_pricing_map()
        return self._pricing

    def record(self, model: str, prompt_tokens: int, completion_tokens: int):
        cost = self._cost_for_model(model, prompt_tokens, completion_tokens)
        with self._lock:
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
            self._total_cost += cost
            self._calls += 1

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "prompt_tokens": self._total_prompt_tokens,
                "completion_tokens": self._total_completion_tokens,
                "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
                "total_cost": round(self._total_cost, 6),
                "calls": self._calls,
            }

    def _cost_for_model(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing_map = self._get_pricing()
        # Exact match first
        pricing = pricing_map.get(model)
        # Prefix match for versioned names like gpt-5.2-2025-12-11
        if pricing is None:
            for prefix, p in pricing_map.items():
                if model.startswith(prefix):
                    pricing = p
                    break
        if pricing is None:
            return 0.0
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


_token_tracker: Optional[TokenTracker] = None


def get_token_tracker() -> TokenTracker:
    """Get the singleton TokenTracker instance."""
    global _token_tracker
    if _token_tracker is None:
        _token_tracker = TokenTracker()
    return _token_tracker


# =============================================================================
# RATE LIMITER
# =============================================================================

class LLMRateLimiter:
    """Rate limiter for LLM API calls"""

    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute
        self.last_call = 0.0
        self._lock = threading.Lock()

    def wait(self):
        """Wait if necessary to respect rate limit"""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()


_llm_limiter: Optional[LLMRateLimiter] = None


def get_llm_limiter() -> LLMRateLimiter:
    global _llm_limiter
    if _llm_limiter is None:
        config = get_config()
        _llm_limiter = LLMRateLimiter(config.rate_limits.llm_calls_per_minute)
    return _llm_limiter


# =============================================================================
# OPENAI CLIENT
# =============================================================================

class OpenAIClient:
    """Client for OpenAI API (supports direct API key and OAuth2 token auth)"""

    def __init__(self):
        configure_rbc_security_certs()

        token, auth_info = fetch_oauth_token()
        base_url = os.getenv("AZURE_BASE_URL") or "https://api.openai.com/v1"

        logger.info(
            "OpenAI client init: auth=%s, base_url=%s",
            auth_info.get("method"), base_url,
        )

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=token, base_url=base_url)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    @staticmethod
    def _extract_content_text(response: Any) -> str:
        """Extract text content from a chat-completions response choice."""
        try:
            message = response.choices[0].message
        except Exception:
            return ""

        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if content is None:
            return ""

        # Defensive parsing for SDK variants that may return structured parts.
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                else:
                    text = getattr(part, "text", None)
                if text:
                    parts.append(str(text))
            return "\n".join(parts).strip()

        return str(content)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 4000,
        temperature: float = 0.3,
        json_mode: bool = False,
        model: str = None
    ) -> str:
        config = get_config()
        model = model or config.llm.models.researcher

        # Apply rate limiting
        get_llm_limiter().wait()

        logger.debug(f"OpenAI completion with model: {model}")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": model,
            "messages": messages,
        }

        # Reasoning and GPT-5 family models should use default temperature handling.
        # GPT-5 parameter compatibility differs by variant/reasoning mode.
        _no_temp = ("o1", "o3", "o4", "gpt-5")
        if not any(model.startswith(p) for p in _no_temp):
            kwargs["temperature"] = temperature

        # Newer models require max_completion_tokens instead of max_tokens
        if any(model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4", "gpt-4.1")):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
            if response.usage:
                get_token_tracker().record(
                    model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )
            text = self._extract_content_text(response)
            if not text.strip():
                finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
                logger.warning(
                    "OpenAI returned empty content (model=%s, finish_reason=%s)",
                    model,
                    finish_reason,
                )
            return text
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete_with_messages(
        self,
        messages: List[Dict[str, str]],
        system: str = None,
        max_tokens: int = 4000,
        temperature: float = 0.3,
        json_mode: bool = False,
        model: str = None
    ) -> str:
        config = get_config()
        model = model or config.llm.models.researcher

        # Apply rate limiting
        get_llm_limiter().wait()

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        kwargs = {
            "model": model,
            "messages": full_messages,
        }

        _no_temp = ("o1", "o3", "o4", "gpt-5")
        if not any(model.startswith(p) for p in _no_temp):
            kwargs["temperature"] = temperature

        if any(model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4", "gpt-4.1")):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
            if response.usage:
                get_token_tracker().record(
                    model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )
            text = self._extract_content_text(response)
            if not text.strip():
                finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
                logger.warning(
                    "OpenAI returned empty content (model=%s, finish_reason=%s)",
                    model,
                    finish_reason,
                )
            return text
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete_with_function(
        self,
        prompt: str,
        function_name: str,
        function_description: str,
        function_parameters: Dict[str, Any],
        system: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.2,
        model: str = None,
        require_tool_call: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Request a function/tool call and return parsed JSON arguments."""
        config = get_config()
        model = model or config.llm.models.researcher

        get_llm_limiter().wait()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": [{
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": function_description,
                    "parameters": function_parameters,
                }
            }],
        }
        if require_tool_call:
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": function_name},
            }
        else:
            kwargs["tool_choice"] = "auto"

        _no_temp = ("o1", "o3", "o4", "gpt-5")
        if not any(model.startswith(p) for p in _no_temp):
            kwargs["temperature"] = temperature

        if any(model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4", "gpt-4.1")):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        try:
            response = self.client.chat.completions.create(**kwargs)
            if response.usage:
                get_token_tracker().record(
                    model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            choice = response.choices[0]
            msg = choice.message
            tool_calls = getattr(msg, "tool_calls", None) or []

            for call in tool_calls:
                fn = getattr(call, "function", None)
                if fn is None:
                    continue
                if getattr(fn, "name", None) != function_name:
                    continue
                raw_args = getattr(fn, "arguments", None) or "{}"
                try:
                    parsed = json.loads(raw_args)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    logger.warning(
                        "Function-call args were not valid JSON (model=%s, function=%s): %r",
                        model, function_name, raw_args[:200]
                    )
                    return None

            # Some models may return JSON in content even when tools are provided.
            text = self._extract_content_text(response).strip()
            if text:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

            finish_reason = getattr(choice, "finish_reason", "unknown")
            logger.warning(
                "No usable function call returned (model=%s, function=%s, finish_reason=%s)",
                model, function_name, finish_reason,
            )
            return None
        except Exception as e:
            logger.error(f"OpenAI API error (function call): {e}")
            raise


# =============================================================================
# CLIENT FACTORY
# =============================================================================

_client: Optional[OpenAIClient] = None
_client_lock = threading.Lock()


def get_llm_client() -> OpenAIClient:
    """Get the OpenAI LLM client"""
    global _client

    if _client is None:
        with _client_lock:
            if _client is None:
                logger.info("Initializing OpenAI LLM client")
                _client = OpenAIClient()

    return _client


def reset_client():
    """Reset the global client instance"""
    global _client
    _client = None
