"""
LLM Client Module for Deep Research Agent
Uses OpenAI API with support for direct API key or OAuth2 token auth.
"""
import os
import threading
import time
from typing import Optional, List, Dict

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_config, get_env_settings
from .oauth import fetch_oauth_token
from .utils.rbc_security import configure_rbc_security_certs
from .logger import get_logger

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
    "gpt-5-mini":    {"input": 1.25,  "output": 5.00},
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
        for role in ("planner", "researcher", "writer", "editor"):
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

    def wait(self):
        """Wait if necessary to respect rate limit"""
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

        # Models that only accept default temperature (reasoning & gpt-5-mini/nano)
        _no_temp = ("o1", "o3", "o4", "gpt-5-mini", "gpt-5-nano")
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
            return response.choices[0].message.content
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

        _no_temp = ("o1", "o3", "o4", "gpt-5-mini", "gpt-5-nano")
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
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
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
