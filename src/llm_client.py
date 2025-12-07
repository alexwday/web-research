"""
LLM Client Module for Deep Research Agent
Supports Anthropic, OpenAI, and OpenRouter
"""
import json
import time
from typing import Optional, List, Dict, Any, Generator
from abc import ABC, abstractmethod

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_config, get_env_settings, LLMProvider
from .logger import get_logger

logger = get_logger(__name__)


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
# BASE CLIENT
# =============================================================================

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 4000,
        temperature: float = 0.3,
        json_mode: bool = False
    ) -> str:
        """Generate a completion"""
        pass
    
    @abstractmethod
    def complete_with_messages(
        self,
        messages: List[Dict[str, str]],
        system: str = None,
        max_tokens: int = 4000,
        temperature: float = 0.3,
        json_mode: bool = False
    ) -> str:
        """Generate a completion with message history"""
        pass


# =============================================================================
# ANTHROPIC CLIENT
# =============================================================================

class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude API"""
    
    def __init__(self):
        settings = get_env_settings()
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
    
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
        
        logger.debug(f"Anthropic completion with model: {model}")
        
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Modify system prompt for JSON mode
        if json_mode and system:
            system = system + "\n\nYou MUST respond with valid JSON only. No other text."
        elif json_mode:
            system = "You MUST respond with valid JSON only. No other text."
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "You are a helpful research assistant.",
                messages=messages
            )
            
            content = response.content[0].text
            
            # Validate JSON if needed
            if json_mode:
                content = self._extract_json(content)
            
            return content
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
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
        
        # Modify system prompt for JSON mode
        if json_mode and system:
            system = system + "\n\nYou MUST respond with valid JSON only. No other text."
        elif json_mode:
            system = "You MUST respond with valid JSON only. No other text."
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "You are a helpful research assistant.",
                messages=messages
            )
            
            content = response.content[0].text
            
            if json_mode:
                content = self._extract_json(content)
            
            return content
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def _extract_json(self, content: str) -> str:
        """Extract JSON from response, handling markdown code blocks"""
        content = content.strip()
        
        # Remove markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        return content.strip()


# =============================================================================
# OPENAI CLIENT
# =============================================================================

class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API"""
    
    def __init__(self):
        settings = get_env_settings()
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.openai_api_key)
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
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        try:
            response = self.client.chat.completions.create(**kwargs)
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
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


# =============================================================================
# OPENROUTER CLIENT
# =============================================================================

class OpenRouterClient(BaseLLMClient):
    """Client for OpenRouter API"""
    
    def __init__(self):
        settings = get_env_settings()
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=settings.openrouter_base_url,
                api_key=settings.openrouter_api_key
            )
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
        
        # OpenRouter uses different model naming
        if not "/" in model:
            model = f"anthropic/{model}"
        
        # Apply rate limiting
        get_llm_limiter().wait()
        
        logger.debug(f"OpenRouter completion with model: {model}")
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        try:
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            
            if json_mode:
                content = self._extract_json(content)
            
            return content
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
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
        
        if not "/" in model:
            model = f"anthropic/{model}"
        
        # Apply rate limiting
        get_llm_limiter().wait()
        
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)
        
        kwargs = {
            "model": model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        try:
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            
            if json_mode:
                content = self._extract_json(content)
            
            return content
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
    
    def _extract_json(self, content: str) -> str:
        """Extract JSON from response"""
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()


# =============================================================================
# CLIENT FACTORY
# =============================================================================

_client: Optional[BaseLLMClient] = None


def get_llm_client() -> BaseLLMClient:
    """Get the configured LLM client"""
    global _client
    
    if _client is None:
        config = get_config()
        provider = config.llm.provider
        
        logger.info(f"Initializing LLM client: {provider}")
        
        if provider == LLMProvider.ANTHROPIC:
            _client = AnthropicClient()
        elif provider == LLMProvider.OPENAI:
            _client = OpenAIClient()
        elif provider == LLMProvider.OPENROUTER:
            _client = OpenRouterClient()
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    return _client


def reset_client():
    """Reset the global client instance"""
    global _client
    _client = None
