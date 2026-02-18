"""Tavily search wrapper with rate limiting."""
import threading
import time
from typing import Any, Dict, List, Optional

from src.config.logger import get_logger
from src.config.settings import get_config, get_env_settings

logger = get_logger(__name__)


class RateLimiter:
    """Simple thread-safe call limiter."""

    def __init__(self, calls_per_minute: int):
        self.interval = 60.0 / max(1, calls_per_minute)
        self.last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()


class TavilySearchTool:
    """Wrapper around TavilyClient.search."""

    def __init__(self, client: Any = None):
        self._client = client
        self._limiter: Optional[RateLimiter] = None

    @property
    def limiter(self) -> RateLimiter:
        if self._limiter is None:
            cfg = get_config()
            self._limiter = RateLimiter(cfg.search.search_calls_per_minute)
        return self._limiter

    def _get_client(self):
        if self._client is not None:
            return self._client

        settings = get_env_settings()
        if not settings.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is not configured")

        from tavily import TavilyClient

        self._client = TavilyClient(api_key=settings.tavily_api_key)
        return self._client

    def search(
        self,
        query: str,
        max_results: int,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        cfg = get_config()
        self.limiter.wait()

        try:
            response = self._get_client().search(
                query,
                search_depth=cfg.search.depth,
                max_results=max_results,
                include_domains=include_domains or None,
                exclude_domains=exclude_domains or None,
                include_raw_content=True,
            )
        except Exception as exc:
            logger.warning("Tavily search error for query '%s': %s", query, exc)
            return []

        results: List[Dict[str, Any]] = []
        for item in response.get("results", []):
            results.append(
                {
                    "query": query,
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("content", ""),
                    "raw_content": item.get("raw_content", ""),
                    "score": item.get("score", 0.0),
                    "published_date": item.get("published_date") or item.get("published"),
                }
            )

        return results

    def extract(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Extract raw content from URLs using Tavily Extract API."""
        if not urls:
            return []
        self.limiter.wait()
        try:
            response = self._get_client().extract(urls)
            return response.get("results", [])
        except Exception as exc:
            logger.warning("Tavily extract error for %d URLs: %s", len(urls), exc)
            return []
