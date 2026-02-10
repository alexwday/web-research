"""Web search and rate limiting."""
import time
import threading
from typing import Optional, List, Dict, Any

from src.config.settings import get_config, get_env_settings
from src.config.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """Simple rate limiter for API calls"""

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


# Rate limiters
_search_limiter: Optional[RateLimiter] = None
_scrape_limiter: Optional[RateLimiter] = None


def get_search_limiter() -> RateLimiter:
    global _search_limiter
    if _search_limiter is None:
        config = get_config()
        _search_limiter = RateLimiter(config.rate_limits.search_calls_per_minute)
    return _search_limiter


def get_scrape_limiter() -> RateLimiter:
    global _scrape_limiter
    if _scrape_limiter is None:
        config = get_config()
        _scrape_limiter = RateLimiter(config.rate_limits.scrape_requests_per_minute)
    return _scrape_limiter


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

def search_tavily(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search using Tavily API with full content extraction"""
    settings = get_env_settings()
    config = get_config()

    if not settings.tavily_api_key:
        raise ValueError("TAVILY_API_KEY not set in environment")

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)

        # Apply rate limiting
        get_search_limiter().wait()

        logger.debug(f"Searching Tavily: {query}")

        response = client.search(
            query,
            search_depth=config.search.depth,
            max_results=max_results,
            include_domains=config.search.include_domains or None,
            exclude_domains=config.search.exclude_domains or None,
            include_raw_content=True  # Get full page content
        )

        results = []
        for r in response.get('results', []):
            results.append({
                'url': r.get('url', ''),
                'title': r.get('title', ''),
                'snippet': r.get('content', ''),
                'raw_content': r.get('raw_content', ''),  # Full page content
                'score': r.get('score', 0.5)
            })

        return results

    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return []


def web_search(query: str, max_results: int = None) -> List[Dict[str, Any]]:
    """Search using Tavily API"""
    config = get_config()
    max_results = max_results or config.search.results_per_query
    return search_tavily(query, max_results)
