"""URL scraping and source extraction."""
import random
import ipaddress
from datetime import datetime, timezone
from typing import Dict, Any, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import get_config, Source
from ..logger import get_logger
from .text import strip_image_data
from .quality import get_domain, is_academic_source, is_blocked_source, calculate_quality_score
from .search import get_scrape_limiter

logger = get_logger(__name__)


# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


def _validate_url(url: str) -> None:
    """Validate that a URL is safe to fetch (prevent SSRF)."""
    import socket
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname")
    try:
        resolved = socket.getaddrinfo(hostname, None)
        for family, _type, proto, canonname, sockaddr in resolved:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                raise ValueError(f"URL resolves to non-public address: {ip}")
    except socket.gaierror:
        pass  # DNS resolution failed — let requests handle it


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.RequestException, ConnectionError))
)
def scrape_url(url: str) -> Tuple[str, str]:
    """
    Scrape content from a URL
    Returns: (title, content)
    """
    config = get_config()

    # Validate URL to prevent SSRF
    _validate_url(url)

    # Apply rate limiting
    get_scrape_limiter().wait()

    logger.debug(f"Scraping: {url}")

    # Get random user agent
    user_agent = random.choice(USER_AGENTS) if config.scraping.rotate_user_agents else USER_AGENTS[0]

    headers = {
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=config.scraping.timeout,
            allow_redirects=True
        )
        response.raise_for_status()

        # Try trafilatura first (better extraction) — pass already-fetched HTML
        try:
            import trafilatura
            content = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=True,
                no_fallback=False
            )
            if content and len(content) > 200:
                content = strip_image_data(content)
                # Get title separately
                soup = BeautifulSoup(response.content, 'html.parser')
                title = (soup.title.string or "") if soup.title else ""
                return title.strip(), content[:config.scraping.max_content_length]
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Trafilatura extraction failed: {e}")

        # Fallback to BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Get title
        title = ""
        if soup.title:
            title = soup.title.string or ""

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header',
                           'aside', 'form', 'button', 'iframe', 'noscript',
                           'img', 'svg', 'picture', 'video', 'audio', 'canvas']):
            element.decompose()

        # Try to find main content
        main_content = None
        for selector in ['article', 'main', '[role="main"]', '.content', '#content', '.post', '.article']:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            # Get body text
            body = soup.find('body')
            text = body.get_text(separator='\n', strip=True) if body else soup.get_text(separator='\n', strip=True)

        # Clean up text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        text = strip_image_data(text)

        # Truncate if needed
        text = text[:config.scraping.max_content_length]

        return title.strip(), text

    except requests.RequestException as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error scraping {url}: {e}")
        return "", ""


def extract_source_info(url: str, search_result: Dict[str, Any] = None, query: str = None) -> Source:
    """
    Extract full source information from a URL
    Uses raw_content from Tavily if available, falls back to scraping
    """
    domain = get_domain(url)

    # Early exit for blocked sources — skip scraping entirely
    if is_blocked_source(url):
        logger.info(f"Blocked source (skipping scrape): {url}")
        return Source(
            url=url,
            title=search_result.get('title', domain) if search_result else domain,
            domain=domain,
            snippet="",
            full_content="",
            quality_score=0.0,
            is_academic=False,
            accessed_at=datetime.now(timezone.utc)
        )

    title = search_result.get('title', '') if search_result else ''
    snippet = search_result.get('snippet', '') if search_result else ''

    # Try to use raw_content from Tavily search results (already fetched)
    full_content = ""
    if search_result and search_result.get('raw_content'):
        full_content = strip_image_data(search_result['raw_content'])
        logger.debug(f"Using Tavily raw_content for {url[:50]}...")
    else:
        # Fall back to scraping if no raw_content available
        try:
            scraped_title, scraped_content = scrape_url(url)
            if scraped_title and not title:
                title = scraped_title
            full_content = scraped_content
        except Exception as e:
            logger.warning(f"Could not scrape {url}: {e}")

    is_academic = is_academic_source(url)
    quality_score = calculate_quality_score(url, title, full_content or snippet, query=query)

    return Source(
        url=url,
        title=title or domain,
        domain=domain,
        snippet=snippet,
        full_content=full_content,
        quality_score=quality_score,
        is_academic=is_academic,
        accessed_at=datetime.now(timezone.utc)
    )
