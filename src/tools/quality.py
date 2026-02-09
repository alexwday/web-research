"""Source quality scoring and domain classification."""
import re
from typing import List
from urllib.parse import urlparse

from ..logger import get_logger

logger = get_logger(__name__)


# Academic domains for quality scoring
ACADEMIC_DOMAINS = {
    'edu', 'ac.uk', 'ac.jp', 'edu.au', 'edu.cn',
    'arxiv.org', 'pubmed.gov', 'ncbi.nlm.nih.gov',
    'scholar.google.com', 'semanticscholar.org',
    'researchgate.net', 'academia.edu', 'jstor.org',
    'springer.com', 'nature.com', 'sciencedirect.com',
    'ieee.org', 'acm.org', 'nih.gov', 'gov'
}

# High-quality domains
HIGH_QUALITY_DOMAINS = {
    'wikipedia.org', 'britannica.com', 'bbc.com', 'bbc.co.uk',
    'nytimes.com', 'washingtonpost.com', 'theguardian.com',
    'reuters.com', 'apnews.com', 'bloomberg.com',
    'techcrunch.com', 'wired.com', 'arstechnica.com',
    'medium.com', 'github.com', 'stackoverflow.com'
}

# Domains known to produce junk, pirated, or paywall-fragment content
BLOCKED_DOMAINS = {
    'acronymattic.com', 'abbreviations.com', 'allacronyms.com',
    'dokumen.pub', 'dokumen.tips',
    'scribd.com',
    'coursehero.com',
    'chegg.com',
    'slideshare.net',
}

# URL patterns that indicate non-article data files or embeddings
BLOCKED_URL_PATTERNS = [
    re.compile(r'\.(txt|csv|tsv|json|jsonl|xml|dat|sql|log|gz|zip|tar|bz2|xz|bin|pkl)(\?.*)?$', re.IGNORECASE),
    re.compile(r'/data/[^/]*\.(txt|csv)', re.IGNORECASE),
    re.compile(r'/resources?/.*embeddings', re.IGNORECASE),
    re.compile(r'/zxcvbn/', re.IGNORECASE),
    re.compile(r'vocab[_.].*\.txt', re.IGNORECASE),
]

# Common English stopwords for relevance scoring
_STOPWORDS = frozenset({
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'and', 'but', 'or', 'if', 'while', 'because', 'until', 'about',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he',
    'him', 'his', 'she', 'her', 'they', 'them', 'their',
})


def get_domain(url: str) -> str:
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def is_academic_source(url: str) -> bool:
    """Check if URL is from an academic source"""
    domain = get_domain(url)

    # Check exact domain
    if domain in ACADEMIC_DOMAINS:
        return True

    # Check domain endings
    for academic in ACADEMIC_DOMAINS:
        if domain.endswith('.' + academic) or domain == academic:
            return True

    return False


def is_blocked_source(url: str) -> bool:
    """Check if a URL matches the domain blocklist or blocked URL patterns."""
    domain = get_domain(url)

    # Check exact domain and parent domains
    if domain in BLOCKED_DOMAINS:
        return True
    # e.g. "www.scribd.com" -> check "scribd.com"
    for blocked in BLOCKED_DOMAINS:
        if domain.endswith('.' + blocked):
            return True

    # Check URL patterns
    for pattern in BLOCKED_URL_PATTERNS:
        if pattern.search(url):
            return True

    return False


def is_junk_content(content: str, min_avg_line_length: int = 15) -> bool:
    """Detect data dumps (word lists, vocab files, CSV-like data).

    Heuristics:
    - Average line length below threshold (word lists have very short lines)
    - High ratio of lines with no spaces (one-word-per-line pattern)
    """
    if not content or len(content) < 200:
        return False

    lines = [line for line in content.splitlines() if line.strip()]
    if len(lines) < 20:
        return False

    # Sample up to 200 lines for efficiency
    sample = lines[:200]
    avg_len = sum(len(line) for line in sample) / len(sample)

    if avg_len < min_avg_line_length:
        # Check ratio of no-space lines (word-per-line pattern)
        no_space = sum(1 for line in sample if ' ' not in line.strip())
        ratio = no_space / len(sample)
        if ratio > 0.6:
            return True

    return False


def content_relevance_score(content: str, query_terms: List[str]) -> float:
    """Score how relevant content is to the query (0.0 - 1.0).

    Checks how many query terms appear in the first 2000 chars of content.
    Returns the ratio of matched terms to total terms.
    """
    if not content or not query_terms:
        return 0.0

    # Use first 2000 chars for speed
    sample = content[:2000].lower()
    matched = sum(1 for term in query_terms if term in sample)
    return matched / len(query_terms)


def _extract_query_terms(query: str) -> List[str]:
    """Extract meaningful search terms from a query string."""
    words = re.findall(r'[a-zA-Z]{2,}', query.lower())
    return [w for w in words if w not in _STOPWORDS]


def calculate_quality_score(url: str, title: str, content: str, query: str = None) -> float:
    """Calculate a quality score for a source (0-1)"""
    # Blocked sources get zero immediately
    if is_blocked_source(url):
        logger.debug(f"Blocked source: {url}")
        return 0.0

    score = 0.5  # Base score
    domain = get_domain(url)

    # Academic bonus
    if is_academic_source(url):
        score += 0.3

    # High-quality domain bonus
    if domain in HIGH_QUALITY_DOMAINS:
        score += 0.2

    # Content length factor (longer = potentially more comprehensive)
    content_len = len(content) if content else 0
    if content_len > 5000:
        score += 0.1
    elif content_len > 2000:
        score += 0.05

    # Penalize very short content
    if content_len < 500:
        score -= 0.1

    # Title quality
    if title and len(title) > 20:
        score += 0.05

    # Junk content penalty (data dumps, word lists)
    if content and is_junk_content(content):
        logger.debug(f"Junk content detected: {url}")
        score -= 0.4

    # Content-relevance penalty when query context is available
    if query and content:
        terms = _extract_query_terms(query)
        if terms:
            relevance = content_relevance_score(content, terms)
            if relevance == 0.0:
                # No query terms found in first 2000 chars — likely irrelevant
                score -= 0.2
            elif relevance < 0.2:
                score -= 0.1

    # Cap score at 1.0
    return min(max(score, 0.0), 1.0)
