"""Text processing utilities: stripping, counting, tokenization."""
import re
from typing import List

from ..config import get_config
from ..logger import get_logger

logger = get_logger(__name__)


def strip_image_data(text: str) -> str:
    """Remove base64 data URIs and other embedded image/binary noise from text.

    OpenAI rejects prompts that contain data-URI images.  This strips them
    out so scraped web content can safely be sent to the LLM.
    """
    # data:image/...;base64,<long blob>  (greedy across whitespace)
    text = re.sub(r'data:image/[^;]{1,20};base64,[A-Za-z0-9+/=\s]{20,}', '[image removed]', text)
    # data:application/octet-stream or other binary data URIs
    text = re.sub(r'data:[a-z]+/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=\s]{100,}', '[binary data removed]', text)
    # Stray long base64 blobs that aren't wrapped in a data URI (>200 chars of pure b64)
    text = re.sub(r'(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{200,}={0,3}(?![A-Za-z0-9+/=])', '[blob removed]', text)
    return text


def count_words(text: str) -> int:
    """Count words in text"""
    if not text:
        return 0
    return len(text.split())


def count_citations(text: str) -> int:
    """Count citations in text (looks for [1], [2] style, [Source: ...], or URLs)"""
    if not text:
        return 0

    # Primary: count [1], [2] style numbered citations (what the LLM actually produces)
    # Negative lookbehind for '](' to avoid matching markdown link text like [text](url)
    numbered_pattern = r'\[(\d+)\](?!\()'
    numbered = len(re.findall(numbered_pattern, text))

    if numbered > 0:
        return numbered

    # Fallback: count [Source: ...] patterns
    citation_pattern = r'\[Source:.*?\]'
    citations = len(re.findall(citation_pattern, text, re.IGNORECASE))

    # Count inline URLs
    url_pattern = r'https?://[^\s\]\)>]+'
    urls = len(re.findall(url_pattern, text))

    # Count markdown footnotes
    footnote_pattern = r'\[\^\d+\]'
    footnotes = len(re.findall(footnote_pattern, text))

    return max(citations, urls // 2, footnotes)


def _resolve_tiktoken_model(model: str = None) -> str:
    """Resolve the model name for tiktoken. Falls back to config writer model."""
    if model is None:
        model = get_config().llm.models.writer
    return model


def count_tokens(text: str, model: str = None) -> int:
    """
    Estimate token count for text
    Uses tiktoken if available, otherwise rough estimate
    """
    model = _resolve_tiktoken_model(model)
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

    except (ImportError, KeyError):
        # Rough estimate: ~4 characters per token
        return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int, model: str = None) -> str:
    """Truncate text to fit within token limit"""
    model = _resolve_tiktoken_model(model)
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)

        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)

    except (ImportError, KeyError):
        # Rough estimate
        chars_per_token = 4
        max_chars = max_tokens * chars_per_token
        return text[:max_chars]
