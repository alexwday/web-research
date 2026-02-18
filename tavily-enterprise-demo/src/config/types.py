"""Pydantic config models for Tavily enterprise demo."""
from typing import Dict, List

from pydantic import BaseModel, Field


class DocNamingEntry(BaseModel):
    """Per-document-type naming overrides for a specific bank."""
    document_aliases: List[str] = Field(default_factory=list)
    url_patterns: List[str] = Field(default_factory=list)


class BankConfig(BaseModel):
    code: str
    name: str
    aliases: List[str] = Field(default_factory=list)
    primary_domains: List[str] = Field(default_factory=list)
    ir_pages: List[str] = Field(default_factory=list)
    doc_naming: Dict[str, DocNamingEntry] = Field(default_factory=dict)


class SearchConfig(BaseModel):
    depth: str = "advanced"
    default_max_results: int = 8
    min_tavily_score: float = 0.15
    search_calls_per_minute: int = 12
    scrape_requests_per_minute: int = 20
    include_domains: List[str] = Field(default_factory=list)
    exclude_domains: List[str] = Field(
        default_factory=lambda: [
            "pinterest.com",
            "quora.com",
            "acronymattic.com",
            "abbreviations.com",
            "dokumen.pub",
            "dokumen.tips",
        ]
    )
    secondary_search_enabled: bool = True


class DownloadConfig(BaseModel):
    directory: str = "report/artifacts"
    timeout_seconds: int = 45
    allowed_extensions: List[str] = Field(default_factory=lambda: ["pdf", "xlsx", "xls"])
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )


class DocumentTarget(BaseModel):
    doc_type: str
    label: str
    required_formats: List[str] = Field(default_factory=lambda: ["pdf"])


class QuarterlyConfig(BaseModel):
    poll_seconds: int = 900
    max_iterations: int = 96
    max_verify_candidates: int = 5
    document_targets: List[DocumentTarget] = Field(
        default_factory=lambda: [
            DocumentTarget(
                doc_type="report_to_shareholders",
                label="Report to Shareholders",
                required_formats=["pdf"],
            ),
            DocumentTarget(
                doc_type="pillar3_disclosure",
                label="Pillar 3 Disclosure",
                required_formats=["pdf", "xlsx"],
            ),
            DocumentTarget(
                doc_type="supplementary_financial_info",
                label="Supplementary Financial Information",
                required_formats=["xlsx"],
            ),
        ]
    )


class LCRConfig(BaseModel):
    query_template: str = (
        "{bank_name} {period} liquidity coverage ratio LCR percentage disclosure"
    )


class HeadlinesConfig(BaseModel):
    default_topics: List[str] = Field(
        default_factory=lambda: [
            "canadian finance",
            "bank earnings",
            "interest rates",
            "capital markets",
            "credit and liquidity",
        ]
    )
    results_per_query: int = 6


class DeepResearchConfig(BaseModel):
    subqueries_per_run: int = 4
    max_sources: int = 12
    max_excerpt_chars: int = 1800


class OutputConfig(BaseModel):
    directory: str = "report"


class LLMRankingConfig(BaseModel):
    enabled: bool = True
    model: str = "gpt-4.1-mini"
    max_tokens: int = 2000
    temperature: float = 0.1


class Config(BaseModel):
    search: SearchConfig = Field(default_factory=SearchConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    quarterly: QuarterlyConfig = Field(default_factory=QuarterlyConfig)
    lcr: LCRConfig = Field(default_factory=LCRConfig)
    headlines: HeadlinesConfig = Field(default_factory=HeadlinesConfig)
    deep_research: DeepResearchConfig = Field(default_factory=DeepResearchConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    llm_ranking: LLMRankingConfig = Field(default_factory=LLMRankingConfig)
    banks: List[BankConfig] = Field(default_factory=list)
