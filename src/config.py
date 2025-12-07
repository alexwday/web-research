"""
Configuration and Pydantic Models for Deep Research Agent
"""
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# ENUMS
# =============================================================================

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"


class SearchProvider(str, Enum):
    TAVILY = "tavily"
    SERPER = "serper"
    BRAVE = "brave"


# =============================================================================
# DATABASE MODELS (Pydantic representations)
# =============================================================================

class ResearchTask(BaseModel):
    """Represents a single research task/topic"""
    id: Optional[int] = None
    parent_id: Optional[int] = None
    topic: str
    description: str
    file_path: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5  # 1-10, higher = more important
    depth: int = 0  # Recursion depth
    word_count: int = 0
    citation_count: int = 0
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    class Config:
        use_enum_values = True


class Source(BaseModel):
    """Represents a source/citation"""
    id: Optional[int] = None
    url: str
    title: str
    domain: str
    snippet: Optional[str] = None
    full_content: Optional[str] = None
    quality_score: float = 0.5
    is_academic: bool = False
    accessed_at: Optional[datetime] = None
    task_ids: List[int] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True


class GlossaryTerm(BaseModel):
    """Represents a glossary entry"""
    id: Optional[int] = None
    term: str
    definition: str
    first_occurrence_task_id: Optional[int] = None
    
    class Config:
        use_enum_values = True


class ResearchSession(BaseModel):
    """Represents a research session/run"""
    id: Optional[int] = None
    query: str
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    total_words: int = 0
    total_sources: int = 0
    status: str = "running"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class LLMModelsConfig(BaseModel):
    planner: str = "claude-sonnet-4-20250514"
    researcher: str = "claude-sonnet-4-20250514"
    writer: str = "claude-sonnet-4-20250514"
    editor: str = "claude-sonnet-4-20250514"


class LLMMaxTokensConfig(BaseModel):
    planner: int = 8000
    researcher: int = 4000
    writer: int = 6000
    editor: int = 8000


class LLMTemperatureConfig(BaseModel):
    planner: float = 0.3
    researcher: float = 0.2
    writer: float = 0.4
    editor: float = 0.2


class LLMConfig(BaseModel):
    provider: LLMProvider = LLMProvider.ANTHROPIC
    models: LLMModelsConfig = Field(default_factory=LLMModelsConfig)
    max_tokens: LLMMaxTokensConfig = Field(default_factory=LLMMaxTokensConfig)
    temperature: LLMTemperatureConfig = Field(default_factory=LLMTemperatureConfig)


class SearchConfig(BaseModel):
    provider: SearchProvider = SearchProvider.TAVILY
    depth: str = "advanced"
    max_results: int = 8
    queries_per_task: int = 3
    include_domains: List[str] = Field(default_factory=list)
    exclude_domains: List[str] = Field(default_factory=lambda: ["pinterest.com", "quora.com"])


class ScrapingConfig(BaseModel):
    max_content_length: int = 15000
    timeout: int = 15
    rotate_user_agents: bool = True
    respect_robots: bool = True
    max_retries: int = 3


class ResearchConfig(BaseModel):
    min_initial_tasks: int = 10
    max_total_tasks: int = 200
    max_recursion_depth: int = 1
    min_words_per_section: int = 500
    max_words_per_section: int = 3000
    min_citations_per_section: int = 2
    enable_recursion: bool = True
    session_delay: int = 2
    max_runtime_hours: int = 24
    max_loops: int = 0


class OutputConfig(BaseModel):
    directory: str = "report"
    report_name: str = "DEEP_RESEARCH_REPORT"
    formats: List[str] = Field(default_factory=lambda: ["markdown", "html"])
    include_toc: bool = True
    include_bibliography: bool = True
    include_glossary: bool = True
    include_summary: bool = True


class QualityConfig(BaseModel):
    validate_citations: bool = True
    check_duplicates: bool = True
    min_source_quality: float = 0.5
    prefer_academic: bool = False
    fact_check: bool = False


class RateLimitsConfig(BaseModel):
    llm_calls_per_minute: int = 20
    search_calls_per_minute: int = 10
    scrape_requests_per_minute: int = 30


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "logs/research.log"
    max_file_size: int = 10
    backup_count: int = 5
    show_progress: bool = True


class DatabaseConfig(BaseModel):
    path: str = "research_state.db"
    wal_mode: bool = True


class Config(BaseModel):
    """Main configuration model"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    rate_limits: RateLimitsConfig = Field(default_factory=RateLimitsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)


# =============================================================================
# ENVIRONMENT SETTINGS
# =============================================================================

class Settings(BaseSettings):
    """Environment-based settings"""
    # API Keys
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    tavily_api_key: Optional[str] = Field(default=None, alias="TAVILY_API_KEY")
    serper_api_key: Optional[str] = Field(default=None, alias="SERPER_API_KEY")
    brave_api_key: Optional[str] = Field(default=None, alias="BRAVE_API_KEY")
    
    # OpenRouter specific
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL"
    )
    
    class Config:
        env_file = ".env"
        extra = "ignore"


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================

def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file"""
    path = Path(config_path)
    
    if path.exists():
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        return Config(**data)
    
    # Check for example config
    example_path = Path("config.example.yaml")
    if example_path.exists():
        print(f"⚠️  No config.yaml found. Using config.example.yaml")
        with open(example_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        return Config(**data)
    
    # Return defaults
    print("⚠️  No configuration file found. Using defaults.")
    return Config()


def get_settings() -> Settings:
    """Get environment settings"""
    return Settings()


# Global instances
_config: Optional[Config] = None
_settings: Optional[Settings] = None


def get_config() -> Config:
    """Get global config instance"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_env_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings
