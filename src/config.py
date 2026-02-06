"""
Configuration and Pydantic Models for Deep Research Agent
"""
import copy
import threading
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from enum import Enum

import yaml
from pydantic import BaseModel, Field
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


# =============================================================================
# DATABASE MODELS (Pydantic representations)
# =============================================================================

class ResearchTask(BaseModel):
    """Represents a single research task/topic"""
    id: Optional[int] = None
    parent_id: Optional[int] = None
    session_id: Optional[int] = None
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
    executive_summary: Optional[str] = None
    conclusion: Optional[str] = None
    report_markdown_path: Optional[str] = None
    report_html_path: Optional[str] = None


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class LLMModelsConfig(BaseModel):
    planner: str = "gpt-4o"
    researcher: str = "gpt-4o-mini"
    writer: str = "gpt-4o"
    editor: str = "gpt-4o"
    discovery: str = "gpt-4o"


class LLMMaxTokensConfig(BaseModel):
    planner: int = 100_000
    researcher: int = 100_000
    writer: int = 100_000
    editor: int = 100_000
    discovery: int = 100_000


class LLMTemperatureConfig(BaseModel):
    planner: float = 0.3
    researcher: float = 0.2
    writer: float = 0.4
    editor: float = 0.2
    discovery: float = 0.2


class LLMConfig(BaseModel):
    models: LLMModelsConfig = Field(default_factory=LLMModelsConfig)
    max_tokens: LLMMaxTokensConfig = Field(default_factory=LLMMaxTokensConfig)
    temperature: LLMTemperatureConfig = Field(default_factory=LLMTemperatureConfig)


class SearchConfig(BaseModel):
    depth: str = "advanced"
    max_results: int = 8
    queries_per_task: int = 3
    include_domains: List[str] = Field(default_factory=list)
    exclude_domains: List[str] = Field(default_factory=lambda: ["pinterest.com", "quora.com"])


class ScrapingConfig(BaseModel):
    max_content_length: int = 15000
    timeout: int = 15
    rotate_user_agents: bool = True


class DiscoveryConfig(BaseModel):
    enabled: bool = True
    frequency: int = 3  # run discovery every N completed tasks
    max_suggestions_per_run: int = 3


class RewriteConfig(BaseModel):
    enabled: bool = True


class ResearchConfig(BaseModel):
    min_initial_tasks: int = 10
    max_total_tasks: int = 200
    max_recursion_depth: int = 1
    min_words_per_section: int = 500
    max_words_per_section: int = 3000
    min_citations_per_section: int = 2
    enable_recursion: bool = True
    task_delay: int = 2
    max_runtime_hours: int = 24
    max_loops: int = -1  # -1 = infinite, 0 = do nothing
    max_concurrent_tasks: int = 3  # number of research tasks to run in parallel


class OutputConfig(BaseModel):
    directory: str = "report"
    report_name: str = "DEEP_RESEARCH_REPORT"
    formats: List[str] = Field(default_factory=lambda: ["markdown", "html"])
    include_toc: bool = True
    include_bibliography: bool = True
    include_glossary: bool = True
    include_summary: bool = True


class QualityConfig(BaseModel):
    min_source_quality: float = 0.5


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
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    rewrite: RewriteConfig = Field(default_factory=RewriteConfig)
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
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    tavily_api_key: Optional[str] = Field(default=None, alias="TAVILY_API_KEY")

    # OAuth / Azure endpoint (corporate environment)
    oauth_url: Optional[str] = Field(default=None, alias="OAUTH_URL")
    oauth_client_id: Optional[str] = Field(default=None, alias="CLIENT_ID")
    oauth_client_secret: Optional[str] = Field(default=None, alias="CLIENT_SECRET")
    azure_base_url: Optional[str] = Field(default=None, alias="AZURE_BASE_URL")

    # Per-agent model name & pricing (override config.yaml when set)
    planner_model_name: Optional[str] = Field(default=None, alias="PLANNER_MODEL_NAME")
    planner_model_input_cost: Optional[float] = Field(default=None, alias="PLANNER_MODEL_INPUT_COST")
    planner_model_output_cost: Optional[float] = Field(default=None, alias="PLANNER_MODEL_OUTPUT_COST")

    researcher_model_name: Optional[str] = Field(default=None, alias="RESEARCHER_MODEL_NAME")
    researcher_model_input_cost: Optional[float] = Field(default=None, alias="RESEARCHER_MODEL_INPUT_COST")
    researcher_model_output_cost: Optional[float] = Field(default=None, alias="RESEARCHER_MODEL_OUTPUT_COST")

    writer_model_name: Optional[str] = Field(default=None, alias="WRITER_MODEL_NAME")
    writer_model_input_cost: Optional[float] = Field(default=None, alias="WRITER_MODEL_INPUT_COST")
    writer_model_output_cost: Optional[float] = Field(default=None, alias="WRITER_MODEL_OUTPUT_COST")

    editor_model_name: Optional[str] = Field(default=None, alias="EDITOR_MODEL_NAME")
    editor_model_input_cost: Optional[float] = Field(default=None, alias="EDITOR_MODEL_INPUT_COST")
    editor_model_output_cost: Optional[float] = Field(default=None, alias="EDITOR_MODEL_OUTPUT_COST")

    discovery_model_name: Optional[str] = Field(default=None, alias="DISCOVERY_MODEL_NAME")
    discovery_model_input_cost: Optional[float] = Field(default=None, alias="DISCOVERY_MODEL_INPUT_COST")
    discovery_model_output_cost: Optional[float] = Field(default=None, alias="DISCOVERY_MODEL_OUTPUT_COST")

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
_config_lock = threading.Lock()
_settings_lock = threading.Lock()


def get_config() -> Config:
    """Get global config instance, with env var overrides for model names."""
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = load_config()
                _apply_env_model_overrides(_config)
    return _config


def _apply_env_model_overrides(config: Config) -> None:
    """Override config.yaml model names with env vars when set."""
    settings = get_env_settings()
    if settings.planner_model_name:
        config.llm.models.planner = settings.planner_model_name
    if settings.researcher_model_name:
        config.llm.models.researcher = settings.researcher_model_name
    if settings.writer_model_name:
        config.llm.models.writer = settings.writer_model_name
    if settings.editor_model_name:
        config.llm.models.editor = settings.editor_model_name
    if settings.discovery_model_name:
        config.llm.models.discovery = settings.discovery_model_name


def get_env_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                _settings = get_settings()
    return _settings


def set_config(config: Config) -> None:
    """Replace the global config singleton (thread-safe)."""
    global _config
    with _config_lock:
        _config = config


def apply_overrides(base: Config, overrides: dict) -> Config:
    """
    Deep-copy *base* config and apply dotted-key overrides.

    Keys use dot notation matching the Config model hierarchy, e.g.
    ``"research.max_total_tasks": 15``.  String values are coerced to
    the target field's type (int / float / bool).
    """
    data = copy.deepcopy(base.model_dump())

    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target = data
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        field_name = parts[-1]

        # Type coercion: inspect the current value to decide target type
        current = target.get(field_name)
        if isinstance(value, str):
            if isinstance(current, bool):
                value = value.lower() in ("true", "1", "yes", "on")
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)

        target[field_name] = value

    return Config(**data)


# =============================================================================
# RESEARCH PRESETS
# =============================================================================

RESEARCH_PRESETS = {
    "quick": {
        "label": "Quick",
        "description": "Fast overview, 3\u20135 tasks, minimal depth",
        "overrides": {
            "research.min_initial_tasks": 3,
            "research.max_total_tasks": 5,
            "research.min_words_per_section": 100,
            "research.max_words_per_section": 500,
            "research.min_citations_per_section": 1,
            "research.enable_recursion": False,
            "research.max_recursion_depth": 0,
            "research.max_runtime_hours": 1,
            "research.max_loops": 5,
            "research.max_concurrent_tasks": 1,
            "search.queries_per_task": 1,
            "search.max_results": 3,
            "discovery.enabled": False,
            "rewrite.enabled": False,
        },
    },
    "standard": {
        "label": "Standard",
        "description": "Balanced research, 8\u201315 tasks",
        "overrides": {
            "research.min_initial_tasks": 8,
            "research.max_total_tasks": 15,
            "research.min_words_per_section": 500,
            "research.max_words_per_section": 2000,
            "research.min_citations_per_section": 3,
            "research.enable_recursion": True,
            "research.max_recursion_depth": 1,
            "research.max_runtime_hours": 6,
            "research.max_loops": 15,
            "research.max_concurrent_tasks": 2,
            "search.queries_per_task": 2,
            "search.max_results": 5,
            "discovery.enabled": True,
            "discovery.frequency": 5,
            "discovery.max_suggestions_per_run": 2,
            "rewrite.enabled": True,
        },
    },
    "deep": {
        "label": "Deep",
        "description": "Thorough analysis, 15\u201330 tasks, recursive",
        "overrides": {
            "research.min_initial_tasks": 15,
            "research.max_total_tasks": 30,
            "research.min_words_per_section": 1000,
            "research.max_words_per_section": 3000,
            "research.min_citations_per_section": 5,
            "research.enable_recursion": True,
            "research.max_recursion_depth": 2,
            "research.max_runtime_hours": 24,
            "research.max_loops": 30,
            "research.max_concurrent_tasks": 3,
            "search.queries_per_task": 3,
            "search.max_results": 8,
            "discovery.enabled": True,
            "discovery.frequency": 3,
            "discovery.max_suggestions_per_run": 3,
            "rewrite.enabled": True,
        },
    },
    "exhaustive": {
        "label": "Exhaustive",
        "description": "Maximum depth, 30\u201350 tasks, deep recursion",
        "overrides": {
            "research.min_initial_tasks": 30,
            "research.max_total_tasks": 50,
            "research.min_words_per_section": 2000,
            "research.max_words_per_section": 5000,
            "research.min_citations_per_section": 8,
            "research.enable_recursion": True,
            "research.max_recursion_depth": 3,
            "research.max_runtime_hours": 48,
            "research.max_loops": 50,
            "research.max_concurrent_tasks": 4,
            "search.queries_per_task": 4,
            "search.max_results": 10,
            "discovery.enabled": True,
            "discovery.frequency": 2,
            "discovery.max_suggestions_per_run": 5,
            "rewrite.enabled": True,
        },
    },
}
