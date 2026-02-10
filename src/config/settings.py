"""
Configuration classes, singletons, and loaders for Deep Research Agent.
"""
import copy
import threading
from pathlib import Path
from typing import Optional, List

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class LLMModelsConfig(BaseModel):
    planner: str = "gpt-4o"
    researcher: str = "gpt-4o-mini"
    writer: str = "gpt-4o"
    editor: str = "gpt-4o"
    outline_designer: str = "gpt-4o"
    synthesizer: str = "gpt-4o"
    analyzer: str = "gpt-4o-mini"
    refiner: str = "gpt-4o-mini"


class LLMMaxTokensConfig(BaseModel):
    planner: int = 100_000
    researcher: int = 100_000
    writer: int = 100_000
    editor: int = 100_000
    outline_designer: int = 100_000
    synthesizer: int = 100_000
    analyzer: int = 4000
    refiner: int = 4000


class LLMTemperatureConfig(BaseModel):
    planner: float = 0.3
    researcher: float = 0.2
    writer: float = 0.4
    editor: float = 0.2
    outline_designer: float = 0.3
    synthesizer: float = 0.3
    analyzer: float = 0.2
    refiner: float = 0.4


class LLMConfig(BaseModel):
    models: LLMModelsConfig = Field(default_factory=LLMModelsConfig)
    max_tokens: LLMMaxTokensConfig = Field(default_factory=LLMMaxTokensConfig)
    temperature: LLMTemperatureConfig = Field(default_factory=LLMTemperatureConfig)


class SearchConfig(BaseModel):
    depth: str = "advanced"
    results_per_query: int = 3
    queries_per_task: int = 3
    pre_plan_queries: int = 5
    pre_plan_max_results: int = 8
    gap_fill_queries: int = 2
    gap_fill_max_results: int = 3
    min_tavily_score: float = 0.1
    include_domains: List[str] = Field(default_factory=list)
    exclude_domains: List[str] = Field(default_factory=lambda: [
        "pinterest.com", "quora.com",
        "acronymattic.com", "abbreviations.com",
        "dokumen.pub", "dokumen.tips",
    ])


class ScrapingConfig(BaseModel):
    max_content_length: int = 15000
    timeout: int = 15
    rotate_user_agents: bool = True


class GapAnalysisConfig(BaseModel):
    enabled: bool = True
    max_new_sections: int = 3
    max_gap_fill_tasks: int = 10


class SynthesisConfig(BaseModel):
    min_words_per_section: int = 500
    max_words_per_section: int = 3000
    min_citations_per_section: int = 2
    max_concurrent: int = 2
    style: str = "balanced"  # "confident", "balanced", or "thorough"


class ResearchConfig(BaseModel):
    min_initial_tasks: int = 10
    max_total_tasks: int = 200
    max_recursion_depth: int = 1
    tasks_per_section: int = 3
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
    min_source_quality: float = 0.55


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
    path: str = "data/research_state.db"
    wal_mode: bool = True


class QueryRefinementConfig(BaseModel):
    enabled: bool = True
    min_questions: int = 3
    max_questions: int = 5


class Config(BaseModel):
    """Main configuration model"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    gap_analysis: GapAnalysisConfig = Field(default_factory=GapAnalysisConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    rate_limits: RateLimitsConfig = Field(default_factory=RateLimitsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    query_refinement: QueryRefinementConfig = Field(default_factory=QueryRefinementConfig)


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

    analyzer_model_name: Optional[str] = Field(default=None, alias="ANALYZER_MODEL_NAME")
    analyzer_model_input_cost: Optional[float] = Field(default=None, alias="ANALYZER_MODEL_INPUT_COST")
    analyzer_model_output_cost: Optional[float] = Field(default=None, alias="ANALYZER_MODEL_OUTPUT_COST")

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
        print(f"\u26a0\ufe0f  No config.yaml found. Using config.example.yaml")
        with open(example_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        return Config(**data)

    # Return defaults
    print("\u26a0\ufe0f  No configuration file found. Using defaults.")
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
    if settings.analyzer_model_name:
        config.llm.models.analyzer = settings.analyzer_model_name


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
