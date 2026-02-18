"""Config and environment loading for Tavily enterprise demo."""
import copy
import threading
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .types import BankConfig, Config, DocNamingEntry

load_dotenv()


def _default_banks() -> list[BankConfig]:
    return [
        BankConfig(
            code="rbc",
            name="Royal Bank of Canada",
            aliases=["RBC"],
            primary_domains=["rbc.com"],
            ir_pages=[
                "https://www.rbc.com/investor-relations/financial-information.html",
            ],
            doc_naming={
                "pillar3_disclosure": DocNamingEntry(
                    document_aliases=["Supplementary Regulatory Capital Disclosure"],
                ),
                "supplementary_financial_info": DocNamingEntry(
                    url_patterns=["supp.pdf", "supp.xlsx"],
                ),
            },
        ),
        BankConfig(
            code="td",
            name="Toronto-Dominion Bank",
            aliases=["TD Bank", "TD"],
            primary_domains=["td.com"],
            ir_pages=[
                "https://www.td.com/ca/en/about-td/for-investors/investor-relations/financial-information/financial-reports/quarterly-results/quarterly-results-{year}",
                "https://www.td.com/ca/en/about-td/for-investors/investor-relations/financial-information/financial-reports/annual-reports/annual-report-{year}",
            ],
            doc_naming={
                "pillar3_disclosure": DocNamingEntry(
                    document_aliases=["Supplemental Regulatory Disclosure"],
                    url_patterns=["regulatory-disclosure", "supp-regulatory"],
                ),
                "supplementary_financial_info": DocNamingEntry(
                    url_patterns=["supplemental-financial"],
                ),
            },
        ),
        BankConfig(
            code="bns",
            name="Scotiabank",
            aliases=["Bank of Nova Scotia", "BNS"],
            primary_domains=["scotiabank.com"],
        ),
        BankConfig(
            code="bmo",
            name="Bank of Montreal",
            aliases=["BMO"],
            primary_domains=["bmo.com"],
        ),
        BankConfig(
            code="cm",
            name="Canadian Imperial Bank of Commerce",
            aliases=["CIBC", "CM"],
            primary_domains=["cibc.com"],
        ),
        BankConfig(
            code="na",
            name="National Bank of Canada",
            aliases=["NBC", "National Bank"],
            primary_domains=["nbc.ca", "bnc.ca"],
        ),
    ]


class Settings(BaseSettings):
    """Environment settings."""

    tavily_api_key: Optional[str] = Field(default=None, alias="TAVILY_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")

    oauth_url: Optional[str] = Field(default=None, alias="OAUTH_URL")
    oauth_client_id: Optional[str] = Field(default=None, alias="CLIENT_ID")
    oauth_client_secret: Optional[str] = Field(default=None, alias="CLIENT_SECRET")
    azure_base_url: Optional[str] = Field(default=None, alias="AZURE_BASE_URL")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


_config: Optional[Config] = None
_settings: Optional[Settings] = None
_config_lock = threading.Lock()
_settings_lock = threading.Lock()


def _ensure_banks(config: Config) -> Config:
    if not config.banks:
        config.banks = _default_banks()
    return config


def load_config(config_path: str = "config.yaml") -> Config:
    """Load config from YAML, falling back to defaults."""
    path = Path(config_path)

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return _ensure_banks(Config(**data))

    example_path = Path("config.yaml.example")
    if example_path.exists():
        with open(example_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return _ensure_banks(Config(**data))

    return _ensure_banks(Config())


def get_settings() -> Settings:
    return Settings()


def get_env_settings() -> Settings:
    global _settings
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                _settings = get_settings()
    return _settings


def get_config() -> Config:
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = load_config()
    return _config


def set_config(config: Config) -> None:
    global _config
    with _config_lock:
        _config = config


def apply_overrides(base: Config, overrides: dict) -> Config:
    """Apply dotted key overrides to a config copy."""
    data = copy.deepcopy(base.model_dump())
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target = data
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        field_name = parts[-1]

        current = target.get(field_name)
        if isinstance(value, str):
            if isinstance(current, bool):
                value = value.lower() in ("true", "1", "yes", "on")
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)

        target[field_name] = value

    updated = Config(**data)
    return _ensure_banks(updated)
