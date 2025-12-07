"""
Deep Research Agent - A 24-hour automated research system
"""

__version__ = "1.0.0"
__author__ = "Deep Research Agent"

from .config import get_config, get_env_settings, load_config
from .database import get_database, DatabaseManager
from .orchestrator import run_research, ResearchOrchestrator
from .agents import PlannerAgent, ResearcherAgent, EditorAgent
from .compiler import ReportCompiler
from .tools import web_search, scrape_url, extract_source_info
from .llm_client import get_llm_client
from .logger import get_logger

__all__ = [
    # Config
    "get_config",
    "get_env_settings", 
    "load_config",
    
    # Database
    "get_database",
    "DatabaseManager",
    
    # Orchestrator
    "run_research",
    "ResearchOrchestrator",
    
    # Agents
    "PlannerAgent",
    "ResearcherAgent",
    "EditorAgent",
    
    # Compiler
    "ReportCompiler",
    
    # Tools
    "web_search",
    "scrape_url",
    "extract_source_info",
    
    # LLM
    "get_llm_client",
    
    # Logger
    "get_logger",
]
