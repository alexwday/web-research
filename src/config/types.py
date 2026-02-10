"""
Domain models (enums + Pydantic data models) for Deep Research Agent.
"""
from typing import Optional, List
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# =============================================================================
# ENUMS
# =============================================================================

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SectionStatus(str, Enum):
    PLANNED = "planned"
    RESEARCHING = "researching"
    READY = "ready"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"


# =============================================================================
# DATABASE MODELS (Pydantic representations)
# =============================================================================

class ResearchTask(BaseModel):
    """Represents a single research task/topic"""
    id: Optional[int] = None
    parent_id: Optional[int] = None
    session_id: Optional[int] = None
    section_id: Optional[int] = None
    topic: str
    description: str
    file_path: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5  # 1-10, higher = more important
    depth: int = 0  # Recursion depth
    word_count: int = 0
    citation_count: int = 0
    is_gap_fill: bool = False
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0

    class Config:
        use_enum_values = True


class ReportSection(BaseModel):
    """Represents a section in the report outline"""
    id: Optional[int] = None
    session_id: Optional[int] = None
    title: str
    description: str = ""
    position: int = 0
    status: SectionStatus = SectionStatus.PLANNED
    synthesized_content: Optional[str] = None
    word_count: int = 0
    citation_count: int = 0
    is_gap_fill: bool = False
    created_at: Optional[datetime] = None
    synthesized_at: Optional[datetime] = None

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
    extracted_content: Optional[str] = None
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
    refined_brief: Optional[str] = None
    refinement_qa: Optional[str] = None
    cancel_requested_at: Optional[datetime] = None
