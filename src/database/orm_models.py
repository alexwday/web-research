"""ORM models for the Deep Research Agent database."""
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean,
    DateTime, ForeignKey, Index, Table
)
from sqlalchemy.orm import declarative_base, relationship, backref

from ..config import (
    TaskStatus, SectionStatus, ResearchTask, ReportSection,
    Source, GlossaryTerm, ResearchSession
)

Base = declarative_base()


# Association table for tasks and sources
task_source_association = Table(
    'task_source_association',
    Base.metadata,
    Column('task_id', Integer, ForeignKey('tasks.id'), primary_key=True),
    Column('source_id', Integer, ForeignKey('sources.id'), primary_key=True),
    Column('position', Integer, default=0),
    Column('extracted_content', Text, nullable=True),
)


class TaskModel(Base):
    """SQLAlchemy model for research tasks"""
    __tablename__ = 'tasks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    parent_id = Column(Integer, ForeignKey('tasks.id'), nullable=True)
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=True)
    section_id = Column(Integer, ForeignKey('sections.id'), nullable=True)
    topic = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    file_path = Column(String(500), nullable=False)
    status = Column(String(20), default='pending')
    priority = Column(Integer, default=5)
    depth = Column(Integer, default=0)
    word_count = Column(Integer, default=0)
    citation_count = Column(Integer, default=0)
    is_gap_fill = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)

    # Relationships
    children = relationship("TaskModel", backref=backref("parent", remote_side="TaskModel.id"), foreign_keys=[parent_id])
    sources = relationship("SourceModel", secondary=task_source_association, back_populates="tasks")

    def to_pydantic(self) -> ResearchTask:
        return ResearchTask(
            id=self.id,
            parent_id=self.parent_id,
            session_id=self.session_id,
            section_id=self.section_id,
            topic=self.topic,
            description=self.description,
            file_path=self.file_path,
            status=TaskStatus(self.status),
            priority=self.priority,
            depth=self.depth,
            word_count=self.word_count,
            citation_count=self.citation_count,
            is_gap_fill=self.is_gap_fill or False,
            created_at=self.created_at,
            completed_at=self.completed_at,
            error_message=self.error_message,
            retry_count=self.retry_count or 0
        )


class SourceModel(Base):
    """SQLAlchemy model for sources/citations"""
    __tablename__ = 'sources'

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(2000), unique=True, nullable=False)
    title = Column(String(500), nullable=False)
    domain = Column(String(200), nullable=False)
    snippet = Column(Text, nullable=True)
    # Retained for debugging and potential future re-synthesis of sections
    full_content = Column(Text, nullable=True)
    extracted_content = Column(Text, nullable=True)
    quality_score = Column(Float, default=0.5)
    is_academic = Column(Boolean, default=False)
    accessed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    tasks = relationship("TaskModel", secondary=task_source_association, back_populates="sources")

    def to_pydantic(self) -> Source:
        return Source(
            id=self.id,
            url=self.url,
            title=self.title,
            domain=self.domain,
            snippet=self.snippet,
            full_content=self.full_content,
            extracted_content=self.extracted_content,
            quality_score=self.quality_score,
            is_academic=self.is_academic,
            accessed_at=self.accessed_at,
            task_ids=[t.id for t in self.tasks]
        )


class GlossaryModel(Base):
    """SQLAlchemy model for glossary terms"""
    __tablename__ = 'glossary'

    id = Column(Integer, primary_key=True, autoincrement=True)
    term = Column(String(200), unique=True, nullable=False)
    definition = Column(Text, nullable=False)
    first_occurrence_task_id = Column(Integer, ForeignKey('tasks.id'), nullable=True)
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=True)

    def to_pydantic(self) -> GlossaryTerm:
        return GlossaryTerm(
            id=self.id,
            term=self.term,
            definition=self.definition,
            first_occurrence_task_id=self.first_occurrence_task_id
        )


class RunEventModel(Base):
    """SQLAlchemy model for run activity events."""
    __tablename__ = 'run_events'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=True)
    task_id = Column(Integer, ForeignKey('tasks.id'), nullable=True)
    event_type = Column(String(30), nullable=False)
    query_group = Column(String(50), nullable=True)
    query_text = Column(String(1000), nullable=True)
    url = Column(String(2000), nullable=True)
    title = Column(String(500), nullable=True)
    snippet = Column(Text, nullable=True)
    quality_score = Column(Float, nullable=True)
    phase = Column(String(30), nullable=True)
    severity = Column(String(10), nullable=True)
    payload_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index('ix_run_events_timeline', 'session_id', 'created_at', 'id'),
        Index('ix_run_events_type', 'session_id', 'event_type'),
    )


class SectionModel(Base):
    """SQLAlchemy model for report sections"""
    __tablename__ = 'sections'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True, default="")
    position = Column(Integer, default=0)
    status = Column(String(20), default='planned')
    synthesized_content = Column(Text, nullable=True)
    word_count = Column(Integer, default=0)
    citation_count = Column(Integer, default=0)
    is_gap_fill = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    synthesized_at = Column(DateTime, nullable=True)

    def to_pydantic(self) -> ReportSection:
        return ReportSection(
            id=self.id,
            session_id=self.session_id,
            title=self.title,
            description=self.description or "",
            position=self.position,
            status=SectionStatus(self.status),
            synthesized_content=self.synthesized_content,
            word_count=self.word_count,
            citation_count=self.citation_count,
            is_gap_fill=self.is_gap_fill or False,
            created_at=self.created_at,
            synthesized_at=self.synthesized_at,
        )


class SessionModel(Base):
    """SQLAlchemy model for research sessions"""
    __tablename__ = 'sessions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime, nullable=True)
    total_tasks = Column(Integer, default=0)
    completed_tasks = Column(Integer, default=0)
    total_words = Column(Integer, default=0)
    total_sources = Column(Integer, default=0)
    status = Column(String(20), default='running')
    executive_summary = Column(Text, nullable=True)
    conclusion = Column(Text, nullable=True)
    report_markdown_path = Column(String(500), nullable=True)
    report_html_path = Column(String(500), nullable=True)
    refined_brief = Column(Text, nullable=True)
    refinement_qa = Column(Text, nullable=True)

    def to_pydantic(self) -> ResearchSession:
        return ResearchSession(
            id=self.id,
            query=self.query,
            started_at=self.started_at,
            ended_at=self.ended_at,
            total_tasks=self.total_tasks,
            completed_tasks=self.completed_tasks,
            total_words=self.total_words,
            total_sources=self.total_sources,
            status=self.status,
            executive_summary=self.executive_summary,
            conclusion=self.conclusion,
            report_markdown_path=self.report_markdown_path,
            report_html_path=self.report_html_path,
            refined_brief=self.refined_brief,
            refinement_qa=self.refinement_qa,
        )
