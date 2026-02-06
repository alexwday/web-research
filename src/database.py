"""
Database Module for Deep Research Agent
Handles all state persistence using SQLite
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, Boolean,
    DateTime, ForeignKey, Table, func, text
)
from sqlalchemy.orm import declarative_base, relationship, backref, sessionmaker

from .config import (
    get_config, TaskStatus, ResearchTask, Source, GlossaryTerm, ResearchSession
)

Base = declarative_base()


# =============================================================================
# ORM MODELS
# =============================================================================

# Association table for tasks and sources
task_source_association = Table(
    'task_source_association',
    Base.metadata,
    Column('task_id', Integer, ForeignKey('tasks.id'), primary_key=True),
    Column('source_id', Integer, ForeignKey('sources.id'), primary_key=True),
    Column('position', Integer, default=0),
)


class TaskModel(Base):
    """SQLAlchemy model for research tasks"""
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    parent_id = Column(Integer, ForeignKey('tasks.id'), nullable=True)
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=True)
    topic = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    file_path = Column(String(500), nullable=False)
    status = Column(String(20), default='pending')
    priority = Column(Integer, default=5)
    depth = Column(Integer, default=0)
    word_count = Column(Integer, default=0)
    citation_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    children = relationship("TaskModel", backref=backref("parent", remote_side="TaskModel.id"), foreign_keys=[parent_id])
    sources = relationship("SourceModel", secondary=task_source_association, back_populates="tasks")
    
    def to_pydantic(self) -> ResearchTask:
        return ResearchTask(
            id=self.id,
            parent_id=self.parent_id,
            session_id=self.session_id,
            topic=self.topic,
            description=self.description,
            file_path=self.file_path,
            status=TaskStatus(self.status),
            priority=self.priority,
            depth=self.depth,
            word_count=self.word_count,
            citation_count=self.citation_count,
            created_at=self.created_at,
            completed_at=self.completed_at,
            error_message=self.error_message
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


class SearchEventModel(Base):
    """SQLAlchemy model for search activity events (queries and results)"""
    __tablename__ = 'search_events'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=True)
    task_id = Column(Integer, ForeignKey('tasks.id'), nullable=True)  # null = planning phase
    event_type = Column(String(20), nullable=False)  # "query" or "result"
    query_group = Column(String(50), nullable=False)  # links results to their parent query
    query_text = Column(String(1000), nullable=True)
    url = Column(String(2000), nullable=True)
    title = Column(String(500), nullable=True)
    snippet = Column(Text, nullable=True)
    quality_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


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
        )


# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, db_path: str = None):
        config = get_config()
        self.db_path = db_path or config.database.path
        self.wal_mode = config.database.wal_mode
        
        # Create sync engine for initialization
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            connect_args={"check_same_thread": False}
        )
        
        # Session factory
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Create all tables
        Base.metadata.create_all(self.engine)

        # Enable WAL mode for better concurrency
        if self.wal_mode:
            with self.engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.commit()

        # Run migrations for existing databases
        self._migrate_session_columns()
        self._migrate_glossary_columns()
        self._migrate_association_columns()

    def _migrate_session_columns(self):
        """Add missing columns to the sessions table for existing databases."""
        new_columns = {
            "executive_summary": "TEXT",
            "conclusion": "TEXT",
            "report_markdown_path": "VARCHAR(500)",
            "report_html_path": "VARCHAR(500)",
        }
        with self.engine.connect() as conn:
            rows = conn.execute(text("PRAGMA table_info(sessions)")).fetchall()
            existing = {row[1] for row in rows}
            for col_name, col_type in new_columns.items():
                if col_name not in existing:
                    conn.execute(text(
                        f"ALTER TABLE sessions ADD COLUMN {col_name} {col_type}"
                    ))
            conn.commit()
    
    def _migrate_glossary_columns(self):
        """Add session_id column to glossary table for existing databases."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("PRAGMA table_info(glossary)")).fetchall()
            existing = {row[1] for row in rows}
            if "session_id" not in existing:
                conn.execute(text(
                    "ALTER TABLE glossary ADD COLUMN session_id INTEGER REFERENCES sessions(id)"
                ))
                conn.commit()

    def _migrate_association_columns(self):
        """Add position column to task_source_association for existing databases."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("PRAGMA table_info(task_source_association)")).fetchall()
            existing = {row[1] for row in rows}
            if "position" not in existing:
                conn.execute(text(
                    "ALTER TABLE task_source_association ADD COLUMN position INTEGER DEFAULT 0"
                ))
                conn.commit()

    def get_sync_session(self):
        """Get a synchronous session"""
        return self.Session()
    
    # =========================================================================
    # SESSION OPERATIONS
    # =========================================================================
    
    def create_session(self, query: str) -> ResearchSession:
        """Create a new research session"""
        with self.get_sync_session() as session:
            db_session = SessionModel(query=query)
            session.add(db_session)
            session.commit()
            session.refresh(db_session)
            return db_session.to_pydantic()
    
    def get_current_session(self) -> Optional[ResearchSession]:
        """Get the current running session"""
        with self.get_sync_session() as session:
            result = session.query(SessionModel).filter(
                SessionModel.status == 'running'
            ).order_by(SessionModel.started_at.desc()).first()
            return result.to_pydantic() if result else None
    
    def get_session_by_id(self, session_id: int) -> Optional[ResearchSession]:
        """Get a session by ID"""
        with self.get_sync_session() as session:
            result = session.query(SessionModel).filter(
                SessionModel.id == session_id
            ).first()
            return result.to_pydantic() if result else None
    
    def update_session(self, session_id: int, **kwargs) -> bool:
        """Update session fields"""
        with self.get_sync_session() as session:
            result = session.query(SessionModel).filter(
                SessionModel.id == session_id
            ).update(kwargs)
            session.commit()
            return result > 0
    
    def complete_session(self, session_id: int):
        """Mark a session as completed"""
        self.update_session(
            session_id,
            status='completed',
            ended_at=datetime.now(timezone.utc)
        )
    
    # =========================================================================
    # TASK OPERATIONS
    # =========================================================================
    
    def add_task(self, task: ResearchTask, session_id: int = None) -> ResearchTask:
        """Add a new task"""
        with self.get_sync_session() as session:
            db_task = TaskModel(
                parent_id=task.parent_id,
                session_id=session_id,
                topic=task.topic,
                description=task.description,
                file_path=task.file_path,
                status=task.status.value if isinstance(task.status, TaskStatus) else task.status,
                priority=task.priority,
                depth=task.depth,
                created_at=datetime.now(timezone.utc)
            )
            session.add(db_task)
            session.commit()
            session.refresh(db_task)
            return db_task.to_pydantic()
    
    def add_tasks_bulk(self, tasks: List[ResearchTask], session_id: int = None) -> List[ResearchTask]:
        """Add multiple tasks at once"""
        with self.get_sync_session() as session:
            db_tasks = []
            for task in tasks:
                db_task = TaskModel(
                    parent_id=task.parent_id,
                    session_id=session_id,
                    topic=task.topic,
                    description=task.description,
                    file_path=task.file_path,
                    status=task.status.value if isinstance(task.status, TaskStatus) else task.status,
                    priority=task.priority,
                    depth=task.depth,
                    created_at=datetime.now(timezone.utc)
                )
                session.add(db_task)
                db_tasks.append(db_task)
            
            session.commit()
            
            # Refresh all to get IDs
            for db_task in db_tasks:
                session.refresh(db_task)
            
            return [t.to_pydantic() for t in db_tasks]
    
    def get_next_task(self, session_id: int = None) -> Optional[ResearchTask]:
        """Get the next pending task (highest priority first)"""
        with self.get_sync_session() as session:
            query = session.query(TaskModel).filter(
                TaskModel.status == 'pending'
            )
            if session_id is not None:
                query = query.filter(TaskModel.session_id == session_id)
            result = query.order_by(
                TaskModel.priority.desc(),
                TaskModel.depth.asc(),
                TaskModel.id.asc()
            ).first()
            return result.to_pydantic() if result else None
    
    def get_task_by_id(self, task_id: int) -> Optional[ResearchTask]:
        """Get a task by ID"""
        with self.get_sync_session() as session:
            result = session.query(TaskModel).filter(
                TaskModel.id == task_id
            ).first()
            return result.to_pydantic() if result else None
    
    def get_all_tasks(self, status: TaskStatus = None, session_id: int = None) -> List[ResearchTask]:
        """Get all tasks, optionally filtered by status and/or session"""
        with self.get_sync_session() as session:
            query = session.query(TaskModel)
            if status:
                query = query.filter(TaskModel.status == status.value)
            if session_id is not None:
                query = query.filter(TaskModel.session_id == session_id)
            results = query.order_by(TaskModel.id).all()
            return [r.to_pydantic() for r in results]
    
    def update_task(self, task_id: int, **kwargs) -> bool:
        """Update task fields"""
        with self.get_sync_session() as session:
            # Convert enums to values
            for key, value in kwargs.items():
                if isinstance(value, TaskStatus):
                    kwargs[key] = value.value
            
            result = session.query(TaskModel).filter(
                TaskModel.id == task_id
            ).update(kwargs)
            session.commit()
            return result > 0
    
    def mark_task_complete(self, task_id: int, word_count: int = 0, citation_count: int = 0):
        """Mark a task as completed"""
        self.update_task(
            task_id,
            status=TaskStatus.COMPLETED.value,
            completed_at=datetime.now(timezone.utc),
            word_count=word_count,
            citation_count=citation_count
        )
    
    def mark_task_failed(self, task_id: int, error_message: str):
        """Mark a task as failed"""
        self.update_task(
            task_id,
            status=TaskStatus.FAILED.value,
            error_message=error_message
        )
    
    def get_task_count(self, status: TaskStatus = None, session_id: int = None) -> int:
        """Get count of tasks"""
        with self.get_sync_session() as session:
            query = session.query(func.count(TaskModel.id))
            if status:
                query = query.filter(TaskModel.status == status.value)
            if session_id is not None:
                query = query.filter(TaskModel.session_id == session_id)
            return query.scalar() or 0
    
    def get_total_word_count(self, session_id: int = None) -> int:
        """Get total word count across all completed tasks"""
        with self.get_sync_session() as session:
            query = session.query(func.sum(TaskModel.word_count)).filter(
                TaskModel.status == TaskStatus.COMPLETED.value
            )
            if session_id is not None:
                query = query.filter(TaskModel.session_id == session_id)
            result = query.scalar()
            return result or 0
    
    def get_in_progress_task(self, session_id: int = None) -> Optional[ResearchTask]:
        """Get the currently in-progress task (if any)."""
        with self.get_sync_session() as session:
            query = session.query(TaskModel).filter(
                TaskModel.status == TaskStatus.IN_PROGRESS.value
            )
            if session_id is not None:
                query = query.filter(TaskModel.session_id == session_id)
            result = query.first()
            return result.to_pydantic() if result else None

    def get_recent_completed_tasks(self, limit: int = 5, session_id: int = None) -> List[ResearchTask]:
        """Get the most recently completed tasks, ordered newest first."""
        with self.get_sync_session() as session:
            query = session.query(TaskModel).filter(
                TaskModel.status == TaskStatus.COMPLETED.value
            )
            if session_id is not None:
                query = query.filter(TaskModel.session_id == session_id)
            results = query.order_by(TaskModel.completed_at.desc()).limit(limit).all()
            return [r.to_pydantic() for r in results]

    # =========================================================================
    # SOURCE OPERATIONS
    # =========================================================================
    
    def add_source(self, source: Source, task_id: int = None, position: int = 0) -> Source:
        """Add a new source, linking it to a task with a presentation position."""
        with self.get_sync_session() as session:
            # Check if source already exists
            existing = session.query(SourceModel).filter(
                SourceModel.url == source.url
            ).first()

            if existing:
                # Link to task if provided (with position for citation ordering)
                if task_id:
                    assoc_exists = session.execute(
                        task_source_association.select().where(
                            task_source_association.c.task_id == task_id,
                            task_source_association.c.source_id == existing.id,
                        )
                    ).first()
                    if not assoc_exists:
                        session.execute(
                            task_source_association.insert().values(
                                task_id=task_id,
                                source_id=existing.id,
                                position=position,
                            )
                        )
                        session.commit()
                return existing.to_pydantic()

            # Create new source
            db_source = SourceModel(
                url=source.url,
                title=source.title,
                domain=source.domain,
                snippet=source.snippet,
                full_content=source.full_content,
                quality_score=source.quality_score,
                is_academic=source.is_academic,
                accessed_at=datetime.now(timezone.utc)
            )
            session.add(db_source)
            session.flush()  # get the id before inserting association

            # Link to task with position
            if task_id:
                session.execute(
                    task_source_association.insert().values(
                        task_id=task_id,
                        source_id=db_source.id,
                        position=position,
                    )
                )

            session.commit()
            session.refresh(db_source)
            return db_source.to_pydantic()
    
    def get_all_sources(self) -> List[Source]:
        """Get all sources"""
        with self.get_sync_session() as session:
            results = session.query(SourceModel).order_by(SourceModel.id).all()
            return [r.to_pydantic() for r in results]
    
    def get_source_by_url(self, url: str) -> Optional[Source]:
        """Get a source by URL"""
        with self.get_sync_session() as session:
            result = session.query(SourceModel).filter(
                SourceModel.url == url
            ).first()
            return result.to_pydantic() if result else None
    
    def get_source_count(self, session_id: int = None) -> int:
        """Get count of sources"""
        with self.get_sync_session() as session:
            if session_id is not None:
                return session.query(func.count(SourceModel.id.distinct())).join(
                    task_source_association,
                    SourceModel.id == task_source_association.c.source_id
                ).join(
                    TaskModel,
                    TaskModel.id == task_source_association.c.task_id
                ).filter(
                    TaskModel.session_id == session_id
                ).scalar() or 0
            return session.query(func.count(SourceModel.id)).scalar() or 0
    
    # =========================================================================
    # GLOSSARY OPERATIONS
    # =========================================================================
    
    def add_glossary_term(self, term: GlossaryTerm, session_id: int = None) -> GlossaryTerm:
        """Add a glossary term"""
        with self.get_sync_session() as session:
            # Check if exists
            existing = session.query(GlossaryModel).filter(
                GlossaryModel.term.ilike(term.term)
            ).first()

            if existing:
                return existing.to_pydantic()

            db_term = GlossaryModel(
                term=term.term,
                definition=term.definition,
                first_occurrence_task_id=term.first_occurrence_task_id,
                session_id=session_id
            )
            session.add(db_term)
            session.commit()
            session.refresh(db_term)
            return db_term.to_pydantic()
    
    def get_all_glossary_terms(self) -> List[GlossaryTerm]:
        """Get all glossary terms"""
        with self.get_sync_session() as session:
            results = session.query(GlossaryModel).order_by(GlossaryModel.term).all()
            return [r.to_pydantic() for r in results]

    def get_glossary_terms_for_session(self, session_id: int) -> List[GlossaryTerm]:
        """Get glossary terms scoped to a specific session"""
        with self.get_sync_session() as session:
            results = session.query(GlossaryModel).filter(
                GlossaryModel.session_id == session_id
            ).order_by(GlossaryModel.term).all()
            return [r.to_pydantic() for r in results]
    
    # =========================================================================
    # SEARCH EVENT OPERATIONS
    # =========================================================================

    def add_search_event(self, **kwargs) -> None:
        """Insert a search event (query or result)."""
        with self.get_sync_session() as session:
            event = SearchEventModel(**kwargs)
            session.add(event)
            session.commit()

    def get_search_events(self, session_id: int) -> List[SearchEventModel]:
        """Get all search events for a session, ordered by created_at."""
        with self.get_sync_session() as session:
            return session.query(SearchEventModel).filter(
                SearchEventModel.session_id == session_id
            ).order_by(SearchEventModel.created_at).all()

    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_statistics(self, session_id: int = None) -> Dict[str, Any]:
        """Get research statistics, optionally scoped to a session"""
        if session_id is not None:
            glossary_count = len(self.get_glossary_terms_for_session(session_id))
        else:
            glossary_count = len(self.get_all_glossary_terms())
        return {
            "total_tasks": self.get_task_count(session_id=session_id),
            "pending_tasks": self.get_task_count(TaskStatus.PENDING, session_id=session_id),
            "completed_tasks": self.get_task_count(TaskStatus.COMPLETED, session_id=session_id),
            "failed_tasks": self.get_task_count(TaskStatus.FAILED, session_id=session_id),
            "total_sources": self.get_source_count(session_id=session_id),
            "total_words": self.get_total_word_count(session_id=session_id),
            "glossary_terms": glossary_count
        }

    # =========================================================================
    # SESSION LISTING & SOURCE SCOPING
    # =========================================================================

    def get_all_sessions(self) -> List[ResearchSession]:
        """Get all sessions ordered by started_at DESC"""
        with self.get_sync_session() as session:
            results = session.query(SessionModel).order_by(
                SessionModel.started_at.desc()
            ).all()
            return [r.to_pydantic() for r in results]

    def get_most_recent_session(self) -> Optional[ResearchSession]:
        """Get the most recent session regardless of status"""
        with self.get_sync_session() as session:
            result = session.query(SessionModel).order_by(
                SessionModel.started_at.desc()
            ).first()
            return result.to_pydantic() if result else None

    def get_sources_for_task(self, task_id: int) -> List[Source]:
        """Get sources linked to a specific task, ordered by presentation position."""
        with self.get_sync_session() as session:
            results = session.query(SourceModel).join(
                task_source_association,
                SourceModel.id == task_source_association.c.source_id
            ).filter(
                task_source_association.c.task_id == task_id
            ).order_by(task_source_association.c.position).all()
            return [r.to_pydantic() for r in results]

    def get_sources_for_session(self, session_id: int) -> List[Source]:
        """Get sources linked to tasks in a specific session"""
        with self.get_sync_session() as session:
            results = session.query(SourceModel).join(
                task_source_association,
                SourceModel.id == task_source_association.c.source_id
            ).join(
                TaskModel,
                TaskModel.id == task_source_association.c.task_id
            ).filter(
                TaskModel.session_id == session_id
            ).distinct().order_by(SourceModel.id).all()
            return [r.to_pydantic() for r in results]


# Global database manager instance
_db: Optional[DatabaseManager] = None
_db_lock = __import__('threading').Lock()


def get_database() -> DatabaseManager:
    """Get global database instance"""
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                _db = DatabaseManager()
    return _db


def reset_database():
    """Reset the global database instance"""
    global _db
    _db = None
