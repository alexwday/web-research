"""
Database Module for Deep Research Agent
Handles all state persistence using SQLite with async support
"""
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, Boolean,
    DateTime, ForeignKey, Enum, Table, select, update, delete, func, text
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

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
    Column('source_id', Integer, ForeignKey('sources.id'), primary_key=True)
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
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    children = relationship("TaskModel", backref="parent", remote_side=[id])
    sources = relationship("SourceModel", secondary=task_source_association, back_populates="tasks")
    
    def to_pydantic(self) -> ResearchTask:
        return ResearchTask(
            id=self.id,
            parent_id=self.parent_id,
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
    full_content = Column(Text, nullable=True)
    quality_score = Column(Float, default=0.5)
    is_academic = Column(Boolean, default=False)
    accessed_at = Column(DateTime, default=datetime.utcnow)
    
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
    
    def to_pydantic(self) -> GlossaryTerm:
        return GlossaryTerm(
            id=self.id,
            term=self.term,
            definition=self.definition,
            first_occurrence_task_id=self.first_occurrence_task_id
        )


class SessionModel(Base):
    """SQLAlchemy model for research sessions"""
    __tablename__ = 'sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    total_tasks = Column(Integer, default=0)
    completed_tasks = Column(Integer, default=0)
    total_words = Column(Integer, default=0)
    total_sources = Column(Integer, default=0)
    status = Column(String(20), default='running')
    
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
            status=self.status
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
    
    @asynccontextmanager
    async def get_session(self):
        """Get a database session"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
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
            ended_at=datetime.utcnow()
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
                created_at=datetime.utcnow()
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
                    created_at=datetime.utcnow()
                )
                session.add(db_task)
                db_tasks.append(db_task)
            
            session.commit()
            
            # Refresh all to get IDs
            for db_task in db_tasks:
                session.refresh(db_task)
            
            return [t.to_pydantic() for t in db_tasks]
    
    def get_next_task(self) -> Optional[ResearchTask]:
        """Get the next pending task (highest priority first)"""
        with self.get_sync_session() as session:
            result = session.query(TaskModel).filter(
                TaskModel.status == 'pending'
            ).order_by(
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
    
    def get_all_tasks(self, status: TaskStatus = None) -> List[ResearchTask]:
        """Get all tasks, optionally filtered by status"""
        with self.get_sync_session() as session:
            query = session.query(TaskModel)
            if status:
                query = query.filter(TaskModel.status == status.value)
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
            completed_at=datetime.utcnow(),
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
    
    def get_task_count(self, status: TaskStatus = None) -> int:
        """Get count of tasks"""
        with self.get_sync_session() as session:
            query = session.query(func.count(TaskModel.id))
            if status:
                query = query.filter(TaskModel.status == status.value)
            return query.scalar() or 0
    
    def get_total_word_count(self) -> int:
        """Get total word count across all completed tasks"""
        with self.get_sync_session() as session:
            result = session.query(func.sum(TaskModel.word_count)).filter(
                TaskModel.status == TaskStatus.COMPLETED.value
            ).scalar()
            return result or 0
    
    # =========================================================================
    # SOURCE OPERATIONS
    # =========================================================================
    
    def add_source(self, source: Source, task_id: int = None) -> Source:
        """Add a new source"""
        with self.get_sync_session() as session:
            # Check if source already exists
            existing = session.query(SourceModel).filter(
                SourceModel.url == source.url
            ).first()
            
            if existing:
                # Link to task if provided
                if task_id:
                    task = session.query(TaskModel).get(task_id)
                    if task and task not in existing.tasks:
                        existing.tasks.append(task)
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
                accessed_at=datetime.utcnow()
            )
            
            # Link to task if provided
            if task_id:
                task = session.query(TaskModel).get(task_id)
                if task:
                    db_source.tasks.append(task)
            
            session.add(db_source)
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
    
    def get_source_count(self) -> int:
        """Get count of sources"""
        with self.get_sync_session() as session:
            return session.query(func.count(SourceModel.id)).scalar() or 0
    
    # =========================================================================
    # GLOSSARY OPERATIONS
    # =========================================================================
    
    def add_glossary_term(self, term: GlossaryTerm) -> GlossaryTerm:
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
                first_occurrence_task_id=term.first_occurrence_task_id
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
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get research statistics"""
        return {
            "total_tasks": self.get_task_count(),
            "pending_tasks": self.get_task_count(TaskStatus.PENDING),
            "completed_tasks": self.get_task_count(TaskStatus.COMPLETED),
            "failed_tasks": self.get_task_count(TaskStatus.FAILED),
            "total_sources": self.get_source_count(),
            "total_words": self.get_total_word_count(),
            "glossary_terms": len(self.get_all_glossary_terms())
        }


# Global database manager instance
_db: Optional[DatabaseManager] = None


def get_database() -> DatabaseManager:
    """Get global database instance"""
    global _db
    if _db is None:
        _db = DatabaseManager()
    return _db


def reset_database():
    """Reset the global database instance"""
    global _db
    _db = None
