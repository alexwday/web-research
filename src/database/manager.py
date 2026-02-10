"""DatabaseManager class — all database operations."""
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine, func, text, or_, and_
from sqlalchemy.orm import sessionmaker

from ..config import (
    get_config, TaskStatus, SectionStatus, ResearchTask, ReportSection,
    Source, GlossaryTerm, ResearchSession
)
from .orm_models import (
    Base, task_source_association,
    TaskModel, SourceModel, GlossaryModel, RunEventModel,
    SectionModel, SessionModel,
)


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
        self._migrate_task_columns()
        self._migrate_section_tables()
        self._migrate_refinement_columns()
        self._migrate_source_columns()
        self._migrate_run_events()
        self._migrate_cancel_requested_column()

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
        """Add missing columns to task_source_association for existing databases."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("PRAGMA table_info(task_source_association)")).fetchall()
            existing = {row[1] for row in rows}
            changed = False
            if "position" not in existing:
                conn.execute(text(
                    "ALTER TABLE task_source_association ADD COLUMN position INTEGER DEFAULT 0"
                ))
                changed = True
            if "extracted_content" not in existing:
                conn.execute(text(
                    "ALTER TABLE task_source_association ADD COLUMN extracted_content TEXT"
                ))
                changed = True
            if changed:
                conn.commit()

    def _migrate_task_columns(self):
        """Add retry_count column to tasks table for existing databases."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("PRAGMA table_info(tasks)")).fetchall()
            existing = {row[1] for row in rows}
            if "retry_count" not in existing:
                conn.execute(text(
                    "ALTER TABLE tasks ADD COLUMN retry_count INTEGER DEFAULT 0"
                ))
                conn.commit()

    def _migrate_section_tables(self):
        """Add section_id and is_gap_fill columns to tasks for existing databases."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("PRAGMA table_info(tasks)")).fetchall()
            existing = {row[1] for row in rows}
            if "section_id" not in existing:
                conn.execute(text(
                    "ALTER TABLE tasks ADD COLUMN section_id INTEGER REFERENCES sections(id)"
                ))
            if "is_gap_fill" not in existing:
                conn.execute(text(
                    "ALTER TABLE tasks ADD COLUMN is_gap_fill BOOLEAN DEFAULT 0"
                ))
            conn.commit()

    def _migrate_refinement_columns(self):
        """Add refined_brief and refinement_qa columns to sessions for existing databases."""
        new_columns = {
            "refined_brief": "TEXT",
            "refinement_qa": "TEXT",
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

    def _migrate_source_columns(self):
        """Add extracted_content column to sources table for existing databases."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("PRAGMA table_info(sources)")).fetchall()
            existing = {row[1] for row in rows}
            if "extracted_content" not in existing:
                conn.execute(text(
                    "ALTER TABLE sources ADD COLUMN extracted_content TEXT"
                ))
                conn.commit()

    def _migrate_run_events(self):
        """Migrate search_events -> run_events for existing databases."""
        with self.engine.connect() as conn:
            old_exists = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='search_events'"
            )).fetchone()
            if not old_exists:
                return  # Fresh DB or already migrated

            # Copy data (new columns get NULL)
            conn.execute(text("""
                INSERT INTO run_events (id, session_id, task_id, event_type, query_group,
                                        query_text, url, title, snippet, quality_score,
                                        phase, severity, payload_json, created_at)
                SELECT id, session_id, task_id, event_type, query_group,
                       query_text, url, title, snippet, quality_score,
                       NULL, NULL, NULL, created_at
                FROM search_events
            """))
            conn.execute(text("DROP TABLE search_events"))
            conn.commit()

    def _migrate_cancel_requested_column(self):
        """Add cancel_requested_at column to sessions table for existing databases."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("PRAGMA table_info(sessions)")).fetchall()
            existing = {row[1] for row in rows}
            if "cancel_requested_at" not in existing:
                conn.execute(text(
                    "ALTER TABLE sessions ADD COLUMN cancel_requested_at DATETIME"
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
                section_id=task.section_id,
                topic=task.topic,
                description=task.description,
                file_path=task.file_path,
                status=task.status.value if isinstance(task.status, TaskStatus) else task.status,
                priority=task.priority,
                depth=task.depth,
                is_gap_fill=task.is_gap_fill,
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
                    section_id=task.section_id,
                    topic=task.topic,
                    description=task.description,
                    file_path=task.file_path,
                    status=task.status.value if isinstance(task.status, TaskStatus) else task.status,
                    priority=task.priority,
                    depth=task.depth,
                    is_gap_fill=task.is_gap_fill,
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
        """Mark a task as failed and increment retry_count."""
        with self.get_sync_session() as session:
            task = session.query(TaskModel).filter(TaskModel.id == task_id).first()
            if task:
                task.status = TaskStatus.FAILED.value
                task.error_message = error_message
                task.retry_count = (task.retry_count or 0) + 1
                session.commit()

    def retry_failed_tasks(self, session_id: int, max_retries: int = 2) -> int:
        """Reset retryable FAILED tasks back to PENDING.

        Returns the number of tasks reset.
        """
        with self.get_sync_session() as session:
            tasks = session.query(TaskModel).filter(
                TaskModel.status == TaskStatus.FAILED.value,
                TaskModel.session_id == session_id,
                (TaskModel.retry_count == None) | (TaskModel.retry_count < max_retries),  # noqa: E711
            ).all()
            for task in tasks:
                task.status = TaskStatus.PENDING.value
                task.error_message = None
            session.commit()
            return len(tasks)

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

    def get_in_progress_tasks(self, session_id: int = None) -> List[ResearchTask]:
        """Get all currently in-progress tasks."""
        with self.get_sync_session() as session:
            query = session.query(TaskModel).filter(
                TaskModel.status == TaskStatus.IN_PROGRESS.value
            )
            if session_id is not None:
                query = query.filter(TaskModel.session_id == session_id)
            results = query.order_by(TaskModel.id).all()
            return [r.to_pydantic() for r in results]

    def get_next_tasks(self, count: int = 1, session_id: int = None) -> List[ResearchTask]:
        """Atomically claim up to `count` pending tasks by marking them IN_PROGRESS.

        Returns claimed tasks ordered by priority desc, depth asc, id asc.
        """
        with self.get_sync_session() as session:
            query = session.query(TaskModel).filter(
                TaskModel.status == 'pending'
            )
            if session_id is not None:
                query = query.filter(TaskModel.session_id == session_id)
            tasks = query.order_by(
                TaskModel.priority.desc(),
                TaskModel.depth.asc(),
                TaskModel.id.asc()
            ).limit(count).all()

            for task in tasks:
                task.status = TaskStatus.IN_PROGRESS.value

            session.commit()
            return [t.to_pydantic() for t in tasks]

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
    # SECTION OPERATIONS
    # =========================================================================

    def add_section(self, section: ReportSection, session_id: int = None) -> ReportSection:
        """Add a new report section"""
        with self.get_sync_session() as session:
            db_section = SectionModel(
                session_id=session_id,
                title=section.title,
                description=section.description,
                position=section.position,
                status=section.status.value if isinstance(section.status, SectionStatus) else section.status,
                is_gap_fill=section.is_gap_fill,
                created_at=datetime.now(timezone.utc)
            )
            session.add(db_section)
            session.commit()
            session.refresh(db_section)
            return db_section.to_pydantic()

    def add_sections_bulk(self, sections: List[ReportSection], session_id: int = None) -> List[ReportSection]:
        """Add multiple sections at once"""
        with self.get_sync_session() as session:
            db_sections = []
            for sec in sections:
                db_section = SectionModel(
                    session_id=session_id,
                    title=sec.title,
                    description=sec.description,
                    position=sec.position,
                    status=sec.status.value if isinstance(sec.status, SectionStatus) else sec.status,
                    is_gap_fill=sec.is_gap_fill,
                    created_at=datetime.now(timezone.utc)
                )
                session.add(db_section)
                db_sections.append(db_section)
            session.commit()
            for db_section in db_sections:
                session.refresh(db_section)
            return [s.to_pydantic() for s in db_sections]

    def get_all_sections(self, session_id: int = None) -> List[ReportSection]:
        """Get all sections, optionally filtered by session, ordered by position"""
        with self.get_sync_session() as session:
            query = session.query(SectionModel)
            if session_id is not None:
                query = query.filter(SectionModel.session_id == session_id)
            results = query.order_by(SectionModel.position).all()
            return [r.to_pydantic() for r in results]

    def update_section(self, section_id: int, **kwargs) -> bool:
        """Update section fields"""
        with self.get_sync_session() as session:
            for key, value in kwargs.items():
                if isinstance(value, SectionStatus):
                    kwargs[key] = value.value
            result = session.query(SectionModel).filter(
                SectionModel.id == section_id
            ).update(kwargs)
            session.commit()
            return result > 0

    def mark_section_synthesized(self, section_id: int, content: str, word_count: int = 0, citation_count: int = 0):
        """Mark a section as synthesized with its content"""
        self.update_section(
            section_id,
            status=SectionStatus.COMPLETE.value,
            synthesized_content=content,
            word_count=word_count,
            citation_count=citation_count,
            synthesized_at=datetime.now(timezone.utc)
        )

    def get_tasks_for_section(self, section_id: int) -> List[ResearchTask]:
        """Get all tasks assigned to a section"""
        with self.get_sync_session() as session:
            results = session.query(TaskModel).filter(
                TaskModel.section_id == section_id
            ).order_by(TaskModel.id).all()
            return [r.to_pydantic() for r in results]

    def get_sources_for_section(self, section_id: int) -> List[Source]:
        """Get all sources linked to tasks in a section, ordered by position"""
        with self.get_sync_session() as session:
            rows = session.query(
                SourceModel,
                task_source_association.c.task_id,
                task_source_association.c.position,
                task_source_association.c.extracted_content,
            ).join(
                task_source_association,
                SourceModel.id == task_source_association.c.source_id
            ).join(
                TaskModel,
                TaskModel.id == task_source_association.c.task_id
            ).filter(
                TaskModel.section_id == section_id
            ).order_by(
                task_source_association.c.position.asc(),
                task_source_association.c.task_id.asc()
            ).all()

            # Deduplicate by source ID while preserving first appearance order.
            by_source: Dict[int, Source] = {}
            ordered_ids: List[int] = []
            for source_model, task_id, _position, assoc_extracted in rows:
                if source_model.id in by_source:
                    continue
                src = source_model.to_pydantic()
                src.task_ids = [task_id] if task_id is not None else []
                src.extracted_content = assoc_extracted
                by_source[source_model.id] = src
                ordered_ids.append(source_model.id)
            return [by_source[sid] for sid in ordered_ids]

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
                    elif assoc_exists.position != position:
                        # Keep latest prompt ordering stable across retries/re-runs.
                        session.execute(
                            task_source_association.update().where(
                                task_source_association.c.task_id == task_id,
                                task_source_association.c.source_id == existing.id,
                            ).values(position=position)
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

    def update_source_extraction(self, task_id: int, source_id: int, extracted_content: str):
        """Update extracted content for a task-source association row."""
        with self.get_sync_session() as session:
            session.execute(
                task_source_association.update().where(
                    task_source_association.c.task_id == task_id,
                    task_source_association.c.source_id == source_id,
                ).values(extracted_content=extracted_content)
            )
            session.commit()

    def get_processed_urls_by_task(self, session_id: int) -> Dict[int, set[str]]:
        """Return {task_id: {url, ...}} for session results that have been processed."""
        processed: Dict[int, set[str]] = {}
        with self.get_sync_session() as session:
            rows = session.query(
                TaskModel.id,
                SourceModel.url,
                task_source_association.c.extracted_content,
                SourceModel.full_content,
                SourceModel.snippet,
            ).join(
                task_source_association,
                TaskModel.id == task_source_association.c.task_id
            ).join(
                SourceModel,
                SourceModel.id == task_source_association.c.source_id
            ).filter(
                TaskModel.session_id == session_id
            ).all()

            for task_id, url, assoc_extracted, full_content, snippet in rows:
                if not url:
                    continue
                if any([
                    bool((assoc_extracted or "").strip()),
                    bool((full_content or "").strip()),
                    bool((snippet or "").strip()),
                ]):
                    processed.setdefault(task_id, set()).add(url)
        return processed

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
    # RUN EVENT OPERATIONS
    # =========================================================================

    def add_run_event(self, **kwargs) -> None:
        """Insert a run event (query, result, phase_changed, etc.)."""
        with self.get_sync_session() as session:
            event = RunEventModel(**kwargs)
            session.add(event)
            session.commit()

    def get_run_events(self, session_id: int) -> List[RunEventModel]:
        """Get all run events for a session, ordered by created_at."""
        with self.get_sync_session() as session:
            return session.query(RunEventModel).filter(
                RunEventModel.session_id == session_id
            ).order_by(RunEventModel.created_at).all()

    def get_run_events_paginated(
        self,
        session_id: int,
        cursor_created_at: Optional[datetime] = None,
        cursor_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[RunEventModel]:
        """Get run events with keyset pagination using (created_at, id).

        Leverages the ix_run_events_timeline index for efficient paging.
        """
        with self.get_sync_session() as session:
            query = session.query(RunEventModel).filter(
                RunEventModel.session_id == session_id
            )
            if cursor_created_at is not None and cursor_id is not None:
                query = query.filter(
                    or_(
                        RunEventModel.created_at > cursor_created_at,
                        and_(
                            RunEventModel.created_at == cursor_created_at,
                            RunEventModel.id > cursor_id,
                        ),
                    )
                )
            return query.order_by(
                RunEventModel.created_at, RunEventModel.id
            ).limit(limit).all()

    def get_rejected_results(self, session_id: int) -> List[Dict[str, Any]]:
        """Return result events whose URLs were NOT saved as sources.

        These are results that were quality-rejected during scraping/filtering.
        Returns list of dicts with url, title, snippet, quality_score, task_id, query_group.
        """
        with self.get_sync_session() as session:
            # Get all source URLs for this session
            source_urls = set()
            src_rows = session.query(SourceModel.url).join(
                task_source_association,
                SourceModel.id == task_source_association.c.source_id
            ).join(
                TaskModel,
                TaskModel.id == task_source_association.c.task_id
            ).filter(TaskModel.session_id == session_id).all()
            for (url,) in src_rows:
                source_urls.add(url)

            # Get all result events not in source_urls
            events = session.query(RunEventModel).filter(
                RunEventModel.session_id == session_id,
                RunEventModel.event_type == "result",
                RunEventModel.url != None,  # noqa: E711
            ).order_by(RunEventModel.created_at).all()

            rejected = []
            seen_urls = set()
            for ev in events:
                if ev.url in source_urls or ev.url in seen_urls:
                    continue
                seen_urls.add(ev.url)
                rejected.append({
                    "url": ev.url,
                    "title": ev.title or "Untitled",
                    "snippet": ev.snippet or "",
                    "quality_score": ev.quality_score,
                    "task_id": ev.task_id,
                    "query_group": ev.query_group,
                })
            return rejected

    def get_run_queries_by_task(self, session_id: int) -> Dict[int, List[Dict]]:
        """Return {task_id: [{query_text, query_group}, ...]} for query events in a session."""
        result: Dict[int, List[Dict]] = {}
        with self.get_sync_session() as session:
            events = session.query(RunEventModel).filter(
                RunEventModel.session_id == session_id,
                RunEventModel.task_id != None,  # noqa: E711
                RunEventModel.event_type == "query",
            ).order_by(RunEventModel.created_at).all()
            for ev in events:
                result.setdefault(ev.task_id, []).append({
                    "query_text": ev.query_text or "",
                    "query_group": ev.query_group or "",
                })
        return result

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
            rows = session.query(
                SourceModel,
                task_source_association.c.extracted_content,
            ).join(
                task_source_association,
                SourceModel.id == task_source_association.c.source_id
            ).filter(
                task_source_association.c.task_id == task_id
            ).order_by(task_source_association.c.position).all()

            sources: List[Source] = []
            for source_model, assoc_extracted in rows:
                src = source_model.to_pydantic()
                src.task_ids = [task_id]
                src.extracted_content = assoc_extracted
                sources.append(src)
            return sources

    def get_sources_for_session(self, session_id: int) -> List[Source]:
        """Get sources linked to tasks in a specific session"""
        with self.get_sync_session() as session:
            rows = session.query(
                SourceModel,
                task_source_association.c.task_id,
                task_source_association.c.extracted_content,
            ).join(
                task_source_association,
                SourceModel.id == task_source_association.c.source_id
            ).join(
                TaskModel,
                TaskModel.id == task_source_association.c.task_id
            ).filter(
                TaskModel.session_id == session_id
            ).order_by(SourceModel.id, task_source_association.c.task_id).all()

            by_source: Dict[int, Source] = {}
            ordered_ids: List[int] = []
            for source_model, task_id, assoc_extracted in rows:
                if source_model.id not in by_source:
                    src = source_model.to_pydantic()
                    src.task_ids = []
                    src.extracted_content = None
                    by_source[source_model.id] = src
                    ordered_ids.append(source_model.id)

                src = by_source[source_model.id]
                if task_id is not None and task_id not in src.task_ids:
                    src.task_ids.append(task_id)
                if not src.extracted_content and assoc_extracted:
                    src.extracted_content = assoc_extracted

            return [by_source[sid] for sid in ordered_ids]
