"""
Database Module for Deep Research Agent
Handles all state persistence using SQLite
"""
import threading
from typing import Optional

from .orm_models import (
    Base,
    task_source_association,
    TaskModel,
    SourceModel,
    GlossaryModel,
    RunEventModel,
    SectionModel,
    SessionModel,
)
from .manager import DatabaseManager


# Global database manager instance
_db: Optional[DatabaseManager] = None
_db_lock = threading.Lock()


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
