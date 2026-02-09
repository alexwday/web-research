"""
Tests for src.database.DatabaseManager — CRUD operations for
sessions, tasks, sections, sources, glossary terms, and search events.
"""
from datetime import datetime, timezone

import pytest

from src.config import (
    TaskStatus, SectionStatus, ResearchTask, ReportSection,
    Source, GlossaryTerm,
)


# =========================================================================
# Session operations
# =========================================================================

class TestSessionCRUD:
    def test_create_session(self, db):
        session = db.create_session("Test query")
        assert session.id is not None
        assert session.query == "Test query"
        assert session.status == "running"

    def test_get_session_by_id(self, db):
        session = db.create_session("Query A")
        fetched = db.get_session_by_id(session.id)
        assert fetched is not None
        assert fetched.query == "Query A"

    def test_get_session_by_id_nonexistent(self, db):
        result = db.get_session_by_id(9999)
        assert result is None

    def test_get_current_session(self, db):
        db.create_session("Running session")
        current = db.get_current_session()
        assert current is not None
        assert current.status == "running"

    def test_get_current_session_none_when_completed(self, db):
        session = db.create_session("Will complete")
        db.complete_session(session.id)
        assert db.get_current_session() is None

    def test_get_all_sessions(self, db):
        db.create_session("First")
        db.create_session("Second")
        sessions = db.get_all_sessions()
        assert len(sessions) == 2

    def test_get_most_recent_session(self, db):
        db.create_session("Old")
        newest = db.create_session("New")
        recent = db.get_most_recent_session()
        assert recent.id == newest.id

    def test_update_session(self, db):
        session = db.create_session("Update me")
        db.update_session(session.id, total_tasks=5, total_words=1000)
        updated = db.get_session_by_id(session.id)
        assert updated.total_tasks == 5
        assert updated.total_words == 1000

    def test_complete_session(self, db):
        session = db.create_session("Complete me")
        db.complete_session(session.id)
        completed = db.get_session_by_id(session.id)
        assert completed.status == "completed"
        assert completed.ended_at is not None


# =========================================================================
# Task operations
# =========================================================================

class TestTaskCRUD:
    def _make_task(self, **kwargs):
        defaults = dict(
            topic="Test topic",
            description="Test description",
            file_path="/tmp/test_task.md",
            priority=5,
        )
        defaults.update(kwargs)
        return ResearchTask(**defaults)

    def test_add_task(self, db):
        session = db.create_session("Q")
        task = db.add_task(self._make_task(), session_id=session.id)
        assert task.id is not None
        assert task.topic == "Test topic"

    def test_add_tasks_bulk(self, db):
        session = db.create_session("Q")
        tasks = db.add_tasks_bulk([
            self._make_task(topic="T1"),
            self._make_task(topic="T2"),
            self._make_task(topic="T3"),
        ], session_id=session.id)
        assert len(tasks) == 3
        assert {t.topic for t in tasks} == {"T1", "T2", "T3"}

    def test_get_task_by_id(self, db):
        session = db.create_session("Q")
        task = db.add_task(self._make_task(), session_id=session.id)
        fetched = db.get_task_by_id(task.id)
        assert fetched is not None
        assert fetched.topic == "Test topic"

    def test_get_next_task(self, db):
        session = db.create_session("Q")
        t1 = db.add_task(self._make_task(topic="Low", priority=3), session_id=session.id)
        t2 = db.add_task(self._make_task(topic="High", priority=8), session_id=session.id)
        nxt = db.get_next_task(session_id=session.id)
        assert nxt.id == t2.id  # highest priority first

    def test_get_next_tasks(self, db):
        session = db.create_session("Q")
        for i in range(5):
            db.add_task(self._make_task(topic=f"T{i}"), session_id=session.id)
        batch = db.get_next_tasks(3, session_id=session.id)
        assert len(batch) == 3
        # All should now be in_progress
        for t in batch:
            fetched = db.get_task_by_id(t.id)
            assert fetched.status == TaskStatus.IN_PROGRESS.value

    def test_mark_task_complete(self, db):
        session = db.create_session("Q")
        task = db.add_task(self._make_task(), session_id=session.id)
        db.mark_task_complete(task.id, word_count=500, citation_count=3)
        fetched = db.get_task_by_id(task.id)
        assert fetched.status == TaskStatus.COMPLETED.value
        assert fetched.word_count == 500
        assert fetched.citation_count == 3

    def test_mark_task_failed(self, db):
        session = db.create_session("Q")
        task = db.add_task(self._make_task(), session_id=session.id)
        db.mark_task_failed(task.id, "Something went wrong")
        fetched = db.get_task_by_id(task.id)
        assert fetched.status == TaskStatus.FAILED.value
        assert fetched.error_message == "Something went wrong"

    def test_get_all_tasks_filtered_by_status(self, db):
        session = db.create_session("Q")
        t1 = db.add_task(self._make_task(topic="Pending"), session_id=session.id)
        t2 = db.add_task(self._make_task(topic="Done"), session_id=session.id)
        db.mark_task_complete(t2.id, word_count=100, citation_count=1)

        pending = db.get_all_tasks(TaskStatus.PENDING, session_id=session.id)
        completed = db.get_all_tasks(TaskStatus.COMPLETED, session_id=session.id)
        assert len(pending) == 1
        assert len(completed) == 1
        assert pending[0].topic == "Pending"

    def test_get_task_count(self, db):
        session = db.create_session("Q")
        db.add_tasks_bulk([self._make_task(topic=f"T{i}") for i in range(4)], session_id=session.id)
        assert db.get_task_count(session_id=session.id) == 4
        assert db.get_task_count(TaskStatus.PENDING, session_id=session.id) == 4
        assert db.get_task_count(TaskStatus.COMPLETED, session_id=session.id) == 0

    def test_retry_failed_tasks(self, db):
        session = db.create_session("Q")
        task = db.add_task(self._make_task(), session_id=session.id)
        db.mark_task_failed(task.id, "err")
        retried = db.retry_failed_tasks(session.id, max_retries=2)
        assert retried == 1
        fetched = db.get_task_by_id(task.id)
        assert fetched.status == TaskStatus.PENDING.value

    def test_get_total_word_count(self, db):
        session = db.create_session("Q")
        t1 = db.add_task(self._make_task(topic="T1"), session_id=session.id)
        t2 = db.add_task(self._make_task(topic="T2"), session_id=session.id)
        db.mark_task_complete(t1.id, word_count=200, citation_count=1)
        db.mark_task_complete(t2.id, word_count=300, citation_count=2)
        assert db.get_total_word_count(session_id=session.id) == 500


# =========================================================================
# Section operations
# =========================================================================

class TestSectionCRUD:
    def test_add_section(self, db):
        session = db.create_session("Q")
        section = db.add_section(
            ReportSection(title="Intro", description="Introduction", position=1),
            session_id=session.id,
        )
        assert section.id is not None
        assert section.title == "Intro"

    def test_add_sections_bulk(self, db):
        session = db.create_session("Q")
        sections = db.add_sections_bulk([
            ReportSection(title="S1", description="D1", position=1),
            ReportSection(title="S2", description="D2", position=2),
        ], session_id=session.id)
        assert len(sections) == 2

    def test_get_all_sections_ordered(self, db):
        session = db.create_session("Q")
        db.add_section(ReportSection(title="Second", description="D", position=2), session_id=session.id)
        db.add_section(ReportSection(title="First", description="D", position=1), session_id=session.id)
        sections = db.get_all_sections(session_id=session.id)
        assert sections[0].title == "First"
        assert sections[1].title == "Second"

    def test_mark_section_synthesized(self, db):
        session = db.create_session("Q")
        section = db.add_section(
            ReportSection(title="S1", description="D", position=1),
            session_id=session.id,
        )
        db.mark_section_synthesized(section.id, "Synthesized content here.", 3, 2)
        updated = db.get_all_sections(session_id=session.id)[0]
        assert updated.synthesized_content == "Synthesized content here."
        assert updated.word_count == 3
        assert updated.status in (SectionStatus.COMPLETE.value, "complete")

    def test_get_tasks_for_section(self, populated_db):
        """Verify tasks are correctly associated with their section."""
        db_mod = __import__("src.database", fromlist=["get_database"])
        db = db_mod.get_database()
        section = populated_db.sections[0]
        tasks = db.get_tasks_for_section(section.id)
        assert len(tasks) == 2
        for t in tasks:
            assert section.title in t.topic


# =========================================================================
# Source operations
# =========================================================================

class TestSourceCRUD:
    def test_add_and_get_source(self, db):
        session = db.create_session("Q")
        task = db.add_task(
            ResearchTask(topic="T", description="D", file_path="/tmp/t.md"),
            session_id=session.id,
        )
        source = db.add_source(
            Source(url="https://example.com", title="Example", domain="example.com"),
            task_id=task.id,
            position=1,
        )
        assert source.id is not None

    def test_get_source_by_url(self, db):
        session = db.create_session("Q")
        task = db.add_task(
            ResearchTask(topic="T", description="D", file_path="/tmp/t.md"),
            session_id=session.id,
        )
        db.add_source(
            Source(url="https://unique.example.com", title="Unique", domain="example.com"),
            task_id=task.id, position=1,
        )
        found = db.get_source_by_url("https://unique.example.com")
        assert found is not None
        assert found.title == "Unique"

    def test_get_sources_for_session(self, populated_db):
        db_mod = __import__("src.database", fromlist=["get_database"])
        db = db_mod.get_database()
        sources = db.get_sources_for_session(populated_db.session.id)
        assert len(sources) >= 2

    def test_get_source_count(self, populated_db):
        db_mod = __import__("src.database", fromlist=["get_database"])
        db = db_mod.get_database()
        count = db.get_source_count(populated_db.session.id)
        assert count >= 2


# =========================================================================
# Glossary operations
# =========================================================================

class TestGlossaryCRUD:
    def test_add_glossary_term(self, db):
        session = db.create_session("Q")
        term = db.add_glossary_term(
            GlossaryTerm(term="LLM", definition="Large Language Model"),
            session_id=session.id,
        )
        assert term.id is not None

    def test_get_glossary_terms_for_session(self, populated_db):
        db_mod = __import__("src.database", fromlist=["get_database"])
        db = db_mod.get_database()
        terms = db.get_glossary_terms_for_session(populated_db.session.id)
        assert len(terms) == 1
        assert terms[0].term == "Alignment"


# =========================================================================
# Search events
# =========================================================================

class TestSearchEvents:
    def test_add_and_get_search_events(self, db):
        session = db.create_session("Q")
        task = db.add_task(
            ResearchTask(topic="T", description="D", file_path="/tmp/t.md"),
            session_id=session.id,
        )
        db.add_search_event(
            session_id=session.id,
            task_id=task.id,
            event_type="query",
            query_group="group_1",
            query_text="test query",
        )
        db.add_search_event(
            session_id=session.id,
            task_id=task.id,
            event_type="result",
            query_group="group_1",
            url="https://example.com",
            title="Example Result",
        )
        events = db.get_search_events(session.id)
        assert len(events) == 2

    def test_search_queries_by_task(self, db):
        session = db.create_session("Q")
        task = db.add_task(
            ResearchTask(topic="T", description="D", file_path="/tmp/t.md"),
            session_id=session.id,
        )
        db.add_search_event(
            session_id=session.id,
            task_id=task.id,
            event_type="query",
            query_group="g1",
            query_text="search query 1",
        )
        by_task = db.get_search_queries_by_task(session.id)
        assert task.id in by_task


# =========================================================================
# Statistics
# =========================================================================

class TestStatistics:
    def test_get_statistics(self, populated_db):
        db_mod = __import__("src.database", fromlist=["get_database"])
        db = db_mod.get_database()
        stats = db.get_statistics(session_id=populated_db.session.id)
        assert stats["total_tasks"] == 6  # 2 per section, 3 sections
        assert stats["pending_tasks"] == 6
        assert stats["completed_tasks"] == 0
        assert "total_words" in stats
        assert "total_sources" in stats

    def test_statistics_after_completing_tasks(self, db):
        session = db.create_session("Q")
        t1 = db.add_task(
            ResearchTask(topic="T1", description="D", file_path="/tmp/t1.md"),
            session_id=session.id,
        )
        t2 = db.add_task(
            ResearchTask(topic="T2", description="D", file_path="/tmp/t2.md"),
            session_id=session.id,
        )
        db.mark_task_complete(t1.id, word_count=100, citation_count=2)
        db.mark_task_failed(t2.id, "error")

        stats = db.get_statistics(session_id=session.id)
        assert stats["completed_tasks"] == 1
        assert stats["failed_tasks"] == 1
        assert stats["total_words"] == 100
