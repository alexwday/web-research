"""
Shared fixtures for the Deep Research Agent test suite.

Provides isolated test environments with:
- Temporary database (SQLite file per test)
- Temporary output directories
- Config singleton management
- Database singleton management
"""
import os
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path so `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Core fixtures: config + database isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_environment(tmp_path):
    """Auto-use fixture that gives every test its own config + database.

    Resets the global singletons in src.config and src.database so tests
    never leak state into each other.
    """
    from src.config.settings import Config, set_config
    from src.infra import _database as db_mod
    from src.pipeline import service as svc_mod

    db_file = str(tmp_path / "test_research.db")
    output_dir = str(tmp_path / "report")

    config = Config()
    config.database.path = db_file
    config.database.wal_mode = False  # WAL unnecessary for tests
    config.output.directory = output_dir
    config.output.formats = ["markdown", "html"]
    # Use quick-like settings for fast tests
    config.research.max_total_tasks = 10
    config.research.max_loops = 10
    config.research.max_runtime_hours = 1
    config.research.max_concurrent_tasks = 1
    config.research.tasks_per_section = 2
    config.gap_analysis.enabled = False

    set_config(config)

    # Reset database singleton so it picks up the new config
    with db_mod._db_lock:
        db_mod._db = None

    # Reset service singleton so it doesn't carry state across tests
    with svc_mod._service_lock:
        svc_mod._service = None

    yield config

    # Teardown: reset singletons
    with db_mod._db_lock:
        db_mod._db = None
    with svc_mod._service_lock:
        svc_mod._service = None
    set_config(Config())  # restore pristine defaults


@pytest.fixture
def db():
    """Return a DatabaseManager backed by the test database."""
    from src.infra._database import get_database
    return get_database()


@pytest.fixture
def populated_db(db):
    """Database pre-populated with a session, sections, tasks, and sources.

    Returns a namespace with the created objects for easy reference.
    """
    from types import SimpleNamespace
    from src.config.types import ResearchTask, ReportSection, Source, GlossaryTerm, TaskStatus

    # Session
    session = db.create_session("Test research query about AI safety")

    # Sections
    sections = db.add_sections_bulk([
        ReportSection(title="Introduction to AI Safety", description="Overview of the field", position=1),
        ReportSection(title="Key Challenges", description="Major open problems", position=2),
        ReportSection(title="Future Directions", description="Where the field is heading", position=3),
    ], session_id=session.id)

    # Tasks (2 per section)
    tasks = []
    for i, section in enumerate(sections):
        section_tasks = db.add_tasks_bulk([
            ResearchTask(
                topic=f"Task {i*2+1} for {section.title}",
                description=f"Research subtopic {i*2+1}",
                file_path=f"/tmp/task_{i*2+1}.md",
                section_id=section.id,
                priority=5,
            ),
            ResearchTask(
                topic=f"Task {i*2+2} for {section.title}",
                description=f"Research subtopic {i*2+2}",
                file_path=f"/tmp/task_{i*2+2}.md",
                section_id=section.id,
                priority=5,
            ),
        ], session_id=session.id)
        tasks.extend(section_tasks)

    # Sources linked to first task
    source = db.add_source(
        Source(
            url="https://example.com/ai-safety",
            title="AI Safety Overview",
            domain="example.com",
            snippet="An overview of AI safety research.",
            quality_score=0.8,
            is_academic=False,
        ),
        task_id=tasks[0].id,
        position=1,
    )

    source2 = db.add_source(
        Source(
            url="https://arxiv.org/paper123",
            title="Alignment Research Paper",
            domain="arxiv.org",
            snippet="A paper on alignment.",
            quality_score=0.9,
            is_academic=True,
        ),
        task_id=tasks[0].id,
        position=2,
    )

    # Glossary term
    glossary = db.add_glossary_term(
        GlossaryTerm(term="Alignment", definition="Ensuring AI systems act in accordance with human values"),
        session_id=session.id,
    )

    return SimpleNamespace(
        session=session,
        sections=sections,
        tasks=tasks,
        sources=[source, source2],
        glossary=glossary,
    )


@pytest.fixture
def test_config(isolated_environment):
    """Explicit access to the test Config object."""
    return isolated_environment
