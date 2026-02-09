"""
Tests for src.web — verifies that key web routes return 200 responses.

Uses FastAPI's TestClient with a pre-populated database so pages have
data to render without requiring a running research session.
"""
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.web.app import create_app
from src.database import get_database
from src.config import (
    ResearchTask, ReportSection, Source, GlossaryTerm, TaskStatus,
)


@pytest.fixture
def client(populated_db):
    """FastAPI test client with a populated database.

    Patches is_research_running and get_current_phase so routes that
    check background thread state don't fail.
    """
    with patch("src.web.app.is_research_running", return_value=False), \
         patch("src.web.app.get_current_phase", return_value="idle"), \
         patch("src.web.routes._is_running", return_value=False):
        app = create_app()
        yield TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client_empty(db):
    """FastAPI test client with an empty database (no sessions)."""
    with patch("src.web.app.is_research_running", return_value=False), \
         patch("src.web.app.get_current_phase", return_value="idle"), \
         patch("src.web.routes._is_running", return_value=False):
        app = create_app()
        yield TestClient(app, raise_server_exceptions=False)


# =========================================================================
# HTML page routes
# =========================================================================

class TestHTMLPages:
    def test_dashboard_home(self, client, populated_db):
        resp = client.get(f"/?session={populated_db.session.id}")
        assert resp.status_code == 200

    def test_dashboard_page(self, client, populated_db):
        resp = client.get(f"/dashboard?session={populated_db.session.id}")
        assert resp.status_code == 200

    def test_sessions_page(self, client):
        resp = client.get("/sessions")
        assert resp.status_code == 200

    def test_tasks_page(self, client, populated_db):
        resp = client.get(f"/tasks?session={populated_db.session.id}")
        assert resp.status_code == 200

    def test_sources_page(self, client, populated_db):
        resp = client.get(f"/sources?session={populated_db.session.id}")
        assert resp.status_code == 200

    def test_research_page(self, client, populated_db):
        resp = client.get(f"/research?session={populated_db.session.id}")
        assert resp.status_code == 200

    def test_report_page(self, client, populated_db):
        resp = client.get(f"/report?session={populated_db.session.id}")
        assert resp.status_code == 200

    def test_dashboard_empty_db(self, client_empty):
        """Dashboard should handle an empty database gracefully."""
        resp = client_empty.get("/")
        assert resp.status_code == 200


# =========================================================================
# JSON API routes
# =========================================================================

class TestAPIRoutes:
    def test_api_status(self, client, populated_db):
        resp = client.get(f"/api/status?session={populated_db.session.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert "running" in data
        assert "statistics" in data

    def test_api_tasks(self, client, populated_db):
        resp = client.get(f"/api/tasks?session={populated_db.session.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 6

    def test_api_tasks_by_status(self, client, populated_db):
        resp = client.get(f"/api/tasks?status=pending&session={populated_db.session.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert all(t["status"] == "pending" for t in data)

    def test_api_task_detail(self, client, populated_db):
        task_id = populated_db.tasks[0].id
        resp = client.get(f"/api/tasks/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == task_id

    def test_api_sources(self, client, populated_db):
        resp = client.get(f"/api/sources?session={populated_db.session.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_api_glossary(self, client, populated_db):
        resp = client.get("/api/glossary")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_api_presets(self, client):
        resp = client.get("/api/presets")
        assert resp.status_code == 200
        data = resp.json()
        assert "quick" in data
        assert "standard" in data

    def test_api_costs(self, client):
        resp = client.get("/api/costs")
        assert resp.status_code == 200


# =========================================================================
# HTMX fragment routes
# =========================================================================

class TestFragmentRoutes:
    def test_fragment_progress(self, client, populated_db):
        resp = client.get(f"/fragments/progress?session={populated_db.session.id}")
        assert resp.status_code == 200

    def test_fragment_task_list(self, client, populated_db):
        resp = client.get(f"/fragments/task-list?session={populated_db.session.id}")
        assert resp.status_code == 200

    def test_fragment_stats(self, client, populated_db):
        resp = client.get(f"/fragments/stats?session={populated_db.session.id}")
        assert resp.status_code == 200

    def test_fragment_activity(self, client, populated_db):
        resp = client.get(f"/fragments/activity?session={populated_db.session.id}")
        assert resp.status_code == 200

    def test_fragment_session_info(self, client, populated_db):
        resp = client.get(f"/fragments/session-info?session={populated_db.session.id}")
        assert resp.status_code == 200

    def test_fragment_research_tasks(self, client, populated_db):
        resp = client.get(f"/fragments/research-tasks?session={populated_db.session.id}")
        assert resp.status_code == 200

    def test_fragment_research_sources(self, client, populated_db):
        resp = client.get(f"/fragments/research-sources?session={populated_db.session.id}")
        assert resp.status_code == 200

    def test_fragment_report_page(self, client, populated_db):
        resp = client.get(f"/fragments/report-page?session={populated_db.session.id}")
        assert resp.status_code == 200
