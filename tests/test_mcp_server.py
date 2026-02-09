"""Tests for MCP server tools."""
from src.mcp_server import research_presets, research_status, research_start, research_cancel
from src.service import get_service


class TestResearchPresets:
    def test_returns_presets_dict(self):
        result = research_presets()
        assert "presets" in result
        presets = result["presets"]
        assert "quick" in presets
        assert "standard" in presets
        assert "deep" in presets
        assert "exhaustive" in presets

    def test_preset_structure(self):
        result = research_presets()
        for name, preset in result["presets"].items():
            assert "label" in preset
            assert "description" in preset
            assert "overrides" in preset


class TestResearchStatus:
    def test_no_session_returns_no_session(self):
        result = research_status()
        assert result["status"] == "no_session"

    def test_with_session(self, db):
        session = db.create_session("Test query")
        result = research_status(run_id=session.id)
        assert result["run_id"] == session.id
        assert result["status"] == "running"
        assert "progress" in result
        assert "timing" in result
        assert "counts" in result
        assert "costs" in result

    def test_progress_calculation(self, db):
        from src.config import ResearchTask

        session = db.create_session("Test query")
        t1 = db.add_task(
            ResearchTask(topic="T1", description="D", file_path="/tmp/t1.md"),
            session_id=session.id,
        )
        t2 = db.add_task(
            ResearchTask(topic="T2", description="D", file_path="/tmp/t2.md"),
            session_id=session.id,
        )
        db.mark_task_complete(t1.id, word_count=100, citation_count=2)
        result = research_status(run_id=session.id)
        assert result["progress"]["completed_tasks"] == 1
        assert result["progress"]["total_tasks"] == 2
        assert result["progress"]["pct"] == 50


class TestResearchStart:
    def test_empty_query_returns_error(self):
        result = research_start(query="")
        assert result["status"] == "error"

    def test_whitespace_query_returns_error(self):
        result = research_start(query="   ")
        assert result["status"] == "error"

    def test_already_running(self):
        from unittest.mock import patch

        with patch.object(get_service(), "start_run", return_value={"status": "already_running"}):
            result = research_start(query="Another query")
        assert result["status"] == "already_running"

    def test_preset_passed_to_service(self):
        from unittest.mock import patch

        with patch.object(get_service(), "start_run", return_value={"status": "started"}) as mock:
            research_start(query="Test query", preset="quick")
            mock.assert_called_once_with("Test query", mode="quick", blocking=False)

    def test_start_returns_started_with_run_id(self, db):
        from unittest.mock import patch

        with patch.object(get_service(), "_run_background", return_value={"status": "started"}):
            result = research_start(query="Test AI safety research")
        assert result["status"] == "started"


class TestResearchCancel:
    def test_cancel_not_running(self):
        result = research_cancel()
        assert result["status"] == "not_running"

    def test_cancel_when_running(self):
        from unittest.mock import MagicMock

        service = get_service()
        service._orchestrator = MagicMock()
        try:
            result = research_cancel()
            assert result["status"] == "cancelling"
        finally:
            service._orchestrator = None
