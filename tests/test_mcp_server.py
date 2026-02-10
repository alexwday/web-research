"""Tests for MCP server tools."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

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

    def test_timing_includes_elapsed_seconds(self, db):
        session = db.create_session("Test query")
        # Session was just created, so elapsed should be a small non-negative int
        result = research_status(run_id=session.id)
        elapsed = result["timing"]["elapsed_seconds"]
        assert isinstance(elapsed, int)
        assert elapsed >= 0

    def test_cancel_requested_at_in_status(self, db):
        session = db.create_session("Test query")
        # Before cancellation
        result = research_status(run_id=session.id)
        assert result["cancel_requested_at"] is None

        # Set cancel_requested_at on session
        now = datetime.now(timezone.utc)
        db.update_session(session.id, cancel_requested_at=now)
        result = research_status(run_id=session.id)
        assert result["cancel_requested_at"] is not None
        assert now.isoformat()[:19] in result["cancel_requested_at"]


class TestResearchStart:
    def test_empty_query_returns_error(self):
        result = research_start(query="")
        assert result["status"] == "error"

    def test_whitespace_query_returns_error(self):
        result = research_start(query="   ")
        assert result["status"] == "error"

    def test_already_running(self):
        with patch.object(get_service(), "start_run", return_value={"status": "already_running"}):
            result = research_start(query="Another query")
        assert result["status"] == "already_running"

    def test_preset_passed_to_service(self):
        with patch.object(get_service(), "start_run", return_value={"status": "started"}) as mock:
            research_start(query="Test query", preset="quick")
            mock.assert_called_once_with("Test query", mode="quick", blocking=False)

    def test_start_returns_started_with_run_id(self, db):
        with patch.object(get_service(), "_run_background", return_value={"status": "started"}):
            result = research_start(query="Test AI safety research")
        assert result["status"] == "started"


class TestResearchCancel:
    def test_cancel_not_running(self):
        result = research_cancel()
        assert result["status"] == "not_running"

    def test_cancel_when_running(self):
        service = get_service()
        mock_orch = MagicMock()
        mock_orch.session_id = None  # no DB session to update
        service._orchestrator = mock_orch
        try:
            result = research_cancel()
            assert result["status"] == "cancelling"
        finally:
            service._orchestrator = None


class TestCancellationBehavior:
    """Tests for cancellation hardening."""

    def test_cancel_sets_flag_and_records_timestamp(self, db):
        """Verify cancel sets _cancel_requested flag and records timestamp in DB."""
        from src.orchestrator import ResearchOrchestrator

        session = db.create_session("Cancel test query")
        orchestrator = ResearchOrchestrator(register_signals=False)
        orchestrator.session_id = session.id
        orchestrator.is_running = True

        service = get_service()
        service._orchestrator = orchestrator
        try:
            result = service.cancel_run()
            assert result["status"] == "cancelling"
            assert orchestrator._cancel_requested is True
            assert orchestrator.is_running is False

            # Check DB timestamp was recorded
            updated = db.get_session_by_id(session.id)
            assert updated.cancel_requested_at is not None
        finally:
            service._orchestrator = None

    def test_double_cancel_is_safe(self, db):
        """Two cancel calls don't error."""
        from src.orchestrator import ResearchOrchestrator

        session = db.create_session("Double cancel test")
        orchestrator = ResearchOrchestrator(register_signals=False)
        orchestrator.session_id = session.id
        orchestrator.is_running = True

        service = get_service()
        service._orchestrator = orchestrator
        try:
            result1 = service.cancel_run()
            assert result1["status"] == "cancelling"
            # Second call — orchestrator still exists but is_running is False
            result2 = service.cancel_run()
            assert result2["status"] == "cancelling"
        finally:
            service._orchestrator = None

    def test_cancel_when_not_running(self):
        """Cancel with no orchestrator returns not_running."""
        service = get_service()
        service._orchestrator = None
        result = service.cancel_run()
        assert result["status"] == "not_running"

    def test_cancel_produces_cancelled_in_finalize(self, db):
        """_cancel_requested=True -> 'cancelled' status in _finalize."""
        from src.orchestrator import ResearchOrchestrator
        from src.config import ResearchTask

        session = db.create_session("Finalize cancel test")
        # Add a pending task so status would normally be "partial"
        db.add_task(
            ResearchTask(topic="T1", description="D", file_path="/tmp/t1.md"),
            session_id=session.id,
        )

        orchestrator = ResearchOrchestrator(register_signals=False)
        orchestrator.session_id = session.id
        orchestrator.start_time = datetime.now()
        orchestrator._cancel_requested = True

        result = orchestrator._finalize({"markdown": "/tmp/report.md"})
        updated = db.get_session_by_id(session.id)
        assert updated.status == "cancelled"

    def test_non_cancelled_partial_still_works(self, db):
        """Pending tasks without cancel -> 'partial' status."""
        from src.orchestrator import ResearchOrchestrator
        from src.config import ResearchTask

        session = db.create_session("Partial test")
        db.add_task(
            ResearchTask(topic="T1", description="D", file_path="/tmp/t1.md"),
            session_id=session.id,
        )

        orchestrator = ResearchOrchestrator(register_signals=False)
        orchestrator.session_id = session.id
        orchestrator.start_time = datetime.now()
        orchestrator._cancel_requested = False

        result = orchestrator._finalize({"markdown": "/tmp/report.md"})
        updated = db.get_session_by_id(session.id)
        assert updated.status == "partial"

    def test_mcp_cancel_returns_run_id(self, db):
        """research_cancel() includes run_id in response."""
        from src.orchestrator import ResearchOrchestrator

        session = db.create_session("MCP cancel run_id test")
        orchestrator = ResearchOrchestrator(register_signals=False)
        orchestrator.session_id = session.id
        orchestrator.is_running = True

        service = get_service()
        service._orchestrator = orchestrator
        try:
            result = research_cancel()
            assert result["status"] == "cancelling"
            assert result["run_id"] == session.id
        finally:
            service._orchestrator = None
