"""Tests for MCP server tools."""
import base64
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from src.mcp_server import (
    research_presets, research_status, research_start, research_cancel,
    research_events, research_result,
    resource_runs, resource_run_status, resource_run_events,
    resource_run_tasks, resource_run_sources, resource_run_sections,
    resource_run_artifacts, resource_run_costs,
)
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


class TestResearchEvents:
    def test_no_session_returns_no_session(self):
        result = research_events()
        assert result["status"] == "no_session"
        assert result["events"] == []

    def test_empty_events(self, db):
        session = db.create_session("Events test")
        result = research_events(run_id=session.id)
        assert result["session_id"] == session.id
        assert result["events"] == []
        assert result["next_cursor"] is None

    def test_events_returned_in_order(self, db):
        session = db.create_session("Order test")
        now = datetime.now(timezone.utc)
        for i in range(3):
            db.add_run_event(
                session_id=session.id,
                event_type="query",
                query_text=f"q{i}",
                created_at=now + timedelta(seconds=i),
            )
        result = research_events(run_id=session.id)
        texts = [e["payload"]["query_text"] for e in result["events"]]
        assert texts == ["q0", "q1", "q2"]

    def test_event_structure(self, db):
        session = db.create_session("Structure test")
        db.add_run_event(
            session_id=session.id,
            event_type="result",
            task_id=None,
            query_group="group1",
            query_text="test query",
            url="https://example.com",
            title="Example",
            snippet="A snippet",
            quality_score=0.7,
            phase="researching",
            severity="info",
            payload_json='{"extra": true}',
        )
        result = research_events(run_id=session.id)
        assert len(result["events"]) == 1
        ev = result["events"][0]
        assert "event_id" in ev
        assert "ts" in ev
        assert ev["type"] == "result"
        assert ev["query_group"] == "group1"
        payload = ev["payload"]
        assert payload["query_text"] == "test query"
        assert payload["url"] == "https://example.com"
        assert payload["title"] == "Example"
        assert payload["snippet"] == "A snippet"
        assert payload["quality_score"] == 0.7
        assert payload["phase"] == "researching"
        assert payload["severity"] == "info"
        assert payload["data"] == {"extra": True}

    def test_cursor_pagination(self, db):
        session = db.create_session("Pagination test")
        now = datetime.now(timezone.utc)
        for i in range(5):
            db.add_run_event(
                session_id=session.id,
                event_type="query",
                query_text=f"q{i}",
                created_at=now + timedelta(seconds=i),
            )
        # Page 1
        page1 = research_events(run_id=session.id, limit=2)
        assert len(page1["events"]) == 2
        assert page1["next_cursor"] is not None
        texts1 = [e["payload"]["query_text"] for e in page1["events"]]
        assert texts1 == ["q0", "q1"]

        # Page 2
        page2 = research_events(run_id=session.id, cursor=page1["next_cursor"], limit=2)
        assert len(page2["events"]) == 2
        assert page2["next_cursor"] is not None
        texts2 = [e["payload"]["query_text"] for e in page2["events"]]
        assert texts2 == ["q2", "q3"]

        # Page 3 (last)
        page3 = research_events(run_id=session.id, cursor=page2["next_cursor"], limit=2)
        assert len(page3["events"]) == 1
        assert page3["next_cursor"] is None
        texts3 = [e["payload"]["query_text"] for e in page3["events"]]
        assert texts3 == ["q4"]

    def test_invalid_cursor_treated_as_start(self, db):
        session = db.create_session("Bad cursor test")
        db.add_run_event(
            session_id=session.id,
            event_type="query",
            query_text="q0",
        )
        result = research_events(run_id=session.id, cursor="not-valid-base64!!!")
        assert len(result["events"]) == 1

    def test_limit_clamping(self, db):
        session = db.create_session("Limit clamp test")
        db.add_run_event(
            session_id=session.id,
            event_type="query",
            query_text="q0",
        )
        # limit=0 should be clamped to 1
        result = research_events(run_id=session.id, limit=0)
        assert len(result["events"]) == 1


class TestResearchResult:
    def test_no_session_returns_no_session(self):
        result = research_result()
        assert result["status"] == "no_session"

    def test_basic_result_structure(self, db):
        session = db.create_session("Result test")
        result = research_result(run_id=session.id)
        assert result["run_id"] == session.id
        assert result["status"] == "running"
        assert "artifacts" in result
        assert "summary" in result
        assert "sources" in result

    def test_result_with_sections(self, populated_db):
        result = research_result(run_id=populated_db.session.id)
        sections = result["summary"]["sections"]
        assert len(sections) == 3
        for s in sections:
            assert "title" in s
            assert "position" in s
            assert "word_count" in s
            assert "citation_count" in s

    def test_result_with_sources(self, populated_db):
        result = research_result(run_id=populated_db.session.id)
        sources = result["sources"]
        assert len(sources) == 2
        urls = {s["url"] for s in sources}
        assert "https://example.com/ai-safety" in urls
        assert "https://arxiv.org/paper123" in urls
        for s in sources:
            assert "domain" in s
            assert "quality_score" in s
            assert "is_academic" in s
            assert "task_ids" in s

    def test_result_artifacts(self, db):
        session = db.create_session("Artifacts test")
        db.update_session(
            session.id,
            report_markdown_path="/tmp/report.md",
            report_html_path="/tmp/report.html",
        )
        result = research_result(run_id=session.id)
        assert result["artifacts"]["markdown_path"] == "/tmp/report.md"
        assert result["artifacts"]["html_path"] == "/tmp/report.html"

    def test_result_summary_fields(self, db):
        session = db.create_session("Summary test")
        db.update_session(
            session.id,
            executive_summary="This is the summary.",
            conclusion="This is the conclusion.",
        )
        result = research_result(run_id=session.id)
        assert result["summary"]["executive_summary"] == "This is the summary."
        assert result["summary"]["conclusion"] == "This is the conclusion."


class TestMCPResources:
    def test_resource_runs(self, db):
        db.create_session("Resource runs test")
        data = json.loads(resource_runs())
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "run_id" in data[0]
        assert "query" in data[0]
        assert "status" in data[0]

    def test_resource_run_status(self, db):
        session = db.create_session("Status resource test")
        data = json.loads(resource_run_status(run_id=session.id))
        assert data["run_id"] == session.id
        assert "status" in data

    def test_resource_run_events(self, db):
        session = db.create_session("Events resource test")
        data = json.loads(resource_run_events(run_id=session.id))
        assert "events" in data
        assert data["session_id"] == session.id

    def test_resource_run_tasks(self, populated_db):
        data = json.loads(resource_run_tasks(run_id=populated_db.session.id))
        assert isinstance(data, list)
        assert len(data) == 6  # 2 tasks per section, 3 sections
        assert "task_id" in data[0]
        assert "topic" in data[0]

    def test_resource_run_sources(self, populated_db):
        data = json.loads(resource_run_sources(run_id=populated_db.session.id))
        assert isinstance(data, list)
        assert len(data) == 2
        assert "url" in data[0]
        assert "snippet" in data[0]

    def test_resource_run_sections(self, populated_db):
        data = json.loads(resource_run_sections(run_id=populated_db.session.id))
        assert isinstance(data, list)
        assert len(data) == 3
        assert "section_id" in data[0]
        assert "title" in data[0]

    def test_resource_run_artifacts(self, db):
        session = db.create_session("Artifacts resource test")
        data = json.loads(resource_run_artifacts(run_id=session.id))
        assert data["run_id"] == session.id
        assert "artifacts" in data

    def test_resource_run_costs(self, db):
        session = db.create_session("Costs resource test")
        data = json.loads(resource_run_costs(run_id=session.id))
        assert data["run_id"] == session.id
        assert "costs" in data
