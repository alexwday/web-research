"""
Tests for src.orchestrator.ResearchOrchestrator — end-to-end pipeline
with mocked LLM agents and search calls.

Verifies: session creation, 7-phase progression, task creation,
section synthesis, and report compilation.
"""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.config import (
    ResearchTask, ReportSection, Source, GlossaryTerm,
    TaskStatus, SectionStatus, get_config,
)
from src.database import get_database


# ---------------------------------------------------------------------------
# Helpers for building mock agent responses
# ---------------------------------------------------------------------------

def _mock_design_outline(query, pre_plan_ctx, session_id):
    """Mock OutlineDesignerAgent.design_outline — creates sections in DB."""
    db = get_database()
    sections = db.add_sections_bulk([
        ReportSection(title="Background", description="Historical context", position=1),
        ReportSection(title="Core Analysis", description="Main findings", position=2),
    ], session_id=session_id)
    return sections


def _mock_plan_tasks(section, all_sections, query, session_id, task_budget=None):
    """Mock SectionTaskPlannerAgent.plan_tasks_for_section — creates tasks in DB."""
    db = get_database()
    tasks = db.add_tasks_bulk([
        ResearchTask(
            topic=f"Research {section.title} - subtopic 1",
            description=f"Investigate {section.title} aspect 1",
            file_path=f"/tmp/task_{section.id}_1.md",
            section_id=section.id,
            priority=5,
        ),
        ResearchTask(
            topic=f"Research {section.title} - subtopic 2",
            description=f"Investigate {section.title} aspect 2",
            file_path=f"/tmp/task_{section.id}_2.md",
            section_id=section.id,
            priority=5,
        ),
    ], session_id=session_id)
    return tasks


def _mock_research_task(task, overall_query="", other_sections=None, session_id=None):
    """Mock ResearcherAgent.research_task — returns content + empty follow-ups."""
    content = (
        f"# Research Notes: {task.topic}\n\n"
        f"This section covers {task.description}.\n\n"
        f"Key finding: The topic is well-documented in academic literature. [1]\n\n"
        f"Additional analysis reveals important patterns. [2]\n"
    )
    # Add a source to the database for this task
    db = get_database()
    try:
        db.add_source(
            Source(
                url=f"https://example.com/source-{task.id}",
                title=f"Source for {task.topic}",
                domain="example.com",
                snippet="Relevant research content.",
                quality_score=0.75,
            ),
            task_id=task.id,
            position=1,
        )
    except Exception:
        pass  # Source URL may already exist
    return content, [], []  # (content, new_tasks, glossary_terms)


def _mock_synthesize_section(section, query, all_sections, adjacent, session_id):
    """Mock SynthesisAgent.synthesize_section — returns polished prose."""
    return (
        f"This section examines {section.title.lower()}. "
        f"Through comprehensive analysis of available literature, several key themes emerge. "
        f"The findings demonstrate significant progress in this area. [1] "
        f"Further research supports these conclusions. [2]\n"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOrchestratorPipeline:
    """Test the full 7-phase orchestrator pipeline with mocked agents."""

    @pytest.fixture
    def orchestrator_mocks(self):
        """Patch all agent methods and file I/O on the orchestrator."""
        patches = [
            # Agent methods
            patch("src.agents.PlannerAgent.run_pre_planning",
                  return_value="Pre-planning context: topic is well-studied."),
            patch("src.agents.OutlineDesignerAgent.design_outline",
                  side_effect=_mock_design_outline),
            patch("src.agents.SectionTaskPlannerAgent.plan_tasks_for_section",
                  side_effect=_mock_plan_tasks),
            patch("src.agents.ResearcherAgent.research_task",
                  side_effect=_mock_research_task),
            patch("src.agents.GapAnalysisAgent.analyze_gaps",
                  return_value={"new_tasks": 0, "new_sections": 0}),
            patch("src.agents.SynthesisAgent.synthesize_section",
                  side_effect=_mock_synthesize_section),
            patch("src.agents.EditorAgent.generate_executive_summary",
                  return_value="Executive summary: This report covers the research topic comprehensively."),
            patch("src.agents.EditorAgent.generate_conclusion",
                  return_value="Conclusion: The research reveals important findings across all areas studied."),
            # File I/O
            patch("src.orchestrator.save_markdown"),
            patch("src.orchestrator.read_file", return_value="mock file content"),
            # Suppress Rich console output during tests
            patch("src.orchestrator.print_header"),
            patch("src.orchestrator.print_info"),
            patch("src.orchestrator.print_success"),
            patch("src.orchestrator.print_warning"),
            patch("src.orchestrator.print_error"),
            patch("src.orchestrator.print_task_start"),
            patch("src.orchestrator.print_write"),
            patch("src.orchestrator.print_statistics_table"),
            patch("src.orchestrator.print_task_table"),
            patch("src.orchestrator.print_completion_summary"),
            patch("src.orchestrator.console"),
            patch("src.orchestrator.create_progress_bar"),
        ]
        mocks = [p.start() for p in patches]
        # Make the progress bar context manager work
        progress_mock = mocks[-1]
        bar = MagicMock()
        bar.__enter__ = MagicMock(return_value=bar)
        bar.__exit__ = MagicMock(return_value=False)
        bar.add_task = MagicMock(return_value=0)
        progress_mock.return_value = bar

        yield mocks
        for p in patches:
            p.stop()

    def test_full_pipeline_creates_session(self, orchestrator_mocks, test_config, tmp_path):
        """The orchestrator should create a session in the database."""
        from src.orchestrator import ResearchOrchestrator

        test_config.output.directory = str(tmp_path / "report")
        orch = ResearchOrchestrator(register_signals=False)
        result = orch.run("What is AI safety?")

        db = get_database()
        sessions = db.get_all_sessions()
        assert len(sessions) == 1
        assert "AI safety" in sessions[0].query

    def test_full_pipeline_progresses_through_phases(self, orchestrator_mocks, test_config, tmp_path):
        """The orchestrator should end in 'complete' phase."""
        from src.orchestrator import ResearchOrchestrator

        test_config.output.directory = str(tmp_path / "report")
        orch = ResearchOrchestrator(register_signals=False)
        result = orch.run("What is AI safety?")

        assert orch.phase == "complete"

    def test_full_pipeline_creates_sections(self, orchestrator_mocks, test_config, tmp_path):
        """Sections should be created during the outline design phase."""
        from src.orchestrator import ResearchOrchestrator

        test_config.output.directory = str(tmp_path / "report")
        orch = ResearchOrchestrator(register_signals=False)
        result = orch.run("What is AI safety?")

        db = get_database()
        sections = db.get_all_sections(session_id=orch.session_id)
        assert len(sections) == 2
        assert sections[0].title == "Background"
        assert sections[1].title == "Core Analysis"

    def test_full_pipeline_creates_tasks(self, orchestrator_mocks, test_config, tmp_path):
        """Tasks should be created for each section during task planning."""
        from src.orchestrator import ResearchOrchestrator

        test_config.output.directory = str(tmp_path / "report")
        orch = ResearchOrchestrator(register_signals=False)
        result = orch.run("What is AI safety?")

        db = get_database()
        tasks = db.get_all_tasks(session_id=orch.session_id)
        assert len(tasks) == 4  # 2 sections x 2 tasks each

    def test_full_pipeline_completes_tasks(self, orchestrator_mocks, test_config, tmp_path):
        """All tasks should be marked completed after research execution."""
        from src.orchestrator import ResearchOrchestrator

        test_config.output.directory = str(tmp_path / "report")
        orch = ResearchOrchestrator(register_signals=False)
        result = orch.run("What is AI safety?")

        db = get_database()
        completed = db.get_task_count(TaskStatus.COMPLETED, session_id=orch.session_id)
        total = db.get_task_count(session_id=orch.session_id)
        assert completed == total

    def test_full_pipeline_synthesizes_sections(self, orchestrator_mocks, test_config, tmp_path):
        """Sections should have synthesized content after synthesis phase."""
        from src.orchestrator import ResearchOrchestrator

        test_config.output.directory = str(tmp_path / "report")
        orch = ResearchOrchestrator(register_signals=False)
        result = orch.run("What is AI safety?")

        db = get_database()
        sections = db.get_all_sections(session_id=orch.session_id)
        for section in sections:
            assert section.synthesized_content is not None
            assert len(section.synthesized_content) > 0

    def test_full_pipeline_produces_report(self, orchestrator_mocks, test_config, tmp_path):
        """The pipeline should produce output files."""
        from src.orchestrator import ResearchOrchestrator

        test_config.output.directory = str(tmp_path / "report")
        orch = ResearchOrchestrator(register_signals=False)
        result = orch.run("What is AI safety?")

        assert "output_files" in result
        output_files = result["output_files"]
        # Should have markdown and/or html
        assert len(output_files) > 0

    def test_full_pipeline_stores_session_artifacts(self, orchestrator_mocks, test_config, tmp_path):
        """Session should have executive summary, conclusion, and report paths stored."""
        from src.orchestrator import ResearchOrchestrator

        test_config.output.directory = str(tmp_path / "report")
        orch = ResearchOrchestrator(register_signals=False)
        result = orch.run("What is AI safety?")

        db = get_database()
        session = db.get_session_by_id(orch.session_id)
        assert session.executive_summary is not None
        assert session.conclusion is not None
        assert session.status in ("completed", "completed_with_errors")

    def test_full_pipeline_returns_statistics(self, orchestrator_mocks, test_config, tmp_path):
        """The result dict should include statistics."""
        from src.orchestrator import ResearchOrchestrator

        test_config.output.directory = str(tmp_path / "report")
        orch = ResearchOrchestrator(register_signals=False)
        result = orch.run("What is AI safety?")

        assert "statistics" in result
        stats = result["statistics"]
        assert stats["total_tasks"] == 4
        assert stats["completed_tasks"] == 4


class TestOrchestratorSessionManagement:
    """Test session initialization and resume logic."""

    def test_initialize_session(self, test_config):
        from src.orchestrator import ResearchOrchestrator

        orch = ResearchOrchestrator(register_signals=False)
        session = orch._initialize_session("Test query")
        assert session.id is not None
        assert session.query == "Test query"

    def test_initialize_session_with_refinement(self, test_config):
        from src.orchestrator import ResearchOrchestrator

        orch = ResearchOrchestrator(register_signals=False)
        session = orch._initialize_session(
            "Raw query",
            refined_brief="Enhanced query with context",
            refinement_qa='[{"q": "What scope?", "a": "Broad"}]',
        )
        db = get_database()
        fetched = db.get_session_by_id(session.id)
        assert fetched.refined_brief == "Enhanced query with context"
