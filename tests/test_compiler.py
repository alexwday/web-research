"""
Tests for src.compiler.ReportCompiler — citation remapping, markdown/HTML
output generation, and source deduplication.
"""
import os
from pathlib import Path

import pytest

from src.pipeline.compiler import ReportCompiler
from src.config.types import (
    ResearchTask, ReportSection, Source, GlossaryTerm, TaskStatus,
)
from src.infra._database import get_database


# =========================================================================
# Citation remapping
# =========================================================================

class TestCitationRemapping:
    def test_remap_simple(self):
        content = "First finding [1] and second finding [2]."
        mapping = {1: 5, 2: 12}
        result = ReportCompiler._remap_citations(content, mapping)
        assert "[5]" in result
        assert "[12]" in result
        assert "[1]" not in result

    def test_remap_empty_mapping(self):
        content = "No citations to remap [1]."
        result = ReportCompiler._remap_citations(content, {})
        assert result == content

    def test_remap_preserves_markdown_links(self):
        content = "See [link text](https://example.com) and also [1]."
        mapping = {1: 99}
        result = ReportCompiler._remap_citations(content, mapping)
        assert "[link text](https://example.com)" in result
        assert "[99]" in result

    def test_remap_unmapped_numbers_preserved(self):
        content = "Mapped [1] and unmapped [3]."
        mapping = {1: 10}
        result = ReportCompiler._remap_citations(content, mapping)
        assert "[10]" in result
        assert "[3]" in result  # unmapped stays as-is


# =========================================================================
# Slugify
# =========================================================================

class TestSlugify:
    def test_basic_slugify(self):
        compiler = ReportCompiler()
        assert compiler._slugify("Hello World") == "hello-world"

    def test_special_characters(self):
        compiler = ReportCompiler()
        slug = compiler._slugify("AI Safety: A Primer (2024)")
        assert ":" not in slug
        assert "(" not in slug
        assert slug == "ai-safety-a-primer-2024"

    def test_deduplication(self):
        compiler = ReportCompiler()
        slug1 = compiler._slugify("Introduction")
        slug2 = compiler._slugify("Introduction")
        assert slug1 == "introduction"
        assert slug2 == "introduction-2"


# =========================================================================
# Heading normalization
# =========================================================================

class TestNormalizeHeadings:
    def test_adds_space_after_hash(self):
        compiler = ReportCompiler()
        result = compiler._normalize_headings("##No space")
        assert result == "## No space"

    def test_leaves_correct_headings(self):
        compiler = ReportCompiler()
        result = compiler._normalize_headings("## Already spaced")
        assert result == "## Already spaced"


# =========================================================================
# Report compilation (integration)
# =========================================================================

class TestCompileReport:
    @pytest.fixture
    def compiler_with_data(self, populated_db, test_config, tmp_path):
        """Set up a compiler with sections that have synthesized content."""
        db = get_database()
        session = populated_db.session

        # Complete all tasks
        for task in populated_db.tasks:
            db.mark_task_complete(task.id, word_count=100, citation_count=2)

        # Synthesize all sections
        for section in populated_db.sections:
            db.mark_section_synthesized(
                section.id,
                f"Synthesized content for {section.title}. Key finding here. [1] Another point. [2]",
                word_count=50,
                citation_count=2,
            )

        test_config.output.directory = str(tmp_path / "report")
        test_config.output.formats = ["markdown", "html"]

        compiler = ReportCompiler()
        return compiler, session, populated_db.sections

    def test_compile_produces_markdown(self, compiler_with_data, tmp_path):
        compiler, session, sections = compiler_with_data
        db = get_database()

        # Build chapters from sections
        all_sections = db.get_all_sections(session_id=session.id)
        chapters = []
        for section in all_sections:
            if section.synthesized_content:
                chapters.append({"section": section, "content": section.synthesized_content})

        output = compiler.compile_report(
            query="Test query about AI safety",
            executive_summary="This report examines AI safety.",
            conclusion="AI safety is important.",
            duration_seconds=60,
            session_id=session.id,
            pre_read_chapters=chapters,
        )

        assert "markdown" in output
        md_path = Path(output["markdown"])
        assert md_path.exists()
        content = md_path.read_text()
        assert "Deep Research Report" in content
        assert "AI safety" in content

    def test_compile_produces_html(self, compiler_with_data, tmp_path):
        compiler, session, sections = compiler_with_data
        db = get_database()

        all_sections = db.get_all_sections(session_id=session.id)
        chapters = []
        for section in all_sections:
            if section.synthesized_content:
                chapters.append({"section": section, "content": section.synthesized_content})

        output = compiler.compile_report(
            query="Test query about AI safety",
            executive_summary="This report examines AI safety.",
            conclusion="AI safety is important.",
            duration_seconds=60,
            session_id=session.id,
            pre_read_chapters=chapters,
        )

        assert "html" in output
        html_path = Path(output["html"])
        assert html_path.exists()
        content = html_path.read_text()
        assert "<html" in content
        assert "Deep Research Report" in content

    def test_compile_without_executive_summary(self, compiler_with_data):
        compiler, session, sections = compiler_with_data
        db = get_database()

        all_sections = db.get_all_sections(session_id=session.id)
        chapters = [
            {"section": s, "content": s.synthesized_content}
            for s in all_sections if s.synthesized_content
        ]

        output = compiler.compile_report(
            query="Test query",
            executive_summary=None,
            conclusion=None,
            duration_seconds=30,
            session_id=session.id,
            pre_read_chapters=chapters,
        )
        assert len(output) > 0

    def test_compile_empty_chapters(self, test_config, tmp_path, db):
        """Compilation with no chapters should still produce output."""
        test_config.output.directory = str(tmp_path / "report")
        session = db.create_session("Empty query")
        compiler = ReportCompiler()

        output = compiler.compile_report(
            query="Empty query",
            session_id=session.id,
            pre_read_chapters=[],
        )
        # Should still produce files (possibly empty reports)
        assert isinstance(output, dict)


class TestSourceDeduplication:
    def test_global_sources_deduplicates(self, populated_db):
        """Sources shared across sections should appear only once globally."""
        db = get_database()
        session = populated_db.session

        # Add the same source URL to a second task
        second_task = populated_db.tasks[2]  # task in second section
        try:
            # Try adding same URL — should reuse existing source
            existing = db.get_source_by_url("https://example.com/ai-safety")
            if existing:
                # Link existing source to second task (the add_source handles dedup)
                db.add_source(
                    Source(
                        url="https://example.com/ai-safety",
                        title="AI Safety Overview",
                        domain="example.com",
                    ),
                    task_id=second_task.id,
                    position=1,
                )
        except Exception:
            pass  # IntegrityError means it's already linked

        compiler = ReportCompiler()
        # Build chapters
        sections = db.get_all_sections(session_id=session.id)
        chapters = []
        for section in sections:
            chapters.append({
                "section": section,
                "content": f"Content for {section.title} [1]",
            })

        global_sources, updated = compiler._build_global_sources(chapters)
        # Each unique URL should appear only once
        urls = [s.url for s in global_sources]
        assert len(urls) == len(set(urls))
