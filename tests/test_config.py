"""
Tests for src.config — configuration loading, presets, and overrides.
"""
import pytest

from src.config import (
    Config, apply_overrides, RESEARCH_PRESETS,
    TaskStatus, SectionStatus,
)


class TestConfigDefaults:
    def test_default_config_creation(self):
        config = Config()
        assert config.database.path == "research_state.db"
        assert config.research.max_total_tasks == 200
        assert config.output.directory == "report"
        assert "markdown" in config.output.formats

    def test_default_llm_models(self):
        config = Config()
        assert config.llm.models.planner == "gpt-4o"
        assert config.llm.models.researcher == "gpt-4o-mini"

    def test_default_search_config(self):
        config = Config()
        assert config.search.results_per_query == 3
        assert config.search.depth == "advanced"

    def test_default_gap_analysis(self):
        config = Config()
        assert config.gap_analysis.enabled is True
        assert config.gap_analysis.max_new_sections == 3


class TestEnums:
    def test_task_status_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.SKIPPED.value == "skipped"

    def test_section_status_values(self):
        assert SectionStatus.PLANNED.value == "planned"
        assert SectionStatus.COMPLETE.value == "complete"


class TestApplyOverrides:
    def test_simple_override(self):
        base = Config()
        result = apply_overrides(base, {"research.max_total_tasks": 50})
        assert result.research.max_total_tasks == 50
        # Original unchanged
        assert base.research.max_total_tasks == 200

    def test_nested_override(self):
        base = Config()
        result = apply_overrides(base, {"llm.models.planner": "gpt-5"})
        assert result.llm.models.planner == "gpt-5"

    def test_bool_coercion(self):
        base = Config()
        result = apply_overrides(base, {"gap_analysis.enabled": "false"})
        assert result.gap_analysis.enabled is False

    def test_int_coercion(self):
        base = Config()
        result = apply_overrides(base, {"research.max_total_tasks": "42"})
        assert result.research.max_total_tasks == 42

    def test_float_coercion(self):
        base = Config()
        result = apply_overrides(base, {"quality.min_source_quality": "0.7"})
        assert result.quality.min_source_quality == 0.7

    def test_multiple_overrides(self):
        base = Config()
        result = apply_overrides(base, {
            "research.max_total_tasks": 10,
            "search.results_per_query": 5,
            "gap_analysis.enabled": False,
        })
        assert result.research.max_total_tasks == 10
        assert result.search.results_per_query == 5
        assert result.gap_analysis.enabled is False


class TestPresets:
    def test_all_presets_exist(self):
        assert set(RESEARCH_PRESETS.keys()) == {"quick", "standard", "deep", "exhaustive"}

    def test_preset_has_required_keys(self):
        for name, preset in RESEARCH_PRESETS.items():
            assert "label" in preset, f"{name} missing label"
            assert "description" in preset, f"{name} missing description"
            assert "overrides" in preset, f"{name} missing overrides"

    def test_quick_preset_applies(self):
        base = Config()
        result = apply_overrides(base, RESEARCH_PRESETS["quick"]["overrides"])
        assert result.research.max_total_tasks == 10
        assert result.gap_analysis.enabled is False

    def test_exhaustive_preset_applies(self):
        base = Config()
        result = apply_overrides(base, RESEARCH_PRESETS["exhaustive"]["overrides"])
        assert result.research.max_total_tasks == 100
        assert result.gap_analysis.enabled is True

    def test_preset_overrides_are_valid_config_paths(self):
        """Every dotted key in preset overrides must resolve to a real Config field."""
        base = Config()
        for name, preset in RESEARCH_PRESETS.items():
            # apply_overrides should not raise
            result = apply_overrides(base, preset["overrides"])
            assert isinstance(result, Config), f"Preset '{name}' produced invalid config"
