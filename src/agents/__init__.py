"""
Agents Module for Deep Research Agent
Contains the Planner, Researcher, and Editor agents
"""

from .planner import PlannerAgent
from .researcher import ResearcherAgent
from .outline_designer import OutlineDesignerAgent
from .section_task_planner import SectionTaskPlannerAgent
from .gap_analysis import GapAnalysisAgent
from .synthesis import SynthesisAgent
from .editor import EditorAgent
from .query_refinement import QueryRefinementAgent

# Re-export all prompt constants for backward compatibility
from .prompts import (
    PLANNER_SYSTEM_PROMPT,
    PRE_PLAN_ANALYSIS_SYSTEM_PROMPT,
    RESEARCH_NOTES_SYSTEM_PROMPT,
    SOURCE_EXTRACTION_SYSTEM_PROMPT,
    QUERY_GENERATOR_SYSTEM,
    QUERY_GENERATOR_PROMPT,
    QUERY_GENERATOR_JSON_PROMPT,
    QUERY_GENERATOR_TOOL_NAME,
    QUERY_GENERATOR_TOOL_DESC,
    GAP_ANALYSIS_SYSTEM_PROMPT,
    OUTLINE_DESIGNER_SYSTEM_PROMPT,
    SECTION_TASK_PLANNER_SYSTEM_PROMPT,
    PIPELINE_GAP_ANALYSIS_SYSTEM_PROMPT,
    SECTION_SYNTHESIS_SYSTEM_PROMPT,
    _SYNTHESIS_STYLE_GUIDANCE,
    EXEC_SUMMARY_SYSTEM_PROMPT,
    CONCLUSION_SYSTEM_PROMPT,
    QUERY_REFINEMENT_QUESTIONS_SYSTEM_PROMPT,
    QUERY_REFINEMENT_BRIEF_SYSTEM_PROMPT,
)
