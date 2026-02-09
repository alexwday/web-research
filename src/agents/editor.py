"""EditorAgent — executive summary and conclusion generation."""
from typing import List, Dict

from ..config import get_config
from ..llm_client import get_llm_client
from ..database import get_database
from ..logger import get_logger

from .prompts import EXEC_SUMMARY_SYSTEM_PROMPT, CONCLUSION_SYSTEM_PROMPT

logger = get_logger(__name__)


class EditorAgent:
    """Agent responsible for executive summary and conclusion generation"""

    def __init__(self):
        self.client = get_llm_client()
        self.db = get_database()

    @property
    def config(self):
        return get_config()

    def generate_executive_summary(
        self,
        query: str,
        section_summaries: List[Dict[str, str]],
        report_structure: str = ""
    ) -> str:
        """Generate an executive summary for the report."""
        logger.info("Generating executive summary...")

        sections_text = "\n\n".join(
            f"### {s['topic']}\n{s['summary']}" for s in section_summaries
        )

        structure_block = ""
        if report_structure:
            structure_block = f"\n\nReport Structure:\n{report_structure}\n"

        prompt = f"""Write an executive summary for this research report.

Research Query: {query}
{structure_block}
Section Findings:
{sections_text}

Synthesize the actual findings above into a 300-500 word executive summary."""

        response = self.client.complete(
            prompt=prompt,
            system=EXEC_SUMMARY_SYSTEM_PROMPT,
            max_tokens=self.config.llm.max_tokens.editor,
            temperature=self.config.llm.temperature.editor,
            model=self.config.llm.models.editor
        )

        return response

    def generate_conclusion(
        self,
        query: str,
        section_summaries: List[Dict[str, str]],
        word_count: int,
        report_structure: str = ""
    ) -> str:
        """Generate a conclusion for the report."""
        logger.info("Generating conclusion...")

        sections_text = "\n\n".join(
            f"### {s['topic']}\n{s['summary']}" for s in section_summaries
        )

        structure_block = ""
        if report_structure:
            structure_block = f"\n\nReport Structure:\n{report_structure}\n"

        prompt = f"""Write a conclusion for this research report ({word_count:,} words across {len(section_summaries)} sections).

Research Query: {query}
{structure_block}
Section Findings:
{sections_text}

Write a 400-600 word conclusion that synthesizes findings across sections, identifies overarching themes, and proposes specific future research directions."""

        response = self.client.complete(
            prompt=prompt,
            system=CONCLUSION_SYSTEM_PROMPT,
            max_tokens=self.config.llm.max_tokens.editor,
            temperature=self.config.llm.temperature.editor,
            model=self.config.llm.models.editor
        )

        return response
