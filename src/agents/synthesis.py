"""SynthesisAgent — synthesizes research notes into polished section prose."""
from typing import List, Dict

from ..config import get_config, ReportSection, SectionStatus
from ..llm_client import get_llm_client
from ..tools import read_file
from ..database import get_database
from ..logger import get_logger

from .prompts import SECTION_SYNTHESIS_SYSTEM_PROMPT, _SYNTHESIS_STYLE_GUIDANCE

logger = get_logger(__name__)


class SynthesisAgent:
    """Agent responsible for synthesizing research notes into polished section prose"""

    def __init__(self):
        self.client = get_llm_client()
        self.db = get_database()

    @property
    def config(self):
        return get_config()

    def synthesize_section(
        self,
        section: ReportSection,
        query: str,
        all_sections: List[ReportSection],
        adjacent_summaries: Dict[str, str],
        session_id: int
    ) -> str:
        """Synthesize all research notes for a section into polished prose.

        Args:
            section: The section to synthesize
            query: Original research query
            all_sections: Full outline for context
            adjacent_summaries: {"previous": str, "next": str} summaries
            session_id: Current session ID

        Returns:
            Synthesized markdown content
        """
        logger.info(f"Synthesizing section: {section.title}")

        # Mark section as synthesizing
        self.db.update_section(section.id, status=SectionStatus.SYNTHESIZING.value)

        # Gather all research notes for this section
        tasks = self.db.get_tasks_for_section(section.id)
        completed = [t for t in tasks if t.status == "completed"]

        research_notes = []
        for task in completed:
            content = read_file(task.file_path)
            if content:
                research_notes.append(f"### Research Task: {task.topic}\n\n{content}")

        if not research_notes:
            logger.warning(f"No research notes for section '{section.title}'")
            return ""

        notes_text = "\n\n---\n\n".join(research_notes)

        # Build outline context
        outline_text = "\n".join(
            f"{s.position}. {s.title}" for s in all_sections
        )

        # Build adjacent section context
        adjacent_text = ""
        if adjacent_summaries.get("previous"):
            adjacent_text += f"\n## Previous Section Summary\n{adjacent_summaries['previous']}\n"
        if adjacent_summaries.get("next"):
            adjacent_text += f"\n## Next Section Preview\n{adjacent_summaries['next']}\n"

        style = getattr(self.config.synthesis, 'style', 'balanced')
        style_guidance = _SYNTHESIS_STYLE_GUIDANCE.get(style, _SYNTHESIS_STYLE_GUIDANCE['balanced'])

        system = SECTION_SYNTHESIS_SYSTEM_PROMPT.format(
            min_words=self.config.synthesis.min_words_per_section,
            max_words=self.config.synthesis.max_words_per_section,
            min_citations=self.config.synthesis.min_citations_per_section,
            style_guidance=style_guidance,
        )

        prompt = f"""Synthesize the following research notes into a polished section.

## Research Query
{query}

## Report Outline
{outline_text}

## This Section
Title: {section.title}
Description: {section.description}
{adjacent_text}
## Research Notes
{notes_text}

---
Write a cohesive, well-structured section that synthesizes all the findings above."""

        response = self.client.complete(
            prompt=prompt,
            system=system,
            max_tokens=self.config.llm.max_tokens.synthesizer,
            temperature=self.config.llm.temperature.synthesizer,
            model=self.config.llm.models.synthesizer
        )

        return response
