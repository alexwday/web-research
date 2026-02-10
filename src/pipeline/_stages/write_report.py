"""EditorAgent — executive summary and conclusion generation."""
from typing import List, Dict

from src.config.settings import get_config
from src.infra.llm import get_llm_client
from src.infra._database import get_database
from src.config.logger import get_logger

from src.pipeline._stages._prompts import get_prompt_set

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

        ps = get_prompt_set("write_report", "executive_summary")
        prompt = ps["user"].format(
            query=query,
            structure_block=structure_block,
            sections_text=sections_text,
        )

        response = self.client.complete(
            prompt=prompt,
            system=ps["system"],
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

        ps = get_prompt_set("write_report", "conclusion")
        prompt = ps["user"].format(
            word_count=f"{word_count:,}",
            num_sections=len(section_summaries),
            query=query,
            structure_block=structure_block,
            sections_text=sections_text,
        )

        response = self.client.complete(
            prompt=prompt,
            system=ps["system"],
            max_tokens=self.config.llm.max_tokens.editor,
            temperature=self.config.llm.temperature.editor,
            model=self.config.llm.models.editor
        )

        return response
