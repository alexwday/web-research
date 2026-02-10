"""OutlineDesignerAgent — designs the report outline from pre-planning context."""
import json
from typing import List

from src.config.settings import get_config
from src.config.types import ReportSection, SectionStatus
from src.infra.llm import get_llm_client
from src.infra._database import get_database
from src.config.logger import get_logger

from src.pipeline._stages._prompts import get_prompt_set

logger = get_logger(__name__)


class OutlineDesignerAgent:
    """Agent responsible for designing the report outline from pre-planning context"""

    def __init__(self):
        self.client = get_llm_client()
        self.db = get_database()

    @property
    def config(self):
        return get_config()

    def design_outline(self, query: str, pre_plan_context: str, session_id: int) -> List[ReportSection]:
        """Design report sections from the pre-planning analysis.

        Returns list of ReportSection objects saved to DB.
        """
        target_sections = self.config.research.min_initial_tasks
        max_sections = max(target_sections + 2, int(target_sections * 1.5))
        logger.info(f"Designing report outline (target ~{target_sections} sections, max {max_sections})...")

        ps = get_prompt_set("design_outline", "design_outline")
        system = ps["system"].format(
            target_sections=target_sections,
            max_sections=max_sections,
        )

        prompt = ps["user"].format(
            query=query,
            pre_plan_context=pre_plan_context,
            target_sections=target_sections,
            max_sections=max_sections,
        )

        try:
            response = self.client.complete(
                prompt=prompt,
                system=system,
                max_tokens=self.config.llm.max_tokens.outline_designer,
                temperature=self.config.llm.temperature.outline_designer,
                json_mode=True,
                model=self.config.llm.models.outline_designer
            )

            data = json.loads(response)
            section_list = data.get("sections", [])

            if not section_list:
                raise ValueError("No sections returned by outline designer")

            # Hard-cap: truncate if LLM exceeded the maximum
            if len(section_list) > max_sections:
                logger.warning(f"Outline designer returned {len(section_list)} sections, truncating to {max_sections}")
                section_list = section_list[:max_sections]

            # Create ReportSection objects
            sections = []
            for item in section_list:
                sections.append(ReportSection(
                    title=item.get("title", "Untitled Section"),
                    description=item.get("description", ""),
                    position=item.get("position", len(sections) + 1),
                    status=SectionStatus.PLANNED,
                ))

            # Save to DB
            saved_sections = self.db.add_sections_bulk(sections, session_id)
            logger.info(f"Designed outline with {len(saved_sections)} sections")
            return saved_sections

        except Exception as e:
            logger.error(f"Outline design failed: {e}")
            raise
