"""SectionTaskPlannerAgent — creates research tasks for each section."""
import json
from typing import List

from src.config.settings import get_config
from src.config.types import ResearchTask, ReportSection, TaskStatus, SectionStatus
from src.infra.llm import get_llm_client
from src.pipeline._tools import generate_file_path
from src.infra._database import get_database
from src.config.logger import get_logger

from src.pipeline._stages._prompts import get_prompt_set

logger = get_logger(__name__)


class SectionTaskPlannerAgent:
    """Agent responsible for creating research tasks for each section"""

    def __init__(self):
        self.client = get_llm_client()
        self.db = get_database()

    @property
    def config(self):
        return get_config()

    def plan_tasks_for_section(
        self,
        section: ReportSection,
        all_sections: List[ReportSection],
        query: str,
        session_id: int,
        task_budget: int = None,
    ) -> List[ResearchTask]:
        """Generate research tasks for a single section.

        Returns list of ResearchTask objects saved to DB.
        """
        logger.info(f"Planning tasks for section: {section.title}")

        tasks_per_section = self.config.research.tasks_per_section
        if task_budget is not None:
            tasks_per_section = min(tasks_per_section, task_budget)

        # Build outline context
        outline_text = "\n".join(
            f"{s.position}. {s.title}: {s.description}" for s in all_sections
        )

        ps = get_prompt_set("plan_tasks", "plan_tasks")
        system = ps["system"].format(
            tasks_per_section=tasks_per_section
        )

        prompt = ps["user"].format(
            query=query,
            outline_text=outline_text,
            section_title=section.title,
            section_description=section.description,
            tasks_per_section=tasks_per_section,
        )

        try:
            response = self.client.complete(
                prompt=prompt,
                system=system,
                max_tokens=self.config.llm.max_tokens.planner,
                temperature=self.config.llm.temperature.planner,
                json_mode=True,
                model=self.config.llm.models.planner
            )

            data = json.loads(response)
            task_list = data.get("tasks", [])

            if not task_list:
                # Fallback: create a single task from the section description
                task_list = [{
                    "topic": section.title,
                    "description": section.description,
                    "priority": 5
                }]

            output_dir = f"{self.config.output.directory}/session_{session_id}"
            existing_count = self.db.get_task_count(session_id=session_id)

            pending_tasks = []
            for i, item in enumerate(task_list[:tasks_per_section]):
                file_index = existing_count + i + 1
                pending_tasks.append(ResearchTask(
                    section_id=section.id,
                    topic=item.get("topic", f"Task {i+1} for {section.title}"),
                    description=item.get("description", ""),
                    file_path=generate_file_path(
                        item.get("topic", section.title),
                        output_dir,
                        file_index
                    ),
                    priority=item.get("priority", 5),
                    depth=0,
                    status=TaskStatus.PENDING
                ))

            tasks = self.db.add_tasks_bulk(pending_tasks, session_id)

            # Update section status
            self.db.update_section(section.id, status=SectionStatus.RESEARCHING.value)

            # Update session total tasks
            total = self.db.get_task_count(session_id=session_id)
            self.db.update_session(session_id, total_tasks=total)

            logger.info(f"Created {len(tasks)} tasks for section '{section.title}'")
            return tasks

        except Exception as e:
            logger.error(f"Task planning failed for section '{section.title}': {e}")
            raise
