"""GapAnalysisAgent — identifies gaps after initial research completes."""
import json
from typing import List, Dict

from src.config.settings import get_config
from src.config.types import ResearchTask, ReportSection, TaskStatus, SectionStatus
from src.infra.llm import get_llm_client
from src.pipeline._tools import read_file, generate_file_path
from src.infra._database import get_database
from src.config.logger import get_logger

from src.pipeline._stages._prompts import get_prompt_set

logger = get_logger(__name__)


class GapAnalysisAgent:
    """Agent responsible for identifying gaps after initial research completes"""

    def __init__(self):
        self.client = get_llm_client()
        self.db = get_database()

    @property
    def config(self):
        return get_config()

    def analyze_gaps(
        self,
        query: str,
        sections: List[ReportSection],
        session_id: int
    ) -> dict:
        """Analyze research gaps at both section and cross-section levels.

        Creates new tasks and sections in DB as needed.
        Returns dict with counts of new items created.
        """
        if not self.config.gap_analysis.enabled:
            logger.info("Gap analysis is disabled")
            return {"new_tasks": 0, "new_sections": 0}

        logger.info("Running comprehensive gap analysis...")

        # Build section summaries from completed task notes
        section_summaries = []
        for section in sections:
            tasks = self.db.get_tasks_for_section(section.id)
            completed = [t for t in tasks if t.status == "completed"]
            notes = []
            for t in completed:
                content = read_file(t.file_path)
                if content:
                    words = content.split()
                    notes.append(f"**{t.topic}**: " + " ".join(words[:200]))
            section_summaries.append({
                "title": section.title,
                "description": section.description,
                "research_notes": "\n".join(notes) if notes else "(no research completed)"
            })

        # Build prompt
        outline_text = "\n".join(
            f"{s.position}. {s.title}: {s.description}" for s in sections
        )

        summaries_text = "\n\n".join(
            f"### {s['title']}\n{s['description']}\n\nResearch gathered:\n{s['research_notes']}"
            for s in section_summaries
        )

        ps = get_prompt_set("review_gaps", "analyze_gaps")
        prompt = ps["user"].format(
            query=query,
            outline_text=outline_text,
            summaries_text=summaries_text,
            max_new_sections=self.config.gap_analysis.max_new_sections,
            max_gap_fill_tasks=self.config.gap_analysis.max_gap_fill_tasks,
        )

        try:
            response = self.client.complete(
                prompt=prompt,
                system=ps["system"],
                max_tokens=self.config.llm.max_tokens.analyzer,
                temperature=self.config.llm.temperature.analyzer,
                json_mode=True,
                model=self.config.llm.models.analyzer
            )

            data = json.loads(response)
            return self._process_gaps(data, sections, query, session_id)

        except Exception as e:
            logger.warning(f"Gap analysis failed: {e}")
            return {"new_tasks": 0, "new_sections": 0}

    def _process_gaps(
        self,
        data: dict,
        sections: List[ReportSection],
        query: str,
        session_id: int
    ) -> dict:
        """Process gap analysis results: create tasks and sections in DB."""
        output_dir = self.config.output.directory
        existing_count = self.db.get_task_count(session_id=session_id)
        task_index = existing_count
        total_new_tasks = 0
        total_new_sections = 0
        max_gap_tasks = self.config.gap_analysis.max_gap_fill_tasks

        section_map = {s.title: s for s in sections}

        # Process per-section gaps
        for gap in data.get("section_gaps", []):
            if total_new_tasks >= max_gap_tasks:
                break
            section_title = gap.get("section_title", "")
            section = section_map.get(section_title)
            if not section:
                continue

            for task_data in gap.get("suggested_tasks", []):
                if total_new_tasks >= max_gap_tasks:
                    break
                task_index += 1
                task = ResearchTask(
                    section_id=section.id,
                    topic=task_data.get("topic", "Gap-fill task"),
                    description=task_data.get("description", ""),
                    file_path=generate_file_path(
                        task_data.get("topic", "gap-fill"),
                        output_dir,
                        task_index
                    ),
                    priority=task_data.get("priority", 5),
                    depth=0,
                    is_gap_fill=True,
                    status=TaskStatus.PENDING
                )
                self.db.add_task(task, session_id)
                total_new_tasks += 1

        # Process new sections
        max_new = self.config.gap_analysis.max_new_sections
        max_position = max((s.position for s in sections), default=0)
        for new_sec in data.get("new_sections", [])[:max_new]:
            if total_new_tasks >= max_gap_tasks:
                break
            max_position += 1
            new_section = ReportSection(
                title=new_sec.get("title", "New Section"),
                description=new_sec.get("description", ""),
                position=new_sec.get("position", max_position),
                status=SectionStatus.RESEARCHING,
                is_gap_fill=True,
            )
            saved_section = self.db.add_section(new_section, session_id)
            total_new_sections += 1

            for task_data in new_sec.get("suggested_tasks", []):
                if total_new_tasks >= max_gap_tasks:
                    break
                task_index += 1
                task = ResearchTask(
                    section_id=saved_section.id,
                    topic=task_data.get("topic", "Gap-fill task"),
                    description=task_data.get("description", ""),
                    file_path=generate_file_path(
                        task_data.get("topic", "gap-fill"),
                        output_dir,
                        task_index
                    ),
                    priority=task_data.get("priority", 5),
                    depth=0,
                    is_gap_fill=True,
                    status=TaskStatus.PENDING
                )
                self.db.add_task(task, session_id)
                total_new_tasks += 1

        # Update session task count
        total = self.db.get_task_count(session_id=session_id)
        self.db.update_session(session_id, total_tasks=total)

        logger.info(f"Gap analysis created {total_new_tasks} tasks, {total_new_sections} new sections")
        return {"new_tasks": total_new_tasks, "new_sections": total_new_sections}
