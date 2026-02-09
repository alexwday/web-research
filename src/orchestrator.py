"""
Research Orchestrator for Deep Research Agent
Coordinates the entire research process using a 7-phase pipeline:
  Phase 1: Deep Pre-Planning
  Phase 2: Report Outline Design
  Phase 3: Task Planning per Section
  Phase 4: Research Execution
  Phase 5: Gap Analysis & Fill
  Phase 6: Section Synthesis
  Phase 7: Report Compilation
"""
import signal
from concurrent.futures import ThreadPoolExecutor, wait, as_completed, FIRST_COMPLETED
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from .config import get_config, TaskStatus, SectionStatus, ResearchTask, GlossaryTerm
from .database import get_database
from .agents import (
    PlannerAgent, ResearcherAgent, EditorAgent,
    OutlineDesignerAgent, SectionTaskPlannerAgent, GapAnalysisAgent, SynthesisAgent,
)
from .compiler import ReportCompiler
from .tools import save_markdown, read_file, count_words, count_citations, ensure_directory, generate_file_path
from .logger import (
    get_logger, console, print_header, print_success, print_error,
    print_warning, print_info, print_task_start, print_write,
    print_statistics_table, print_task_table, print_completion_summary,
    create_progress_bar
)

logger = get_logger(__name__)


class ResearchOrchestrator:
    """
    Main orchestrator for the deep research process.
    Coordinates planning, research, and compilation via a 7-phase pipeline.
    """

    def __init__(self, register_signals: bool = True):
        self.db = get_database()
        self.planner = PlannerAgent()
        self.researcher = ResearcherAgent()
        self.outline_designer = OutlineDesignerAgent()
        self.section_planner = SectionTaskPlannerAgent()
        self.gap_analyst = GapAnalysisAgent()
        self.synthesizer = SynthesisAgent()
        self.editor = EditorAgent()
        self.compiler = ReportCompiler()

        self.session_id: Optional[int] = None
        self.query: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.is_running = False
        self.phase: str = "idle"

        # Setup signal handlers for graceful shutdown (only from main thread)
        if register_signals:
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)

    @property
    def config(self):
        return get_config()

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print_warning("\nShutdown signal received. Completing current task...")
        self.is_running = False

    def _set_phase(self, new_phase: str):
        """Set phase and emit a phase_changed run event."""
        import json as _json
        old_phase = self.phase
        self.phase = new_phase
        if self.session_id is not None:
            self.db.add_run_event(
                session_id=self.session_id,
                task_id=None,
                event_type="phase_changed",
                phase=new_phase,
                severity="info",
                payload_json=_json.dumps({"old_phase": old_phase, "new_phase": new_phase}),
            )

    def run(self, query: str, resume: bool = False,
            refined_brief: str = None, refinement_qa: str = None) -> dict:
        """
        Run the full research process via the 7-phase pipeline.

        Args:
            query: The research query/topic
            resume: Whether to resume an existing session
            refined_brief: Optional enhanced research brief from query refinement
            refinement_qa: Optional JSON string of Q&A pairs from refinement

        Returns:
            Dict with output file paths and statistics
        """
        self.query = refined_brief or query
        self.start_time = datetime.now()
        self.is_running = True

        print_header(
            "Deep Research Agent",
            f"Starting comprehensive research on:\n{query[:100]}..."
        )

        try:
            # Initialize or Resume session
            if resume:
                session = self._resume_session()
                if not session:
                    print_info("No existing session found. Starting fresh.")
                    session = self._initialize_session(
                        query, refined_brief=refined_brief,
                        refinement_qa=refinement_qa)
            else:
                session = self._initialize_session(
                    query, refined_brief=refined_brief,
                    refinement_qa=refinement_qa)

            self.session_id = session.id

            # Check if session already has sections (resume case)
            existing_sections = self.db.get_all_sections(session_id=self.session_id)

            if not existing_sections:
                # Phase 1: Deep Pre-Planning
                self._set_phase("pre_planning")
                print_info("Phase 1: Deep pre-planning...")
                pre_plan_ctx = self.planner.run_pre_planning(self.query, self.session_id)

                if not self.is_running:
                    return self._emergency_compile()

                # Phase 2: Report Outline
                self._set_phase("outline_design")
                print_info("Phase 2: Designing report outline...")
                sections = self.outline_designer.design_outline(self.query, pre_plan_ctx, self.session_id)
                print_success(f"Designed outline with {len(sections)} sections")

                if not self.is_running:
                    return self._emergency_compile()

                # Phase 3: Task Planning per Section (parallel)
                self._set_phase("task_planning")
                print_info("Phase 3: Planning research tasks per section...")
                max_total = self.config.research.max_total_tasks
                budget_per_section = max(1, max_total // len(sections)) if sections else max_total

                def _plan_section(sec):
                    if not self.is_running:
                        return sec, []
                    return sec, self.section_planner.plan_tasks_for_section(
                        sec, sections, self.query, self.session_id,
                        task_budget=budget_per_section,
                    )

                total_created = 0
                with ThreadPoolExecutor(max_workers=min(len(sections), 4)) as plan_exec:
                    futures = {plan_exec.submit(_plan_section, s): s for s in sections}
                    for future in as_completed(futures):
                        try:
                            sec, tasks = future.result()
                            total_created += len(tasks)
                            print_info(f"  {sec.title}: {len(tasks)} tasks")
                        except Exception as e:
                            sec = futures[future]
                            logger.error(f"Task planning failed for '{sec.title}': {e}")

                # Show task overview
                all_tasks = self.db.get_all_tasks(session_id=self.session_id)
                print_task_table(all_tasks)
            else:
                sections = existing_sections
                print_info(f"Resuming with {len(sections)} existing sections")

            if not self.is_running:
                return self._emergency_compile()

            # Phase 4: Research Execution
            self._set_phase("researching")
            print_info("Phase 4: Executing research tasks...")
            self._run_research_loop()

            if not self.is_running:
                return self._emergency_compile()

            # Phase 5: Gap Analysis & Fill
            self._set_phase("gap_analysis")
            print_info("Phase 5: Gap analysis...")
            # Reload sections in case they were modified
            sections = self.db.get_all_sections(session_id=self.session_id)
            gap_result = self.gap_analyst.analyze_gaps(self.query, sections, self.session_id)
            if gap_result.get("new_tasks", 0) > 0:
                print_info(f"Gap analysis created {gap_result['new_tasks']} new tasks, "
                          f"{gap_result.get('new_sections', 0)} new sections")
                # Execute gap-fill research tasks
                self._set_phase("researching")
                print_info("Executing gap-fill research tasks...")
                self._run_research_loop()

            if not self.is_running:
                return self._emergency_compile()

            # Phase 6: Section Synthesis
            self._set_phase("synthesizing")
            print_info("Phase 6: Synthesizing sections...")
            sections = self.db.get_all_sections(session_id=self.session_id)
            self._synthesize_all_sections(self.query, sections)

            # Phase 7: Compile
            self._set_phase("compiling")
            print_info("Phase 7: Compiling final report...")
            output_files = self._compile_final_report()

            # Cleanup and Statistics
            self._set_phase("complete")
            return self._finalize(output_files)

        except KeyboardInterrupt:
            print_warning("\nInterrupted by user.")
            return self._emergency_compile()

        except Exception as e:
            logger.exception(f"Research failed: {e}")
            print_error(f"Research failed: {e}")
            return self._emergency_compile()

    def _initialize_session(self, query: str, refined_brief: str = None,
                            refinement_qa: str = None):
        """Initialize a new research session"""
        print_info("Initializing new research session...")

        # Create session in database
        session = self.db.create_session(query)

        # Store refinement data if provided
        if refined_brief or refinement_qa:
            self.db.update_session(
                session.id,
                refined_brief=refined_brief,
                refinement_qa=refinement_qa,
            )

        # Create output directory
        ensure_directory(self.config.output.directory)

        return session

    def _resume_session(self):
        """Resume an existing research session"""
        session = self.db.get_current_session()

        # If no active "running" session exists, allow resuming the most
        # recent session when it still has pending tasks.
        if not session:
            recent = self.db.get_most_recent_session()
            if recent:
                pending = self.db.get_task_count(TaskStatus.PENDING, session_id=recent.id)
                if pending > 0:
                    self.db.update_session(recent.id, status="running", ended_at=None)
                    session = self.db.get_session_by_id(recent.id)

        if session:
            print_info(f"Resuming session #{session.id}")
            print_info(f"Query: {session.query[:100]}...")

            stats = self.db.get_statistics(session_id=session.id)
            print_statistics_table(stats)

            self.query = session.query
            return session

        return None

    def _execute_single_task(self, task, other_sections):
        """Execute a single research task. Thread-safe — called from worker threads."""
        print_task_start(task.topic, task.id)

        content, new_tasks, glossary_terms = self.researcher.research_task(
            task,
            overall_query=self.query or "",
            other_sections=other_sections,
            session_id=self.session_id
        )

        # Save content to file
        save_markdown(task.file_path, content, append=False)

        # Update task statistics
        word_count = count_words(content)
        citation_count = count_citations(content)

        self.db.mark_task_complete(
            task.id,
            word_count=word_count,
            citation_count=citation_count
        )

        print_write(task.file_path, word_count)

        return {
            'task_id': task.id,
            'new_tasks': new_tasks,
            'glossary_terms': glossary_terms,
        }

    def _run_research_loop(self):
        """Execute the main research loop with parallel task execution."""
        print_info("Starting research loop...")

        loop_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 3
        max_loops = float('inf') if self.config.research.max_loops < 0 else self.config.research.max_loops
        max_runtime = timedelta(hours=self.config.research.max_runtime_hours) if self.config.research.max_runtime_hours else None
        max_workers = self.config.research.max_concurrent_tasks

        with create_progress_bar() as progress:
            total_tasks = self.db.get_task_count(session_id=self.session_id)
            progress_id = progress.add_task(
                "[cyan]Researching...",
                total=total_tasks
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                active = {}  # future -> task

                while self.is_running:
                    # Check termination conditions
                    if loop_count >= max_loops:
                        print_info(f"Reached maximum loop count ({max_loops})")
                        break

                    if max_runtime and (datetime.now() - self.start_time) > max_runtime:
                        print_info(f"Reached maximum runtime ({self.config.research.max_runtime_hours}h)")
                        break

                    # Fill available executor slots with new tasks
                    slots = max_workers - len(active)
                    if slots > 0:
                        new_tasks = self.db.get_next_tasks(slots, session_id=self.session_id)
                        if new_tasks:
                            all_tasks = self.db.get_all_tasks(session_id=self.session_id)
                            for task in new_tasks:
                                other_sections = [
                                    f"{t.topic} ({'done' if t.status == 'completed' else 'pending'})"
                                    for t in all_tasks if t.id != task.id
                                ]
                                future = executor.submit(
                                    self._execute_single_task, task, other_sections
                                )
                                active[future] = task

                    if not active:
                        # No running tasks and none to claim — check for retryable failures
                        retried = self.db.retry_failed_tasks(self.session_id, max_retries=2)
                        if retried > 0:
                            print_info(f"Retrying {retried} previously failed task(s)...")
                            continue

                        failed = self.db.get_task_count(TaskStatus.FAILED, session_id=self.session_id)
                        if failed > 0:
                            print_warning(f"No pending tasks remain; {failed} task(s) failed after retries.")
                        else:
                            print_success("All tasks completed!")
                        break

                    # Wait for at least one task to finish (timeout lets us re-check is_running)
                    done, _ = wait(active.keys(), return_when=FIRST_COMPLETED, timeout=2.0)

                    if not done:
                        continue

                    for future in done:
                        task = active.pop(future)
                        try:
                            result = future.result()

                            if result['new_tasks']:
                                self._add_recursive_tasks(result['new_tasks'])
                            if result['glossary_terms']:
                                self._add_glossary_terms(result['glossary_terms'], result['task_id'])

                            loop_count += 1
                            consecutive_failures = 0

                        except Exception as e:
                            logger.error(f"Task {task.id} failed: {e}")
                            self.db.mark_task_failed(task.id, str(e))
                            consecutive_failures += 1
                            if consecutive_failures >= max_consecutive_failures:
                                print_error(
                                    f"Aborting: {consecutive_failures} consecutive "
                                    f"task failures. Last error: {e}"
                                )
                                # Cancel queued (not-yet-started) futures
                                for f in list(active):
                                    f.cancel()
                                active.clear()
                                self.is_running = False
                                break

                        # Update progress
                        completed = self.db.get_task_count(TaskStatus.COMPLETED, session_id=self.session_id)
                        total = self.db.get_task_count(session_id=self.session_id)
                        progress.update(progress_id, completed=completed, total=total)

        # Final progress update
        stats = self.db.get_statistics(session_id=self.session_id)
        print_statistics_table(stats)

    def _add_recursive_tasks(self, new_tasks: List[dict]):
        """Add new tasks discovered during research"""
        # Check if we're at task limit
        current_count = self.db.get_task_count(session_id=self.session_id)
        remaining_capacity = self.config.research.max_total_tasks - current_count

        if remaining_capacity <= 0:
            print_warning("Task limit reached. Skipping new tasks.")
            return

        # Limit new tasks to remaining capacity
        tasks_to_add = new_tasks[:remaining_capacity]

        for i, task_data in enumerate(tasks_to_add, start=1):
            # Use an index to avoid file-path collisions for similarly named tasks.
            file_index = current_count + i
            task = ResearchTask(
                parent_id=task_data.get("parent_id"),
                section_id=task_data.get("section_id"),
                topic=task_data["topic"],
                description=task_data.get("description", ""),
                file_path=generate_file_path(
                    task_data["topic"],
                    self.config.output.directory,
                    file_index
                ),
                priority=task_data.get("priority", 5),
                depth=task_data.get("depth", 1),
                status=TaskStatus.PENDING
            )

            self.db.add_task(task, self.session_id)

        # Keep denormalized session task count in sync with recursive additions.
        self.db.update_session(
            self.session_id,
            total_tasks=self.db.get_task_count(session_id=self.session_id)
        )

        print_info(f"Added {len(tasks_to_add)} new research tasks")

    def _add_glossary_terms(self, terms: List[dict], task_id: int):
        """Add glossary terms discovered during research"""
        for term_data in terms:
            term = GlossaryTerm(
                term=term_data["term"],
                definition=term_data["definition"],
                first_occurrence_task_id=task_id
            )
            self.db.add_glossary_term(term, session_id=self.session_id)

    def _synthesize_all_sections(self, query: str, sections: list):
        """Synthesize all sections in parallel, providing adjacent context."""

        # Pre-compute adjacent context from descriptions (always available,
        # no dependency on other sections' synthesis output).
        def _build_adjacent(i):
            adjacent = {"previous": "", "next": ""}
            if i > 0:
                prev = sections[i - 1]
                adjacent["previous"] = f"**{prev.title}**: {prev.description}"
            if i < len(sections) - 1:
                nxt = sections[i + 1]
                adjacent["next"] = f"**{nxt.title}**: {nxt.description}"
            return adjacent

        # Filter to sections that need synthesis
        to_synthesize = []
        for i, section in enumerate(sections):
            if section.status == SectionStatus.COMPLETE.value or section.status == "complete":
                print_info(f"  Section '{section.title}' already synthesized, skipping")
                continue
            tasks = self.db.get_tasks_for_section(section.id)
            completed = [t for t in tasks if t.status == "completed"]
            if not completed:
                print_warning(f"  Section '{section.title}' has no completed tasks, skipping")
                continue
            to_synthesize.append((i, section))

        if not to_synthesize:
            print_success("Section synthesis complete (nothing to do)")
            return

        def _synthesize_one(idx_section):
            i, section = idx_section
            if not self.is_running:
                return section, None
            adjacent = _build_adjacent(i)
            print_info(f"  Synthesizing: {section.title}")
            content = self.synthesizer.synthesize_section(
                section, query, sections, adjacent, self.session_id
            )
            return section, content

        max_workers = min(len(to_synthesize), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as synth_exec:
            futures = {synth_exec.submit(_synthesize_one, item): item for item in to_synthesize}
            for future in as_completed(futures):
                try:
                    section, content = future.result()
                    if content:
                        word_count = count_words(content)
                        citation_count = count_citations(content)
                        self.db.mark_section_synthesized(
                            section.id, content, word_count, citation_count
                        )
                        section.synthesized_content = content
                        print_info(f"    {section.title}: {word_count} words, {citation_count} citations")
                except Exception as e:
                    item = futures[future]
                    logger.error(f"Synthesis failed for '{item[1].title}': {e}")

        print_success("Section synthesis complete")

    def _compile_final_report(self) -> dict:
        """Compile the final report from synthesized sections."""
        print_info("Compiling final report...")

        # Get all sections with synthesized content
        sections = self.db.get_all_sections(session_id=self.session_id)

        # Build chapters from synthesized sections
        chapters = []
        section_summaries = []
        for section in sections:
            content = section.synthesized_content
            if not content:
                # Fallback: try to read from task files (backward compatibility)
                tasks = self.db.get_tasks_for_section(section.id)
                completed = [t for t in tasks if t.status == "completed"]
                task_contents = []
                for t in completed:
                    tc = read_file(t.file_path)
                    if tc:
                        task_contents.append(tc)
                if task_contents:
                    content = "\n\n---\n\n".join(task_contents)

            if content:
                chapters.append({
                    "section": section,
                    "content": content,
                })
                # Build summary for exec summary/conclusion
                words = content.split()
                summary = " ".join(words[:500])
                if len(words) > 500:
                    summary += " ..."
                section_summaries.append({
                    "topic": section.title,
                    "summary": summary,
                })

        # If no sections exist (old-style session), fall back to task-based compilation
        if not chapters:
            return self._compile_task_based_report()

        # Generate executive summary and conclusion in parallel
        executive_summary = None
        conclusion = None
        total_words = sum(count_words(ch["content"]) for ch in chapters)

        # Build report structure
        report_structure = "\n".join(
            f"{s.position}. {s.title}" for s in sections if s.synthesized_content
        )

        with ThreadPoolExecutor(max_workers=2) as comp_executor:
            futures = {}

            if self.config.output.include_summary:
                self.db.add_run_event(
                    session_id=self.session_id, task_id=None,
                    event_type="agent_action", query_group="exec_summary",
                    query_text="Generating executive summary",
                )
                futures[comp_executor.submit(
                    self.editor.generate_executive_summary,
                    self.query, section_summaries, report_structure
                )] = 'summary'

            self.db.add_run_event(
                session_id=self.session_id, task_id=None,
                event_type="agent_action", query_group="conclusion",
                query_text="Generating conclusion",
            )
            futures[comp_executor.submit(
                self.editor.generate_conclusion,
                self.query, section_summaries, total_words, report_structure
            )] = 'conclusion'

            for future in as_completed(futures):
                label = futures[future]
                try:
                    result = future.result()
                    if label == 'summary':
                        executive_summary = result
                    else:
                        conclusion = result
                except Exception as e:
                    logger.warning(f"Failed to generate {label}: {e}")

        # Calculate duration
        duration_seconds = (datetime.now() - self.start_time).total_seconds()

        # Compile report
        output_files = self.compiler.compile_report(
            self.query,
            executive_summary,
            conclusion,
            duration_seconds,
            session_id=self.session_id,
            pre_read_chapters=chapters
        )

        # Store report artifacts on session
        if self.session_id:
            self.db.update_session(
                self.session_id,
                executive_summary=executive_summary,
                conclusion=conclusion,
                report_markdown_path=output_files.get("markdown"),
                report_html_path=output_files.get("html"),
            )

        return output_files

    def _compile_task_based_report(self) -> dict:
        """Fallback: compile report from tasks directly (for old sessions without sections)."""
        tasks = self.db.get_all_tasks(TaskStatus.COMPLETED, session_id=self.session_id)

        chapters = []
        section_summaries = []
        for task in tasks:
            content = read_file(task.file_path)
            if content:
                chapters.append({"task": task, "content": content})
                words = content.split()
                summary = " ".join(words[:500])
                if len(words) > 500:
                    summary += " ..."
                section_summaries.append({"topic": task.topic, "summary": summary})

        chapters.sort(key=lambda x: x["task"].file_path)

        # Generate exec summary and conclusion
        executive_summary = None
        conclusion = None
        total_words = sum(count_words(ch["content"]) for ch in chapters)

        with ThreadPoolExecutor(max_workers=2) as comp_executor:
            futures = {}
            if self.config.output.include_summary and section_summaries:
                futures[comp_executor.submit(
                    self.editor.generate_executive_summary,
                    self.query, section_summaries, ""
                )] = 'summary'
            if section_summaries:
                futures[comp_executor.submit(
                    self.editor.generate_conclusion,
                    self.query, section_summaries, total_words, ""
                )] = 'conclusion'

            for future in as_completed(futures):
                label = futures[future]
                try:
                    result = future.result()
                    if label == 'summary':
                        executive_summary = result
                    else:
                        conclusion = result
                except Exception as e:
                    logger.warning(f"Failed to generate {label}: {e}")

        duration_seconds = (datetime.now() - self.start_time).total_seconds()

        output_files = self.compiler.compile_report(
            self.query,
            executive_summary,
            conclusion,
            duration_seconds,
            session_id=self.session_id,
            pre_read_chapters=chapters
        )

        if self.session_id:
            self.db.update_session(
                self.session_id,
                executive_summary=executive_summary,
                conclusion=conclusion,
                report_markdown_path=output_files.get("markdown"),
                report_html_path=output_files.get("html"),
            )

        return output_files

    def _emergency_compile(self) -> dict:
        """Emergency compile when interrupted"""
        print_warning("Performing emergency compile of completed work...")

        try:
            duration_seconds = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

            output_files = self.compiler.compile_report(
                self.query or "Interrupted Research",
                None,  # No executive summary
                None,  # No conclusion
                duration_seconds,
                session_id=self.session_id
            )

            return self._finalize(output_files)

        except Exception as e:
            logger.error(f"Emergency compile failed: {e}")
            return {"error": str(e)}

    def _finalize(self, output_files: dict) -> dict:
        """Finalize the research session"""

        # Calculate final statistics
        duration_seconds = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        stats = self.db.get_statistics(session_id=self.session_id)

        # Update session with final stats and mark completed
        if self.session_id:
            self.db.update_session(
                self.session_id,
                completed_tasks=stats["completed_tasks"],
                total_words=stats["total_words"],
                total_sources=stats["total_sources"],
            )

            failed = stats.get("failed_tasks", 0)
            pending = stats.get("pending_tasks", 0)
            if pending > 0 and failed > 0:
                status = "partial_with_errors"
            elif pending > 0:
                status = "partial"
            elif failed > 0:
                status = "completed_with_errors"
            else:
                status = "completed"

            self.db.update_session(
                self.session_id,
                status=status,
                ended_at=datetime.now(timezone.utc)
            )

        # Print completion summary
        print_completion_summary(
            total_tasks=stats["total_tasks"],
            completed_tasks=stats["completed_tasks"],
            total_words=stats["total_words"],
            total_sources=stats["total_sources"],
            duration_seconds=duration_seconds
        )
        if stats.get("pending_tasks", 0) > 0:
            print_warning(
                f"Run finished with {stats['pending_tasks']} pending task(s). "
                "Resume to continue remaining work."
            )

        # Print output file locations
        console.print("\n[bold cyan]Output Files:[/bold cyan]")
        for fmt, path in output_files.items():
            console.print(f"  \u2022 {fmt}: {path}")

        return {
            "output_files": output_files,
            "statistics": stats,
            "duration_seconds": duration_seconds
        }


def run_research(query: str, resume: bool = False) -> dict:
    """
    Main entry point for running research

    Args:
        query: The research query/topic
        resume: Whether to resume an existing session

    Returns:
        Dict with output files and statistics
    """
    orchestrator = ResearchOrchestrator()
    return orchestrator.run(query, resume)
