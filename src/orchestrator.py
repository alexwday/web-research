"""
Research Orchestrator for Deep Research Agent
Coordinates the entire research process
"""
import time
import signal
from datetime import datetime, timedelta
from typing import Optional, List

from .config import get_config, TaskStatus, ResearchTask, GlossaryTerm
from .database import get_database
from .agents import PlannerAgent, ResearcherAgent, DiscoveryAgent, EditorAgent
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
    Coordinates planning, research, and compilation.
    """
    
    def __init__(self, register_signals: bool = True):
        self.db = get_database()
        self.planner = PlannerAgent()
        self.researcher = ResearcherAgent()
        self.discovery = DiscoveryAgent()
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
    
    def run(self, query: str, resume: bool = False) -> dict:
        """
        Run the full research process
        
        Args:
            query: The research query/topic
            resume: Whether to resume an existing session
        
        Returns:
            Dict with output file paths and statistics
        """
        self.query = query
        self.start_time = datetime.now()
        self.is_running = True
        
        print_header(
            "Deep Research Agent",
            f"Starting comprehensive research on:\n{query[:100]}..."
        )
        
        try:
            # Phase 1: Initialize or Resume
            self.phase = "planning"
            if resume:
                session = self._resume_session()
                if not session:
                    print_info("No existing session found. Starting fresh.")
                    session = self._initialize_session(query)
            else:
                session = self._initialize_session(query)
            
            self.session_id = session.id
            
            # Phase 2: Research Loop
            self.phase = "researching"
            self._run_research_loop()
            
            # Phase 3: Compile Report
            self.phase = "compiling"
            output_files = self._compile_final_report()
            
            # Phase 4: Cleanup and Statistics
            self.phase = "complete"
            return self._finalize(output_files)
            
        except KeyboardInterrupt:
            print_warning("\nInterrupted by user.")
            return self._emergency_compile()
            
        except Exception as e:
            logger.exception(f"Research failed: {e}")
            print_error(f"Research failed: {e}")
            return self._emergency_compile()
    
    def _initialize_session(self, query: str):
        """Initialize a new research session"""
        print_info("Initializing new research session...")
        
        # Create session in database
        session = self.db.create_session(query)
        
        # Create output directory
        ensure_directory(self.config.output.directory)
        
        # Generate research plan
        print_info("Planning research strategy...")
        tasks = self.planner.create_plan(query, session.id)
        
        print_success(f"Created research plan with {len(tasks)} tasks")
        
        # Show task overview
        print_task_table(tasks)
        
        return session
    
    def _resume_session(self):
        """Resume an existing research session"""
        session = self.db.get_current_session()
        
        if session:
            print_info(f"Resuming session #{session.id}")
            print_info(f"Query: {session.query[:100]}...")

            stats = self.db.get_statistics(session_id=session.id)
            print_statistics_table(stats)
            
            self.query = session.query
            return session
        
        return None
    
    def _run_research_loop(self):
        """Execute the main research loop"""
        print_info("Starting research loop...")

        loop_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 3
        max_loops = float('inf') if self.config.research.max_loops < 0 else self.config.research.max_loops
        max_runtime = timedelta(hours=self.config.research.max_runtime_hours) if self.config.research.max_runtime_hours else None
        
        with create_progress_bar() as progress:
            # Create progress task
            total_tasks = self.db.get_task_count(session_id=self.session_id)
            task_id = progress.add_task(
                "[cyan]Researching...",
                total=total_tasks
            )

            while self.is_running:
                # Check termination conditions
                if loop_count >= max_loops:
                    print_info(f"Reached maximum loop count ({max_loops})")
                    break

                if max_runtime and (datetime.now() - self.start_time) > max_runtime:
                    print_info(f"Reached maximum runtime ({self.config.research.max_runtime_hours}h)")
                    break

                # Get next task
                task = self.db.get_next_task(session_id=self.session_id)

                if not task:
                    print_success("All tasks completed!")
                    break

                # Update progress
                completed = self.db.get_task_count(TaskStatus.COMPLETED, session_id=self.session_id)
                total = self.db.get_task_count(session_id=self.session_id)
                progress.update(task_id, completed=completed, total=total)
                
                # Execute research
                print_task_start(task.topic, task.id)

                # Build context: list of all other section topics for the researcher
                all_tasks = self.db.get_all_tasks(session_id=self.session_id)
                other_sections = [
                    f"{t.topic} ({'done' if t.status == 'completed' else 'pending'})"
                    for t in all_tasks if t.id != task.id
                ]

                try:
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

                    # Handle new tasks from recursion
                    if new_tasks:
                        self._add_recursive_tasks(new_tasks)

                    # Handle glossary terms
                    if glossary_terms:
                        self._add_glossary_terms(glossary_terms, task.id)

                    loop_count += 1
                    consecutive_failures = 0  # reset on success

                    # Discovery step: run after every N completed tasks
                    if (
                        self.config.discovery.enabled
                        and loop_count % self.config.discovery.frequency == 0
                    ):
                        self._run_discovery()

                    # Delay between tasks
                    time.sleep(self.config.research.task_delay)

                except Exception as e:
                    logger.error(f"Task {task.id} failed: {e}")
                    self.db.mark_task_failed(task.id, str(e))
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print_error(
                            f"Aborting: {consecutive_failures} consecutive "
                            f"task failures. Last error: {e}"
                        )
                        break
                    continue
        
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
        
        for task_data in tasks_to_add:
            task = ResearchTask(
                parent_id=task_data.get("parent_id"),
                topic=task_data["topic"],
                description=task_data.get("description", ""),
                file_path=generate_file_path(
                    task_data["topic"],
                    self.config.output.directory
                ),
                priority=task_data.get("priority", 5),
                depth=task_data.get("depth", 1),
                status=TaskStatus.PENDING
            )
            
            self.db.add_task(task, self.session_id)
        
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
    
    def _run_discovery(self):
        """Run the discovery agent to identify research gaps."""
        print_info("Running discovery agent for gap analysis...")

        all_tasks = self.db.get_all_tasks(session_id=self.session_id)
        completed_tasks = [t for t in all_tasks if t.status == "completed"]

        if not completed_tasks:
            return

        # Build task list for discovery agent
        task_list = [
            {"topic": t.topic, "status": t.status}
            for t in all_tasks
        ]

        # Build summaries from completed sections
        section_summaries = []
        for t in completed_tasks:
            content = read_file(t.file_path)
            if not content:
                continue
            words = content.split()
            summary = " ".join(words[:300])
            if len(words) > 300:
                summary += " ..."
            section_summaries.append({"topic": t.topic, "summary": summary})

        new_tasks = self.discovery.discover_tasks(
            query=self.query or "",
            all_tasks=task_list,
            section_summaries=section_summaries,
            session_id=self.session_id
        )

        if new_tasks:
            self._add_recursive_tasks(new_tasks)
            print_info(f"Discovery agent suggested {len(new_tasks)} new tasks")
        else:
            print_info("Discovery agent found no significant gaps")

    def _build_section_summaries(self, tasks) -> tuple:
        """Build section summaries from completed task files.

        Reads the first ~500 words of each section to give the editor
        real content to synthesize rather than just topic names.

        Returns (summaries, chapters) where chapters is a list of
        {"task": task, "content": full_content} dicts for reuse by the compiler.
        """
        summaries = []
        chapters = []
        for task in tasks:
            content = read_file(task.file_path)
            if not content:
                continue
            chapters.append({"task": task, "content": content})
            # Take first ~500 words as a summary
            words = content.split()
            summary = " ".join(words[:500])
            if len(words) > 500:
                summary += " ..."
            summaries.append({"topic": task.topic, "summary": summary})
        return summaries, chapters

    def _rewrite_sections(self, chapters: list) -> tuple:
        """Reorder and rewrite sections for cohesion.

        Returns (rewritten_chapters, new_section_summaries).
        """
        print_info("Running editor rewrite pass...")

        # Step 1: Determine optimal section order
        topics = [ch["task"].topic for ch in chapters]
        ordered_topics = self.editor.determine_section_order(
            self.query or "", topics
        )

        # Reorder chapters to match
        topic_to_chapter = {ch["task"].topic: ch for ch in chapters}
        ordered_chapters = []
        for topic in ordered_topics:
            if topic in topic_to_chapter:
                ordered_chapters.append(topic_to_chapter[topic])
        # Append any chapters not in the ordered list (safety)
        seen = set(ordered_topics)
        for ch in chapters:
            if ch["task"].topic not in seen:
                ordered_chapters.append(ch)

        print_info(f"Sections reordered: {len(ordered_chapters)} sections")

        # Step 2: Rewrite each section with awareness of neighbors
        toc = [ch["task"].topic for ch in ordered_chapters]
        rewritten_chapters = []
        preceding_summaries = []

        for i, chapter in enumerate(ordered_chapters):
            following_topics = toc[i + 1:]

            rewritten_content = self.editor.rewrite_section(
                query=self.query or "",
                section_topic=chapter["task"].topic,
                section_content=chapter["content"],
                toc=toc,
                preceding_summaries=preceding_summaries,
                following_topics=following_topics,
            )

            rewritten_chapters.append({
                "task": chapter["task"],
                "content": rewritten_content,
            })

            # Save rewritten content to file
            save_markdown(chapter["task"].file_path, rewritten_content, append=False)

            # Build summary of this section for the next iteration
            words = rewritten_content.split()
            summary = " ".join(words[:500])
            if len(words) > 500:
                summary += " ..."
            preceding_summaries.append({
                "topic": chapter["task"].topic,
                "summary": summary,
            })

            print_write(chapter["task"].file_path, count_words(rewritten_content))

        print_info("Editor rewrite pass complete")

        # Rebuild section_summaries from rewritten content
        section_summaries = []
        for ch in rewritten_chapters:
            words = ch["content"].split()
            summary = " ".join(words[:500])
            if len(words) > 500:
                summary += " ..."
            section_summaries.append({
                "topic": ch["task"].topic,
                "summary": summary,
            })

        return rewritten_chapters, section_summaries

    def _compile_final_report(self) -> dict:
        """Compile the final report"""
        print_info("Compiling final report...")

        # Get completed tasks and build section summaries (also pre-reads files)
        tasks = self.db.get_all_tasks(TaskStatus.COMPLETED, session_id=self.session_id)
        section_summaries, pre_read_chapters = self._build_section_summaries(tasks)

        # Editor rewrite pass: reorder and rewrite sections for cohesion
        if self.config.rewrite.enabled and len(pre_read_chapters) > 1:
            pre_read_chapters, section_summaries = self._rewrite_sections(
                pre_read_chapters
            )

        # Generate executive summary
        executive_summary = None
        if self.config.output.include_summary:
            try:
                executive_summary = self.editor.generate_executive_summary(
                    self.query,
                    section_summaries
                )
            except Exception as e:
                logger.warning(f"Failed to generate executive summary: {e}")

        # Generate conclusion
        conclusion = None
        total_words = self.db.get_total_word_count(session_id=self.session_id)
        try:
            conclusion = self.editor.generate_conclusion(
                self.query,
                section_summaries,
                total_words
            )
        except Exception as e:
            logger.warning(f"Failed to generate conclusion: {e}")
        
        # Calculate duration
        duration_seconds = (datetime.now() - self.start_time).total_seconds()
        
        # Compile report (pass pre-read chapters to avoid re-reading files)
        output_files = self.compiler.compile_report(
            self.query,
            executive_summary,
            conclusion,
            duration_seconds,
            session_id=self.session_id,
            pre_read_chapters=pre_read_chapters
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
            self.db.complete_session(self.session_id)
        
        # Print completion summary
        print_completion_summary(
            total_tasks=stats["total_tasks"],
            completed_tasks=stats["completed_tasks"],
            total_words=stats["total_words"],
            total_sources=stats["total_sources"],
            duration_seconds=duration_seconds
        )
        
        # Print output file locations
        console.print("\n[bold cyan]Output Files:[/bold cyan]")
        for fmt, path in output_files.items():
            console.print(f"  • {fmt}: {path}")
        
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
