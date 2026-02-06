"""
Agents Module for Deep Research Agent
Contains the Planner, Researcher, and Editor agents
"""
import json
import re
from typing import List, Dict, Any, Tuple

from .config import get_config, ResearchTask, TaskStatus
from .llm_client import get_llm_client
from .tools import (
    web_search, extract_source_info,
    truncate_to_tokens, generate_file_path
)
from .database import get_database
from .logger import get_logger, print_search, print_scrape

logger = get_logger(__name__)


# =============================================================================
# PROMPTS
# =============================================================================

PLANNER_SYSTEM_PROMPT = """You are a Research Architect specializing in breaking down complex topics into comprehensive, in-depth research plans.

Your task is to analyze a research query and create a detailed plan that, when executed, will result in a comprehensive report covering every important aspect of the topic in depth.

You will be provided with preliminary web search results to ground your plan in real, current information. Use these results to:
- Identify the key subtopics, themes, and angles that actually exist in the literature
- Discover terminology, frameworks, or debates you might not have known about
- Ensure your plan covers what is actually being discussed, not just what you assume

PRIORITY RULES (tasks are executed from highest to lowest priority):
- Priority 9-10: Introduction/Overview, Historical Background, Core Definitions (research first to build foundation)
- Priority 7-8: Main body content, detailed analysis, current state
- Priority 5-6: Applications, case studies, specific examples
- Priority 3-4: Edge cases, alternative perspectives, limitations
- Priority 1-2: Future Directions, Conclusion/Synthesis (research LAST after all body content exists)

Note: The system will enforce these ordering rules automatically, but setting correct priorities helps produce a coherent report.

GUIDELINES:
1. Break the topic into logical chapters/sections following academic structure
2. Each task should be specific enough to research in a single focused session
3. Include tasks for edge cases, controversies, and alternative perspectives
4. Prioritize foundational knowledge before advanced topics
5. Aim for {min_tasks} to {max_tasks} tasks
6. Each task must be a RESEARCH topic — not a meta-task like "Write introduction"
7. Ground your plan in the preliminary search results — do not ignore them

OUTPUT FORMAT:
Output ONLY a valid JSON object with a "tasks" array. Each task object must have:
- "topic": Brief title (max 100 chars)
- "description": Detailed research instructions (2-4 sentences explaining exactly what to investigate)
- "priority": 1-10 (10 = research first, 1 = research last)

Example:
{{
  "tasks": [
    {{
      "topic": "Historical Origins of Machine Learning",
      "description": "Research the early history of machine learning from 1950s-1980s. Focus on key papers, pioneering researchers like Alan Turing, Arthur Samuel, and Frank Rosenblatt. Document the evolution from simple perceptrons to early neural networks.",
      "priority": 9
    }}
  ]
}}"""


RESEARCHER_SYSTEM_PROMPT = """You are a Deep Research Specialist writing a section for an academic report.

WRITING GUIDELINES:
1. Aim for {min_words}-{max_words} words. Prioritize specific facts, data, and analysis over general statements. Every paragraph should contain at least one concrete claim with a citation. Cut filler.
2. Include specific facts, figures, dates, and direct quotes from sources
3. Structure with subheadings (### for main subsections, #### for sub-subsections). Do NOT write a top-level ## heading — the section title is added automatically.
4. Define technical terms when first introduced
5. Address multiple perspectives and controversies if they exist
6. Do not write a general introduction or conclusion for this section — jump directly into the substance
7. NEVER add disclaimers about sources being unavailable, training data cutoffs, or limitations of your knowledge. You HAVE source material — use it. Do not write "Note on sources" paragraphs.

CITATION FORMAT:
- Sources are numbered in the order listed under "Source Material" below
- Cite using numbered references: [1], [2], etc.
- Source 1 = first source, Source 2 = second source, and so on
- Direct quotes: "quoted text" [3]
- Minimum {min_citations} citations required
- You MUST cite from the provided sources. Do not invent citations.

NOTE: Source content may be truncated. Do not assume you have the complete text of any source.

OUTPUT FORMAT:
Write the section in Markdown format.

ABOUT NEW TASKS:
- Most research tasks should NOT spawn new tasks — only do so if something critical was discovered that cannot be covered here
- Never suggest more than 1 new task
- Never suggest tasks that overlap with existing sections listed in the prompt

If you discover something critical, include at the END of your response:

```json
{{
  "new_tasks": [
    {{"topic": "...", "description": "...", "priority": 3}}
  ],
  "glossary_terms": [
    {{"term": "...", "definition": "..."}}
  ]
}}
```

For most tasks, do NOT include any JSON block."""


EXEC_SUMMARY_SYSTEM_PROMPT = """You are a Research Editor writing the executive summary for a comprehensive research report.

Your job is to synthesize the ACTUAL FINDINGS from the section summaries provided — not to restate topic names or write generic filler.

Requirements:
- 300-500 words
- Lead with the single most important finding or insight
- Reference specific data, facts, or conclusions from the sections
- Provide context for why this research matters
- Preview the report structure briefly at the end
- Professional academic prose; no bullet lists"""


CONCLUSION_SYSTEM_PROMPT = """You are a Research Editor writing the conclusion for a comprehensive research report.

Your job is to synthesize findings ACROSS sections, draw connections the individual sections could not, and identify overarching themes.

Requirements:
- 400-600 words
- Do NOT simply restate what each section said — find the throughlines
- Identify 2-3 overarching themes or tensions that emerged
- Discuss practical implications
- Propose specific areas for future research (not vague "more research is needed")
- Professional academic prose; no bullet lists"""


DISCOVERY_SYSTEM_PROMPT = """You are a Research Gap Analyst. Your sole job is to review completed research sections against the original query and identify critical gaps that deserve their own dedicated section.

You will receive:
- The original research query
- The full list of existing tasks (completed and pending)
- Summaries of completed sections

Your task:
1. Compare what has been written against what a comprehensive report on this topic SHOULD cover
2. Identify important sub-topics, perspectives, or angles that are missing
3. Do NOT suggest topics that overlap with existing or pending tasks
4. Do NOT suggest meta-tasks (e.g. "Write introduction") — only research topics
5. Prioritize gaps that would significantly improve report comprehensiveness

OUTPUT FORMAT:
Output ONLY a valid JSON object:
{{
  "suggestions": [
    {{
      "topic": "Brief title (max 100 chars)",
      "description": "2-3 sentences explaining what to investigate and why it matters",
      "priority": 3
    }}
  ]
}}

If no significant gaps exist, return: {{"suggestions": []}}

Priority guidelines:
- 7-8: Critical gap — report is incomplete without this
- 5-6: Important — adds significant value
- 3-4: Nice to have — deepens coverage but not essential"""


REORDER_SYSTEM_PROMPT = """You are a Report Structure Editor. Your job is to determine the optimal reading order for the sections of a research report.

Rules:
- Introduction/Overview/Background sections should come FIRST
- Core concepts and definitions should come early
- Main body content should flow logically from general to specific
- Case studies and applications should come after the concepts they illustrate
- Comparative analysis and alternative perspectives should come after the things they compare
- Conclusion/Future Directions/Synthesis sections should come LAST
- Adjacent sections should have natural thematic connections

OUTPUT FORMAT:
Output ONLY a valid JSON object with an "order" array containing the section topics in optimal reading order:
{{
  "order": [
    "First Section Topic",
    "Second Section Topic",
    ...
  ]
}}

Include ALL sections — do not drop any."""


REWRITE_SYSTEM_PROMPT = """You are a Report Editor performing a cohesion rewrite pass. You are rewriting ONE section of a larger research report to improve flow, reduce repetition, and strengthen transitions.

You will receive:
- The full table of contents (so you know the report structure)
- Summaries of the sections that come BEFORE this one
- The topics of sections that come AFTER this one
- The original content of THIS section

Your task:
1. Rewrite this section so it reads naturally in context:
   - Remove content that repeats what prior sections already covered (reference them instead: "As discussed in the section on X...")
   - Add brief forward references where helpful ("This will be explored further in the section on Y")
   - Ensure terminology is consistent with prior sections
   - Smooth the opening so it transitions naturally from the previous section
2. Preserve ALL substantive content, facts, data, and citations — do not remove information, only restructure and rephrase
3. Keep all [N] citation references exactly as they are — do not renumber or remove them
4. Maintain the same markdown heading structure (### and ####)
5. Keep approximately the same length — this is a cohesion edit, not a summary

OUTPUT: The rewritten section content in Markdown format. Do NOT include the section title (## heading) — only the body content."""


QUERY_GENERATOR_SYSTEM = """You are a search query specialist. Generate maximally diverse search queries — each must target a different angle or source type."""


QUERY_GENERATOR_PROMPT = """Generate {num_queries} search queries for:

Topic: {topic}
Focus: {description}

Each query MUST target a different angle (e.g., one broad overview, one specific/technical, one recent data or news). Vary terminology across queries. 3-8 words each.

Output ONLY the queries, one per line, no numbering or bullets."""


# =============================================================================
# PLANNER AGENT
# =============================================================================

class PlannerAgent:
    """Agent responsible for creating the initial research plan"""

    def __init__(self):
        self.client = get_llm_client()
        self.db = get_database()

    @property
    def config(self):
        return get_config()
    
    def create_plan(self, query: str, session_id: int) -> List[ResearchTask]:
        """
        Analyze the query and create a comprehensive research plan.
        Runs preliminary web searches first to ground the plan in real data.
        """
        logger.info(f"Creating research plan for: {query[:100]}...")

        # Pre-search: gather web context so the planner is informed
        search_context = self._pre_search(query)

        # Format the system prompt with config values
        max_initial = self.config.research.max_total_tasks
        min_initial = min(self.config.research.min_initial_tasks, max_initial)
        system = PLANNER_SYSTEM_PROMPT.format(
            min_tasks=min_initial,
            max_tasks=max_initial
        )

        search_block = ""
        if search_context:
            search_block = f"""

## Preliminary Web Search Results
The following search results were gathered to help you understand the current landscape of this topic. Use them to inform your plan.

{search_context}

---
"""

        prompt = f"""Create a comprehensive research plan for the following query:

{query}
{search_block}
Create an exhaustive plan covering all important aspects of this topic. The goal is to produce a thorough, in-depth report. Favor deep coverage of each aspect over surface-level breadth."""

        try:
            pending_tasks = None

            # Try up to 2 times if the planner returns too few tasks
            for attempt in range(2):
                extra_instruction = ""
                if attempt > 0:
                    extra_instruction = (
                        f"\n\nIMPORTANT: Your previous plan only had "
                        f"{len(pending_tasks)} tasks, but this research "
                        f"requires at least {min_initial}. Generate MORE "
                        f"tasks with finer-grained subtopics.\n"
                    )

                response = self.client.complete(
                    prompt=prompt + extra_instruction,
                    system=system,
                    max_tokens=self.config.llm.max_tokens.planner,
                    temperature=self.config.llm.temperature.planner,
                    json_mode=True,
                    model=self.config.llm.models.planner
                )

                pending_tasks = self._parse_plan_json(response)

                if len(pending_tasks) >= min_initial:
                    break

                logger.warning(
                    f"Planner returned {len(pending_tasks)} tasks, "
                    f"need {min_initial}+. Retrying with stronger prompt..."
                )

            # Save to database
            tasks = self._save_plan_tasks(pending_tasks, session_id)

            logger.info(f"Created plan with {len(tasks)} tasks")
            return tasks

        except Exception as e:
            logger.error(f"Failed to create plan: {e}")
            raise

    def _pre_search(self, query: str) -> str:
        """Run preliminary web searches to give the planner real-world context.

        Executes 2 searches: one broad overview query and one more specific
        angle. Returns formatted context string with titles and snippets.
        """
        logger.info("Running pre-planning web searches...")

        queries = [
            query,
            f"{query} key topics overview",
        ]

        seen_urls = set()
        results = []

        for q in queries:
            print_search(f"[pre-plan] {q}")
            hits = web_search(q, max_results=5)
            for hit in hits:
                url = hit.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append(hit)

        if not results:
            logger.warning("Pre-planning search returned no results")
            return ""

        # Build compact context: title + snippet (no full content needed)
        parts = []
        for i, r in enumerate(results[:10], 1):
            title = r.get("title", "Untitled")
            snippet = r.get("snippet", "")
            url = r.get("url", "")
            parts.append(f"{i}. **{title}**\n   {snippet}\n   Source: {url}")

        context = "\n\n".join(parts)
        logger.info(f"Pre-planning search found {len(results)} results")
        return context
    
    def _parse_plan_json(self, response: str) -> List[dict]:
        """Parse the planner's JSON response into a list of task dicts (no DB writes)."""
        try:
            data = json.loads(response)

            # Handle various response formats
            if isinstance(data, list):
                task_list = data
            elif "tasks" in data:
                task_list = data["tasks"]
            elif "plan" in data:
                task_list = data["plan"]
            else:
                for value in data.values():
                    if isinstance(value, list):
                        task_list = value
                        break
                else:
                    raise ValueError("Could not find task list in response")

            max_tasks = self.config.research.max_total_tasks
            return task_list[:max_tasks]

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            logger.debug(f"Response was: {response[:500]}")
            raise

    def _save_plan_tasks(self, task_list: List[dict], session_id: int) -> List[ResearchTask]:
        """Apply priority rules, create ResearchTask objects, and insert into DB."""
        output_dir = self.config.output.directory

        # Structural keywords — only unambiguous terms to avoid misclassifying
        # content topics (e.g. "Protein Synthesis" should NOT match "synthesis")
        conclusion_keywords = ['conclusion', 'closing remarks', 'wrap-up', 'recap', 'final thoughts', 'key takeaways']
        future_keywords = ['future', 'outlook', 'prediction', 'forecast', 'next steps']
        intro_keywords = ['introduction', 'overview', 'background', 'foundation', 'basics', 'fundamentals']

        pending_tasks = []
        for i, item in enumerate(task_list):
            topic = item.get("topic", f"Task {i+1}")
            topic_lower = topic.lower()
            priority = item.get("priority", 5)

            # Enforce priority rules — use startswith for conclusion/future
            # to avoid false positives on content topics
            if any(topic_lower.startswith(kw) for kw in conclusion_keywords):
                priority = 1
            elif any(topic_lower.startswith(kw) for kw in future_keywords):
                priority = 2
            elif any(kw in topic_lower for kw in intro_keywords) and priority < 9:
                priority = max(priority, 9)

            pending_tasks.append(ResearchTask(
                topic=topic,
                description=item.get("description", ""),
                file_path=generate_file_path(topic, output_dir, i + 1),
                priority=priority,
                depth=0,
                status=TaskStatus.PENDING
            ))

        tasks = self.db.add_tasks_bulk(pending_tasks, session_id)
        self.db.update_session(session_id, total_tasks=len(tasks))
        return tasks


# =============================================================================
# RESEARCHER AGENT
# =============================================================================

class ResearcherAgent:
    """Agent responsible for deep research on individual topics"""

    def __init__(self):
        self.client = get_llm_client()
        self.db = get_database()

    @property
    def config(self):
        return get_config()
    
    def research_task(
        self,
        task: ResearchTask,
        overall_query: str = "",
        other_sections: List[str] = None,
        session_id: int = None
    ) -> Tuple[str, List[Dict], List[Dict]]:
        """
        Perform deep research on a single task
        Returns: (content, new_tasks, glossary_terms)
        """
        logger.info(f"Researching: {task.topic}")

        # Mark task as in progress
        self.db.update_task(task.id, status=TaskStatus.IN_PROGRESS)

        try:
            # Step 1: Generate search queries
            queries = self._generate_queries(task)

            # Step 2: Execute searches and gather content
            search_context = self._execute_searches(queries, task.id)

            # Handle empty search results
            if not search_context or not search_context.strip():
                logger.warning(f"No sources found for task: {task.topic}")
                search_context = (
                    "WARNING: No source material was found for this topic. "
                    "Write based on your training knowledge and clearly note "
                    "that sources were unavailable."
                )

            # Step 3: Synthesize and write
            content, new_tasks, glossary_terms = self._synthesize(
                task, search_context, overall_query, other_sections, session_id=session_id
            )

            return content, new_tasks, glossary_terms

        except Exception as e:
            logger.error(f"Research failed for task {task.id}: {e}")
            raise
    
    def _generate_queries(self, task: ResearchTask) -> List[str]:
        """Generate search queries for the task"""
        prompt = QUERY_GENERATOR_PROMPT.format(
            num_queries=self.config.search.queries_per_task,
            topic=task.topic,
            description=task.description
        )

        response = self.client.complete(
            prompt=prompt,
            system=QUERY_GENERATOR_SYSTEM,
            max_tokens=500,
            temperature=0.5,
            model=self.config.llm.models.researcher
        )
        
        # Parse queries (one per line)
        queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
        
        # Remove any numbering or bullets
        queries = [re.sub(r'^[\d\.\-\*\•]+\s*', '', q) for q in queries]
        
        return queries[:self.config.search.queries_per_task]
    
    def _execute_searches(self, queries: List[str], task_id: int) -> str:
        """Execute searches and aggregate results"""
        all_results = []
        seen_urls = set()
        
        for query in queries:
            print_search(query)
            results = web_search(query)
            
            for r in results:
                url = r.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)
        
        # Extract full content from top results
        context_parts = []
        sources_added = 0
        max_sources = self.config.search.max_results
        
        for result in all_results[:max_sources * 2]:  # Check more than needed
            url = result.get('url', '')
            if not url:
                continue
            
            try:
                print_scrape(url)
                source = extract_source_info(url, result)
                
                # Check quality threshold
                if source.quality_score < self.config.quality.min_source_quality:
                    logger.debug(f"Skipping low-quality source: {url}")
                    continue
                
                # Save source to database (position preserves citation order)
                self.db.add_source(source, task_id, position=sources_added)
                
                # Build context — use configured max_content_length
                content = source.full_content or source.snippet or ""
                max_len = self.config.scraping.max_content_length
                if content:
                    content_str = content[:max_len]
                    if len(content) > max_len:
                        content_str += "\n[... content truncated ...]"
                    context_parts.append(f"""
### Source: {source.title}
URL: {source.url}
Domain: {source.domain}
{'[Academic Source]' if source.is_academic else ''}

{content_str}
""")
                    sources_added += 1
                    
                    if sources_added >= max_sources:
                        break
                        
            except Exception as e:
                logger.warning(f"Failed to extract from {url}: {e}")
                continue
        
        return "\n\n---\n\n".join(context_parts)
    
    def _synthesize(
        self,
        task: ResearchTask,
        search_context: str,
        overall_query: str = "",
        other_sections: List[str] = None,
        session_id: int = None
    ) -> Tuple[str, List[Dict], List[Dict]]:
        """Synthesize research into written content"""

        # Truncate context if too long
        max_context_tokens = 50000  # Leave room for prompt and response
        search_context = truncate_to_tokens(search_context, max_context_tokens)

        system = RESEARCHER_SYSTEM_PROMPT.format(
            min_words=self.config.research.min_words_per_section,
            max_words=self.config.research.max_words_per_section,
            min_citations=self.config.research.min_citations_per_section
        )

        # Build other-sections context
        other_sections_text = ""
        if other_sections:
            other_sections_text = "\n## Other Sections in This Report (do not repeat their content):\n"
            other_sections_text += "\n".join(f"- {s}" for s in other_sections)
            other_sections_text += "\n"

        prompt = f"""Write a section for a research report on: **{overall_query}**

## This Section: {task.topic}

## Research Instructions:
{task.description}
{other_sections_text}
## Source Material:
{search_context}

---

Write this section assuming the reader will read the full report. Do not write a general
introduction or conclusion for this section — jump directly into the substance.
If you discover important sub-topics that need separate investigation, include them in the JSON block at the end."""
        
        response = self.client.complete(
            prompt=prompt,
            system=system,
            max_tokens=self.config.llm.max_tokens.writer,
            temperature=self.config.llm.temperature.writer,
            model=self.config.llm.models.writer
        )
        
        # Parse response for content, new tasks, and glossary
        content, new_tasks, glossary_terms = self._parse_research_response(response, task, session_id=session_id)

        return content, new_tasks, glossary_terms

    def _parse_research_response(
        self,
        response: str,
        task: ResearchTask,
        session_id: int = None
    ) -> Tuple[str, List[Dict], List[Dict]]:
        """Parse the researcher's response"""
        content = response
        new_tasks = []
        glossary_terms = []

        # Check recursion depth
        can_recurse = (
            self.config.research.enable_recursion and
            task.depth < self.config.research.max_recursion_depth
        )

        # Check total task limit
        total_tasks = self.db.get_task_count(session_id=session_id)
        at_task_limit = total_tasks >= self.config.research.max_total_tasks

        # Try to extract JSON metadata block
        data, json_start_pos = self._extract_json_metadata(response)

        if data is not None:
            # Extract new tasks if allowed - LIMIT TO 1 MAX
            if can_recurse and not at_task_limit:
                raw_tasks = data.get("new_tasks", [])
                for t in raw_tasks[:1]:
                    if isinstance(t, dict) and "topic" in t:
                        topic = t["topic"]
                        existing_tasks = self.db.get_all_tasks(session_id=session_id)
                        is_duplicate = any(
                            topic.lower() in et.topic.lower() or et.topic.lower() in topic.lower()
                            for et in existing_tasks
                        )

                        if not is_duplicate:
                            new_tasks.append({
                                "topic": topic,
                                "description": t.get("description", ""),
                                "priority": min(t.get("priority", 3), 4),
                                "parent_id": task.id,
                                "depth": task.depth + 1
                            })

            # Extract glossary terms
            raw_glossary = data.get("glossary_terms", [])
            for g in raw_glossary:
                if isinstance(g, dict) and "term" in g and "definition" in g:
                    glossary_terms.append({
                        "term": g["term"],
                        "definition": g["definition"]
                    })

            # Remove JSON block from content
            content = response[:json_start_pos].strip()

        return content, new_tasks, glossary_terms

    def _extract_json_metadata(self, response: str) -> Tuple[dict, int]:
        """Extract a JSON metadata block (new_tasks / glossary_terms) from response.

        Tries three strategies in order:
        1. Fenced ```json ... ``` block
        2. Fenced ``` ... ``` block whose content is JSON with expected keys
        3. Naked JSON found by key marker + brace-counting (handles trailing text)

        Returns (parsed_dict, start_position) or (None, -1).
        """
        _EXPECTED_KEYS = ('new_tasks', 'glossary_terms')

        # Method 1 & 2: fenced code blocks (```json or plain ```)
        # Iterate all fenced blocks; keep the last valid one (closest to end).
        json_data, json_pos = None, -1
        for fenced in re.finditer(r'```(?:json)?\s*\n?(.*?)\s*```', response, re.DOTALL):
            candidate = fenced.group(1).strip()
            if not candidate.startswith('{'):
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and any(k in parsed for k in _EXPECTED_KEYS):
                    json_data = parsed
                    json_pos = fenced.start()
            except json.JSONDecodeError:
                continue

        if json_data is not None:
            return json_data, json_pos

        # Method 3: naked JSON — search backwards for key marker, brace-count
        for key in ('"new_tasks"', '"glossary_terms"'):
            idx = response.rfind(key)
            if idx == -1:
                continue
            open_idx = response.rfind('{', 0, idx)
            if open_idx == -1:
                continue
            end_idx = self._find_matching_brace(response, open_idx)
            if end_idx is None:
                continue
            try:
                parsed = json.loads(response[open_idx:end_idx])
                if isinstance(parsed, dict) and any(k in parsed for k in _EXPECTED_KEYS):
                    return parsed, open_idx
            except json.JSONDecodeError:
                continue

        return None, -1

    @staticmethod
    def _find_matching_brace(text: str, start: int) -> int:
        """Return index after the closing } that matches text[start] == '{'.

        Returns None if no matching brace is found.
        """
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return i + 1
        return None


# =============================================================================
# DISCOVERY AGENT
# =============================================================================

class DiscoveryAgent:
    """Agent responsible for identifying research gaps after sections are written"""

    def __init__(self):
        self.client = get_llm_client()
        self.db = get_database()

    @property
    def config(self):
        return get_config()

    def discover_tasks(
        self,
        query: str,
        all_tasks: List[Dict[str, str]],
        section_summaries: List[Dict[str, str]],
        session_id: int = None
    ) -> List[Dict]:
        """
        Review completed work and identify gaps in coverage.

        Args:
            query: Original research query
            all_tasks: List of dicts with 'topic', 'status' keys
            section_summaries: List of dicts with 'topic', 'summary' keys
            session_id: Current session ID

        Returns:
            List of new task dicts with 'topic', 'description', 'priority' keys
        """
        logger.info("Running discovery agent for gap analysis...")

        # Build task list context
        tasks_text = "\n".join(
            f"- [{t['status']}] {t['topic']}" for t in all_tasks
        )

        # Build section summaries context
        summaries_text = "\n\n".join(
            f"### {s['topic']}\n{s['summary']}" for s in section_summaries
        )

        prompt = f"""Analyze the following research project for gaps in coverage.

## Original Research Query
{query}

## All Tasks (completed and pending)
{tasks_text}

## Summaries of Completed Sections
{summaries_text}

---

Identify up to {self.config.discovery.max_suggestions_per_run} important topics that are missing and would significantly improve the report's comprehensiveness. Only suggest topics that do NOT overlap with existing tasks."""

        try:
            response = self.client.complete(
                prompt=prompt,
                system=DISCOVERY_SYSTEM_PROMPT,
                max_tokens=self.config.llm.max_tokens.discovery,
                temperature=self.config.llm.temperature.discovery,
                json_mode=True,
                model=self.config.llm.models.discovery
            )

            return self._parse_discovery_response(response, all_tasks, session_id)

        except Exception as e:
            logger.warning(f"Discovery agent failed: {e}")
            return []

    def _parse_discovery_response(
        self,
        response: str,
        all_tasks: List[Dict[str, str]],
        session_id: int = None
    ) -> List[Dict]:
        """Parse discovery agent JSON response into task dicts."""
        try:
            data = json.loads(response)
            suggestions = data.get("suggestions", [])
        except json.JSONDecodeError:
            logger.warning("Discovery agent returned invalid JSON")
            return []

        # Check task limit
        total_tasks = self.db.get_task_count(session_id=session_id)
        remaining_capacity = self.config.research.max_total_tasks - total_tasks
        if remaining_capacity <= 0:
            return []

        existing_topics = [t['topic'].lower() for t in all_tasks]
        new_tasks = []

        for suggestion in suggestions[:self.config.discovery.max_suggestions_per_run]:
            if not isinstance(suggestion, dict) or "topic" not in suggestion:
                continue

            topic = suggestion["topic"]
            # Deduplication check
            is_duplicate = any(
                topic.lower() in et or et in topic.lower()
                for et in existing_topics
            )
            if is_duplicate:
                continue

            new_tasks.append({
                "topic": topic,
                "description": suggestion.get("description", ""),
                "priority": min(suggestion.get("priority", 5), 8),
                "depth": 1,
            })

            if len(new_tasks) >= remaining_capacity:
                break

        return new_tasks


# =============================================================================
# EDITOR AGENT
# =============================================================================

class EditorAgent:
    """Agent responsible for final compilation and editing"""

    def __init__(self):
        self.client = get_llm_client()
        self.db = get_database()

    @property
    def config(self):
        return get_config()
    
    def generate_executive_summary(self, query: str, section_summaries: List[Dict[str, str]]) -> str:
        """Generate an executive summary for the report.

        Args:
            query: The original research query
            section_summaries: List of dicts with 'topic' and 'summary' keys
        """
        logger.info("Generating executive summary...")

        sections_text = "\n\n".join(
            f"### {s['topic']}\n{s['summary']}" for s in section_summaries
        )

        prompt = f"""Write an executive summary for this research report.

Research Query: {query}

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
    
    def generate_conclusion(self, query: str, section_summaries: List[Dict[str, str]], word_count: int) -> str:
        """Generate a conclusion for the report.

        Args:
            query: The original research query
            section_summaries: List of dicts with 'topic' and 'summary' keys
            word_count: Total words in the report body
        """
        logger.info("Generating conclusion...")

        sections_text = "\n\n".join(
            f"### {s['topic']}\n{s['summary']}" for s in section_summaries
        )

        prompt = f"""Write a conclusion for this research report ({word_count:,} words across {len(section_summaries)} sections).

Research Query: {query}

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

    def determine_section_order(self, query: str, section_topics: List[str]) -> List[str]:
        """Determine the optimal reading order for report sections.

        Args:
            query: The original research query
            section_topics: List of section topic strings

        Returns:
            Ordered list of section topics
        """
        logger.info("Determining optimal section order...")

        topics_text = "\n".join(f"- {t}" for t in section_topics)

        prompt = f"""Determine the optimal reading order for these sections of a research report.

Research Query: {query}

Sections (in current arbitrary order):
{topics_text}

Reorder them for the best logical flow and reading experience."""

        try:
            response = self.client.complete(
                prompt=prompt,
                system=REORDER_SYSTEM_PROMPT,
                max_tokens=2000,
                temperature=0.1,
                json_mode=True,
                model=self.config.llm.models.editor
            )

            data = json.loads(response)
            ordered = data.get("order", [])

            # Validate: every original topic must appear exactly once
            original_set = set(section_topics)
            ordered_set = set(ordered)

            if ordered_set == original_set:
                return ordered

            # Fallback: use returned order for known topics, append missing ones
            logger.warning("Reorder response missing/extra topics; patching...")
            result = [t for t in ordered if t in original_set]
            for t in section_topics:
                if t not in ordered_set:
                    result.append(t)
            return result

        except Exception as e:
            logger.warning(f"Section reorder failed: {e}; keeping original order")
            return section_topics

    def rewrite_section(
        self,
        query: str,
        section_topic: str,
        section_content: str,
        toc: List[str],
        preceding_summaries: List[Dict[str, str]],
        following_topics: List[str]
    ) -> str:
        """Rewrite a single section for cohesion within the full report.

        Args:
            query: Original research query
            section_topic: This section's topic
            section_content: Original markdown content of this section
            toc: Full ordered table of contents (topic strings)
            preceding_summaries: Summaries of sections before this one
            following_topics: Topics of sections after this one

        Returns:
            Rewritten section content (markdown, no ## heading)
        """
        logger.info(f"Rewriting section: {section_topic}")

        toc_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(toc))

        preceding_text = ""
        if preceding_summaries:
            preceding_text = "\n\n".join(
                f"### {s['topic']}\n{s['summary']}" for s in preceding_summaries
            )

        following_text = ""
        if following_topics:
            following_text = "\n".join(f"- {t}" for t in following_topics)

        prompt = f"""Rewrite this section for cohesion within the full report.

## Research Query
{query}

## Full Table of Contents
{toc_text}

## Preceding Section Summaries
{preceding_text if preceding_text else "(This is the first section)"}

## Sections After This One
{following_text if following_text else "(This is the last section)"}

## Section to Rewrite: {section_topic}

{section_content}

---

Rewrite the above section content. Preserve all facts, data, and citations. Improve transitions and reduce repetition with preceding sections."""

        try:
            response = self.client.complete(
                prompt=prompt,
                system=REWRITE_SYSTEM_PROMPT,
                max_tokens=self.config.llm.max_tokens.editor,
                temperature=self.config.llm.temperature.editor,
                model=self.config.llm.models.editor
            )
            return response
        except Exception as e:
            logger.warning(f"Rewrite failed for '{section_topic}': {e}; keeping original")
            return section_content
