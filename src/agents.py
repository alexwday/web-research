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
