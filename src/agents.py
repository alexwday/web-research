"""
Agents Module for Deep Research Agent
Contains the Planner, Researcher, and Editor agents
"""
import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

from .config import get_config, ResearchTask, TaskStatus, Source
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
- NEVER use a citation number higher than the number of sources provided. If 3 sources are listed, only [1], [2], [3] are valid.
- If the source material section says "WARNING" or contains no actual sources, write WITHOUT any [N] citation markers at all. Do not fabricate references.

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


RESTRUCTURE_GROUPING_SYSTEM_PROMPT = """You are a Report Structure Architect. Your job is to organize a flat list of research sections into a hierarchical chapter structure with 2-7 primary topic groups.

You will receive:
- The original research query
- A list of section topics with brief summaries

Your task:
1. Identify 2-7 primary topic groups (chapters) that naturally cluster the sections
2. Assign every section to exactly one group
3. Give each group a clear, descriptive title
4. Give each section a concise subtopic title (may differ from the original topic to better fit the group context)

Rules:
- Every original section topic MUST appear exactly once in the output
- Groups should be ordered for optimal reading flow (general/background first, specific/advanced later)
- Subtopic titles should be concise (max 80 chars) and clearly distinguish sections within the same group
- If sections don't cluster naturally, it's fine to have a group with only one section

OUTPUT FORMAT:
Output ONLY a valid JSON object:
{{
  "groups": [
    {{
      "title": "Primary Topic Title",
      "sections": [
        {{"original_topic": "exact original section topic", "subtopic_title": "new concise title"}}
      ]
    }}
  ]
}}"""


RESTRUCTURE_REWRITE_SYSTEM_PROMPT = """You are a Report Editor performing a contextual rewrite of one section within a topic group. You are rewriting this section so it reads naturally as part of its group chapter.

You will receive:
- The original research query
- The full hierarchical report structure (all groups and their subtopics, so you understand the big picture)
- The group (chapter) title this section belongs to
- The subtopic title assigned to this section
- Summaries of the subtopics that PRECEDE this one in the group (so you can avoid repetition and build transitions)
- The topics of subtopics that FOLLOW this one in the group (so you can add forward references)
- A summary of all other groups in the report (for broader context)
- The original content of this section

Your task:
1. Rewrite this section so it reads naturally within its group:
   - Remove content that repeats what preceding subtopics already covered (reference them instead: "As discussed above in the subtopic on X...")
   - Add brief forward references where helpful ("This will be explored further in the subtopic on Y")
   - Avoid repeating information covered by siblings in the same group
   - Ensure terminology is consistent with the group context
   - Smooth the opening to transition naturally from the previous subtopic (or introduce the group theme if this is the first subtopic)
2. Preserve ALL substantive content, facts, data, and citations — do not remove information, only restructure and rephrase
3. Keep all [N] citation references exactly as they are — do not renumber or remove them
4. Maintain markdown heading structure (### and ####)
5. Keep approximately the same length — this is a contextual edit, not a summary

OUTPUT: The rewritten section content in Markdown format. Do NOT include any heading — only the body content."""


PRE_PLAN_ANALYSIS_SYSTEM_PROMPT = """You are a Research Analyst. Analyze the provided web page content and extract structured insights relevant to the research query.

For the given page, extract:
1. **Key Entities**: Specific names, products, organizations, frameworks, technologies, or people mentioned
2. **Subtopics Covered**: Main themes and subtopics discussed on this page
3. **Gaps/Missing Angles**: What important aspects of the research query does this page NOT address?
4. **Notable Claims**: Specific data points, statistics, quotes, or assertions worth investigating further
5. **Relevance Assessment**: How relevant is this page to the research query? (high/medium/low)

OUTPUT FORMAT:
Return ONLY a valid JSON object:
{{
  "entities": ["entity1", "entity2"],
  "subtopics": ["subtopic1", "subtopic2"],
  "gaps": ["gap1", "gap2"],
  "notable_claims": ["claim1", "claim2"],
  "relevance": "high"
}}"""


GAP_ANALYSIS_SYSTEM_PROMPT = """You are a Research Gap Analyst. Review the gathered source material against the task requirements and identify what critical information is still missing.

You will receive:
- The task topic and description
- The overall research query for context
- Summaries of the source material already gathered

Your job:
1. Assess whether the gathered sources adequately cover the task requirements
2. Identify specific gaps — missing perspectives, data, entities, or subtopics
3. Generate targeted search queries to fill those gaps

OUTPUT FORMAT:
Return ONLY a valid JSON object:
{{
  "has_gaps": true,
  "gap_summary": "Brief description of what's missing",
  "queries": ["targeted query 1", "targeted query 2"]
}}

If the gathered material is sufficient, return:
{{
  "has_gaps": false,
  "gap_summary": "",
  "queries": []
}}"""


QUERY_GENERATOR_SYSTEM = """You are a search query specialist. Generate maximally diverse search queries — each must target a different angle or source type."""


QUERY_GENERATOR_PROMPT = """Generate {num_queries} search queries for:

Overall Research: {overall_query}
Section Topic: {topic}
Focus: {description}

Each query MUST target a different angle (e.g., one broad overview, one specific/technical, one recent data or news). Vary terminology across queries. 3-8 words each. Queries should be relevant to both the section topic and the broader research context.

Output ONLY the queries, one per line, no numbering or bullets."""


QUERY_GENERATOR_JSON_PROMPT = """Generate {num_queries} search queries for:

Overall Research: {overall_query}
Section Topic: {topic}
Focus: {description}

Each query MUST target a different angle (e.g., broad overview, specific/technical angle, recent evidence/news angle).
Use concise search phrasing and varied terminology. Queries should be relevant to both the section topic and the broader research context.

Return ONLY a JSON object in this exact shape:
{{"queries": ["query 1", "query 2"]}}
"""

QUERY_GENERATOR_TOOL_NAME = "emit_search_queries"
QUERY_GENERATOR_TOOL_DESC = "Return diverse web search queries for the research task."


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
        search_context = self._pre_search(query, session_id=session_id)

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

## Pre-Planning Research Analysis
The following analysis was gathered from real web sources to help you understand the current landscape of this topic. Use the discovered entities, subtopics, and gaps to create highly specific, targeted research tasks.

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

    def _generate_planning_queries(self, query: str, num_queries: int = None) -> List[str]:
        """Use the LLM to produce diverse, concise search queries for pre-planning."""
        if num_queries is None:
            num_queries = self.config.search.pre_plan_queries
        prompt = (
            f"Generate {num_queries} diverse web search queries to gather background "
            f"information for planning research on the following topic:\n\n"
            f"{query}\n\n"
            f"Each query should be 3-8 words, target a different angle "
            f"(e.g. broad overview, key debates, recent developments), "
            f"and use concise search-engine phrasing.\n\n"
            f'Return ONLY a JSON object: {{"queries": ["query1", "query2"]}}'
        )
        try:
            tool_schema = {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": num_queries + 2,
                    }
                },
                "required": ["queries"],
                "additionalProperties": False,
            }
            result = self.client.complete_with_function(
                prompt=prompt,
                system=QUERY_GENERATOR_SYSTEM,
                function_name=QUERY_GENERATOR_TOOL_NAME,
                function_description=QUERY_GENERATOR_TOOL_DESC,
                function_parameters=tool_schema,
                max_tokens=self.config.llm.max_tokens.researcher,
                temperature=0.3,
                model=self.config.llm.models.researcher,
                require_tool_call=True,
            )
            if result and "queries" in result:
                queries = [q.strip() for q in result["queries"] if q.strip()][:num_queries]
                if queries:
                    logger.info(f"LLM generated {len(queries)} planning queries: {queries}")
                    return queries
        except Exception as e:
            logger.warning(f"LLM planning query generation failed, using fallback: {e}")

        # Fallback: use the raw query
        return [query]

    def _run_single_pre_search(self, q: str, session_id: int = None) -> List[dict]:
        """Execute a single pre-planning search query. Thread-safe."""
        qg = uuid.uuid4().hex[:12]
        print_search(f"[pre-plan] {q}")
        self.db.add_search_event(
            session_id=session_id, task_id=None,
            event_type="query", query_group=qg, query_text=q,
        )
        hits = web_search(q, max_results=self.config.search.pre_plan_max_results)
        for hit in hits:
            url = hit.get("url", "")
            if url:
                self.db.add_search_event(
                    session_id=session_id, task_id=None,
                    event_type="result", query_group=qg,
                    url=url,
                    title=hit.get("title", ""),
                    snippet=hit.get("snippet", ""),
                )
        return hits

    def _scrape_pre_plan_result(self, result: dict, session_id: int = None) -> Source:
        """Scrape a single search result and return a Source object. Thread-safe.

        Returns None if scraping fails or quality is too low.
        """
        url = result.get("url", "")
        if not url:
            return None
        try:
            print_scrape(url)
            source = extract_source_info(url, result)
            if source.quality_score < self.config.quality.min_source_quality:
                logger.info(f"[pre-plan] Skipping low-quality source: {url}")
                return None
            if not (source.full_content or source.snippet):
                return None
            return source
        except Exception as e:
            logger.warning(f"[pre-plan] Failed to scrape {url}: {e}")
            return None

    def _analyze_pre_plan_page(self, source: Source, query: str) -> dict:
        """Run LLM analysis on a scraped page, returning analysis dict. Thread-safe.

        Returns None if analysis fails.
        """
        content = source.full_content or source.snippet or ""
        max_len = self.config.scraping.max_content_length
        if len(content) > max_len:
            content = content[:max_len] + "\n[... content truncated ...]"

        prompt = (
            f"Research Query: {query}\n\n"
            f"Page Title: {source.title}\n"
            f"URL: {source.url}\n\n"
            f"Page Content:\n{content}"
        )
        try:
            response = self.client.complete(
                prompt=prompt,
                system=PRE_PLAN_ANALYSIS_SYSTEM_PROMPT,
                max_tokens=self.config.llm.max_tokens.analyzer,
                temperature=self.config.llm.temperature.analyzer,
                json_mode=True,
                model=self.config.llm.models.analyzer,
            )
            return json.loads(response)
        except Exception as e:
            logger.warning(f"[pre-plan] Analysis failed for {source.url}: {e}")
            return None

    def _pre_search(self, query: str, session_id: int = None) -> str:
        """Run preliminary web searches to give the planner real-world context.

        4-phase pipeline:
        1. Search: Run diverse queries in parallel -> deduplicated results
        2. Scrape: Scrape top results in parallel -> Source objects
        3. Analyze: LLM analysis on each scraped page -> rich entity/subtopic data
        4. Format: Build rich context string for the planner

        Falls back to snippet-only context if scraping or analysis fails.
        """
        logger.info("Running pre-planning web searches...")

        # Phase 1: Search
        queries = self._generate_planning_queries(query)
        seen_urls = set()
        results = []

        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            futures = [
                executor.submit(self._run_single_pre_search, q, session_id)
                for q in queries
            ]
            for future in as_completed(futures):
                try:
                    for hit in future.result():
                        url = hit.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            results.append(hit)
                except Exception as e:
                    logger.warning(f"Pre-planning search failed: {e}")

        if not results:
            logger.warning("Pre-planning search returned no results")
            return ""

        logger.info(f"Pre-planning search found {len(results)} unique results")

        # Phase 2: Scrape top results in parallel (cap at 30)
        scrape_targets = results[:30]
        sources = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(self._scrape_pre_plan_result, r, session_id)
                for r in scrape_targets
            ]
            for future in as_completed(futures):
                try:
                    source = future.result()
                    if source is not None:
                        sources.append(source)
                except Exception as e:
                    logger.warning(f"Pre-plan scrape error: {e}")

        logger.info(f"Pre-planning scraped {len(sources)} pages successfully")

        # Fallback to snippet-only if scraping produced nothing
        if not sources:
            logger.warning("Pre-planning scraping failed; falling back to snippets")
            return self._format_snippet_context(results)

        # Phase 3: Analyze each scraped page in parallel
        analyses = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._analyze_pre_plan_page, src, query): src
                for src in sources
            }
            for future in as_completed(futures):
                src = futures[future]
                try:
                    analysis = future.result()
                    if analysis is not None:
                        analyses.append((src, analysis))
                except Exception as e:
                    logger.warning(f"Pre-plan analysis error: {e}")

        logger.info(f"Pre-planning analyzed {len(analyses)} pages")

        # Fallback to content previews if analysis failed
        if not analyses:
            logger.warning("Pre-planning analysis failed; falling back to content previews")
            return self._format_content_preview_context(sources)

        # Phase 4: Format rich context
        return self._format_analysis_context(analyses)

    def _format_snippet_context(self, results: list) -> str:
        """Format search results as snippet-only context (fallback)."""
        parts = []
        for i, r in enumerate(results[:10], 1):
            title = r.get("title", "Untitled")
            snippet = r.get("snippet", "")
            url = r.get("url", "")
            parts.append(f"{i}. **{title}**\n   {snippet}\n   Source: {url}")
        return "\n\n".join(parts)

    def _format_content_preview_context(self, sources: list) -> str:
        """Format scraped sources as content previews (fallback)."""
        parts = []
        for i, src in enumerate(sources[:10], 1):
            content = src.full_content or src.snippet or ""
            preview = content[:1500]
            if len(content) > 1500:
                preview += "..."
            parts.append(
                f"{i}. **{src.title}**\n"
                f"   URL: {src.url}\n"
                f"   Content Preview: {preview}"
            )
        return "\n\n".join(parts)

    def _format_analysis_context(self, analyses: list) -> str:
        """Format LLM analyses into rich context for the planner."""
        # Collect all entities and subtopics across pages for a summary
        all_entities = set()
        all_subtopics = set()
        all_gaps = set()

        parts = []
        for i, (src, analysis) in enumerate(analyses[:15], 1):
            entities = analysis.get("entities", [])
            subtopics = analysis.get("subtopics", [])
            gaps = analysis.get("gaps", [])
            claims = analysis.get("notable_claims", [])
            relevance = analysis.get("relevance", "medium")

            all_entities.update(entities)
            all_subtopics.update(subtopics)
            all_gaps.update(gaps)

            part = f"### Source {i}: {src.title}\n"
            part += f"URL: {src.url}\n"
            part += f"Relevance: {relevance}\n"
            if entities:
                part += f"Key Entities: {', '.join(entities[:10])}\n"
            if subtopics:
                part += f"Subtopics: {', '.join(subtopics[:8])}\n"
            if claims:
                part += f"Notable Claims:\n"
                for claim in claims[:5]:
                    part += f"  - {claim}\n"
            if gaps:
                part += f"Gaps: {', '.join(gaps[:5])}\n"

            parts.append(part)

        # Add a summary section at the top
        summary = "### Cross-Source Summary\n"
        if all_entities:
            summary += f"**All Entities Discovered**: {', '.join(sorted(all_entities)[:30])}\n"
        if all_subtopics:
            summary += f"**All Subtopics Found**: {', '.join(sorted(all_subtopics)[:20])}\n"
        if all_gaps:
            summary += f"**Identified Gaps**: {', '.join(sorted(all_gaps)[:15])}\n"
        summary += "\n---\n"

        return summary + "\n\n".join(parts)
    
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
            queries = self._generate_queries(task, overall_query=overall_query)

            # Step 2: Execute searches and gather content
            search_context, initial_source_count = self._execute_searches(queries, task.id, session_id=session_id)

            # Handle empty search results
            has_sources = bool(search_context and search_context.strip())
            if not has_sources:
                logger.warning(f"No sources found for task: {task.topic}")
                search_context = (
                    "WARNING: No source material was found for this topic. "
                    "Write based on your training knowledge. "
                    "Do NOT include any [N] citation markers."
                )

            # Step 3 (gap-fill): If we have sources and gap-fill is enabled,
            # check for missing information and run targeted follow-up searches
            gap_fill_queries_count = self.config.search.gap_fill_queries
            if has_sources and gap_fill_queries_count > 0:
                try:
                    gap_queries = self._identify_gaps(task, search_context, overall_query)
                    if gap_queries:
                        # Collect URLs already seen so gap-fill doesn't re-scrape them
                        existing_sources = self.db.get_sources_for_task(task.id)
                        existing_urls = {s.url for s in existing_sources}
                        gap_context = self._execute_gap_fill_searches(
                            gap_queries, task.id, existing_urls, session_id,
                            source_number_offset=initial_source_count
                        )
                        if gap_context:
                            search_context += (
                                "\n\n---\n\n## Additional Sources (Gap-Fill)\n\n"
                                + gap_context
                            )
                except Exception as e:
                    logger.warning(f"Gap-fill failed for task {task.id}: {e}")

            # Step 4: Synthesize and write
            content, new_tasks, glossary_terms = self._synthesize(
                task, search_context, overall_query, other_sections, session_id=session_id
            )

            # Safety net: strip phantom citations if task has no real sources
            if not has_sources:
                stripped = re.sub(r'(?<!\])\[(\d+)\](?!\()', '', content)
                if stripped != content:
                    logger.warning(f"Stripped phantom citations from sourceless task {task.id}")
                    content = stripped

            return content, new_tasks, glossary_terms

        except Exception as e:
            logger.error(f"Research failed for task {task.id}: {e}")
            raise
    
    def _generate_queries(self, task: ResearchTask, overall_query: str = "") -> List[str]:
        """Generate search queries for the task.

        Retries once on empty LLM response, then falls back to a
        synthetic query derived from the task topic and description.
        """
        num_queries = max(1, self.config.search.queries_per_task)
        logger.info(f"Generating {num_queries} queries for: {task.topic[:60]}")

        tool_schema = {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": max(num_queries, 8),
                }
            },
            "required": ["queries"],
            "additionalProperties": False,
        }

        # Attempt 0: native tool/function call for structured output.
        try:
            tool_payload = self.client.complete_with_function(
                prompt=QUERY_GENERATOR_JSON_PROMPT.format(
                    num_queries=num_queries,
                    overall_query=overall_query or task.topic,
                    topic=task.topic,
                    description=task.description
                ),
                system=QUERY_GENERATOR_SYSTEM,
                function_name=QUERY_GENERATOR_TOOL_NAME,
                function_description=QUERY_GENERATOR_TOOL_DESC,
                function_parameters=tool_schema,
                max_tokens=self.config.llm.max_tokens.researcher,
                temperature=0.3,
                model=self.config.llm.models.researcher,
                require_tool_call=True,
            )
            if tool_payload:
                logger.info(f"Query generator tool payload: {str(tool_payload)[:200]!r}")
                result = self._parse_query_response(json.dumps(tool_payload), num_queries)
                if result:
                    if len(result) < num_queries:
                        for q in self._build_fallback_queries(task, num_queries):
                            if q.lower() not in {x.lower() for x in result}:
                                result.append(q)
                            if len(result) >= num_queries:
                                break
                    logger.info(f"Parsed {len(result)} queries (tool): {result}")
                    return result
        except Exception as e:
            logger.warning(f"Tool-based query generation failed; falling back: {e}")

        json_prompt = QUERY_GENERATOR_JSON_PROMPT.format(
            num_queries=num_queries,
            overall_query=overall_query or task.topic,
            topic=task.topic,
            description=task.description
        )

        base_prompt = QUERY_GENERATOR_PROMPT.format(
            num_queries=num_queries,
            overall_query=overall_query or task.topic,
            topic=task.topic,
            description=task.description
        )

        attempts = [
            {
                "prompt": json_prompt,
                "json_mode": True,
                "label": "json",
            },
            {
                "prompt": (
                    base_prompt
                    + "\n\nIMPORTANT: Return at least one concrete search query. "
                    + "Do not return blank output."
                ),
                "json_mode": False,
                "label": "text",
            },
        ]

        for attempt_idx, attempt in enumerate(attempts, start=1):
            response = self.client.complete(
                prompt=attempt["prompt"],
                system=QUERY_GENERATOR_SYSTEM,
                max_tokens=self.config.llm.max_tokens.researcher,
                temperature=0.5 + ((attempt_idx - 1) * 0.2),
                json_mode=attempt["json_mode"],
                model=self.config.llm.models.researcher
            )

            response_text = (response or "").strip()
            logger.info(
                f"Query generator raw response (attempt {attempt_idx}, {attempt['label']}): "
                f"{response_text[:200]!r}"
            )

            result = self._parse_query_response(response_text, num_queries)
            if result:
                if len(result) < num_queries:
                    # Fill short outputs deterministically so downstream stages
                    # always run with the configured query count.
                    for q in self._build_fallback_queries(task, num_queries):
                        if q.lower() not in {x.lower() for x in result}:
                            result.append(q)
                        if len(result) >= num_queries:
                            break
                logger.info(f"Parsed {len(result)} queries: {result}")
                return result

            logger.warning(f"Query generator returned empty (attempt {attempt_idx})")

        fallback_queries = self._build_fallback_queries(task, num_queries)
        logger.warning(f"Using fallback queries: {fallback_queries}")
        return fallback_queries

    def _parse_query_response(self, response: str, num_queries: int) -> List[str]:
        """Parse query-generation output from plain text, fenced blocks, or JSON."""
        if not response:
            return []

        text = response.strip()

        # Accept fenced output and extract inner content.
        fenced = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()

        candidates: List[str] = []

        # Try JSON first.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                candidates = [str(item) for item in parsed]
            elif isinstance(parsed, dict):
                val = (
                    parsed.get("queries")
                    or parsed.get("search_queries")
                    or parsed.get("query")
                )
                if isinstance(val, list):
                    candidates = [str(item) for item in val]
                elif isinstance(val, str):
                    candidates = [val]
        except json.JSONDecodeError:
            pass

        # Fallback to line parsing.
        if not candidates:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if len(lines) == 1 and num_queries > 1 and any(sep in lines[0] for sep in (";", "|")):
                lines = [part.strip() for part in re.split(r"[;|]", lines[0]) if part.strip()]
            candidates = lines

        cleaned: List[str] = []
        seen = set()

        for candidate in candidates:
            q = candidate.strip().strip("`").strip()
            q = re.sub(r'^[\d\)\.\-\*\•]+\s*', '', q)
            q = re.sub(r'^queries?\s*:\s*', '', q, flags=re.IGNORECASE)
            q = q.strip().strip('"').strip("'")
            q = re.sub(r'\s+', ' ', q)
            if not q:
                continue

            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(q)

            if len(cleaned) >= num_queries:
                break

        return cleaned

    def _build_fallback_queries(self, task: ResearchTask, num_queries: int) -> List[str]:
        """Build deterministic fallback queries when the model returns no usable output."""
        topic = " ".join(task.topic.split())
        desc = " ".join(task.description.split())
        desc_short = " ".join(desc.split()[:12])

        candidates = [
            topic,
            f"{topic} overview",
            f"{topic} key themes analysis",
            f"{topic} recent evidence studies",
            f"{topic} case studies",
        ]
        if desc_short:
            candidates.insert(1, f"{topic} {desc_short}")

        fallbacks: List[str] = []
        seen = set()
        for candidate in candidates:
            query = " ".join(candidate.split()[:12]).strip()
            if not query:
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            fallbacks.append(query)
            if len(fallbacks) >= num_queries:
                break

        if not fallbacks:
            fallbacks = [topic[:120].strip() or "research topic overview"]

        return fallbacks
    
    def _search_single_query(self, query: str, task_id: int, session_id: int = None) -> List[dict]:
        """Execute a single search query and log events. Thread-safe."""
        qg = uuid.uuid4().hex[:12]
        print_search(query)
        logger.info(f"Searching: {query}")
        self.db.add_search_event(
            session_id=session_id, task_id=task_id,
            event_type="query", query_group=qg, query_text=query,
        )
        results = web_search(query)
        logger.info(f"Search returned {len(results)} results")

        for r in results:
            url = r.get('url', '')
            if url:
                self.db.add_search_event(
                    session_id=session_id, task_id=task_id,
                    event_type="result", query_group=qg,
                    url=url,
                    title=r.get('title', ''),
                    snippet=r.get('snippet', ''),
                )
        return results

    def _execute_searches(self, queries: List[str], task_id: int, session_id: int = None) -> Tuple[str, int]:
        """Execute searches in parallel and aggregate results.

        Returns (context_string, source_count).
        """
        logger.info(f"Executing searches with {len(queries)} queries")

        # Phase 1: Run all search queries in parallel
        all_results = []
        seen_urls = set()

        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            futures = [
                executor.submit(self._search_single_query, q, task_id, session_id)
                for q in queries
            ]
            for future in as_completed(futures):
                try:
                    for r in future.result():
                        url = r.get('url', '')
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append(r)
                except Exception as e:
                    logger.warning(f"Search query failed: {e}")

        logger.info(f"Total unique results: {len(all_results)}")

        # Phase 2: Extract full content from top results
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
                logger.info(f"Source: {url[:60]} quality={source.quality_score} content_len={len(source.full_content or '')}")

                # Check quality threshold
                if source.quality_score < self.config.quality.min_source_quality:
                    logger.info(f"Skipping low-quality source: {url}")
                    continue

                # Build context — use configured max_content_length
                content = source.full_content or source.snippet or ""
                max_len = self.config.scraping.max_content_length
                if content:
                    # Save source to database only when it will appear in the prompt
                    # (position must match prompt order for citation correctness)
                    self.db.add_source(source, task_id, position=sources_added)

                    sources_added += 1
                    content_str = content[:max_len]
                    if len(content) > max_len:
                        content_str += "\n[... content truncated ...]"
                    context_parts.append(
                        f"\n### Source {sources_added}: {source.title}\n"
                        f"URL: {source.url}\n"
                        f"Domain: {source.domain}\n"
                        f"{'[Academic Source]' if source.is_academic else ''}\n\n"
                        f"{content_str}\n"
                    )

                    if sources_added >= max_sources:
                        break

            except Exception as e:
                logger.warning(f"Failed to extract from {url}: {e}")
                continue

        return "\n\n---\n\n".join(context_parts), sources_added

    def _identify_gaps(
        self, task: ResearchTask, search_context: str, overall_query: str
    ) -> List[str]:
        """Analyze gathered sources vs task requirements and return gap-fill queries.

        Returns a list of targeted search queries (possibly empty) to fill gaps.
        """
        max_gap_queries = self.config.search.gap_fill_queries
        if max_gap_queries <= 0:
            return []

        # Build a condensed summary of what we have (truncate to keep prompt small)
        context_preview = search_context[:8000]
        if len(search_context) > 8000:
            context_preview += "\n[... additional sources omitted for brevity ...]"

        prompt = (
            f"Overall Research Query: {overall_query}\n\n"
            f"Task Topic: {task.topic}\n"
            f"Task Description: {task.description}\n\n"
            f"Gathered Source Material (summary):\n{context_preview}\n\n"
            f"Maximum gap-fill queries to suggest: {max_gap_queries}"
        )

        try:
            response = self.client.complete(
                prompt=prompt,
                system=GAP_ANALYSIS_SYSTEM_PROMPT,
                max_tokens=2000,
                temperature=0.2,
                json_mode=True,
                model=self.config.llm.models.researcher,
            )
            data = json.loads(response)
            if not data.get("has_gaps", False):
                logger.info(f"Gap analysis: no gaps found for task {task.id}")
                return []

            queries = [q.strip() for q in data.get("queries", []) if q.strip()]
            queries = queries[:max_gap_queries]
            if queries:
                logger.info(
                    f"Gap analysis found {len(queries)} gap-fill queries for task {task.id}: {queries}"
                )
            return queries
        except Exception as e:
            logger.warning(f"Gap analysis LLM call failed for task {task.id}: {e}")
            return []

    def _execute_gap_fill_searches(
        self,
        queries: List[str],
        task_id: int,
        existing_urls: set,
        session_id: int = None,
        source_number_offset: int = 0,
    ) -> str:
        """Execute gap-fill search queries, scrape new results, and return context.

        Skips URLs already seen in the initial search. Saves sources with position
        offset of 100 to keep gap-fill citations after initial sources.
        """
        max_results = self.config.search.gap_fill_max_results
        if max_results <= 0:
            return ""

        # Run all gap-fill searches in parallel
        all_results = []
        seen_urls = set(existing_urls)

        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            futures = [
                executor.submit(self._search_single_query, q, task_id, session_id)
                for q in queries
            ]
            for future in as_completed(futures):
                try:
                    for r in future.result():
                        url = r.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append(r)
                except Exception as e:
                    logger.warning(f"Gap-fill search failed: {e}")

        if not all_results:
            logger.info(f"Gap-fill searches returned no new results for task {task_id}")
            return ""

        logger.info(f"Gap-fill found {len(all_results)} new results for task {task_id}")

        # Scrape and build context
        context_parts = []
        sources_added = 0

        for result in all_results[: max_results * 2]:
            url = result.get("url", "")
            if not url:
                continue
            try:
                print_scrape(url)
                source = extract_source_info(url, result)
                if source.quality_score < self.config.quality.min_source_quality:
                    continue

                content = source.full_content or source.snippet or ""
                max_len = self.config.scraping.max_content_length
                if content:
                    # Save source only when it will appear in the prompt
                    # Position offset 100+ so gap-fill citations sort after initial sources
                    self.db.add_source(source, task_id, position=100 + sources_added)

                    sources_added += 1
                    source_num = source_number_offset + sources_added
                    content_str = content[:max_len]
                    if len(content) > max_len:
                        content_str += "\n[... content truncated ...]"
                    context_parts.append(
                        f"### Source {source_num}: {source.title}\n"
                        f"URL: {source.url}\n"
                        f"Domain: {source.domain}\n"
                        f"{'[Academic Source]' if source.is_academic else ''}\n\n"
                        f"{content_str}"
                    )
                    if sources_added >= max_results:
                        break
            except Exception as e:
                logger.warning(f"Gap-fill scrape failed for {url}: {e}")
                continue

        logger.info(f"Gap-fill added {sources_added} new sources for task {task_id}")
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

        # Safety net: strip any remaining trailing JSON block that the
        # structured extraction missed (malformed JSON, unexpected format, etc.)
        content = self._strip_trailing_json(content)

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

        # Method 1 & 2: fenced code blocks (```json/```JSON or plain ```)
        # Iterate all fenced blocks; keep the last valid one (closest to end).
        json_data, json_pos = None, -1
        for fenced in re.finditer(r'```(?:json)?\s*\n?(.*?)\s*```', response, re.DOTALL | re.IGNORECASE):
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

    @staticmethod
    def _strip_trailing_json(content: str) -> str:
        """Remove trailing JSON blocks (fenced or naked) from section content.

        Safety net for when _extract_json_metadata fails to parse the block
        (malformed JSON, unexpected keys, uppercase ```JSON tag, etc.).
        """
        text = content.rstrip()

        # 1. Trailing fenced code block containing a JSON object
        m = re.search(r'\n```[a-zA-Z]*\s*\n[\s\S]*?\n```\s*$', text)
        if m and '{' in m.group():
            return text[:m.start()].rstrip()

        # 2. Trailing naked JSON object — scan backwards for lines starting
        #    with '{' and try to parse from there to end-of-string.
        lines = text.split('\n')
        for i in range(len(lines) - 1, max(len(lines) - 30, -1), -1):
            if lines[i].strip().startswith('{'):
                candidate = '\n'.join(lines[i:]).strip()
                try:
                    json.loads(candidate)
                    return '\n'.join(lines[:i]).rstrip()
                except json.JSONDecodeError:
                    continue  # inner brace; keep scanning for outer {

        return content


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

        # Build task list context (include descriptions for overlap detection)
        tasks_text = "\n".join(
            f"- [{t['status']}] {t['topic']}: {t.get('description', '')}" for t in all_tasks
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
    
    def generate_executive_summary(
        self,
        query: str,
        section_summaries: List[Dict[str, str]],
        report_structure: str = ""
    ) -> str:
        """Generate an executive summary for the report.

        Args:
            query: The original research query
            section_summaries: List of dicts with 'topic' and 'summary' keys
            report_structure: Optional hierarchical structure description (groups/chapters)
        """
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
        """Generate a conclusion for the report.

        Args:
            query: The original research query
            section_summaries: List of dicts with 'topic' and 'summary' keys
            word_count: Total words in the report body
            report_structure: Optional hierarchical structure description (groups/chapters)
        """
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

    def determine_section_order(
        self,
        query: str,
        section_topics: List[str],
        section_summaries: List[Dict[str, str]] = None
    ) -> List[str]:
        """Determine the optimal reading order for report sections.

        Args:
            query: The original research query
            section_topics: List of section topic strings
            section_summaries: Optional list of dicts with 'topic' and 'summary' keys

        Returns:
            Ordered list of section topics
        """
        logger.info("Determining optimal section order...")

        if section_summaries:
            topics_text = "\n".join(
                f"- **{s['topic']}**: {s['summary'][:200]}" for s in section_summaries
            )
        else:
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
                max_tokens=self.config.llm.max_tokens.editor,
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

    def group_sections_into_topics(
        self,
        query: str,
        section_summaries: List[Dict[str, str]]
    ) -> List[Dict]:
        """Group sections into primary topic clusters.

        Args:
            query: Original research query
            section_summaries: List of dicts with 'topic' and 'summary' keys

        Returns:
            List of group dicts: [{"title": str, "sections": [{"original_topic": str, "subtopic_title": str}]}]
        """
        logger.info("Grouping sections into topic clusters...")

        sections_text = "\n".join(
            f"- **{s['topic']}**: {s['summary']}" for s in section_summaries
        )

        prompt = f"""Organize the following research sections into 2-7 primary topic groups.

## Research Query
{query}

## Sections
{sections_text}

Group these sections into logical chapters and assign subtopic titles."""

        try:
            response = self.client.complete(
                prompt=prompt,
                system=RESTRUCTURE_GROUPING_SYSTEM_PROMPT,
                max_tokens=self.config.llm.max_tokens.editor,
                temperature=0.1,
                json_mode=True,
                model=self.config.llm.models.editor
            )

            data = json.loads(response)
            groups = data.get("groups", [])

            if not groups:
                raise ValueError("Empty groups returned")

            # Validate all original topics are present
            assigned_topics = set()
            for group in groups:
                for section in group.get("sections", []):
                    assigned_topics.add(section.get("original_topic", ""))

            original_topics = {s["topic"] for s in section_summaries}
            missing = original_topics - assigned_topics

            if missing:
                logger.warning(f"Grouping missed {len(missing)} sections; appending to last group")
                last_group = groups[-1]
                for topic in missing:
                    last_group["sections"].append({
                        "original_topic": topic,
                        "subtopic_title": topic,
                    })

            return groups

        except Exception as e:
            logger.warning(f"Section grouping failed: {e}; using flat fallback")
            # Fallback: one group per section
            return [
                {
                    "title": s["topic"],
                    "sections": [{"original_topic": s["topic"], "subtopic_title": s["topic"]}],
                }
                for s in section_summaries
            ]

    def rewrite_section_in_group(
        self,
        query: str,
        group_title: str,
        subtopic_title: str,
        section_content: str,
        full_toc: str,
        preceding_siblings: List[Dict[str, str]],
        following_sibling_titles: List[str],
        other_groups_summary: str
    ) -> str:
        """Rewrite a section within its topic-group context.

        Args:
            query: Original research query
            group_title: The primary topic (chapter) this section belongs to
            subtopic_title: The assigned subtopic title
            section_content: Original markdown content
            full_toc: Full hierarchical TOC string (all groups + subtopics)
            preceding_siblings: List of {"subtopic_title": str, "content_or_summary": str}
                for subtopics before this one in the group
            following_sibling_titles: List of subtopic title strings after this one
            other_groups_summary: Text listing all other groups and subtopics

        Returns:
            Rewritten section content (markdown, no heading)
        """
        logger.info(f"Rewriting section in group context: {subtopic_title}")

        preceding_text = ""
        if preceding_siblings:
            parts = []
            for sib in preceding_siblings:
                parts.append(f"### {sib['subtopic_title']}\n{sib['content_or_summary']}")
            preceding_text = "\n\n".join(parts)

        following_text = ""
        if following_sibling_titles:
            following_text = "\n".join(f"- {t}" for t in following_sibling_titles)

        prompt = f"""Rewrite this section within its topic-group context.

## Research Query
{query}

## Full Report Structure
{full_toc}

## Current Group (Chapter): {group_title}

## This Section: {subtopic_title}

## Preceding Subtopics in This Group
{preceding_text if preceding_text else "(This is the first subtopic in the group)"}

## Following Subtopics in This Group
{following_text if following_text else "(This is the last subtopic in the group)"}

## Other Groups in the Report
{other_groups_summary if other_groups_summary else "(No other groups)"}

## Section Content to Rewrite

{section_content}

---

Rewrite the above section content. Preserve all facts, data, and citations. Improve coherence within the group context."""

        try:
            response = self.client.complete(
                prompt=prompt,
                system=RESTRUCTURE_REWRITE_SYSTEM_PROMPT,
                max_tokens=self.config.llm.max_tokens.editor,
                temperature=self.config.llm.temperature.editor,
                model=self.config.llm.models.editor
            )
            return response
        except Exception as e:
            logger.warning(f"Restructure rewrite failed for '{subtopic_title}': {e}; keeping original")
            return section_content
