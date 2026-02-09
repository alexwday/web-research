"""
Agents Module for Deep Research Agent
Contains the Planner, Researcher, and Editor agents
"""
import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

from .config import get_config, ResearchTask, TaskStatus, SectionStatus, ReportSection, Source
from .llm_client import get_llm_client
from .tools import (
    web_search, extract_source_info, is_blocked_source,
    truncate_to_tokens, generate_file_path, read_file
)
from .database import get_database
from .logger import get_logger, print_search, print_scrape

logger = get_logger(__name__)


# =============================================================================
# PROMPTS
# =============================================================================

PLANNER_SYSTEM_PROMPT = """You are a Research Architect specializing in deep pre-planning analysis for comprehensive research projects.

Your task is to analyze a research query and the provided web search results to build a thorough understanding of the topic landscape. You are NOT creating research tasks — you are building a rich analytical foundation that will be used to design a report outline.

Use the preliminary web search results to:
- Identify the key subtopics, themes, and angles that actually exist in the literature
- Discover terminology, frameworks, or debates you might not have known about
- Map the landscape of what is actually being discussed about this topic
- Identify areas of consensus, controversy, and uncertainty
- Note important entities, organizations, products, or frameworks

OUTPUT FORMAT:
Provide a structured analysis as a JSON object:
{{
  "topic_landscape": "2-3 paragraph overview of what you learned about this topic from the sources",
  "key_themes": ["theme 1", "theme 2", ...],
  "key_entities": ["entity 1", "entity 2", ...],
  "controversies": ["controversy 1", ...],
  "knowledge_gaps": ["gap 1", "gap 2", ...],
  "recommended_scope": "1-2 sentences on what a comprehensive report should cover"
}}"""


RESEARCH_NOTES_SYSTEM_PROMPT = """You are a Deep Research Specialist gathering findings for a research report section.

Your job is to produce STRUCTURED RESEARCH NOTES — not polished prose. These notes will later be synthesized into a cohesive section by another agent. Focus on extracting and organizing information from your sources.

OUTPUT STRUCTURE:
Organize your findings under these headings:

### Key Findings
- Bullet points of the most important facts, claims, and insights, each with a citation [N]
- Be specific: include numbers, dates, names, percentages

### Data & Statistics
- Any quantitative data, figures, metrics, or measurements found in sources
- Include the context for each data point [N]

### Notable Quotes
- Direct quotes that are particularly insightful or authoritative
- "Quoted text here" [N]

### Conflicting Viewpoints
- Areas where sources disagree or present different perspectives
- Note which sources support which position [N]

### Gaps / Follow-up Needed
- Information that was NOT found but would be valuable
- Questions raised by the research that remain unanswered

CITATION FORMAT:
- Sources are numbered in the order listed under "Source Material" below
- Cite using numbered references: [1], [2], etc.
- Source 1 = first source, Source 2 = second source, and so on
- You MUST cite from the provided sources. Do not invent citations.
- NEVER use a citation number higher than the number of sources provided.
- If the source material section says "WARNING" or contains no actual sources, write WITHOUT any [N] citation markers. Do not fabricate references.

NOTE: Source content may be truncated. Do not assume you have the complete text of any source.

ABOUT NEW TASKS:
- Most research tasks should NOT spawn new tasks — only do so if something critical was discovered that cannot be covered here
- Never suggest more than 1 new task

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


OUTLINE_DESIGNER_SYSTEM_PROMPT = """You are a Report Architect. Given a research query and a deep pre-planning analysis, design the sections of a comprehensive research report.

Each section should represent a coherent chapter that covers a specific aspect of the topic. The sections will later have focused research tasks created for each one.

GUIDELINES:
1. Design approximately {target_sections} sections (no more than {max_sections})
2. Order sections for optimal reading flow (background first, then core content, then advanced/future topics)
3. Each section needs a clear, specific title and a description of what it should cover
4. Do NOT include "Executive Summary" or "Conclusion" — these are generated separately
5. Sections should be non-overlapping and collectively exhaustive
6. Ground your outline in the pre-planning analysis — reflect what was actually discovered

OUTPUT FORMAT:
Output ONLY a valid JSON object:
{{
  "sections": [
    {{
      "title": "Section Title",
      "description": "2-3 sentences describing what this section should cover, including specific angles and subtopics",
      "position": 1
    }}
  ]
}}"""


SECTION_TASK_PLANNER_SYSTEM_PROMPT = """You are a Research Task Planner. Given one section of a report outline and the full outline for context, generate focused research tasks for that section.

Each task is an investigation unit — it should focus on gathering specific information, not on writing. The research notes from these tasks will later be synthesized into the section's prose.

GUIDELINES:
1. Generate {tasks_per_section} tasks for this section
2. Each task should target a different angle, subtopic, or source type within the section
3. Tasks should be specific enough to execute with 1-3 search queries each
4. Avoid overlap between tasks — each should contribute unique information
5. Consider the full report outline to avoid duplicating work done in other sections

OUTPUT FORMAT:
Output ONLY a valid JSON object:
{{
  "tasks": [
    {{
      "topic": "Brief, specific research focus (max 100 chars)",
      "description": "2-3 sentences explaining exactly what to investigate and what kind of information to find",
      "priority": 5
    }}
  ]
}}"""


PIPELINE_GAP_ANALYSIS_SYSTEM_PROMPT = """You are a Research Gap Analyst performing a comprehensive post-research review. Your job is to identify what's missing from the research at TWO levels:

1. **Per-section gaps**: For existing sections, identify specific data, perspectives, or evidence that the research tasks did not adequately cover.
2. **Cross-section gaps**: Identify entirely new sections that should be added to the report.

You will receive:
- The original research query
- The report outline (all sections with descriptions)
- Summaries of all research notes gathered per section

ANALYSIS GUIDELINES:
- Compare gathered material against each section's description
- Look for sections where research is thin (few findings, few citations)
- Identify perspectives or angles that no section covers
- Consider whether the report would leave important questions unanswered
- Be selective: only suggest gaps that would SIGNIFICANTLY improve the report

OUTPUT FORMAT:
Output ONLY a valid JSON object:
{{
  "section_gaps": [
    {{
      "section_title": "exact title of existing section",
      "gap_description": "what is missing and why it matters",
      "suggested_tasks": [
        {{
          "topic": "specific research focus",
          "description": "what to investigate",
          "priority": 6
        }}
      ]
    }}
  ],
  "new_sections": [
    {{
      "title": "New Section Title",
      "description": "What this section should cover and why it's needed",
      "position": 99,
      "suggested_tasks": [
        {{
          "topic": "specific research focus",
          "description": "what to investigate",
          "priority": 5
        }}
      ]
    }}
  ]
}}

If no significant gaps exist, return: {{"section_gaps": [], "new_sections": []}}"""


SECTION_SYNTHESIS_SYSTEM_PROMPT = """You are a Research Report Writer. Your job is to synthesize research notes from multiple investigation tasks into polished, cohesive prose for one section of a larger report.

You will receive:
- The section title and description
- Research notes from all tasks for this section (with citations)
- The full report outline (for context on how this section fits)
- Brief summaries of adjacent sections (for transitions)

WRITING GUIDELINES:
1. Aim for {min_words}-{max_words} words
2. Synthesize findings across all research notes — do not just concatenate them
3. Structure with subheadings (### for main subsections, #### for sub-subsections). Do NOT write a top-level ## heading — the section title is added automatically.
4. Include specific facts, figures, dates, and direct quotes from the research notes
5. Address multiple perspectives and controversies found in the notes
6. Ensure smooth flow: each paragraph should build on the previous one
7. Do not write a general introduction or conclusion for this section
8. Reference adjacent sections where helpful ("As explored in [adjacent section title]...")
9. Minimum {min_citations} citations required, using the same [N] format from the research notes
10. NEVER add disclaimers about sources or knowledge limitations

STYLE GUIDANCE:
{style_guidance}

CITATION FORMAT:
- Preserve the [N] citation numbers from the research notes exactly as they appear
- These numbers will be remapped to global numbering during compilation
- Only use citation numbers that appear in the research notes provided

OUTPUT: Write the section in Markdown format."""

# Style guidance text mapped from config synthesis.style
_SYNTHESIS_STYLE_GUIDANCE = {
    "confident": (
        "Present findings authoritatively. Do NOT discuss gaps, limitations, "
        "missing data, or suggest follow-up research. Write as a finished "
        "report, not a research proposal."
    ),
    "balanced": (
        "You may briefly note 1-2 significant limitations where directly "
        "relevant, but focus primarily on presenting findings."
    ),
    "thorough": (
        "Where relevant, note important evidentiary gaps and suggest specific "
        "follow-up, but keep gap discussion proportional to findings."
    ),
}


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


SOURCE_EXTRACTION_SYSTEM_PROMPT = """You are a Research Extractor. Given a web page and a research task, extract the key findings relevant to the task.

Extract:
1. Key facts, statistics, and data points
2. Important quotes or claims
3. Relevant context and background
4. Specific examples or case studies

Output concise, structured notes in Markdown. Use bullet points.
Focus only on information relevant to the research task.
Do NOT add commentary or analysis - just extract what the page says.
Keep output under 1000 words."""


QUERY_GENERATOR_SYSTEM = """You are a search query specialist. Your job is to decompose a research topic into short, focused search queries. Each query should retrieve results for ONE specific sub-aspect. Never combine multiple concepts into a single query."""


QUERY_GENERATOR_PROMPT = """Break down this research task into {num_queries} separate, focused search queries.

Overall Research: {overall_query}
Section Topic: {topic}
Task Focus: {description}

Rules:
- Each query MUST be 3-8 words. No exceptions.
- Each query should target ONE specific fact, concept, or data point.
- DO NOT list multiple keywords/topics in one query.
- Vary the angles: e.g., one for definitions, one for recent data, one for mechanisms.

BAD query (too many concepts crammed in):
  polymetallic nodules cobalt crusts sulfides Ni Co Cu grades CCZ ISA maps data

GOOD queries (focused, short):
  polymetallic nodule metal concentrations
  cobalt-rich crust locations Pacific
  ISA seabed mining exploration contracts

Output ONLY the queries, one per line, no numbering or bullets."""


QUERY_GENERATOR_JSON_PROMPT = """Break down this research task into {num_queries} separate, focused search queries.

Overall Research: {overall_query}
Section Topic: {topic}
Task Focus: {description}

Rules:
- Each query MUST be 3-8 words. No exceptions.
- Each query should target ONE specific fact, concept, or data point.
- DO NOT list multiple keywords/topics in one query.
- Vary the angles: e.g., one for definitions, one for recent data, one for mechanisms.

BAD query (too many concepts crammed in):
  polymetallic nodules cobalt crusts sulfides Ni Co Cu grades CCZ ISA maps data

GOOD queries (focused, short):
  polymetallic nodule metal concentrations
  cobalt-rich crust locations Pacific
  ISA seabed mining exploration contracts

Return ONLY a JSON object: {{"queries": ["query 1", "query 2", ...]}}
"""

QUERY_GENERATOR_TOOL_NAME = "emit_search_queries"
QUERY_GENERATOR_TOOL_DESC = "Return diverse web search queries for the research task."


QUERY_REFINEMENT_QUESTIONS_SYSTEM_PROMPT = """You are a Research Scope Analyst. Given a research query, identify ambiguities and generate clarifying multiple-choice questions to help narrow the scope.

Your goal is to produce {min_questions}-{max_questions} questions that target:
- Scope: How broad or narrow should the research be?
- Timeframe: What time period is most relevant?
- Audience: Who is the intended reader?
- Depth: Technical depth vs. high-level overview?
- Sub-topics: Which specific aspects matter most?

Each question should have 3-4 answer options that represent meaningfully different research directions.

OUTPUT FORMAT:
Return ONLY a valid JSON object:
{{
  "questions": [
    {{
      "question": "Clear, specific question text",
      "options": ["Option A", "Option B", "Option C"]
    }}
  ]
}}"""


QUERY_REFINEMENT_BRIEF_SYSTEM_PROMPT = """You are a Research Brief Writer. Given a research query and the user's answers to clarifying questions, synthesize an enhanced research directive.

Your brief should be 2-4 paragraphs that:
- Incorporate the user's specific preferences and answers
- Clearly define the research scope, depth, and focus areas
- Provide actionable guidance for a research team
- Maintain the original query's intent while adding precision

OUTPUT FORMAT:
Return ONLY a valid JSON object:
{{
  "brief": "The full research brief text..."
}}"""


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
    
    def run_pre_planning(self, query: str, session_id: int) -> str:
        """
        Run deep pre-planning: web searches + analysis to build topic understanding.
        Returns a context string for use by the outline designer.
        """
        logger.info(f"Running deep pre-planning for: {query[:100]}...")

        # Pre-search: gather web context
        search_context = self._pre_search(query, session_id=session_id)

        if not search_context:
            logger.warning("Pre-planning produced no search context")
            return ""

        # Run structured analysis on the gathered context
        system = PLANNER_SYSTEM_PROMPT

        prompt = f"""Analyze the following research topic and gathered source material to build a deep understanding of the topic landscape.

## Research Query
{query}

## Pre-Planning Research
{search_context}

---
Provide a thorough analysis of the topic landscape based on these sources."""

        try:
            response = self.client.complete(
                prompt=prompt,
                system=system,
                max_tokens=self.config.llm.max_tokens.planner,
                temperature=self.config.llm.temperature.planner,
                json_mode=True,
                model=self.config.llm.models.planner
            )

            # Return the raw search context plus the structured analysis
            # so the outline designer has both rich source data and synthesis
            result = f"## Raw Source Analysis\n{search_context}\n\n## Structured Analysis\n{response}"
            logger.info("Pre-planning analysis complete")
            return result

        except Exception as e:
            logger.warning(f"Pre-planning analysis failed: {e}; returning raw context")
            return search_context

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
                        "maxItems": num_queries,
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

        # Skip blocked sources before scraping
        if is_blocked_source(url):
            logger.info(f"[pre-plan] Blocked source: {url}")
            return None

        # Skip results with very low Tavily relevance score
        min_tavily = getattr(self.config.search, 'min_tavily_score', 0.3)
        tavily_score = result.get('score', 1.0)
        if tavily_score < min_tavily:
            logger.info(f"[pre-plan] Low Tavily score ({tavily_score:.2f}): {url}")
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
    
    @staticmethod
    def _parse_json_response(response: str) -> dict:
        """Parse a JSON response, handling common LLM quirks."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try extracting from fenced block
            m = re.search(r'```(?:json)?\s*\n?(.*?)\s*```', response, re.DOTALL)
            if m:
                return json.loads(m.group(1))
            raise


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
            search_context, initial_source_count = self._execute_searches(
                queries, task.id, session_id=session_id,
                task_topic=task.topic, task_description=task.description,
                overall_query=overall_query,
            )

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
                    "maxItems": num_queries,
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
        # Request extra results from Tavily as buffer for filtering
        tavily_limit = max(5, self.config.search.results_per_query * 3)
        results = web_search(query, max_results=tavily_limit)
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

    def _extract_source_content(self, source: Source, task_topic: str, task_description: str, overall_query: str) -> str:
        """Run LLM extraction on a single source. Thread-safe. Returns extracted text or empty string."""
        content = source.full_content or source.snippet or ""
        if not content.strip():
            return ""

        max_len = self.config.scraping.max_content_length
        content_trimmed = content[:max_len]

        prompt = (
            f"Research Query: {overall_query}\n"
            f"Task Topic: {task_topic}\n"
            f"Task Description: {task_description}\n\n"
            f"Source: {source.title}\n"
            f"URL: {source.url}\n\n"
            f"--- PAGE CONTENT ---\n{content_trimmed}\n--- END ---"
        )

        try:
            result = self.client.complete(
                prompt=prompt,
                system=SOURCE_EXTRACTION_SYSTEM_PROMPT,
                max_tokens=self.config.llm.max_tokens.analyzer,
                temperature=self.config.llm.temperature.analyzer,
                model=self.config.llm.models.analyzer,
            )
            return result.strip() if result else ""
        except Exception as e:
            logger.warning(f"Source extraction failed for {source.url}: {e}")
            return ""

    def _execute_searches(
        self, queries: List[str], task_id: int, session_id: int = None,
        task_topic: str = "", task_description: str = "", overall_query: str = "",
    ) -> Tuple[str, int]:
        """Execute searches in parallel, extract per-source content, and aggregate results.

        Each query contributes up to ``results_per_query`` sources so that every
        query is represented in the final context.

        Returns (context_string, source_count).
        """
        results_per_query = self.config.search.results_per_query
        logger.info(
            f"Executing searches with {len(queries)} queries, "
            f"{results_per_query} results per query"
        )

        # Phase 1: Run all search queries in parallel, track per-query results
        query_results: Dict[int, List[dict]] = {}

        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            futures = {
                executor.submit(self._search_single_query, q, task_id, session_id): i
                for i, q in enumerate(queries)
            }
            for future in as_completed(futures):
                qi = futures[future]
                try:
                    query_results[qi] = future.result()
                except Exception as e:
                    logger.warning(f"Search query {qi} failed: {e}")
                    query_results[qi] = []

        total_raw = sum(len(r) for r in query_results.values())
        logger.info(f"Total raw results across {len(queries)} queries: {total_raw}")

        # Phase 2: For each query (in order), scrape up to results_per_query sources.
        # URLs are deduped across queries so the same page isn't scraped twice.
        saved_sources = []  # list of (position, Source pydantic object with id)
        seen_urls: set = set()
        source_counter = 0
        min_tavily = getattr(self.config.search, 'min_tavily_score', 0.3)

        for qi in sorted(query_results.keys()):
            results = query_results[qi]
            query_source_count = 0

            for result in results:
                if query_source_count >= results_per_query:
                    break

                url = result.get('url', '')
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)  # mark seen regardless of outcome

                if is_blocked_source(url):
                    logger.info(f"Blocked source (skipping): {url}")
                    continue

                tavily_score = result.get('score', 1.0)
                if tavily_score < min_tavily:
                    logger.info(f"Low Tavily score ({tavily_score:.2f}), skipping: {url}")
                    continue

                try:
                    print_scrape(url)
                    source = extract_source_info(url, result)
                    logger.info(f"Source: {url[:60]} quality={source.quality_score} content_len={len(source.full_content or '')}")

                    if source.quality_score < self.config.quality.min_source_quality:
                        logger.info(f"Skipping low-quality source: {url}")
                        continue

                    content = source.full_content or source.snippet or ""
                    if content:
                        db_source = self.db.add_source(source, task_id, position=source_counter)
                        saved_sources.append((source_counter, source, db_source))
                        source_counter += 1
                        query_source_count += 1

                except Exception as e:
                    logger.warning(f"Failed to extract from {url}: {e}")
                    continue

            logger.info(f"Query {qi}: kept {query_source_count}/{results_per_query} sources")

        # Phase 3: Run LLM extraction on each source in parallel
        extraction_results = {}  # position -> extracted_content
        if saved_sources and task_topic:
            with ThreadPoolExecutor(max_workers=min(len(saved_sources), 4)) as executor:
                future_map = {
                    executor.submit(
                        self._extract_source_content, src, task_topic, task_description, overall_query
                    ): pos
                    for pos, src, db_src in saved_sources
                }
                for future in as_completed(future_map):
                    pos = future_map[future]
                    try:
                        extracted = future.result()
                        if extracted:
                            extraction_results[pos] = extracted
                    except Exception as e:
                        logger.warning(f"Extraction future failed for position {pos}: {e}")

            # Persist extracted content to DB
            for pos, src, db_src in saved_sources:
                extracted = extraction_results.get(pos)
                if extracted and db_src.id:
                    try:
                        self.db.update_source_extraction(task_id, db_src.id, extracted)
                    except Exception as e:
                        logger.warning(f"Failed to save extraction for source {db_src.id}: {e}")

        # Phase 4: Build context string from extracted content (or fall back to raw)
        context_parts = []
        for pos, src, db_src in saved_sources:
            extracted = extraction_results.get(pos)
            if extracted:
                context_parts.append(
                    f"\n### Source {pos + 1}: {src.title}\n"
                    f"URL: {src.url}\n"
                    f"Domain: {src.domain}\n"
                    f"{'[Academic Source]' if src.is_academic else ''}\n\n"
                    f"{extracted}\n"
                )
            else:
                # Fall back to raw content
                content = src.full_content or src.snippet or ""
                max_len = self.config.scraping.max_content_length
                content_str = content[:max_len]
                if len(content) > max_len:
                    content_str += "\n[... content truncated ...]"
                context_parts.append(
                    f"\n### Source {pos + 1}: {src.title}\n"
                    f"URL: {src.url}\n"
                    f"Domain: {src.domain}\n"
                    f"{'[Academic Source]' if src.is_academic else ''}\n\n"
                    f"{content_str}\n"
                )

        return "\n\n---\n\n".join(context_parts), source_counter

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
        min_tavily = getattr(self.config.search, 'min_tavily_score', 0.3)

        for result in all_results[: max_results * 2]:
            url = result.get("url", "")
            if not url:
                continue

            # Skip blocked sources
            if is_blocked_source(url):
                logger.info(f"Gap-fill blocked source: {url}")
                continue

            # Skip low Tavily relevance
            tavily_score = result.get('score', 1.0)
            if tavily_score < min_tavily:
                logger.info(f"Gap-fill low Tavily score ({tavily_score:.2f}): {url}")
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
        """Produce research notes from gathered sources"""

        # Truncate context if too long
        max_context_tokens = 50000  # Leave room for prompt and response
        search_context = truncate_to_tokens(search_context, max_context_tokens)

        system = RESEARCH_NOTES_SYSTEM_PROMPT

        # For confident style, replace gap analysis section with a skip instruction
        style = getattr(self.config.synthesis, 'style', 'balanced')
        if style == 'confident':
            system = system.replace(
                "### Gaps / Follow-up Needed\n"
                "- Information that was NOT found but would be valuable\n"
                "- Questions raised by the research that remain unanswered",
                "### Gaps / Follow-up Needed\n"
                "Skip gap analysis — focus only on findings from available sources."
            )

        # Build other-sections context
        other_sections_text = ""
        if other_sections:
            other_sections_text = "\n## Other Sections in This Report (do not repeat their content):\n"
            other_sections_text += "\n".join(f"- {s}" for s in other_sections)
            other_sections_text += "\n"

        prompt = f"""Gather research notes for a report on: **{overall_query}**

## This Task: {task.topic}

## Research Instructions:
{task.description}
{other_sections_text}
## Source Material:
{search_context}

---

Extract and organize findings from the source material into structured research notes.
Focus on facts, data, and specific claims with citations.
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
# OUTLINE DESIGNER AGENT
# =============================================================================

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

        system = OUTLINE_DESIGNER_SYSTEM_PROMPT.format(
            target_sections=target_sections,
            max_sections=max_sections,
        )

        prompt = f"""Design a report outline for the following research query.

## Research Query
{query}

## Pre-Planning Analysis
{pre_plan_context}

---
Design approximately {target_sections} sections (no more than {max_sections}) that collectively cover this topic. Each section should be a coherent chapter covering a specific aspect."""

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


# =============================================================================
# SECTION TASK PLANNER AGENT
# =============================================================================

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

        system = SECTION_TASK_PLANNER_SYSTEM_PROMPT.format(
            tasks_per_section=tasks_per_section
        )

        prompt = f"""Generate research tasks for the following section.

## Research Query
{query}

## Full Report Outline
{outline_text}

## Target Section
Title: {section.title}
Description: {section.description}

---
Generate {tasks_per_section} focused research tasks for this section."""

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

            output_dir = self.config.output.directory
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


# =============================================================================
# GAP ANALYSIS AGENT
# =============================================================================

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

        prompt = f"""Analyze the following research project for gaps.

## Research Query
{query}

## Report Outline
{outline_text}

## Research Notes Per Section
{summaries_text}

---
Identify per-section gaps and any new sections needed. Max {self.config.gap_analysis.max_new_sections} new sections, max {self.config.gap_analysis.max_gap_fill_tasks} total new tasks."""

        try:
            response = self.client.complete(
                prompt=prompt,
                system=PIPELINE_GAP_ANALYSIS_SYSTEM_PROMPT,
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


# =============================================================================
# SYNTHESIS AGENT
# =============================================================================

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


# =============================================================================
# EDITOR AGENT
# =============================================================================

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
        """Generate a conclusion for the report."""
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


# =============================================================================
# QUERY REFINEMENT AGENT
# =============================================================================

class QueryRefinementAgent:
    """Agent responsible for generating clarifying questions and synthesizing research briefs."""

    def __init__(self):
        self.client = get_llm_client()

    @property
    def config(self):
        return get_config()

    def generate_questions(self, query: str) -> List[Dict[str, Any]]:
        """Generate clarifying multiple-choice questions for a research query.

        Returns list of dicts: [{"question": "...", "options": ["A", "B", "C"]}, ...]
        """
        logger.info(f"Generating refinement questions for: {query[:100]}")

        qr_config = self.config.query_refinement
        system = QUERY_REFINEMENT_QUESTIONS_SYSTEM_PROMPT.format(
            min_questions=qr_config.min_questions,
            max_questions=qr_config.max_questions,
        )

        prompt = f"Generate clarifying questions for this research query:\n\n{query}"

        try:
            response = self.client.complete(
                prompt=prompt,
                system=system,
                max_tokens=self.config.llm.max_tokens.refiner,
                temperature=self.config.llm.temperature.refiner,
                json_mode=True,
                model=self.config.llm.models.refiner,
            )
            data = json.loads(response)
            questions = data.get("questions", [])
            # Validate structure
            valid = []
            for q in questions:
                if isinstance(q, dict) and "question" in q and "options" in q:
                    valid.append({
                        "question": q["question"],
                        "options": q["options"][:4],  # cap at 4 options
                    })
            logger.info(f"Generated {len(valid)} refinement questions")
            return valid[:qr_config.max_questions]
        except Exception as e:
            logger.error(f"Failed to generate refinement questions: {e}")
            raise

    def synthesize_brief(self, query: str, qa_pairs: List[Dict[str, str]]) -> str:
        """Synthesize an enhanced research brief from query + Q&A pairs.

        Args:
            query: Original research query
            qa_pairs: List of {"question": "...", "answer": "..."}

        Returns:
            The research brief string
        """
        logger.info("Synthesizing research brief from Q&A pairs")

        qa_text = "\n".join(
            f"Q: {pair['question']}\nA: {pair['answer']}"
            for pair in qa_pairs
        )

        prompt = f"""Original research query:
{query}

User's answers to clarifying questions:
{qa_text}

Synthesize an enhanced research brief that incorporates these preferences."""

        try:
            response = self.client.complete(
                prompt=prompt,
                system=QUERY_REFINEMENT_BRIEF_SYSTEM_PROMPT,
                max_tokens=self.config.llm.max_tokens.refiner,
                temperature=self.config.llm.temperature.refiner,
                json_mode=True,
                model=self.config.llm.models.refiner,
            )
            data = json.loads(response)
            brief = data.get("brief", "")
            if not brief:
                raise ValueError("Empty brief returned from LLM")
            logger.info(f"Synthesized brief ({len(brief)} chars)")
            return brief
        except Exception as e:
            logger.error(f"Failed to synthesize brief: {e}")
            raise
