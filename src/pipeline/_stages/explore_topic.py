"""PlannerAgent — creates the initial research plan via deep pre-planning."""
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from src.config.settings import get_config
from src.config.types import Source
from src.infra.llm import get_llm_client
from src.pipeline._tools import web_search, extract_source_info, is_blocked_source
from src.infra._database import get_database
from src.config.logger import get_logger, print_search, print_scrape

from src.pipeline._stages._prompts import get_prompt_set

logger = get_logger(__name__)


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
        ps = get_prompt_set("explore_topic", "analyze_landscape")
        system = ps["system"]
        prompt = ps["user"].format(query=query, search_context=search_context)

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
        ps = get_prompt_set("explore_topic", "generate_planning_queries")
        prompt = ps["user"].format(num_queries=num_queries, query=query)
        try:
            tool_def = ps["tool"]
            tool_schema = dict(tool_def["parameters"])
            tool_schema["properties"]["queries"]["maxItems"] = num_queries
            result = self.client.complete_with_function(
                prompt=prompt,
                system=ps["system"],
                function_name=tool_def["name"],
                function_description=tool_def["description"],
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
        self.db.add_run_event(
            session_id=session_id, task_id=None,
            event_type="query", query_group=qg, query_text=q,
        )
        hits = web_search(q, max_results=self.config.search.pre_plan_max_results)
        for hit in hits:
            url = hit.get("url", "")
            if url:
                self.db.add_run_event(
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

        ps = get_prompt_set("explore_topic", "analyze_page")
        prompt = ps["user"].format(
            query=query, title=source.title, url=source.url, content=content,
        )
        try:
            response = self.client.complete(
                prompt=prompt,
                system=ps["system"],
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
        import re
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try extracting from fenced block
            m = re.search(r'```(?:json)?\s*\n?(.*?)\s*```', response, re.DOTALL)
            if m:
                return json.loads(m.group(1))
            raise
