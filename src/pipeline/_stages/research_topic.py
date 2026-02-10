"""ResearcherAgent — deep research on individual topics."""
import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

from src.config.settings import get_config
from src.config.types import ResearchTask, TaskStatus, Source
from src.infra.llm import get_llm_client
from src.pipeline._tools import (
    web_search, extract_source_info, is_blocked_source,
    truncate_to_tokens,
)
from src.infra._database import get_database
from src.config.logger import get_logger, print_search, print_scrape

from src.pipeline._stages._prompts import get_prompt_set

logger = get_logger(__name__)


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

        ps = get_prompt_set("research_topic", "generate_queries")
        tool_def = ps["tool"]
        tool_schema = dict(tool_def["parameters"])
        tool_schema["properties"]["queries"]["maxItems"] = num_queries

        fmt_kwargs = dict(
            num_queries=num_queries,
            overall_query=overall_query or task.topic,
            topic=task.topic,
            description=task.description,
        )

        # Attempt 0: native tool/function call for structured output.
        try:
            tool_payload = self.client.complete_with_function(
                prompt=ps["user_json"].format(**fmt_kwargs),
                system=ps["system"],
                function_name=tool_def["name"],
                function_description=tool_def["description"],
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

        attempts = [
            {
                "prompt": ps["user_json"].format(**fmt_kwargs),
                "json_mode": True,
                "label": "json",
            },
            {
                "prompt": ps["user_text"].format(**fmt_kwargs),
                "json_mode": False,
                "label": "text",
            },
        ]

        for attempt_idx, attempt in enumerate(attempts, start=1):
            response = self.client.complete(
                prompt=attempt["prompt"],
                system=ps["system"],
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
        self.db.add_run_event(
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
                self.db.add_run_event(
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

        es = get_prompt_set("research_topic", "extract_source")
        prompt = es["user"].format(
            overall_query=overall_query,
            task_topic=task_topic,
            task_description=task_description,
            title=source.title,
            url=source.url,
            content_trimmed=content_trimmed,
        )

        try:
            result = self.client.complete(
                prompt=prompt,
                system=es["system"],
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

        ig = get_prompt_set("research_topic", "identify_gaps")
        prompt = ig["user"].format(
            overall_query=overall_query,
            topic=task.topic,
            description=task.description,
            context_preview=context_preview,
            max_gap_queries=max_gap_queries,
        )

        try:
            response = self.client.complete(
                prompt=prompt,
                system=ig["system"],
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

        sn = get_prompt_set("research_topic", "synthesize_notes")
        system = sn["system"]

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

        prompt = sn["user"].format(
            overall_query=overall_query,
            topic=task.topic,
            description=task.description,
            other_sections_text=other_sections_text,
            search_context=search_context,
        )

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
