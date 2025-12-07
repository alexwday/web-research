"""
Agents Module for Deep Research Agent
Contains the Planner, Researcher, and Editor agents
"""
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .config import get_config, ResearchTask, Source, GlossaryTerm, TaskStatus
from .llm_client import get_llm_client
from .tools import (
    web_search, extract_source_info, count_words, count_citations,
    truncate_to_tokens, generate_file_path
)
from .database import get_database
from .logger import get_logger, print_search, print_scrape

logger = get_logger(__name__)


# =============================================================================
# PROMPTS
# =============================================================================

PLANNER_SYSTEM_PROMPT = """You are a Research Architect specializing in breaking down complex topics into comprehensive, exhaustive research plans.

Your task is to analyze a research query and create a detailed plan that, when executed, will result in a book-length, comprehensive report covering every aspect of the topic.

CRITICAL PRIORITY RULES (tasks are executed from highest to lowest priority):
- Priority 10: Introduction/Overview (MUST be researched first)
- Priority 9: Historical Background, Core Definitions
- Priority 7-8: Main body content, analysis, current state
- Priority 5-6: Applications, case studies, specific examples
- Priority 3-4: Edge cases, alternative perspectives, limitations
- Priority 1-2: Conclusion, Future Directions, Summary (MUST be researched LAST after all other content exists)

GUIDELINES:
1. Break the topic into logical chapters/sections following academic structure:
   - Introduction/Overview (priority: 10)
   - Historical Background/Context (priority: 9)
   - Core Concepts and Definitions (priority: 9)
   - Detailed Analysis sections (priority: 7-8)
   - Current State/Applications (priority: 6-7)
   - Challenges and Limitations (priority: 4-5)
   - Future Directions (priority: 2)
   - Conclusion (priority: 1 - ALWAYS LAST)

2. Each task should be specific enough to research in a single focused session
3. Tasks should build upon each other logically
4. Include tasks for edge cases, controversies, and alternative perspectives
5. Prioritize foundational knowledge before advanced topics
6. Aim for {min_tasks} to {max_tasks} initial tasks
7. DO NOT include meta-tasks like "Write introduction" - each task should be a RESEARCH topic

OUTPUT FORMAT:
You MUST output ONLY a valid JSON object with a "tasks" array. Each task object must have:
- "topic": Brief title (max 100 chars)
- "description": Detailed research instructions (2-4 sentences explaining exactly what to investigate)
- "priority": 1-10 following the rules above (10 = research first, 1 = research last)
- "dependencies": List of topic names this depends on (empty for independent tasks)

Example:
{{
  "tasks": [
    {{
      "topic": "Historical Origins of Machine Learning",
      "description": "Research the early history of machine learning from 1950s-1980s. Focus on key papers, pioneering researchers like Alan Turing, Arthur Samuel, and Frank Rosenblatt. Document the evolution from simple perceptrons to early neural networks.",
      "priority": 9,
      "dependencies": []
    }},
    {{
      "topic": "Conclusion and Synthesis",
      "description": "Synthesize all findings into a comprehensive conclusion. Summarize key insights, evaluate the current state of the field, and provide final recommendations.",
      "priority": 1,
      "dependencies": ["all other topics"]
    }}
  ]
}}"""


RESEARCHER_SYSTEM_PROMPT = """You are a Deep Research Specialist. Your task is to thoroughly research a specific topic and write a comprehensive, well-cited section for an academic report.

RESEARCH GUIDELINES:
1. Use the search results and scraped content provided to write detailed analysis
2. Be VERBOSE and COMPREHENSIVE - aim for {min_words} to {max_words} words
3. Include specific facts, figures, dates, and quotes from sources
4. Cite EVERY claim with the source URL using format: [Source: URL]
5. Structure with clear headings (## for main, ### for sub)
6. Define technical terms when first introduced
7. Include relevant examples and case studies
8. Address multiple perspectives and controversies if they exist

CITATION REQUIREMENTS:
- Minimum {min_citations} citations required
- Use inline citations: [Source: https://example.com]
- Include direct quotes when appropriate: "quote" [Source: URL]

OUTPUT FORMAT:
Write the section in Markdown format. 

IMPORTANT ABOUT NEW TASKS:
- Only suggest a new sub-topic if it is ABSOLUTELY ESSENTIAL and could not be covered in this section
- Most research tasks should NOT spawn new tasks
- Never suggest more than 1 new task
- Never suggest tasks that overlap with common research topics like "history", "future", "applications"
- Set priority to 3 (low) for any suggested task

If you genuinely discover something critical that needs its own section, include at the END:

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

For most research tasks, you should NOT include any JSON block - just write the content."""


EDITOR_SYSTEM_PROMPT = """You are a Research Editor. Your task is to create a polished executive summary and introduction for a comprehensive research report.

Based on the research topics and any provided content, write:
1. An executive summary (300-500 words) highlighting key findings
2. A compelling introduction that frames the research
3. Key takeaways or highlights

Write in a professional, academic tone suitable for publication."""


QUERY_GENERATOR_PROMPT = """Generate {num_queries} distinct, highly effective search queries to find comprehensive information about:

Topic: {topic}
Specific Focus: {description}

Requirements:
- Queries should be diverse and cover different aspects
- Include both broad and specific queries
- Consider academic, news, and technical sources
- Each query should be 3-8 words for optimal search results

Output ONLY the queries, one per line, no numbering or bullets."""


# =============================================================================
# PLANNER AGENT
# =============================================================================

class PlannerAgent:
    """Agent responsible for creating the initial research plan"""
    
    def __init__(self):
        self.config = get_config()
        self.client = get_llm_client()
        self.db = get_database()
    
    def create_plan(self, query: str, session_id: int) -> List[ResearchTask]:
        """
        Analyze the query and create a comprehensive research plan
        """
        logger.info(f"Creating research plan for: {query[:100]}...")
        
        # Format the system prompt with config values
        system = PLANNER_SYSTEM_PROMPT.format(
            min_tasks=self.config.research.min_initial_tasks,
            max_tasks=self.config.research.max_total_tasks // 2  # Leave room for recursion
        )
        
        prompt = f"""Create a comprehensive research plan for the following query:

{query}

Remember to create an exhaustive plan covering all aspects of this topic. The goal is to produce a book-length report that leaves no stone unturned."""
        
        try:
            response = self.client.complete(
                prompt=prompt,
                system=system,
                max_tokens=self.config.llm.max_tokens.planner,
                temperature=self.config.llm.temperature.planner,
                json_mode=True,
                model=self.config.llm.models.planner
            )
            
            # Parse the response
            tasks = self._parse_plan_response(response, session_id)
            
            logger.info(f"Created plan with {len(tasks)} tasks")
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to create plan: {e}")
            raise
    
    def _parse_plan_response(self, response: str, session_id: int) -> List[ResearchTask]:
        """Parse the planner's JSON response into ResearchTask objects"""
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
                # Try to find any list in the response
                for value in data.values():
                    if isinstance(value, list):
                        task_list = value
                        break
                else:
                    raise ValueError("Could not find task list in response")
            
            tasks = []
            output_dir = self.config.output.directory
            
            # Keywords that indicate tasks that should run last
            conclusion_keywords = ['conclusion', 'summary', 'synthesis', 'final', 'closing', 'wrap-up', 'recap']
            future_keywords = ['future', 'outlook', 'prediction', 'forecast', 'next steps']
            intro_keywords = ['introduction', 'overview', 'background', 'foundation', 'basics', 'fundamentals']
            
            for i, item in enumerate(task_list):
                topic = item.get("topic", f"Task {i+1}")
                topic_lower = topic.lower()
                priority = item.get("priority", 5)
                
                # Enforce priority rules based on topic content
                if any(kw in topic_lower for kw in conclusion_keywords):
                    priority = 1  # Conclusion always last
                elif any(kw in topic_lower for kw in future_keywords):
                    priority = 2  # Future directions second to last
                elif any(kw in topic_lower for kw in intro_keywords) and priority < 9:
                    priority = max(priority, 9)  # Intro should be early
                
                task = ResearchTask(
                    topic=topic,
                    description=item.get("description", ""),
                    file_path=generate_file_path(
                        topic,
                        output_dir,
                        i + 1
                    ),
                    priority=priority,
                    depth=0,
                    status=TaskStatus.PENDING
                )
                
                # Save to database
                saved_task = self.db.add_task(task, session_id)
                tasks.append(saved_task)
            
            # Update session with task count
            self.db.update_session(session_id, total_tasks=len(tasks))
            
            return tasks
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            logger.debug(f"Response was: {response[:500]}")
            raise


# =============================================================================
# RESEARCHER AGENT
# =============================================================================

class ResearcherAgent:
    """Agent responsible for deep research on individual topics"""
    
    def __init__(self):
        self.config = get_config()
        self.client = get_llm_client()
        self.db = get_database()
    
    def research_task(self, task: ResearchTask) -> Tuple[str, List[Dict], List[Dict]]:
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
            
            # Step 3: Synthesize and write
            content, new_tasks, glossary_terms = self._synthesize(task, search_context)
            
            return content, new_tasks, glossary_terms
            
        except Exception as e:
            logger.error(f"Research failed for task {task.id}: {e}")
            self.db.mark_task_failed(task.id, str(e))
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
                
                # Save source to database
                self.db.add_source(source, task_id)
                
                # Build context
                content = source.full_content or source.snippet or ""
                if content:
                    context_parts.append(f"""
### Source: {source.title}
URL: {source.url}
Domain: {source.domain}
{'[Academic Source]' if source.is_academic else ''}

{content[:8000]}
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
        search_context: str
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
        
        prompt = f"""Research and write a comprehensive section on the following topic:

## Topic: {task.topic}

## Research Instructions:
{task.description}

## Source Material:
{search_context}

---

Write a detailed, well-cited section now. Be thorough and comprehensive.
If you discover important sub-topics that need separate investigation, include them in the JSON block at the end."""
        
        response = self.client.complete(
            prompt=prompt,
            system=system,
            max_tokens=self.config.llm.max_tokens.writer,
            temperature=self.config.llm.temperature.writer,
            model=self.config.llm.models.writer
        )
        
        # Parse response for content, new tasks, and glossary
        content, new_tasks, glossary_terms = self._parse_research_response(response, task)
        
        return content, new_tasks, glossary_terms
    
    def _parse_research_response(
        self, 
        response: str, 
        task: ResearchTask
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
        total_tasks = self.db.get_task_count()
        at_task_limit = total_tasks >= self.config.research.max_total_tasks
        
        # Try to extract JSON block
        if "```json" in response:
            try:
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    data = json.loads(json_str)
                    
                    # Extract new tasks if allowed - LIMIT TO 1 MAX
                    if can_recurse and not at_task_limit:
                        raw_tasks = data.get("new_tasks", [])
                        # Only take the first task to prevent task explosion
                        for t in raw_tasks[:1]:
                            if isinstance(t, dict) and "topic" in t:
                                # Check if this task is too similar to existing tasks
                                topic = t["topic"]
                                existing_tasks = self.db.get_all_tasks()
                                is_duplicate = any(
                                    topic.lower() in et.topic.lower() or et.topic.lower() in topic.lower()
                                    for et in existing_tasks
                                )
                                
                                if not is_duplicate:
                                    new_tasks.append({
                                        "topic": topic,
                                        "description": t.get("description", ""),
                                        "priority": min(t.get("priority", 3), 4),  # Cap at 4 for discovered tasks
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
                    content = response[:json_match.start()].strip()
                    
            except (json.JSONDecodeError, AttributeError) as e:
                logger.debug(f"Could not parse JSON block: {e}")
        
        return content, new_tasks, glossary_terms


# =============================================================================
# EDITOR AGENT
# =============================================================================

class EditorAgent:
    """Agent responsible for final compilation and editing"""
    
    def __init__(self):
        self.config = get_config()
        self.client = get_llm_client()
        self.db = get_database()
    
    def generate_executive_summary(self, query: str, topics: List[str]) -> str:
        """Generate an executive summary for the report"""
        logger.info("Generating executive summary...")
        
        topics_text = "\n".join(f"- {t}" for t in topics[:30])
        
        prompt = f"""Based on the following research query and topics covered, write an executive summary.

Original Research Query:
{query}

Topics Researched:
{topics_text}

Write a compelling executive summary (300-500 words) that:
1. Introduces the research topic and its significance
2. Highlights key themes and findings
3. Provides context for why this research matters
4. Previews the structure of the report"""
        
        response = self.client.complete(
            prompt=prompt,
            system=EDITOR_SYSTEM_PROMPT,
            max_tokens=self.config.llm.max_tokens.editor,
            temperature=self.config.llm.temperature.editor,
            model=self.config.llm.models.editor
        )
        
        return response
    
    def generate_conclusion(self, query: str, topics: List[str], word_count: int) -> str:
        """Generate a conclusion for the report"""
        logger.info("Generating conclusion...")
        
        topics_text = "\n".join(f"- {t}" for t in topics[:30])
        
        prompt = f"""Based on the following research, write a comprehensive conclusion.

Original Research Query:
{query}

Topics Covered:
{topics_text}

Total Words Written: {word_count:,}

Write a thoughtful conclusion (400-600 words) that:
1. Synthesizes the key findings
2. Discusses implications
3. Identifies areas for future research
4. Provides final thoughts and recommendations"""
        
        response = self.client.complete(
            prompt=prompt,
            system=EDITOR_SYSTEM_PROMPT,
            max_tokens=self.config.llm.max_tokens.editor,
            temperature=self.config.llm.temperature.editor,
            model=self.config.llm.models.editor
        )
        
        return response
