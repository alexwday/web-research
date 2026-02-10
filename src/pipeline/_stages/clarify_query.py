"""QueryRefinementAgent — generates clarifying questions and synthesizes research briefs."""
import json
from typing import List, Dict, Any

from src.config.settings import get_config
from src.infra.llm import get_llm_client
from src.config.logger import get_logger

from src.pipeline._stages._prompts import get_prompt_set

logger = get_logger(__name__)


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
        ps = get_prompt_set("clarify_query", "generate_questions")
        system = ps["system"].format(
            min_questions=qr_config.min_questions,
            max_questions=qr_config.max_questions,
        )

        prompt = ps["user"].format(query=query)

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

        ps = get_prompt_set("clarify_query", "synthesize_brief")
        prompt = ps["user"].format(query=query, qa_text=qa_text)

        try:
            response = self.client.complete(
                prompt=prompt,
                system=ps["system"],
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
