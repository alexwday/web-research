"""QueryRefinementAgent — generates clarifying questions and synthesizes research briefs."""
import json
from typing import List, Dict, Any

from ..config import get_config
from ..llm_client import get_llm_client
from ..logger import get_logger

from .prompts import (
    QUERY_REFINEMENT_QUESTIONS_SYSTEM_PROMPT,
    QUERY_REFINEMENT_BRIEF_SYSTEM_PROMPT,
)

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
