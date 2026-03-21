"""
Query Expander for handling vague or broad questions.
Breaks down complex queries into specific sub-questions.
"""
from typing import List, Dict, Any, Optional
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger
import json
import re

from src.utils.prompts import QUERY_EXPANSION_PROMPT
from src.config import get_settings


class QueryExpander:
    """
    Expands vague queries into specific sub-questions.

    Handles:
    - Broad questions (e.g., "How is the company doing?")
    - Multi-faceted questions
    - Ambiguous queries
    """

    def __init__(
        self,
        llm_model: str = "claude-3-5-sonnet-20241022",
        llm_provider: str = "anthropic",
        temperature: float = 0.3
    ):
        """
        Initialize query expander.

        Args:
            llm_model: Model name (e.g., "claude-3-5-sonnet-20241022" or "gpt-4o-mini")
            llm_provider: "anthropic" or "openai"
            temperature: LLM temperature for creativity
        """
        self.settings = get_settings()
        self.logger = logger.bind(module="query_expander")
        self._llm_model = llm_model
        self._llm_provider = llm_provider
        self._temperature = temperature
        self._llm = None  # lazily initialised on first use

        self.logger.info(f"Query Expander initialised with {llm_provider} model: {llm_model}")

    @property
    def llm(self):
        """Return the LLM, initialising it on first access."""
        if self._llm is None:
            if self._llm_provider.lower() == "anthropic":
                self._llm = ChatAnthropic(
                    model=self._llm_model,
                    temperature=self._temperature,
                    anthropic_api_key=self.settings.anthropic_api_key,
                )
            else:  # openai
                self._llm = ChatOpenAI(
                    model=self._llm_model,
                    temperature=self._temperature,
                    api_key=self.settings.openai_api_key,
                )
        return self._llm

    def is_vague_query(self, query: str) -> bool:
        """
        Detect if a query is vague or broad.

        Args:
            query: User's query

        Returns:
            True if query is vague
        """
        vague_patterns = [
            r'\bhow\s+(is|are|was|were)\s+\w+\s+(doing|performing)',
            r'\btell\s+me\s+about',
            r'\bwhat\s+about',
            r'\bgive\s+me\s+(an\s+)?overview',
            r'\bsummarize',
            r'\bgeneral\s+information',
            r'\bwhat\s+do\s+you\s+know',
            r'\banything\s+about',
        ]

        query_lower = query.lower()

        # Check patterns
        for pattern in vague_patterns:
            if re.search(pattern, query_lower):
                return True

        # Check length - very short queries are often vague
        word_count = len(query.split())
        if word_count <= 5:
            # Short and potentially vague
            vague_words = ['how', 'what', 'overview', 'summary', 'general', 'about']
            if any(word in query_lower for word in vague_words):
                return True

        return False

    def expand_query(
        self,
        query: str,
        max_sub_questions: int = 5,
        domain: str = "financial 10-K filing"
    ) -> Dict[str, Any]:
        """
        Expand a vague query into specific sub-questions.

        Args:
            query: User's original query
            max_sub_questions: Maximum number of sub-questions
            domain: Document domain context

        Returns:
            Dictionary with sub-questions and metadata
        """
        self.logger.info(f"Expanding query: {query}")

        try:
            # Create expansion prompt
            expansion_prompt = f"""{QUERY_EXPANSION_PROMPT}

Original Question: "{query}"

Context: This is about a {domain}.

Generate {max_sub_questions} specific, answerable sub-questions that would help comprehensively answer the original question.

Format your response as a JSON array of strings:
["sub-question 1", "sub-question 2", ...]

Sub-questions:"""

            messages = [
                SystemMessage(content="You are a helpful assistant that breaks down vague questions into specific sub-questions."),
                HumanMessage(content=expansion_prompt)
            ]

            response = self.llm.invoke(messages)
            content = response.content.strip()

            # Extract JSON from response
            sub_questions = self._parse_sub_questions(content, max_sub_questions)

            return {
                "original_query": query,
                "is_vague": True,
                "sub_questions": sub_questions,
                "num_sub_questions": len(sub_questions),
                "status": "expanded"
            }

        except Exception as e:
            self.logger.error(f"Error expanding query: {str(e)}")
            return {
                "original_query": query,
                "is_vague": False,
                "sub_questions": [query],  # Fallback to original
                "num_sub_questions": 1,
                "status": "error",
                "error": str(e)
            }

    def _parse_sub_questions(
        self,
        llm_response: str,
        max_questions: int
    ) -> List[str]:
        """
        Parse sub-questions from LLM response.

        Args:
            llm_response: Raw LLM response
            max_questions: Maximum number to extract

        Returns:
            List of sub-questions
        """
        try:
            # Try to extract JSON array
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group(0))
                return questions[:max_questions]

            # Fallback: Extract numbered lines
            lines = llm_response.split('\n')
            questions = []

            for line in lines:
                # Match patterns like "1. Question" or "- Question"
                match = re.match(r'^\s*[\d\-\*\.]+\s*(.+)$', line)
                if match:
                    question = match.group(1).strip('"').strip()
                    if question:
                        questions.append(question)

            return questions[:max_questions]

        except Exception as e:
            self.logger.warning(f"Could not parse sub-questions: {str(e)}")
            return []

    def suggest_refinements(self, query: str) -> List[str]:
        """
        Suggest ways to refine a vague query.

        Args:
            query: User's query

        Returns:
            List of suggested refinements
        """
        refinement_prompt = f"""The user asked: "{query}"

This question is too vague. Suggest 3 ways they could make it more specific.

Format your response as a JSON array of strings:
["suggestion 1", "suggestion 2", "suggestion 3"]

Suggestions:"""

        try:
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=refinement_prompt)
            ]

            response = self.llm.invoke(messages)
            suggestions = self._parse_sub_questions(response.content, max_questions=3)

            return suggestions

        except Exception as e:
            self.logger.error(f"Error generating suggestions: {str(e)}")
            return []

    def process_query(
        self,
        query: str,
        auto_expand: bool = True
    ) -> Dict[str, Any]:
        """
        Process a query - expand if vague, return as-is if specific.

        Args:
            query: User's query
            auto_expand: Whether to automatically expand vague queries

        Returns:
            Processing result
        """
        is_vague = self.is_vague_query(query)

        if is_vague and auto_expand:
            return self.expand_query(query)
        elif is_vague and not auto_expand:
            return {
                "original_query": query,
                "is_vague": True,
                "sub_questions": [query],
                "suggestions": self.suggest_refinements(query),
                "status": "vague_not_expanded"
            }
        else:
            return {
                "original_query": query,
                "is_vague": False,
                "sub_questions": [query],
                "status": "specific"
            }


def create_query_expander(model: str = "gpt-4o-mini") -> QueryExpander:
    """
    Factory function to create query expander.

    Args:
        model: LLM model name

    Returns:
        QueryExpander instance
    """
    return QueryExpander(llm_model=model)


if __name__ == "__main__":
    # Test the query expander
    expander = QueryExpander()

    test_queries = [
        "How is Apple doing?",
        "What was Apple's revenue in FY2025?",
        "Tell me about the company",
        "What were the R&D expenses in 2025 compared to 2024?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print(f"Is vague: {expander.is_vague_query(query)}")

        if expander.is_vague_query(query):
            result = expander.expand_query(query)
            print(f"Sub-questions:")
            for i, sq in enumerate(result['sub_questions'], 1):
                print(f"  {i}. {sq}")
