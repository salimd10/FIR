"""
Unit tests for QueryExpander.
"""
import pytest
from src.agents.query_expander import QueryExpander


class TestQueryExpander:
    """Test the query expander."""

    @pytest.fixture
    def expander(self):
        """Create query expander instance."""
        return QueryExpander()

    def test_detect_vague_query(self, expander):
        """Test vague query detection."""
        vague_queries = [
            "How is Apple doing?",
            "Tell me about the company",
            "What about revenue?",
            "Give me an overview",
        ]

        for query in vague_queries:
            assert expander.is_vague_query(query) is True

    def test_detect_specific_query(self, expander):
        """Test specific query detection."""
        specific_queries = [
            "What was Apple's revenue in FY2025?",
            "Calculate the year-over-year growth rate of R&D expenses",
            "What were the total assets reported on page 45?",
        ]

        for query in specific_queries:
            assert expander.is_vague_query(query) is False

    def test_parse_sub_questions_json(self, expander):
        """Test parsing sub-questions from JSON format."""
        llm_response = '["Question 1?", "Question 2?", "Question 3?"]'
        questions = expander._parse_sub_questions(llm_response, max_questions=5)

        assert len(questions) == 3
        assert "Question 1?" in questions

    def test_parse_sub_questions_numbered(self, expander):
        """Test parsing numbered list format."""
        llm_response = """
        1. What was the revenue?
        2. What was the profit?
        3. What are the risks?
        """
        questions = expander._parse_sub_questions(llm_response, max_questions=5)

        assert len(questions) == 3

    def test_max_questions_limit(self, expander):
        """Test that max_questions limit is respected."""
        llm_response = '["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]'
        questions = expander._parse_sub_questions(llm_response, max_questions=3)

        assert len(questions) == 3

    def test_process_specific_query(self, expander):
        """Test processing a specific query (no expansion)."""
        query = "What was Apple's revenue in FY2025?"
        result = expander.process_query(query, auto_expand=False)

        assert result["is_vague"] is False
        assert result["status"] == "specific"
        assert len(result["sub_questions"]) == 1
        assert result["sub_questions"][0] == query
