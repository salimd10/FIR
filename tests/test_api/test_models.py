"""
Unit tests for Pydantic models.
"""
import pytest
from src.api.models import (
    QueryRequest,
    QueryResponse,
    Citation,
    HealthResponse
)


class TestQueryRequest:
    """Test QueryRequest model."""

    def test_create_basic_request(self):
        """Test creating a basic query request."""
        request = QueryRequest(question="What is the revenue?")

        assert request.question == "What is the revenue?"
        assert request.return_sources is True
        assert request.max_sources == 5
        assert request.top_k == 5
        assert request.expand_query is False

    def test_create_custom_request(self):
        """Test creating a request with custom parameters."""
        request = QueryRequest(
            question="Calculate growth rate",
            return_sources=False,
            max_sources=10,
            top_k=8,
            expand_query=True
        )

        assert request.return_sources is False
        assert request.max_sources == 10
        assert request.top_k == 8
        assert request.expand_query is True

    def test_missing_required_field(self):
        """Test that missing required field raises error."""
        with pytest.raises(Exception):
            QueryRequest()


class TestCitation:
    """Test Citation model."""

    def test_create_citation(self):
        """Test creating a citation."""
        citation = Citation(
            citation_id=1,
            text="Sample text",
            page_number=42,
            section="Risk Factors",
            score=0.95,
            chunk_type="text"
        )

        assert citation.citation_id == 1
        assert citation.page_number == 42
        assert citation.score == 0.95


class TestQueryResponse:
    """Test QueryResponse model."""

    def test_create_response(self):
        """Test creating a query response."""
        response = QueryResponse(
            query_id="test-123",
            question="Test question",
            answer="Test answer",
            citations=[],
            confidence=0.85,
            processing_time_ms=150
        )

        assert response.query_id == "test-123"
        assert response.confidence == 0.85
        assert response.status == "success"

    def test_response_with_citations(self):
        """Test response with citations."""
        citation = Citation(
            citation_id=1,
            text="Source text",
            page_number=10,
            section="Section A",
            score=0.9,
            chunk_type="text"
        )

        response = QueryResponse(
            query_id="test-123",
            question="Question",
            answer="Answer",
            citations=[citation],
            confidence=0.9,
            processing_time_ms=200
        )

        assert len(response.citations) == 1
        assert response.citations[0].citation_id == 1


class TestHealthResponse:
    """Test HealthResponse model."""

    def test_create_health_response(self):
        """Test creating a health response."""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            qdrant_connected=True,
            embeddings_available=True
        )

        assert response.status == "healthy"
        assert response.qdrant_connected is True
