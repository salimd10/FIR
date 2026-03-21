"""
Unit tests for ChunkingStrategy.
"""
import pytest
from src.ingestion.chunking_strategy import SemanticChunker


class TestSemanticChunker:
    """Test the semantic chunking strategy."""

    @pytest.fixture
    def chunker(self):
        """Create chunker instance."""
        return SemanticChunker(
            chunk_size=512,
            chunk_overlap=50
        )

    def test_chunk_text(self, chunker):
        """Test basic text chunking."""
        text = "This is a test. " * 100  # Create long text

        chunks = chunker.chunk_text(
            text=text,
            metadata={"page": 1, "section": "Test"}
        )

        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)

    def test_chunk_metadata_preservation(self, chunker):
        """Test that metadata is preserved in chunks."""
        text = "Test content."
        metadata = {
            "page": 42,
            "section": "Risk Factors",
            "source": "test.pdf"
        }

        chunks = chunker.chunk_text(text, metadata)

        for chunk in chunks:
            assert chunk["metadata"]["page_number"] == 42
            assert chunk["metadata"]["section"] == "Risk Factors"

    def test_empty_text(self, chunker):
        """Test chunking empty text."""
        chunks = chunker.chunk_text("", metadata={})
        assert len(chunks) == 0

    def test_short_text(self, chunker):
        """Test chunking text shorter than chunk size."""
        text = "Short text."
        chunks = chunker.chunk_text(text, metadata={})

        assert len(chunks) == 1
        assert chunks[0]["content"] == text

    def test_token_counting(self, chunker):
        """Test token counting functionality."""
        text = "This is a test sentence."
        token_count = chunker._count_tokens(text)

        assert isinstance(token_count, int)
        assert token_count > 0
