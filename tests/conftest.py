"""
Pytest configuration and fixtures.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "page": 1,
        "section": "Test Section",
        "source": "test.pdf"
    }


@pytest.fixture
def sample_chunk():
    """Sample chunk for testing."""
    return {
        "content": "This is sample content for testing.",
        "metadata": {
            "page_number": 1,
            "section": "Test Section",
            "chunk_type": "text"
        },
        "chunk_id": "test-chunk-1"
    }
