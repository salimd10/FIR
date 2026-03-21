"""
Citation engine for tracking and formatting source attributions.
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import re


class CitationEngine:
    """
    Engine for managing citations and source tracking.
    Ensures every answer can be traced back to source documents.
    """

    def __init__(self):
        """Initialize citation engine."""
        self.logger = logger.bind(module="citation_engine")

    def create_citations(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        answer: str
    ) -> List[Dict[str, Any]]:
        """
        Create structured citations from retrieved chunks.

        Args:
            retrieved_chunks: Chunks retrieved for the answer
            answer: Generated answer text

        Returns:
            List of citation dictionaries
        """
        citations = []

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            citation = {
                "citation_id": idx,
                "text": chunk.get("content", "")[:500],  # First 500 chars
                "full_text": chunk.get("content", ""),
                "page_number": chunk.get("page_number", "N/A"),
                "section": chunk.get("section_title", "N/A"),
                "chunk_type": chunk.get("chunk_type", "text"),
                "relevance_score": chunk.get("rrf_score", chunk.get("score", 0)),
                "used_in_answer": self._is_used_in_answer(
                    chunk.get("content", ""),
                    answer
                )
            }
            citations.append(citation)

        return citations

    def _is_used_in_answer(self, chunk_content: str, answer: str) -> bool:
        """
        Heuristic to check if chunk content was used in answer.

        Args:
            chunk_content: Content of the chunk
            answer: Generated answer

        Returns:
            True if likely used, False otherwise
        """
        # Simple heuristic: check for common significant phrases
        # Extract phrases of 4+ words from chunk
        chunk_words = chunk_content.lower().split()
        answer_lower = answer.lower()

        # Look for 4-word phrases from chunk in answer
        for i in range(len(chunk_words) - 3):
            phrase = " ".join(chunk_words[i:i+4])
            if phrase in answer_lower:
                return True

        return False

    def format_citation(
        self,
        page_number: Any,
        section: str = None
    ) -> str:
        """
        Format a citation string.

        Args:
            page_number: Page number
            section: Section name

        Returns:
            Formatted citation string
        """
        if section:
            return f"(Page {page_number}, {section})"
        return f"(Page {page_number})"

    def extract_citations_from_text(self, text: str) -> List[str]:
        """
        Extract citation markers from text.

        Args:
            text: Text containing citations

        Returns:
            List of citation strings
        """
        # Pattern: (Page X, Section) or (Page X)
        pattern = r'\(Page\s+\d+(?:,\s+[^)]+)?\)'
        return re.findall(pattern, text)

    def add_citations_to_answer(
        self,
        answer: str,
        citations: List[Dict[str, Any]],
        mode: str = "inline"
    ) -> str:
        """
        Add citations to answer text.

        Args:
            answer: Answer text
            citations: List of citations
            mode: 'inline' or 'footnote'

        Returns:
            Answer with citations
        """
        if mode == "inline":
            # Citations should already be in the answer from LLM
            return answer

        elif mode == "footnote":
            # Add footnote-style citations at the end
            footnotes = ["\n\n**Sources:**"]
            for citation in citations:
                if citation.get("used_in_answer", True):
                    footnotes.append(
                        f"[{citation['citation_id']}] "
                        f"Page {citation['page_number']}, "
                        f"{citation['section']}: "
                        f"{citation['text']}..."
                    )

            return answer + "\n".join(footnotes)

        return answer

    def validate_citations(
        self,
        answer: str,
        citations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that answer has proper citations.

        Args:
            answer: Answer text
            citations: Available citations

        Returns:
            Validation result
        """
        extracted_citations = self.extract_citations_from_text(answer)

        return {
            "has_citations": len(extracted_citations) > 0,
            "citation_count": len(extracted_citations),
            "available_sources": len(citations),
            "citations": extracted_citations
        }

    def create_citation_map(
        self,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Create a map of page numbers to chunks.

        Args:
            retrieved_chunks: Retrieved chunks

        Returns:
            Dictionary mapping page numbers to chunk lists
        """
        citation_map = {}

        for chunk in retrieved_chunks:
            page = chunk.get("page_number", 0)
            if page not in citation_map:
                citation_map[page] = []
            citation_map[page].append(chunk)

        return citation_map


# Convenience functions
def create_citations(
    chunks: List[Dict[str, Any]],
    answer: str
) -> List[Dict[str, Any]]:
    """
    Quick function to create citations.

    Args:
        chunks: Retrieved chunks
        answer: Generated answer

    Returns:
        List of citations
    """
    engine = CitationEngine()
    return engine.create_citations(chunks, answer)


if __name__ == "__main__":
    # Test citation engine
    engine = CitationEngine()

    # Test citation formatting
    citation = engine.format_citation(39, "Consolidated Statement of Operations")
    print(f"Formatted citation: {citation}")

    # Test extraction
    text = "Apple's R&D was $31.4B (Page 39, Consolidated Statement). Sales grew (Page 45)."
    extracted = engine.extract_citations_from_text(text)
    print(f"Extracted citations: {extracted}")
