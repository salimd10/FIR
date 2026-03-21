"""
Smart chunking strategy for financial documents.
Preserves semantic boundaries, table integrity, and contextual relationships.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tiktoken
from loguru import logger


@dataclass
class DocumentChunk:
    """
    Represents a chunk of document content with metadata.
    """
    chunk_id: str
    content: str
    chunk_type: str  # 'text', 'table', 'mixed'
    page_number: int
    section_title: Optional[str] = None
    token_count: int = 0
    is_table: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FinancialDocumentChunker:
    """
    Advanced chunking strategy specifically designed for financial documents.

    Key Features:
    - Preserves table integrity (never split tables)
    - Maintains semantic boundaries (sections, paragraphs)
    - Includes context headers with each chunk
    - Handles footnotes and references
    - Optimized for RAG retrieval
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        model_name: str = "gpt-4"
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Number of overlapping tokens between chunks
            model_name: Model name for tokenization
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger.bind(module="chunker")

        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.logger.warning(f"Model {model_name} not found, using cl100k_base")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def chunk_document(
        self,
        parsed_document: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Chunk a parsed document into semantic chunks.

        Args:
            parsed_document: Document from PDF parser

        Returns:
            List of document chunks
        """
        elements = parsed_document.get("content", [])
        metadata = parsed_document.get("metadata", {})

        chunks = []
        chunk_counter = 0

        # Process elements by section
        current_section_elements = []
        current_section = None

        for element in elements:
            element_section = element.get("section")

            # Start new section
            if element_section and element_section != current_section:
                # Process accumulated elements from previous section
                if current_section_elements:
                    section_chunks = self._chunk_section(
                        current_section_elements,
                        current_section,
                        chunk_counter
                    )
                    chunks.extend(section_chunks)
                    chunk_counter += len(section_chunks)

                # Reset for new section
                current_section = element_section
                current_section_elements = [element]
            else:
                current_section_elements.append(element)

        # Process remaining elements
        if current_section_elements:
            section_chunks = self._chunk_section(
                current_section_elements,
                current_section,
                chunk_counter
            )
            chunks.extend(section_chunks)

        self.logger.info(f"Created {len(chunks)} chunks from document")
        return chunks

    def _chunk_section(
        self,
        elements: List[Dict[str, Any]],
        section_title: Optional[str],
        start_counter: int
    ) -> List[DocumentChunk]:
        """
        Chunk elements within a section.

        Args:
            elements: Elements in the section
            section_title: Title of the section
            start_counter: Starting chunk counter

        Returns:
            List of chunks for this section
        """
        chunks = []
        current_chunk_elements = []
        current_token_count = 0

        # Add section header to all chunks
        section_header = f"## {section_title}\n\n" if section_title else ""
        header_token_count = self._count_tokens(section_header)

        for element in elements:
            element_content = element.get("enhanced_content") or element.get("content", "")
            element_token_count = self._count_tokens(element_content)

            # Special handling for tables - never split them
            if element.get("is_table"):
                # If table is too large, create dedicated chunk
                if element_token_count > self.chunk_size:
                    # Save current chunk first
                    if current_chunk_elements:
                        chunk = self._create_chunk_from_elements(
                            current_chunk_elements,
                            section_title,
                            section_header,
                            start_counter + len(chunks)
                        )
                        chunks.append(chunk)
                        current_chunk_elements = []
                        current_token_count = 0

                    # Create dedicated chunk for large table
                    chunk = self._create_table_chunk(
                        element,
                        section_title,
                        section_header,
                        start_counter + len(chunks)
                    )
                    chunks.append(chunk)
                    continue

                # If adding table exceeds chunk size, create new chunk
                if current_token_count + element_token_count > self.chunk_size:
                    if current_chunk_elements:
                        chunk = self._create_chunk_from_elements(
                            current_chunk_elements,
                            section_title,
                            section_header,
                            start_counter + len(chunks)
                        )
                        chunks.append(chunk)
                        current_chunk_elements = []
                        current_token_count = 0

            # Check if adding element exceeds chunk size
            if current_token_count + element_token_count > self.chunk_size:
                if current_chunk_elements:
                    # Create chunk from current elements
                    chunk = self._create_chunk_from_elements(
                        current_chunk_elements,
                        section_title,
                        section_header,
                        start_counter + len(chunks)
                    )
                    chunks.append(chunk)

                    # Start new chunk with overlap
                    current_chunk_elements = self._get_overlap_elements(
                        current_chunk_elements,
                        self.chunk_overlap
                    )
                    current_token_count = sum(
                        self._count_tokens(e.get("content", ""))
                        for e in current_chunk_elements
                    )

            # Add element to current chunk
            current_chunk_elements.append(element)
            current_token_count += element_token_count

        # Create final chunk
        if current_chunk_elements:
            chunk = self._create_chunk_from_elements(
                current_chunk_elements,
                section_title,
                section_header,
                start_counter + len(chunks)
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk_from_elements(
        self,
        elements: List[Dict[str, Any]],
        section_title: Optional[str],
        section_header: str,
        chunk_id: int
    ) -> DocumentChunk:
        """
        Create a DocumentChunk from a list of elements.

        Args:
            elements: List of elements to combine
            section_title: Section title
            section_header: Formatted section header
            chunk_id: Chunk identifier

        Returns:
            DocumentChunk object
        """
        # Combine element content
        content_parts = [section_header] if section_header else []

        for element in elements:
            content = element.get("enhanced_content") or element.get("content", "")
            if content:
                content_parts.append(content)

        full_content = "\n\n".join(content_parts)

        # Determine page number (use first element's page)
        page_number = elements[0].get("page_number", 1) if elements else 1

        # Determine chunk type
        has_table = any(e.get("is_table") for e in elements)
        chunk_type = "table" if has_table and len(elements) == 1 else "mixed" if has_table else "text"

        # Collect metadata
        metadata = {
            "element_count": len(elements),
            "element_types": [e.get("type") for e in elements],
            "has_table": has_table,
        }

        return DocumentChunk(
            chunk_id=f"chunk_{chunk_id}",
            content=full_content,
            chunk_type=chunk_type,
            page_number=page_number,
            section_title=section_title,
            token_count=self._count_tokens(full_content),
            is_table=has_table,
            metadata=metadata
        )

    def _create_table_chunk(
        self,
        table_element: Dict[str, Any],
        section_title: Optional[str],
        section_header: str,
        chunk_id: int
    ) -> DocumentChunk:
        """
        Create a dedicated chunk for a large table.

        Args:
            table_element: Table element
            section_title: Section title
            section_header: Formatted section header
            chunk_id: Chunk identifier

        Returns:
            DocumentChunk object
        """
        content = section_header + "\n\n" + (
            table_element.get("enhanced_content") or
            table_element.get("content", "")
        )

        metadata = {
            "element_count": 1,
            "element_types": [table_element.get("type")],
            "has_table": True,
            "table_data": table_element.get("table_data"),
            "financial_values": table_element.get("financial_values", {}),
        }

        return DocumentChunk(
            chunk_id=f"chunk_{chunk_id}",
            content=content,
            chunk_type="table",
            page_number=table_element.get("page_number", 1),
            section_title=section_title,
            token_count=self._count_tokens(content),
            is_table=True,
            metadata=metadata
        )

    def _get_overlap_elements(
        self,
        elements: List[Dict[str, Any]],
        overlap_tokens: int
    ) -> List[Dict[str, Any]]:
        """
        Get elements for chunk overlap.

        Args:
            elements: Current chunk elements
            overlap_tokens: Number of tokens to overlap

        Returns:
            Elements for overlap
        """
        overlap_elements = []
        token_count = 0

        # Start from the end and work backwards
        for element in reversed(elements):
            element_tokens = self._count_tokens(element.get("content", ""))

            if token_count + element_tokens > overlap_tokens:
                break

            overlap_elements.insert(0, element)
            token_count += element_tokens

        return overlap_elements

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def create_retrieval_optimized_chunks(
        self,
        chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """
        Add retrieval-optimized metadata to chunks.
        This includes keywords, entity extraction, etc.

        Args:
            chunks: List of chunks

        Returns:
            Enhanced chunks
        """
        for chunk in chunks:
            # Add keyword metadata for better retrieval
            chunk.metadata["keywords"] = self._extract_keywords(chunk.content)

            # Add summary for large chunks
            if chunk.token_count > 800:
                chunk.metadata["summary"] = self._create_chunk_summary(chunk.content)

        return chunks

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords for better retrieval.
        Simple implementation - can be enhanced with NLP.

        Args:
            text: Chunk content

        Returns:
            List of keywords
        """
        # Financial keywords to look for
        financial_terms = [
            "revenue", "sales", "income", "expense", "cost",
            "asset", "liability", "equity", "cash flow",
            "r&d", "research", "development", "operating",
            "net income", "gross margin", "ebitda",
            "segment", "region", "geographic"
        ]

        keywords = []
        text_lower = text.lower()

        for term in financial_terms:
            if term in text_lower:
                keywords.append(term)

        return keywords

    def _create_chunk_summary(self, text: str) -> str:
        """
        Create a brief summary of chunk content.
        Simple implementation - takes first few sentences.

        Args:
            text: Chunk content

        Returns:
            Summary string
        """
        sentences = text.split('.')[:2]
        return '.'.join(sentences).strip()[:200] + "..."


# Convenience function
def chunk_financial_document(
    parsed_document: Dict[str, Any],
    chunk_size: int = 1024,
    chunk_overlap: int = 128
) -> List[DocumentChunk]:
    """
    Quick function to chunk a financial document.

    Args:
        parsed_document: Parsed document from PDF parser
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks

    Returns:
        List of DocumentChunk objects
    """
    chunker = FinancialDocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return chunker.chunk_document(parsed_document)


if __name__ == "__main__":
    # Test the chunker
    print("Chunking strategy module loaded successfully")
    print(f"Default chunk size: 1024 tokens")
    print(f"Default overlap: 128 tokens")
