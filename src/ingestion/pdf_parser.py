"""
Layout-aware PDF parser for financial documents.
Uses unstructured.io for high-resolution parsing with table preservation.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import pdfplumber
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Element,
    Table,
    Title,
    NarrativeText,
    ListItem,
    Footer,
    Header
)
from loguru import logger
import json


class FinancialPDFParser:
    """
    Advanced PDF parser specifically designed for financial documents.
    Handles complex layouts, nested tables, and footnotes.
    """

    def __init__(self):
        """Initialize the PDF parser."""
        self.logger = logger.bind(module="pdf_parser")

    def parse_document(
        self,
        pdf_path: Path,
        strategy: str = "hi_res"
    ) -> Dict[str, Any]:
        """
        Parse a PDF document with layout awareness.

        Args:
            pdf_path: Path to the PDF file
            strategy: Parsing strategy ('hi_res', 'fast', 'ocr_only')

        Returns:
            Dictionary containing parsed elements with metadata
        """
        self.logger.info(f"Parsing PDF: {pdf_path} with strategy: {strategy}")

        try:
            # Use unstructured.io for layout-aware parsing
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy=strategy,
                infer_table_structure=True,
                include_page_breaks=True,
                extract_images_in_pdf=False,  # We focus on text and tables
                languages=["eng"],
            )

            # Extract metadata using pdfplumber for precise page info
            pdf_metadata = self._extract_metadata(pdf_path)

            # Process elements into structured format
            structured_content = self._process_elements(elements, pdf_path)

            return {
                "metadata": pdf_metadata,
                "content": structured_content,
                "total_elements": len(elements),
                "tables_found": sum(1 for e in elements if isinstance(e, Table)),
            }

        except Exception as e:
            self.logger.error(f"Error parsing PDF {pdf_path}: {str(e)}")
            raise

    def _extract_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract document metadata using pdfplumber.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with document metadata
        """
        with pdfplumber.open(pdf_path) as pdf:
            return {
                "filename": pdf_path.name,
                "total_pages": len(pdf.pages),
                "pdf_metadata": pdf.metadata or {},
            }

    def _process_elements(
        self,
        elements: List[Element],
        pdf_path: Path
    ) -> List[Dict[str, Any]]:
        """
        Process unstructured elements into structured format.

        Args:
            elements: List of parsed elements from unstructured
            pdf_path: Path to PDF file

        Returns:
            List of structured content dictionaries
        """
        structured_content = []
        current_page = 1
        current_section = None

        for idx, element in enumerate(elements):
            # Get element metadata
            metadata = element.metadata.to_dict() if hasattr(element, 'metadata') else {}

            # Update page number if available
            if metadata.get('page_number'):
                current_page = metadata['page_number']

            # Update section from titles
            if isinstance(element, Title):
                current_section = str(element)

            # Create structured element
            structured_element = {
                "id": f"element_{idx}",
                "type": element.category if hasattr(element, 'category') else type(element).__name__,
                "content": str(element),
                "page_number": current_page,
                "section": current_section,
                "metadata": metadata,
            }

            # Special handling for tables
            if isinstance(element, Table):
                structured_element["is_table"] = True
                structured_element["table_data"] = self._extract_table_structure(
                    element,
                    pdf_path,
                    current_page
                )
            else:
                structured_element["is_table"] = False

            structured_content.append(structured_element)

        return structured_content

    def _extract_table_structure(
        self,
        table_element: Element,
        pdf_path: Path,
        page_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Extract detailed table structure using pdfplumber as fallback.

        Args:
            table_element: Table element from unstructured
            pdf_path: Path to PDF file
            page_number: Current page number

        Returns:
            Dictionary with table structure or None
        """
        try:
            # Try to get HTML representation from unstructured
            table_html = None
            if hasattr(table_element.metadata, 'text_as_html'):
                table_html = table_element.metadata.text_as_html

            # Use pdfplumber for precise table extraction
            with pdfplumber.open(pdf_path) as pdf:
                if page_number <= len(pdf.pages):
                    page = pdf.pages[page_number - 1]
                    tables = page.extract_tables()

                    if tables:
                        # Take the first table on the page (can be enhanced)
                        table_data = tables[0]

                        return {
                            "rows": len(table_data),
                            "columns": len(table_data[0]) if table_data else 0,
                            "data": table_data,
                            "html": table_html,
                        }

            return None

        except Exception as e:
            self.logger.warning(f"Could not extract table structure: {str(e)}")
            return None

    def extract_tables_from_page(
        self,
        pdf_path: Path,
        page_number: int
    ) -> List[List[List[str]]]:
        """
        Extract all tables from a specific page.

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)

        Returns:
            List of tables (each table is a list of rows)
        """
        with pdfplumber.open(pdf_path) as pdf:
            if page_number <= len(pdf.pages):
                page = pdf.pages[page_number - 1]
                return page.extract_tables()
        return []

    def get_page_text(
        self,
        pdf_path: Path,
        page_number: int
    ) -> str:
        """
        Extract raw text from a specific page.

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)

        Returns:
            Raw text content
        """
        with pdfplumber.open(pdf_path) as pdf:
            if page_number <= len(pdf.pages):
                page = pdf.pages[page_number - 1]
                return page.extract_text() or ""
        return ""

    def detect_footnotes(
        self,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect and mark footnotes in the document.

        Args:
            elements: List of structured elements

        Returns:
            Updated elements with footnote markers
        """
        for element in elements:
            content = element.get("content", "").strip()

            # Simple heuristic for footnote detection
            # Footnotes usually start with numbers or symbols and are shorter
            if element.get("type") == "Footer" or (
                len(content) < 200 and
                content and
                content[0].isdigit() or content.startswith("*")
            ):
                element["is_footnote"] = True
            else:
                element["is_footnote"] = False

        return elements


# Convenience functions
def parse_financial_document(pdf_path: Path) -> Dict[str, Any]:
    """
    Quick function to parse a financial PDF document.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Parsed document structure
    """
    parser = FinancialPDFParser()
    return parser.parse_document(pdf_path)


if __name__ == "__main__":
    # Test the parser
    import sys

    if len(sys.argv) > 1:
        test_path = Path(sys.argv[1])
        if test_path.exists():
            result = parse_financial_document(test_path)
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"File not found: {test_path}")
    else:
        print("Usage: python pdf_parser.py <path_to_pdf>")
