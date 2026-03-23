"""
Layout-aware PDF parser for financial documents.
Uses pdfplumber for text and table extraction.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
import pdfplumber
from loguru import logger
import json


class FinancialPDFParser:
    """
    PDF parser for financial documents using pdfplumber.
    Extracts text elements and tables page by page, preserving
    section context and table structure.
    """

    def __init__(self):
        self.logger = logger.bind(module="pdf_parser")

    def parse_document(
        self,
        pdf_path: Path,
        strategy: str = "fast"  # kept for API compatibility, unused
    ) -> Dict[str, Any]:
        """
        Parse a PDF document into structured elements.

        Args:
            pdf_path: Path to the PDF file
            strategy: Ignored (kept for API compatibility)

        Returns:
            Dictionary with metadata, content elements, counts
        """
        self.logger.info(f"Parsing PDF: {pdf_path.name}")

        structured_content: List[Dict[str, Any]] = []
        tables_found = 0
        element_idx = 0
        current_section: Optional[str] = None

        with pdfplumber.open(pdf_path) as pdf:
            pdf_metadata = {
                "filename": pdf_path.name,
                "total_pages": len(pdf.pages),
                "pdf_metadata": pdf.metadata or {},
            }

            for page_num, page in enumerate(pdf.pages, start=1):
                # --- Extract tables first so we can skip their bounding boxes ---
                page_tables = page.extract_tables() or []
                table_bboxes = [t.bbox for t in page.find_tables()] if page_tables else []

                for table_data in page_tables:
                    if not table_data:
                        continue
                    # Convert None cells to empty strings
                    clean_data = [
                        [cell or "" for cell in row]
                        for row in table_data
                    ]
                    content_str = self._table_to_text(clean_data)
                    structured_content.append({
                        "id": f"element_{element_idx}",
                        "type": "Table",
                        "content": content_str,
                        "page_number": page_num,
                        "section": current_section,
                        "metadata": {"page_number": page_num},
                        "is_table": True,
                        "table_data": {
                            "rows": len(clean_data),
                            "columns": len(clean_data[0]) if clean_data else 0,
                            "data": clean_data,
                            "html": None,
                        },
                    })
                    element_idx += 1
                    tables_found += 1

                # --- Extract text, splitting into paragraphs ---
                raw_text = page.extract_text() or ""
                paragraphs = self._split_into_paragraphs(raw_text)

                for para in paragraphs:
                    if not para.strip():
                        continue

                    elem_type = self._classify_element(para)
                    if elem_type == "Title":
                        current_section = para.strip()

                    structured_content.append({
                        "id": f"element_{element_idx}",
                        "type": elem_type,
                        "content": para.strip(),
                        "page_number": page_num,
                        "section": current_section,
                        "metadata": {"page_number": page_num},
                        "is_table": False,
                        "table_data": None,
                    })
                    element_idx += 1

        self.logger.info(
            f"Parsed {pdf_metadata['total_pages']} pages, "
            f"{len(structured_content)} elements, {tables_found} tables"
        )

        return {
            "metadata": pdf_metadata,
            "content": structured_content,
            "total_elements": len(structured_content),
            "tables_found": tables_found,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split raw page text into paragraph-sized chunks."""
        # Split on blank lines first
        blocks = re.split(r"\n{2,}", text)
        result = []
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            # If block is very long, further split on sentence boundaries
            if len(block) > 1500:
                sentences = re.split(r"(?<=[.!?])\s+", block)
                chunk, chunks = "", []
                for s in sentences:
                    if len(chunk) + len(s) > 1000 and chunk:
                        chunks.append(chunk.strip())
                        chunk = s
                    else:
                        chunk = (chunk + " " + s).strip()
                if chunk:
                    chunks.append(chunk)
                result.extend(chunks)
            else:
                result.append(block)
        return result

    def _classify_element(self, text: str) -> str:
        """Heuristically classify a text block as Title or NarrativeText."""
        stripped = text.strip()
        lines = stripped.splitlines()
        # Title heuristics: short, no trailing period, ALL CAPS or Title Case
        if len(lines) <= 2 and len(stripped) <= 120:
            if stripped.isupper():
                return "Title"
            if stripped.istitle() and not stripped.endswith("."):
                return "Title"
            if re.match(r"^(ITEM|PART|NOTE|SECTION)\s+\d+", stripped, re.IGNORECASE):
                return "Title"
        return "NarrativeText"

    def _table_to_text(self, table_data: List[List[str]]) -> str:
        """Convert table rows to a readable text representation."""
        if not table_data:
            return ""
        lines = []
        for row in table_data:
            lines.append(" | ".join(str(cell) for cell in row))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Utility methods (kept for API compatibility)
    # ------------------------------------------------------------------

    def _extract_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        with pdfplumber.open(pdf_path) as pdf:
            return {
                "filename": pdf_path.name,
                "total_pages": len(pdf.pages),
                "pdf_metadata": pdf.metadata or {},
            }

    def extract_tables_from_page(
        self,
        pdf_path: Path,
        page_number: int
    ) -> List[List[List[str]]]:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number <= len(pdf.pages):
                return pdf.pages[page_number - 1].extract_tables() or []
        return []

    def get_page_text(self, pdf_path: Path, page_number: int) -> str:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number <= len(pdf.pages):
                return pdf.pages[page_number - 1].extract_text() or ""
        return ""

    def detect_footnotes(
        self,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        for element in elements:
            content = element.get("content", "").strip()
            element["is_footnote"] = (
                element.get("type") == "Footer"
                or (len(content) < 200 and bool(content)
                    and (content[0].isdigit() or content.startswith("*")))
            )
        return elements


# Convenience function
def parse_financial_document(pdf_path: Path) -> Dict[str, Any]:
    return FinancialPDFParser().parse_document(pdf_path)


if __name__ == "__main__":
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
