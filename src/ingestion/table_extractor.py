"""
Table extraction and conversion to Markdown format.
Preserves table structure for better LLM comprehension.
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import re


class TableToMarkdownConverter:
    """
    Converts financial tables to Markdown format while preserving structure.
    This ensures LLMs can understand table relationships and perform calculations.
    """

    def __init__(self):
        """Initialize the converter."""
        self.logger = logger.bind(module="table_converter")

    def convert_table_to_markdown(
        self,
        table_data: List[List[str]],
        has_header: bool = True,
        caption: Optional[str] = None
    ) -> str:
        """
        Convert a table (2D list) to Markdown format.

        Args:
            table_data: 2D list representing table rows and columns
            has_header: Whether first row is a header
            caption: Optional table caption

        Returns:
            Markdown formatted table string
        """
        if not table_data or not table_data[0]:
            return ""

        # Clean and normalize cell values
        cleaned_data = self._clean_table_data(table_data)

        # Build markdown table
        markdown_lines = []

        # Add caption if provided
        if caption:
            markdown_lines.append(f"**{caption}**\n")

        # Determine column widths for alignment
        num_cols = len(cleaned_data[0])

        # Add header row
        if has_header and len(cleaned_data) > 0:
            header = cleaned_data[0]
            markdown_lines.append("| " + " | ".join(header) + " |")
            markdown_lines.append("| " + " | ".join(["---"] * num_cols) + " |")
            data_rows = cleaned_data[1:]
        else:
            # Create generic header
            markdown_lines.append("| " + " | ".join([f"Col {i+1}" for i in range(num_cols)]) + " |")
            markdown_lines.append("| " + " | ".join(["---"] * num_cols) + " |")
            data_rows = cleaned_data

        # Add data rows
        for row in data_rows:
            # Ensure row has correct number of columns
            padded_row = row + [""] * (num_cols - len(row))
            markdown_lines.append("| " + " | ".join(padded_row[:num_cols]) + " |")

        return "\n".join(markdown_lines)

    def _clean_table_data(self, table_data: List[List[str]]) -> List[List[str]]:
        """
        Clean and normalize table cell values.

        Args:
            table_data: Raw table data

        Returns:
            Cleaned table data
        """
        cleaned = []

        for row in table_data:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cell_value = ""
                else:
                    cell_value = str(cell).strip()
                    # Remove excessive whitespace
                    cell_value = re.sub(r'\s+', ' ', cell_value)
                    # Escape pipe characters that would break markdown
                    cell_value = cell_value.replace('|', '\\|')

                cleaned_row.append(cell_value)
            cleaned.append(cleaned_row)

        return cleaned

    def extract_financial_values(
        self,
        table_data: List[List[str]]
    ) -> Dict[str, List[float]]:
        """
        Extract numerical financial values from table.
        Useful for pre-processing financial data.

        Args:
            table_data: Table data

        Returns:
            Dictionary mapping column names to lists of numerical values
        """
        if not table_data or len(table_data) < 2:
            return {}

        financial_data = {}
        headers = table_data[0]

        for col_idx, header in enumerate(headers):
            values = []
            for row in table_data[1:]:
                if col_idx < len(row):
                    cell_value = row[col_idx]
                    # Try to parse financial value
                    num_value = self._parse_financial_number(cell_value)
                    if num_value is not None:
                        values.append(num_value)

            if values:
                financial_data[header] = values

        return financial_data

    def _parse_financial_number(self, value: str) -> Optional[float]:
        """
        Parse a financial number from string.
        Handles formats like: $1,234.56, (1,234.56), 1.2B, etc.

        Args:
            value: String representation of number

        Returns:
            Parsed float or None
        """
        if not value or not isinstance(value, str):
            return None

        # Remove common financial formatting
        cleaned = value.strip()

        # Handle parentheses (negative numbers)
        is_negative = cleaned.startswith('(') and cleaned.endswith(')')
        if is_negative:
            cleaned = cleaned[1:-1]

        # Remove currency symbols and commas
        cleaned = re.sub(r'[$€£¥,]', '', cleaned)

        # Handle abbreviations (B for billion, M for million, K for thousand)
        multiplier = 1.0
        if cleaned.endswith('B'):
            multiplier = 1_000_000_000
            cleaned = cleaned[:-1]
        elif cleaned.endswith('M'):
            multiplier = 1_000_000
            cleaned = cleaned[:-1]
        elif cleaned.endswith('K'):
            multiplier = 1_000
            cleaned = cleaned[:-1]

        # Try to parse as float
        try:
            num_value = float(cleaned) * multiplier
            return -num_value if is_negative else num_value
        except (ValueError, AttributeError):
            return None

    def create_table_with_context(
        self,
        table_data: List[List[str]],
        section_title: Optional[str] = None,
        preceding_text: Optional[str] = None,
        following_text: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> str:
        """
        Create a table with surrounding context for better RAG retrieval.

        Args:
            table_data: Table data
            section_title: Section where table appears
            preceding_text: Text before the table
            following_text: Text after the table (e.g., footnotes)
            page_number: Page number

        Returns:
            Formatted string with table and context
        """
        parts = []

        # Add section title
        if section_title:
            parts.append(f"## {section_title}\n")

        # Add page reference
        if page_number:
            parts.append(f"*(Page {page_number})*\n")

        # Add preceding context
        if preceding_text:
            parts.append(f"{preceding_text}\n")

        # Add the table
        markdown_table = self.convert_table_to_markdown(table_data)
        parts.append(markdown_table)

        # Add following context (e.g., footnotes)
        if following_text:
            parts.append(f"\n{following_text}")

        return "\n".join(parts)

    def merge_tables_with_headers(
        self,
        tables: List[List[List[str]]],
        shared_header: Optional[List[str]] = None
    ) -> List[List[str]]:
        """
        Merge multiple tables that share the same header structure.
        Useful for multi-page tables.

        Args:
            tables: List of tables to merge
            shared_header: Optional shared header to use

        Returns:
            Merged table
        """
        if not tables:
            return []

        # Use first table's header if not provided
        if shared_header is None:
            shared_header = tables[0][0] if tables[0] else []

        merged = [shared_header]

        for table in tables:
            # Skip header row for subsequent tables
            data_rows = table[1:] if len(table) > 1 else []
            merged.extend(data_rows)

        return merged


class FinancialTableProcessor:
    """
    Higher-level processor for financial tables.
    Combines extraction, conversion, and enhancement.
    """

    def __init__(self):
        """Initialize the processor."""
        self.converter = TableToMarkdownConverter()
        self.logger = logger.bind(module="table_processor")

    def process_table_element(
        self,
        element: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a table element from parsed PDF.

        Args:
            element: Element dictionary from PDF parser

        Returns:
            Enhanced element with markdown table
        """
        if not element.get("is_table") or not element.get("table_data"):
            return element

        table_data = element["table_data"].get("data", [])
        if not table_data:
            return element

        # Convert to markdown
        markdown_table = self.converter.convert_table_to_markdown(table_data)

        # Extract financial values
        financial_values = self.converter.extract_financial_values(table_data)

        # Create enhanced content with context
        enhanced_content = self.converter.create_table_with_context(
            table_data=table_data,
            section_title=element.get("section"),
            page_number=element.get("page_number")
        )

        # Update element
        element["markdown_table"] = markdown_table
        element["enhanced_content"] = enhanced_content
        element["financial_values"] = financial_values
        element["content"] = enhanced_content  # Replace original content

        return element

    def identify_key_financial_tables(
        self,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify key financial tables (income statement, balance sheet, etc.)

        Args:
            elements: List of document elements

        Returns:
            List of key financial table elements
        """
        key_tables = []
        keywords = [
            "consolidated statement",
            "balance sheet",
            "income statement",
            "cash flow",
            "operations",
            "stockholders' equity",
            "segment",
            "reconciliation",
        ]

        for element in elements:
            if not element.get("is_table"):
                continue

            section = (element.get("section") or "").lower()
            content = (element.get("content") or "").lower()

            # Check if table matches key financial statement patterns
            if any(keyword in section or keyword in content for keyword in keywords):
                element["is_key_financial_table"] = True
                key_tables.append(element)

        self.logger.info(f"Identified {len(key_tables)} key financial tables")
        return key_tables


# Convenience functions
def convert_table_to_markdown(table_data: List[List[str]]) -> str:
    """Quick function to convert table to markdown."""
    converter = TableToMarkdownConverter()
    return converter.convert_table_to_markdown(table_data)


if __name__ == "__main__":
    # Test the converter
    sample_table = [
        ["", "2025", "2024", "2023"],
        ["Net Sales", "$400.0B", "$385.0B", "$370.0B"],
        ["Cost of Sales", "$(220.0B)", "$(210.0B)", "$(200.0B)"],
        ["Gross Margin", "$180.0B", "$175.0B", "$170.0B"],
    ]

    converter = TableToMarkdownConverter()
    markdown = converter.convert_table_to_markdown(sample_table)
    print(markdown)
    print("\n" + "="*50 + "\n")

    # Test financial value extraction
    values = converter.extract_financial_values(sample_table)
    print("Extracted values:", values)
