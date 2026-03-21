"""
Unit tests for FinancialCalculatorTool.
"""
import pytest
from src.agents.calculator_tool import FinancialCalculatorTool


class TestFinancialCalculatorTool:
    """Test the calculator tool."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return FinancialCalculatorTool()

    def test_simple_calculation(self, calculator):
        """Test a simple calculation."""
        code = "2 + 2"
        result = calculator.calculate(code)

        assert result["success"] is True
        assert "4" in result["result"]
        assert result["error"] is None

    def test_percentage_change(self, calculator):
        """Test percentage change calculation."""
        code = "((31.4 - 29.9) / 29.9) * 100"
        result = calculator.calculate(code)

        assert result["success"] is True
        assert result["error"] is None

        # Result should be around 5.02%
        value = float(result["result"])
        assert 5.0 < value < 5.1

    def test_financial_calculation(self, calculator):
        """Test financial calculation with large numbers."""
        code = "(0.60 * 400) - (0.40 * 400)"
        result = calculator.calculate(code)

        assert result["success"] is True
        assert "80" in result["result"]

    def test_unsafe_code_detection(self, calculator):
        """Test that unsafe code is rejected."""
        unsafe_codes = [
            "import os",
            "import sys",
            "eval('malicious')",
            "exec('malicious')",
            "open('file.txt')",
        ]

        for code in unsafe_codes:
            result = calculator.calculate(code)
            assert result["success"] is False
            assert "unsafe" in result["error"].lower()

    def test_calculation_history(self, calculator):
        """Test calculation history tracking."""
        calculator.calculate("1 + 1")
        calculator.calculate("2 * 3")

        history = calculator.get_calculation_history()

        assert len(history) == 2
        assert history[0]["code"] == "1 + 1"
        assert history[1]["code"] == "2 * 3"

    def test_clear_history(self, calculator):
        """Test clearing calculation history."""
        calculator.calculate("1 + 1")
        assert len(calculator.get_calculation_history()) == 1

        calculator.clear_history()
        assert len(calculator.get_calculation_history()) == 0

    def test_division_by_zero(self, calculator):
        """Test error handling for division by zero."""
        code = "1 / 0"
        result = calculator.calculate(code)

        assert result["success"] is False
        assert result["error"] is not None

    def test_langchain_tool_creation(self, calculator):
        """Test LangChain tool creation."""
        tool = calculator.create_langchain_tool()

        assert tool.name == "financial_calculator"
        assert callable(tool.func)
