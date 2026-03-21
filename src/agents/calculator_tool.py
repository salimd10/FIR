"""
Calculator tool using Python REPL for multi-step financial calculations.
Ensures LLM doesn't perform mental math.
"""
from typing import Dict, Any, Optional, List
from langchain_core.tools import Tool
from loguru import logger
import re
import io
import sys


class _SimplePythonREPL:
    """Minimal Python REPL replacement for langchain_experimental.utilities.PythonREPL."""

    def run(self, code: str) -> str:
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer
        try:
            result = eval(code)  # noqa: S307
            sys.stdout = old_stdout
            if result is not None:
                return str(result)
            return buffer.getvalue()
        except SyntaxError:
            sys.stdout = old_stdout
            exec(code)  # noqa: S102
            return buffer.getvalue()
        finally:
            sys.stdout = old_stdout


class FinancialCalculatorTool:
    """
    Safe Python REPL for financial calculations.
    LLM extracts numbers and generates calculation code.
    """

    def __init__(self):
        """Initialize the calculator tool."""
        self.python_repl = _SimplePythonREPL()
        self.logger = logger.bind(module="calculator_tool")
        self.calculation_history = []

    def calculate(self, python_code: str) -> Dict[str, Any]:
        """
        Execute Python code for calculations.

        Args:
            python_code: Python code to execute

        Returns:
            Dictionary with result and execution details
        """
        self.logger.info(f"Executing calculation code")

        try:
            # Security check - only allow safe operations
            if not self._is_safe_code(python_code):
                return {
                    "success": False,
                    "error": "Code contains potentially unsafe operations",
                    "result": None
                }

            # Execute the code
            result = self.python_repl.run(python_code)

            # Log the calculation
            calculation_record = {
                "code": python_code,
                "result": str(result),
                "success": True
            }
            self.calculation_history.append(calculation_record)

            return {
                "success": True,
                "result": str(result).strip(),
                "code": python_code,
                "error": None
            }

        except Exception as e:
            self.logger.error(f"Calculation error: {str(e)}")
            return {
                "success": False,
                "result": None,
                "code": python_code,
                "error": str(e)
            }

    def _is_safe_code(self, code: str) -> bool:
        """
        Check if code is safe to execute.

        Args:
            code: Python code

        Returns:
            True if safe, False otherwise
        """
        # Blacklist dangerous operations
        dangerous_patterns = [
            r'import\s+os',
            r'import\s+sys',
            r'import\s+subprocess',
            r'__import__',
            r'eval\s*\(',
            r'exec\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                self.logger.warning(f"Unsafe pattern detected: {pattern}")
                return False

        return True

    def create_langchain_tool(self) -> Tool:
        """
        Create a LangChain Tool instance.

        Returns:
            LangChain Tool
        """
        return Tool(
            name="financial_calculator",
            description=(
                "A Python REPL for performing financial calculations. "
                "Use this tool whenever you need to calculate percentages, "
                "differences, ratios, or any mathematical operations. "
                "Input should be valid Python code that performs the calculation. "
                "Example: '((31.4 - 29.9) / 29.9) * 100' for percentage change."
            ),
            func=lambda code: self.calculate(code)["result"]
        )

    def get_calculation_history(self) -> List[Dict[str, Any]]:
        """
        Get history of calculations performed.

        Returns:
            List of calculation records
        """
        return self.calculation_history

    def clear_history(self):
        """Clear calculation history."""
        self.calculation_history = []


class FinancialCalculationPrompts:
    """
    Prompts for guiding LLM to use calculator tool correctly.
    """

    @staticmethod
    def get_calculation_instructions() -> str:
        """
        Get instructions for LLM on how to use calculator.

        Returns:
            Instruction string
        """
        return """
When you need to perform any mathematical calculation:

1. **Extract** the relevant numbers from the context
2. **Identify** the calculation needed (percentage change, difference, ratio, etc.)
3. **Generate** Python code to perform the calculation
4. **Use** the financial_calculator tool with your Python code
5. **Interpret** the result in your answer

**Important Rules:**
- NEVER perform mental math or approximate calculations
- ALWAYS use the calculator for any arithmetic
- Show your calculation steps clearly
- Include units (dollars, percentages, etc.) in your final answer

**Example for percentage change:**
If R&D was $29.9B in 2024 and $31.4B in 2025:

Step 1: Extract values
- R&D_2024 = 29.9  # in billions
- R&D_2025 = 31.4  # in billions

Step 2: Generate calculation code
```python
((31.4 - 29.9) / 29.9) * 100
```

Step 3: Use calculator tool with the code

Step 4: Result: 5.02%

Step 5: Format answer: "R&D expenses increased by 5.02% from FY2024 to FY2025."
"""

    @staticmethod
    def get_common_calculations() -> Dict[str, str]:
        """
        Get templates for common financial calculations.

        Returns:
            Dictionary of calculation templates
        """
        return {
            "percentage_change": "((new_value - old_value) / old_value) * 100",
            "year_over_year_growth": "((current_year - previous_year) / previous_year) * 100",
            "difference": "value1 - value2",
            "ratio": "numerator / denominator",
            "percentage_of_total": "(part / total) * 100",
            "compound_growth_rate": "(((end_value / start_value) ** (1 / num_years)) - 1) * 100",
            "margin": "((revenue - cost) / revenue) * 100",
        }


# Convenience functions
def create_calculator_tool() -> FinancialCalculatorTool:
    """
    Create a calculator tool instance.

    Returns:
        FinancialCalculatorTool instance
    """
    return FinancialCalculatorTool()


if __name__ == "__main__":
    # Test the calculator
    calc = FinancialCalculatorTool()

    # Test percentage change
    code = "((31.4 - 29.9) / 29.9) * 100"
    result = calc.calculate(code)
    print(f"Result: {result}")

    # Test difference
    code2 = "(0.60 * 400) - (0.40 * 400)"
    result2 = calc.calculate(code2)
    print(f"Result2: {result2}")

    print(f"\nCalculation history: {calc.get_calculation_history()}")
