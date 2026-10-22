"""``core.calculator``: the calculator tool of the MATH and GPQA runners."""

import pytest
from langchain_core.tools import tool

from core import calculator


@pytest.mark.parametrize(
    "doc",
    [
        calculator.CALCULATOR_DOC,
        calculator.CALCULATOR_DOC_NARROW,
        calculator.CALCULATOR_DOC_DECIMAL_PI,
        calculator.CALCULATOR_DOC_BRIEF,
    ],
)
def test_tool_description_is_its_docstring(doc):
    wrapped = tool(calculator.make_calculator(doc))
    assert wrapped.name == "calculator"
    assert wrapped.description == doc.rstrip()
    assert wrapped.invoke({"expression": "2 ** 3"}) == "8"


def test_evaluate_allows_only_math_names():
    assert calculator.evaluate("sqrt(16) + 1") == "5.0"
    assert calculator.evaluate("floor(pi)") == "3"
    assert calculator.evaluate("open('x')").startswith("ERROR:")
    assert calculator.evaluate("__import__('os')").startswith("ERROR:")
