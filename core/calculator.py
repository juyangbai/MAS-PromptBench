"""The calculator tool of the MATH and GPQA runners.

A tool's docstring is the description the model sees, so every wording the
runners use is kept verbatim, indentation included. The runner applies its
framework's wrapper: ``tool(make_calculator(DOC))`` for LangChain,
``crewai.tools.tool("calculator")(...)`` for CrewAI, the bare function for
AutoGen, and :func:`evaluate` with :data:`CALCULATOR_DESCRIPTION` /
:data:`CALCULATOR_PARAMETERS` for the OpenAI Agents SDK.
"""

from __future__ import annotations

import math
from collections.abc import Callable

CALCULATOR_DOC = (
    "Evaluate a numeric Python expression (arithmetic + math functions).\n\n"
    "    Supports +, -, *, /, **, parentheses, and math functions (sqrt, log, log10,\n"
    "    log2, exp, sin, cos, tan, asin, acos, atan, floor, ceil, pow, pi, e).\n"
    '    Example: calculator("(4/3) * pi * 2**3")\n'
    "    "
)
CALCULATOR_DOC_NARROW = (
    "Evaluate a numeric Python expression (arithmetic + math functions).\n\n"
    "    Supports +, -, *, /, **, parentheses, and math functions (sqrt, log,\n"
    "    log10, log2, exp, sin, cos, tan, asin, acos, atan, floor, ceil, pow,\n"
    '    pi, e). Example: calculator("(4/3) * pi * 2**3")\n'
    "    "
)
CALCULATOR_DOC_DECIMAL_PI = (
    "Evaluate a numeric Python expression (arithmetic + math functions).\n\n"
    "    Supports +, -, *, /, **, parentheses, and math functions (sqrt, log,\n"
    "    log10, log2, exp, sin, cos, tan, asin, acos, atan, floor, ceil, pow,\n"
    '    pi, e). Example: calculator("(4/3) * 3.14159 * 2**3")\n'
    "    "
)
CALCULATOR_DOC_BRIEF = (
    "Evaluate a numeric Python expression (arithmetic + math functions\n"
    '    like sqrt, log, sin, pi, e). Example: calculator("(4/3) * pi * 2**3")\n'
    "    "
)
# The OpenAI Agents SDK tool's description and JSON parameters.
CALCULATOR_DESCRIPTION = (
    "Evaluate a numeric Python expression with arithmetic and the native allow-listed math functions."
)
CALCULATOR_PARAMETERS = {
    "type": "object",
    "properties": {"expression": {"type": "string"}},
    "required": ["expression"],
}

_MATH_NAMES = (
    "sqrt", "log", "log10", "log2", "exp", "sin", "cos", "tan", "asin", "acos", "atan", "floor", "ceil", "pow", "pi", "e",
)  # fmt: skip


def evaluate(expression: str) -> str:
    """Evaluate ``expression`` with the allow-listed ``math`` names and no builtins."""
    allowed = {name: getattr(math, name) for name in _MATH_NAMES}
    allowed["__builtins__"] = {}
    try:
        return str(eval(expression, allowed))
    except Exception as e:
        return f"ERROR: {e}"


def make_calculator(doc: str) -> Callable[[str], str]:
    """A ``calculator(expression)`` function documented by ``doc``, ready for a framework's tool wrapper."""

    def calculator(expression: str) -> str:
        return evaluate(expression)

    calculator.__doc__ = doc
    return calculator
