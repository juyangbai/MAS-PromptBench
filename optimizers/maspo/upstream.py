"""Definitions taken verbatim from the upstream MASPO reference implementation.

Source: https://github.com/wangzx1219/MASPO (commit
``e79aa8ea00b14c1a28f5c13500676459456a3670``), the release accompanying
"MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems"
(arXiv:2605.06623). Credit for these definitions belongs to the MASPO authors;
the upstream repository ships no license file, so no license is asserted here.

Only what the port uses is reproduced: the ``TaskType`` enum (``config.py``)
and the MATH entries of ``PROMPT_OPTIMIZE_TEMPLATE`` and
``INTERMEDIATE_COMPARE_TEMPLATE`` (``prompts.py``). The strings are unchanged.
"""
from __future__ import annotations

from enum import Enum


class TaskType(Enum):
    MATH = "math"
    MATH_CHOICE = "math_choice"
    REASONING_CHOICE = "reasoning_choice"
    CODE = "code"


PROMPT_OPTIMIZE_TEMPLATE = {
    TaskType.MATH: """
You are optimizing a prompt for a specific agent in a multi-agent mathematical reasoning system.
CRITICAL: The agent's core role and responsibilities MUST be preserved in the optimized prompt.

Agent Type: {agent_type}
Current System Role: {role_description}

Sample Execution Traces (Question + Context + Agent Output)
```
{samples}
```
Requirements:
```
{requirements}
```
Reference prompt:
```
{prompt}
```
Provide your analysis, optimization points, and the complete optimized prompt using the following XML format:
<analyse>Analyse what drawbacks exist in the results produced by the reference prompt and how to improve them.</analyse>
<modification>One sentence summary of the key improvement</modification>
<prompt>Provide the complete optimized prompt</prompt>
""",
}

INTERMEDIATE_COMPARE_TEMPLATE = {
    TaskType.MATH: """
You are comparing two intermediate outputs from an agent in a multi-agent mathematical reasoning system.
Your task is to decide which output is more likely to help the system eventually produce the CORRECT FINAL ANSWER.

Prefer the output that:
- Contains mathematically accurate reasoning (no factual/logical errors)
- Clearly states intermediate results or assumptions
- Avoids misleading statements or ambiguous conclusions
- Provides enough detail for downstream agents to verify or build upon

Problem: {question}
Output A:
{output_a}
Output B:
{output_b}

Which output is more conducive to obtaining the correct final answer? Respond ONLY with "A" or "B".
""",
}

__all__ = ["INTERMEDIATE_COMPARE_TEMPLATE", "PROMPT_OPTIMIZE_TEMPLATE", "TaskType"]
