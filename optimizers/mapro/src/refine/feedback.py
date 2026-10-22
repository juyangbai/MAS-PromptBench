"""Topology-aware feedback collection (Stage 3).

For an executed task the MAPRO loop (``native``) produces:
  * global feedback f_g  — did the system succeed, and if not how (expected vs got);
  * local feedback f_l   — *blames*: traversing the graph in reverse topological
    order, each agent critiques the upstream parent(s) whose output hindered it.
    Blame directed at parent i is accumulated into f_l[i].
This module holds the blame prompt, its parser and the feedback aggregation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm import GenerationConfig
from ..mas.graph import MASGraph

_BLAME_SYS = (
    "You perform blame assignment in a multi-agent system. You are terse and output only the requested BLAME lines."
)

_BLAME_TMPL = """The multi-agent system's final result was {outcome}.

Task:
{task}

Agent '{role}' ({aid}) received the following upstream inputs and produced the
output below.

Upstream inputs:
{parents}

Agent '{role}' output:
{output}

For each upstream parent listed above, state in ONE short sentence whether and
how its output hindered agent '{role}'. If a parent's output was adequate, write
'OK'. Format each line exactly as:  BLAME <parent_id>: <one sentence>"""

_BLAME_CFG = GenerationConfig(temperature=0.2, max_tokens=256, enable_thinking=False)
_BLAME_RE = re.compile(r"BLAME\s+([^\s:]+)\s*:\s*(.+)")


def _clip(s: str, n: int = 1200) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + "…"


@dataclass
class TaskFeedback:
    """Global feedback of one task and the blame each agent received."""

    f_g: str
    blames: dict[str, str] = field(default_factory=dict)  # aid -> critique it received


def aggregate_feedback(graph: MASGraph, per_task: list[TaskFeedback], max_len: int = 800) -> tuple[str, dict[str, str]]:
    """Combine per-task feedback into one global summary + per-agent blame string."""
    n = len(per_task)
    n_fail = sum(1 for f in per_task if "INCORRECTLY" in f.f_g)
    f_g = f"Over {n} sampled tasks, {n - n_fail} succeeded and {n_fail} failed."
    blames: dict[str, str] = {}
    for aid in graph.agent_ids:
        pieces = [f.blames.get(aid, "") for f in per_task if f.blames.get(aid, "")]
        joined = " ".join(pieces).strip()
        blames[aid] = joined[:max_len] + ("…" if len(joined) > max_len else "")
    return f_g, blames
