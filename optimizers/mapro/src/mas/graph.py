"""Multi-agent system as a DAG.

An `Agent` carries a `base_prompt` (its role instruction — this is what MAPRO
optimizes). `MASGraph` wires agents with directed `Edge`s that encode
information flow (parent output -> child input). The task input is global
context available to every agent; edges represent *inter-agent* dependencies
only (matching the paper's joint score T(P) = ∏ g(p_i) · ∏_{(i,j)∈E} g(p_i,p_j)).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Agent:
    """One agent: its role and the base prompt MAPRO optimizes."""

    id: str
    role: str  # short human label, e.g. "solver", "reflector"
    base_prompt: str  # seed role instruction; candidate pool derives from this
    reasoning_style: str = ""  # DMAD debaters: "cot" | "stepback" | "pot"; "" otherwise


@dataclass(frozen=True)
class Edge:
    """Directed information flow from ``src``'s output to ``dst``'s input."""

    src: str
    dst: str


@dataclass
class MASGraph:
    """Agents wired as a DAG, with the agent whose output is the system answer."""

    agents: dict[str, Agent]
    edges: list[Edge]
    output_agent: str  # whose output is the system's final answer
    name: str = "mas"
    task_preamble: str = ""  # optional shared instruction prepended to every task

    def __post_init__(self) -> None:
        ids = set(self.agents)
        for e in self.edges:
            if e.src not in ids or e.dst not in ids:
                raise ValueError(f"edge {e} references unknown agent")
        if self.output_agent not in ids:
            raise ValueError(f"output_agent {self.output_agent!r} not in agents")
        self._parents: dict[str, list[str]] = {a: [] for a in ids}
        self._children: dict[str, list[str]] = {a: [] for a in ids}
        for e in self.edges:
            self._parents[e.dst].append(e.src)
            self._children[e.src].append(e.dst)

    # --- structure queries ---
    @property
    def agent_ids(self) -> list[str]:
        return list(self.agents.keys())

    def parents(self, aid: str) -> list[str]:
        return list(self._parents[aid])

    def children(self, aid: str) -> list[str]:
        return list(self._children[aid])

    def topo_order(self) -> list[str]:
        """Kahn's algorithm; deterministic (insertion order tie-break)."""
        indeg = {a: len(self._parents[a]) for a in self.agents}
        ready = [a for a in self.agents if indeg[a] == 0]
        order: list[str] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for c in self._children[n]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
        if len(order) != len(self.agents):
            raise ValueError("MASGraph is not a DAG (cycle detected)")
        return order
