"""Team specs: the agents of a multi-agent topology at team size r.

``configs/teams/<dataset>.yaml`` declares, per dataset and topology, the team at every
size in :data:`TEAM_SIZES`; :func:`spec` returns it as a :class:`TeamSpec`. A
topologies/ LangGraph runner builds its team from ``spec(topology, dataset)``
(r = :data:`BASE_SIZE`), and each teamsizes/ runner executes that runner with
another r (:mod:`core.variant`). The agents' recursion limit is the same at every r.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import yaml

from core.paths import CONFIGS_DIR

CONFIG_DIR = CONFIGS_DIR / "teams"
TEAM_SIZES = (2, 4, 8, 10)
BASE_SIZE = 4

_COUNT_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
_DELEGATION_NOTE = (
    "Delegation: when you want a specific worker to act, call the "
    "matching delegate_to_<worker> tool with clear instructions "
    "(instead of merely addressing them in free-form text). "
)


@dataclass(frozen=True)
class Stage:
    """One stage of a sequential pipeline."""

    role: str
    tools: tuple[str, ...]
    task: str  # user-message template, filled with the instance fields via str.format


@dataclass(frozen=True)
class Worker:
    """A centralized worker: its tools, its system-prompt suffix and the manager's ``delegate_to_<role>`` tool."""

    role: str
    delegate: str  # the delegate tool's description
    tools: tuple[str, ...] = ()
    prompt_suffix: str = ""  # appended to the worker's role prompt


@dataclass(frozen=True)
class TeamSpec:
    """The team of one (topology, dataset) at size ``size``."""

    topology: str
    dataset: str
    size: int
    recursion_limit: int | None = None  # each agent's ReAct loop (None: no agent loops)
    role: str | None = None  # independent / decentralized: the replicated role
    n_rounds: int | None = None  # decentralized
    stages: tuple[Stage, ...] = ()  # sequential
    manager: str | None = None  # centralized: the manager's prompt key
    workers: tuple[Worker, ...] = ()  # centralized
    manager_tools: tuple[str, ...] = ()  # centralized, besides the delegate tools
    max_turns: int | None = None  # centralized: manager + worker turns before the run stops

    @property
    def n_agents(self) -> int:
        """Replicas (independent) or peers (decentralized): the team size."""
        return self.size

    @property
    def roles(self) -> tuple[str, ...]:
        """Prompt roles in team order."""
        if self.stages:
            return tuple(stage.role for stage in self.stages)
        if self.manager:
            return (self.manager, *(worker.role for worker in self.workers))
        return (self.role,)

    @property
    def delegation_note(self) -> str:
        """Tells the manager to delegate through its delegate tools and names the workers."""
        names = [worker.role for worker in self.workers]
        if len(names) == 1:
            return _DELEGATION_NOTE + f"The only worker is: {names[0]}."
        return _DELEGATION_NOTE + f"The {_COUNT_WORDS[len(names)]} workers are: {', '.join(names)}."


@cache
def _config() -> dict:
    """All team specs, merged from configs/teams/<dataset>.yaml."""
    merged: dict = {}
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        merged.update(yaml.safe_load(path.read_text()) or {})
    return merged


def defined(topology: str, dataset: str) -> bool:
    """Whether ``configs/teams/<dataset>.yaml`` declares teams for this (topology, dataset)."""
    return topology in (_config().get(dataset) or {})


@cache
def spec(topology: str, dataset: str, size: int | None = None) -> TeamSpec:
    """The team of ``topology`` on ``dataset`` at ``size`` (default :data:`BASE_SIZE`)."""
    size = BASE_SIZE if size is None else size
    if size not in TEAM_SIZES:
        raise ValueError(f"team size must be one of {TEAM_SIZES}, got {size}")
    if not defined(topology, dataset):
        raise KeyError(f"no {topology} team for {dataset!r} in configs/teams/{dataset}.yaml")
    cfg = _config()[dataset][topology]
    team = TeamSpec(topology, dataset, size, cfg.get("recursion_limit"), **_FIELDS[topology](cfg, size))
    if topology in ("sequential", "centralized") and len(team.roles) != size:
        raise ValueError(f"{topology}/{dataset} r={size} lists {len(team.roles)} agents")
    return team


def _replicated(cfg: dict, size: int) -> dict:
    return {"role": cfg["role"], "n_rounds": cfg.get("n_rounds")}


def _sequential(cfg: dict, size: int) -> dict:
    stages = cfg["sizes"][size]
    return {"stages": tuple(Stage(s["role"], tuple(s["tools"]), cfg["tasks"][s["role"]]) for s in stages)}


def _centralized(cfg: dict, size: int) -> dict:
    team = cfg["sizes"][size]
    return {
        "manager": team["manager"],
        "workers": tuple(_worker(cfg, role) for role in team["workers"]),
        "manager_tools": tuple(cfg["manager_tools"]),
        "max_turns": team["max_turns"],
    }


def _worker(cfg: dict, role: str) -> Worker:
    """Worker ``role``: its ``tools_by_worker`` entry (default ``worker_tools``) and ``prompt_suffixes`` entry."""
    tools = cfg.get("tools_by_worker", {}).get(role, cfg["worker_tools"])
    return Worker(role, cfg["delegates"][role], tuple(tools), cfg.get("prompt_suffixes", {}).get(role, ""))


_FIELDS = {
    "independent": _replicated,
    "decentralized": _replicated,
    "sequential": _sequential,
    "centralized": _centralized,
}
