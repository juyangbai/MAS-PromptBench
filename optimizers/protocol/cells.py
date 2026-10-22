"""The experiment grid: which (method, task, topology, framework, communication,
team size, model) cells exist, and how each maps to a real-runner adapter.

Table 2 runs each topology in its native framework; Tables 3-7 use LangGraph;
Tables 4/5 vary communication format / team size on the application tasks;
Table 6 adds the four extra methods; Table 7 repeats the application cells
with a second task model. A cell shared by several tables is one job.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from .config import COMMUNICATIONS, PROTOCOL_ID, TASK_MODELS, TEAM_SIZES, TOPOLOGIES
from .errors import UnsupportedBaselineCell
from .schema import CellSpec

# Config
TASKS = ("gpqa", "hotpotqa", "math", "lcb", "apps", "swe", "bfcl", "toolhop", "apibank")
APPLICATION_TASKS = ("hotpotqa", "lcb", "bfcl")
PRIMARY_METHODS = ("gepa", "mipro", "mapro", "maspo")
ALL_METHODS = ("gepa", "mipro", "hivemind", "mapro", "mamut_gepa", "maspo", "maspob", "tavo")
BASELINE_METHODS = ("hivemind", "mamut_gepa", "mapro", "maspo", "maspob", "tavo")
SINGLE_AGENT_METHODS = ("gepa", "mipro")
MULTI_TOPOLOGIES = TOPOLOGIES[1:]
QWEN_MODEL = TASK_MODELS["qwen"]
LLAMA_MODEL = TASK_MODELS["llama"]
NATIVE_FRAMEWORK = {
    "single": "langgraph",
    "independent": "langgraph",
    "sequential": "crewai",
    "centralized": "autogen",
    "decentralized": "openai_agents",
}
FRAMEWORK_KEYS = {
    ("sequential", "crewai"): "sequential_crewai",
    ("centralized", "autogen"): "centralized_autogen",
    ("decentralized", "openai_agents"): "decentralized_openai_agents",
}
EXPECTED_COUNTS = {
    "qwen_configurations": 558,
    "qwen_jobs": 1674,
    "runner_conditions_per_seed": 132,
    "runner_conditions_all_seeds": 396,
    "llama_additional_configurations": 54,
    "llama_additional_jobs": 162,
    "all_configurations": 612,
    "all_jobs": 1836,
}


@dataclass(frozen=True, order=True)
class GridCell:
    """One configuration of the experiment grid and the tables that report it."""

    method: str
    task: str
    topology: str
    framework: str
    communication: str
    team_size: int
    task_model: str = QWEN_MODEL
    source_tables: tuple[int, ...] = ()

    @property
    def identity(self) -> tuple[object, ...]:
        """Method and runtime condition, without the source tables."""
        return (
            self.method,
            self.task,
            self.topology,
            self.framework,
            self.communication,
            self.team_size,
            self.task_model,
        )

    @property
    def runner_identity(self) -> tuple[object, ...]:
        """Method-independent runtime condition (shared seed-bundle baseline)."""
        return (self.task, self.topology, self.framework, self.communication, self.team_size, self.task_model)

    @property
    def registry_key(self) -> str:
        """The bridge registry key of the runtime condition."""
        return registry_key(self.topology, self.framework, self.communication, self.team_size)


def _supports(method: str, topology: str) -> bool:
    return topology != "single" or method in SINGLE_AGENT_METHODS


def build_grid(include_llama: bool = False) -> list[GridCell]:
    """All Qwen cells; with ``include_llama`` also the Table-7 Llama cells."""
    cells: dict[tuple, GridCell] = {}

    def add(
        table,
        methods: Iterable[str],
        tasks: Iterable[str],
        topologies: Iterable[str],
        framework_for,
        communications: Iterable[str] = ("freeform",),
        team_sizes: Iterable[int] = (4,),
        model: str = QWEN_MODEL,
    ) -> None:
        for method in methods:
            for task in tasks:
                for topology in topologies:
                    if not _supports(method, topology):
                        continue
                    for communication in communications:
                        for size in team_sizes:
                            actual_size = 1 if topology == "single" else size
                            cell = GridCell(
                                method,
                                task,
                                topology,
                                framework_for(topology),
                                communication,
                                actual_size,
                                model,
                                (table,),
                            )
                            prior = cells.get(cell.identity)
                            if prior is None:
                                cells[cell.identity] = cell
                            else:
                                tables = tuple(sorted(set(prior.source_tables + (table,))))
                                cells[cell.identity] = replace(prior, source_tables=tables)

    add(2, PRIMARY_METHODS, TASKS, TOPOLOGIES, lambda t: NATIVE_FRAMEWORK[t])
    add(3, PRIMARY_METHODS, TASKS, TOPOLOGIES, lambda _: "langgraph")
    add(4, PRIMARY_METHODS, APPLICATION_TASKS, MULTI_TOPOLOGIES, lambda _: "langgraph", COMMUNICATIONS)
    add(5, PRIMARY_METHODS, APPLICATION_TASKS, MULTI_TOPOLOGIES, lambda _: "langgraph", ("freeform",), TEAM_SIZES)
    add(6, ALL_METHODS, APPLICATION_TASKS, TOPOLOGIES, lambda _: "langgraph")
    add(7, PRIMARY_METHODS, APPLICATION_TASKS, TOPOLOGIES, lambda _: "langgraph")
    if include_llama:
        add(7, PRIMARY_METHODS, APPLICATION_TASKS, TOPOLOGIES, lambda _: "langgraph", model=LLAMA_MODEL)
    return sorted(cells.values())


def grid_assertions() -> dict[str, int]:
    """Check the deduplicated grid sizes against the frozen counts."""
    qwen = build_grid(False)
    full = build_grid(True)
    runners = {cell.runner_identity for cell in qwen}
    values = {
        "qwen_configurations": len(qwen),
        "qwen_jobs": len(qwen) * 3,
        "runner_conditions_per_seed": len(runners),
        "runner_conditions_all_seeds": len(runners) * 3,
        "llama_additional_configurations": len(full) - len(qwen),
        "llama_additional_jobs": (len(full) - len(qwen)) * 3,
        "all_configurations": len(full),
        "all_jobs": len(full) * 3,
    }
    if values != EXPECTED_COUNTS:
        raise AssertionError(f"grid mismatch: {values} != {EXPECTED_COUNTS}")
    return values


def find_cell(
    method: str, task: str, topology: str, framework: str, communication: str, team_size: int, task_model: str
) -> GridCell | None:
    """The grid cell with exactly this identity, or None."""
    key = (method, task, topology, framework, communication, 1 if topology == "single" else team_size, task_model)
    for cell in build_grid(include_llama=True):
        if cell.identity == key:
            return cell
    return None


def runner_condition_in_grid(
    task: str, topology: str, framework: str, communication: str, team_size: int, task_model: str
) -> bool:
    """Whether any method's grid cell uses this runtime condition."""
    key = (task, topology, framework, communication, 1 if topology == "single" else team_size, task_model)
    return any(cell.runner_identity == key for cell in build_grid(include_llama=True))


def required_cells(method: str) -> tuple[GridCell, ...]:
    """Every grid cell (Qwen and Llama) assigned to one method of :data:`BASELINE_METHODS`."""
    if method not in BASELINE_METHODS:
        raise KeyError(f"unknown baseline method {method!r}")
    return tuple(cell for cell in build_grid(True) if cell.method == method)


def _grid_key(value) -> tuple[object, ...]:
    return (
        value.method,
        value.task,
        value.topology,
        value.framework,
        value.communication,
        value.team_size,
        value.task_model,
    )


def validate_cell(method: str, cell: CellSpec) -> None:
    """Fail before any model call unless ``cell`` is exactly one of ``method``'s grid cells."""
    if cell.protocol_id != PROTOCOL_ID:
        raise UnsupportedBaselineCell(f"{method} requires protocol_id={PROTOCOL_ID!r}, got {cell.protocol_id!r}")
    if cell.method != method:
        raise UnsupportedBaselineCell(f"expected method={method!r}, got {cell.method!r}")
    if _grid_key(cell) not in {_grid_key(candidate) for candidate in required_cells(method)}:
        raise UnsupportedBaselineCell(
            "cell is not in the experiment grid: "
            f"task={cell.task}, topology={cell.topology}, framework={cell.framework}, "
            f"communication={cell.communication}, team_size={cell.team_size}, task_model={cell.task_model}"
        )


def in_grid(cell: CellSpec) -> bool:
    """Whether ``cell`` is exactly one of its method's grid cells."""
    return _grid_key(cell) in {
        _grid_key(candidate) for candidate in build_grid(True) if candidate.method == cell.method
    }


# Mapping to the real-runner registry
def registry_key(topology: str, framework: str, communication: str = "freeform", team_size: int = 4) -> str:
    """The ``optimizers.bridge.registry`` topology key for one runtime condition."""
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topology!r}")
    if framework != "langgraph":
        key = FRAMEWORK_KEYS.get((topology, framework))
        if key is None:
            raise ValueError(f"no {framework!r} runtime for topology {topology!r}")
        if communication != "freeform" or team_size != 4:
            raise ValueError("communication and team-size variants exist only for LangGraph runtimes")
        return key
    if communication != "freeform":
        if topology == "single":
            raise ValueError("the single topology has no communication variants")
        return f"{topology}_communications_{communication}"
    if topology != "single" and team_size != 4:
        return f"{topology}_r{team_size}"
    return topology


def parse_registry_key(key: str) -> dict[str, object]:
    """Inverse of :func:`registry_key`; a bare topology means the LangGraph default."""
    for (topology, framework), name in FRAMEWORK_KEYS.items():
        if key == name:
            return {"topology": topology, "framework": framework}
    for topology in MULTI_TOPOLOGIES:
        prefix = f"{topology}_communications_"
        if key.startswith(prefix) and key[len(prefix) :] in COMMUNICATIONS:
            return {"topology": topology, "framework": "langgraph", "communication": key[len(prefix) :]}
        if key.startswith(f"{topology}_r") and key[len(topology) + 2 :].isdigit():
            return {"topology": topology, "framework": "langgraph", "team_size": int(key[len(topology) + 2 :])}
    if key in TOPOLOGIES:
        return {"topology": key}
    raise ValueError(f"unknown topology or registry key {key!r}")


__all__ = [
    "ALL_METHODS",
    "APPLICATION_TASKS",
    "BASELINE_METHODS",
    "EXPECTED_COUNTS",
    "GridCell",
    "LLAMA_MODEL",
    "NATIVE_FRAMEWORK",
    "PRIMARY_METHODS",
    "QWEN_MODEL",
    "SINGLE_AGENT_METHODS",
    "TASKS",
    "build_grid",
    "find_cell",
    "grid_assertions",
    "in_grid",
    "parse_registry_key",
    "registry_key",
    "required_cells",
    "runner_condition_in_grid",
    "validate_cell",
]
