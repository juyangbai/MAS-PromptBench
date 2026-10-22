"""Inventory of golden cells.

A cell id is a logical name (independent of file layout); ``locate`` maps it
to the code that implements it today. Inventory is derived from the file tree
and the bridge registry (listed once per optimizer), so a refactor that drops
or adds a runner shows up in ``test_inventory``.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASETS = ("gpqa", "hotpotqa", "math", "lcb", "apps", "bfcl", "swe", "apibank", "toolhop")
OPTIMIZERS = ("gepa", "mipro")
TEAM_SIZES = (2, 4, 8, 10)
FORMATS = ("freeform", "semi_structured", "structured_soft")

AREAS = (
    "topologies",
    "teamsizes",
    "communications",
    "registry",
    "prompts",
    "datasets",
    "cli",
    "scorers",
    "static",
    "methods",
)

# Cells run by ``pytest -m golden_fast``: one per framework / area shape.
FAST_CELLS = (
    "topologies/single/langgraph/gpqa",
    "topologies/independent/langgraph/bfcl",
    "topologies/sequential/crewai/hotpotqa",
    "topologies/sequential/langgraph/toolhop",
    "topologies/centralized/autogen/math",
    "topologies/centralized/langgraph/lcb",
    "topologies/decentralized/langgraph/apibank",
    "topologies/decentralized/openai_agents/gpqa",
    "teamsizes/centralized/hotpotqa/r2",
    "communications/sequential/bfcl/structured_soft",
    "registry/gepa/gpqa/single",
    "registry/gepa/hotpotqa/centralized_autogen",
    "prompts/gepa/hotpotqa",
    "datasets/gepa/gpqa",
    "datasets/gepa/hotpotqa",
    "cli/topologies/single",
    "scorers/gpqa",
    "scorers/communications",
    "static/decoding",
)


@cache
def first_eval_id(dataset: str) -> str:
    manifest = json.loads((REPO / "benchmarks" / dataset / f"{dataset}_eval_ids.json").read_text())
    return str(manifest["ids"][0])


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def module_name(path: Path) -> str:
    return _rel(path)[:-3].replace("/", ".")


# ---------------------------------------------------------------- locators
def topology_path(topology: str, framework: str, dataset: str) -> Path:
    if topology in ("single", "independent"):
        return REPO / "topologies" / topology / dataset / f"{framework}_{dataset}.py"
    return REPO / "topologies" / topology / framework / dataset / f"{framework}_{dataset}.py"


def teamsize_path(topology: str, dataset: str, size: int) -> Path:
    return REPO / "teamsizes" / topology / dataset / f"{dataset}_r{size}.py"


def communication_path(topology: str, dataset: str, fmt: str) -> Path:
    return REPO / "communications" / topology / dataset / f"{dataset}_{fmt}.py"


# ---------------------------------------------------------------- inventory
def topology_cells() -> list[dict]:
    cells = []
    for path in sorted((REPO / "topologies").glob("*/*/*.py")) + sorted((REPO / "topologies").glob("*/*/*/*.py")):
        rel = path.relative_to(REPO / "topologies").parts
        match = re.fullmatch(r"([a-z_]+)_([a-z]+)\.py", rel[-1])
        if not match or match.group(2) not in DATASETS or rel[-2] != match.group(2):
            continue
        framework, dataset = match.groups()
        topology = rel[0]
        if topology_path(topology, framework, dataset) != path:
            continue
        cells.append(
            {
                "id": f"topologies/{topology}/{framework}/{dataset}",
                "area": "topologies",
                "kind": "runner",
                "topology": topology,
                "framework": framework,
                "dataset": dataset,
                "path": _rel(path),
                "module": module_name(path),
            }
        )
    return cells


def teamsize_cells() -> list[dict]:
    cells = []
    for path in sorted((REPO / "teamsizes").glob("*/*/*_r*.py")):
        topology, dataset, name = path.relative_to(REPO / "teamsizes").parts
        match = re.fullmatch(rf"{dataset}_r(\d+)\.py", name)
        if not match:
            continue
        size = int(match.group(1))
        cells.append(
            {
                "id": f"teamsizes/{topology}/{dataset}/r{size}",
                "area": "teamsizes",
                "kind": "runner",
                "topology": topology,
                "framework": "langgraph",
                "dataset": dataset,
                "team_size": size,
                "path": _rel(path),
                "module": module_name(path),
            }
        )
    return cells


def communication_cells() -> list[dict]:
    cells = []
    for path in sorted((REPO / "communications").glob("*/*/*.py")):
        topology, dataset, name = path.relative_to(REPO / "communications").parts
        match = re.fullmatch(rf"{dataset}_({'|'.join(FORMATS)})\.py", name)
        if not match:
            continue
        cells.append(
            {
                "id": f"communications/{topology}/{dataset}/{match.group(1)}",
                "area": "communications",
                "kind": "communication",
                "topology": topology,
                "framework": "langgraph",
                "dataset": dataset,
                "format": match.group(1),
                "path": _rel(path),
                "module": module_name(path),
            }
        )
    return cells


@cache
def registry_keys() -> dict[str, tuple[str, ...]]:
    """Registry keys per dataset; both optimizers share the bridge registry.

    ``optimizers.bridge.registry`` imports only the adapter interface, so the
    parent process stays free of framework imports.
    """
    from optimizers.bridge.registry import DATASET_ADAPTERS

    return {ds: tuple(sorted(keys)) for ds, keys in DATASET_ADAPTERS.items()}


def _base_topology(key: str) -> str:
    return key.split("_", 1)[0]


def registry_cells() -> list[dict]:
    cells = []
    for optimizer in OPTIMIZERS:
        for dataset, keys in sorted(registry_keys().items()):
            for key in keys:
                cells.append(
                    {
                        "id": f"registry/{optimizer}/{dataset}/{key}",
                        "area": "registry",
                        "kind": "registry",
                        "optimizer": optimizer,
                        "dataset": dataset,
                        "key": key,
                        "topology": _base_topology(key),
                        "depends": [f"datasets/{optimizer}/{dataset}"],
                    }
                )
    return cells


def prompt_cells() -> list[dict]:
    return [
        {
            "id": f"prompts/{optimizer}/{dataset}",
            "area": "prompts",
            "kind": "prompts",
            "optimizer": optimizer,
            "dataset": dataset,
            "keys": list(registry_keys()[dataset]),
        }
        for optimizer in OPTIMIZERS
        for dataset in sorted(registry_keys())
    ]


def dataset_cells() -> list[dict]:
    return [
        {
            "id": f"datasets/{optimizer}/{dataset}",
            "area": "datasets",
            "kind": "dataset",
            "optimizer": optimizer,
            "dataset": dataset,
        }
        for optimizer in OPTIMIZERS
        for dataset in sorted(registry_keys())
    ]


def cli_cells() -> list[dict]:
    groups: dict[str, list[str]] = {}
    for cell in topology_cells() + teamsize_cells() + communication_cells():
        group = "/".join(cell["id"].split("/")[:2])
        groups.setdefault(group, []).append(cell["path"])
    return [
        {"id": f"cli/{group}", "area": "cli", "kind": "cli", "paths": sorted(paths)}
        for group, paths in sorted(groups.items())
    ]


def scorer_cells() -> list[dict]:
    cells = [
        {
            "id": f"scorers/{dataset}",
            "area": "scorers",
            "kind": "scorer",
            "dataset": dataset,
            "depends": [f"datasets/{optimizer}/{dataset}" for optimizer in OPTIMIZERS],
        }
        for dataset in DATASETS
    ]
    cells.append({"id": "scorers/communications", "area": "scorers", "kind": "comm_parser"})
    return cells


def static_cells() -> list[dict]:
    return [{"id": "static/decoding", "area": "static", "kind": "decoding"}]


def method_cells() -> list[dict]:
    from tests.golden.methods import method_cells as cells

    return cells()


def all_cells() -> list[dict]:
    cells = (
        topology_cells()
        + teamsize_cells()
        + communication_cells()
        + dataset_cells()
        + registry_cells()
        + prompt_cells()
        + cli_cells()
        + scorer_cells()
        + static_cells()
        + method_cells()
    )
    for cell in cells:
        topology = cell.get("topology")
        cell["concurrent"] = bool(cell.get("concurrent")) or topology in ("independent", "decentralized")
        cell["fast"] = cell["id"] in FAST_CELLS
    ids = [cell["id"] for cell in cells]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate golden cell ids")
    return cells


def cells_by_id() -> dict[str, dict]:
    return {cell["id"]: cell for cell in all_cells()}
