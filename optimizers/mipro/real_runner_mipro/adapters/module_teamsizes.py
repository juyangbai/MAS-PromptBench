"""Team-size adapters backed by the real ``teamsizes`` modules.

These adapters intentionally live under ``optimizers/mipro`` only. They make
the real team-size runners look like normal GEPA pairs while leaving the
baseline ``teamsizes/`` implementations untouched.
"""
from __future__ import annotations

import ast
import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from real_runner_mipro.adapters.module_hotpotqa import ModuleHotpotQAAdapter
from real_runner_mipro.adapters.module_lcb import ModuleLCBAdapter
from real_runner_mipro.registry import TEAM_SIZES


SUPPORTED_TEAMSIZES_DATASETS = ("hotpotqa", "lcb")
SUPPORTED_TEAMSIZES_BASE_TOPOLOGIES = ("independent", "decentralized", "sequential", "centralized")
SUPPORTED_TEAM_SIZES = TEAM_SIZES  # single source of truth: registry.TEAM_SIZES


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def teamsizes_topology(base_topology: str, team_size: int) -> str:
    return f"{base_topology}_r{team_size}"


def teamsizes_module_name(dataset: str, base_topology: str, team_size: int) -> str:
    return f"teamsizes.{base_topology}.{dataset}.{dataset}_r{team_size}"


def class_name(dataset: str, base_topology: str, team_size: int) -> str:
    dataset_prefix = {"hotpotqa": "HotpotQATeamSizes", "lcb": "LCBTeamSizes"}[dataset]
    topo_prefix = "".join(part.title() for part in base_topology.split("_"))
    return f"{dataset_prefix}{topo_prefix}R{team_size}Adapter"


@lru_cache(maxsize=None)
def role_catalog(dataset: str, base_topology: str) -> tuple[str, ...]:
    roles_path = repo_root() / "configs" / "prompts" / "roles.yaml"
    cfg = yaml.safe_load(roles_path.read_text())
    roles = cfg["topologies"][base_topology]["benchmarks"][dataset]
    return tuple(roles.keys())


def _literal_tuple_first_strings(node: ast.AST) -> list[str]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []
    roles: list[str] = []
    for item in node.elts:
        if not isinstance(item, (ast.Tuple, ast.List)) or not item.elts:
            continue
        first = item.elts[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            roles.append(first.value)
    return roles


@lru_cache(maxsize=None)
def roles_from_module_source(module_name: str, dataset: str, base_topology: str) -> tuple[str, ...]:
    catalog = role_catalog(dataset, base_topology)
    if base_topology in {"independent", "decentralized"}:
        return catalog

    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return catalog
    source = Path(spec.origin).read_text()
    tree = ast.parse(source)

    load_prompt_roles: list[str] = []
    stage_roles: list[str] = []
    worker_roles: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> Any:
            func = node.func
            if isinstance(func, ast.Name) and func.id == "_load_prompt" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    load_prompt_roles.append(arg.value)
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> Any:
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "stages":
                    stage_roles.extend(_literal_tuple_first_strings(node.value))
                elif target.id == "worker_specs":
                    worker_roles.extend(_literal_tuple_first_strings(node.value))
            self.generic_visit(node)

    Visitor().visit(tree)

    if base_topology == "sequential" and stage_roles:
        roles = stage_roles
    elif base_topology == "centralized" and worker_roles:
        manager_role = next((r for r in load_prompt_roles if r.startswith("manager")), "manager")
        roles = [manager_role, *worker_roles]
    else:
        roles = load_prompt_roles or list(catalog)

    valid = set(catalog)
    deduped: list[str] = []
    for role in roles:
        if role in valid and role not in deduped:
            deduped.append(role)
    return tuple(deduped or catalog)


def contract_role(role: str) -> str:
    if role.startswith("manager_r") and role.removeprefix("manager_r").isdigit():
        return "manager"
    return role


def trace_role(role: str) -> str:
    return contract_role(role)


class TeamSizesAdapterMixin:
    """Shared behavior for HotpotQA/LCB team-size adapters."""

    team_size: int
    base_topology: str
    teamsizes_enabled = True

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        n_agents: int | None = None,
        n_rounds: int | None = None,
    ):
        resolved_agents = n_agents if n_agents is not None else self.team_size
        resolved_rounds = n_rounds
        if self.base_topology == "decentralized" and resolved_rounds is None:
            resolved_rounds = int(os.environ.get("DECENTRALIZED_N_ROUNDS", os.environ.get("N_ROUNDS", "2")))
        super().__init__(prompts=prompts, n_agents=resolved_agents, n_rounds=resolved_rounds)

    def describe_runtime(self, example: Any | None = None) -> dict:
        meta = super().describe_runtime(example)
        meta.update(
            {
                "teamsizes_enabled": True,
                "team_size": self.team_size,
                "base_topology": self.base_topology,
                "teamsizes_module": self.module_name,
            }
        )
        return meta

    def role_detail(self, role: str, out: dict) -> str:
        return super().role_detail(trace_role(role), out)

    def _role_detail(self, role: str, out: dict) -> str:
        return super()._role_detail(trace_role(role), out)


class TeamSizesLCBAdapter(TeamSizesAdapterMixin, ModuleLCBAdapter):
    dataset = "lcb"
    framework = "langgraph"


class TeamSizesHotpotQAAdapter(TeamSizesAdapterMixin, ModuleHotpotQAAdapter):
    dataset = "hotpotqa"
    framework = "langgraph"


def _make_adapter_class(dataset: str, base_topology: str, team_size: int):
    module_name = teamsizes_module_name(dataset, base_topology, team_size)
    base_class = TeamSizesHotpotQAAdapter if dataset == "hotpotqa" else TeamSizesLCBAdapter
    attrs = {
        "topology": teamsizes_topology(base_topology, team_size),
        "base_topology": base_topology,
        "prompt_topology": base_topology,
        "team_size": team_size,
        "roles_": list(roles_from_module_source(module_name, dataset, base_topology)),
        "module_name": module_name,
        "__module__": __name__,
    }
    return type(class_name(dataset, base_topology, team_size), (base_class,), attrs)


TEAMSIZES_ADAPTER_NAMES: dict[tuple[str, str, int], str] = {}
for _dataset in SUPPORTED_TEAMSIZES_DATASETS:
    for _base_topology in SUPPORTED_TEAMSIZES_BASE_TOPOLOGIES:
        for _team_size in SUPPORTED_TEAM_SIZES:
            _cls = _make_adapter_class(_dataset, _base_topology, _team_size)
            globals()[_cls.__name__] = _cls
            TEAMSIZES_ADAPTER_NAMES[(_dataset, _base_topology, _team_size)] = _cls.__name__


__all__ = [
    "SUPPORTED_TEAMSIZES_DATASETS",
    "SUPPORTED_TEAMSIZES_BASE_TOPOLOGIES",
    "SUPPORTED_TEAM_SIZES",
    "TEAMSIZES_ADAPTER_NAMES",
    "class_name",
    "teamsizes_module_name",
    "teamsizes_topology",
    *sorted(TEAMSIZES_ADAPTER_NAMES.values()),
]
