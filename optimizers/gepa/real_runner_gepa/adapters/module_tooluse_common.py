"""Shared adapters for API-Bank and ToolHop real-runner GEPA pairs."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from communications.communication_formats import STRICT_COMMUNICATION_FIELDS, collect_reports
from real_runner_gepa.adapters.gpqa_common import coerce_instance
from real_runner_gepa.adapters.module_common import import_real_module, module_lock
from real_runner_gepa.lm import TASK_MODEL, next_task_endpoint
from real_runner_gepa.output_contracts import append_output_contract
from real_runner_gepa.registry import TEAM_SIZES


BASE_TOPOLOGIES = ("independent", "decentralized", "sequential", "centralized")
COMMUNICATIONS_FORMATS = ("freeform", "semi_structured", "structured_soft")
DEFAULT_CORE_TEAM_SIZE = 4
COMMUNICATION_FIELDS = (
    "communication_format",
    "communication_parse_ok",
    *STRICT_COMMUNICATION_FIELDS,
    "communication_parse_errors",
    "communication_parse_warnings",
    "communication_report_ok_count",
    "communication_report_total",
    "communication_reports",
    "communication_rendered_reports",
    "communication_inflight_handoffs",
    "communication_inflight_handoff_count",
    "communication_inflight_all_parse_ok",
)

TOOLHOP_GENERALIZATION_GUARDRAIL = """

TOOLHOP GENERALIZATION GUARDRAIL:
Use training feedback to improve the reusable process, not to memorize facts. Do not
embed sample-specific names, titles, dates, IDs, tool outputs, or gold answers into
the prompt. Do not add successful-example sections, domain-specific knowledge
sections, concrete answer-tag examples, or examples containing concrete names,
titles, dates, or answers from train/validation traces. Use placeholders such as
ENTITY, DATE, TOOL_OUTPUT, and VALUE instead. For each new ToolHop instance,
derive intermediate values only from the current question, current tool schemas, and
current tool outputs. If a remembered fact conflicts with a current tool result,
trust the current tool result.
""".strip()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def prompt_path(topology: str, dataset: str, role: str) -> Path:
    return repo_root() / "configs" / "prompts" / topology / dataset / f"{role}.txt"


def append_toolhop_generalization_guardrail(dataset: str, text: str) -> str:
    if dataset != "toolhop":
        return text
    if "TOOLHOP GENERALIZATION GUARDRAIL:" in text:
        return text
    return text.rstrip() + "\n\n" + TOOLHOP_GENERALIZATION_GUARDRAIL


def load_prompts(
    dataset: str,
    topology: str,
    roles: list[str],
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    overrides = overrides or {}
    prompts: dict[str, str] = {}
    for role in roles:
        text = overrides.get(role, prompt_path(topology, dataset, role).read_text().strip())
        prompts[role] = append_toolhop_generalization_guardrail(dataset, text)
    return prompts


@lru_cache(maxsize=None)
def roles_from_yaml(dataset: str, topology: str) -> tuple[str, ...]:
    cfg = yaml.safe_load((repo_root() / "configs" / "prompts" / "roles.yaml").read_text())
    roles = cfg["topologies"][topology]["benchmarks"][dataset]
    return tuple(roles.keys())


def sequential_roles(dataset: str, team_size: int | None = None) -> list[str]:
    roles = list(roles_from_yaml(dataset, "sequential"))
    return roles[:team_size] if team_size is not None else roles


def centralized_worker_roles(dataset: str) -> list[str]:
    return [
        role
        for role in roles_from_yaml(dataset, "centralized")
        if role != "manager" and not role.startswith("manager_r")
    ]


def centralized_roles(dataset: str, team_size: int | None = None) -> tuple[str, list[str]]:
    workers = centralized_worker_roles(dataset)
    if team_size is None:
        return "manager", workers
    manager = f"manager_r{team_size}" if team_size in {8, 10} else "manager"
    return manager, workers[: max(team_size - 1, 0)]


def class_name(dataset_prefix: str, base_topology: str, suffix: str) -> str:
    topo = "".join(part.title() for part in base_topology.split("_"))
    return f"{dataset_prefix}{topo}{suffix}Adapter"


def communications_class_name(dataset_prefix: str, base_topology: str, fmt: str) -> str:
    fmt_prefix = "".join(part.title() for part in fmt.split("_"))
    return class_name(dataset_prefix + "Communications", base_topology, fmt_prefix)


def teamsizes_class_name(dataset_prefix: str, base_topology: str, team_size: int) -> str:
    return class_name(dataset_prefix + "TeamSizes", base_topology, f"R{team_size}")


def profile_for(dataset: str, base_topology: str, team_size: int | None = None) -> dict[str, Any]:
    effective_team_size = team_size
    if effective_team_size is None and base_topology in {"sequential", "centralized"}:
        effective_team_size = DEFAULT_CORE_TEAM_SIZE
    if base_topology in {"single", "independent"}:
        roles = ["solver"]
        return {"roles": roles, "final_role": "solver", "stage_roles": None, "worker_roles": None}
    if base_topology == "decentralized":
        roles = ["debater"]
        return {"roles": roles, "final_role": "debater", "stage_roles": None, "worker_roles": None}
    if base_topology == "sequential":
        roles = sequential_roles(dataset, effective_team_size)
        return {"roles": roles, "final_role": roles[-1], "stage_roles": tuple(roles), "worker_roles": None}
    if base_topology == "centralized":
        manager, workers = centralized_roles(dataset, effective_team_size)
        roles = [manager, *workers]
        return {"roles": roles, "final_role": manager, "stage_roles": None, "worker_roles": tuple(workers)}
    raise ValueError(f"unsupported topology {base_topology!r}")


class CommonToolUseAdapterBase:
    """Prompt-mutable adapter that delegates to a shared real runner module."""

    dataset: str
    dataset_prefix: str
    common_module_name: str
    topology: str
    prompt_topology: str
    runner_topology: str
    style: str
    roles_: list[str]
    final_role: str
    stage_roles: tuple[str, ...] | None = None
    worker_roles: tuple[str, ...] | None = None
    team_size: int | None = None
    communications_format: str | None = None

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        n_agents: int | None = None,
        n_rounds: int | None = None,
    ):
        self._prompts = load_prompts(self.dataset, self.prompt_topology, self.roles_, prompts)
        self.n_agents = n_agents if n_agents is not None else self._default_n_agents()
        self.n_rounds = n_rounds if n_rounds is not None else self._default_n_rounds()

    def _default_n_agents(self) -> int | None:
        if self.team_size is not None:
            return self.team_size
        if self.runner_topology == "independent":
            return int(os.environ.get(f"{self.dataset.upper()}_INDEPENDENT_N_AGENTS", os.environ.get("INDEPENDENT_N_AGENTS", "4")))
        if self.runner_topology == "decentralized":
            return int(os.environ.get(f"{self.dataset.upper()}_DECENTRALIZED_N_AGENTS", os.environ.get("DECENTRALIZED_N_AGENTS", "4")))
        return None

    def _default_n_rounds(self) -> int | None:
        if self.runner_topology == "decentralized":
            return int(os.environ.get(f"{self.dataset.upper()}_DECENTRALIZED_N_ROUNDS", os.environ.get("DECENTRALIZED_N_ROUNDS", "2")))
        return None

    def roles(self) -> list[str]:
        return list(self.roles_)

    def get_prompt(self, role: str) -> str:
        self._check_role(role)
        return self._prompts[role]

    def set_prompt(self, role: str, text: str) -> None:
        self._check_role(role)
        self._prompts[role] = append_toolhop_generalization_guardrail(self.dataset, text)

    def reset(self) -> None:
        return None

    def __getstate__(self):
        return self.__dict__.copy()

    def describe_runtime(self, example: Any | None = None) -> dict:
        instance = coerce_instance(example) if example is not None else {}
        return {
            "topology": self.topology,
            "runner_topology": self.runner_topology,
            "dataset": self.dataset,
            "style": self.style,
            "roles": self.roles(),
            "final_role": self.final_role,
            "stage_roles": list(self.stage_roles or []),
            "worker_roles": list(self.worker_roles or []),
            "team_size": self.team_size,
            "communications_format": self.communications_format,
            "n_agents": self.n_agents,
            "n_rounds": self.n_rounds,
            "module": self.common_module_name,
            "example_id": instance.get("id"),
        }

    def prompt_for_role(self, role: str, prompt_suffix: str = "") -> str:
        self._check_role(role)
        text = append_toolhop_generalization_guardrail(self.dataset, self._prompts[role])
        if prompt_suffix and prompt_suffix not in text:
            text = text.rstrip() + "\n" + prompt_suffix.strip()
        return append_output_contract(text, self.dataset, self.prompt_topology, role)

    @contextmanager
    def patched_common(self, module):
        restore: dict[str, Any] = {}

        def patch(name: str, value: Any) -> None:
            restore[name] = getattr(module, name, None)
            setattr(module, name, value)

        if self.dataset == "apibank":
            patch("_load_prompt", lambda topology, role, style, prompt_suffix="": self.prompt_for_role(role, prompt_suffix))
        elif self.dataset == "toolhop":
            patch("_system_prompt", lambda topology, role, style, prompt_suffix="": self.prompt_for_role(role, prompt_suffix))
        if hasattr(module, "VLLM_BASE_URL"):
            patch("VLLM_BASE_URL", next_task_endpoint())
        if hasattr(module, "MODEL_ID"):
            patch("MODEL_ID", os.environ.get("MODEL_ID", os.environ.get("TASK_MODEL", TASK_MODEL)))
        if self.n_agents is not None:
            if hasattr(module, "INDEPENDENT_N_AGENTS"):
                patch("INDEPENDENT_N_AGENTS", self.n_agents)
            if hasattr(module, "DECENTRALIZED_N_AGENTS"):
                patch("DECENTRALIZED_N_AGENTS", self.n_agents)
        if self.n_rounds is not None and hasattr(module, "DECENTRALIZED_N_ROUNDS"):
            patch("DECENTRALIZED_N_ROUNDS", self.n_rounds)
        try:
            yield
        finally:
            for name, value in restore.items():
                setattr(module, name, value)

    def run_example(self, example: Any) -> dict:
        instance = coerce_instance(example)
        module = import_real_module(self.common_module_name)
        try:
            with module_lock(self.common_module_name), self.patched_common(module):
                out = module.solve_topology(
                    instance,
                    style=self.style,
                    topology=self.runner_topology,
                    role=self.final_role,
                    roles=self.stage_roles,
                    worker_roles=self.worker_roles,
                )
        except TimeoutError as exc:
            return self._runtime_failure_output(instance, exc)
        except Exception as exc:
            if type(exc).__name__ == "BadRequestError":
                return self._runtime_failure_output(instance, exc)
            raise

        if self.communications_format is not None:
            out = dict(out)
            out.update(collect_reports(out, topology=self.runner_topology, fmt=self.communications_format, dataset=self.dataset))
        result = self.adapter_output(module, instance, out)
        for key in COMMUNICATION_FIELDS:
            if key in out:
                result[key] = out[key]
        return result

    def adapter_output(self, module: Any, instance: dict, out: dict) -> dict:
        raise NotImplementedError

    def format_role_trace(self, role: str, output: Any) -> str:
        self._check_role(role)
        if not isinstance(output, dict):
            return str(output)
        runner_output = output.get("runner_output") or {}
        fields = [
            f"role={role}",
            f"dataset={self.dataset}",
            f"topology={self.topology}",
            f"winner={output.get('winner')}",
            f"selected_answer={output.get('answer') or output.get('predicted_answer')}",
            self.role_detail(role, runner_output),
        ]
        for key in COMMUNICATION_FIELDS:
            if key in output:
                fields.append(f"{key}={output.get(key)}")
        return "\n".join(fields)

    def role_detail(self, role: str, out: dict) -> str:
        if "by_stage" in out:
            return f"{role}_text={str((out.get('by_stage') or {}).get(role, ''))[:1200]}"
        if role == self.final_role and isinstance(out.get("manager"), dict):
            return f"{role}_manager={json.dumps(out.get('manager'), default=str)[:1200]}"
        if isinstance(out.get("workers"), list):
            matches = [item for item in out["workers"] if item.get("role") == role]
            if matches:
                return f"{role}_worker={json.dumps(matches[0], default=str)[:1200]}"
        for key in ("per_agent", "per_peer"):
            if key in out:
                return f"{role}_{key}={json.dumps(out.get(key), default=str)[:1200]}"
        if isinstance(out.get("stage_outputs"), list):
            matches = [item for item in out["stage_outputs"] if item.get("role") == role]
            if matches:
                return f"{role}_stage={json.dumps(matches[0], default=str)[:1200]}"
        return f"{role}_output={json.dumps(out, default=str)[:1200]}"

    def _runtime_failure_output(self, instance: dict, exc: BaseException) -> dict:
        raw = (
            f"ERROR: real {self.dataset} runner failed before producing a valid final output "
            f"for task {instance.get('id') or '<unknown>'}: {type(exc).__name__}: {exc}"
        )
        return {
            "model_output": [],
            "answer": None,
            "answer_text": raw,
            "predicted_answer": "",
            "winner": None,
            "buckets": {},
            "raw": raw,
            "runner_output": {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "id": instance.get("id"),
                "raw": raw,
            },
        }

    def _check_role(self, role: str) -> None:
        if role not in self.roles_:
            raise KeyError(f"Unknown role {role!r}; expected one of {self.roles_}")
