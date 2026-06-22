"""communications communication-format adapters backed by real ``topologies`` modules.

The model solves with the normal task prompts.  communications syntax is rendered by
infrastructure from raw runner reports before metadata is scored or recorded,
so GEPA optimizes task behavior instead of JSON/tag formatting obedience.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from communications.communication_formats import (
    BASE_MODULES,
    FORMATS,
    begin_handoff_recording,
    collect_reports,
    end_handoff_recording,
)

from real_runner_mipro.adapters.module_hotpotqa import ModuleHotpotQAAdapter
from real_runner_mipro.adapters.module_lcb import ModuleLCBAdapter
from real_runner_mipro.output_contracts import append_output_contract


SUPPORTED_COMMUNICATIONS_DATASETS = ("hotpotqa", "lcb")
SUPPORTED_COMMUNICATIONS_BASE_TOPOLOGIES = ("independent", "decentralized", "sequential", "centralized")
SUPPORTED_COMMUNICATIONS_FORMATS = ("freeform", "semi_structured", "structured_soft")


ROLE_CATALOG: dict[str, dict[str, list[str]]] = {
    "hotpotqa": {
        "independent": ["solver"],
        "decentralized": ["debater"],
        "sequential": ["planner", "retriever", "reasoner", "writer"],
        "centralized": ["manager", "retriever_worker", "reasoner_worker", "writer_worker"],
    },
    "lcb": {
        "independent": ["coder"],
        "decentralized": ["debater"],
        "sequential": ["analyzer", "coder", "tester", "debugger"],
        "centralized": ["manager", "analyzer_worker", "coder_worker", "tester_worker"],
    },
}


def communications_topology(base_topology: str, fmt: str) -> str:
    return f"{base_topology}_communications_{fmt}"


def communications_proxy_module_name(dataset: str, base_topology: str, fmt: str) -> str:
    return f"communications.{base_topology}.{dataset}.{dataset}_{fmt}"


def communications_class_name(dataset: str, base_topology: str, fmt: str) -> str:
    dataset_prefix = {"hotpotqa": "HotpotQACommunications", "lcb": "LCBCommunications"}[dataset]
    topo_prefix = "".join(part.title() for part in base_topology.split("_"))
    fmt_prefix = "".join(part.title() for part in fmt.split("_"))
    return f"{dataset_prefix}{topo_prefix}{fmt_prefix}Adapter"


class CommunicationsAdapterMixin:
    """Shared behavior for fixed-format communications GEPA pairs."""

    communications_enabled = True
    communications_format: str
    base_topology: str
    communications_module: str

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        n_agents: int | None = None,
        n_rounds: int | None = None,
    ):
        resolved_agents = n_agents
        resolved_rounds = n_rounds
        if self.base_topology == "independent" and resolved_agents is None:
            resolved_agents = int(os.environ.get("INDEPENDENT_N_AGENTS", os.environ.get("N_AGENTS", "4")))
        if self.base_topology == "decentralized":
            if resolved_agents is None:
                resolved_agents = int(os.environ.get("DECENTRALIZED_N_AGENTS", os.environ.get("N_AGENTS", "4")))
            if resolved_rounds is None:
                resolved_rounds = int(os.environ.get("DECENTRALIZED_N_ROUNDS", os.environ.get("N_ROUNDS", "2")))
        super().__init__(prompts=prompts, n_agents=resolved_agents, n_rounds=resolved_rounds)

    @contextmanager
    def _communications_runtime_format(self, module):
        missing = object()
        original = getattr(module, "COMMUNICATION_FORMAT", missing)
        setattr(module, "COMMUNICATION_FORMAT", self.communications_format)
        try:
            yield
        finally:
            if original is missing:
                try:
                    delattr(module, "COMMUNICATION_FORMAT")
                except AttributeError:
                    pass
            else:
                setattr(module, "COMMUNICATION_FORMAT", original)

    @contextmanager
    def _patched_module(self, module):
        parent = super()
        parent_ctx = getattr(parent, "_patched_module", None) or getattr(parent, "patched_module")
        with parent_ctx(module):
            with self._communications_runtime_format(module):
                yield

    @contextmanager
    def patched_module(self, module):
        parent = super()
        parent_ctx = getattr(parent, "patched_module", None) or getattr(parent, "_patched_module")
        with parent_ctx(module):
            with self._communications_runtime_format(module):
                yield

    def describe_runtime(self, example: Any | None = None) -> dict:
        meta = super().describe_runtime(example)
        meta.update(
            {
                "communications_enabled": True,
                "communications_format": self.communications_format,
                "base_topology": self.base_topology,
                "communications_module": self.communications_module,
                "communications_base_module": self.module_name,
            }
        )
        return meta

    def run_example(self, example: Any) -> dict:
        token = begin_handoff_recording()
        handoffs: list[dict] = []
        try:
            result = super().run_example(example)
        finally:
            handoffs = end_handoff_recording(token)
        runner_output = result.get("runner_output") or {}
        runner_output["communication_inflight_handoffs"] = handoffs
        runner_output["communication_inflight_handoff_count"] = len(handoffs)
        runner_output["communication_inflight_all_parse_ok"] = all(
            bool(item.get("ok")) for item in handoffs
        )
        reports = collect_reports(runner_output, topology=self.base_topology, fmt=self.communications_format, dataset=self.dataset)
        runner_output.update(reports)
        result["runner_output"] = runner_output
        result.update(reports)
        for key in (
            "communication_inflight_handoffs",
            "communication_inflight_handoff_count",
            "communication_inflight_all_parse_ok",
        ):
            if key in runner_output:
                result[key] = runner_output[key]
        return result

    def format_role_trace(self, role: str, output: Any) -> str:
        base = super().format_role_trace(role, output)
        if not isinstance(output, dict):
            return base
        runner_output = output.get("runner_output") or {}
        fields = [
            f"communication_format={runner_output.get('communication_format', self.communications_format)}",
            f"communication_parse_ok={runner_output.get('communication_parse_ok')}",
            f"communication_all_parse_ok={runner_output.get('communication_all_parse_ok')}",
            f"communication_parse_rate={runner_output.get('communication_parse_rate')}",
            f"communication_required_report_count={runner_output.get('communication_required_report_count')}",
            f"communication_missing_roles={runner_output.get('communication_missing_roles') or []}",
            f"communication_infra_error={runner_output.get('communication_infra_error')}",
            f"communication_report_ok_count={runner_output.get('communication_report_ok_count')}",
            f"communication_report_total={runner_output.get('communication_report_total')}",
            f"communication_inflight_handoff_count={runner_output.get('communication_inflight_handoff_count')}",
            f"communication_inflight_all_parse_ok={runner_output.get('communication_inflight_all_parse_ok')}",
            f"communication_parse_warnings={runner_output.get('communication_parse_warnings') or []}",
        ]
        return base + "\n" + "\n".join(fields)


class CommunicationsHotpotQAAdapter(CommunicationsAdapterMixin, ModuleHotpotQAAdapter):
    dataset = "hotpotqa"
    framework = "langgraph"

    def _prompt_for_module(self, module, role: str) -> str:
        text = self._prompts[role]
        nudge = getattr(module, "_OUTPUT_FORMAT_NUDGE", "")
        if nudge and "Answer: <short-form>" not in text:
            text = text + nudge
        return append_output_contract(text, self.dataset, self.prompt_topology, role)


class CommunicationsLCBAdapter(CommunicationsAdapterMixin, ModuleLCBAdapter):
    dataset = "lcb"
    framework = "langgraph"

    def prompt_for_role(self, module, role: str) -> str:
        text = self._prompts[role]
        nudge = getattr(module, "_OUTPUT_FORMAT_NUDGE", "")
        appendix = getattr(module, "_OUTPUT_FORMAT_APPENDIX", "")
        if nudge and nudge not in text:
            text += nudge
        if appendix and appendix not in text:
            text += appendix
        if self.prompt_topology == "centralized" and role == "manager":
            direct_nudge = (
                "\n\nCompile-time robustness: if delegation/tool calls are not "
                "strictly necessary, solve directly and emit the final fenced "
                "```python``` solution. If you do call a tool, keep arguments "
                "short valid JSON strings."
            )
            if direct_nudge not in text:
                text += direct_nudge
        if role in {"solver", "coder", "debugger", "debater"}:
            code_first_nudge = (
                "\n\nCode-first requirement: do not write a long explanation. "
                "Your final answer must start with ```python and contain the "
                "complete submitted solution before any prose."
            )
            if code_first_nudge not in text:
                text += code_first_nudge
        return append_output_contract(text, self.dataset, self.prompt_topology, role)


def _make_adapter_class(dataset: str, base_topology: str, fmt: str):
    if fmt not in FORMATS:
        raise ValueError(f"unknown communications format {fmt!r}")
    module_name = BASE_MODULES[(base_topology, dataset)]
    base_class = CommunicationsHotpotQAAdapter if dataset == "hotpotqa" else CommunicationsLCBAdapter
    attrs = {
        "topology": communications_topology(base_topology, fmt),
        "base_topology": base_topology,
        "prompt_topology": base_topology,
        "communications_format": fmt,
        "communications_module": communications_proxy_module_name(dataset, base_topology, fmt),
        "roles_": list(ROLE_CATALOG[dataset][base_topology]),
        "module_name": module_name,
        "__module__": __name__,
    }
    return type(communications_class_name(dataset, base_topology, fmt), (base_class,), attrs)


COMMUNICATIONS_ADAPTER_NAMES: dict[tuple[str, str, str], str] = {}
for _dataset in SUPPORTED_COMMUNICATIONS_DATASETS:
    for _base_topology in SUPPORTED_COMMUNICATIONS_BASE_TOPOLOGIES:
        for _fmt in SUPPORTED_COMMUNICATIONS_FORMATS:
            _cls = _make_adapter_class(_dataset, _base_topology, _fmt)
            globals()[_cls.__name__] = _cls
            COMMUNICATIONS_ADAPTER_NAMES[(_dataset, _base_topology, _fmt)] = _cls.__name__


__all__ = [
    "SUPPORTED_COMMUNICATIONS_DATASETS",
    "SUPPORTED_COMMUNICATIONS_BASE_TOPOLOGIES",
    "SUPPORTED_COMMUNICATIONS_FORMATS",
    "COMMUNICATIONS_ADAPTER_NAMES",
    "communications_class_name",
    "communications_proxy_module_name",
    "communications_topology",
    *sorted(COMMUNICATIONS_ADAPTER_NAMES.values()),
]
