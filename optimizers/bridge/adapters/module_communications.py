"""communications communication-format adapters backed by real ``topologies`` modules.

The model solves with the normal task prompts.  communications syntax is rendered by
infrastructure from raw runner reports before metadata is scored or recorded,
so optimizers improve task behavior instead of JSON/tag formatting obedience.
Each call runs a fresh copy of the runner with the pair's ``COMMUNICATION_FORMAT``
preset, so the runner itself renders the handoffs in that format.
"""

from __future__ import annotations

import os
from typing import Any

from communications.communication_formats import runner_module
from core.communication import FORMATS, append_contract, begin_handoff_recording, collect_reports, end_handoff_recording
from optimizers.bridge.adapters.bfcl_common import execution_prompt
from optimizers.bridge.adapters.module_bfcl import ModuleBFCLAdapter
from optimizers.bridge.adapters.module_common import import_isolated_real_module
from optimizers.bridge.adapters.module_hotpotqa import ModuleHotpotQAAdapter
from optimizers.bridge.adapters.module_lcb import ModuleLCBAdapter

SUPPORTED_COMMUNICATIONS_DATASETS = ("hotpotqa", "lcb", "bfcl")
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
    "bfcl": {
        "independent": ["caller"],
        "decentralized": ["debater"],
        "sequential": ["analyzer", "inspector", "caller", "verifier"],
        "centralized": ["manager", "inspector_worker", "caller_worker", "validator_worker"],
    },
}


def communications_topology(base_topology: str, fmt: str) -> str:
    return f"{base_topology}_communications_{fmt}"


def communications_proxy_module_name(dataset: str, base_topology: str, fmt: str) -> str:
    return f"communications.{base_topology}.{dataset}.{dataset}_{fmt}"


def communications_class_name(dataset: str, base_topology: str, fmt: str) -> str:
    dataset_prefix = {
        "hotpotqa": "HotpotQACommunications",
        "lcb": "LCBCommunications",
        "bfcl": "BFCLCommunications",
    }[dataset]
    topo_prefix = "".join(part.title() for part in base_topology.split("_"))
    fmt_prefix = "".join(part.title() for part in fmt.split("_"))
    return f"{dataset_prefix}{topo_prefix}{fmt_prefix}Adapter"


class CommunicationsAdapterMixin:
    """Shared behavior for fixed-format communications pairs."""

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

    def load_module(self):
        """A fresh copy of the runner with the pair's ``COMMUNICATION_FORMAT`` preset: it renders the handoffs."""
        return import_isolated_real_module(self.module_name, COMMUNICATION_FORMAT=self.communications_format)

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
        runner_output["communication_inflight_all_parse_ok"] = all(bool(item.get("ok")) for item in handoffs)
        reports = collect_reports(
            runner_output, topology=self.base_topology, fmt=self.communications_format, dataset=self.dataset
        )
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


class CommunicationsLCBAdapter(CommunicationsAdapterMixin, ModuleLCBAdapter):
    dataset = "lcb"
    framework = "langgraph"


class CommunicationsBFCLAdapter(CommunicationsAdapterMixin, ModuleBFCLAdapter):
    """BFCL keeps the format contract in the role prompts.

    This matches ``communications/<topology>/bfcl/bfcl_<fmt>.py`` exactly:
    the protected output contract first, then the inter-agent format
    contract (a no-op for ``freeform``). In-flight handoffs are still
    rendered by the topology runner and scored by ``collect_reports``.
    """

    dataset = "bfcl"
    framework = "langgraph"

    def _prompt_for_module(self, module, role: str) -> str:
        text = execution_prompt(self._prompts[role], self.prompt_topology, role)
        return append_contract(text, self.communications_format, self.dataset)


def _make_adapter_class(dataset: str, base_topology: str, fmt: str):
    if fmt not in FORMATS:
        raise ValueError(f"unknown communications format {fmt!r}")
    module_name = runner_module(base_topology, dataset)
    base_class = {
        "hotpotqa": CommunicationsHotpotQAAdapter,
        "lcb": CommunicationsLCBAdapter,
        "bfcl": CommunicationsBFCLAdapter,
    }[dataset]
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
