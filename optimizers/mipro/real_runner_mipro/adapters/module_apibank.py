"""API-Bank adapters backed by the real shared runner."""
from __future__ import annotations

import json
from typing import Any

from real_runner_mipro.adapters.module_tooluse_common import (
    BASE_TOPOLOGIES,
    COMMUNICATIONS_FORMATS,
    TEAM_SIZES,
    CommonToolUseAdapterBase,
    communications_class_name,
    profile_for,
    teamsizes_class_name,
)


DATASET = "apibank"
DATASET_PREFIX = "APIBank"
COMMON_MODULE = "topologies.single.apibank.langgraph_apibank"
CORE_TOPOLOGIES = {
    "single": ("single", "single_langgraph"),
    "independent": ("independent", "independent_langgraph"),
    "sequential": ("sequential", "sequential_langgraph"),
    "sequential_crewai": ("sequential", "sequential_crewai"),
    "centralized": ("centralized", "centralized_langgraph"),
    "centralized_autogen": ("centralized", "centralized_autogen"),
    "decentralized": ("decentralized", "decentralized_langgraph"),
    "decentralized_openai": ("decentralized", "decentralized_openai"),
}
CORE_CLASS_NAMES = {
    "single": "SingleAPIBankAdapter",
    "independent": "IndependentAPIBankAdapter",
    "sequential": "SequentialAPIBankAdapter",
    "sequential_crewai": "SequentialCrewAIAPIBankAdapter",
    "centralized": "CentralizedAPIBankAdapter",
    "centralized_autogen": "CentralizedAutoGenAPIBankAdapter",
    "decentralized": "DecentralizedAPIBankAdapter",
    "decentralized_openai": "DecentralizedOpenAIAPIBankAdapter",
}


class APIBankAdapter(CommonToolUseAdapterBase):
    dataset = DATASET
    dataset_prefix = DATASET_PREFIX
    common_module_name = COMMON_MODULE

    def adapter_output(self, module: Any, instance: dict, out: dict) -> dict:
        raw = self._fallback_raw(out)
        predicted = out.get("predicted_answer") or module.extract_api_call(raw)
        answer_text = predicted or raw
        return {
            "model_output": [],
            "answer": predicted,
            "answer_text": answer_text,
            "predicted_answer": predicted,
            "winner": out.get("winner"),
            "buckets": out.get("buckets") or {},
            "raw": raw,
            "runner_output": out,
        }

    def _fallback_raw(self, out: dict) -> str:
        if out.get("raw"):
            return str(out.get("raw") or "")
        if isinstance(out.get("by_stage"), dict):
            stages = out.get("by_stage") or {}
            return str(stages.get(self.final_role) or stages.get("verifier") or "")
        if isinstance(out.get("manager"), dict):
            return str(out["manager"].get("predicted_answer") or "")
        if isinstance(out.get("per_agent"), list) and out["per_agent"]:
            return str(out["per_agent"][0].get("predicted_answer") or "")
        if isinstance(out.get("per_peer"), list) and out["per_peer"]:
            return str(out["per_peer"][0].get("predicted_answer") or "")
        return str(out.get("predicted_answer") or "")


def _make_core_class(topology: str):
    runner_topology, style = CORE_TOPOLOGIES[topology]
    profile = profile_for(DATASET, runner_topology)
    attrs = {
        "topology": topology,
        "prompt_topology": runner_topology,
        "runner_topology": runner_topology,
        "style": style,
        "roles_": list(profile["roles"]),
        "final_role": profile["final_role"],
        "stage_roles": profile["stage_roles"],
        "worker_roles": profile["worker_roles"],
        "__module__": __name__,
    }
    return type(CORE_CLASS_NAMES[topology], (APIBankAdapter,), attrs)


for _topology in CORE_TOPOLOGIES:
    _cls = _make_core_class(_topology)
    globals()[_cls.__name__] = _cls


TEAMSIZES_ADAPTER_NAMES: dict[tuple[str, int], str] = {}
for _base_topology in BASE_TOPOLOGIES:
    for _team_size in TEAM_SIZES:
        _profile = profile_for(DATASET, _base_topology, _team_size)
        _class_name = teamsizes_class_name(DATASET_PREFIX, _base_topology, _team_size)
        _attrs = {
            "topology": f"{_base_topology}_r{_team_size}",
            "prompt_topology": _base_topology,
            "runner_topology": _base_topology,
            "style": f"{_base_topology}_apibank_r{_team_size}",
            "roles_": list(_profile["roles"]),
            "final_role": _profile["final_role"],
            "stage_roles": _profile["stage_roles"],
            "worker_roles": _profile["worker_roles"],
            "team_size": _team_size,
            "__module__": __name__,
        }
        _cls = type(_class_name, (APIBankAdapter,), _attrs)
        globals()[_class_name] = _cls
        TEAMSIZES_ADAPTER_NAMES[(_base_topology, _team_size)] = _class_name


COMMUNICATIONS_ADAPTER_NAMES: dict[tuple[str, str], str] = {}
for _base_topology in BASE_TOPOLOGIES:
    for _fmt in COMMUNICATIONS_FORMATS:
        _profile = profile_for(DATASET, _base_topology)
        _class_name = communications_class_name(DATASET_PREFIX, _base_topology, _fmt)
        _attrs = {
            "topology": f"{_base_topology}_communications_{_fmt}",
            "prompt_topology": _base_topology,
            "runner_topology": _base_topology,
            "style": f"{_base_topology}_apibank_communications_{_fmt}",
            "roles_": list(_profile["roles"]),
            "final_role": _profile["final_role"],
            "stage_roles": _profile["stage_roles"],
            "worker_roles": _profile["worker_roles"],
            "communications_format": _fmt,
            "__module__": __name__,
        }
        _cls = type(_class_name, (APIBankAdapter,), _attrs)
        globals()[_class_name] = _cls
        COMMUNICATIONS_ADAPTER_NAMES[(_base_topology, _fmt)] = _class_name


__all__ = [
    "APIBankAdapter",
    "TEAMSIZES_ADAPTER_NAMES",
    "COMMUNICATIONS_ADAPTER_NAMES",
    *CORE_CLASS_NAMES.values(),
    *sorted(TEAMSIZES_ADAPTER_NAMES.values()),
    *sorted(COMMUNICATIONS_ADAPTER_NAMES.values()),
]
