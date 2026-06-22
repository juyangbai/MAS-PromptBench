"""ToolHop adapters backed by the real shared runner."""
from __future__ import annotations

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


DATASET = "toolhop"
DATASET_PREFIX = "ToolHop"
COMMON_MODULE = "topologies.single.toolhop.langgraph_toolhop"
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
    "single": "SingleToolHopAdapter",
    "independent": "IndependentToolHopAdapter",
    "sequential": "SequentialToolHopAdapter",
    "sequential_crewai": "SequentialCrewAIToolHopAdapter",
    "centralized": "CentralizedToolHopAdapter",
    "centralized_autogen": "CentralizedAutoGenToolHopAdapter",
    "decentralized": "DecentralizedToolHopAdapter",
    "decentralized_openai": "DecentralizedOpenAIToolHopAdapter",
}


class ToolHopAdapter(CommonToolUseAdapterBase):
    dataset = DATASET
    dataset_prefix = DATASET_PREFIX
    common_module_name = COMMON_MODULE

    def adapter_output(self, module: Any, instance: dict, out: dict) -> dict:
        raw = self._fallback_raw(out)
        predicted = out.get("predicted_answer") or module.extract_answer(raw)
        answer_text = raw or (f"<answer>{predicted}</answer>" if predicted else "")
        return {
            "model_output": [],
            "answer": answer_text,
            "answer_text": answer_text,
            "predicted_answer": predicted,
            "winner": out.get("winner"),
            "buckets": out.get("buckets") or {},
            "raw": raw,
            "runner_output": out,
        }

    def _fallback_raw(self, out: dict) -> str:
        if out.get("final_content"):
            return str(out.get("final_content") or "")
        if isinstance(out.get("by_stage"), dict):
            stages = out.get("by_stage") or {}
            return str(stages.get(self.final_role) or stages.get("verifier") or "")
        if isinstance(out.get("manager"), dict):
            return str(out["manager"].get("final_content") or out["manager"].get("predicted_answer") or "")
        if isinstance(out.get("per_agent"), list) and out["per_agent"]:
            return str(out["per_agent"][0].get("final_content") or out["per_agent"][0].get("predicted_answer") or "")
        if isinstance(out.get("per_peer"), list) and out["per_peer"]:
            return str(out["per_peer"][0].get("final_content") or out["per_peer"][0].get("predicted_answer") or "")
        messages = out.get("messages") or []
        if messages:
            return str(module._last_assistant_content(messages))
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
    return type(CORE_CLASS_NAMES[topology], (ToolHopAdapter,), attrs)


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
            "style": f"{_base_topology}_toolhop_r{_team_size}",
            "roles_": list(_profile["roles"]),
            "final_role": _profile["final_role"],
            "stage_roles": _profile["stage_roles"],
            "worker_roles": _profile["worker_roles"],
            "team_size": _team_size,
            "__module__": __name__,
        }
        _cls = type(_class_name, (ToolHopAdapter,), _attrs)
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
            "style": f"{_base_topology}_toolhop_communications_{_fmt}",
            "roles_": list(_profile["roles"]),
            "final_role": _profile["final_role"],
            "stage_roles": _profile["stage_roles"],
            "worker_roles": _profile["worker_roles"],
            "communications_format": _fmt,
            "__module__": __name__,
        }
        _cls = type(_class_name, (ToolHopAdapter,), _attrs)
        globals()[_class_name] = _cls
        COMMUNICATIONS_ADAPTER_NAMES[(_base_topology, _fmt)] = _class_name


__all__ = [
    "ToolHopAdapter",
    "TEAMSIZES_ADAPTER_NAMES",
    "COMMUNICATIONS_ADAPTER_NAMES",
    *CORE_CLASS_NAMES.values(),
    *sorted(TEAMSIZES_ADAPTER_NAMES.values()),
    *sorted(COMMUNICATIONS_ADAPTER_NAMES.values()),
]
