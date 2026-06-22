"""APPS adapters backed by the real topology modules."""
from __future__ import annotations

import os
from typing import Any

from real_runner_gepa.adapters.gpqa_common import coerce_instance
from real_runner_gepa.adapters.module_common import (
    ModuleAdapterBase,
    call_with_supported_kwargs,
    fenced_code,
    import_real_module,
    module_lock,
)


class ModuleAPPSAdapter(ModuleAdapterBase):
    dataset = "apps"

    def run_example(self, example: Any) -> dict:
        instance = coerce_instance(example)
        module = import_real_module(self.module_name)
        last_exc = None
        for _ in range(2):
            try:
                with module_lock(self.module_name), self.patched_module(module):
                    out = call_with_supported_kwargs(
                        module.solve,
                        instance["problem"],
                        starter_code=instance.get("starter_code"),
                        input_output=instance.get("input_output"),
                    )
                break
            except TimeoutError as exc:
                return self._runtime_failure_output(instance, exc)
            except Exception as exc:
                if type(exc).__name__ == "BadRequestError":
                    return self._runtime_failure_output(instance, exc)
                last_exc = exc
        else:
            raise last_exc
        code = out.get("code") or self._extract_code_from_output(module, out)
        raw = out.get("raw") or self._fallback_raw(out)
        return {
            "model_output": [],
            "answer": code,
            "answer_text": fenced_code(code) if code else raw,
            "code": code,
            "winner": out.get("winner"),
            "buckets": {},
            "raw": raw,
            "runner_output": out,
        }

    def _runtime_failure_output(self, instance: dict, exc: BaseException) -> dict:
        raw = (
            "ERROR: real APPS runner failed before producing a final fenced "
            f"python solution for task {instance.get('id') or '<unknown>'}: "
            f"{type(exc).__name__}: {exc}"
        )
        return {
            "model_output": [],
            "answer": None,
            "answer_text": raw,
            "code": None,
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

    def _fallback_raw(self, out: dict) -> str:
        if "by_stage" in out:
            stages = out.get("by_stage") or {}
            return str(stages.get(self.roles_[-1]) or stages.get("coder") or "")
        if "per_agent" in out:
            return str((out.get("per_agent") or [{}])[0].get("raw", ""))
        if "per_peer" in out:
            raw = str((out.get("per_peer") or [{}])[0].get("raw", ""))
            if raw:
                return raw
        for ctx in reversed(out.get("all_contexts") or []):
            for message in reversed(ctx or []):
                content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
                if isinstance(content, str) and content.strip():
                    return content
        messages = out.get("messages") or []
        for message in reversed(messages):
            content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                return content
        return ""

    def _extract_code_from_output(self, module, out: dict) -> str | None:
        extractor = getattr(module, "extract_code", None)
        if extractor is None:
            return None
        containers = []
        if out.get("messages"):
            containers.append(out.get("messages") or [])
        containers.extend(out.get("all_contexts") or [])
        for key in ("per_agent", "per_peer"):
            for item in out.get(key) or []:
                raw = item.get("raw") if isinstance(item, dict) else None
                if isinstance(raw, str):
                    code = extractor(raw)
                    if code:
                        return code
        for messages in containers:
            for message in reversed(messages or []):
                content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
                if not isinstance(content, str) or not content.strip():
                    continue
                code = extractor(content)
                if code:
                    return code
        return None


class SingleAPPSAdapter(ModuleAPPSAdapter):
    topology = "single"
    framework = "langgraph"
    prompt_topology = "single"
    roles_ = ["solver"]
    module_name = "topologies.single.apps.langgraph_apps"


class IndependentAPPSAdapter(ModuleAPPSAdapter):
    topology = "independent"
    framework = "langgraph"
    prompt_topology = "independent"
    roles_ = ["coder"]
    module_name = "topologies.independent.apps.langgraph_apps"

    def __init__(self, *args, n_agents: int | None = None, **kwargs):
        super().__init__(*args, n_agents=n_agents or int(os.environ.get("INDEPENDENT_N_AGENTS", "4")), **kwargs)


class DecentralizedAPPSAdapter(ModuleAPPSAdapter):
    topology = "decentralized"
    framework = "langgraph"
    prompt_topology = "decentralized"
    roles_ = ["debater"]
    module_name = "topologies.decentralized.langgraph.apps.langgraph_apps"

    def __init__(self, *args, n_agents: int | None = None, n_rounds: int | None = None, **kwargs):
        super().__init__(
            *args,
            n_agents=n_agents or int(os.environ.get("DECENTRALIZED_N_AGENTS", "4")),
            n_rounds=n_rounds or int(os.environ.get("DECENTRALIZED_N_ROUNDS", "2")),
            **kwargs,
        )


class DecentralizedOpenAIAPPSAdapter(DecentralizedAPPSAdapter):
    topology = "decentralized_openai"
    framework = "openai"
    module_name = "topologies.decentralized.openai.apps.openai_apps"


class SequentialAPPSAdapter(ModuleAPPSAdapter):
    topology = "sequential"
    framework = "langgraph"
    prompt_topology = "sequential"
    roles_ = ["analyzer", "coder", "tester", "debugger"]
    module_name = "topologies.sequential.langgraph.apps.langgraph_apps"


class SequentialCrewAIAPPSAdapter(SequentialAPPSAdapter):
    topology = "sequential_crewai"
    framework = "crewai"
    module_name = "topologies.sequential.crewai.apps.crewai_apps"


class CentralizedAPPSAdapter(ModuleAPPSAdapter):
    topology = "centralized"
    framework = "langgraph"
    prompt_topology = "centralized"
    roles_ = ["manager", "analyzer_worker", "coder_worker", "tester_worker"]
    module_name = "topologies.centralized.langgraph.apps.langgraph_apps"


class CentralizedAutoGenAPPSAdapter(CentralizedAPPSAdapter):
    topology = "centralized_autogen"
    framework = "autogen"
    module_name = "topologies.centralized.autogen.apps.autogen_apps"
