"""Sequential/BFCL real-runner adapter."""
from __future__ import annotations

import json
import operator
from collections.abc import Callable
from typing import Annotated, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from real_runner_mipro.adapters.bfcl_common import (
    coerce_instance,
    compact_role_trace,
    default_chat_model,
    execution_prompt,
    extract_canonical,
    flatten_user_request,
    load_prompts,
    schemas_text,
)


TOPOLOGY = "sequential"
DATASET = "bfcl"
ROLES = ["analyzer", "inspector", "caller", "verifier"]


TASK_DESCRIPTIONS = {
    "analyzer": (
        "Parse the user request below to extract the intent, entities, "
        "and implicit constraints. Do NOT reference the schemas yet.\n\n"
        "USER REQUEST:\n{user_request}"
    ),
    "inspector": (
        "Given the Analyzer's summary and the function schema(s) below, map "
        "the intent onto a specific function and propose argument values.\n\n"
        "USER REQUEST:\n{user_request}\n\nSCHEMAS:\n{schemas_text}"
    ),
    "caller": (
        "Emit the call in BFCL canonical JSON. Output a SINGLE fenced ```json "
        "block containing a list of dicts where each key is the function name "
        "and each value is the arg dict.\n\nUSER REQUEST:\n{user_request}\n\n"
        "SCHEMAS:\n{schemas_text}"
    ),
    "verifier": (
        "Validate the Caller's canonical JSON against the schema(s). If "
        "correct, re-emit the SAME JSON. If wrong, emit corrected canonical "
        "JSON. Output a SINGLE fenced ```json``` block as the FINAL answer.\n\n"
        "USER REQUEST:\n{user_request}\n\nSCHEMAS:\n{schemas_text}"
    ),
}


def _merge_dict(a: dict | None, b: dict | None) -> dict:
    out = dict(a or {})
    out.update(b or {})
    return out


class SequentialState(TypedDict, total=False):
    inputs: dict
    by_stage: Annotated[dict, _merge_dict]
    messages: Annotated[list, operator.add]


def _escape_braces(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")


def _format_user(template: str, inputs: dict, by_stage: dict, prior_roles: list[str]) -> str:
    body = template.format(**inputs)
    for role in prior_roles:
        body += f"\n\n--- PRIOR STAGE: {role} ---\n{by_stage.get(role, '')}"
    return body


class SequentialBFCLAdapter:
    topology = TOPOLOGY
    dataset = DATASET

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        model_factory: Callable[[int], Any] | None = None,
    ):
        self._prompts = load_prompts(TOPOLOGY, ROLES, prompts)
        self.model_factory = model_factory or default_chat_model
        self._compiled = None

    def roles(self) -> list[str]:
        return list(ROLES)

    def get_prompt(self, role: str) -> str:
        self._check_role(role)
        return self._prompts[role]

    def set_prompt(self, role: str, text: str) -> None:
        self._check_role(role)
        self._prompts[role] = text
        self.reset()

    def reset(self) -> None:
        self._compiled = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_compiled"] = None
        return state

    def run_example(self, example: Any) -> dict:
        instance = coerce_instance(example)
        compiled = self._compiled_graph()
        inputs = {
            "user_request": _escape_braces(flatten_user_request(instance["question"])),
            "schemas_text": _escape_braces(schemas_text(instance)),
        }
        result = compiled.invoke({"inputs": inputs, "by_stage": {}, "messages": []})
        by_stage = result.get("by_stage") or {}
        model_output = extract_canonical(by_stage.get("verifier", "")) or extract_canonical(
            by_stage.get("caller", "")
        ) or []
        return {
            "model_output": model_output,
            "winner": "verifier",
            "buckets": [],
            "by_role": by_stage,
        }

    def format_role_trace(self, role: str, output: Any) -> str:
        self._check_role(role)
        if not isinstance(output, dict):
            return str(output)
        role_text = (output.get("by_role") or {}).get(role, "")
        return compact_role_trace(
            role=role,
            model_output=output.get("model_output") or [],
            winner=output.get("winner"),
            buckets=output.get("buckets") or [],
            details=[f"{role}_text={str(role_text)[:1200]}"],
        )

    def describe_runtime(self, example: Any | None = None) -> dict:
        instance = coerce_instance(example) if example is not None else {}
        return {
            "topology": self.topology,
            "dataset": self.dataset,
            "roles": self.roles(),
            "tool_count": len(instance.get("function") or []),
            "prompt_prefix": self.get_prompt(ROLES[0])[:80],
        }

    def _compiled_graph(self):
        if self._compiled is None:
            self._compiled = self._build_graph().compile()
        return self._compiled

    def _build_graph(self):
        graph = StateGraph(SequentialState)
        prior: list[str] = []
        for idx, role in enumerate(ROLES):
            graph.add_node(role, self._make_node(role, idx, list(prior)))
            prior.append(role)
        graph.add_edge(START, ROLES[0])
        for src, dst in zip(ROLES, ROLES[1:]):
            graph.add_edge(src, dst)
        graph.add_edge(ROLES[-1], END)
        return graph

    def _make_node(self, role: str, seed: int, prior_roles: list[str]):
        template = TASK_DESCRIPTIONS[role]

        def node(state: SequentialState) -> dict:
            user = _format_user(template, state["inputs"], state.get("by_stage") or {}, prior_roles)
            ai = self.model_factory(seed).invoke(
                [SystemMessage(content=execution_prompt(self._prompts[role], TOPOLOGY, role)), HumanMessage(content=user)]
            )
            return {"by_stage": {role: ai.content or ""}, "messages": [ai]}

        return node

    @staticmethod
    def _check_role(role: str) -> None:
        if role not in ROLES:
            raise KeyError(f"Unknown role {role!r}; expected one of {ROLES}")
