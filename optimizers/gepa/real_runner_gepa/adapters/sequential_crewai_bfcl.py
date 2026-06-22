"""Sequential/CrewAI BFCL real-runner adapter."""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from real_runner_gepa.adapters.bfcl_common import (
    coerce_instance,
    execution_prompt,
    compact_role_trace,
    extract_canonical,
    flatten_user_request,
    load_prompts,
    schemas_text,
)
from real_runner_gepa.lm import TASK_MODEL, next_task_endpoint


TOPOLOGY = "sequential_crewai"
PROMPT_TOPOLOGY = "sequential"
DATASET = "bfcl"
ROLES = ["analyzer", "inspector", "caller", "verifier"]


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


class SequentialCrewAIBFCLAdapter:
    topology = TOPOLOGY
    dataset = DATASET

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        llm_factory: Callable[[], Any] | None = None,
    ):
        self._prompts = load_prompts(PROMPT_TOPOLOGY, ROLES, prompts)
        self.llm_factory = llm_factory or self._default_llm_factory

    def roles(self) -> list[str]:
        return list(ROLES)

    def get_prompt(self, role: str) -> str:
        self._check_role(role)
        return self._prompts[role]

    def set_prompt(self, role: str, text: str) -> None:
        self._check_role(role)
        self._prompts[role] = text

    def reset(self) -> None:
        return None

    def __getstate__(self):
        return self.__dict__.copy()

    def run_example(self, example: Any) -> dict:
        instance = coerce_instance(example)
        crew = self._build_crew()
        result = crew.kickoff(
            inputs={
                "user_request": _escape_braces(flatten_user_request(instance["question"])),
                "schemas_text": _escape_braces(schemas_text(instance)),
            }
        )
        stages = {}
        try:
            stages["analyzer"] = result.tasks_output[0].raw
            stages["inspector"] = result.tasks_output[1].raw
            stages["caller"] = result.tasks_output[2].raw
            stages["verifier"] = result.tasks_output[3].raw
        except (AttributeError, IndexError):
            stages = {
                "analyzer": "",
                "inspector": "",
                "caller": "",
                "verifier": getattr(result, "raw", ""),
            }

        final = getattr(result, "raw", "")
        model_output = extract_canonical(stages.get("verifier", "") or final)
        if model_output is None:
            model_output = extract_canonical(stages.get("caller", ""))
        return {
            "model_output": model_output or [],
            "winner": "verifier",
            "buckets": [],
            "by_role": stages,
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
            "framework": "crewai",
            "roles": self.roles(),
            "tool_count": len(instance.get("function") or []),
            "prompt_prefix": self.get_prompt("analyzer")[:80],
        }

    def _build_crew(self):
        from crewai import Agent, Crew, Process, Task

        llm = self.llm_factory()
        analyzer = Agent(
            role="BFCL Analyzer",
            goal="Parse the user request to extract intent, entities, and constraints.",
            backstory=execution_prompt(self._prompts["analyzer"], PROMPT_TOPOLOGY, "analyzer"),
            tools=[],
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )
        inspector = Agent(
            role="BFCL Inspector",
            goal="Map the request onto a specific function and argument plan.",
            backstory=execution_prompt(self._prompts["inspector"], PROMPT_TOPOLOGY, "inspector"),
            tools=[],
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )
        caller = Agent(
            role="BFCL Caller",
            goal="Emit BFCL canonical JSON calls matching the schema.",
            backstory=execution_prompt(self._prompts["caller"], PROMPT_TOPOLOGY, "caller"),
            tools=[],
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )
        verifier = Agent(
            role="BFCL Verifier",
            goal="Validate and emit the final BFCL canonical JSON calls.",
            backstory=execution_prompt(self._prompts["verifier"], PROMPT_TOPOLOGY, "verifier"),
            tools=[],
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        analyze_task = Task(
            description=(
                "Parse the user request below to extract the intent, entities, "
                "and any implicit constraints. Do NOT reference the schemas yet.\n\n"
                "USER REQUEST:\n{user_request}"
            ),
            expected_output="A structured summary of intent, entities, and constraints.",
            agent=analyzer,
        )
        inspect_task = Task(
            description=(
                "Given the Analyzer's summary and the function schema(s) below, map "
                "the intent onto a specific function and propose argument values.\n\n"
                "USER REQUEST:\n{user_request}\n\nSCHEMAS:\n{schemas_text}"
            ),
            expected_output="A plan with function name, arguments, and expected types.",
            agent=inspector,
            context=[analyze_task],
        )
        call_task = Task(
            description=(
                "Emit the call in BFCL canonical JSON. Follow the Inspector's plan "
                "exactly. Output a SINGLE fenced ```json block containing a list of "
                "dicts with one key per dict where the key is the function name and "
                "the value is the arg dict.\n\nUSER REQUEST:\n{user_request}\n\n"
                "SCHEMAS:\n{schemas_text}"
            ),
            expected_output="A fenced ```json block containing canonical call dicts.",
            agent=caller,
            context=[analyze_task, inspect_task],
        )
        verify_task = Task(
            description=(
                "Validate the Caller's canonical JSON against the schema(s). If "
                "correct, re-emit the SAME JSON. If wrong, emit corrected canonical "
                "JSON. Output a SINGLE fenced ```json``` block as the FINAL answer.\n\n"
                "USER REQUEST:\n{user_request}\n\nSCHEMAS:\n{schemas_text}"
            ),
            expected_output="A SINGLE fenced ```json block containing the final canonical call list.",
            agent=verifier,
            context=[analyze_task, inspect_task, call_task],
        )
        return Crew(
            agents=[analyzer, inspector, caller, verifier],
            tasks=[analyze_task, inspect_task, call_task, verify_task],
            process=Process.sequential,
            verbose=False,
        )

    @staticmethod
    def _default_llm_factory():
        from crewai import LLM

        return LLM(
            model=f"openai/{os.environ.get('MODEL_ID', TASK_MODEL)}",
            base_url=next_task_endpoint(),
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            temperature=0.2,
            top_p=0.9,
            seed=0,
            max_tokens=1024,
            extra_body={
                "repetition_penalty": 1.05,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    @staticmethod
    def _check_role(role: str) -> None:
        if role not in ROLES:
            raise KeyError(f"Unknown role {role!r}; expected one of {ROLES}")
