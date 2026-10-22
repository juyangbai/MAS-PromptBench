"""A runner applies the communication format it is loaded with (``COMMUNICATION_FORMAT``, see ``core.variant``)."""

import importlib
import sys

import pytest

from core import variant
from core.communication import append_contract

SEQUENTIAL_HOTPOTQA = "topologies.sequential.langgraph.hotpotqa.langgraph_hotpotqa"
SEQUENTIAL_APIBANK = "topologies.sequential.langgraph.apibank.langgraph_apibank"

# Each prompt loader a runner may have: role-only, API-Bank's four arguments, ToolHop's
# ``_system_prompt``, and a module-level prompt completed after its output nudge.
PROMPTS = {
    "role-only _load_prompt": (SEQUENTIAL_HOTPOTQA, lambda runner: runner._load_prompt("planner")),
    "API-Bank _load_prompt": (
        SEQUENTIAL_APIBANK,
        lambda runner: runner._load_prompt("sequential", "verifier", "sequential_langgraph", "Be brief."),
    ),
    "ToolHop _system_prompt": (
        "topologies.centralized.langgraph.toolhop.langgraph_toolhop",
        lambda runner: runner._system_prompt("centralized", "manager", "centralized_langgraph"),
    ),
    "SYSTEM_PROMPT": (
        "topologies.decentralized.langgraph.hotpotqa.langgraph_hotpotqa",
        lambda runner: runner.SYSTEM_PROMPT,
    ),
}


@pytest.mark.parametrize("fmt", ["semi_structured", "structured_soft"])
@pytest.mark.parametrize("loader", list(PROMPTS))
def test_each_prompt_loader_appends_the_contract(loader, fmt):
    name, prompt = PROMPTS[loader]
    plain = importlib.import_module(name)
    preset = variant.module(name, COMMUNICATION_FORMAT=fmt)
    assert prompt(preset) == append_contract(prompt(plain), fmt, plain.task.DATASET) != prompt(plain)
    assert prompt(variant.module(name, COMMUNICATION_FORMAT="freeform")) == prompt(plain)


def test_two_formats_in_one_process_keep_their_own_prompts_and_handoffs():
    plain = importlib.import_module(SEQUENTIAL_HOTPOTQA)
    semi = variant.module(SEQUENTIAL_HOTPOTQA, COMMUNICATION_FORMAT="semi_structured")
    soft = variant.module(SEQUENTIAL_HOTPOTQA, COMMUNICATION_FORMAT="structured_soft")
    assert "[STATUS]" in semi._load_prompt("writer") and "JSON_REPORT:" not in semi._load_prompt("writer")
    assert "JSON_REPORT:" in soft._load_prompt("writer") and "[STATUS]" not in soft._load_prompt("writer")
    assert semi._format_stage_handoff("planner", "Plan.").startswith("[STATUS]\ncompleted")
    assert soft._format_stage_handoff("planner", "Plan.").startswith("JSON_REPORT:\n")
    assert plain._format_stage_handoff("planner", "Plan.") == "Plan."
    assert (plain.COMMUNICATION_FORMAT, plain.COMMUNICATION.fmt) == ("freeform", "freeform")


def test_tool_use_runners_tell_a_freeform_run_from_a_plain_one():
    plain = importlib.import_module(SEQUENTIAL_APIBANK)
    freeform = variant.module(SEQUENTIAL_APIBANK, COMMUNICATION_FORMAT="freeform")
    assert plain.COMMUNICATION_FORMAT is None and freeform.COMMUNICATION_FORMAT == "freeform"
    assert plain.COMMUNICATION == freeform.COMMUNICATION


def test_communications_entries_run_their_own_runner():
    semi = importlib.import_module("communications.sequential.hotpotqa.hotpotqa_semi_structured")
    soft = importlib.import_module("communications.sequential.hotpotqa.hotpotqa_structured_soft")
    assert semi.BASE_MODULE != soft.BASE_MODULE
    assert sys.modules[semi.BASE_MODULE].COMMUNICATION.fmt == semi.COMMUNICATION_FORMAT == "semi_structured"
    assert sys.modules[soft.BASE_MODULE].COMMUNICATION.fmt == soft.COMMUNICATION_FORMAT == "structured_soft"
    assert importlib.import_module(SEQUENTIAL_HOTPOTQA).COMMUNICATION_FORMAT == "freeform"
