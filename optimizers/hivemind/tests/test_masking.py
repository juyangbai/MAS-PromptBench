"""Coalition masking through the protocol runner's ``optimizer_control`` hook, and its verification."""

from __future__ import annotations

import pytest

from optimizers.bridge.adapters import centralized_bfcl, module_hotpotqa, module_lcb, sequential_bfcl
from optimizers.bridge.adapters.decentralized_bfcl import DecentralizedBFCLAdapter
from optimizers.bridge.adapters.independent_bfcl import IndependentBFCLAdapter
from optimizers.bridge.adapters.module_common import import_isolated_real_module
from optimizers.protocol.errors import NativeIntegrationError, RunnerContractError
from optimizers.protocol.schema import CellSpec, PromptBundle, RunRecord

from .. import coalition
from ..optimizer import _require_control_ack
from ..runtime import CoalitionExecutionHook, coalition_execution, runtime_support, validated_control, verify_mask
from ..topology import TOPOLOGY_ACK_SCHEMA, coalition_game, expected_control_evidence
from .fakes import (
    BFCL_WORKERS,
    Script,
    ScriptedChatModel,
    TelemetryHook,
    bfcl_runtime,
    fake_examples,
    hivemind_cell,
    hotpotqa_runtime,
    protocol_runner,
)

HOTPOTQA_ROLES = ("manager", "reasoner_worker", "retriever_worker", "writer_worker")


def _controlled(runner, control) -> PromptBundle:
    seed = runner.seed_bundle
    return PromptBundle(roles=dict(seed.roles), metadata={**dict(seed.metadata), "optimizer_control": control})


def _sources(record: RunRecord) -> set[str]:
    return {str(message.get("source")) for message in record.messages}


def _bfcl_instance() -> dict:
    instance = dict(fake_examples("bfcl")[2]["task_instance"])
    instance.pop("ground_truth")
    return instance


# Control requests
def test_control_requests_must_be_canonical_partitions_inside_the_frozen_scope():
    cell = hivemind_cell("hotpotqa", "centralized")
    game = coalition_game("centralized", HOTPOTQA_ROLES)
    control = game.control(["retriever_worker"])
    assert control["masked_workers"] == ["reasoner_worker", "writer_worker"]
    assert validated_control(cell, HOTPOTQA_ROLES, dict(control)) == control
    assert runtime_support(cell, HOTPOTQA_ROLES) == (True, "centralized_native_player_coalition")
    for other, reason in [
        (CellSpec(method="gepa", task="hotpotqa", topology="centralized", framework="langgraph"), "not_hivemind"),
        (
            CellSpec(method="hivemind", task="hotpotqa", topology="centralized", framework="langgraph", team_size=8),
            "outside_frozen_hivemind_scope",
        ),
        (
            CellSpec(method="hivemind", task="math", topology="centralized", framework="langgraph"),
            "outside_frozen_hivemind_scope",
        ),
        (
            CellSpec(method="hivemind", task="hotpotqa", topology="centralized", framework="autogen"),
            "outside_frozen_hivemind_scope",
        ),
    ]:
        assert runtime_support(other, HOTPOTQA_ROLES) == (False, reason)
        with pytest.raises(RunnerContractError, match="unavailable"):
            validated_control(other, HOTPOTQA_ROLES, control)
    for bad, message in [
        ("not a mapping", "must be a mapping"),
        ({**control, "active_workers": ["retriever_worker", "retriever_worker"]}, "duplicates"),
        ({**control, "masked_workers": ["writer_worker"]}, "exact role partition"),
        ({**control, "active_workers": "retriever_worker"}, "invalid"),
        ({**control, "extra": True}, "not canonical"),
        ({**control, "schema": "other"}, "not canonical"),
    ]:
        with pytest.raises(RunnerContractError, match=message):
            validated_control(cell, HOTPOTQA_ROLES, bad)
    with pytest.raises(RunnerContractError, match="topology_runtime_role_contract_invalid"):
        validated_control(hivemind_cell("hotpotqa", "sequential"), ("planner", "writer"), control)


# Centralized masking mechanics on the real adapters and modules
@pytest.mark.parametrize(
    "module_name,masked_cls,patch_name,other_tool",
    [
        (
            "topologies.centralized.langgraph.hotpotqa.langgraph_hotpotqa",
            coalition.MaskedCentralizedHotpotQAAdapter,
            "_patched_module",
            "wikipedia_search",
        ),
        (
            "topologies.centralized.langgraph.lcb.langgraph_lcb",
            coalition.MaskedCentralizedLCBAdapter,
            "patched_module",
            "python_exec",
        ),
    ],
)
def test_module_mask_removes_masked_delegation_tools_and_restores_the_module(
    module_name, masked_cls, patch_name, other_tool
):
    module = import_isolated_real_module(module_name)
    masked_worker = masked_cls.roles_[2]
    adapter = masked_cls(masked_workers=frozenset({masked_worker}))
    originals = {
        name: getattr(module, name)
        for name in (
            "DELEGATION_TOOLS",
            "DELEGATION_NAMES",
            "MANAGER_TOOLS",
            "_manager_tool_node",
            "_MANAGER_TERMINATE_NUDGE",
            "_load_prompt",
        )
    }
    with getattr(adapter, patch_name)(module):
        assert module._load_prompt is not originals["_load_prompt"]  # the runtime's own patches still apply
        assert f"delegate_to_{masked_worker}" not in module.DELEGATION_NAMES
        assert len(module.DELEGATION_NAMES) == 2
        names = [tool.name for tool in module.MANAGER_TOOLS]
        assert other_tool in names and f"delegate_to_{masked_worker}" not in names
        assert set(module._manager_tool_node.tools_by_name) == set(names)
        assert "DO NOT EXIST" in module._MANAGER_TERMINATE_NUDGE
    assert all(getattr(module, name) is value for name, value in originals.items())
    game = coalition_game("centralized", masked_cls.roles_)
    expected = expected_control_evidence(game.control(sorted(set(masked_cls.roles_[1:]) - {masked_worker})))
    assert adapter.coalition_control_evidence() == expected
    with pytest.raises(ValueError, match="unknown masked workers"):
        masked_cls(masked_workers=frozenset({"manager"}))


def test_masked_centralized_bfcl_binds_builds_and_routes_only_active_workers():
    script = Script(BFCL_WORKERS)
    adapter = coalition.build_masked_runner(
        centralized_bfcl.CentralizedBFCLAdapter, frozenset({"caller_worker"}), model_factory=script.model
    )
    assert isinstance(adapter, coalition.MaskedCentralizedBFCLAdapter)
    assert "caller_worker" not in adapter._build_graph().nodes
    output = adapter.run_example(_bfcl_instance())
    sources = {message["source"] for message in output["messages"]}
    assert {"inspector_worker", "validator_worker"} <= sources and "caller_worker" not in sources
    manager_tools = [call["tools"] for call in script.calls if call["role"] == "manager"]
    assert all(tools == ("delegate_to_inspector_worker", "delegate_to_validator_worker") for tools in manager_tools)
    # The manager still tried the masked tool once: it bounced back to the manager.
    assert script.roles_called().count("manager") == 4
    assert "worker:caller_worker" not in script.roles_called()
    game = coalition_game("centralized", centralized_bfcl.ROLES)
    assert adapter.coalition_control_evidence() == expected_control_evidence(
        game.control(["inspector_worker", "validator_worker"])
    )
    unmasked_script = Script(BFCL_WORKERS)
    plain = centralized_bfcl.CentralizedBFCLAdapter(model_factory=unmasked_script.model)
    assert "caller_worker" in {message["source"] for message in plain.run_example(_bfcl_instance())["messages"]}
    with pytest.raises(ValueError, match="no coalition-masked runner"):
        coalition.build_masked_runner(sequential_bfcl.SequentialBFCLAdapter, frozenset())
    assert set(coalition.MASKED_CENTRALIZED) == {
        centralized_bfcl.CentralizedBFCLAdapter,
        module_hotpotqa.CentralizedHotpotQAAdapter,
        module_lcb.CentralizedLCBAdapter,
    }


# Through the protocol runner
@pytest.mark.parametrize(
    "active",
    [
        ["reasoner_worker", "retriever_worker", "writer_worker"],
        ["retriever_worker"],
        [],
    ],
)
def test_hook_executes_and_acknowledges_centralized_coalitions(hotpotqa_script, active):
    cell = hivemind_cell("hotpotqa", "centralized", budget=20)
    runner, budget, data = protocol_runner(cell, hotpotqa_runtime(cell, module_hotpotqa.CentralizedHotpotQAAdapter))
    control = coalition_game("centralized", runner.required_roles).control(active)
    with coalition_execution():
        record = runner.run(data.row("tr2"), _controlled(runner, control), 11)
    assert record.usable and budget.snapshot()["charged"] == 1
    _require_control_ack(record, control)
    ack = record.metadata["runtime_metadata"]["execution_control"]
    assert ack["masked_workers"] == control["masked_workers"] and ack["allow_zero_model_calls"] is False
    assert ack["bound_delegation_tools"] == [f"delegate_to_{worker}" for worker in active]
    assert not _sources(record) & set(control["masked_workers"])
    assert set(active) <= _sources(record)
    for tools in (call["tools"] for call in hotpotqa_script.calls if call["role"] == "manager"):
        assert set(tools) == {"wikipedia_search", "wikipedia_page", *(f"delegate_to_{w}" for w in active)}
    assert record.score == (1.0 if "retriever_worker" in active else 0.0)
    assert record.usage.model_calls == len(hotpotqa_script.calls)


@pytest.mark.parametrize(
    "variant,message",
    [
        ("unmasked", "masked workers executed"),
        ("different_mask", "evidence differs"),
        ("never_executed", "evidence unavailable"),
    ],
)
def test_unmasked_or_misreported_execution_is_never_acknowledged(hotpotqa_script, monkeypatch, variant, message):
    cell = hivemind_cell("hotpotqa", "centralized", budget=20)
    runner, budget, data = protocol_runner(cell, hotpotqa_runtime(cell, module_hotpotqa.CentralizedHotpotQAAdapter))
    control = coalition_game("centralized", runner.required_roles).control(["retriever_worker", "writer_worker"])
    built = {
        "unmasked": lambda base, masked, **kw: base(**kw),
        "different_mask": lambda base, masked, **kw: coalition.MaskedCentralizedHotpotQAAdapter(
            masked_workers=frozenset({"reasoner_worker", "writer_worker"})
        ),
        "never_executed": lambda base, masked, **kw: _NeverRuns(masked_workers=masked),
    }[variant]
    monkeypatch.setattr(coalition, "build_masked_runner", built)
    with coalition_execution():
        record = runner.run(data.row("tr2"), _controlled(runner, control), 11)
    assert record.status == "infrastructure_failure" and record.metadata["failure_stage"] == "runtime_contract"
    assert message in record.error and record.metadata["execution_attempts"] == 3
    snapshot = budget.snapshot()
    assert (snapshot["charged"], snapshot["infrastructure_failures"], snapshot["reserved"]) == (0, 3, 0)
    with pytest.raises(NativeIntegrationError, match="did not acknowledge"):
        _require_control_ack(record, control)


class _NeverRuns(coalition.MaskedCentralizedHotpotQAAdapter):
    """Answers without executing the masked graph, so it has no coalition evidence."""

    def run_example(self, example):
        return {
            "answer": "value-tr2",
            "messages": [{"source": "manager", "content": "Answer: value-tr2"}],
            "telemetry": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "n_llm_calls": 1,
                "n_tool_calls": 0,
            },
        }


def test_controlled_bundles_require_the_hook_and_records_require_the_exact_ack(hotpotqa_script):
    cell = hivemind_cell("hotpotqa", "centralized", budget=20)
    runner, budget, data = protocol_runner(cell, hotpotqa_runtime(cell, module_hotpotqa.CentralizedHotpotQAAdapter))
    game = coalition_game("centralized", runner.required_roles)
    control = game.control(["retriever_worker"])
    with pytest.raises(RunnerContractError, match="no execution hook"):
        runner.run(data.row("tr2"), _controlled(runner, control), 11)
    hook = CoalitionExecutionHook()
    with pytest.raises(RunnerContractError, match="not built by this execution hook"):
        hook.verify(runner.executor.runtime, object(), control, {})
    with pytest.raises(RunnerContractError, match="masked workers executed"):
        verify_mask({"messages": [{"source": "writer_worker", "content": "x"}]}, control)
    verify_mask({"messages": [{"source": "writer_worker", "content": "x"}]}, game.control(game.players))
    assert budget.snapshot()["attempted"] == 0 and hotpotqa_script.calls == []
    with coalition_execution(hook):
        record = runner.run(data.row("tr2"), _controlled(runner, control), 11)
    _require_control_ack(record, control)
    for wrong in (game.control(["writer_worker"]), game.control([])):
        with pytest.raises(NativeIntegrationError):
            _require_control_ack(record, wrong)
    bare = RunRecord(cell_id=cell.cell_id, example_id="tr2", status="success", score=1.0)
    with pytest.raises(NativeIntegrationError):
        _require_control_ack(bare, control)


@pytest.mark.parametrize(
    "topology,adapter_class,active,seeds",
    [
        ("sequential", sequential_bfcl.SequentialBFCLAdapter, ["analyzer", "caller"], [0, 2]),
        ("independent", IndependentBFCLAdapter, ["caller::1", "caller::3"], [1, 3]),
        ("decentralized", DecentralizedBFCLAdapter, ["debater::0", "debater::2"], [0, 2]),
    ],
)
def test_hook_masks_native_players_of_noncentralized_topologies(topology, adapter_class, active, seeds):
    script = Script()
    cell = hivemind_cell("bfcl", topology, budget=20)
    runner, budget, data = protocol_runner(cell, bfcl_runtime(cell, adapter_class, script))
    control = coalition_game(topology, runner.required_roles).control(active)
    with coalition_execution(TelemetryHook(script)):
        record = runner.run(data.row("tr2"), _controlled(runner, control), 11)
    assert record.status == "success" and budget.snapshot()["charged"] == 1
    _require_control_ack(record, control)
    ack = record.metadata["runtime_metadata"]["execution_control"]
    assert ack["ack_schema"] == TOPOLOGY_ACK_SCHEMA and ack["topology"] == topology
    assert sorted({call["seed"] for call in script.calls}) == seeds
    output = record.final_output
    if topology == "sequential":
        assert list(output["by_role"]) == active and output["winner"] == "caller"
    elif topology == "independent":
        assert [agent["agent_id"] for agent in output["per_agent"]] == seeds
    else:
        assert [peer["peer"] for peer in output["per_peer"]] == seeds and output["winner"] == 0
    assert output["model_output"] == [{"answer": {"value": "value-tr2"}}]


def test_sequential_module_mask_skips_masked_stages(hotpotqa_script):
    cell = hivemind_cell("hotpotqa", "sequential", budget=20)
    runner, budget, data = protocol_runner(cell, hotpotqa_runtime(cell, module_hotpotqa.SequentialHotpotQAAdapter))
    control = coalition_game("sequential", runner.required_roles).control(["planner", "retriever"])
    with coalition_execution():
        record = runner.run(data.row("tr2"), _controlled(runner, control), 11)
    _require_control_ack(record, control)
    assert list(record.final_output["runner_output"]["by_stage"]) == ["planner", "retriever"]
    assert len(hotpotqa_script.calls) == 2 and record.score == 1.0
    assert record.final_output["answer"] == "value-tr2"


@pytest.mark.parametrize(
    "topology,adapter_class,players",
    [
        ("independent", module_hotpotqa.IndependentHotpotQAAdapter, ("per_agent", "agent_id")),
        ("decentralized", module_hotpotqa.DecentralizedHotpotQAAdapter, ("per_peer", "peer")),
    ],
)
def test_module_backed_ensembles_run_only_the_active_replicas(
    hotpotqa_script, monkeypatch, topology, adapter_class, players
):
    import langchain_openai

    class ScriptedChatOpenAI(ScriptedChatModel):
        def __init__(self, **kwargs):
            super().__init__(script=hotpotqa_script, seed=int(kwargs.get("seed") or 0))

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", ScriptedChatOpenAI)
    cell = hivemind_cell("hotpotqa", topology, budget=20)
    runner, budget, data = protocol_runner(cell, hotpotqa_runtime(cell, adapter_class))
    role = runner.required_roles[0]
    control = coalition_game(topology, runner.required_roles).control([f"{role}::1", f"{role}::3"])
    with coalition_execution(TelemetryHook(hotpotqa_script)):
        record = runner.run(data.row("tr2"), _controlled(runner, control), 11)
    _require_control_ack(record, control)
    key, identity = players
    assert [entry[identity] for entry in record.final_output["runner_output"][key]] == [1, 3]
    if topology == "independent":
        assert sorted(call["seed"] for call in hotpotqa_script.calls) == [1, 3]
    else:
        assert len(hotpotqa_script.calls) == 2 * 2 and record.final_output["winner"] in {1, 3}
    assert record.score == 1.0 and budget.snapshot()["charged"] == 1


def test_empty_noncentralized_coalition_is_a_charged_zero_call_abstention():
    script = Script()
    cell = hivemind_cell("bfcl", "sequential", budget=20)
    runner, budget, data = protocol_runner(cell, bfcl_runtime(cell, sequential_bfcl.SequentialBFCLAdapter, script))
    control = coalition_game("sequential", runner.required_roles).control([])
    with coalition_execution(TelemetryHook(script)):
        record = runner.run(data.row("tr2"), _controlled(runner, control), 11)
    assert script.calls == [] and record.usage.model_calls == 0
    assert (record.status, record.score) == ("semantic_failure", 0.0)
    assert record.final_output["runner_output"]["coalition_abstention"] is True
    assert record.metadata["runtime_metadata"]["execution_control"]["allow_zero_model_calls"] is True
    assert budget.snapshot()["charged"] == 1
    _require_control_ack(record, control)
