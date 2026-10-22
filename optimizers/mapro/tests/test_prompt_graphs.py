"""Prompt graphs, per-role contexts, blame and judged MAP selection for the four multi-agent topologies."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from optimizers.protocol.rollouts import native_role_order

from .. import native
from ..integration import _SeededAsyncChat, _SeededProbe
from ..src.mas.graph import Edge
from ..src.reward.demos import DemoPool
from .fakes import MARK, CentralizedAdapter, FakeBackend, SequentialAdapter, mapro_cell, mapro_runtime

SEEDS = {
    "manager": "m",
    "coder_worker": "c",
    "tester_worker": "t",
    "planner": "p",
    "writer": "w",
    "checker": "k",
    "solver": "s",
    "debater": "d",
}


def _graph(topology, roles, team_size=4):
    return native.build_prompt_graph(list(roles), {r: SEEDS[r] for r in roles}, topology, team_size)


def test_centralized_graph_is_a_manager_star():
    graph = _graph("centralized", ["manager", "coder_worker", "tester_worker"])
    assert graph.edges == [Edge("manager", "coder_worker"), Edge("manager", "tester_worker")]
    assert graph.output_agent == "manager" and graph.children("manager") == ["coder_worker", "tester_worker"]
    assert getattr(graph, "factor_edges", graph.edges) == graph.edges
    assert getattr(graph, "tied_replicas", 1) == 1 and graph.topo_order()[0] == "manager"


def test_sequential_graph_chains_the_native_stage_order():
    graph = _graph("sequential", ["planner", "writer", "checker"])
    assert graph.edges == [Edge("planner", "writer"), Edge("writer", "checker")]
    assert graph.output_agent == "checker" and graph.topo_order() == ["planner", "writer", "checker"]
    assert (graph.tied_replicas, graph.tied_peer_edges, graph.factor_edges) == (1, 0, graph.edges)


@pytest.mark.parametrize(
    "topology,team_size,peers,factors",
    [
        ("independent", 4, 0, []),
        ("independent", 8, 0, []),
        ("decentralized", 4, 12, [Edge("debater", "debater")]),
        ("decentralized", 2, 2, [Edge("debater", "debater")]),
    ],
)
def test_shared_prompt_graphs_tie_replicas_and_keep_peer_factors(topology, team_size, peers, factors):
    role = "solver" if topology == "independent" else "debater"
    graph = _graph(topology, [role], team_size)
    assert graph.edges == [] and graph.output_agent == role
    assert (graph.tied_replicas, graph.tied_peer_edges, graph.factor_edges) == (team_size, peers, factors)


def test_unsupported_prompt_variable_layout_is_refused():
    with pytest.raises(ValueError, match="unsupported native prompt-variable topology"):
        _graph("independent", ["solver", "writer"])
    with pytest.raises(ValueError):
        _graph("ring", ["solver"])


def _record(messages):
    return {
        "score": 0.0,
        "gold": "",
        "got": "wrong",
        "messages": [{"source": "user", "content": "the task"}, *messages],
    }


def test_topology_contexts_follow_native_visibility():
    central = _graph("centralized", ["manager", "coder_worker", "tester_worker"])
    record = _record(
        [
            {
                "source": "manager",
                "content": "plan",
                "tool_calls": [{"name": "delegate_to_coder_worker", "args": {"instructions": "write it"}}],
            },
            {"source": "coder_worker", "content": "code"},
            {"source": "tester_worker", "content": "tests"},
        ]
    )
    contexts = native.topology_contexts(record, central, "centralized")
    assert contexts["manager"] == "the task"
    assert contexts["coder_worker"].endswith("MANAGER INSTRUCTIONS TO coder_worker:\nwrite it")
    assert contexts["tester_worker"].endswith("MANAGER INSTRUCTIONS TO tester_worker:\nplan")  # fallback
    chain = _graph("sequential", ["planner", "writer", "checker"])
    messages = [
        {"source": "planner", "content": "p"},
        {"source": "writer", "content": "w"},
        {"source": "checker", "content": "k"},
    ]
    contexts = native.topology_contexts(_record(messages), chain, "sequential")
    visible = {role: json.loads(text.split("CONTEXT:\n", 1)[1]) for role, text in contexts.items()}
    assert [m["source"] for m in visible["planner"]] == ["user"]
    assert [m["source"] for m in visible["checker"]] == ["user", "planner", "writer"]
    peers = [{"source": "debater", "content": "view"}]
    assert (
        json.loads(
            native.topology_contexts(_record(peers), _graph("decentralized", ["debater"]), "decentralized")[
                "debater"
            ].split("CONTEXT:\n", 1)[1]
        )[-1]
        == peers[0]
    )
    assert native.topology_contexts(_record(peers), _graph("independent", ["solver"]), "independent")[
        "solver"
    ].endswith("CONTEXT:\n[]")


def test_blame_targets_parents_only_and_only_on_failures():
    backend = FakeBackend()
    shim = _SeededAsyncChat(mapro_cell("sequential"), backend, "mapro_reflection", thinking=True)
    chain = _graph("sequential", ["planner", "writer", "checker"])
    messages = [{"source": r, "content": f"{r} out"} for r in ("planner", "writer", "checker")]
    feedback = asyncio.run(native.topology_blame(shim, chain, "sequential", _record(messages)))
    assert "INCORRECTLY" in feedback.f_g and feedback.blames["checker"] == ""
    assert feedback.blames["planner"].startswith("(from writer)") and feedback.blames["writer"].startswith(
        "(from checker)"
    )
    star = _graph("centralized", ["manager", "coder_worker", "tester_worker"])
    record = _record([{"source": "manager", "content": "go"}, {"source": "coder_worker", "content": "c"}])
    feedback = asyncio.run(native.topology_blame(shim, star, "centralized", record))
    assert feedback.blames["manager"].startswith("(from coder_worker)")  # tester never activated
    assert feedback.blames["coder_worker"] and feedback.blames["tester_worker"]
    calls = len(backend.requests)
    solved = asyncio.run(native.topology_blame(shim, chain, "sequential", dict(_record(messages), score=1.0)))
    assert len(backend.requests) == calls and not any(solved.blames.values())


@pytest.mark.parametrize(
    "topology,roles,node_calls,edge_calls",
    [
        ("centralized", ["manager", "coder_worker", "tester_worker"], 3 * 3 * 2, 2 * 3 * 3 * 2),
        ("sequential", ["planner", "writer", "checker"], 3 * 3 * 2, 2 * 3 * 3 * 2),
        ("independent", ["solver"], 3 * 2, 0),
        ("decentralized", ["debater"], 3 * 2, 3 * 2),
    ],
)
def test_stage2_selects_the_judged_map_assignment(topology, roles, node_calls, edge_calls):
    cell = mapro_cell(topology)
    backend = FakeBackend()
    judge = _SeededAsyncChat(cell, backend, "mapro_node_edge_judge")
    probe = _SeededProbe(cell, backend, 2)
    graph = _graph(topology, roles)
    pools = {r: [SEEDS[r], f"{MARK} {r} a", f"{MARK} {r} b"] for r in roles}
    contexts = [{r: f"context {t} for {r}" for r in roles} for t in range(2)]
    Y = {r: [[probe.complete(cand, contexts[t][r]) for t in range(2)] for cand in pools[r]] for r in roles}
    backend.requests.clear()
    assignment, result, node_scores, edge_scores = asyncio.run(
        native.stage2_select(None, judge, graph, pools, contexts, Y, {r: DemoPool() for r in roles})
    )
    judged = [r for r in backend.requests if r["phase"] == "mapro_node_edge_judge"]
    assert len(judged) == node_calls + edge_calls
    assert all((r["thinking"], r["max_output_tokens"], r["temperature"]) == (False, 8, 0.2) for r in judged)
    assert assignment == {r: f"{MARK} {r} a" for r in roles} and set(result.assignment.values()) == {1}
    assert all(np.allclose(node_scores[r], [0.10, 0.95, 0.95]) for r in roles)
    if topology == "decentralized":
        diagonal = edge_scores[("debater", "debater")]
        assert np.allclose(np.diag(diagonal), [0.10, 0.95, 0.95]) and diagonal[0, 1] == 0.0  # off-diagonal unscored
        assert result.log_score == pytest.approx(4 * np.log(0.95) + 12 * np.log(0.95))
    if topology == "independent":
        assert edge_scores == {} and result.log_score == pytest.approx(4 * np.log(0.95))


def test_sequential_roles_use_the_native_stage_order_and_manager_leads_centralized():
    sequential = mapro_cell("sequential")
    runtime = mapro_runtime(sequential)
    bundle = runtime.seed_bundle()
    assert list(bundle.roles) == sorted(SequentialAdapter.ROLES)
    assert native_role_order(sequential, list(bundle.roles)) == list(SequentialAdapter.ROLES)
    central = mapro_cell("centralized")
    roles = native_role_order(central, list(mapro_runtime(central).seed_bundle().roles))
    assert roles[0] != "manager" and set(roles) == set(CentralizedAdapter.ROLES)  # sorted bundle order
