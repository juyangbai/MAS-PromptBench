"""MAP core: exact max-product BP, tied shared-prompt factors, demos and output parsing (no model calls)."""

from __future__ import annotations

import asyncio
import itertools

import numpy as np
import pytest

from ..native import tied_log_quality
from ..src.infer.bp import MAPProblem, brute_force_map, map_infer
from ..src.refine.mutate import _clean, _split_variants, init_candidates, mutate_pool
from ..src.reward.demos import Demo, DemoPool, critic_update
from ..src.reward.node_reward import node_score_listwise
from ..src.reward.parse import parse_score, parse_score_list


def _random_problem(rng, n, k, edge_prob=0.5):
    names = [f"v{i}" for i in range(n)]
    node = {v: rng.uniform(0.01, 1.0, size=k) for v in names}
    edges = {}
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                u, w = (names[i], names[j]) if rng.random() < 0.5 else (names[j], names[i])
                edges[(u, w)] = rng.uniform(0.01, 1.0, size=(k, k))
    return MAPProblem(names, {v: k for v in names}, node, edges)


def _score(problem, assignment):
    return sum(float(arr[tuple(assignment[x] for x in vs)]) for vs, arr in problem.log_factors())


def test_bp_matches_brute_force_on_random_graphs():
    rng = np.random.default_rng(12345)
    for _ in range(60):
        n, k = int(rng.integers(1, 6)), int(rng.integers(2, 5))
        problem = _random_problem(rng, n, k, edge_prob=float(rng.uniform(0.3, 0.9)))
        got, exact = map_infer(problem), brute_force_map(problem)
        assert abs(got.log_score - exact.log_score) < 1e-6
        assert abs(_score(problem, got.assignment) - exact.log_score) < 1e-6


def test_bp_on_small_hand_checked_star():
    # manager -> two workers, K=2. Node-only MAP would pick manager=0; the edges flip it.
    problem = MAPProblem(
        ["manager", "a", "b"],
        {"manager": 2, "a": 2, "b": 2},
        {"manager": np.array([0.9, 0.6]), "a": np.array([0.5, 0.5]), "b": np.array([0.8, 0.4])},
        {("manager", "a"): np.array([[0.1, 0.1], [0.9, 0.2]]), ("manager", "b"): np.array([[0.2, 0.1], [0.9, 0.9]])},
    )
    result = map_infer(problem)
    expected = max(
        itertools.product(range(2), repeat=3), key=lambda c: _score(problem, dict(zip(["manager", "a", "b"], c)))
    )
    assert tuple(result.assignment[v] for v in ("manager", "a", "b")) == expected == (1, 0, 0)
    assert result.treewidth == 1
    assert result.log_score == pytest.approx(np.log(0.6 * 0.5 * 0.8 * 0.9 * 0.9))


def test_bp_chain_triangle_and_single_node():
    rng = np.random.default_rng(7)
    names = ["solver", "reflector", "refiner"]
    problem = MAPProblem(
        names,
        {v: 5 for v in names},
        {v: rng.uniform(0.01, 1, 5) for v in names},
        {
            ("solver", "reflector"): rng.uniform(0.01, 1, (5, 5)),
            ("solver", "refiner"): rng.uniform(0.01, 1, (5, 5)),
            ("reflector", "refiner"): rng.uniform(0.01, 1, (5, 5)),
        },
    )
    got, exact = map_infer(problem), brute_force_map(problem)
    assert got.assignment == exact.assignment and got.treewidth == 2
    single = map_infer(MAPProblem(["a"], {"a": 4}, {"a": np.array([0.1, 0.9, 0.3, 0.2])}, {}))
    assert single.assignment == {"a": 1} and single.treewidth == 0


def test_tied_shared_prompt_retains_replica_and_peer_factors_exactly():
    # Node-only scoring would pick 0; directed peer compatibility must pick 1.
    nodes, edges, n = [0.9, 0.7], [0.1, 0.95], 4
    collapsed = tied_log_quality(nodes, edges, n, n * (n - 1))
    expanded = [
        sum(np.log(nodes[k]) for _ in range(n)) + sum(np.log(edges[k]) for i in range(n) for j in range(n) if i != j)
        for k in range(2)
    ]
    assert np.allclose(collapsed, expanded)
    assert int(np.argmax(collapsed)) == 1 != int(np.argmax(nodes))
    assert np.allclose(tied_log_quality(nodes, None, 3, 0), 3 * np.log(nodes))


def test_demo_pool_renders_at_most_three_balanced_demonstrations():
    pool = DemoPool()
    for index in range(4):
        pool.add(Demo(prompt=f"p{index}", output=f"good {index}", label="+"))
        pool.add(Demo(prompt=f"n{index}", output=f"bad {index}", label="-"))
    lines = pool.render().splitlines()
    assert pool.max_demos == 3 and len(lines) == 3
    assert [line.split("]")[0] for line in lines] == ["[GOOD", "[BAD", "[GOOD"]
    assert "good 3" in lines[0] and "bad 3" in lines[1]
    critic = DemoPool()
    critic_update(critic, "p1", "chosen out", [0.2, 0.9, 0.3], ["p0", "p1", "p2"], ["o0", "chosen out", "o2"], True)
    assert [d.output for d in critic.positives] == ["chosen out"] and [d.output for d in critic.negatives] == ["o0"]


def test_score_and_list_parsing():
    assert parse_score("0.73") == 0.73 and parse_score("8/10") == 0.8 and parse_score("") == 0.5
    assert parse_score_list("Here are the 3 scores:\n0.62\n0.55\n0.48", 3) == [0.62, 0.55, 0.48]
    assert parse_score_list("Candidate 2: 0.85\nCandidate 1: 0.60\nCandidate 3: 0.40", 3) == [0.60, 0.85, 0.40]
    assert parse_score_list("-0.5\n0.3", 2) == [0.0, 0.3] and parse_score_list("garbage", 2) == [0.5, 0.5]


def test_variant_and_label_cleaning():
    for raw in ["VARIANT 1: text", "**VARIANT 3:** text", "1. VARIANT: text", "Variant 2 – text", "ROLE PROMPT: text"]:
        assert _clean(raw) == "text", raw
    blocks = (
        "Here are 3 variants:\n\nVARIANT 1:\nYou are A.\nDo X.\n\nVARIANT 2:\nYou are B.\n\n**VARIANT 3:** You are C."
    )
    assert _split_variants(blocks) == ["You are A.\nDo X.", "You are B.", "You are C."]


class _Rewriter:
    def __init__(self, reply: str) -> None:
        self.reply, self.prompts = reply, []

    async def chat_text(self, prompt, system=None, cfg=None):
        self.prompts.append(prompt)
        return self.reply


def test_pool_init_keeps_seed_and_pads_and_mutation_cycles_three_operators():
    pool = asyncio.run(init_candidates(_Rewriter("VARIANT 1: alpha\nVARIANT 2: beta"), "solver", "seed", 5))
    assert pool == ["seed", "alpha", "beta", "seed", "seed"]
    rewriter = _Rewriter("rewritten")
    mutated = asyncio.run(mutate_pool(rewriter, "solver", "best", "f_g", "blame", 5, nonce="2"))
    assert mutated == ["best"] + ["rewritten"] * 4
    operators = [next(op for op in ("ADDING", "REPLACEMENT", "REORGANIZATION") if op in p) for p in rewriter.prompts]
    assert operators == ["ADDING", "REPLACEMENT", "REORGANIZATION", "ADDING"]
    assert [p.count("Revision attempt 2.") for p in rewriter.prompts] == [1, 1, 1, 1]


def test_listwise_node_judge_aligns_candidate_prompts_and_outputs():
    judge = _Rewriter("0.80\n0.35")
    scores = asyncio.run(
        node_score_listwise(judge, "solver", "task", ["PROMPT_A", "PROMPT_B"], ["OUTPUT_A", "OUTPUT_B"], DemoPool())
    )
    prompt = judge.prompts[0]
    assert scores == [0.8, 0.35]
    assert prompt.index("PROMPT_A") < prompt.index("OUTPUT_A") < prompt.index("PROMPT_B") < prompt.index("OUTPUT_B")
    with pytest.raises(ValueError, match="identical original-position order"):
        asyncio.run(node_score_listwise(judge, "solver", "task", ["p0", "p1"], ["y0"], DemoPool()))
