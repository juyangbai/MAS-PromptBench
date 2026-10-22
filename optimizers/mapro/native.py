"""MAPRO native loop over a prompt-mutable protocol session.

Per-role prompt pools (K=5 including the seed), LLM-judged node/edge
potentials on the cell's prompt graph, exact max-product MAP selection
(``src/infer/bp.py``), topology-aware blame feedback and trust-region pool
mutation between rounds (MAPRO, arXiv:2510.07475, Algorithm 1).

Prompt graphs: centralized is a directed manager-to-worker star; sequential
chains the native stage order; independent and decentralized replicas share
one prompt variable, solved as a constrained MAP that keeps every replica node
factor and (decentralized) every directed peer factor on the diagonal.

The caller supplies the three model surfaces: ``shim`` (pool initialization,
blame and mutation; reflection model), ``judge`` (node/edge potentials) and
``probe`` (single-agent candidate outputs), the latter two on the task model.
Full-MAS rollouts happen only in :func:`run_assignment` through ``runner``;
each assignment evaluation is charged batch-atomically on ``budget`` first.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import numpy as np

from optimizers.protocol.errors import NativeIntegrationError
from optimizers.protocol.schema import StopReason
from optimizers.protocol.session import NativeRolloutBudget

from .regime import MAPROSettings
from .src.infer.bp import MAPProblem, MAPResult, map_infer
from .src.llm import ChatModel
from .src.mas.graph import Agent, Edge, MASGraph
from .src.refine.feedback import _BLAME_CFG, _BLAME_RE, _BLAME_SYS, _BLAME_TMPL, TaskFeedback, _clip, aggregate_feedback
from .src.refine.mutate import init_candidates, mutate_pool
from .src.reward.demos import DemoPool, critic_update
from .src.reward.edge_reward import edge_score, edge_score_listwise
from .src.reward.node_reward import node_score, node_score_listwise

# MAPRO_LISTWISE=1 aligns the reward-call shape with paper Fig. 4 and
# MAPRO_PAPER_ANCHOR=1 anchors on the latest MAP selection (Alg. 1). The
# protocol integration refuses both: its regime is pointwise + best-so-far.
_ENV = MAPROSettings.from_env()
MAPRO_LISTWISE_FLAG = _ENV.listwise
MAPRO_PAPER_ANCHOR_FLAG = _ENV.paper_anchor


# Prompt graphs
def build_star_graph(roles: list[str], seed_prompts: dict[str, str]) -> MASGraph:
    """Star over the centralized roles (``roles[0]`` is the manager hub).

    MASGraph must be a DAG, so the bidirectional group chat is modeled with
    directed manager->worker edges; the blame pass handles both directions.
    """
    hub = roles[0]
    agents = {r: Agent(id=r, role=r, base_prompt=seed_prompts[r]) for r in roles}
    edges = [Edge(hub, w) for w in roles[1:]]
    return MASGraph(agents=agents, edges=edges, output_agent=hub, name="centralized-star")


def build_prompt_graph(roles: list[str], seed_prompts: dict[str, str], topology: str, team_size: int = 1) -> MASGraph:
    """The prompt-variable graph of a topology (star, chain, or one tied replica variable)."""
    if topology == "centralized":
        return build_star_graph(roles, seed_prompts)
    agents = {r: Agent(id=r, role=r, base_prompt=seed_prompts[r]) for r in roles}
    if topology == "sequential":
        edges = [Edge(left, right) for left, right in zip(roles, roles[1:])]
    elif topology in {"single", "independent", "decentralized"} and len(roles) == 1:
        # Replicas share one prompt parameter. No fictitious manager/worker edge.
        edges = []
    else:
        raise ValueError("unsupported native prompt-variable topology")
    graph = MASGraph(agents=agents, edges=edges, output_agent=roles[-1], name=f"native-{topology}-prompt-graph")
    graph.tied_replicas = int(team_size) if topology in {"independent", "decentralized"} else 1
    graph.tied_peer_edges = int(team_size) * (int(team_size) - 1) if topology == "decentralized" else 0
    graph.factor_edges = [Edge(roles[0], roles[0])] if graph.tied_peer_edges else edges
    return graph


def tied_log_quality(
    node_scores: np.ndarray, diagonal_edge_scores: np.ndarray | None, replicas: int, peer_edges: int
) -> np.ndarray:
    """Exact constrained MAP: all replicas use the same prompt variable.

    Independent replicas retain every node factor; communicating peers
    additionally retain each directed peer factor evaluated on the
    shared-candidate diagonal.
    """
    values = int(replicas) * np.log(np.clip(np.asarray(node_scores), 1e-6, 1.0))
    if peer_edges:
        values = values + int(peer_edges) * np.log(np.clip(np.asarray(diagonal_edge_scores), 1e-6, 1.0))
    return values


def topology_contexts(record: dict, graph: MASGraph, topology: str) -> dict[str, str]:
    """Per-role context x_i from one train transcript: what each role saw before acting."""
    if topology == "centralized":
        return role_contexts(record, graph.agent_ids, graph.output_agent)
    task = _clip(_task_text(record), 1500)
    messages = record.get("messages") or []
    result = {}
    for role in graph.agent_ids:
        if topology == "sequential":
            first = next((i for i, item in enumerate(messages) if item.get("source") == role), len(messages))
            visible = messages[:first]
        elif topology == "decentralized":
            visible = messages
        else:
            visible = []
        result[role] = f"TASK:\n{task}\n\nNATIVE SOURCE-TAGGED CONTEXT:\n" + json.dumps(
            visible, ensure_ascii=False, default=str
        )
    return result


async def topology_blame(shim: ChatModel, graph: MASGraph, topology: str, record: dict) -> TaskFeedback:
    """Blame feedback of one train record: each child critiques its parents on an incorrect answer."""
    if topology == "centralized":
        return await star_blame(shim, graph, graph.output_agent, record)
    correct = record["score"] > 0
    feedback = (
        f"The system answered correctly (final answer: {record['got']})."
        if correct
        else f"The system answered INCORRECTLY (expected: {record['gold']}; produced: {record['got']})."
    )
    blames = {role: [] for role in graph.agent_ids}
    if not correct:
        messages = record.get("messages") or []

        def outputs(source: str) -> str:
            return "\n".join(str(m.get("content", "")) for m in messages if m.get("source") == source)

        for role in reversed(graph.agent_ids):
            parents = graph.parents(role)
            if not parents:
                continue
            prompt = _BLAME_TMPL.format(
                outcome="INCORRECT",
                task=_clip(_task_text(record), 800),
                role=role,
                aid=role,
                parents="\n".join(f"--- parent {p} ---\n{_clip(outputs(p), 600)}" for p in parents),
                output=_clip(outputs(role), 600),
            )
            response = await shim.chat_text(prompt, system=_BLAME_SYS, cfg=_BLAME_CFG)
            for parent, critique in _BLAME_RE.findall(response or ""):
                if parent in parents and critique.strip().upper() != "OK":
                    blames[parent].append(f"(from {role}) {critique.strip()}")
    return TaskFeedback(f_g=feedback, blames={role: " ".join(values) for role, values in blames.items()})


# Train-side assignment evaluation with source-tagged transcripts (the caller
# charges the budget BEFORE calling; batch-atomic).
def _row_gold(row) -> str:
    for key in ("answer", "solution", "gold", "expected_answer"):
        value = getattr(row, key, None)
        if value:
            return str(value)
    return ""


def _output_messages(out) -> list:
    """Return the runner transcript across the top-level and nested output shapes."""
    if not isinstance(out, dict):
        return []
    top_level = out.get("messages")
    if isinstance(top_level, list) and top_level:
        return top_level
    runner_output = out.get("runner_output")
    if isinstance(runner_output, dict):
        nested = runner_output.get("messages")
        if isinstance(nested, list):
            return nested
    return top_level if isinstance(top_level, list) else []


def _prediction_from_adapter_output(runner, role, out):
    from optimizers.bridge.programs import prediction_from_adapter_output

    return prediction_from_adapter_output(runner, role, out)


def run_assignment(
    runner: Any,
    plain_metric: Callable[[Any, Any], float],
    roles: list[str],
    assignment: dict[str, str],
    rows: Sequence[Any],
    num_threads: int,
) -> tuple[float, list[dict]]:
    """Full-MAS train evaluation of one prompt assignment: mean score and per-row transcripts."""
    for role in roles:
        runner.set_prompt(role, assignment[role])

    def one(row):
        try:
            out = runner.run_example(row)
            if isinstance(out, dict) and out.get("runner_status") in {"success", "semantic_failure"}:
                score = float(out.get("runner_score") or 0.0)
                got = str(out.get("answer_text") or "")
            else:
                pred = _prediction_from_adapter_output(runner, roles[0], out)
                score = float(plain_metric(row, pred))
                got = str(getattr(pred, "answer", "") or "")
        except Exception as exc:  # never raise per-example
            if type(exc).__name__ in {"NativeInfrastructureExhausted", "NativeIntegrationError"}:
                raise
            out, score, got = {"error": repr(exc), "messages": []}, 0.0, ""
        return {
            "id": str(getattr(row, "id", "")),
            "score": score,
            "gold": _row_gold(row)[:400],
            "got": got[:400],
            "error": out.get("error") if isinstance(out, dict) else None,
            "messages": _output_messages(out),
        }

    with ThreadPoolExecutor(max_workers=min(num_threads, max(1, len(rows)))) as ex:
        records = list(ex.map(one, rows))
    score = sum(r["score"] for r in records) / max(1, len(records))
    return score, records


# Per-role contexts x_i from a source-tagged transcript (Stage 2 step 1).
def _task_text(record: dict) -> str:
    for m in record.get("messages") or []:
        if m.get("source") in ("user", "human") and m.get("content"):
            return m["content"]
    return ""


def _delegations(messages: list[dict], hub: str, role: str) -> list[str]:
    """Manager delegation instructions to `role`: delegate_to_<role> tool calls,
    falling back to the last hub message preceding the role's first message."""
    instr: list[str] = []
    for m in messages:
        if m.get("source") != hub:
            continue
        for tc in m.get("tool_calls") or []:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name != f"delegate_to_{role}":
                continue
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            if isinstance(args, dict):
                instr.append(str(args.get("instructions") or json.dumps(args, default=str)))
            elif args:
                instr.append(str(args))
    if not instr:
        first = next((i for i, m in enumerate(messages) if m.get("source") == role), None)
        if first is not None:
            prior = [m.get("content", "") for m in messages[:first] if m.get("source") == hub and m.get("content")]
            if prior:
                instr = [prior[-1]]
    return instr


def role_contexts(record: dict, roles: list[str], hub: str) -> dict[str, str]:
    """Centralized contexts: the task for the hub, the hub's delegations for each worker."""
    task = _clip(_task_text(record), 1500)
    msgs = record.get("messages") or []
    ctx = {hub: task}
    for role in roles:
        if role == hub:
            continue
        instr = _delegations(msgs, hub, role)
        block = (
            "\n---\n".join(_clip(i, 700) for i in instr[-2:])
            if instr
            else "(the manager did not delegate to you in this rollout; respond according to your role)"
        )
        ctx[role] = f"TASK:\n{task}\n\nMANAGER INSTRUCTIONS TO {role}:\n{block}"
    return ctx


# Stage 2: judged node/edge potentials + exact MAP via max-product BP.
async def stage2_select(
    shim: ChatModel,
    judge: ChatModel,
    graph: MASGraph,
    pools: dict[str, list[str]],
    contexts_list: list[dict[str, str]],
    Y: dict[str, list[list[str]]],
    demos: dict[str, DemoPool],
    iteration: int | None = None,
) -> tuple[dict[str, str], MAPResult, dict[str, np.ndarray], dict[tuple[str, str], np.ndarray]]:
    """Judge node/edge potentials of every pool candidate and select the exact MAP assignment."""
    roles = graph.agent_ids
    factor_edges = getattr(graph, "factor_edges", graph.edges)
    K = {r: len(pools[r]) for r in roles}

    node_jobs, node_idx = [], []
    edge_jobs, edge_idx = [], []
    if MAPRO_LISTWISE_FLAG:
        # paper Eq. 4/5: one call per (role, task) / (edge, k, task) ranks the pool
        for r in roles:
            for ti, ctx in enumerate(contexts_list):
                node_jobs.append(
                    node_score_listwise(
                        judge,
                        graph.agents[r].role,
                        ctx[r],
                        list(pools[r]),
                        [Y[r][k][ti] for k in range(K[r])],
                        demos[r],
                    )
                )
                node_idx.append((r,))
        for e in factor_edges:
            i, j = e.src, e.dst
            for k in range(K[i]):
                for ti in range(len(contexts_list)):
                    edge_jobs.append(
                        edge_score_listwise(judge, graph.agents[j].role, Y[i][k][ti], list(pools[j]), demos[j])
                    )
                    edge_idx.append((i, j, k))
    else:
        for r in roles:
            for k in range(K[r]):
                for ti, ctx in enumerate(contexts_list):
                    node_jobs.append(node_score(judge, graph.agents[r].role, ctx[r], Y[r][k][ti], demos[r]))
                    node_idx.append((r, k))
        for e in factor_edges:
            i, j = e.src, e.dst
            for k in range(K[i]):
                for l in range(K[j]):
                    if i == j and k != l:
                        continue  # infeasible off-diagonal shared-prompt assignment
                    for ti in range(len(contexts_list)):
                        edge_jobs.append(edge_score(judge, graph.agents[j].role, Y[i][k][ti], pools[j][l], demos[j]))
                        edge_idx.append((i, j, k, l))
    # Node and edge potentials are independent fan-outs (edge jobs never read
    # node values), so one combined gather; submission stays node-first.
    all_vals = await asyncio.gather(*node_jobs, *edge_jobs)
    node_vals = all_vals[: len(node_jobs)]
    edge_vals = all_vals[len(node_jobs) :]
    node_scores = {r: np.zeros(K[r]) for r in roles}
    node_cnt = {r: np.zeros(K[r]) for r in roles}
    edge_scores = {(e.src, e.dst): np.zeros((K[e.src], K[e.dst])) for e in factor_edges}
    edge_cnt = {key: np.zeros_like(mat) for key, mat in edge_scores.items()}
    if MAPRO_LISTWISE_FLAG:
        for (r,), vals in zip(node_idx, node_vals):
            node_scores[r] += np.asarray(vals[: K[r]])
            node_cnt[r] += 1
        for (i, j, k), row in zip(edge_idx, edge_vals):
            edge_scores[(i, j)][k, :] += np.asarray(row[: K[j]])
            edge_cnt[(i, j)][k, :] += 1
    else:
        for (r, k), v in zip(node_idx, node_vals):
            node_scores[r][k] += v
            node_cnt[r][k] += 1
        for (i, j, k, l), v in zip(edge_idx, edge_vals):
            edge_scores[(i, j)][k, l] += v
            edge_cnt[(i, j)][k, l] += 1
    node_scores = {r: node_scores[r] / np.maximum(1, node_cnt[r]) for r in roles}
    for key in edge_scores:
        edge_scores[key] /= np.maximum(1, edge_cnt[key])

    nodes = {r: np.clip(node_scores[r], 1e-6, 1.0) for r in roles}
    edges = {key: np.clip(mat, 1e-6, 1.0) for key, mat in edge_scores.items()}
    tied_logs = None
    if len(roles) == 1 and getattr(graph, "tied_replicas", 1) > 1:
        role = roles[0]
        diagonal = np.diag(edge_scores[(role, role)]) if (role, role) in edge_scores else None
        tied_logs = tied_log_quality(node_scores[role], diagonal, graph.tied_replicas, graph.tied_peer_edges)
        # Common scale preserves MAP while avoiding underflow before log BP.
        nodes, edges = {role: np.exp(tied_logs - np.max(tied_logs))}, {}
    problem = MAPProblem(
        var_names=list(roles),
        cardinalities=K,
        node_potentials=nodes,
        edge_potentials=edges,
    )
    res = map_infer(problem)
    if tied_logs is not None:
        res.log_score = float(tied_logs[res.assignment[roles[0]]])
    assignment = {r: pools[r][res.assignment[r]] for r in roles}
    return assignment, res, node_scores, edge_scores


# Stage 3: blame over the bidirectional star (adapted from refine/feedback.py):
# each activated worker critiques the manager's delegation (blames the hub);
# the manager critiques the workers' reports (blames the workers). f_g carries
# the metric verdict plus expected-vs-got.
async def star_blame(shim: ChatModel, graph: MASGraph, hub: str, record: dict) -> TaskFeedback:
    """Blame over the bidirectional star: workers critique the hub, the hub critiques the workers."""
    correct = record["score"] > 0
    outcome = "CORRECT" if correct else "INCORRECT"
    f_g = (
        f"The system answered correctly (final answer: {record['got']})."
        if correct
        else f"The system answered INCORRECTLY (expected: {record['gold']}; produced: {record['got']})."
    )
    blames: dict[str, list[str]] = {aid: [] for aid in graph.agent_ids}
    if not correct:
        msgs = record.get("messages") or []
        task = _clip(_task_text(record), 800)
        by_role = {
            aid: [m.get("content", "") for m in msgs if m.get("source") == aid and m.get("content")]
            for aid in graph.agent_ids
        }
        jobs, valid_parents, order = [], [], []
        # Workers critique the hub's delegation (reverse topo: leaves first).
        for w in graph.children(hub):
            if not by_role[w]:
                continue  # never activated in this rollout
            deleg = _delegations(msgs, hub, w)
            deleg_text = deleg[-1] if deleg else "(no explicit delegation found)"
            parents_block = f"--- parent {hub} ({graph.agents[hub].role}) ---\n{_clip(deleg_text, 600)}"
            prompt = _BLAME_TMPL.format(
                outcome=outcome,
                task=task,
                role=graph.agents[w].role,
                aid=w,
                parents=parents_block,
                output=_clip("\n".join(by_role[w]), 600),
            )
            jobs.append(shim.chat_text(prompt, system=_BLAME_SYS, cfg=_BLAME_CFG))
            valid_parents.append({hub})
            order.append(w)
        # The hub critiques the workers' reports.
        parents_block = "\n".join(
            f"--- parent {w} ({graph.agents[w].role}) ---\n"
            f"{_clip(by_role[w][-1] if by_role[w] else '(worker was never activated)', 600)}"
            for w in graph.children(hub)
        )
        prompt = _BLAME_TMPL.format(
            outcome=outcome,
            task=task,
            role=graph.agents[hub].role,
            aid=hub,
            parents=parents_block,
            output=_clip(by_role[hub][-1] if by_role[hub] else "", 600),
        )
        jobs.append(shim.chat_text(prompt, system=_BLAME_SYS, cfg=_BLAME_CFG))
        valid_parents.append(set(graph.children(hub)))
        order.append(hub)

        resps = await asyncio.gather(*jobs)
        for critic_id, valid, resp in zip(order, valid_parents, resps):
            for pid, crit in _BLAME_RE.findall(resp or ""):
                crit = crit.strip()
                if pid in valid and crit.upper() != "OK":
                    blames[pid].append(f"(from {graph.agents[critic_id].role}) {crit}")
    return TaskFeedback(f_g=f_g, blames={aid: " ".join(v) for aid, v in blames.items()})


# The MAPRO loop (Algorithm 1) over the protocol session.
async def optimize_mapro(
    args: SimpleNamespace,
    runner: Any,
    roles: list[str],
    seed_prompts: dict[str, str],
    train: Sequence[Any],
    plain_metric: Callable[[Any, Any], float],
    budget: NativeRolloutBudget,
    log: Callable[[str], None] = print,
    *,
    shim: ChatModel,
    judge: ChatModel,
    probe: Any,
    on_iteration: Callable[[dict], None] | None = None,
    topology: str = "centralized",
    team_size: int = 1,
) -> dict:
    """Run MAPRO (Algorithm 1) until patience, the iteration cap or the budget stops it; returns the best state."""
    if shim is None or judge is None or probe is None:
        raise NativeIntegrationError("MAPRO requires the reflection, judge and probe model surfaces")
    graph = build_prompt_graph(roles, seed_prompts, topology, team_size)
    demos = {r: DemoPool() for r in roles}

    eval_rows = train if args.eval_batch <= 0 else train[: args.eval_batch]
    score_n = max(1, min(args.score_batch, len(eval_rows)))

    # Stage 1: init pools (seed + K-1 reflection variants) and the charged seed
    # train eval. The seed eval reads only seed_prompts, never `pools`, so it
    # runs concurrently with pool initialization; its batch-atomic charge is
    # reserved before launch.
    if not budget.try_charge(len(eval_rows)):
        raise NativeIntegrationError(
            f"budget {budget.total} cannot cover the initial seed train eval ({len(eval_rows)} rollouts)"
        )
    if hasattr(runner, "set_context"):
        runner.set_context(iteration=0, event="mapro_seed_train_eval")

    def _seed_eval():
        return run_assignment(runner, plain_metric, roles, dict(seed_prompts), eval_rows, args.num_threads)

    async def _init_pools():
        return await asyncio.gather(*[init_candidates(shim, r, seed_prompts[r], args.K) for r in roles])

    pool_lists, (seed_train, incumbent_records) = await asyncio.gather(_init_pools(), asyncio.to_thread(_seed_eval))
    pools = dict(zip(roles, pool_lists))
    best_assignment, best_train = dict(seed_prompts), seed_train
    best_records = incumbent_records
    incumbent = dict(seed_prompts)
    incumbent_train = seed_train
    trajectory = [[0, seed_train]]
    history: list[dict] = []
    non_seed_evals = 0
    log(f"[mapro] iter 0 (seed): train={seed_train:.3f} rollouts={budget.used}/{budget.total}")

    deltas: list[float] = []
    prev = seed_train
    stop_reason = StopReason.MAX_ITERS

    for t in range(1, args.max_iters + 1):
        # Never start an iteration whose eval batch cannot be fully afforded.
        if budget.remaining() < len(eval_rows):
            stop_reason = StopReason.BUDGET
            budget.stops += 1
            break

        # Stage 2: contexts from the incumbent's latest train transcripts,
        # single-agent probes for every candidate, judged potentials, exact MAP.
        contexts_list = [topology_contexts(rec, graph, topology) for rec in incumbent_records[:score_n]]
        jobs, idx = [], []
        for r in roles:
            for k, cand in enumerate(pools[r]):
                for ti, ctx in enumerate(contexts_list):
                    jobs.append((cand, ctx[r]))
                    idx.append((r, k, ti))
        outs = await asyncio.to_thread(probe.batch, jobs)
        Y = {r: [["" for _ in contexts_list] for _ in pools[r]] for r in roles}
        for (r, k, ti), o in zip(idx, outs):
            Y[r][k][ti] = o
        assignment, bp_res, node_scores, edge_scores = await stage2_select(
            shim, judge, graph, pools, contexts_list, Y, demos, iteration=t
        )

        # Evaluate the MAP assignment on the train batch (charged, batch-atomic).
        is_incumbent = all(assignment[r].strip() == incumbent[r].strip() for r in roles)
        evaluated = False
        if is_incumbent:
            # The anchor can be latest-selection (paper mode) or best-so-far.
            # Its prompt, score, and records are maintained as one coherent state.
            val, records = incumbent_train, incumbent_records
        else:
            if not budget.try_charge(len(eval_rows)):
                stop_reason = StopReason.BUDGET
                break
            if hasattr(runner, "set_context"):
                runner.set_context(iteration=t, event="mapro_assignment_eval")
            val, records = await asyncio.to_thread(
                run_assignment, runner, plain_metric, roles, assignment, eval_rows, args.num_threads
            )
            evaluated = True
            if any(assignment[r].strip() != seed_prompts[r].strip() for r in roles):
                non_seed_evals += 1
            if val > best_train:
                best_assignment, best_train = dict(assignment), val
                best_records = records
        trajectory.append([t, val])
        log(
            f"[mapro] iter {t}: train={val:.3f} (best={best_train:.3f}) "
            f"idx={bp_res.assignment} logscore={bp_res.log_score:.2f} "
            f"evaluated={evaluated} rollouts={budget.used}/{budget.total}"
        )

        # Stage 3: blame feedback on the mutation base's transcripts (failures
        # first): best-so-far (trust region, default) or the latest MAP selection
        # (paper Alg. 1, MAPRO_PAPER_ANCHOR=1).
        if MAPRO_PAPER_ANCHOR_FLAG:
            mutate_base = dict(assignment)
            base_records = records
        else:
            mutate_base = best_assignment
            base_records = best_records
        order = sorted(range(len(base_records)), key=lambda i: base_records[i]["score"])
        sampled = [base_records[i] for i in order[: args.feedback_samples]]
        fbs = list(await asyncio.gather(*[topology_blame(shim, graph, topology, rec) for rec in sampled]))
        f_g, blames = aggregate_feedback(graph, fbs)
        examples = [fb.f_g for fb in fbs if "INCORRECTLY" in fb.f_g][:2]
        if examples:
            f_g = f_g + " Examples: " + " | ".join(examples)
        n_correct = sum(1 for rec in sampled if rec["score"] > 0)
        # Critic labels describe the selected MAP assignment, never the separate
        # feedback/mutation anchor. Use its full eval batch: failure-first feedback
        # samples are deliberately biased and would starve positive demonstrations.
        n_correct_selected = sum(1 for rec in records if rec["score"] >= 1.0)
        selected_success = n_correct_selected >= max(1, len(records)) / 2

        # Critic demo update (score-driven, no extra LLM call).
        if not args.no_demos:
            for r in roles:
                ci = bp_res.assignment[r]
                cand_out0 = [Y[r][k][0] for k in range(len(pools[r]))]
                critic_update(
                    demos[r],
                    chosen_prompt=assignment[r],
                    chosen_output=cand_out0[ci],
                    node_scores=node_scores[r],
                    candidate_prompts=pools[r],
                    candidate_outputs=cand_out0,
                    success=selected_success,
                )

        # Mutate around the anchor: best-so-far (trust region) or latest P* (paper).
        new_pool_lists = await asyncio.gather(
            *[mutate_pool(shim, r, mutate_base[r], f_g, blames.get(r, ""), args.K, nonce=str(t)) for r in roles]
        )
        pools = dict(zip(roles, new_pool_lists))
        if MAPRO_PAPER_ANCHOR_FLAG:
            # Paper anchoring: next-round context state follows the latest MAP
            # selection even when it did not improve best-so-far.
            incumbent = dict(assignment)
            incumbent_train = val
            incumbent_records = records
        else:
            incumbent = dict(best_assignment)
            incumbent_train = best_train
            incumbent_records = best_records

        history.append(
            {
                "iter": t,
                "train_score": val,
                "evaluated": evaluated,
                "selected_is_incumbent": is_incumbent,
                "assignment_idx": {r: int(bp_res.assignment[r]) for r in roles},
                "bp_log_score": float(bp_res.log_score),
                "bp_treewidth": int(bp_res.treewidth),
                "node_scores": {r: [round(float(x), 4) for x in node_scores[r]] for r in roles},
                "edge_scores": {f"{i}->{j}": np.round(m, 4).tolist() for (i, j), m in edge_scores.items()},
                "selected_roles_changed_vs_seed": [
                    r for r in roles if assignment[r].strip() != seed_prompts[r].strip()
                ],
                "n_correct_feedback": n_correct,
                "n_correct_selected": n_correct_selected,
                "critic_success": selected_success,
                "blame_chars": {r: len(blames.get(r, "")) for r in roles},
                "rollouts_used": budget.used,
            }
        )
        if on_iteration is not None:
            on_iteration(
                {
                    "iteration": t,
                    "selected_assignment": dict(assignment),
                    "incumbent": dict(incumbent),
                    "best_assignment": dict(best_assignment),
                    "selected_score": float(val),
                    "best_train": float(best_train),
                    "evaluated": bool(evaluated),
                    "pools": {role: list(values) for role, values in pools.items()},
                    "trajectory": list(trajectory),
                    "history": list(history),
                    "non_seed_evals": non_seed_evals,
                    "rollouts_used": budget.used,
                }
            )

        # Stage 4: patience termination on the train trajectory.
        delta = val - prev
        deltas.append(delta)
        prev = val
        if t >= args.patience and max(deltas[-args.patience :]) <= args.eps:
            stop_reason = StopReason.PATIENCE
            break

    return {
        "best_assignment": best_assignment,
        "best_train": best_train,
        "seed_train": seed_train,
        "trajectory": trajectory,
        "history": history,
        "iterations_run": len(history),
        "stop_reason": stop_reason,
        "non_seed_evals": non_seed_evals,
        "eval_batch_size": len(eval_rows),
        "score_batch_size": score_n,
        "judge_usage_9b": dict(judge.usage),
        "probe_usage_9b": dict(probe.usage),
        "judge_errors_122b": shim.n_errors,
        "graph_edges": [[e.src, e.dst] for e in graph.edges],
        "tied_replicas": getattr(graph, "tied_replicas", 1),
        "tied_peer_factors": getattr(graph, "tied_peer_edges", 0),
        "final_pool_sizes": {r: len(pools[r]) for r in roles},
    }


__all__ = [
    "MAPRO_LISTWISE_FLAG",
    "MAPRO_PAPER_ANCHOR_FLAG",
    "build_prompt_graph",
    "build_star_graph",
    "optimize_mapro",
    "role_contexts",
    "run_assignment",
    "stage2_select",
    "star_blame",
    "tied_log_quality",
    "topology_blame",
    "topology_contexts",
]
