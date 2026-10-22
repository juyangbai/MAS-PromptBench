"""MASPO multi-granularity evolutionary beam search over one role-prompt bundle.

Re-implementation of the upstream full ``optimize_all_fixed_rounds`` regime
("MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems",
arXiv:2605.06623). The upstream ``MAPromptOptimizer`` is bound to its own
``MAS``/``InferenceCache`` classes, so the loop is re-implemented over the
protocol runner; every parent-baseline and candidate minibatch evaluation is a
charged full-MAS rollout.

Kept from the paper/release: beam K=2, K_sub=2 variations per parent (one per
minibatch half), minibatch |B|=10 resampled per node, T=3 depth steps per role
turn, coordinate ascent over roles (terminal last), Eq. 5 joint reward
0.4 local / 0.4 lookahead / 0.2 global - 0.5 (0.7 local / 0.3 global fallback
when no successor pair is evaluable; terminal: global - 0.5), Eq. 6
misalignment mining/injection (buffer 3, injection 5) and Eq. 8 beam refresh
with the release's 0.7/0.3 weights. Proposals run at temperature 0.7 and
pairwise judgments at 0.0 (upstream values).

Adaptations: the global granularity is the exact dataset metric (win 1 /
tie 0.5 / loss 0 against the parent rollout on the same item and seed);
MATH uses the upstream templates, every other task a role-aware adaptation
(no upstream template matches these role sets); the requirement string is the
task's protected output contract; judge replies other than exactly ``A``/``B``
(or ``<choose>A|B</choose>``) count as a loss.
"""

from __future__ import annotations

import random
import re
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Any

import yaml

from optimizers.protocol import REPO_ROOT
from optimizers.protocol.errors import NativeIntegrationError

from .upstream import INTERMEDIATE_COMPARE_TEMPLATE, PROMPT_OPTIMIZE_TEMPLATE, TaskType

# Config
MIS_CAP = 3  # upstream: node-best candidate's priority-sorted top-3 misalignment cases
MIS_INJECT = 5  # upstream: random.sample(union of predecessor buffers, 5)
LOOKAHEAD_WEIGHTS = (0.4, 0.4, 0.2)  # local, successor, global (upstream "4:4:2")
FALLBACK_WEIGHTS = (0.7, 0.3)  # local, global (no successor pair; beam refresh)
SCORE_OFFSET = 0.5
PROPOSAL_TEMPERATURE = 0.7
JUDGE_TEMPERATURE = 0.0
PROPOSAL_MAX_TOKENS = 16384  # native ceiling; the reflection client applies the common one
JUDGE_MAX_TOKENS = 1024
ROLES_CATALOG = REPO_ROOT / "configs" / "prompts" / "roles.yaml"
_GENERIC_ROLE_DESC = "Specialist agent in a manager/worker multi-agent system."
_QUESTION_KEYS = ("problem", "question", "issue_description", "prompt", "input")


# Role descriptions and task contracts
@cache
def _role_descriptions(dataset: str) -> dict[str, str]:
    """The task's centralized role catalog (used for every topology, as retained;
    roles absent from it get the generic description)."""
    try:
        catalog = yaml.safe_load(ROLES_CATALOG.read_text(encoding="utf-8"))
        return dict(catalog["topologies"]["centralized"]["benchmarks"][dataset])
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return {}


def role_description(dataset: str, role: str) -> str:
    """The proposer's description of a role (managers of any round share one)."""
    descriptions = _role_descriptions(dataset)
    if role in descriptions:
        return descriptions[role]
    if role.startswith("manager_r") and role.removeprefix("manager_r").isdigit():
        return descriptions.get("manager", _GENERIC_ROLE_DESC)
    return _GENERIC_ROLE_DESC


def dataset_contract(task: str) -> str:
    """The task's protected final-output contract (the MASPO requirement string)."""
    from optimizers.bridge.output_contracts import DATASET_CONTRACTS

    return DATASET_CONTRACTS[task]


def regime_mode(rounds_per_turn: int) -> str:
    """Result label: the paper regime uses T=3; other round counts are budget adaptations."""
    if rounds_per_turn < 1:
        raise ValueError("rounds per turn must be at least 1")
    return "maspo-full" if rounds_per_turn == 3 else "maspo-budget-adapted"


_ROLE_AWARE_PROMPT_OPTIMIZE_TEMPLATE = """
You are optimizing one role's prompt in a centralized manager/worker multi-agent system.
The role and the system's delegation/tool interfaces are fixed and MUST be preserved.

Dataset: {dataset}
Agent Role: {agent_type}
Role Responsibility: {role_description}

Sample Execution Traces (Task + Role Context + Role Output):
<traces>
{samples}
</traces>

Role-aware requirements:
<requirements>
{requirements}
</requirements>

Reference prompt:
<reference_prompt>
{prompt}
</reference_prompt>

Analyze the role output only against this role's responsibility and its usefulness to
downstream agents. Preserve exact role names, delegation tool names, schemas, and output
interfaces. Do not turn a nonterminal specialist into the final-answer agent, and do not
assume every role writes Python or emits the system's final artifact.

Return exactly:
<analyse>Specific defects in the traces and how the prompt should address them.</analyse>
<modification>One-sentence summary of the key improvement.</modification>
<prompt>The complete optimized prompt, with the original role and interfaces preserved.</prompt>
"""

_ROLE_AWARE_INTERMEDIATE_COMPARE_TEMPLATE = """
You are comparing two outputs from one role in a centralized manager/worker system.

Dataset: {dataset}
Agent Role: {agent_type}
Role Responsibility: {role_description}

Task and available schema/constraints:
{question}

System final-output contract (context for downstream usefulness; a nonterminal role
must not be penalized for reporting in its own role-specific format):
{requirements}

Dataset-specific checks:
{domain_criteria}

Output A:
{output_a}

Output B:
{output_b}

Choose the output that is more correct for this role, more faithful to the supplied
schema/constraints, and more useful to downstream agents. Prefer grounded, complete,
non-hallucinated detail. Respond ONLY with "A" or "B".
"""


# Rows and rollouts
def _row_id(row: Any) -> str:
    value = row.get("id") if isinstance(row, Mapping) else getattr(row, "id", None)
    return str(id(row) if value is None else value)


def _row_question(row: Any) -> str:
    for key in _QUESTION_KEYS:
        value = row.get(key) if isinstance(row, Mapping) else getattr(row, key, None)
        if value:
            return str(value)
    return ""


def _rollout_one(runner: Any, row: Any) -> dict[str, Any]:
    """One charged full-MAS rollout under the runner's current bundle.

    Only the runner's canonical score is used; infrastructure and contract
    errors propagate (they must not be scored as task failures).
    """
    out = runner.run_example(row)
    if not isinstance(out, Mapping) or out.get("runner_status") not in {"success", "semantic_failure"}:
        raise NativeIntegrationError("MASPO rollout returned no scored runner observation")
    score = float(out.get("runner_score") or 0.0)
    messages = out.get("messages")
    if not messages:
        nested = out.get("runner_output")
        if isinstance(nested, Mapping):
            messages = nested.get("messages")
    messages = messages if isinstance(messages, list) else []
    question = _row_question(row)
    task_context = next(
        (
            str(m.get("content") or "")
            for m in messages
            if isinstance(m, Mapping) and m.get("source") == "user" and m.get("content")
        ),
        question,
    )
    return {
        "id": _row_id(row),
        "question": question,
        "task_context": task_context,
        "messages": messages,
        "score": score,
        "error": out.get("error"),
    }


def run_bundle_minibatch(runner: Any, bundle: Mapping[str, str], rows: list, num_threads: int) -> list[dict]:
    """One rollout per row under ``bundle``; callers reserve ``len(rows)`` first."""
    for role, text in bundle.items():
        runner.set_prompt(role, text)
    if not rows:
        return []
    with ThreadPoolExecutor(max_workers=min(num_threads, max(1, len(rows)))) as pool:
        return list(pool.map(lambda row: _rollout_one(runner, row), rows))


def run_candidate_minibatches(
    runners: list, bundles: list[Mapping[str, str]], item_lists: list[list], num_threads: int
) -> list[list[dict]]:
    """Evaluate several candidate minibatches through one shared pool.

    Upstream evaluates all candidates of a node concurrently; each candidate
    runs on its own session (a session holds one bundle at a time). Callers
    reserve budget first.
    """
    for runner, bundle in zip(runners, bundles):
        for role, text in bundle.items():
            runner.set_prompt(role, text)
    jobs = [(ci, ii, row) for ci, items in enumerate(item_lists) for ii, row in enumerate(items)]
    results: list[list] = [[None] * len(items) for items in item_lists]
    if not jobs:
        return results
    with ThreadPoolExecutor(max_workers=min(num_threads, len(jobs))) as pool:
        outputs = list(pool.map(lambda job: _rollout_one(runners[job[0]], job[2]), jobs))
    for (ci, ii, _), record in zip(jobs, outputs):
        results[ci][ii] = record
    return results


def role_output(messages: list[dict], role: str) -> str:
    """The role's own contribution in a rollout (upstream ``node_outputs_raw[aid]``)."""
    return _raw_role_output(messages, role) or "(role produced no output in this rollout)"


def _raw_role_output(messages: list[dict], role: str) -> str:
    contents = [m.get("content", "") for m in messages if m.get("source") == role]
    return "\n---\n".join(str(c) for c in contents[-3:] if c).strip()


def _neutral_output_pair(output_cand: str, output_base: str) -> bool:
    """True when a forced A/B comparison has no attributable evidence."""
    cand, base = output_cand.strip(), output_base.strip()
    return not cand or not base or cand == base


def role_context(messages: list[dict], role: str, question: str) -> str:
    """The role's input context (upstream ``cache.get_context_for_node``): task + recent peers."""
    parts = [f"Task: {question[:2000]}"]
    others = [
        (m.get("source", "?"), m.get("content", "")) for m in messages if m.get("source") != role and m.get("content")
    ]
    for source, content in others[-8:]:
        parts.append(f"[{source}] {str(content)[:400]}")
    return "\n".join(parts)[:5000]


# Judge
class Judge:
    """Upstream ``_propose_new_prompt`` / ``_compare_intermediate`` with call counters."""

    def __init__(self, reflection: Any, task_type: TaskType, requirement: str, dataset: str) -> None:
        self.reflection = reflection
        self.task_type = task_type
        self.requirement = requirement
        self.dataset = dataset
        self.topology = "centralized"
        self.terminal_role: str | None = None
        self._lock = threading.Lock()
        self.n_proposal_calls = 0
        self.n_judge_calls = 0
        self.n_proposal_extract_failures = 0

    def propose_new_prompt(self, role: str, old_p: str, qa: dict[str, dict[str, str]]) -> str:
        """Rewrite one role's prompt from sampled problems, contexts and outputs."""
        samples_block = "\n\n".join(
            f"Problem {i + 1}:\n{q.strip()[:2000]}\n\nContext:\n"
            f"{data['context'].strip() or '(no context)'}\n\nAgent Output:\n"
            f"{data['output'].strip()[:4500]}"
            for i, (q, data) in enumerate(qa.items())
        )
        is_terminal = role == "manager" or (role.startswith("manager_r") and role.removeprefix("manager_r").isdigit())
        scope = (
            "This is the terminal manager, so preserve and satisfy the final-output contract exactly."
            if is_terminal
            else "This is a nonterminal specialist. Preserve its report-to-manager format; "
            "the final-output contract below is system context, not this role's output format."
        )
        topology = self.topology
        if topology != "centralized":
            is_terminal = role == self.terminal_role
            scope = (
                "This prompt produces the system's final output. Preserve its native output contract."
                if is_terminal
                else "This is an intermediate native pipeline stage. Preserve its stage output and downstream "
                "input contracts."
            )
            if topology in {"independent", "decentralized"}:
                scope += (
                    " All native replicas share this prompt; preserve their existing aggregation and "
                    "communication rules."
                )
        full_requirement = (
            "Ensure the agent's role, responsibilities, and input format remain consistent. "
            + scope
            + "\n"
            + self.requirement
        )
        template = (
            PROMPT_OPTIMIZE_TEMPLATE[TaskType.MATH] if self.dataset == "math" else _ROLE_AWARE_PROMPT_OPTIMIZE_TEMPLATE
        )
        if topology != "centralized":
            template = template.replace("centralized manager/worker", f"{topology}")
        prompt = template.format(
            dataset=self.dataset,
            agent_type=role,
            role_description=role_description(self.dataset, role),
            requirements=full_requirement,
            prompt=old_p,
            samples=samples_block,
        )
        raw = self.reflection.complete(prompt, temperature=PROPOSAL_TEMPERATURE, max_tokens=PROPOSAL_MAX_TOKENS)
        with self._lock:
            self.n_proposal_calls += 1
        try:
            return raw.split("<prompt>")[1].split("</prompt>")[0].strip() or old_p
        except IndexError:
            with self._lock:
                self.n_proposal_extract_failures += 1
            return old_p

    def compare_intermediate(self, role: str, question: str, output_cand: str, output_base: str) -> bool:
        """True when the candidate output (always ``A``, as upstream) wins."""
        if self.dataset == "math":
            template = INTERMEDIATE_COMPARE_TEMPLATE[TaskType.MATH]
            domain_criteria = ""
        elif self.dataset == "bfcl":
            template = _ROLE_AWARE_INTERMEDIATE_COMPARE_TEMPLATE
            domain_criteria = (
                "Check function-name and argument grounding against the supplied schemas, "
                "required fields and types, call ordering, and canonical JSON when this "
                "role is responsible for a call. Never reward invented functions or values."
            )
        else:
            template = _ROLE_AWARE_INTERMEDIATE_COMPARE_TEMPLATE
            domain_criteria = (
                "Check algorithmic correctness, constraint coverage, complexity, edge cases, "
                "and actionable collaboration. Only a code-producing role should be judged "
                "for code completeness; stdin/stdout and callable formats are task-dependent."
            )
        prompt = template.format(
            dataset=self.dataset,
            agent_type=role,
            role_description=role_description(self.dataset, role),
            requirements=self.requirement,
            domain_criteria=domain_criteria,
            question=question.strip()[:8000],
            output_a=output_cand.strip() or "(empty)",
            output_b=output_base.strip() or "(empty)",
        )
        try:
            response = self.reflection.complete(prompt, temperature=JUDGE_TEMPERATURE, max_tokens=JUDGE_MAX_TOKENS)
        except Exception:
            return False  # unreachable judge: the candidate does not win
        with self._lock:
            self.n_judge_calls += 1
        # Upstream treats an empty or malformed verdict as a candidate win; fail closed instead.
        match = re.fullmatch(r"(?:<choose>\s*)?([AB])(?:\s*</choose>)?", response.strip(), re.IGNORECASE)
        return bool(match and match.group(1).upper() == "A")


# Search
def _sample_minibatch(train_rows, minibatch, rng, mis_buffers, predecessors):
    """Per-node minibatch: predecessors' misalignment cases (sampled to 5), then a random fill."""
    injected: list = []
    seen: set = set()
    if mis_buffers is not None and predecessors:
        union = []
        for pred in predecessors:
            for row in mis_buffers.get(pred, []):
                if _row_id(row) not in seen:
                    seen.add(_row_id(row))
                    union.append(row)
        injected = rng.sample(union, MIS_INJECT) if len(union) > MIS_INJECT else union
    inj_ids = {_row_id(r) for r in injected}
    # Upstream excludes the entire predecessor union from the random fill.
    excluded = set(seen) if (mis_buffers is not None and predecessors) else inj_ids
    pool = [r for r in train_rows if _row_id(r) not in excluded]
    need = max(0, minibatch - len(injected))
    fill = rng.sample(pool, min(need, len(pool)))
    return (list(injected) + fill)[:minibatch]


def evaluate_candidate(
    judge: Judge,
    role: str,
    is_terminal: bool,
    base_runs: list[dict],
    cand_runs: list[dict],
    refl_inflight: int,
    successor_role: str | None = None,
    mis_out: list | None = None,
    items: list | None = None,
) -> dict:
    """Upstream ``_evaluate_candidate`` with the lookahead reward and misalignment mining.

    global = exact metric (candidate vs parent rollout: win 1 / tie 0.5 /
    loss 0); local and lookahead = pairwise judge of the role's and the
    successor's own outputs. Returns the joint ``score`` (Eq. 5, minus
    :data:`SCORE_OFFSET`), its component rates and ``n_pairs``. With
    ``mis_out`` and ``items``, the mined misalignment cases are appended to
    ``mis_out`` as ``(priority, item)``.
    """
    pairs = list(zip(base_runs, cand_runs))
    n = len(pairs)
    if n == 0:
        return {"score": 0.0, "rate_local": None, "rate_global": None, "n_pairs": 0}

    rate_global = _global_win_rate(pairs)
    if is_terminal:
        return {"score": rate_global - SCORE_OFFSET, "rate_local": None, "rate_global": rate_global, "n_pairs": n}

    local_credits = _local_credits(judge, role, pairs, refl_inflight)
    rate_local = sum(local_credits) / n
    w_local, w_global = FALLBACK_WEIGHTS
    if successor_role is None:
        score = rate_global * w_global + rate_local * w_local - SCORE_OFFSET
        return {"score": score, "rate_local": rate_local, "rate_global": rate_global, "n_pairs": n}

    next_credit_by_i = _successor_credits(judge, successor_role, pairs, refl_inflight)
    if mis_out is not None and items is not None:
        mis_out.extend(_misalignment_cases(pairs, local_credits, next_credit_by_i, items))
    if next_credit_by_i:
        rate_next = sum(next_credit_by_i.values()) / len(next_credit_by_i)
        a_local, a_next, a_global = LOOKAHEAD_WEIGHTS
        score = a_local * rate_local + a_next * rate_next + a_global * rate_global - SCORE_OFFSET
    else:
        rate_next = None
        score = rate_global * w_global + rate_local * w_local - SCORE_OFFSET
    return {
        "score": score,
        "rate_local": rate_local,
        "rate_next": rate_next,
        "rate_global": rate_global,
        "n_pairs": n,
    }


def _global_win_rate(pairs: list[tuple[dict, dict]]) -> float:
    """Exact-metric win rate of the candidate over the parent rollouts (a tie counts half)."""
    wins_global = 0.0
    for base, cand in pairs:
        if cand["score"] > base["score"] + 1e-9:
            wins_global += 1.0
        elif abs(cand["score"] - base["score"]) <= 1e-9:
            wins_global += 0.5  # exact-metric tie; upstream's forced-choice judge is a coin flip
    return wins_global / len(pairs)


def _judge_outputs(judge: Judge, role: str, jobs: list[tuple[dict, str, str]], refl_inflight: int) -> list[bool]:
    """Pairwise verdicts (candidate wins) of ``jobs`` = ``[(parent run, candidate output, parent output)]``."""

    def one(job):
        base, output_cand, output_base = job
        return judge.compare_intermediate(role, base.get("task_context") or base["question"], output_cand, output_base)

    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=min(refl_inflight, len(jobs))) as pool:
        return list(pool.map(one, jobs))


def _local_credits(judge: Judge, role: str, pairs: list[tuple[dict, dict]], refl_inflight: int) -> list[float]:
    """Per pair: 1 / 0 when the judge prefers the candidate's / parent's own role output, 0.5 when neutral."""
    credits = [0.5] * len(pairs)
    judged, jobs = [], []
    for i, (base, cand) in enumerate(pairs):
        output_cand = _raw_role_output(cand["messages"], role)
        output_base = _raw_role_output(base["messages"], role)
        if not _neutral_output_pair(output_cand, output_base):
            judged.append(i)
            jobs.append((base, output_cand, output_base))
    for i, won in zip(judged, _judge_outputs(judge, role, jobs, refl_inflight)):
        credits[i] = 1.0 if won else 0.0
    return credits


def _successor_output(messages: list[dict], successor_role: str) -> str:
    """The successor's last three non-empty messages."""
    contents = [m.get("content", "") for m in messages if m.get("source") == successor_role]
    return "\n---\n".join(str(c) for c in contents[-3:] if c)


def _successor_credits(
    judge: Judge, successor_role: str, pairs: list[tuple[dict, dict]], refl_inflight: int
) -> dict[int, float]:
    """Eq. 5 lookahead credit per pair index, judged on the successor's outputs.

    Pairs where either successor output is missing get no credit; neutral pairs
    get 0.5 (and come first, in pair order, then the judged pairs).
    """
    next_credit_by_i: dict[int, float] = {}
    judged, jobs = [], []
    for i, (base, cand) in enumerate(pairs):
        sb = _successor_output(base["messages"], successor_role)
        sc = _successor_output(cand["messages"], successor_role)
        if not (sb and sc):
            continue
        if _neutral_output_pair(sc, sb):
            next_credit_by_i[i] = 0.5
        else:
            judged.append(i)
            jobs.append((base, sc, sb))
    for i, won in zip(judged, _judge_outputs(judge, successor_role, jobs, refl_inflight)):
        next_credit_by_i[i] = 1.0 if won else 0.0
    return next_credit_by_i


def _misalignment_cases(
    pairs: list[tuple[dict, dict]], local_credits: list[float], next_credit_by_i: dict[int, float], items: list
) -> list[tuple[int, Any]]:
    """Eq. 6 misalignment cases: local wins whose successor or global outcome failed.

    Upstream priority: 0 = successor and global fail, 1 = successor fail,
    2 = global fail.
    """
    cases = []
    for i, (base, cand) in enumerate(pairs):
        if local_credits[i] != 1.0 or i >= len(items):
            continue
        glob_fail = cand["score"] < base["score"] - 1e-9
        next_fail = next_credit_by_i.get(i) == 0.0
        if next_fail and glob_fail:
            priority = 0
        elif next_fail:
            priority = 1
        elif glob_fail:
            priority = 2
        else:
            continue
        cases.append((priority, items[i]))
    return cases


class RoleState:
    """Upstream ``AgentOptState``: beam, best node and depth of one role."""

    def __init__(self, role: str, seed_prompt: str) -> None:
        self.role = role
        node = {"prompt": seed_prompt, "cumulative_score": 0.0}
        self.beam = [node]
        self.best_overall = dict(node)
        self.depth = 0


def step_role(
    state: RoleState,
    runner: Any,
    cand_runners: list,
    judge: Judge,
    best_prompt: dict[str, str],
    train_rows: list,
    budget: Any,
    rng: random.Random,
    num_threads: int,
    refl_inflight: int,
    beam_width: int,
    k_sub: int,
    minibatch: int,
    is_terminal: bool,
    successor_role: str | None = None,
    mis_buffers: dict[str, list] | None = None,
    predecessors: list[str] | None = None,
) -> dict:
    """One beam-search depth step for one role (upstream ``process_single_node`` loop)."""
    role = state.role
    entry = {"role": role, "depth": state.depth, "minibatch_ids": [], "nodes": []}
    all_children: list[dict] = []
    best_node_this_step = dict(state.best_overall)

    for node in list(state.beam):
        if budget.exhausted:
            break
        eval_samples = _sample_minibatch(train_rows, minibatch, rng, mis_buffers, predecessors)
        entry["minibatch_ids"].append([_row_id(r) for r in eval_samples])
        temp_map = dict(best_prompt)
        temp_map[role] = node["prompt"]

        # Parent-baseline minibatch (charged).
        granted = budget.reserve(len(eval_samples))
        items = eval_samples[:granted]
        node_entry = {"parent_cumulative": node["cumulative_score"], "baseline_rollouts": granted, "candidates": []}
        entry["nodes"].append(node_entry)
        if not items:
            all_children.append(dict(node))
            break
        base_runs = run_bundle_minibatch(runner, temp_map, items, num_threads)

        def qa_from(runs):
            return {
                r["question"]: {
                    "context": role_context(r["messages"], role, r.get("task_context") or r["question"]),
                    "output": role_output(r["messages"], role),
                }
                for r in runs
            }

        # K_sub variations, one per minibatch half.
        mid = len(base_runs) // 2
        halves = [base_runs[:mid] or base_runs, base_runs[mid:] or base_runs]
        halves = (halves * k_sub)[:k_sub]
        with ThreadPoolExecutor(max_workers=max(1, k_sub)) as pool:
            proposals = list(
                pool.map(
                    lambda runs, parent=node: judge.propose_new_prompt(role, parent["prompt"], qa_from(runs)), halves
                )
            )
        candidates = list(dict.fromkeys(proposals))

        node_best = {"prompt": node["prompt"], "score": 0.0, "mis": []}
        # Phase 1: reserve each candidate's minibatch in candidate order.
        planned: list[tuple[str, int, list]] = []
        folds: list[tuple[str, int | None]] = []
        for cand in candidates:
            if cand == node["prompt"]:
                folds.append(("identical", None))  # upstream: identical proposal scores 0.0, no rollouts
                continue
            granted_c = budget.reserve(len(items))
            folds.append(("eval", len(planned)))
            planned.append((cand, granted_c, items[:granted_c]))
        # Phase 2: evaluate the granted candidate minibatches concurrently.
        cand_runs_lists: list[list[dict]] = []
        if planned:
            bundles = [{**temp_map, role: cand} for cand, _, _ in planned]
            cand_runs_lists = run_candidate_minibatches(
                cand_runners[: len(planned)], bundles, [p[2] for p in planned], num_threads
            )
        # Phase 3: score candidates in order.
        infos = []
        mis_by_cand: dict[int, list] = {}
        for pi, ((_cand, _granted_c, cand_items), cand_runs) in enumerate(zip(planned, cand_runs_lists)):
            cand_mis: list = []
            infos.append(
                evaluate_candidate(
                    judge,
                    role,
                    is_terminal,
                    base_runs[: len(cand_runs)],
                    cand_runs,
                    refl_inflight,
                    successor_role=successor_role,
                    mis_out=(cand_mis if mis_buffers is not None else None),
                    items=cand_items,
                )
            )
            cand_mis.sort(key=lambda t: t[0])
            mis_by_cand[pi] = [row for _priority, row in cand_mis[:MIS_CAP]]
        # Phase 4: fold results in candidate order (upstream tie-breaks).
        for kind, pi in folds:
            if kind == "identical":
                node_entry["candidates"].append({"identical_to_parent": True, "score": 0.0})
                all_children.append(dict(node))
                continue
            cand, granted_c, _cand_items = planned[pi]
            info = infos[pi]
            node_entry["candidates"].append(
                {
                    "score": round(info["score"], 4),
                    "rate_local": info["rate_local"],
                    "rate_global": info["rate_global"],
                    "n_pairs": info["n_pairs"],
                    "rollouts": granted_c,
                    "prompt_chars": len(cand),
                }
            )
            if info["score"] > 0:
                all_children.append({"prompt": cand, "cumulative_score": node["cumulative_score"] + info["score"]})
                if info["score"] > node_best["score"]:
                    node_best = {"prompt": cand, "score": info["score"], "mis": mis_by_cand.get(pi, [])}
            else:
                all_children.append(dict(node))
        if not candidates:
            all_children.append(dict(node))

        node_cumulative = node["cumulative_score"] + max(0.0, node_best["score"])
        if node_cumulative > best_node_this_step["cumulative_score"]:
            best_node_this_step = {
                "prompt": node_best["prompt"],
                "cumulative_score": node_cumulative,
                "mis": node_best.get("mis", []),
            }

    if all_children:
        all_children.sort(key=lambda x: x["cumulative_score"], reverse=True)
        state.beam = all_children[:beam_width]

    # Anchor shift (coordinate ascent): a strict improvement updates the shared bundle.
    anchor_updated = False
    if best_node_this_step["cumulative_score"] > state.best_overall["cumulative_score"] + 1e-6:
        state.best_overall = {
            "prompt": best_node_this_step["prompt"],
            "cumulative_score": best_node_this_step["cumulative_score"],
        }
        best_prompt[role] = best_node_this_step["prompt"]
        anchor_updated = True
        if mis_buffers is not None and best_node_this_step.get("mis"):
            # The anchor-updating candidate's cases replace the role's buffer.
            mis_buffers[role] = list(best_node_this_step["mis"])

    state.depth += 1
    entry["anchor_updated"] = anchor_updated
    entry["best_cumulative"] = round(state.best_overall["cumulative_score"], 4)
    entry["rollouts_used_after"] = budget.used
    return entry


def beam_refresh(
    state: RoleState,
    runner: Any,
    judge: Judge,
    best_prompt: dict[str, str],
    train_rows: list,
    budget: Any,
    rng: random.Random,
    num_threads: int,
    refl_inflight: int,
    minibatch: int,
    is_terminal: bool,
    beam_width: int = 2,
) -> dict:
    """Eq. 8 beam refresh at a role revisit.

    Re-score each non-anchor beam node against the current global best on a
    fresh minibatch (0.7 local + 0.3 global, no lookahead; anchor pinned at
    0.0), re-rank, reset ``best_overall`` to the refreshed top node as upstream
    does unconditionally, and shift the anchor when the top node is not it.
    """
    role = state.role
    anchor_prompt = best_prompt[role]
    entry = {"role": role, "depth": state.depth, "refresh": True, "nodes": [], "anchor_updated": False}
    others = [n for n in state.beam if n["prompt"].strip() != anchor_prompt.strip()]
    if not others or budget.exhausted:
        anchor_node = {"prompt": anchor_prompt, "cumulative_score": 0.0}
        state.beam = ([anchor_node] + [{"prompt": n["prompt"], "cumulative_score": 0.0} for n in others])[:beam_width]
        state.best_overall = dict(anchor_node)
        entry["skipped"] = "no-non-anchor-nodes" if not others else "budget-exhausted"
        entry["best_cumulative"] = 0.0
        entry["rollouts_used_after"] = budget.used
        return entry

    fresh = rng.sample(train_rows, min(minibatch, len(train_rows)))
    if budget.cap - budget.used < len(fresh) + 1:
        # The anchor batch alone would exhaust the budget with nothing to compare against.
        anchor_node = {"prompt": anchor_prompt, "cumulative_score": 0.0}
        state.beam = ([anchor_node] + [{"prompt": n["prompt"], "cumulative_score": 0.0} for n in others])[:beam_width]
        state.best_overall = dict(anchor_node)
        entry.update({"skipped": "budget-tail", "best_cumulative": 0.0, "rollouts_used_after": budget.used})
        return entry
    granted = budget.reserve(len(fresh))
    items = fresh[:granted]
    refreshed = [{"prompt": anchor_prompt, "cumulative_score": 0.0}]
    if items:
        base_runs = run_bundle_minibatch(runner, dict(best_prompt), items, num_threads)
        for node in others:
            g2 = budget.reserve(len(items))
            node_items = items[:g2]
            if not node_items:
                refreshed.append({"prompt": node["prompt"], "cumulative_score": 0.0})
                continue
            node_map = dict(best_prompt)
            node_map[role] = node["prompt"]
            node_runs = run_bundle_minibatch(runner, node_map, node_items, num_threads)
            info = evaluate_candidate(judge, role, is_terminal, base_runs[: len(node_runs)], node_runs, refl_inflight)
            refreshed.append({"prompt": node["prompt"], "cumulative_score": info["score"]})
            entry["nodes"].append(
                {"refreshed_score": round(info["score"], 4), "rollouts": g2, "prompt_chars": len(node["prompt"])}
            )
    else:
        refreshed += [{"prompt": n["prompt"], "cumulative_score": 0.0} for n in others]

    refreshed.sort(key=lambda x: x["cumulative_score"], reverse=True)
    state.beam = refreshed[:beam_width]
    state.best_overall = dict(refreshed[0])
    if refreshed[0]["cumulative_score"] > 1e-6 and refreshed[0]["prompt"].strip() != anchor_prompt.strip():
        best_prompt[role] = refreshed[0]["prompt"]
        entry["anchor_updated"] = True
    entry["best_cumulative"] = round(refreshed[0]["cumulative_score"], 4)
    entry["rollouts_used_after"] = budget.used
    return entry


__all__ = [
    "FALLBACK_WEIGHTS",
    "JUDGE_TEMPERATURE",
    "Judge",
    "LOOKAHEAD_WEIGHTS",
    "MIS_CAP",
    "MIS_INJECT",
    "PROPOSAL_TEMPERATURE",
    "RoleState",
    "SCORE_OFFSET",
    "TaskType",
    "beam_refresh",
    "dataset_contract",
    "evaluate_candidate",
    "regime_mode",
    "role_description",
    "run_bundle_minibatch",
    "run_candidate_minibatches",
    "step_role",
]
