"""Shared ToolHop helpers for team-size variants.

The dataset/tool execution path is owned by the canonical self-contained runner
``topologies.single.toolhop.langgraph_toolhop``.
This module adds the team-size sweep layer used by ``teamsizes/<topo>/toolhop``:
run N seeded replicas, majority-vote the extracted final answers, and write the
same predictions/results/traces artifacts as the other team-size datasets.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from topologies.single.toolhop import langgraph_toolhop as _base


MODEL_ID = _base.MODEL_ID
VLLM_BASE_URL = _base.VLLM_BASE_URL
HF_DATASET = _base.HF_DATASET
load_instances = _base.load_instances
dataset_summary = _base.dataset_summary

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _answer_key(answer: str | None) -> str:
    return (answer or "").strip().removesuffix(".0").replace(",", "").lower()


def _sum_telemetry(per_agent: list[dict]) -> dict:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "n_llm_calls": 0,
        "n_tool_calls": 0,
    }
    for agent in per_agent:
        telemetry = agent.get("telemetry") or {}
        for key in totals:
            totals[key] += int(telemetry.get(key) or 0)
    return totals


def _solve_replica(
    sample: dict,
    *,
    style: str,
    topology: str,
    role: str,
    seed: int,
) -> dict:
    start = time.time()
    try:
        out = _base.solve(
            sample,
            style=style,
            topology=topology,
            role=role,
            seed=seed,
        )
    except Exception as exc:
        return {
            "seed": seed,
            "solve_s": round(time.time() - start, 1),
            "error": f"{type(exc).__name__}: {exc}",
            "messages": [],
            "telemetry": {},
        }

    messages = out.get("messages") or []
    final_content = _base._last_assistant_content(messages)
    predicted = _base.extract_answer(final_content)
    correct = _base.score_answer(
        str(sample.get("answer", "")),
        final_content,
        _base._previous_tool_content(messages),
    )
    return {
        "seed": seed,
        "solve_s": round(float(out.get("solve_s") or 0.0), 1),
        "predicted_answer": predicted,
        "answer_key": _answer_key(predicted),
        "answer_correct": int(correct),
        "correct": bool(correct),
        "turns": sum(1 for message in messages if message.get("role") == "assistant"),
        "tool_calls": sum(len(message.get("tool_calls") or []) for message in messages),
        "messages": messages,
        "telemetry": out.get("telemetry") or {},
    }


def _choose_winner(per_agent: list[dict]) -> int | None:
    candidates = [
        (idx, agent.get("answer_key") or "")
        for idx, agent in enumerate(per_agent)
        if not agent.get("error") and agent.get("predicted_answer") is not None
    ]
    if not candidates:
        return None
    counts = Counter(key for _, key in candidates)
    first_seen: dict[str, int] = {}
    for idx, key in candidates:
        first_seen.setdefault(key, idx)
    winner_key = max(counts, key=lambda key: (counts[key], -first_seen[key]))
    return first_seen[winner_key]


def _write_trace(path: Path, summary: dict, per_agent: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(
            f"style: {summary.get('style')}\n"
            f"team_size: {summary.get('team_size')}\n"
            f"winner: {summary.get('winner')}\n"
            f"predicted_answer: {summary.get('predicted_answer')}\n"
            f"gold_answer: {summary.get('gold_answer')}\n\n"
        )
        for idx, agent in enumerate(per_agent):
            f.write(f"=== AGENT {idx} seed={agent.get('seed')} ===\n")
            if agent.get("error"):
                f.write(f"error: {agent['error']}\n\n")
                continue
            f.write(
                f"predicted_answer: {agent.get('predicted_answer')}\n"
                f"correct: {agent.get('correct')}\n"
                f"turns: {agent.get('turns')} tool_calls: {agent.get('tool_calls')}\n"
            )
            messages = agent.get("messages") or []
            final = _base._last_assistant_content(messages)
            if final:
                f.write("\nfinal assistant:\n" + final + "\n")
            f.write("\n")


def run_one(
    instance: dict,
    out_dir: Path,
    *,
    style: str,
    topology: str,
    role: str,
    team_size: int,
) -> dict:
    iid = instance["id"]
    summary: dict[str, Any] = {
        "id": iid,
        "idx": iid,
        "question": instance.get("question"),
        "gold_answer": instance.get("answer"),
        "style": style,
        "team_size": team_size,
        "n_agents": team_size,
    }

    t0 = time.time()
    per_agent = [
        _solve_replica(instance, style=style, topology=topology, role=role, seed=seed)
        for seed in range(team_size)
    ]
    summary["solve_s"] = round(time.time() - t0, 1)
    summary["per_agent"] = [
        {k: v for k, v in agent.items() if k != "messages"}
        for agent in per_agent
    ]
    summary["buckets"] = dict(Counter(
        agent.get("answer_key") or ""
        for agent in per_agent
        if not agent.get("error")
    ))

    winner = _choose_winner(per_agent)
    summary["winner"] = winner
    if winner is None:
        summary["predicted_answer"] = ""
        summary["answer_correct"] = 0
        summary["correct"] = False
        summary["error"] = "all ToolHop replicas failed"
        summary["stage"] = "solve"
    else:
        winning = per_agent[winner]
        predicted = winning.get("predicted_answer") or ""
        summary["predicted_answer"] = predicted
        summary["answer_correct"] = int(
            _base.score_answer(str(instance.get("answer", "")), predicted)
        )
        summary["correct"] = bool(summary["answer_correct"])

    summary["tool_calls"] = sum(int(agent.get("tool_calls") or 0) for agent in per_agent)
    summary["turns"] = sum(int(agent.get("turns") or 0) for agent in per_agent)
    summary.update(_sum_telemetry(per_agent))

    _write_trace(out_dir / "traces" / f"{iid}.txt", summary, per_agent)
    return summary


def run_batch(
    *,
    style: str,
    topology: str,
    role: str,
    team_size: int,
    limit: int | None = None,
    offset: int = 0,
    only: list[str | int] | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    out_dir = out_dir or (_REPO_ROOT / "results" / "toolhop" / style)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_instances(limit=limit, offset=offset, only=only)
    if verbose:
        print(f"loaded {len(rows)} instance(s) from {HF_DATASET} ({style}, N={team_size})")

    preds_path = out_dir / "predictions.jsonl"
    results_path = out_dir / "results.jsonl"
    correct = 0
    with preds_path.open("a") as fp, results_path.open("a") as fr:
        for index, row in enumerate(rows, 1):
            if verbose:
                print(f"\n[{index}/{len(rows)}] {row['id']}")
            summary = run_one(
                row,
                out_dir,
                style=style,
                topology=topology,
                role=role,
                team_size=team_size,
            )
            correct += int(bool(summary.get("correct")))
            fp.write(
                json.dumps(
                    {
                        "idx": row["id"],
                        "id": row["id"],
                        "question": row.get("question"),
                        "predicted_answer": summary.get("predicted_answer"),
                        "model_name_or_path": MODEL_ID,
                    },
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
            fr.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")
            fp.flush()
            fr.flush()
            if verbose:
                compact = {k: v for k, v in summary.items() if k not in {"per_agent", "buckets"}}
                print(f"  -> {json.dumps(compact, ensure_ascii=False, default=str)}")

    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": (correct / len(rows)) if rows else 0.0,
        "style": style,
        "team_size": team_size,
    }


def main(*, style: str, topology: str, role: str, team_size: int) -> int:
    parser = argparse.ArgumentParser(description=f"ToolHop team-size runner ({style}).")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--batch", action="store_true", help="accepted for CLI uniformity; this runner always runs a batch")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "results" / "toolhop" / style))
    parser.add_argument(
        "--smoke-dataset",
        action="store_true",
        help="Only load and validate ToolHop; do not call the model or execute tools.",
    )
    args = parser.parse_args()

    if args.smoke_dataset:
        print(json.dumps(dataset_summary(limit=args.limit), indent=2, ensure_ascii=False, default=str))
        return 0

    run_batch(
        style=style,
        topology=topology,
        role=role,
        team_size=team_size,
        limit=args.limit if not args.only else None,
        offset=args.offset,
        only=args.only,
        out_dir=Path(args.out_dir),
    )
    return 0
