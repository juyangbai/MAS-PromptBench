"""Shared API-Bank helpers for team-size variants."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from topologies.single.apibank import langgraph_apibank as _base


MODEL_ID = _base.MODEL_ID
VLLM_BASE_URL = _base.VLLM_BASE_URL
BENCHMARK_NAME = _base.BENCHMARK_NAME
load_instances = _base.load_instances
dataset_summary = _base.dataset_summary

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _answer_key(answer: str | None) -> str:
    try:
        name, params = _base.parse_api_call(answer)
    except Exception:
        return (answer or "").strip().lower()
    return json.dumps({"name": name, "params": params}, sort_keys=True, default=str)


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
        out = _base.solve(sample, style=style, topology=topology, role=role, seed=seed)
    except Exception as exc:
        return {
            "seed": seed,
            "solve_s": round(time.time() - start, 1),
            "error": f"{type(exc).__name__}: {exc}",
            "raw": "",
            "telemetry": {},
        }
    raw = out.get("raw") or ""
    predicted = _base.extract_api_call(raw)
    scored = _base.score_prediction(sample, predicted)
    return {
        "seed": seed,
        "solve_s": round(float(out.get("solve_s") or 0.0), 1),
        "raw": raw,
        "predicted_answer": predicted,
        "answer_key": _answer_key(predicted),
        "answer_correct": int(bool(scored.get("correct"))),
        "correct": bool(scored.get("correct")),
        "stage": scored.get("stage"),
        "error": scored.get("error"),
        "turns": 1,
        "tool_calls": 0,
        "telemetry": out.get("telemetry") or {},
    }


def _choose_winner(per_agent: list[dict]) -> int | None:
    candidates = [
        (idx, agent.get("answer_key") or "")
        for idx, agent in enumerate(per_agent)
        if str(agent.get("predicted_answer") or "").strip()
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
    path.write_text(
        json.dumps(
            {
                "summary": summary,
                "per_agent": per_agent,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


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
        "file": instance.get("file"),
        "sample_id": instance.get("sample_id"),
        "question": _base._format_chat_history(instance.get("chat_history") or []),
        "gold_api_call": instance.get("gold_api_call"),
        "gold_api_name": instance.get("ground_truth", {}).get("api_name"),
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
    summary["per_agent"] = [{k: v for k, v in agent.items() if k != "raw"} for agent in per_agent]
    summary["buckets"] = dict(
        Counter(
            agent.get("answer_key") or ""
            for agent in per_agent
            if str(agent.get("predicted_answer") or "").strip()
        )
    )
    winner = _choose_winner(per_agent)
    summary["winner"] = winner
    if winner is None:
        summary["predicted_answer"] = ""
        summary["answer_correct"] = 0
        summary["correct"] = False
        summary["stage"] = "solve"
        summary["error"] = "all API-Bank replicas failed"
    else:
        predicted = per_agent[winner].get("predicted_answer") or ""
        scored = _base.score_prediction(instance, predicted)
        summary["predicted_answer"] = predicted
        summary["answer_correct"] = int(bool(scored.get("correct")))
        summary["correct"] = bool(scored.get("correct"))
        summary["stage"] = scored.get("stage")
        summary["error"] = scored.get("error")
    summary["tool_calls"] = 0
    summary["turns"] = len(per_agent)
    summary.update(_sum_telemetry(per_agent))
    safe_id = iid.replace("/", "_").replace(":", "_")
    _write_trace(Path(out_dir) / "traces" / f"{safe_id}.json", summary, per_agent)
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
    level: str | int | None = None,
) -> dict:
    out_dir = out_dir or (_REPO_ROOT / "results" / "apibank" / style)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_instances(limit=limit, offset=offset, only=only, level=level)
    if verbose:
        print(f"loaded {len(rows)} instance(s) from {_base.benchmark_name(level)} ({style}, N={team_size})")
    preds_path = out_dir / "predictions.jsonl"
    results_path = out_dir / "results.jsonl"
    correct = 0
    with preds_path.open("a") as fp, results_path.open("a") as fr:
        for index, row in enumerate(rows, 1):
            if verbose:
                print(f"\n[{index}/{len(rows)}] {row['id']}")
            summary = run_one(row, out_dir, style=style, topology=topology, role=role, team_size=team_size)
            correct += int(bool(summary.get("correct")))
            fp.write(
                json.dumps(
                    {
                        "idx": row["id"],
                        "id": row["id"],
                        "question": summary.get("question"),
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
                print(f"  -> {json.dumps(summary, ensure_ascii=False, default=str)}")
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": (correct / len(rows)) if rows else 0.0,
        "style": style,
        "team_size": team_size,
    }


def main(
    *,
    style: str,
    topology: str,
    role: str,
    team_size: int,
) -> int:
    parser = argparse.ArgumentParser(description=f"{BENCHMARK_NAME} team-size runner")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--batch", action="store_true", help="accepted for CLI uniformity; this runner always runs a batch")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "results" / "apibank" / style))
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--level", default=None, help="API-Bank slice: all, 1, 2, or 3")
    parser.add_argument("--curated-path", default=None)
    parser.add_argument(
        "--toolsearcher-scorer",
        choices=["official", "upstream", "keyword"],
        default=None,
        help="ToolSearcher scorer.",
    )
    args = parser.parse_args()
    if args.curated_path:
        os.environ["APIBANK_CURATED_PATH"] = str(Path(args.curated_path).expanduser().resolve())
    if args.toolsearcher_scorer:
        os.environ["APIBANK_TOOLSEARCHER_SCORER"] = args.toolsearcher_scorer
    if args.summary:
        print(json.dumps(dataset_summary(limit=args.limit, level=args.level), indent=2, default=str))
        return 0
    run_batch(
        style=style,
        topology=topology,
        role=role,
        team_size=team_size,
        limit=args.limit,
        offset=args.offset,
        only=args.only,
        out_dir=Path(args.out_dir),
        level=args.level,
    )
    return 0
