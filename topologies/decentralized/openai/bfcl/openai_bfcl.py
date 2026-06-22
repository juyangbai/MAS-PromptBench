"""Decentralized debate topology specialized for BFCL, OpenAI SDK."""

# Config
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path

from huggingface_hub import hf_hub_download
from openai import OpenAI

# Shared telemetry.
_TOPO_ROOT = str(Path(__file__).resolve().parents[4])
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import (  # noqa: E402
    openai_sdk_telemetry, openai_sdk_accumulate, normalize,
)


_TELEM_ACC: dict = {
    "prompt_tokens": 0, "completion_tokens": 0,
    "total_tokens": 0, "n_llm_calls": 0, "n_tool_calls": 0,
}


def _reset_telem_acc() -> None:
    for k in _TELEM_ACC:
        _TELEM_ACC[k] = 0

from bfcl_eval.constants.enums import Language
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

N_AGENTS = int(os.environ.get("DECENTRALIZED_N_AGENTS", "4"))
N_ROUNDS = int(os.environ.get("DECENTRALIZED_N_ROUNDS", "2"))

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "decentralized" / "bfcl"

HF_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
AST_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple")


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


SYSTEM_PROMPT = _load_prompt("debater")


# Model registration (same as other bfcl topologies)
def _register_model_with_bfcl(model_id: str) -> None:
    if model_id in MODEL_CONFIG_MAPPING:
        return
    template = MODEL_CONFIG_MAPPING["qwen3-8b"]
    MODEL_CONFIG_MAPPING[model_id] = ModelConfig(
        model_name=model_id,
        display_name=model_id,
        url=template.url,
        org=template.org,
        license=template.license,
        model_handler=template.model_handler,
        is_fc_model=True,
        underscore_to_dot=False,
    )


_register_model_with_bfcl(MODEL_ID)


# Client
def _build_client() -> OpenAI:
    # Explicit timeout + retries so transient vLLM hiccups on long parallel
    # generations don't kill a 25-row batch with APIConnectionError.
    return OpenAI(
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        timeout=300.0,
        max_retries=5,
    )


def _completion_kwargs() -> dict:
    return {
        "model": MODEL_ID,
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": 0,
        "max_tokens": 1024,
        "extra_body": {
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def _chat(client: OpenAI, messages: list[dict]) -> dict:
    """Single LLM call — BFCL debaters have no tools, pure text output."""
    resp = client.chat.completions.create(messages=messages, **_completion_kwargs())
    openai_sdk_accumulate(_TELEM_ACC, resp)
    msg = resp.choices[0].message
    return msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)


# Debate loop
def _peer_injection(others_final: list[dict], user_content: str) -> dict:
    body = ["These are the final calls from other peer agents in the previous round:"]
    for i, m in enumerate(others_final):
        body.append(f"\nPeer {i + 1}:\n```\n{m.get('content') or ''}\n```")
    body.append(
        "\nCompare their function calls against yours. Revise ONLY if a peer "
        "picked a better function or caught an error in yours. Re-emit your "
        "final canonical call as a SINGLE fenced ```json``` block at the "
        "end.\n\nOriginal request:\n" + user_content
    )
    return {"role": "user", "content": "\n".join(body)}


def _format_task(user_request: str, schemas_text: str) -> str:
    return (
        "USER REQUEST:\n"
        f"{user_request}\n\n"
        "SCHEMAS:\n"
        f"{schemas_text}\n\n"
        "Emit your final canonical call list as a SINGLE fenced ```json``` "
        "block. Canonical form is a list of dicts with one key per dict: "
        "[{\"fn_name\": {\"arg\": value, ...}}, ...]. "
        "For a single call, emit a one-element list. For parallel calls, "
        "emit a multi-element list."
    )


def run_debate(user_request: str, schemas_text: str) -> list[list[dict]]:
    client = _build_client()
    user_content = _format_task(user_request, schemas_text)
    contexts: list[list[dict]] = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        for _ in range(N_AGENTS)
    ]
    round_finals: list[list[dict]] = []
    for r in range(N_ROUNDS):
        this_round: list[dict] = []
        for i, ctx in enumerate(contexts):
            if r > 0:
                others = [round_finals[r - 1][j] for j in range(N_AGENTS) if j != i]
                ctx.append(_peer_injection(others, user_content))
            final_msg = _chat(client, ctx)
            ctx.append(final_msg)
            this_round.append(final_msg)
        round_finals.append(this_round)
    return contexts


# Output parsing
_FENCED_RE = re.compile(r"```(?:\w*)\s*([\s\S]*?)\s*```")


def extract_canonical(text: str) -> list[dict] | None:
    """Extract the last fenced JSON list-of-dicts from `text`."""
    candidates: list[str] = [m.group(1) for m in _FENCED_RE.finditer(text)]
    for cand in reversed(candidates):
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
            return parsed
    return None


# Scoring
def score_one(
    function_schemas: list[dict],
    model_output: list[dict],
    ground_truth: list[dict],
    category: str,
) -> dict:
    """Delegate to bfcl-eval's AST checker."""
    return ast_checker(
        function_schemas, model_output, ground_truth,
        Language.PYTHON, category, MODEL_ID,
    )


# Dataset
def load_instances(
    category: str,
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    main = Path(
        hf_hub_download(HF_DATASET, f"BFCL_v3_{category}.json", repo_type="dataset")
    )
    ans = Path(
        hf_hub_download(HF_DATASET, f"possible_answer/BFCL_v3_{category}.json", repo_type="dataset")
    )
    rows = [json.loads(line) for line in main.read_text().splitlines() if line.strip()]
    gt_by_id = {
        json.loads(line)["id"]: json.loads(line)
        for line in ans.read_text().splitlines() if line.strip()
    }
    if only:
        want = set(only)
        rows = [r for r in rows if r["id"] in want]
    rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    missing = [r["id"] for r in rows if r["id"] not in gt_by_id]
    if missing:
        raise RuntimeError(f"no ground truth for ids: {missing[:5]}")
    return rows, [gt_by_id[r["id"]] for r in rows]


# Aggregation (best-of-N over ast_checker)
def best_of_n(
    per_peer_calls: list[list[dict] | None],
    function_schemas: list[dict],
    ground_truth: list[dict],
    category: str,
) -> tuple[int | None, list[dict]]:
    scored = []
    for i, call in enumerate(per_peer_calls):
        if not call:
            scored.append({"peer": i, "call": call, "valid": False, "report": None})
            continue
        try:
            r = ast_checker(
                function_schemas, call, ground_truth,
                Language.PYTHON, category, MODEL_ID,
            )
        except Exception as e:
            scored.append({"peer": i, "call": call, "valid": False,
                           "report": {"error": f"{type(e).__name__}: {e}"}})
            continue
        valid = bool(r.get("valid"))
        scored.append({"peer": i, "call": call, "valid": valid, "report": r})
    passing = [s for s in scored if s["valid"]]
    if passing:
        winner = min(passing, key=lambda s: s["peer"])
    elif any(s["call"] for s in scored):
        with_call = [s for s in scored if s["call"]]
        winner = min(with_call, key=lambda s: s["peer"])
    else:
        return None, scored
    return winner["peer"], scored


# Orchestration
def _flatten_user_request(question: list) -> str:
    if not question:
        return ""
    turns = question[0] if isinstance(question[0], list) else question
    parts = []
    for msg in turns:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"[{role}] {content}")
        else:
            parts.append(str(msg))
    return "\n".join(parts)


def solve(instance: dict, ground_truth: dict | None = None, category: str = "simple") -> dict:
    _reset_telem_acc()
    user_request = _flatten_user_request(instance["question"])
    schemas_text = json.dumps(instance["function"], indent=2)
    contexts = run_debate(user_request, schemas_text)

    per_peer = []
    for i, ctx in enumerate(contexts):
        final = ctx[-1].get("content") or ""
        call = extract_canonical(final)
        per_peer.append({"peer": i, "call": call, "raw": final})

    telem_src = dict(_TELEM_ACC)
    if telem_src["n_llm_calls"] == 0:
        telem_src = openai_sdk_telemetry(contexts)
    telem = normalize(telem_src)
    if ground_truth is not None:
        winner_idx, scored = best_of_n(
            [p["call"] for p in per_peer],
            instance["function"],
            ground_truth["ground_truth"],
            category,
        )
        return {
            "model_output": per_peer[winner_idx]["call"] if winner_idx is not None else [],
            "winner": winner_idx,
            "per_peer": scored,
            "all_contexts": contexts,
            "telemetry": telem,
        }
    return {
        "model_output": per_peer[0]["call"] or [],
        "winner": 0,
        "per_peer": per_peer,
        "all_contexts": contexts,
        "telemetry": telem,
    }


# Batch runner
def run_one(
    instance: dict,
    ground_truth: dict,
    category: str,
    out_dir: Path,
) -> dict:
    """Solve one BFCL instance with N×R debate, best-of-N pick, score the
    winner. Writes per-peer summaries to out_dir/traces/<id>.txt."""
    iid = instance["id"]
    summary: dict = {
        "id": iid, "category": category,
        "n_peers": N_AGENTS, "n_rounds": N_ROUNDS,
    }

    try:
        out = solve(instance, ground_truth=ground_truth, category=category)
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["stage"] = "solve"
        return summary

    summary["winner"] = out.get("winner")
    summary["model_output"] = out.get("model_output") or []
    summary["tool_calls"] = len(summary["model_output"])
    summary.update(out.get("telemetry") or {})

    trace_path = out_dir / "traces" / f"{iid}.txt"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w") as f:
        f.write(f"winner: peer {out.get('winner')}\n\n")
        for s in out.get("per_peer") or []:
            call = s.get("call") or []
            call_str = json.dumps(call, sort_keys=True) if call else "(no call)"
            f.write(f"=== peer {s.get('peer')} ===\n{call_str}\n")
            if s.get("valid") is not None:
                f.write(f"  valid={s['valid']}  error_type={s.get('error_type')}\n")
            f.write("\n")

    try:
        checker_result = score_one(
            instance["function"],
            summary["model_output"],
            ground_truth["ground_truth"],
            category,
        )
    except Exception as e:
        summary["valid"] = False
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["stage"] = "score"
        return summary

    summary["valid"] = bool(checker_result.get("valid"))
    summary["error_type"] = checker_result.get("error_type")
    if not summary["valid"]:
        summary["score_error"] = (checker_result.get("error") or [])[:3]
    return summary


def run_batch(
    category: str,
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Iterate one AST subset; write predictions.jsonl + results.jsonl."""
    _default_root = Path(__file__).resolve().parents[4]
    out_dir = out_dir or (_default_root / "results" / "bfcl_decentralized")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, gts = load_instances(category, limit, offset, only)
    if verbose:
        print(f"loaded {len(rows)} instance(s) from {HF_DATASET} / {category}  "
              f"(N={N_AGENTS}, R={N_ROUNDS})")

    preds_path = out_dir / "predictions.jsonl"
    results_path = out_dir / "results.jsonl"

    valid = 0
    with preds_path.open("a") as fp, results_path.open("a") as fr:
        for i, (row, gt) in enumerate(zip(rows, gts), start=1):
            if verbose:
                print(f"\n[{i}/{len(rows)}] {row['id']}  ({category})")
            summary = run_one(row, gt, category, out_dir)
            if summary.get("valid"):
                valid += 1

            fp.write(json.dumps({
                "id": row["id"],
                "category": category,
                "model_output": summary.get("model_output"),
                "model_name_or_path": MODEL_ID,
            }) + "\n")
            fr.write(json.dumps(summary) + "\n")
            fp.flush()
            fr.flush()
            if verbose:
                print(f"  -> {json.dumps(summary)}")

    if verbose:
        print(
            f"\ndone: valid {valid}/{len(rows)}"
            f"\n      predictions -> {preds_path}"
            f"\n      results     -> {results_path}"
        )
    return {
        "n": len(rows),
        "valid": valid,
        "valid_rate": (valid / len(rows)) if rows else 0.0,
        "category": category,
    }


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decentralized-topology BFCL runner (debate)."
    )
    parser.add_argument(
        "--category", default="simple", choices=list(AST_CATEGORIES),
        help="BFCL subset (default: simple).",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--only", action="append", default=None, metavar="INSTANCE_ID",
    )
    _default_out = str(
        Path(__file__).resolve().parents[4] / "results" / "bfcl_decentralized"
    )
    parser.add_argument("--out-dir", default=_default_out)
    args = parser.parse_args()

    run_batch(
        category=args.category,
        limit=args.limit if not args.only else None,
        offset=args.offset,
        only=args.only,
        out_dir=Path(args.out_dir),
    )
