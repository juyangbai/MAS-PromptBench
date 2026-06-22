"""Centralized topology specialized for BFCL, AutoGen."""

# Config
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path
from typing import Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Shared telemetry.
_TOPO_ROOT = str(Path(__file__).resolve().parents[4])
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import autogen_telemetry, normalize  # noqa: E402

from huggingface_hub import hf_hub_download

from bfcl_eval.constants.enums import Language
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "centralized" / "bfcl"

HF_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
AST_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple")


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


# Model registration in bfcl-eval
def _register_model_with_bfcl(model_id: str) -> None:
    """Tell bfcl-eval how to handle function names for `model_id`.

    `ast_checker` -> `convert_func_name` looks up the model in
    MODEL_CONFIG_MAPPING to decide whether to rewrite '.' -> '_' in
    function names. Qwen3.5-9B handles dots fine (same as the
    registered qwen3-8b/-14b entries), but our model name isn't in
    the registry, so without this we hit KeyError on any instance
    with a dotted name (e.g. `math.factorial`).
    """
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


# LLM
def _build_client() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model=MODEL_ID,
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "qwen",
            "structured_output": False,
        },
        temperature=0.2,
        top_p=0.9,
        seed=0,
        # BFCL calls are short; bounded output is fine.
        max_tokens=1024,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


# Team
_MANAGER_TERMINATE_NUDGE = (
    "\n\nWhen you emit the final fenced ```json``` block containing "
    "the canonical call list, immediately follow it with the literal "
    "string TERMINATE on its own line so the group-chat knows to stop."
)


def build_team() -> SelectorGroupChat:
    client = _build_client()

    manager = AssistantAgent(
        "manager",
        description="Coordinator that plans the call, dispatches composition + validation, and emits the final canonical JSON.",
        model_client=client,
        system_message=_load_prompt("manager") + _MANAGER_TERMINATE_NUDGE,
    )

    inspector_worker = AssistantAgent(
        "inspector_worker",
        description="Reads the schema and returns an argument plan.",
        model_client=client,
        system_message=_load_prompt("inspector_worker"),
    )

    caller_worker = AssistantAgent(
        "caller_worker",
        description="Composes the canonical JSON call per manager instruction.",
        model_client=client,
        system_message=_load_prompt("caller_worker"),
    )

    validator_worker = AssistantAgent(
        "validator_worker",
        description="Checks the call against the schema (name exists, required params present, types valid).",
        model_client=client,
        system_message=_load_prompt("validator_worker"),
    )

    # Force manager-routing: after any worker speaks, the manager MUST be
    # the next speaker (so workers never chain turns with each other).
    def _selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
        if not messages:
            return manager.name
        if messages[-1].source != manager.name:
            return manager.name
        return None

    selector_prompt = (
        "You are coordinating a 4-agent team on a BFCL function-calling task.\n"
        "Select the next agent to act.\n\n{roles}\n\n"
        "Conversation so far:\n{history}\n\n"
        "Pick exactly one agent from {participants}."
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(24)

    return SelectorGroupChat(
        [manager, inspector_worker, caller_worker, validator_worker],
        model_client=client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        selector_func=_selector_func,
        allow_repeated_speaker=True,
    )


# Prompt scaffolding
def format_task(user_request: str, schemas_text: str) -> str:
    """Build the single task string sent to the group chat.

    The user request + schemas both go into the initial `user` message
    so every agent sees them from the start of the dialogue.
    """
    return (
        "USER REQUEST:\n"
        f"{user_request}\n\n"
        "SCHEMAS:\n"
        f"{schemas_text}\n\n"
        "Emit the final canonical call list as a SINGLE fenced ```json "
        "block. Canonical form is a list of dicts; each dict has exactly "
        "ONE key equal to the ACTUAL function name from one of the "
        "schemas above, and the value is the arguments dict. "
        "Do NOT use the literal string 'fn_name' as the key, and do NOT "
        "use the shape {\"fn_name\": \"<name>\", \"args\": {...}}. "
        "Example for a schema named calculate_area: "
        "[{\"calculate_area\": {\"width\": 5, \"height\": 3}}]. "
        "For parallel calls, emit multiple such dicts in the list."
    )


# Output parsing
_FENCED_RE = re.compile(r"```(?:\w*)\s*([\s\S]*?)\s*```")

# Common non-canonical shapes the model emits instead of {name: args}:
#   {"fn_name": "<real>", "args": {...}}           (model misreads the template)
#   {"name": "<real>", "arguments": {...}}         (OpenAI tool-call shape)
#   {"function_name": "<real>", "arguments": {...}}
_NAME_ARGS_PAIRS = (
    ("fn_name", "args"),
    ("name", "arguments"),
    ("function_name", "arguments"),
    ("function", "arguments"),
)


def _normalize_call(d: dict) -> dict:
    """Normalize the common wrong shapes back to canonical {name: args}.

    Leaves a dict alone if it's already canonical (single string-keyed arg dict).
    """
    for name_key, args_key in _NAME_ARGS_PAIRS:
        if (
            name_key in d
            and args_key in d
            and isinstance(d[name_key], str)
            and isinstance(d[args_key], dict)
        ):
            return {d[name_key]: d[args_key]}
    return d


def extract_canonical(text: str) -> list[dict] | None:
    """Extract the last fenced JSON list-of-dicts from `text`, normalizing
    common non-canonical shapes emitted by chat models.

    Returns None if no fenced JSON parses to a non-empty list-of-dicts.
    Strips trailing TERMINATE so the fence regex doesn't misalign.
    """
    text = re.sub(r"\bTERMINATE\b", "", text)
    candidates: list[str] = [m.group(1) for m in _FENCED_RE.finditer(text)]
    for cand in reversed(candidates):
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
            return [_normalize_call(d) for d in parsed]
    return None


# Scoring
def score_one(
    function_schemas: list[dict],
    model_output: list[dict],
    ground_truth: list[dict],
    category: str,
) -> dict:
    """Delegate to bfcl-eval's AST checker (aligned to other topologies)."""
    return ast_checker(
        function_schemas,
        model_output,
        ground_truth,
        Language.PYTHON,
        category,
        MODEL_ID,
    )


# Dataset
def load_instances(
    category: str,
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load (rows, ground_truths) for one BFCL subset."""
    main = Path(
        hf_hub_download(HF_DATASET, f"BFCL_v3_{category}.json", repo_type="dataset")
    )
    ans = Path(
        hf_hub_download(
            HF_DATASET,
            f"possible_answer/BFCL_v3_{category}.json",
            repo_type="dataset",
        )
    )

    rows = [json.loads(line) for line in main.read_text().splitlines() if line.strip()]
    gt_by_id = {
        json.loads(line)["id"]: json.loads(line)
        for line in ans.read_text().splitlines()
        if line.strip()
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


# Orchestration
def _flatten_user_request(question: list) -> str:
    """BFCL stores `question` as [[msgs...]]; for single-turn subsets we
    take the concatenated user-turn contents."""
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


async def solve_async(instance: dict) -> dict:
    """Run the centralized team on one BFCL instance.

    Returns:
        {
            "model_output": canonical call list [{fn: {arg: val}}] or [],
            "raw":          manager's last message content,
            "messages":     list of {source, content} from every turn,
        }
    """
    team = build_team()
    user_request = _flatten_user_request(instance["question"])
    schemas_text = json.dumps(instance["function"], indent=2)
    task = format_task(user_request, schemas_text)

    result = await team.run(task=task)
    messages = [
        {
            "source": getattr(m, "source", None),
            "content": getattr(m, "content", None)
            if isinstance(getattr(m, "content", None), str)
            else str(getattr(m, "content", "")),
        }
        for m in result.messages
    ]
    manager_msgs = [m for m in messages if m["source"] == "manager"]
    final = manager_msgs[-1]["content"] if manager_msgs else ""
    # Prefer manager's final fenced JSON; fall back to caller_worker's.
    model_output = extract_canonical(final)
    if model_output is None:
        caller_msgs = [m for m in messages if m["source"] == "caller_worker"]
        if caller_msgs:
            model_output = extract_canonical(caller_msgs[-1]["content"])
    return {
        "model_output": model_output or [],
        "raw": final,
        "messages": messages,
        "telemetry": normalize(autogen_telemetry(result)),
    }


def solve(instance: dict) -> dict:
    return asyncio.run(solve_async(instance))


# Batch runner
def run_one(
    instance: dict,
    ground_truth: dict,
    category: str,
    out_dir: Path,
) -> dict:
    """Solve one BFCL instance via the manager-worker team and score the
    canonical output. Writes group-chat trace to out_dir/traces/<id>.txt."""
    iid = instance["id"]
    summary: dict = {"id": iid, "category": category}

    try:
        out = solve(instance)
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["stage"] = "solve"
        return summary

    summary["model_output"] = out.get("model_output") or []
    summary["n_messages"] = len(out.get("messages") or [])
    summary["tool_calls"] = len(summary["model_output"])
    summary.update(out.get("telemetry") or {})

    trace_path = out_dir / "traces" / f"{iid}.txt"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w") as f:
        for m in out.get("messages") or []:
            src = m.get("source", "?")
            content = m.get("content", "")
            f.write(f"=== {str(src).upper()} ===\n{content}\n\n")

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
    out_dir = out_dir or (_default_root / "results" / "bfcl_centralized")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, gts = load_instances(category, limit, offset, only)
    if verbose:
        print(f"loaded {len(rows)} instance(s) from {HF_DATASET} / {category}")

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
        description="Centralized-topology BFCL runner (AutoGen SelectorGroupChat)."
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
        Path(__file__).resolve().parents[4] / "results" / "bfcl_centralized"
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
