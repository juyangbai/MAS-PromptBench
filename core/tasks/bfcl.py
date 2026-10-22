"""BFCL: data, schema tools, call parsing and voting, AST scoring and batch files.

The benchmark covers the four AST-scoreable single-turn subsets of BFCL v3
(gorilla-llm/Berkeley-Function-Calling-Leaderboard). A prediction is a canonical
call list ``[{"<function>": {<arguments>}}, ...]`` scored by bfcl-eval's AST
checker against the instance's possible answers. The single and independent
runners read it from the model's first native tool-call turn; the other
topologies parse it from a fenced JSON block (:mod:`core.bfcl_calls`).

A runner's batch writes one line per instance to ``predictions.jsonl``
(``{id, category, model_output, model_name_or_path}``) and its summary to
``results.jsonl`` (both emptied when the batch starts), and a text trace to
``traces/<id>.txt``.
"""

from __future__ import annotations

import json
import keyword
import logging
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from core import batch, cli, voting
from core.bfcl_calls import extract_canonical  # noqa: F401  (the fenced-JSON parser of the runners)

logger = logging.getLogger(__name__)

DATASET = "bfcl"
HF_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
AST_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple")
DEFAULT_CATEGORY = "simple"
DEFAULT_LIMIT = 5  # instances the command line evaluates unless --limit or --only says otherwise
SOURCE = f"BFCL v3 from {HF_DATASET}"
EPILOG = "examples:\n  %(prog)s --category multiple --limit 20\n  %(prog)s --only simple_0 simple_1"

# Appended to the centralized manager prompts.
TERMINATE_NUDGE = (
    "\n\nWhen you emit the final fenced ```json``` block containing "
    "the canonical call list, immediately follow it with the literal "
    "string TERMINATE on its own line so the group-chat knows to stop."
)


# Data
def load_instances(
    category: str,
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Rows of one subset and their possible answers (``{id, ground_truth}``), aligned by id.

    ``only`` selects ids before ``offset`` and ``limit`` apply; a row without a
    possible answer raises.
    """
    from huggingface_hub import hf_hub_download

    def jsonl(filename: str) -> list[dict]:
        path = Path(hf_hub_download(HF_DATASET, filename, repo_type="dataset"))
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    rows = jsonl(f"BFCL_v3_{category}.json")
    gt_by_id = {gt["id"]: gt for gt in jsonl(f"possible_answer/BFCL_v3_{category}.json")}
    if only:
        wanted = set(only)
        rows = [row for row in rows if row["id"] in wanted]
    rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    missing = [row["id"] for row in rows if row["id"] not in gt_by_id]
    if missing:
        raise RuntimeError(f"no ground truth for ids: {missing[:5]}")
    return rows, [gt_by_id[row["id"]] for row in rows]


def add_arguments(parser) -> None:
    """``--category``: the BFCL subset."""
    parser.add_argument(
        "--category", default=DEFAULT_CATEGORY, choices=list(AST_CATEGORIES), help="BFCL subset (default: simple)."
    )


def load_pairs(
    category: str = DEFAULT_CATEGORY,
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
) -> list[tuple[dict, dict]]:
    """``(row, possible answer)`` pairs: the command line's instances."""
    rows, gts = load_instances(category, limit, offset, only)
    return list(zip(rows, gts))


# Prompt formatting
def flatten_question(question: list) -> str:
    """The turns of a single-turn ``question`` (``[[messages]]``) as ``[role] content`` lines."""
    if not question:
        return ""
    turns = question[0] if isinstance(question[0], list) else question
    parts = []
    for msg in turns:
        if isinstance(msg, dict):
            parts.append(f"[{msg.get('role', 'user')}] {msg.get('content', '')}")
        else:
            parts.append(str(msg))
    return "\n".join(parts)


def render_schemas(functions: list[dict]) -> str:
    """The function schemas as indented JSON."""
    return json.dumps(functions, indent=2)


def stage_inputs(instance: dict) -> dict:
    """The fields of the sequential stage templates (``user_request``, ``schemas_text``).

    The values are not escaped: ``str.format`` (LangGraph) and CrewAI's input
    interpolation insert them as they are, so the model reads single braces.
    """
    return {
        "user_request": flatten_question(instance["question"]),
        "schemas_text": render_schemas(instance["function"]),
    }


def format_task(user_request: str, schemas_text: str) -> str:
    """The centralized team's task message: the request, the schemas and the canonical output form."""
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
        'use the shape {"fn_name": "<name>", "args": {...}}. '
        "Example for a schema named calculate_area: "
        '[{"calculate_area": {"width": 5, "height": 3}}]. '
        "For parallel calls, emit multiple such dicts in the list."
    )


def format_debate_task(user_request: str, schemas_text: str) -> str:
    """The LangGraph debate peers' task message."""
    return (
        "USER REQUEST:\n"
        f"{user_request}\n\n"
        "SCHEMAS:\n"
        f"{schemas_text}\n\n"
        "Emit your final canonical call list as a SINGLE fenced ```json``` "
        "block. Canonical form is a list of dicts with one key per dict: "
        '[{"fn_name": {"arg": value, ...}}, ...]. '
        "For a single call, emit a one-element list. For parallel calls, "
        "emit a multi-element list."
    )


def agents_input(instance: dict) -> str:
    """Task text given to every Agents SDK peer: the raw request turns and the schemas."""
    return (
        "USER REQUEST:\n"
        + json.dumps(instance["question"], ensure_ascii=False)
        + "\n\nFUNCTION SCHEMAS:\n"
        + json.dumps(instance["function"], ensure_ascii=False, indent=2)
        + "\n\nReturn the selected function call in canonical JSON."
    )


# Function schemas as LangChain tools (single and independent runners)
_SCHEMA_TYPES = {
    "integer": int,
    "string": str,
    "float": float,
    "number": float,
    "boolean": bool,
    "dict": dict,
    "any": Any,
}


def _py_type(prop: dict) -> Any:
    """Python annotation of a BFCL JSON-schema property (BFCL adds the "dict", "tuple" and "any" types)."""
    kind = (prop or {}).get("type", "any")
    if kind in ("array", "tuple"):
        items = prop.get("items") or {}
        return list[_py_type(items) if items else Any]
    return _SCHEMA_TYPES.get(kind, Any)


def _field_name(name: str) -> str:
    """A pydantic-safe field name for a parameter: no leading underscore, no keyword.

    The schema's name becomes the field's alias, but the tool schema the model
    sees lists the safe name (:func:`parameter_names` maps it back).
    """
    safe = name.lstrip("_") or "field"
    return safe + "_" if keyword.iskeyword(safe) else safe


def parameter_names(schema: dict) -> dict[str, str]:
    """The schema's parameter names, keyed by the names its :func:`schema_to_tool` tool shows the model."""
    properties = (schema.get("parameters") or {}).get("properties") or {}
    return {_field_name(name): name for name in properties}


def schema_to_tool(schema: dict):
    """A LangChain tool for one BFCL function schema; calling it returns "" (only the call is scored)."""
    from langchain_core.tools import StructuredTool
    from pydantic import Field, create_model

    params = schema.get("parameters") or {}
    required = set(params.get("required") or [])
    fields: dict[str, Any] = {}
    for name, prop in (params.get("properties") or {}).items():
        py_type = _py_type(prop)
        description = (prop or {}).get("description", "")
        safe = _field_name(name)
        alias = {"alias": name} if safe != name else {}
        if name in required:
            fields[safe] = (py_type, Field(..., description=description, **alias))
        else:
            fields[safe] = (py_type | None, Field(None, description=description, **alias))
    args_model = create_model(re.sub(r"\W+", "_", schema["name"]) + "Args", **fields) if fields else None
    return StructuredTool.from_function(
        func=lambda **_: "",
        name=schema["name"],
        description=schema.get("description", ""),
        args_schema=args_model,
    )


# Calls and scoring
def extract_first_tool_calls(messages: list) -> list[dict]:
    """The tool calls of the earliest message that has any: BFCL scores the first commitment."""
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            return tool_calls
    return []


def to_canonical(tool_calls: list[dict], function_schemas: Sequence[dict] = ()) -> list[dict]:
    """LangChain tool calls ``[{"name", "args", "id"}]`` as canonical calls ``[{name: args}]``.

    The arguments of a call to one of ``function_schemas`` (the functions the
    :func:`schema_to_tool` tools were built from) get back their schema names,
    e.g. ``from_`` -> ``_from``: the AST checker expects the schema's names.
    """
    names = {schema["name"]: parameter_names(schema) for schema in function_schemas}
    canonical = []
    for tc in tool_calls:
        original = names.get(tc["name"], {})
        canonical.append({tc["name"]: {original.get(arg, arg): value for arg, value in (tc.get("args") or {}).items()}})
    return canonical


def register_model(model_id: str) -> None:
    """Make ``model_id`` known to bfcl-eval as a model that keeps dotted function names.

    The AST checker looks the model up to decide whether to rewrite '.' to '_'
    in function names (for models that cannot emit dots) and fails on an
    unknown model. An unknown model is registered as a clone of ``qwen3-8b``;
    scoring reads only its ``underscore_to_dot``.
    """
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

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


def score_one(
    function_schemas: list[dict],
    model_output: list[dict],
    ground_truth: list[dict],
    category: str,
    model_id: str,
) -> dict:
    """bfcl-eval's AST checker report (``valid``, ``error``, ``error_type``) for one prediction.

    The checker branches on ``category``: parallel subsets match calls in any
    order, multiple subsets pick one function, simple matches a single call.
    """
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    return ast_checker(function_schemas, model_output, ground_truth, Language.PYTHON, category, model_id)


# Voting over several agents' calls
def canonical_key(model_output: list[dict]) -> str:
    """Vote key of a call list: arguments sorted within each call, calls sorted (parallel subsets ignore order)."""
    calls = [{fn: dict(sorted((args or {}).items())) for fn, args in call.items()} for call in model_output]
    calls.sort(key=lambda call: json.dumps(call, sort_keys=True))
    return json.dumps(calls, sort_keys=True)


def select_call(calls: list[list[dict] | None]) -> int:
    """Index of the submitted call list: the vote over the canonical keys (:mod:`core.voting`; no call abstains)."""
    return voting.majority([canonical_key(call) if call else None for call in calls])


def majority_vote(answers: list[dict]) -> dict:
    """The submitted replica record: :func:`select_call` over the answers' ``model_output``, in ``agent_id`` order."""
    ranked = sorted(answers, key=lambda answer: answer["agent_id"])
    return ranked[select_call([answer.get("model_output") for answer in ranked])]


def vote_counts(answers: list[dict]) -> list[tuple[str, int]]:
    """``(canonical key, count)`` over the answers that have a call, most votes first."""
    counts = Counter(canonical_key(a["model_output"]) for a in answers if a.get("model_output"))
    return sorted(counts.items(), key=lambda kv: -kv[1])


# Per-instance summaries and traces
def solve_failed(summary: dict, exc: Exception) -> dict:
    """Record a failed solve on the instance summary."""
    summary["error"] = f"{type(exc).__name__}: {exc}"
    summary["stage"] = "solve"
    return summary


def add_verdict(
    summary: dict, score_one: Callable[..., dict], instance: dict, ground_truth: dict, category: str
) -> dict:
    """Score ``summary["model_output"]`` with a runner's ``score_one`` and record the verdict.

    Sets ``valid``, ``error_type`` and, for an invalid call, the first three
    checker errors as ``score_error``; a checker failure sets ``valid`` False,
    ``error`` and ``stage`` "score".
    """
    try:
        report = score_one(instance["function"], summary["model_output"], ground_truth["ground_truth"], category)
    except Exception as e:
        summary["valid"] = False
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["stage"] = "score"
        return summary
    summary["valid"] = bool(report.get("valid"))
    summary["error_type"] = report.get("error_type")
    if not summary["valid"]:
        summary["score_error"] = (report.get("error") or [])[:3]
    return summary


def sections_trace(sections: Iterable[tuple[str, str]]) -> str:
    """Trace text of ``(title, text)`` sections, each as ``=== title ===`` and the text."""
    return "".join(f"=== {title} ===\n{text}\n\n" for title, text in sections)


def messages_trace(messages: list[dict]) -> str:
    """Trace text of a group chat's ``{source, content}`` messages."""
    return sections_trace((str(m.get("source", "?")).upper(), m.get("content", "")) for m in messages)


def peer_trace(out: dict) -> str:
    """Trace text of a debate: the winner, then each peer's call and, when scored, its verdict."""
    text = f"winner: peer {out.get('winner')}\n\n"
    for peer in out.get("per_peer") or []:
        call = peer.get("call") or []
        text += f"=== peer {peer.get('peer')} ===\n{json.dumps(call, sort_keys=True) if call else '(no call)'}\n"
        if peer.get("valid") is not None:
            text += f"  valid={peer['valid']}  error_type={peer.get('error_type')}\n"
        text += "\n"
    return text


def write_trace(out_dir: Path, instance_id: str, text: str) -> None:
    """Write an instance's trace to ``out_dir/traces/<id>.txt``."""
    path = out_dir / "traces" / f"{instance_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# Batches
RunOne = Callable[[dict, dict, str, Path], dict]


def run_batch(
    run_one: RunOne,
    rows: list[dict],
    gts: list[dict],
    category: str,
    out_dir: Path,
    *,
    model_id: str,
    predictions: Path | None = None,
    verbose: bool = True,
    hidden: tuple[str, ...] = (),
) -> dict:
    """Score every row with ``run_one(row, possible answer, category, out_dir)``.

    Writes each prediction to ``predictions`` (default ``out_dir/predictions.jsonl``)
    and each summary to ``out_dir/results.jsonl``, both emptied first; returns
    ``{n, valid, valid_rate, category}``. Verbose progress omits the summary keys in
    ``hidden``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = Path(predictions or out_dir / "predictions.jsonl")
    results_path = out_dir / "results.jsonl"

    def prediction(pair: tuple[dict, dict], summary: dict) -> dict:
        return {
            "id": pair[0]["id"],
            "category": category,
            "model_output": summary.get("model_output"),
            "model_name_or_path": model_id,
        }

    def summarize(summaries: list[dict]) -> dict:
        valid = sum(1 for summary in summaries if summary.get("valid"))
        n = len(summaries)
        return {"n": n, "valid": valid, "valid_rate": (valid / n) if n else 0.0, "category": category}

    def progress(index: int, total: int, summary: dict, done: list[dict]) -> str:
        return f"  -> {json.dumps({k: v for k, v in summary.items() if k not in hidden})}"

    def banner(result: dict) -> str:
        return (
            f"\ndone: valid {result['valid']}/{result['n']}\n"
            f"      predictions -> {preds_path}\n      results     -> {results_path}"
        )

    return batch.run_batch(
        list(zip(rows, gts)),
        lambda index, pair: run_one(pair[0], pair[1], category, out_dir),
        summarize=summarize,
        out_path=results_path,
        outputs=[batch.Output(preds_path, prediction)],
        verbose=verbose,
        header=lambda index, total, pair: f"\n[{index + 1}/{total}] {pair[0]['id']}  ({category})",
        progress=progress,
        banner=banner,
        raw_summary=True,
    )


def evaluate(
    run_one: RunOne,
    category: str,
    limit: int | None,
    offset: int,
    only: list[str] | None,
    out_dir: Path,
    *,
    model_id: str,
    verbose: bool = True,
    note: str = "",
    hidden: tuple[str, ...] = (),
) -> dict:
    """Load a slice of one subset and :func:`run_batch` it into ``out_dir`` (a runner's ``run_batch``).

    ``note`` follows the "loaded ..." line (e.g. the team size).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, gts = load_instances(category, limit, offset, only)
    if verbose:
        logger.info("loaded %d instance(s) from %s / %s%s", len(rows), HF_DATASET, category, note)
    return run_batch(run_one, rows, gts, category, out_dir, model_id=model_id, verbose=verbose, hidden=hidden)


def cli_main(
    argv: list[str] | None,
    *,
    description: str,
    run_one: RunOne,
    model_id: str,
    default_out_dir: Path,
    hidden: tuple[str, ...] = (),
    preflight: Callable[[], None] | None = None,
) -> int:
    """A runner's command line (:func:`core.cli.main`); outputs default to ``default_out_dir``."""

    def run_pairs(pairs: list[tuple[dict, dict]], category: str, out_path: Path, out_dir: Path | None = None) -> dict:
        rows = [row for row, _ in pairs]
        gts = [gt for _, gt in pairs]
        out_dir = Path(out_dir or default_out_dir)
        return run_batch(run_one, rows, gts, category, out_dir, model_id=model_id, predictions=out_path, hidden=hidden)

    return cli.main(
        argv,
        description=description,
        load_instances=load_pairs,
        run_batch=run_pairs,
        source=SOURCE,
        predictions=default_out_dir / "predictions.jsonl",
        add_arguments=add_arguments,
        default_limit=DEFAULT_LIMIT,
        epilog=EPILOG,
        preflight=preflight,
    )
