"""Local BFCL loader/scorer copy for the real-runner pilot."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import dspy

from real_runner_gepa.datasets.split_utils import train_val_split_excluding_real_eval


HF_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
DEFAULT_CATEGORY = "simple"
TASK_MODEL = os.environ.get("TASK_MODEL", "Qwen/Qwen3.5-9B")


def _flatten_messages(question_field) -> str:
    if not question_field:
        return ""
    if isinstance(question_field, str):
        return question_field
    parts = []
    for thread in question_field:
        if isinstance(thread, list):
            for msg in thread:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    if role in ("user", "human"):
                        parts.append(str(msg.get("content", "")))
                else:
                    parts.append(str(msg))
        else:
            parts.append(str(thread))
    return "\n".join(p for p in parts if p)


def _format_tools_description(schemas: list[dict]) -> str:
    if not schemas:
        return "(no tools available)"
    blocks = []
    for schema in schemas:
        name = schema.get("name", "?")
        desc = schema.get("description", "")
        params = schema.get("parameters") or {}
        properties = params.get("properties") or {}
        required = set(params.get("required") or [])
        param_lines = []
        for pname, prop in properties.items():
            typ = (prop or {}).get("type", "any")
            pdesc = (prop or {}).get("description", "")
            req = "*" if pname in required else ""
            param_lines.append(f"    - {pname}{req}: {typ}  {pdesc}")
        params_block = "\n".join(param_lines) if param_lines else "    (no parameters)"
        blocks.append(f"- {name}: {desc}\n  parameters (* = required):\n{params_block}")
    return "\n".join(blocks)


def _bfcl_question_to_nested_messages(question_field) -> list:
    if isinstance(question_field, list):
        return question_field
    return [[{"role": "user", "content": str(question_field)}]]


def load_all(category: str | None = None) -> list[dspy.Example]:
    """Load one BFCL category as examples carrying real-runner instance data."""
    from huggingface_hub import hf_hub_download

    category = category or os.environ.get("BFCL_CATEGORY", DEFAULT_CATEGORY)
    main_path = hf_hub_download(
        HF_DATASET, f"BFCL_v3_{category}.json", repo_type="dataset"
    )
    ans_path = hf_hub_download(
        HF_DATASET,
        f"possible_answer/BFCL_v3_{category}.json",
        repo_type="dataset",
    )

    rows = [
        json.loads(line)
        for line in Path(main_path).read_text().splitlines()
        if line.strip()
    ]
    gt_by_id = {
        json.loads(line)["id"]: json.loads(line)
        for line in Path(ans_path).read_text().splitlines()
        if line.strip()
    }

    examples: list[dspy.Example] = []
    for row in rows:
        rid = row.get("id")
        if rid is None or rid not in gt_by_id:
            continue
        functions = row.get("function") or []
        question_text = _flatten_messages(row.get("question"))
        if not question_text or not functions:
            continue
        gt = gt_by_id[rid]
        instance = {
            "id": str(rid),
            "question": _bfcl_question_to_nested_messages(row.get("question")),
            "function": functions,
            "ground_truth": gt.get("ground_truth") or [],
            "category": category,
        }
        examples.append(
            dspy.Example(
                id=str(rid),
                question=question_text,
                tools_description=_format_tools_description(functions),
                function=functions,
                ground_truth=gt.get("ground_truth") or [],
                category=category,
                task_instance=instance,
                answer=json.dumps(gt.get("ground_truth") or []),
            ).with_inputs("task_instance")
        )
    return examples


def train_val_split(
    examples: list[dspy.Example],
    train_size: int,
    val_size: int,
    seed: int = 0,
    offset: int = 0,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    return train_val_split_excluding_real_eval("bfcl", examples, train_size, val_size, seed, offset)


def score_model_output(
    function_schemas: list[dict],
    model_calls: list[dict] | None,
    ground_truth: list[dict],
    category: str,
    model_id: str,
) -> dict:
    if not model_calls:
        return {"ok": False, "detail": "model emitted no parseable tool calls"}
    try:
        from bfcl_eval.constants.enums import Language
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    except Exception as exc:
        return _fallback_score(model_calls, ground_truth, f"bfcl_eval import: {exc}")

    if model_id not in MODEL_CONFIG_MAPPING:
        template = MODEL_CONFIG_MAPPING.get("qwen3-8b") or next(
            iter(MODEL_CONFIG_MAPPING.values())
        )
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

    try:
        result = ast_checker(
            function_schemas,
            model_calls,
            ground_truth,
            Language.PYTHON,
            category,
            model_id,
        )
    except Exception as exc:
        return _fallback_score(model_calls, ground_truth, f"ast_checker: {exc}")

    if isinstance(result, dict):
        ok = bool(result.get("valid", False))
        detail = result.get("error", "") or json.dumps(result)[:300]
    else:
        ok = bool(result)
        detail = str(result)[:300]
    return {"ok": ok, "detail": detail or ("ok" if ok else "invalid")}


def _function_names(calls: list[dict]) -> list[str]:
    names: list[str] = []
    for call in calls or []:
        if isinstance(call, dict):
            names.extend(str(name) for name in call.keys())
    return names


def _schema_summary(schemas: list[dict]) -> str:
    blocks: list[str] = []
    for schema in schemas or []:
        props = ((schema.get("parameters") or {}).get("properties") or {})
        required = set((schema.get("parameters") or {}).get("required") or [])
        fields = []
        for name, prop in props.items():
            req = "required" if name in required else "optional"
            typ = (prop or {}).get("type", "any")
            fields.append(f"{name}:{typ}:{req}")
        blocks.append(f"{schema.get('name', '?')}({', '.join(fields)})")
    return "; ".join(blocks)


def _trace_agent_text(pred_trace) -> str:
    if not pred_trace:
        return ""
    parts: list[str] = []
    for _, _, outputs in pred_trace:
        role_trace = getattr(outputs, "agent_trace", None)
        if role_trace:
            parts.append(str(role_trace))
        elif outputs is not None:
            parts.append(str(outputs))
    return "\n".join(parts)


def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """DSPy-compatible metric used by the generic all-dataset GEPA runner."""
    calls = getattr(prediction, "tool_calls", None) or []
    schemas = getattr(example, "function", []) or []
    gold_calls = getattr(example, "ground_truth", []) or []
    category = getattr(example, "category", "simple") or "simple"
    result = score_model_output(schemas, calls, gold_calls, category, TASK_MODEL)
    score = 1.0 if result["ok"] else 0.0
    role = pred_name or "program"
    agent_trace = _trace_agent_text(pred_trace) or getattr(prediction, "agent_trace", "")
    if score:
        feedback = (
            f"Correct for role {role}. {result['detail'][:300]} "
            "Keep preserving exact function names, required arguments, and schema types."
        )
    else:
        failure_kind = "Format failure" if not calls else "Reasoning/tool-call failure"
        feedback = (
            f"{failure_kind} for role {role}. {result['detail'][:500]}\n"
            f"Gold functions: {_function_names(gold_calls)}. "
            f"Emitted functions: {_function_names(calls)}.\n"
            f"Gold calls: {json.dumps(gold_calls, default=str)[:900]}\n"
            f"Emitted calls: {json.dumps(calls, default=str)[:900]}\n"
            f"Available schema summary: {_schema_summary(schemas)[:1200]}\n"
            f"Real-runner trace:\n{agent_trace[:1600]}\n"
            "Actionable fix: choose only listed tools, include every required "
            "argument, preserve exact parameter names, and convert values to "
            "the schema type before calling the tool."
        )
    return dspy.Prediction(score=score, feedback=feedback)


def _fallback_score(model_calls: list[dict], ground_truth: list[dict], reason: str) -> dict:
    model_names = set()
    for call in model_calls or []:
        if isinstance(call, dict):
            model_names.update(call.keys())
    gold_names = set()
    for call in ground_truth or []:
        if isinstance(call, dict):
            gold_names.update(call.keys())
    overlap = model_names & gold_names
    return {
        "ok": bool(overlap),
        "detail": (
            f"fallback ({reason}); model={sorted(model_names)} "
            f"gold={sorted(gold_names)} overlap={sorted(overlap)}"
        ),
    }
