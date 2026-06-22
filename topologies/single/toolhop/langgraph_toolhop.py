"""Shared ToolHop loader, solver, scorer, and CLI helpers.

ToolHop rows include OpenAI-style tool schemas and Python source for local
tools. Real ToolHop solving requires executing those dataset-provided
functions, so execution is gated by `TOOLHOP_ALLOW_DATASET_EXEC=1`.
"""

from __future__ import annotations

import argparse
import ast
import builtins as _builtins
import json
import math
import os
import re
import statistics
import time
from collections import Counter
from datetime import date as _date
from datetime import datetime as _datetime_class
from datetime import timedelta as _timedelta
import datetime as _datetime_module
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from topologies.output_contracts import append_output_contract
from topologies.telemetry import normalize, openai_sdk_accumulate
from communications.communication_formats import normalize_report, render_report


HF_DATASET = "bytedance-research/ToolHop"
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
DEFAULT_MAX_TURNS = int(os.environ.get("TOOLHOP_MAX_TURNS", "9"))
_TOOL_RESULT_CHAR_BUDGET = int(os.environ.get("TOOLHOP_TOOL_RESULT_CHAR_BUDGET", "6000"))
INDEPENDENT_N_AGENTS = int(os.environ.get("TOOLHOP_INDEPENDENT_N_AGENTS", os.environ.get("INDEPENDENT_N_AGENTS", "4")))
DECENTRALIZED_N_AGENTS = int(os.environ.get("TOOLHOP_DECENTRALIZED_N_AGENTS", os.environ.get("DECENTRALIZED_N_AGENTS", "4")))
DECENTRALIZED_N_ROUNDS = int(os.environ.get("TOOLHOP_DECENTRALIZED_N_ROUNDS", os.environ.get("DECENTRALIZED_N_ROUNDS", "2")))

SEQUENTIAL_ROLES = ("planner", "caller", "checker", "verifier")
CENTRALIZED_WORKER_ROLES = ("planner_worker", "caller_worker", "validator_worker")
FINAL_ANSWER_ROLES = {"solver", "verifier", "manager", "debater"}

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_PROMPTS_ROOT = _REPO_ROOT / "configs" / "prompts"

try:
    import pytz as _pytz
except Exception:  # pragma: no cover - optional benchmark helper dependency
    _pytz = None

try:
    from dateutil.relativedelta import relativedelta as _relativedelta
except Exception:  # pragma: no cover - optional benchmark helper dependency
    _relativedelta = None

try:
    from babel.dates import format_date as _format_date
except Exception:  # pragma: no cover - optional benchmark helper dependency
    _format_date = None


class _DatetimeCompat:
    """Compatibility object for snippets using either datetime import style."""

    _CLASS_ATTRS = {
        "combine",
        "fromisoformat",
        "fromtimestamp",
        "now",
        "strptime",
        "today",
        "utcfromtimestamp",
        "utcnow",
    }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _datetime_class(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in self._CLASS_ATTRS and hasattr(_datetime_class, name):
            return getattr(_datetime_class, name)
        if hasattr(_datetime_module, name):
            return getattr(_datetime_module, name)
        return getattr(_datetime_class, name)


_SAFE_IMPORT_ROOTS = {
    "babel",
    "base64",
    "binascii",
    "calendar",
    "cmath",
    "collections",
    "csv",
    "dateutil",
    "datetime",
    "dicttoxml",
    "fractions",
    "functools",
    "holidays",
    "io",
    "itertools",
    "json",
    "locale",
    "math",
    "numbers",
    "numpy",
    "pytz",
    "re",
    "roman",
    "statistics",
    "string",
    "sympy",
    "time",
    "_strptime",
    "unicodedata",
    "urllib",
    "xml",
}


def _safe_import(
    name: str,
    globals: dict | None = None,
    locals: dict | None = None,
    fromlist: tuple | list = (),
    level: int = 0,
) -> Any:
    if level:
        raise ImportError("relative imports are not allowed in ToolHop snippets")
    root = str(name).split(".", 1)[0]
    if root not in _SAFE_IMPORT_ROOTS:
        raise ImportError(f"import of {name!r} is not allowed in ToolHop snippets")
    return _builtins.__import__(name, globals, locals, fromlist, level)


_SAFE_BUILTINS = {
    "__build_class__": _builtins.__build_class__,
    "__import__": _safe_import,
    "abs": abs,
    "all": all,
    "any": any,
    "ascii": ascii,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "Exception": Exception,
    "filter": filter,
    "float": float,
    "format": format,
    "hex": hex,
    "IndexError": IndexError,
    "int": int,
    "isinstance": isinstance,
    "iter": iter,
    "KeyError": KeyError,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "NotImplementedError": NotImplementedError,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "RuntimeError": RuntimeError,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "TimeoutError": TimeoutError,
    "TypeError": TypeError,
    "tuple": tuple,
    "type": type,
    "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
    "zip": zip,
}


def _dataset_path() -> Path:
    return Path(
        hf_hub_download(
            repo_id=HF_DATASET,
            repo_type="dataset",
            filename="data/ToolHop.json",
        )
    )


def load_instances(
    limit: int | None = None,
    offset: int = 0,
    only: list[str | int] | None = None,
) -> list[dict]:
    """Load ToolHop rows. `only` accepts integer ids or their string form."""
    with _dataset_path().open(encoding="utf-8") as f:
        rows = json.load(f)
    if only:
        wanted = {str(item) for item in only}
        rows = [row for row in rows if str(row["id"]) in wanted]
    rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _literal_type(value: str) -> str:
    try:
        return type(ast.literal_eval(value.strip())).__name__
    except Exception:
        return "raw_string"


def _stats(values: list[int]) -> dict:
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 3),
        "median": statistics.median(values),
    }


def dataset_summary(limit: int | None = None) -> dict:
    """Safe structural smoke test. Does not execute dataset tool code."""
    rows = load_instances(limit=limit)
    required = {"id", "question", "answer", "tools", "functions"}
    missing = Counter()
    ids = []
    tool_counts = []
    function_counts = []
    source_errors = []
    schema_errors = []
    name_mismatches = []
    answer_types = Counter()

    for sample in rows:
        sid = sample.get("id")
        ids.append(sid)
        missing.update(required - set(sample))
        tools = sample.get("tools") or {}
        functions = sample.get("functions") or []
        tool_counts.append(len(tools))
        function_counts.append(len(functions))

        function_names = []
        for index, source in enumerate(functions):
            try:
                tree = ast.parse(source)
                function_names.extend(
                    node.name for node in tree.body if isinstance(node, ast.FunctionDef)
                )
                compile(source, f"toolhop_{sid}_{index}.py", "exec")
            except Exception as exc:
                source_errors.append((sid, index, type(exc).__name__, str(exc)[:160]))

        schema_names = []
        for key, schema in tools.items():
            if not isinstance(schema, dict):
                schema_errors.append((sid, key, "schema_not_dict"))
                continue
            schema_names.append(schema.get("name"))
            params = schema.get("parameters")
            if (
                not isinstance(params, dict)
                or params.get("type") != "object"
                or "properties" not in params
            ):
                schema_errors.append((sid, schema.get("name"), "bad_parameters"))

        if set(schema_names) != set(function_names):
            name_mismatches.append(
                (
                    sid,
                    sorted(set(schema_names) - set(function_names))[:5],
                    sorted(set(function_names) - set(schema_names))[:5],
                )
            )
        answer_types[_literal_type(str(sample.get("answer", "")))] += 1

    return {
        "num_samples": len(rows),
        "unique_ids": len(set(ids)),
        "duplicates": len(ids) - len(set(ids)),
        "missing_fields": dict(missing),
        "tools_per_sample": _stats(tool_counts),
        "functions_per_sample": _stats(function_counts),
        "answer_literal_types": dict(answer_types),
        "source_parse_or_compile_errors": len(source_errors),
        "source_error_examples": source_errors[:5],
        "schema_errors": len(schema_errors),
        "schema_error_examples": schema_errors[:5],
        "tool_function_name_mismatches": len(name_mismatches),
        "name_mismatch_examples": name_mismatches[:5],
    }


_SAFE_EXCEPTION_BASES = {"Exception", "ValueError", "RuntimeError"}


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _is_safe_exception_class(node: ast.ClassDef) -> bool:
    if node.decorator_list or node.keywords:
        return False
    if not node.bases:
        return False
    for base in node.bases:
        if not isinstance(base, ast.Name) or base.id not in _SAFE_EXCEPTION_BASES:
            return False
    return all(isinstance(stmt, ast.Pass) or _is_docstring(stmt) for stmt in node.body)


def _function_only_module(source: str, sid: Any, index: int) -> ast.Module:
    """Return a module containing only docstrings and function definitions.

    Some ToolHop function snippets include top-level "example usage" calls
    after the actual function. Those are not part of the tool surface and must
    not run during loading.
    """
    tree = ast.parse(source)
    body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            body.append(node)
            continue
        if isinstance(node, ast.ClassDef) and _is_safe_exception_class(node):
            body.append(node)
            continue
        if _is_docstring(node):
            body.append(node)
            continue
        # Drop imports, example calls, assignments, comments parsed as no-ops,
        # etc. Imports used by ToolHop snippets are provided explicitly in the
        # restricted namespace below.
    if not any(isinstance(node, ast.FunctionDef) for node in body):
        raise RuntimeError(f"ToolHop function source {sid}/{index} defines no function")
    return ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))


def _tool_namespace(sample: dict) -> dict[str, Any]:
    return {
        "__builtins__": _SAFE_BUILTINS,
        "__name__": f"toolhop_{sample.get('id', 'sample')}",
        "date": _date,
        "datetime": _DatetimeCompat(),
        "format_date": _format_date,
        "math": math,
        "pytz": _pytz,
        "re": re,
        "relativedelta": _relativedelta,
        "timedelta": _timedelta,
    }


def _tool_schema_entries(sample: dict) -> list[dict[str, Any]]:
    schemas = [
        (index, schema)
        for index, schema in enumerate((sample.get("tools") or {}).values())
        if isinstance(schema, dict) and schema.get("name")
    ]
    name_counts = Counter(str(schema.get("name")) for _, schema in schemas)
    seen: Counter[str] = Counter()
    entries: list[dict[str, Any]] = []
    for index, schema in schemas:
        name = str(schema.get("name"))
        occurrence = seen[name]
        seen[name] += 1
        runtime_name = name
        if name_counts[name] > 1:
            suffix = f"__{occurrence}"
            runtime_name = name[: 64 - len(suffix)] + suffix
        entries.append(
            {
                "index": index,
                "schema": schema,
                "name": name,
                "runtime_name": runtime_name,
            }
        )
    return entries


def _exec_function_source(sample: dict, source: str, index: int) -> dict[str, Any]:
    namespace = _tool_namespace(sample)
    module = _function_only_module(source, sample.get("id"), index)
    exec(compile(module, f"toolhop_{sample.get('id')}_{index}.py", "exec"), namespace)
    return namespace


def _shared_function_namespace(sample: dict) -> dict[str, Any]:
    namespace = _tool_namespace(sample)
    for index, source in enumerate(sample.get("functions") or []):
        module = _function_only_module(source, sample.get("id"), index)
        exec(compile(module, f"toolhop_{sample.get('id')}_{index}.py", "exec"), namespace)
    return namespace


def _function_map(sample: dict) -> dict[str, Any]:
    if os.environ.get("TOOLHOP_ALLOW_DATASET_EXEC") != "1":
        raise RuntimeError(
            "ToolHop solving requires executing dataset-provided Python tool "
            "functions. Set TOOLHOP_ALLOW_DATASET_EXEC=1 only when you intend "
            "to run the benchmark."
        )

    sources = list(sample.get("functions") or [])
    functions: dict[str, Any] = {}
    shared_namespace: dict[str, Any] | None = None
    for entry in _tool_schema_entries(sample):
        fn = None
        index = int(entry["index"])
        name = str(entry["name"])
        if index < len(sources):
            namespace = _exec_function_source(sample, sources[index], index)
            fn = namespace.get(name)
        if not callable(fn):
            if shared_namespace is None:
                shared_namespace = _shared_function_namespace(sample)
            fn = shared_namespace.get(name)
        if callable(fn):
            functions[str(entry["runtime_name"])] = fn
    return functions


def _tools_payload(sample: dict) -> list[dict]:
    payload = []
    for entry in _tool_schema_entries(sample):
        schema = dict(entry["schema"])
        schema["name"] = entry["runtime_name"]
        payload.append({"type": "function", "function": schema})
    return payload


def _system_prompt(topology: str, role: str, style: str, prompt_suffix: str = "") -> str:
    prompt_path = _PROMPTS_ROOT / topology / "toolhop" / f"{role}.txt"
    if prompt_path.exists():
        prompt = prompt_path.read_text().strip()
    else:
        prompt = (
            "You are solving ToolHop, a multi-hop tool-use benchmark. Use the "
            "provided tools to answer the user's question. Keep intermediate "
            "reasoning concise. Use tool outputs to continue the chain. The final "
            "response must be short."
            f"\n\nImplementation style: {style}."
        )
    if prompt_suffix:
        prompt = prompt.rstrip() + "\n" + prompt_suffix
    return append_output_contract(prompt, "toolhop", topology, role)


def _user_prompt(sample: dict) -> str:
    return (
        "Answer the question using the available tools. If the final answer is "
        "a date, use YYYY-MM-DD. If it is a name, use Firstname Lastname. If it "
        "contains a number, output the number, not a word, with no leading "
        "zeroes.\n\nQuestion: "
        + str(sample["question"])
    )


def _completion_kwargs(seed: int | None = None) -> dict:
    return {
        "model": MODEL_ID,
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": 0 if seed is None else int(seed),
        "max_tokens": int(os.environ.get("TOOLHOP_MAX_TOKENS", "512")),
        "extra_body": {
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def _client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "ToolHop model solving requires the openai Python package. Run it "
            "inside the mas-promptbench conda environment or install openai."
        ) from exc

    return OpenAI(
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
        timeout=300.0,
        max_retries=5,
    )


def _message_dict(message, tool_call_payloads: list[dict] | None = None) -> dict:
    payload = message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else dict(message)
    message_dict = {
        key: payload[key]
        for key in ("role", "content")
        if key in payload and payload[key] is not None
    }
    if tool_call_payloads is not None:
        message_dict["tool_calls"] = [
            _sanitize_tool_call_for_message(tool_call)
            for tool_call in tool_call_payloads
        ]
    elif payload.get("tool_calls"):
        message_dict["tool_calls"] = [
            _sanitize_tool_call_for_message(_tool_call_dict(tool_call, index))
            for index, tool_call in enumerate(payload["tool_calls"])
        ]
    return message_dict


def _tool_call_dict(tool_call, index: int = 0) -> dict:
    payload = tool_call.model_dump(exclude_none=True) if hasattr(tool_call, "model_dump") else dict(tool_call)
    normalized = dict(payload)
    normalized["id"] = str(normalized.get("id") or f"tool_call_{index}")
    normalized["type"] = normalized.get("type") or "function"
    normalized["function"] = dict(normalized.get("function") or {})
    return normalized


def _sanitize_tool_call_for_message(tool_call: dict) -> dict:
    """Return a chat-history-safe tool call payload.

    vLLM validates prior assistant tool calls on the next request. Small
    models sometimes emit malformed JSON in ``function.arguments``; if that
    raw string is stored in ``messages``, the next request fails before the
    agent can recover. Keep the original payload for execution, but store
    valid JSON in chat history.
    """
    sanitized = dict(tool_call)
    function = dict(sanitized.get("function") or {})
    function["arguments"] = _valid_json_arguments(function.get("arguments"))
    sanitized["function"] = function
    return sanitized


def _valid_json_arguments(raw_args: Any) -> str:
    if raw_args in (None, ""):
        return "{}"
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except Exception:
            return json.dumps({"_malformed_arguments": raw_args}, ensure_ascii=False)
        return json.dumps(parsed, ensure_ascii=False)
    return json.dumps(raw_args, ensure_ascii=False, default=str)


def _parse_tool_arguments(raw_args: Any) -> tuple[dict[str, Any] | None, str | None]:
    if raw_args in (None, ""):
        return {}, None
    if isinstance(raw_args, dict):
        return raw_args, None
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except Exception as exc:
            return None, str(exc)
        if not isinstance(parsed, dict):
            return None, f"expected a JSON object, got {type(parsed).__name__}"
        return parsed, None
    return None, f"expected a JSON object string, got {type(raw_args).__name__}"


def _tool_result_content(result: Any) -> str:
    content = json.dumps(result, ensure_ascii=False, default=str)
    if len(content) > _TOOL_RESULT_CHAR_BUDGET:
        return (
            content[:_TOOL_RESULT_CHAR_BUDGET]
            + f"\n... [truncated tool result: {len(content)} chars]"
        )
    return content


def _execute_tool_call(tool_call: dict, functions: dict[str, Any]) -> dict:
    function = tool_call.get("function") or {}
    name = function.get("name")
    raw_args = function.get("arguments") or "{}"
    args, parse_error = _parse_tool_arguments(raw_args)
    if parse_error:
        content = f"an error occurred when parsing arguments for {name}: {parse_error}"
    else:
        if name not in functions:
            content = f"an error occurred when calling {name}: unknown tool"
        else:
            try:
                result = functions[name](**args)
                content = _tool_result_content(result)
            except Exception as exc:
                content = f"an error occurred when calling {name}: {exc}"
    return {"role": "tool", "tool_call_id": tool_call.get("id"), "content": content}


def _needs_final_answer(messages: list[dict]) -> bool:
    return not _ANSWER_RE.search(_last_assistant_content(messages))


def _finalization_context(messages: list[dict], char_budget: int = 6000) -> str:
    question = ""
    for message in messages:
        if message.get("role") == "user":
            question = str(message.get("content") or "")
            break

    chunks = []
    for message in messages[2:]:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if role == "tool":
            chunks.append("Tool observation: " + content)
        elif role == "assistant":
            chunks.append("Assistant attempt: " + content)
        elif role == "user":
            chunks.append("Instruction: " + content)

    recent = "\n\n".join(chunks)
    if len(recent) > char_budget:
        recent = recent[-char_budget:]
    return (
        "Original task:\n"
        f"{question}\n\n"
        "Available observations and attempts:\n"
        f"{recent}\n\n"
        "Return only the best final answer in exactly this format: "
        "<answer>VALUE</answer>."
    )


def _force_final_answer(client: Any, messages: list[dict], telemetry: dict, seed: int | None) -> None:
    messages.append(
        {
            "role": "user",
            "content": (
                "The tool-call budget is exhausted. Do not call any more tools. "
                "Output only the best final answer using exactly this format: "
                "<answer>VALUE</answer>. Do not explain. Do not include any "
                "text before or after the answer tag."
            ),
        }
    )
    kwargs = _completion_kwargs(seed)
    kwargs["temperature"] = 0.0
    kwargs["max_tokens"] = int(os.environ.get("TOOLHOP_FINAL_MAX_TOKENS", "64"))
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict answer extractor. Output exactly one XML "
                        "tag of the form <answer>VALUE</answer> and nothing else. "
                        "Do not explain or include uncertainty."
                    ),
                },
                {"role": "user", "content": _finalization_context(messages)},
            ],
            **kwargs,
        )
    except Exception:
        messages.append({"role": "assistant", "content": "<answer></answer>"})
    else:
        openai_sdk_accumulate(telemetry, response)
        messages.append(_message_dict(response.choices[0].message))


def solve(
    sample: dict,
    *,
    style: str,
    topology: str,
    role: str,
    max_turns: int = DEFAULT_MAX_TURNS,
    seed: int | None = None,
    prompt_suffix: str = "",
    extra_context: str = "",
) -> dict:
    """Run one OpenAI-compatible tool loop on a ToolHop sample."""
    functions = _function_map(sample)
    client = _client()
    tools = _tools_payload(sample)
    user_prompt = _user_prompt(sample)
    if extra_context:
        user_prompt += (
            "\n\nCONTEXT FROM OTHER AGENTS OR PRIOR STAGES:\n"
            + str(extra_context).strip()
        )
    messages: list[dict] = [
        {"role": "system", "content": _system_prompt(topology, role, style, prompt_suffix)},
        {"role": "user", "content": user_prompt},
    ]
    telemetry = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "n_llm_calls": 0,
        "n_tool_calls": 0,
    }

    start = time.time()
    exhausted = True
    for _ in range(max_turns):
        response = client.chat.completions.create(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            **_completion_kwargs(seed),
        )
        openai_sdk_accumulate(telemetry, response)
        message = response.choices[0].message
        tool_calls = [
            _tool_call_dict(tool_call, index)
            for index, tool_call in enumerate(getattr(message, "tool_calls", None) or [])
        ]
        assistant_msg = _message_dict(message, tool_calls)
        messages.append(assistant_msg)
        if not tool_calls:
            exhausted = False
            break
        for tool_call_payload in tool_calls:
            telemetry["n_tool_calls"] += 1
            messages.append(_execute_tool_call(tool_call_payload, functions))

    if _needs_final_answer(messages) and (exhausted or role in FINAL_ANSWER_ROLES):
        _force_final_answer(client, messages, telemetry, seed)

    return {
        "messages": messages,
        "solve_s": time.time() - start,
        "telemetry": normalize(telemetry),
    }


def extract_answer(text: str | None) -> str:
    text = text or ""
    match = _ANSWER_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def score_answer(answer_text: str, solution_str: str, prev_tool_content: str = "") -> bool:
    if "<answer>" in solution_str:
        solution_str = solution_str.split("<answer>")[-1]
    if "</answer>" in solution_str:
        solution_str = solution_str.split("</answer>")[0]

    try:
        ground_truth = ast.literal_eval(answer_text.strip())
    except Exception:
        if str(answer_text).removesuffix(".0").lower() in (
            str(solution_str).removesuffix(".0").replace(",", "").lower()
        ):
            return True
    else:
        try:
            solution = ast.literal_eval(solution_str.strip())
        except Exception:
            pass
        else:
            if ground_truth == solution:
                return True

    return bool(
        prev_tool_content
        and answer_text.removesuffix(".0").lower()
        in prev_tool_content.removesuffix(".0").replace(",", "").lower()
    )


def _last_assistant_content(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message.get("content") or "")
    return ""


def _previous_tool_content(messages: list[dict]) -> str:
    seen_final_assistant = False
    for index in range(len(messages) - 1, 0, -1):
        role = messages[index].get("role")
        if role == "assistant":
            seen_final_assistant = True
            continue
        if seen_final_assistant and role == "tool":
            return str(messages[index].get("content") or "")
    return ""


def _answer_key(answer: str | None) -> str:
    return (answer or "").strip().removesuffix(".0").replace(",", "").lower()


def _sum_telemetry(agent_outputs: list[dict]) -> dict:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "n_llm_calls": 0,
        "n_tool_calls": 0,
    }
    for output in agent_outputs:
        telemetry = output.get("telemetry") or {}
        for key in totals:
            totals[key] += int(telemetry.get(key) or 0)
    return totals


def _compact_agent(agent: dict) -> dict:
    return {
        key: value
        for key, value in agent.items()
        if key not in {"messages", "telemetry"}
    }


def _solve_agent(
    sample: dict,
    *,
    style: str,
    topology: str,
    role: str,
    seed: int,
    prompt_suffix: str = "",
    extra_context: str = "",
) -> dict:
    start = time.time()
    try:
        out = solve(
            sample,
            style=style,
            topology=topology,
            role=role,
            seed=seed,
            prompt_suffix=prompt_suffix,
            extra_context=extra_context,
        )
    except Exception as exc:
        return {
            "role": role,
            "seed": seed,
            "solve_s": round(time.time() - start, 1),
            "error": f"{type(exc).__name__}: {exc}",
            "messages": [],
            "telemetry": {},
        }

    messages = out.get("messages") or []
    final_content = _last_assistant_content(messages)
    predicted = extract_answer(final_content)
    previous_tool_content = _previous_tool_content(messages)
    correct = score_answer(
        str(sample.get("answer", "")),
        final_content,
        previous_tool_content,
    )
    return {
        "role": role,
        "seed": seed,
        "solve_s": round(float(out.get("solve_s") or 0.0), 1),
        "predicted_answer": predicted,
        "answer_key": _answer_key(predicted),
        "answer_correct": int(correct),
        "correct": bool(correct),
        "turns": sum(1 for message in messages if message.get("role") == "assistant"),
        "tool_calls": sum(len(message.get("tool_calls") or []) for message in messages),
        "final_content": final_content,
        "previous_tool_content": previous_tool_content,
        "messages": messages,
        "telemetry": out.get("telemetry") or {},
    }


def _choose_winner(agents: list[dict]) -> int | None:
    candidates = [
        (idx, agent.get("answer_key") or "")
        for idx, agent in enumerate(agents)
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


def _communications_format_from_style(style: str) -> str | None:
    match = re.search(r"_communications_(freeform|semi_structured|structured_soft)(?:$|_)", style or "")
    return match.group(1) if match else None


def _report_text_for_context(report: dict) -> str:
    return str(
        report.get("raw")
        or report.get("final_content")
        or report.get("predicted_answer")
        or report.get("raw_tail")
        or ""
    )


def _fit_context_chunks(chunks: list[str], *, char_budget: int) -> str:
    selected: list[str] = []
    size = 0
    for chunk in reversed(chunks):
        extra = len(chunk) + (2 if selected else 0)
        if selected and size + extra > char_budget:
            continue
        selected.append(chunk)
        size += extra
        if size >= char_budget:
            break
    selected.reverse()
    return "\n\n".join(selected)


def _reports_context(reports: list[dict], *, char_budget: int = 5000, communications_format: str | None = None) -> str:
    chunks = []
    for report in reports:
        label = report.get("role") or f"agent_{report.get('seed', '?')}"
        text = _report_text_for_context(report)
        if not text:
            continue
        if communications_format:
            try:
                normalized = normalize_report(
                    str(label),
                    text[-700:],
                    dataset="toolhop",
                    topology="handoff",
                    payload={"seed": report.get("seed")},
                )
                rendered = render_report(normalized, communications_format)
                chunks.append(f"{label}:\n{rendered}")
            except Exception:
                chunks.append(f"{label}:\n{text[-1200:]}")
        else:
            chunks.append(f"{label}:\n{text[-1200:]}")
    if communications_format:
        return _fit_context_chunks(chunks, char_budget=char_budget)
    context = "\n\n".join(chunks)
    return context[-char_budget:]


def solve_topology(
    sample: dict,
    *,
    style: str,
    topology: str,
    role: str,
    prompt_suffix: str = "",
    roles: list[str] | tuple[str, ...] | None = None,
    worker_roles: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Solve one ToolHop row using the requested topology shape."""
    start = time.time()
    topology_key = topology.replace("_openai", "")
    communications_format = _communications_format_from_style(style)

    if topology_key == "single":
        agent = _solve_agent(
            sample,
            style=style,
            topology=topology,
            role=role,
            seed=0,
            prompt_suffix=prompt_suffix,
        )
        return {
            **agent,
            "n_agents": 1,
            "messages": agent.get("messages") or [],
            "solve_s": agent.get("solve_s", round(time.time() - start, 1)),
            "telemetry": agent.get("telemetry") or {},
        }

    raise ValueError(  # unreachable for correct calls (runner is single-topology)
        f"this runner handles 'single'; received topology={topology!r}"
    )
def _write_trace(path: Path, messages: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for message in messages:
            f.write(f"=== {str(message.get('role', '?')).upper()} ===\n")
            if message.get("content"):
                f.write(str(message.get("content")) + "\n")
            for tool_call in message.get("tool_calls") or []:
                f.write("[tool_call] " + json.dumps(tool_call, ensure_ascii=False, default=str) + "\n")
            f.write("\n")


def _write_topology_trace(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def run_one(
    instance: dict,
    out_dir: Path,
    *,
    style: str,
    topology: str,
    role: str,
) -> dict:
    iid = instance["id"]
    summary: dict[str, Any] = {
        "id": iid,
        "idx": iid,
        "question": instance.get("question"),
        "gold_answer": instance.get("answer"),
        "style": style,
    }

    try:
        out = solve_topology(instance, style=style, topology=topology, role=role)
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["stage"] = "solve"
        return summary

    summary["solve_s"] = round(float(out.get("solve_s") or 0.0), 1)
    summary["predicted_answer"] = out.get("predicted_answer", "")
    summary["answer_correct"] = int(bool(out.get("answer_correct")))
    summary["correct"] = bool(out.get("correct"))
    summary["turns"] = int(out.get("turns") or 0)
    summary["tool_calls"] = int(out.get("tool_calls") or 0)
    for key in ("n_agents", "n_rounds", "winner", "buckets", "per_agent", "per_peer", "by_stage", "stage_outputs", "workers", "manager"):
        if key in out:
            summary[key] = out[key]
    summary.update(out.get("telemetry") or {})
    if out.get("messages"):
        _write_trace(out_dir / "traces" / f"{iid}.txt", out.get("messages") or [])
    else:
        _write_topology_trace(out_dir / "traces" / f"{iid}.json", {"summary": summary})
    return summary


def run_batch(
    *,
    style: str,
    topology: str,
    role: str,
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
        print(f"loaded {len(rows)} instance(s) from {HF_DATASET} ({style})")

    preds_path = out_dir / "predictions.jsonl"
    results_path = out_dir / "results.jsonl"
    correct = 0
    with preds_path.open("a") as fp, results_path.open("a") as fr:
        for index, row in enumerate(rows, 1):
            if verbose:
                print(f"\n[{index}/{len(rows)}] {row['id']}")
            summary = run_one(row, out_dir, style=style, topology=topology, role=role)
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
                print(f"  -> {json.dumps(summary, ensure_ascii=False, default=str)}")

    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": (correct / len(rows)) if rows else 0.0,
        "style": style,
    }


def main(*, style: str, topology: str, role: str) -> int:
    parser = argparse.ArgumentParser(description=f"ToolHop runner ({style}).")
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
        limit=args.limit if not args.only else None,
        offset=args.offset,
        only=args.only,
        out_dir=Path(args.out_dir),
    )
    return 0


STYLE = "single_langgraph"
TOPOLOGY = "single"
ROLE = "solver"

if __name__ == "__main__":
    raise SystemExit(main(style=STYLE, topology=TOPOLOGY, role=ROLE))
