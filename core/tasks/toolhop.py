"""ToolHop: multi-hop tool-use questions, sandboxed dataset tools, the tool loop, scoring and batch records.

Rows of bytedance-research/ToolHop (``data/ToolHop.json``) carry a question, its
answer, OpenAI-style tool schemas and the Python source of each tool. Solving
executes that dataset-provided source in a restricted namespace, so it requires
``TOOLHOP_ALLOW_DATASET_EXEC=1``. An agent runs a tool-calling loop on the raw
OpenAI client and ends with ``<answer>VALUE</answer>``; the answer is scored
with ToolHop's own matcher (literal equality, substring, or the last tool result).
"""

from __future__ import annotations

import ast
import builtins as _builtins
import datetime as _datetime_module
import json
import math
import os
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime as _datetime_class
from datetime import timedelta as _timedelta
from pathlib import Path
from typing import Any

from core import agent_runs, cli
from core.batch import attempt
from core.llm import completion_kwargs, openai_client
from core.output_contracts import append_output_contract
from core.paths import PROMPTS_DIR
from core.telemetry import normalize, openai_sdk_accumulate, sum_telemetry

DATASET = "toolhop"
HF_DATASET = "bytedance-research/ToolHop"
DEFAULT_MAX_TURNS = int(os.environ.get("TOOLHOP_MAX_TURNS", "9"))
_TOOL_RESULT_CHAR_BUDGET = int(os.environ.get("TOOLHOP_TOOL_RESULT_CHAR_BUDGET", "6000"))
# Roles whose run must end with an <answer> tag (forced if the loop ended without one).
FINAL_ANSWER_ROLES = {"solver", "verifier", "manager", "debater"}


# Data
def _dataset_path() -> Path:
    from huggingface_hub import hf_hub_download

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
                function_names.extend(node.name for node in tree.body if isinstance(node, ast.FunctionDef))
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
            if not isinstance(params, dict) or params.get("type") != "object" or "properties" not in params:
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


# Sandbox: the dataset's tool code runs with allow-listed builtins, imports and helpers
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


# Tools: the row's functions (sandboxed) and their schemas
_SAFE_EXCEPTION_BASES = {"Exception", "ValueError", "RuntimeError"}


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


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


def function_map(sample: dict) -> dict[str, Any]:
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


@dataclass(frozen=True)
class Tool:
    """One dataset tool as a function tool: its schema's ``parameters`` and ``call(arguments)``."""

    name: str
    description: str
    parameters: Any
    call: Callable[[dict], Any]


def function_tools(sample: dict, functions: dict[str, Any]) -> list[Tool]:
    """The row's tool schemas as tools backed by ``functions`` (see :func:`function_map`)."""

    def call(arguments: dict, *, name: str) -> Any:
        if name not in functions:
            raise KeyError(f"unknown ToolHop function {name}")
        return functions[name](**dict(arguments))

    return [
        Tool(
            name=str(entry["runtime_name"]),
            description=str(entry["schema"].get("description") or ""),
            parameters=entry["schema"].get("parameters"),
            call=lambda arguments, name=str(entry["runtime_name"]): call(arguments, name=name),
        )
        for entry in _tool_schema_entries(sample)
    ]


# Prompts
_GENERIC_PROMPT = (
    "You are solving ToolHop, a multi-hop tool-use benchmark. Use the "
    "provided tools to answer the user's question. Keep intermediate "
    "reasoning concise. Use tool outputs to continue the chain. The final "
    "response must be short."
)


def system_prompt(topology: str, role: str, style: str, prompt_suffix: str = "") -> str:
    """System prompt of ``role``: its prompt file (else a generic prompt naming ``style``)
    plus ``prompt_suffix``, under the role's protected output contract."""
    path = PROMPTS_DIR / topology / DATASET / f"{role}.txt"
    if path.exists():
        prompt = path.read_text().strip()
    else:
        prompt = _GENERIC_PROMPT + f"\n\nImplementation style: {style}."
    if prompt_suffix:
        prompt = prompt.rstrip() + "\n" + prompt_suffix
    return append_output_contract(prompt, DATASET, topology, role)


def user_prompt(sample: dict) -> str:
    """The question with ToolHop's answer-format instructions."""
    return (
        "Answer the question using the available tools. If the final answer is "
        "a date, use YYYY-MM-DD. If it is a name, use Firstname Lastname. If it "
        "contains a number, output the number, not a word, with no leading "
        "zeroes.\n\nQuestion: " + str(sample["question"])
    )


def user_message(sample: dict, extra_context: str = "") -> str:
    """:func:`user_prompt` plus the reports of other agents or earlier stages, if any."""
    message = user_prompt(sample)
    if extra_context:
        message += "\n\nCONTEXT FROM OTHER AGENTS OR PRIOR STAGES:\n" + str(extra_context).strip()
    return message


# The tool loop
def client(base_url: str) -> Any:
    """Raw OpenAI client (600 s timeout, 5 SDK retries)."""
    return openai_client(base_url=base_url)


def _message_dict(message, tool_call_payloads: list[dict] | None = None) -> dict:
    payload = message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else dict(message)
    message_dict = {key: payload[key] for key in ("role", "content") if key in payload and payload[key] is not None}
    if tool_call_payloads is not None:
        message_dict["tool_calls"] = [_sanitize_tool_call_for_message(tool_call) for tool_call in tool_call_payloads]
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
        return content[:_TOOL_RESULT_CHAR_BUDGET] + f"\n... [truncated tool result: {len(content)} chars]"
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
    return not _ANSWER_RE.search(last_assistant_content(messages))


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


def _force_final_answer(llm: Any, messages: list[dict], telemetry: dict, seed: int | None, *, model: str) -> None:
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
    kwargs = completion_kwargs(seed, model=model)
    kwargs["temperature"] = 0.0
    kwargs["max_tokens"] = int(os.environ.get("TASK_MODEL_MAX_TOKENS", "32768"))
    try:
        response = llm.chat.completions.create(
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


def tool_loop(
    llm: Any,
    sample: dict,
    functions: dict[str, Any],
    system: str,
    user: str,
    *,
    model: str,
    role: str,
    seed: int | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> dict:
    """Let the model call the row's tools until it answers without a tool call or
    ``max_turns`` requests are used; then force an ``<answer>`` tag if the
    budget ran out or ``role`` must answer. Returns ``{"messages", "solve_s", "telemetry"}``."""
    tools = _tools_payload(sample)
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    telemetry = normalize(None)
    start = time.time()
    exhausted = True
    for _ in range(max_turns):
        response = llm.chat.completions.create(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            **completion_kwargs(seed, model=model),
        )
        openai_sdk_accumulate(telemetry, response)
        message = response.choices[0].message
        tool_calls = [
            _tool_call_dict(tool_call, index)
            for index, tool_call in enumerate(getattr(message, "tool_calls", None) or [])
        ]
        messages.append(_message_dict(message, tool_calls))
        if not tool_calls:
            exhausted = False
            break
        for tool_call in tool_calls:
            telemetry["n_tool_calls"] += 1
            messages.append(_execute_tool_call(tool_call, functions))

    if _needs_final_answer(messages) and (exhausted or role in FINAL_ANSWER_ROLES):
        _force_final_answer(llm, messages, telemetry, seed, model=model)
    return {"messages": messages, "solve_s": time.time() - start, "telemetry": normalize(telemetry)}


# Answers and scoring
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


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
        and answer_text.removesuffix(".0").lower() in prev_tool_content.removesuffix(".0").replace(",", "").lower()
    )


def last_assistant_content(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message.get("content") or "")
    return ""


def previous_tool_content(messages: list[dict]) -> str:
    seen_final_assistant = False
    for index in range(len(messages) - 1, 0, -1):
        role = messages[index].get("role")
        if role == "assistant":
            seen_final_assistant = True
            continue
        if seen_final_assistant and role == "tool":
            return str(messages[index].get("content") or "")
    return ""


def answer_key(answer: str | None) -> str:
    return (answer or "").strip().removesuffix(".0").replace(",", "").lower()


# Agents: one tool loop each, and how a topology combines them
def scored_answer(sample: dict, final_content: str, prev_tool_content: str) -> dict:
    """The answer of a final message and its score."""
    predicted = extract_answer(final_content)
    correct = score_answer(str(sample.get("answer", "")), final_content, prev_tool_content)
    return {
        "predicted_answer": predicted,
        "answer_key": answer_key(predicted),
        "answer_correct": int(correct),
        "correct": bool(correct),
    }


def solve_agent(solve: Callable[..., dict], sample: dict, *, role: str, seed: int, **kwargs: Any) -> dict:
    """Run ``solve(sample, role=, seed=, **kwargs)`` for one agent and score it; a failure becomes ``error``."""
    out, latency_s, error = attempt(lambda: solve(sample, role=role, seed=seed, **kwargs))
    if error is not None:
        return {
            "role": role,
            "seed": seed,
            "solve_s": round(latency_s, 1),
            "error": error,
            "messages": [],
            "telemetry": {},
        }
    messages = out.get("messages") or []
    final_content = last_assistant_content(messages)
    prev_tool_content = previous_tool_content(messages)
    return {
        "role": role,
        "seed": seed,
        "solve_s": round(float(out.get("solve_s") or 0.0), 1),
        **scored_answer(sample, final_content, prev_tool_content),
        "turns": sum(1 for message in messages if message.get("role") == "assistant"),
        "tool_calls": sum(len(message.get("tool_calls") or []) for message in messages),
        "final_content": final_content,
        "previous_tool_content": prev_tool_content,
        "messages": messages,
        "telemetry": out.get("telemetry") or {},
    }


def answer_fields(agent: dict) -> dict:
    """A topology's prediction fields, taken from the agent whose answer it submits."""
    return {
        "predicted_answer": agent.get("predicted_answer", ""),
        "answer_correct": int(bool(agent.get("correct"))),
        "correct": bool(agent.get("correct")),
    }


def usage(agents: list[dict]) -> dict:
    """Tool calls and model turns summed over ``agents``."""
    return {
        "tool_calls": sum(int(agent.get("tool_calls") or 0) for agent in agents),
        "turns": sum(int(agent.get("turns") or 0) for agent in agents),
    }


def vote_buckets(agents: list[dict]) -> dict[str, int]:
    """Votes per answer key among the agents that finished without an error."""
    return dict(Counter(agent.get("answer_key") or "" for agent in agents if not agent.get("error")))


def choose_winner(agents: list[dict]) -> int | None:
    """Index of the first agent with the most common answer (None if every agent failed)."""
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


def compact_agent(agent: dict) -> dict:
    """An agent's entry in a topology output: without messages and telemetry."""
    return {key: value for key, value in agent.items() if key not in {"messages", "telemetry"}}


# Batch records
def _write_messages(path: Path, messages: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for message in messages:
            f.write(f"=== {str(message.get('role', '?')).upper()} ===\n")
            if message.get("content"):
                f.write(str(message.get("content")) + "\n")
            for tool_call in message.get("tool_calls") or []:
                f.write("[tool_call] " + json.dumps(tool_call, ensure_ascii=False, default=str) + "\n")
            f.write("\n")


def run_instance(
    instance: dict,
    out_dir: Path,
    *,
    style: str,
    solve: Callable[[], dict],
    debate_errors: bool = False,
) -> dict:
    """Record of one instance solved by ``solve()`` (a topology's output); writes its trace to ``traces/``.

    ``debate_errors`` records an ``error`` of the output with stage ``debate``.
    """
    iid = instance["id"]
    summary: dict[str, Any] = {
        "id": iid,
        "idx": iid,
        "question": instance.get("question"),
        "gold_answer": instance.get("answer"),
        "style": style,
    }
    try:
        out = solve()
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
    if debate_errors and out.get("error"):
        summary["error"] = out["error"]
        summary["stage"] = "debate"
    for key in agent_runs.OUTPUT_KEYS:
        if key in out:
            summary[key] = out[key]
    summary.update(out.get("telemetry") or {})
    if out.get("messages"):
        _write_messages(out_dir / "traces" / f"{iid}.txt", out.get("messages") or [])
    else:
        agent_runs.write_json(out_dir / "traces" / f"{iid}.json", {"summary": summary})
    return summary


def _prediction(instance: dict, record: dict) -> dict:
    return {
        "idx": instance["id"],
        "id": instance["id"],
        "question": instance.get("question"),
        "predicted_answer": record.get("predicted_answer"),
    }


def run_rows(
    instances: list[dict],
    run_one: Callable[[dict, Path], dict],
    *,
    style: str,
    model_id: str,
    out_dir: Path | None = None,
    predictions: Path | None = None,
    team_size: int | None = None,
    verbose: bool = True,
) -> dict:
    """:func:`core.agent_runs.run_rows` for ToolHop (``out_dir`` default ``results/toolhop/<style>``).

    The progress of a team-size run leaves the replicas and votes out.
    """
    return agent_runs.run_rows(
        instances,
        run_one,
        dataset=DATASET,
        source=HF_DATASET,
        style=style,
        model_id=model_id,
        prediction=_prediction,
        out_dir=out_dir,
        predictions=predictions,
        team_size=team_size,
        hidden=("per_agent", "buckets") if team_size else (),
        verbose=verbose,
    )


# Team sizes: majority vote over seeded replicas of one role
_NOT_IN_REPLICA = {"role", "final_content", "previous_tool_content"}


def solve_replicas(
    solve: Callable[..., dict], sample: dict, *, style: str, topology: str, role: str, n: int
) -> list[dict]:
    """``n`` scored tool loops of ``solve`` under ``role`` (seed = replica index)."""
    agents = [solve_agent(solve, sample, style=style, topology=topology, role=role, seed=seed) for seed in range(n)]
    return [{key: value for key, value in agent.items() if key not in _NOT_IN_REPLICA} for agent in agents]


def _write_replicas(path: Path, summary: dict, replicas: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(
            f"style: {summary.get('style')}\n"
            f"team_size: {summary.get('team_size')}\n"
            f"winner: {summary.get('winner')}\n"
            f"predicted_answer: {summary.get('predicted_answer')}\n"
            f"gold_answer: {summary.get('gold_answer')}\n\n"
        )
        for idx, agent in enumerate(replicas):
            f.write(f"=== AGENT {idx} seed={agent.get('seed')} ===\n")
            if agent.get("error"):
                f.write(f"error: {agent['error']}\n\n")
                continue
            f.write(
                f"predicted_answer: {agent.get('predicted_answer')}\n"
                f"correct: {agent.get('correct')}\n"
                f"turns: {agent.get('turns')} tool_calls: {agent.get('tool_calls')}\n"
            )
            final = last_assistant_content(agent.get("messages") or [])
            if final:
                f.write("\nfinal assistant:\n" + final + "\n")
            f.write("\n")


def run_replicas(
    instance: dict,
    out_dir: Path,
    *,
    style: str,
    team_size: int,
    solve: Callable[[], list[dict]],
) -> dict:
    """Record of one instance solved by ``solve()`` (the replicas); the winning answer is scored again."""
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
    start = time.time()
    replicas = solve()
    summary["solve_s"] = round(time.time() - start, 1)
    summary["per_agent"] = [{key: value for key, value in agent.items() if key != "messages"} for agent in replicas]
    summary["buckets"] = vote_buckets(replicas)
    winner = choose_winner(replicas)
    summary["winner"] = winner
    if winner is None:
        summary["predicted_answer"] = ""
        summary["answer_correct"] = 0
        summary["correct"] = False
        summary["error"] = "all ToolHop replicas failed"
        summary["stage"] = "solve"
    else:
        predicted = replicas[winner].get("predicted_answer") or ""
        summary["predicted_answer"] = predicted
        summary["answer_correct"] = int(score_answer(str(instance.get("answer", "")), predicted))
        summary["correct"] = bool(summary["answer_correct"])
    summary.update(usage(replicas))
    summary.update(sum_telemetry(replicas))
    _write_replicas(out_dir / "traces" / f"{iid}.txt", summary, replicas)
    return summary


# Command line
DEFAULT_LIMIT = 5
SMOKE_DATASET = cli.InfoMode(
    "--smoke-dataset", dataset_summary, help="Only load and validate ToolHop; do not call the model or execute tools."
)


def main(
    argv: list[str] | None,
    *,
    description: str,
    run_one: Callable[[dict, Path], dict],
    style: str,
    model_id: str,
    team_size: int | None = None,
    preflight: Callable[[], None] | None = None,
) -> int:
    """Command line of a ToolHop runner (:mod:`core.cli`); ``--smoke-dataset`` prints :func:`dataset_summary`."""

    def run_batch(instances: list[dict], out_dir: Path | None = None, out_path: Path | None = None) -> dict:
        return run_rows(
            instances,
            run_one,
            style=style,
            model_id=model_id,
            out_dir=out_dir,
            predictions=out_path,
            team_size=team_size,
        )

    return cli.main(
        argv,
        description=description,
        load_instances=load_instances,
        run_batch=run_batch,
        default_limit=DEFAULT_LIMIT,
        info=SMOKE_DATASET,
        preflight=preflight,
    )
