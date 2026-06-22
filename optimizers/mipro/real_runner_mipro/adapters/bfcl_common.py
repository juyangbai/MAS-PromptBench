"""Shared BFCL real-runner adapter utilities."""
from __future__ import annotations

import json
import keyword
import os
import re
from pathlib import Path
from typing import Any, List, Optional

from real_runner_mipro.lm import TASK_MODEL, next_task_endpoint
from real_runner_mipro.output_contracts import append_output_contract


BFCL_DATASET = "bfcl"
DEFAULT_RECURSION_LIMIT = 100


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return workspace_root().parent.parent


def prompt_path(topology: str, role: str) -> Path:
    return repo_root() / "configs" / "prompts" / topology / "bfcl" / f"{role}.txt"


def load_prompt(topology: str, role: str) -> str:
    return prompt_path(topology, role).read_text().strip()


def load_prompts(topology: str, roles: list[str], overrides: dict[str, str] | None = None) -> dict[str, str]:
    overrides = overrides or {}
    return {role: overrides.get(role, load_prompt(topology, role)) for role in roles}


def execution_prompt(prompt: str, topology: str, role: str) -> str:
    return append_output_contract(prompt, BFCL_DATASET, topology, role)


def coerce_instance(example: Any) -> dict:
    if isinstance(example, dict):
        return example
    if hasattr(example, "toDict"):
        return example.toDict()
    data = dict(getattr(example, "__dict__", {}))
    if not data:
        raise TypeError(f"Cannot coerce {type(example).__name__} to instance dict")
    return data


def flatten_user_request(question: list) -> str:
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


def schemas_text(instance: dict) -> str:
    return json.dumps(instance.get("function") or [], indent=2)


_PRIMITIVE_TYPE_MAP = {
    "integer": int,
    "string": str,
    "float": float,
    "number": float,
    "boolean": bool,
    "dict": dict,
    "any": Any,
}


def py_type_of(prop: dict) -> Any:
    t = (prop or {}).get("type", "any")
    if t in ("array", "tuple"):
        items = prop.get("items") or {}
        item_type = py_type_of(items) if items else Any
        return List[item_type]
    return _PRIMITIVE_TYPE_MAP.get(t, Any)


def sanitize_field_name(name: str) -> str:
    safe = name.lstrip("_") or "field"
    if keyword.iskeyword(safe):
        safe = safe + "_"
    return safe


def schema_to_tool(schema: dict):
    from langchain_core.tools import StructuredTool
    from pydantic import Field, create_model

    params = schema.get("parameters") or {}
    properties = params.get("properties") or {}
    required = set(params.get("required") or [])

    fields: dict[str, Any] = {}
    for name, prop in properties.items():
        py_type = py_type_of(prop)
        desc = (prop or {}).get("description", "")
        safe_name = sanitize_field_name(name)
        alias_kw = {"alias": name} if safe_name != name else {}
        if name in required:
            fields[safe_name] = (py_type, Field(..., description=desc, **alias_kw))
        else:
            fields[safe_name] = (Optional[py_type], Field(None, description=desc, **alias_kw))

    safe_name = re.sub(r"\W+", "_", schema["name"]) + "Args"
    args_model = create_model(safe_name, **fields) if fields else None

    return StructuredTool.from_function(
        func=lambda **_: "",
        name=schema["name"],
        description=schema.get("description", ""),
        args_schema=args_model,
    )


def default_chat_model(seed: int = 0):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.environ.get("MODEL_ID", TASK_MODEL),
        base_url=next_task_endpoint(),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        temperature=0.2,
        top_p=0.9,
        seed=seed,
        max_tokens=1024,
        timeout=300.0,
        max_retries=5,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


def recursion_limit(env_name: str | None = None, default: int = DEFAULT_RECURSION_LIMIT) -> int:
    """Read a positive LangGraph recursion limit from the environment."""
    raw = os.environ.get(env_name or "", "") if env_name else ""
    raw = raw or os.environ.get("REAL_RUNNER_RECURSION_LIMIT", "")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def extract_first_tool_calls(messages: list) -> list[dict]:
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            return tool_calls
    return []


def to_canonical(tool_calls: list[dict]) -> list[dict]:
    return [{tc["name"]: dict(tc.get("args") or {})} for tc in tool_calls]


FENCED_RE = re.compile(r"```(?:\w*)\s*([\s\S]*?)\s*```")
NAME_ARGS_PAIRS = (
    ("fn_name", "args"),
    ("name", "arguments"),
    ("function_name", "arguments"),
    ("function", "arguments"),
)


def normalize_call(d: dict) -> dict:
    for name_key, args_key in NAME_ARGS_PAIRS:
        if (
            name_key in d
            and args_key in d
            and isinstance(d[name_key], str)
            and isinstance(d[args_key], dict)
        ):
            return {d[name_key]: d[args_key]}
    return d


def extract_canonical(text: str) -> list[dict] | None:
    text = re.sub(r"\bTERMINATE\b", "", text or "")
    candidates = [m.group(1) for m in FENCED_RE.finditer(text)]
    for cand in reversed(candidates):
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
            return [normalize_call(d) for d in parsed]
    return None


def canonical_key(model_output: list[dict]) -> str:
    normalized = []
    for call in model_output or []:
        normalized_call = {}
        for fn, args in call.items():
            normalized_call[fn] = dict(sorted((args or {}).items()))
        normalized.append(normalized_call)
    normalized.sort(key=lambda c: json.dumps(c, sort_keys=True))
    return json.dumps(normalized, sort_keys=True)


def compact_role_trace(
    *,
    role: str,
    model_output: list[dict],
    winner: Any = None,
    buckets: Any = None,
    details: list[str] | None = None,
) -> str:
    lines = [
        f"role={role}",
        f"winner={winner}",
        f"majority_buckets={buckets or []}",
        f"selected_model_output={model_output or []}",
    ]
    lines.extend(details or [])
    return "\n".join(lines)
