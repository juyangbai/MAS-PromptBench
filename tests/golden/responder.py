"""Deterministic scripted "model" behind the fake chat-completions server.

The reply is a pure function of (request body, per-cell script), so runs are
reproducible even when a topology issues requests concurrently. The policy
plays a cooperative agent along the shortest path that still exercises every
coordination mechanism once:

1. AutoGen ``SelectorGroupChat`` speaker selection: pick the next worker that
   has not spoken yet (at most ``MAX_DELEGATIONS`` workers), then the manager.
2. Domain tool calls (``script["domain_calls"]``, e.g. BFCL ground truth) when
   the request offers those functions and none was called yet.
3. ``delegate_to_*`` tools (LangGraph hub-and-spoke): call the next not yet
   called delegation tool, at most ``MAX_DELEGATIONS`` per conversation.
4. One exercise call of an ordinary tool when ``script["exercise_tools"]``
   allows it and no tool has run yet in this conversation.
5. Otherwise the dataset's final answer (``script["final"]``) - wrapped in the
   CrewAI ``Final Answer:`` format when the prompt asks for it, followed by
   ``TERMINATE`` when the prompt asks for that marker. An AutoGen-style
   manager (TERMINATE contract, no delegation tools) first hands the floor to
   the selector and terminates once it is selected again right after itself.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

MAX_DELEGATIONS = 3

_SELECTOR_MARKERS = ("Pick exactly one agent from", "No valid name was mentioned", "select the next role from")
_PARTICIPANTS_RE = re.compile(r"\[((?:'[^']+'(?:,\s*)?)+)\]")
MANAGER_HANDOFF = "PLAN: hand the next step to the most suitable worker and wait for its report."
DEFAULT_TOOL_STRING = "2+3"


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def _tool_names(body: dict) -> list[str]:
    names = []
    for tool in body.get("tools") or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        name = (fn or {}).get("name") if isinstance(fn, dict) else None
        if name:
            names.append(name)
    return names


def _tool_schema(body: dict, name: str) -> dict:
    for tool in body.get("tools") or []:
        fn = tool.get("function") or {}
        if fn.get("name") == name:
            return fn.get("parameters") or {}
    return {}


def _called_tools(messages: list[dict]) -> list[str]:
    called = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            if fn.get("name"):
                called.append(fn["name"])
    return called


def _example_value(name: str, schema: dict) -> Any:
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "string")
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    if kind in ("integer",):
        return 1
    if kind in ("number", "float"):
        return 1.5
    if kind == "boolean":
        return True
    if kind in ("array", "tuple"):
        return [_example_value(name, schema.get("items") or {"type": "string"})]
    if kind in ("object", "dict"):
        return {}
    for option in schema.get("anyOf") or []:
        if option.get("type") != "null":
            return _example_value(name, option)
    lowered = name.lower()
    if "code" in lowered or "program" in lowered or "source" in lowered:
        return "print(2 + 3)"
    if "instruction" in lowered or "task" in lowered or "request" in lowered:
        return "Handle your part of the task and report back."
    if "expression" in lowered or "expr" in lowered:
        return DEFAULT_TOOL_STRING
    return "golden"


def example_arguments(schema: dict) -> dict:
    props = schema.get("properties") or {}
    required = schema.get("required") or list(props)
    return {name: _example_value(name, props.get(name) or {}) for name in required if name in props}


def _selector_reply(text: str) -> str | None:
    matches = _PARTICIPANTS_RE.findall(text)
    if not matches:
        return None
    participants = [p.strip().strip("'") for p in matches[-1].split(",") if p.strip()]
    if not participants:
        return None
    manager = "manager" if "manager" in participants else participants[0]
    workers = [p for p in participants if p != manager]
    history = text.split("Conversation so far:", 1)[-1]
    spoken = [w for w in workers if re.search(rf"(?m)^{re.escape(w)}:", history)]
    if len(spoken) < min(MAX_DELEGATIONS, len(workers)):
        for worker in workers:
            if worker not in spoken:
                return worker
    return manager


def _final_text(script: dict, system_text: str, prompt_text: str) -> str:
    final = script.get("final") or "Answer: unknown"
    if "Final Answer:" in system_text and "Thought:" in system_text:
        final = f"Thought: I now know the final answer\nFinal Answer: {final}"
    if "TERMINATE" in system_text:
        final = f"{final}\nTERMINATE"
    return final


def respond(body: dict, script: dict) -> dict:
    messages = [m for m in (body.get("messages") or []) if isinstance(m, dict)]
    system_text = "\n".join(_text(m.get("content")) for m in messages if m.get("role") in ("system", "developer"))
    all_text = "\n".join(_text(m.get("content")) for m in messages)
    offered = _tool_names(body)
    called = _called_tools(messages)
    has_tool_results = any(m.get("role") == "tool" for m in messages)

    # 1. AutoGen speaker selection.
    if not offered and any(marker in all_text for marker in _SELECTOR_MARKERS):
        first_user = next((_text(m.get("content")) for m in messages if m.get("role") == "user"), "")
        choice = _selector_reply(first_user)
        if choice:
            return {"content": choice}

    # 2. Domain tool calls (BFCL ground truth) when the functions are offered.
    domain = script.get("domain_calls") or []
    if domain and offered:
        names = [call["name"] for call in domain]
        if all(name in offered for name in names) and not (set(names) & set(called)):
            return {"content": None, "tool_calls": [dict(call) for call in domain]}

    # 3. Hub-and-spoke delegation tools.
    delegation = [name for name in offered if name.startswith("delegate_to_")]
    if delegation:
        done = [name for name in called if name.startswith("delegate_to_")]
        pending = [name for name in delegation if name not in done]
        if pending and len(done) < MAX_DELEGATIONS:
            name = pending[0]
            return {
                "content": None,
                "tool_calls": [{"name": name, "arguments": example_arguments(_tool_schema(body, name))}],
            }

    # 4. One exercise call of an ordinary tool (not in AutoGen group chats,
    #    whose tool-call summaries would replace the agents' replies).
    autogen_style = any(m.get("role") == "user" and m.get("name") for m in messages)
    allowed = script.get("exercise_tools")
    if allowed and not autogen_style and not has_tool_results and not called:
        candidates = [name for name in offered if not name.startswith("delegate_to_")]
        if allowed != "*":
            candidates = [name for name in candidates if name in set(allowed)]
        if candidates:
            name = candidates[0]
            arguments = (script.get("tool_args") or {}).get(name) or example_arguments(_tool_schema(body, name))
            return {"content": None, "tool_calls": [{"name": name, "arguments": arguments}]}

    # 5. AutoGen-style manager: hand off first, terminate when re-selected.
    if autogen_style and "TERMINATE" in system_text and not delegation:
        last = messages[-1] if messages else {}
        if last.get("role") != "assistant":
            return {"content": MANAGER_HANDOFF}

    return {"content": _final_text(script, system_text, all_text)}


# Script helpers used by the drivers -------------------------------------------------


def bfcl_domain_calls(ground_truth: list) -> list[dict]:
    """Canonical BFCL ground truth -> one concrete call per function.

    ``{"fn": {"arg": [v1, v2]}}``: the first non-empty acceptable value is
    used; arguments whose only acceptable value is "" (optional) are omitted.
    """
    calls = []
    for entry in ground_truth or []:
        if not isinstance(entry, dict):
            continue
        for name, args in entry.items():
            concrete = {}
            for arg, options in (args or {}).items():
                values = options if isinstance(options, list) else [options]
                chosen = next((v for v in values if v != ""), None)
                if chosen is None:
                    continue
                concrete[arg] = chosen
            calls.append({"name": name, "arguments": concrete})
    return calls


def bfcl_final(calls: list[dict]) -> str:
    payload = [{c["name"]: c["arguments"]} for c in calls]
    return "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


def apibank_final(gold_call: str | None) -> str:
    return (gold_call or "[UNKNOWN()]").strip()


def literal_or_text(value: Any) -> str:
    if isinstance(value, str):
        try:
            return str(ast.literal_eval(value))
        except Exception:
            return value
    return str(value)
