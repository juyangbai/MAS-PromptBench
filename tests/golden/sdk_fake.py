"""Stand-in for the OpenAI Agents SDK turn loop (decentralized/openai_agents).

openai-agents needs openai>=3 and an isolated install that is usually not
importable in the main environment, so the golden harness always replaces
``OpenAIAgentsSDKInvoker.invoke`` (even when the SDK is installed, to keep
goldens environment independent). Everything around the SDK - debate
orchestration, per-(peer, round) seeds, instructions, round inputs, tool
specs and handlers, transcript/telemetry assembly, answer selection - runs
for real. Each SDK model turn is recorded as an ``agents_sdk`` pseudo request
holding exactly what the runner handed to the SDK (instructions, input, tool
schemas, decoding settings); the SDK's own wire format is not covered.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Callable
from typing import Any

from tests.golden import scrub


class UserError(Exception):
    """Mirrors ``agents.exceptions.UserError`` (tool handler failure)."""


class MaxTurnsExceeded(Exception):
    """Mirrors ``agents.exceptions.MaxTurnsExceeded``."""


def install(record: Callable[[str, dict], None], respond: Callable[[dict], dict]) -> None:
    from topologies.decentralized.openai_agents import agents_sdk_base as base

    def invoke(self, *, name, instructions, input_text, request_seed, tools=()):
        specs = {spec.name: spec for spec in tools}
        body: dict[str, Any] = {
            "model": self.model_id,
            "base_url": self.base_url,
            "agent_name": name,
            "instructions": instructions,
            "input": input_text,
            "seed": request_seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "thinking": self.thinking,
            "max_turns": self.max_turns,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": dict(spec.parameters),
                    },
                }
                for spec in tools
            ],
        }
        messages: list[dict] = [{"role": "system", "content": instructions}, {"role": "user", "content": input_text}]
        items: list[dict] = []
        usage = base.Usage()
        for turn in range(int(self.max_turns)):
            request = dict(body, turn=turn, messages=list(messages))
            record("agents_sdk", request)
            reply = respond({"messages": messages, "tools": body["tools"]})
            prompt_tokens = max(1, len(json.dumps(scrub.to_data(messages))) // 4)
            completion_tokens = max(1, len(json.dumps(scrub.to_data(reply))) // 4)
            usage = usage + base.Usage(
                model_calls=1,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            calls = reply.get("tool_calls") or []
            if not calls:
                content = reply.get("content") or ""
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content, "annotations": []}],
                    }
                )
                events = [it for it in items if it["type"] in ("function_call", "function_call_output")]
                n_calls = sum(1 for it in items if it["type"] == "function_call")
                usage = base.Usage(
                    model_calls=usage.model_calls,
                    tool_calls=n_calls,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                )
                return base.AgentTurnResult(output=content, usage=usage, items=items, tool_events=events)
            for index, call in enumerate(calls):
                arguments = json.dumps(call.get("arguments") or {}, ensure_ascii=False)
                call_id = "call_" + hashlib.sha1(f"{name}|{request_seed}|{turn}|{index}".encode()).hexdigest()[:16]
                items.append(
                    {"type": "function_call", "call_id": call_id, "name": call["name"], "arguments": arguments}
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": call["name"], "arguments": arguments},
                            }
                        ],
                    }
                )
                spec = specs.get(call["name"])
                if spec is None:
                    raise UserError(f"Error running tool {call['name']}: unknown tool")
                try:
                    value = spec.handler(json.loads(arguments))
                    if inspect.isawaitable(value):
                        value = asyncio.new_event_loop().run_until_complete(value)
                except Exception as exc:  # the SDK surfaces handler failures as UserError
                    raise UserError(f"Error running tool {call['name']}: {exc}") from exc
                output = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
                items.append({"type": "function_call_output", "call_id": call_id, "output": output})
                messages.append({"role": "tool", "tool_call_id": call_id, "content": output})
        raise MaxTurnsExceeded(f"Max turns ({self.max_turns}) exceeded")

    base.OpenAIAgentsSDKInvoker.invoke = invoke
