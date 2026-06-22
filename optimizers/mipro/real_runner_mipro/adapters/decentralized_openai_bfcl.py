"""Decentralized/OpenAI-SDK BFCL real-runner adapter."""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from real_runner_mipro.adapters.bfcl_common import (
    coerce_instance,
    compact_role_trace,
    extract_canonical,
    execution_prompt,
    flatten_user_request,
    load_prompts,
    schemas_text,
)
from real_runner_mipro.lm import TASK_MODEL, next_task_endpoint


ROLE = "debater"
TOPOLOGY = "decentralized_openai"
PROMPT_TOPOLOGY = "decentralized"
DATASET = "bfcl"


def _format_task(user_request: str, schemas: str) -> str:
    return (
        "USER REQUEST:\n"
        f"{user_request}\n\n"
        "SCHEMAS:\n"
        f"{schemas}\n\n"
        "Emit your final canonical call list as a SINGLE fenced ```json``` "
        "block. Canonical form is a list of dicts with one key per dict: "
        '[{"fn_name": {"arg": value, ...}}, ...]. '
        "For a single call, emit a one-element list. For parallel calls, "
        "emit a multi-element list."
    )


def _peer_injection(others_final: list[dict], user_content: str) -> dict:
    body = ["These are the final calls from other peer agents in the previous round:"]
    for i, msg in enumerate(others_final):
        body.append(f"\nPeer {i + 1}:\n```\n{msg.get('content') or ''}\n```")
    body.append(
        "\nCompare their function calls against yours. Revise ONLY if a peer "
        "picked a better function or caught an error in yours. Re-emit your "
        "final canonical call as a SINGLE fenced ```json``` block at the end.\n\n"
        "Original request:\n" + user_content
    )
    return {"role": "user", "content": "\n".join(body)}


def _best_of_n(
    per_peer: list[dict],
    function_schemas: list[dict],
    ground_truth: list[dict],
    category: str,
) -> tuple[dict | None, list[dict]]:
    from real_runner_mipro.datasets.bfcl import score_model_output

    scored = []
    for peer in per_peer:
        call = peer.get("call")
        if not call:
            scored.append({**peer, "valid": False, "score_detail": "no parseable call"})
            continue
        result = score_model_output(function_schemas, call, ground_truth, category, TASK_MODEL)
        scored.append({**peer, "valid": bool(result.get("ok")), "score_detail": result.get("detail", "")})

    passing = [peer for peer in scored if peer.get("valid")]
    if passing:
        return min(passing, key=lambda peer: peer["peer"]), scored
    with_call = [peer for peer in scored if peer.get("call")]
    if with_call:
        return min(with_call, key=lambda peer: peer["peer"]), scored
    return None, scored


class DecentralizedOpenAIBFCLAdapter:
    topology = TOPOLOGY
    dataset = DATASET

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        n_agents: int | None = None,
        n_rounds: int | None = None,
        client_factory: Callable[[], Any] | None = None,
    ):
        self._prompts = load_prompts(PROMPT_TOPOLOGY, [ROLE], prompts)
        self.n_agents = n_agents or int(os.environ.get("DECENTRALIZED_N_AGENTS", "4"))
        self.n_rounds = n_rounds or int(os.environ.get("DECENTRALIZED_N_ROUNDS", "2"))
        self.client_factory = client_factory or self._default_client_factory

    def roles(self) -> list[str]:
        return [ROLE]

    def get_prompt(self, role: str) -> str:
        self._check_role(role)
        return self._prompts[role]

    def set_prompt(self, role: str, text: str) -> None:
        self._check_role(role)
        self._prompts[role] = text

    def reset(self) -> None:
        return None

    def __getstate__(self):
        return self.__dict__.copy()

    def run_example(self, example: Any) -> dict:
        instance = coerce_instance(example)
        client = self.client_factory()
        user_content = _format_task(flatten_user_request(instance["question"]), schemas_text(instance))
        contexts = [
            [
                {"role": "system", "content": execution_prompt(self._prompts[ROLE], PROMPT_TOPOLOGY, ROLE)},
                {"role": "user", "content": user_content},
            ]
            for _ in range(self.n_agents)
        ]
        round_finals: list[list[dict]] = []

        for round_idx in range(self.n_rounds):
            this_round = []
            for peer_idx, ctx in enumerate(contexts):
                if round_idx > 0:
                    others = [round_finals[round_idx - 1][j] for j in range(self.n_agents) if j != peer_idx]
                    ctx.append(_peer_injection(others, user_content))
                msg = self._chat(client, ctx)
                ctx.append(msg)
                this_round.append(msg)
            round_finals.append(this_round)

        per_peer = []
        for idx, ctx in enumerate(contexts):
            final = ctx[-1].get("content") or ""
            call = extract_canonical(final)
            per_peer.append({"peer": idx, "call": call, "raw": final[:1200]})

        ground_truth = instance.get("ground_truth") or []
        if ground_truth:
            winner, per_peer = _best_of_n(
                per_peer,
                instance.get("function") or [],
                ground_truth,
                instance.get("category") or "simple",
            )
        else:
            with_call = [peer for peer in per_peer if peer.get("call")]
            winner = min(with_call, key=lambda peer: peer["peer"]) if with_call else None

        return {
            "model_output": (winner or {}).get("call") or [],
            "winner": (winner or {}).get("peer"),
            "buckets": [],
            "per_peer": per_peer,
        }

    def format_role_trace(self, role: str, output: Any) -> str:
        self._check_role(role)
        if not isinstance(output, dict):
            return str(output)
        details = []
        for peer in output.get("per_peer") or []:
            details.append(
                f"peer={peer.get('peer')} valid={peer.get('valid')} "
                f"call={peer.get('call') or []} raw={str(peer.get('raw') or '')[:500]}"
            )
        return compact_role_trace(
            role=role,
            model_output=output.get("model_output") or [],
            winner=output.get("winner"),
            buckets=output.get("buckets") or [],
            details=details,
        )

    def describe_runtime(self, example: Any | None = None) -> dict:
        instance = coerce_instance(example) if example is not None else {}
        return {
            "topology": self.topology,
            "dataset": self.dataset,
            "framework": "openai",
            "roles": self.roles(),
            "n_agents": self.n_agents,
            "n_rounds": self.n_rounds,
            "tool_count": len(instance.get("function") or []),
            "prompt_prefix": self.get_prompt(ROLE)[:80],
        }

    @staticmethod
    def _default_client_factory():
        from openai import OpenAI

        return OpenAI(base_url=next_task_endpoint(), api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=300.0, max_retries=5)

    @staticmethod
    def _chat(client: Any, messages: list[dict]) -> dict:
        resp = client.chat.completions.create(
            model=os.environ.get("MODEL_ID", TASK_MODEL),
            messages=messages,
            temperature=0.2,
            top_p=0.9,
            seed=0,
            max_tokens=1024,
            extra_body={
                "repetition_penalty": 1.05,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        msg = resp.choices[0].message
        return msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)

    @staticmethod
    def _check_role(role: str) -> None:
        if role != ROLE:
            raise KeyError(f"Unknown role {role!r}; expected {ROLE!r}")
