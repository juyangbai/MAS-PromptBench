"""The reflection client: request body, telemetry, and the logical-seed wrapper's common policy."""

from __future__ import annotations

import json

from ..reflection import LogicalTextReflectionClient, ReflectionClient
from .fakes import FakeOpenAI, fake_cell


def test_client_sends_the_callers_sampling_and_records_telemetry():
    wire = FakeOpenAI()
    client = ReflectionClient(client=wire, model="reflection-model")
    text = client.complete(
        "reflect",
        request_seed=123,
        temperature=0.7,
        top_p=1.0,
        max_output_tokens=48000,
        thinking=True,
        system="system text",
        phase="proposal",
        role="writer",
    )
    assert text == "improved prompt"
    body = wire.bodies[-1]
    assert body["messages"] == [{"role": "system", "content": "system text"}, {"role": "user", "content": "reflect"}]
    assert (body["model"], body["temperature"], body["top_p"], body["max_tokens"], body["seed"]) == (
        "reflection-model",
        0.7,
        1.0,
        48000,
        123,
    )
    assert body["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    snapshot = client.snapshot()
    assert snapshot["usage"]["model_calls"] == 1 and snapshot["usage"]["input_tokens"] == 11
    request = snapshot["requests"][0]
    assert (request["phase"], request["role"], request["request_seed"], request["finish_reason"]) == (
        "proposal",
        "writer",
        123,
        "stop",
    )
    assert client.usage == {"prompt_tokens": 11, "completion_tokens": 7, "n_calls": 1}


def test_wrapper_enforces_the_common_policy_and_keeps_native_sampling():
    wire = FakeOpenAI()
    raw = ReflectionClient(client=wire, model="Qwen/Qwen3.5-122B-A10B-FP8", max_retries=3)
    wrapped = LogicalTextReflectionClient(
        cell=fake_cell(method="tavo"), client=raw, method="tavo", default_temperature=0.5, default_top_p=1.0
    )
    assert wrapped.complete("reflect", temperature=0.3, max_tokens=1536) == "improved prompt"
    body = wire.bodies[-1]
    assert body["max_tokens"] == 48000 and body["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert (body["temperature"], body["top_p"]) == (0.3, 1.0)
    assert body["seed"] == wrapped.requests[-1]["request_seed"] and wrapped.requests[-1]["status"] == "success"
    assert wrapped.requests[-1]["finish_reason"] == "stop" and wrapped.requests[-1]["thinking"] is True
    assert wrapped.requests[-1]["phase"] == "tavo_reflection"
    wrapped.complete("reflect")
    assert wire.bodies[-1]["temperature"] == 0.5 and wire.bodies[-1]["seed"] != body["seed"]
    assert wrapped.usage == {"prompt_tokens": 22, "completion_tokens": 14, "n_calls": 2}


def test_construction_contacts_no_endpoint_and_rotates_over_endpoints():
    lazy = ReflectionClient()
    assert lazy._clients == {} and json.dumps(lazy.usage)
    rotating = ReflectionClient(model="task-model", base_urls=("http://a.invalid/v1", "http://b.invalid/v1"))
    assert rotating.endpoint == "http://a.invalid/v1" and rotating._clients == {}
