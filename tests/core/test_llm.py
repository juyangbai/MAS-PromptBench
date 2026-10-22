"""``core.llm`` builders reproduce the runners' client kwargs exactly.

The ``runner_*`` helpers below are the expressions the runners evaluate today
(module-level ``MODEL_ID`` / ``VLLM_BASE_URL`` plus per-call env reads).
"""

import os
import sys
import types

import pytest

from core import llm


def runner_endpoint():
    return (
        os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B"),
        os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
    )


def runner_chat_openai(seed=None, timeout=False, max_tokens=True):
    model_id, base_url = runner_endpoint()
    kwargs = dict(
        model=model_id,
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        temperature=float(os.environ.get("TASK_MODEL_TEMPERATURE", "0.0")),
        top_p=float(os.environ.get("TASK_MODEL_TOP_P", "0.9")),
        seed=int(os.environ.get("REQUEST_SEED", "0")) if seed is None else seed,
    )
    if max_tokens:
        kwargs["max_tokens"] = int(os.environ.get("TASK_MODEL_MAX_TOKENS", "32768"))
    if timeout:
        kwargs.update(timeout=600.0, max_retries=5)
    kwargs["extra_body"] = {"repetition_penalty": 1.05, "chat_template_kwargs": {"enable_thinking": False}}
    return kwargs


def runner_crewai_llm(drop_params=False):
    model_id, base_url = runner_endpoint()
    kwargs = dict(
        model=f"openai/{model_id}",
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        temperature=float(os.environ.get("TASK_MODEL_TEMPERATURE", "0.0")),
        top_p=float(os.environ.get("TASK_MODEL_TOP_P", "0.9")),
        seed=int(os.environ.get("REQUEST_SEED", "0")),
        max_tokens=int(os.environ.get("TASK_MODEL_MAX_TOKENS", "32768")),
    )
    if drop_params:
        kwargs["additional_drop_params"] = []
    kwargs["extra_body"] = {"repetition_penalty": 1.05, "chat_template_kwargs": {"enable_thinking": False}}
    return kwargs


def runner_autogen_client():
    model_id, base_url = runner_endpoint()
    return dict(
        model=model_id,
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "qwen",
            "structured_output": False,
        },
        temperature=float(os.environ.get("TASK_MODEL_TEMPERATURE", "0.0")),
        top_p=float(os.environ.get("TASK_MODEL_TOP_P", "0.9")),
        seed=int(os.environ.get("REQUEST_SEED", "0")),
        max_tokens=int(os.environ.get("TASK_MODEL_MAX_TOKENS", "32768")),
        extra_body={"repetition_penalty": 1.05, "chat_template_kwargs": {"enable_thinking": False}},
    )


def runner_toolhop_client():
    return dict(
        base_url=runner_endpoint()[1],
        api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
        timeout=600.0,
        max_retries=5,
    )


def runner_apibank_client():
    return dict(
        base_url=runner_endpoint()[1],
        api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
        timeout=float(os.environ.get("APIBANK_REQUEST_TIMEOUT", "60")),
        max_retries=int(os.environ.get("APIBANK_OPENAI_MAX_RETRIES", "0")),
    )


def runner_completion_kwargs(seed=None):
    return {
        "model": runner_endpoint()[0],
        "temperature": float(os.environ.get("TASK_MODEL_TEMPERATURE", "0.0")),
        "top_p": float(os.environ.get("TASK_MODEL_TOP_P", "0.9")),
        "seed": 0 if seed is None else int(seed),
        "max_tokens": int(os.environ.get("TASK_MODEL_MAX_TOKENS", "32768")),
        "extra_body": {"repetition_penalty": 1.05, "chat_template_kwargs": {"enable_thinking": False}},
    }


@pytest.fixture(params=["clean_env", "custom_env", "blank_key"])
def env(request):
    if request.param == "blank_key":
        monkeypatch = request.getfixturevalue("custom_env")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        return monkeypatch
    return request.getfixturevalue(request.param)


def test_chat_openai_kwargs(env):
    assert llm.chat_openai_kwargs() == runner_chat_openai()
    assert llm.chat_openai_kwargs(seed=0) == runner_chat_openai(seed=0)
    assert llm.chat_openai_kwargs(seed=3) == runner_chat_openai(seed=3)
    assert llm.chat_openai_kwargs(timeout=600.0, max_retries=5) == runner_chat_openai(timeout=True)
    assert llm.chat_openai_kwargs(include_max_tokens=False) == runner_chat_openai(max_tokens=False)


def test_crewai_llm_kwargs(env):
    assert llm.crewai_llm_kwargs() == runner_crewai_llm()
    assert llm.crewai_llm_kwargs(additional_drop_params=[]) == runner_crewai_llm(drop_params=True)


def test_autogen_client_kwargs(env):
    assert llm.autogen_client_kwargs() == runner_autogen_client()


def test_openai_client_kwargs(env):
    assert llm.openai_client_kwargs() == runner_toolhop_client()
    env.setenv("APIBANK_REQUEST_TIMEOUT", "45")
    timeout = float(os.environ.get("APIBANK_REQUEST_TIMEOUT", "60"))
    max_retries = int(os.environ.get("APIBANK_OPENAI_MAX_RETRIES", "0"))
    assert llm.openai_client_kwargs(timeout=timeout, max_retries=max_retries) == runner_apibank_client()


def test_completion_kwargs(env):
    assert llm.completion_kwargs() == runner_completion_kwargs()
    assert llm.completion_kwargs(4) == runner_completion_kwargs(4)
    assert list(llm.completion_kwargs()) == list(runner_completion_kwargs())


def test_patched_module_constants_win(custom_env):
    """Runners pass their (optimizer-patched) MODEL_ID / VLLM_BASE_URL."""
    kwargs = llm.chat_openai_kwargs(model="patched-model", base_url="http://patched/v1")
    assert (kwargs["model"], kwargs["base_url"]) == ("patched-model", "http://patched/v1")
    assert llm.crewai_llm_kwargs(model="patched-model")["model"] == "openai/patched-model"
    assert llm.autogen_client_kwargs(base_url="http://patched/v1")["base_url"] == "http://patched/v1"
    assert llm.openai_client_kwargs(base_url="http://patched/v1")["base_url"] == "http://patched/v1"
    assert llm.completion_kwargs(model="patched-model")["model"] == "patched-model"


def test_kwargs_are_fresh(clean_env):
    first = llm.autogen_client_kwargs()
    first["model_info"]["family"] = "other"
    first["extra_body"]["repetition_penalty"] = 1.0
    assert llm.autogen_client_kwargs() == runner_autogen_client()


class _Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _fake_module(monkeypatch, name, **attrs):
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        monkeypatch.setitem(sys.modules, ".".join(parts[:i]), types.ModuleType(".".join(parts[:i])))
    for key, value in attrs.items():
        setattr(sys.modules[name], key, value)


def test_builders_pass_kwargs_to_the_framework_class(custom_env):
    _fake_module(custom_env, "langchain_openai", ChatOpenAI=_Recorder)
    _fake_module(custom_env, "crewai", LLM=_Recorder)
    _fake_module(custom_env, "autogen_ext.models.openai", OpenAIChatCompletionClient=_Recorder)
    _fake_module(custom_env, "openai", OpenAI=_Recorder)
    assert llm.chat_openai(seed=2, timeout=600.0, max_retries=5).kwargs == runner_chat_openai(seed=2, timeout=True)
    assert llm.crewai_llm(additional_drop_params=[]).kwargs == runner_crewai_llm(drop_params=True)
    assert llm.autogen_client().kwargs == runner_autogen_client()
    assert llm.openai_client().kwargs == runner_toolhop_client()


def test_real_langchain_and_openai_clients(custom_env):
    langchain_openai = pytest.importorskip("langchain_openai")
    openai = pytest.importorskip("openai")
    chat = llm.chat_openai()
    assert isinstance(chat, langchain_openai.ChatOpenAI)
    assert (chat.model_name, chat.temperature, chat.top_p, chat.seed, chat.max_tokens) == (
        "meta-llama/Llama-3.1-8B-Instruct",
        0.2,
        0.95,
        7,
        1024,
    )
    assert chat.extra_body == {"repetition_penalty": 1.05, "chat_template_kwargs": {"enable_thinking": False}}
    client = llm.openai_client()
    assert isinstance(client, openai.OpenAI)
    assert (client.api_key, client.timeout, client.max_retries) == ("key-123", 600.0, 5)
    assert str(client.base_url).rstrip("/") == "http://127.0.0.1:9999/v1"
