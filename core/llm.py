"""Chat-client builders for the runners' OpenAI-compatible endpoint.

Each builder reproduces one client style of the runners with the decoding
protocol of :mod:`core.settings`: temperature, top_p, seed, max_tokens and the
vLLM ``extra_body`` (``repetition_penalty`` 1.05, ``chat_template_kwargs``
``enable_thinking`` False). ``model`` and ``base_url`` default to MODEL_ID and
VLLM_BASE_URL; runners pass their module-level ``MODEL_ID`` / ``VLLM_BASE_URL``
because the prompt optimizers patch those per rollout. Framework imports are
deferred to the builder that needs them.

Client constructions in the runners (communications/ reuses the topology
runners; team-size API-Bank / ToolHop runners reuse the single-topology ones):

1. LangChain ``ChatOpenAI`` -- LangGraph single, sequential, centralized and
   decentralized runners for gpqa, hotpotqa, math, lcb, apps, bfcl, swe, and the
   matching team-size runners: model, base_url, api_key, temperature, top_p,
   seed=REQUEST_SEED, max_tokens, extra_body. ``chat_openai()``
2. ``ChatOpenAI`` with a per-replica seed (the replica index) -- independent
   topology and team-size runners. ``chat_openai(seed=i)``
3. ``ChatOpenAI`` plus ``timeout=600.0, max_retries=5`` -- decentralized BFCL
   (topology and team-size runners). ``chat_openai(timeout=600.0, max_retries=5)``
4. ``ChatOpenAI`` without ``max_tokens`` -- single/swe.
   ``chat_openai(include_max_tokens=False)``
5. CrewAI ``LLM`` with ``model="openai/<MODEL_ID>"`` -- sequential/crewai apps,
   bfcl, hotpotqa, lcb, math, swe. ``crewai_llm()``
6. CrewAI ``LLM`` plus ``additional_drop_params=[]`` -- sequential/crewai/gpqa.
   ``crewai_llm(additional_drop_params=[])``
7. AutoGen ``OpenAIChatCompletionClient`` with a fixed ``model_info`` (family
   "qwen", function calling and JSON output, no vision or structured output)
   -- centralized/autogen apps, bfcl, gpqa, hotpotqa, lcb, math, swe.
   ``autogen_client()``
8. Raw ``openai.OpenAI`` with ``timeout=600.0, max_retries=5`` (an empty API
   key counts as unset) -- every ToolHop runner. ``openai_client()``
9. Raw ``openai.OpenAI`` with ``timeout=float($APIBANK_REQUEST_TIMEOUT or 60)``
   and ``max_retries=int($APIBANK_OPENAI_MAX_RETRIES or 0)`` -- every API-Bank
   runner. ``openai_client(timeout=..., max_retries=...)``
10. Request kwargs for the raw clients (ToolHop, API-Bank): model plus the
    decoding kwargs, with ``seed`` 0 when not given (REQUEST_SEED is not read).
    ToolHop's answer-finalization call overrides ``temperature`` to 0.0.
    ``completion_kwargs(seed)``
11. OpenAI Agents SDK -- decentralized/openai_agents, built in
    ``topologies.decentralized.openai_agents.agents_sdk_base``: ``AsyncOpenAI(
    base_url, api_key, max_retries=0)`` and ``ModelSettings(temperature, top_p,
    max_tokens, include_usage=True, extra_args={"seed": <per-peer seed>},
    extra_body={"chat_template_kwargs": {"enable_thinking": False}})`` (no
    repetition_penalty). It needs openai>=3 from an isolated install, so it is
    not built here; its values come from the same settings.
12. Framework templates outside the benchmark (``single/langgraph_base.py``,
    ``independent/langgraph_base.py``, ``centralized/autogen/autogen_base.py``):
    ``ChatOpenAI(model="gpt-4o-mini", temperature=0)``, ``AsyncOpenAI()``,
    ``OpenAIChatCompletionClient(model="gpt-4o-mini")``.
"""

from __future__ import annotations

from typing import Any

from core import settings

AUTOGEN_MODEL_INFO = {
    "vision": False,
    "function_calling": True,
    "json_output": True,
    "family": "qwen",
    "structured_output": False,
}


def _endpoint(model: str | None, base_url: str | None) -> tuple[str, str]:
    return (
        settings.model_id() if model is None else model,
        settings.base_url() if base_url is None else base_url,
    )


def chat_openai_kwargs(
    *,
    model: str | None = None,
    base_url: str | None = None,
    seed: int | None = None,
    include_max_tokens: bool = True,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> dict[str, Any]:
    """Keyword arguments of a runner's LangChain ``ChatOpenAI``."""
    model, base_url = _endpoint(model, base_url)
    decoding = settings.decoding(seed=seed, include_max_tokens=include_max_tokens)
    body = decoding.pop("extra_body")
    kwargs: dict[str, Any] = {"model": model, "base_url": base_url, "api_key": settings.api_key(), **decoding}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    kwargs["extra_body"] = body
    return kwargs


def chat_openai(
    *,
    model: str | None = None,
    base_url: str | None = None,
    seed: int | None = None,
    include_max_tokens: bool = True,
    timeout: float | None = None,
    max_retries: int | None = None,
):
    """LangChain ``ChatOpenAI`` built from :func:`chat_openai_kwargs`."""
    from langchain_openai import ChatOpenAI

    kwargs = chat_openai_kwargs(
        model=model,
        base_url=base_url,
        seed=seed,
        include_max_tokens=include_max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )
    return ChatOpenAI(**kwargs)


def crewai_llm_kwargs(
    *,
    model: str | None = None,
    base_url: str | None = None,
    seed: int | None = None,
    additional_drop_params: list[str] | None = None,
) -> dict[str, Any]:
    """Keyword arguments of a runner's CrewAI ``LLM`` (``model`` without the
    ``openai/`` provider prefix, which is added here)."""
    model, base_url = _endpoint(model, base_url)
    decoding = settings.decoding(seed=seed)
    body = decoding.pop("extra_body")
    kwargs: dict[str, Any] = {
        "model": f"openai/{model}",
        "base_url": base_url,
        "api_key": settings.api_key(),
        **decoding,
    }
    if additional_drop_params is not None:
        kwargs["additional_drop_params"] = list(additional_drop_params)
    kwargs["extra_body"] = body
    return kwargs


def crewai_llm(
    *,
    model: str | None = None,
    base_url: str | None = None,
    seed: int | None = None,
    additional_drop_params: list[str] | None = None,
):
    """CrewAI ``LLM`` built from :func:`crewai_llm_kwargs`."""
    from crewai import LLM

    kwargs = crewai_llm_kwargs(model=model, base_url=base_url, seed=seed, additional_drop_params=additional_drop_params)
    return LLM(**kwargs)


def autogen_client_kwargs(
    *,
    model: str | None = None,
    base_url: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Keyword arguments of a runner's AutoGen ``OpenAIChatCompletionClient``."""
    model, base_url = _endpoint(model, base_url)
    return {
        "model": model,
        "base_url": base_url,
        "api_key": settings.api_key(),
        "model_info": dict(AUTOGEN_MODEL_INFO),
        **settings.decoding(seed=seed),
    }


def autogen_client(*, model: str | None = None, base_url: str | None = None, seed: int | None = None):
    """AutoGen ``OpenAIChatCompletionClient`` built from :func:`autogen_client_kwargs`."""
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    return OpenAIChatCompletionClient(**autogen_client_kwargs(model=model, base_url=base_url, seed=seed))


def openai_client_kwargs(
    *,
    base_url: str | None = None,
    timeout: float = 600.0,
    max_retries: int = 5,
) -> dict[str, Any]:
    """Keyword arguments of a runner's raw ``openai.OpenAI`` client."""
    return {
        "base_url": settings.base_url() if base_url is None else base_url,
        "api_key": settings.api_key(blank_as_unset=True),
        "timeout": timeout,
        "max_retries": max_retries,
    }


def openai_client(*, base_url: str | None = None, timeout: float = 600.0, max_retries: int = 5):
    """Raw ``openai.OpenAI`` built from :func:`openai_client_kwargs`."""
    from openai import OpenAI

    return OpenAI(**openai_client_kwargs(base_url=base_url, timeout=timeout, max_retries=max_retries))


def completion_kwargs(seed: int | None = None, *, model: str | None = None) -> dict[str, Any]:
    """Per-request kwargs for ``client.chat.completions.create`` (raw clients)."""
    return {
        "model": settings.model_id() if model is None else model,
        "temperature": settings.temperature(),
        "top_p": settings.top_p(),
        "seed": 0 if seed is None else int(seed),
        "max_tokens": settings.max_tokens(),
        "extra_body": settings.extra_body(),
    }
