"""Context fit of reflection requests (no endpoint, no tokenizer)."""

from __future__ import annotations

import copy

from ...reflection import capture_reflection_responses
from .. import context_fit
from ..context_fit import PROMPT_FIT_SCHEMA, ContextFitLM, fit_to_context
from .fakes import FakeLM


def test_requests_that_fit_pass_through_untouched():
    inner = FakeLM()
    lm = fit_to_context(inner)
    assert fit_to_context(lm) is lm and lm.kwargs is inner.kwargs and lm.model == inner.model
    with capture_reflection_responses() as observed:
        lm("```\nWrite.\n```", max_tokens=48000)
    assert inner.calls[0]["prompt"] == "```\nWrite.\n```" and not observed
    assert isinstance(copy.deepcopy(lm), ContextFitLM)
    clone = lm.copy(temperature=0.5)
    assert isinstance(clone, ContextFitLM) and clone.kwargs["temperature"] == 0.5


def test_oversized_prompts_are_trimmed_before_the_call(monkeypatch):
    monkeypatch.setenv("REFLECTION_CONTEXT_LIMIT", "60000")
    inner = FakeLM()
    lm = ContextFitLM(inner)
    prompt = "```\nWrite.\n```\n" + "x" * 40000
    with capture_reflection_responses() as observed:
        lm(prompt, max_tokens=48000)
    sent = inner.calls[0]["prompt"]
    assert len(sent) < len(prompt) and "characters removed from the middle" in sent and sent.startswith("```")
    assert observed and observed[0]["schema"] == PROMPT_FIT_SCHEMA and observed[0]["pre_fit"] is True
    request = {"prompt": sent, "max_tokens": 48000}
    assert context_fit.count_prompt_tokens(request, None) + 48000 + 256 <= 60000
