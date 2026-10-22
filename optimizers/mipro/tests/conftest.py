"""MIPRO tests run offline and size reflection prompts without a tokenizer."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _offline_context_fit(offline, monkeypatch: pytest.MonkeyPatch) -> None:
    from optimizers.protocol.methods import context_fit

    # No local tokenizer lookups: the character estimate is used instead.
    monkeypatch.setattr(context_fit, "_token_counter", lambda model: None)
