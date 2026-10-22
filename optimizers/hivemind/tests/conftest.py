"""The scripted-model fixture shared by the HiveMind tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def hotpotqa_script(monkeypatch: pytest.MonkeyPatch):
    """Scripted chat model behind the real HotpotQA module adapters."""
    import optimizers.bridge.adapters.module_hotpotqa as module

    from .fakes import HOTPOTQA_WORKERS, Script

    script = Script(HOTPOTQA_WORKERS)
    monkeypatch.setattr(module, "default_chat_model", lambda seed=0, **_: script.model(seed))
    return script
