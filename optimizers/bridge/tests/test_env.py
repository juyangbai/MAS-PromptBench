"""The shared endpoint and trace-compaction variables: the names read and their defaults."""

from __future__ import annotations

import huggingface_hub
import pytest

from optimizers.bridge import lm, programs
from optimizers.bridge.datasets import bfcl
from optimizers.bridge.mipro_programs import MIPRO_VIEW
from optimizers.protocol.settings import ProtocolSettings

VARIABLES = (
    "TASK_ENDPOINTS",
    "VLLM_BASE_URL",
    "REFLECTION_MODEL_BASE_URL",
    "REFLECTION_COMPACT_DATASETS",
    "MIPRO_REFLECTION_COMPACT_DATASETS",
    "BFCL_CATEGORY",
)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _compacted(*datasets: str) -> dict[str, tuple[bool, ...]]:
    views = {"gepa": programs.GEPA_VIEW, "mipro": MIPRO_VIEW}
    return {name: tuple(programs._compact_enabled(d, view) for d in datasets) for name, view in views.items()}


def test_endpoint_defaults(env):
    settings = ProtocolSettings.from_env()
    assert settings.task_endpoints == ()
    assert settings.reflection_endpoint() == "http://localhost:8200/v1"
    assert lm.task_endpoints() == ("http://localhost:8000/v1",)


def test_endpoint_variables_are_read(env):
    env.setenv("VLLM_BASE_URL", "http://fallback.invalid/v1")
    assert ProtocolSettings.from_env().task_endpoints == ("http://fallback.invalid/v1",)
    env.setenv("TASK_ENDPOINTS", "http://a.invalid/v1, http://b.invalid/v1")
    env.setenv("REFLECTION_MODEL_BASE_URL", "http://reflection.invalid/v1")
    settings = ProtocolSettings.from_env()
    assert settings.task_endpoints == lm.task_endpoints() == ("http://a.invalid/v1", "http://b.invalid/v1")
    assert settings.reflection_endpoint() == "http://reflection.invalid/v1"
    env.setenv("REFLECTION_MODEL_BASE_URL", "")
    assert ProtocolSettings.from_env().reflection_endpoint() == "http://localhost:8200/v1"


def test_compaction_variables_are_read(env):
    assert _compacted("lcb", "math") == {"gepa": (True, False), "mipro": (True, False)}
    env.setenv("REFLECTION_COMPACT_DATASETS", "math")
    assert _compacted("lcb", "math") == {"gepa": (False, True), "mipro": (False, True)}
    env.setenv("MIPRO_REFLECTION_COMPACT_DATASETS", "*")
    assert _compacted("lcb", "math") == {"gepa": (False, True), "mipro": (True, True)}


def test_bfcl_loads_the_categories_its_fixed_splits_span(env, tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text("")
    requested: list[str] = []

    def download(repo_id: str, filename: str, **kwargs: object) -> str:
        requested.append(filename)
        return str(empty)

    env.setattr(huggingface_hub, "hf_hub_download", download)
    assert bfcl.load_all() == []
    assert requested[::2] == [f"BFCL_v3_{category}.json" for category in bfcl.AST_CATEGORIES]
    requested.clear()
    env.setenv("BFCL_CATEGORY", "multiple")
    bfcl.load_all()
    assert requested == ["BFCL_v3_multiple.json", "possible_answer/BFCL_v3_multiple.json"]
