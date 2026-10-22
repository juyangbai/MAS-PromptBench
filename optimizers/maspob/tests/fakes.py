"""Fake reflection LM, grid cells, protocol runners and embeddings for offline MASPOB tests."""

from __future__ import annotations

import hashlib
import importlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from optimizers.protocol.runner import ProtocolRunner
from optimizers.protocol.schema import CellSpec
from optimizers.protocol.tests import fakes as protocol_fakes
from optimizers.protocol.tests.fakes import FakeReflectionBackend


def _missing(*modules: str) -> list[str]:
    missing = []
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    return missing


GNN_MISSING = _missing("numpy", "torch", "torch_geometric")
requires_gnn = pytest.mark.skipif(
    bool(GNN_MISSING),
    reason=(
        f"MASPOB GNN tests need {', '.join(GNN_MISSING)} (pip install numpy torch torch_geometric, or set "
        "MASPOB_TEST_SITE_PACKAGES to a directory containing torch_geometric)"
    ),
)


class FakeReflectionLM(FakeReflectionBackend):
    """Offline reflection model: a deterministic, contract-preserving rewrite of the seed prompt.

    The output depends only on the meta-prompt. A quarter of the variants carry
    ``GOOD`` and a quarter ``BAD``, which the protocol ``FakeAdapter`` turns into
    all-correct / all-wrong writers. ``invalid`` returns text that breaks the
    prompt's executable interface; ``error`` raises.
    """

    def __init__(self, *, invalid: bool = False, error: bool = False) -> None:
        super().__init__()
        self.invalid = invalid
        self.error = error

    def respond(self, prompt: str, request: Mapping[str, Any]) -> str:
        if self.error:
            raise ConnectionError("offline reflection endpoint refused the connection")
        if self.invalid:
            return "This rewrite invents a {new_placeholder} and therefore breaks the interface contract."
        original = re.search(r"ORIGINAL instruction:\n---\n(.*)\n---", prompt, re.S).group(1).strip()
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        tag = {0: " GOOD", 1: " BAD"}.get(int(digest, 16) % 4, "")
        return f"```\nHere is the prompt:\n{original} Follow style variant {digest} and stay precise.{tag}\n```"


def grid_cell(*, topology: str = "sequential", task: str = "hotpotqa", seed: int = 0, budget: int = 600) -> CellSpec:
    """A table-6 MASPOB cell bound to the protocol's fake task data."""
    return CellSpec(
        method="maspob",
        task=task,
        topology=topology,
        framework="langgraph",
        optimizer_seed=seed,
        split_hash=protocol_fakes.fake_task_data().split_hash,
        budget=budget,
        source_tables=(6,),
    )


def protocol_runner(cell: CellSpec, **kwargs: Any) -> tuple[ProtocolRunner, Any, Any]:
    """The budget-owning protocol runner over the fake two-role adapter (planner, writer)."""
    return protocol_fakes.protocol_runner(cell, data=protocol_fakes.fake_task_data(), **kwargs)


def fake_embeddings(pool: Mapping[str, list[str]], roles: Sequence[str]) -> tuple[list[Any], dict[str, Any]]:
    """Deterministic 16-d unit vectors: two marker features plus text-hash features."""
    import torch

    tensors = []
    for role in roles:
        rows = []
        for text in pool[role]:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            features = [float("GOOD" in text), float("BAD" in text)] + [byte / 255.0 for byte in digest[:14]]
            rows.append(features)
        tensors.append(torch.nn.functional.normalize(torch.tensor(rows, dtype=torch.float32), dim=1))
    return tensors, {"backend": "offline-fake", "dim": 16}


__all__ = ["FakeReflectionLM", "GNN_MISSING", "fake_embeddings", "grid_cell", "protocol_runner", "requires_gnn"]
