"""TAVO tests run offline with the Eq. 3 credit default (``TAVO_CREDIT`` unset)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _offline_default_credit(offline, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVO_CREDIT", raising=False)
