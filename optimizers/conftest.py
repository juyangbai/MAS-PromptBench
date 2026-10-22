"""Fixtures shared by the bridge, protocol and method tests.

* DSPy's on-disk cache lives in a temporary directory for the session;
* process-global protocol state (the runner's role-order cache and the fake
  adapter's script and call log) is reset around every test;
* ``offline`` refuses every socket connection and DNS lookup.
"""

from __future__ import annotations

import os
import shutil
import socket
import tempfile
from collections.abc import Iterator

import pytest

_CACHE = tempfile.mkdtemp(prefix="optimizer-tests-dspy-")
os.environ["DSPY_CACHEDIR"] = _CACHE


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    shutil.rmtree(_CACHE, ignore_errors=True)


@pytest.fixture(autouse=True)
def _protocol_state() -> Iterator[None]:
    from optimizers.protocol import runner
    from optimizers.protocol.tests.fakes import reset_fake

    role_orders = dict(runner._ROLE_ORDERS)
    reset_fake()
    yield
    reset_fake()
    runner._ROLE_ORDERS.clear()
    runner._ROLE_ORDERS.update(role_orders)


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that opens a network connection."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("offline test attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
