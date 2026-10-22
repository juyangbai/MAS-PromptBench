"""The OpenAI Agents SDK directory is first on the path for the whole process."""

from __future__ import annotations

import os
import site
import subprocess
import sys
import textwrap

import pytest

from core.paths import REPO_ROOT
from topologies.decentralized.openai_agents import agents_sdk_base as base

SITE_PACKAGES = site.getsitepackages()[0]


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    """An SDK directory providing ``typing_extensions`` (imported by the main
    environment in this process) and a module nobody imported."""
    root = tmp_path / "sdk"
    root.mkdir()
    (root / "typing_extensions.py").write_text('ORIGIN = "sdk"\n')
    (root / "sdk_probe_module.py").write_text("")
    monkeypatch.setenv("OPENAI_AGENTS_PATH", str(root))
    monkeypatch.setattr(base, "_SDK", None)
    monkeypatch.setattr(base, "_import_sdk", lambda: "sdk entry points")
    return root.resolve()


def test_sdk_dir_is_the_configured_directory_when_it_exists(sdk, tmp_path, monkeypatch):
    assert base.sdk_dir() == sdk
    monkeypatch.setenv("OPENAI_AGENTS_PATH", str(tmp_path / "missing"))
    assert base.sdk_dir() is None
    monkeypatch.delenv("OPENAI_AGENTS_PATH")
    assert base.sdk_dir() == (base.VENDOR_SDK_DIR.resolve() if base.VENDOR_SDK_DIR.is_dir() else None)


def test_first_on_path_means_ahead_of_every_site_packages_directory(sdk, monkeypatch):
    monkeypatch.setattr(sys, "path", [str(REPO_ROOT), str(sdk), SITE_PACKAGES])
    assert base.sdk_first_on_path(sdk)
    monkeypatch.setattr(sys, "path", [str(REPO_ROOT), SITE_PACKAGES, str(sdk)])
    assert not base.sdk_first_on_path(sdk)
    monkeypatch.setattr(sys, "path", [str(REPO_ROOT), SITE_PACKAGES])
    assert not base.sdk_first_on_path(sdk)


def test_load_refuses_a_directory_that_is_not_first(sdk, monkeypatch):
    monkeypatch.setattr(sys, "path", [str(REPO_ROOT), SITE_PACKAGES, str(sdk)])
    with pytest.raises(RuntimeError, match="is not first on PYTHONPATH"):
        base.load_agents_sdk()
    assert base._SDK is None


def test_load_refuses_packages_already_imported_from_elsewhere(sdk, monkeypatch):
    # The main environment's copy.
    import typing_extensions  # noqa: F401

    monkeypatch.setattr(sys, "path", [str(REPO_ROOT), str(sdk), SITE_PACKAGES])
    with pytest.raises(RuntimeError, match="already imported typing_extensions from elsewhere"):
        base.load_agents_sdk()


def test_load_imports_once_when_the_directory_serves_the_process(sdk, monkeypatch):
    (sdk / "typing_extensions.py").unlink()
    monkeypatch.setattr(sys, "path", [str(REPO_ROOT), str(sdk), SITE_PACKAGES])
    path_before = list(sys.path)
    assert base.load_agents_sdk() == "sdk entry points"
    monkeypatch.setattr(base, "_import_sdk", lambda: pytest.fail("imported twice"))
    assert base.load_agents_sdk() == "sdk entry points"
    assert sys.path == path_before


def test_failed_import_names_the_reason(sdk, tmp_path, monkeypatch):
    def broken():
        raise ImportError("cannot import name 'sentinel'")

    monkeypatch.setattr(base, "_import_sdk", broken)
    (sdk / "typing_extensions.py").unlink()
    monkeypatch.setattr(sys, "path", [str(REPO_ROOT), str(sdk), SITE_PACKAGES])
    with pytest.raises(RuntimeError, match="does not import .*sentinel.*\n.*requirements-openai-agents.txt"):
        base.load_agents_sdk()
    monkeypatch.setenv("OPENAI_AGENTS_PATH", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError) as unavailable:
        base.load_agents_sdk()
    assert str(unavailable.value) == base._INSTALL_HINT


_CHILD = textwrap.dedent(
    """
    import typing_extensions  # imported before the SDK directory is in place

    from topologies.decentralized.openai_agents import agents_sdk_base as base

    base.reexec_with_sdk_first()
    print(getattr(typing_extensions, "ORIGIN", "main"), base.sdk_first_on_path(base.sdk_dir()))
    """
)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ((), "sdk True"),
        # PYTHONPATH is ignored: one restart, then the process reports it instead of looping.
        (("-E",), "main False"),
    ],
)
def test_reexec_restarts_the_process_with_the_directory_first(sdk, flags, expected):
    environment = {
        **os.environ,
        "OPENAI_AGENTS_PATH": str(sdk),
        "PYTHONPATH": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-B", *flags, "-c", _CHILD],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == expected.split()
