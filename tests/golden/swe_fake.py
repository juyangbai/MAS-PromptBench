"""Offline stand-ins for the SWE-bench network/container boundary.

* ``git clone <url> <dest>`` creates a small deterministic toy repository at
  ``dest`` instead of contacting GitHub; the instance base commit is treated
  as an alias of the toy HEAD (``checkout --detach <sha>`` / ``reset --hard
  <sha>`` are no-ops, ``rev-parse <sha>`` resolves to HEAD). Every other git
  command runs for real on the toy repository.
* ``singularity`` / ``apptainer`` invocations fail fast (exit 127) so the
  per-instance image pull/eval path is never attempted.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

TOY_FILES = {
    "README.md": "Toy repository used by the golden harness.\n",
    "buggy.py": "def buggy():\n    return 'replace me if needed'\n",
}
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
_COMMIT_REF_RE = re.compile(r"^[0-9a-f]{40}(\^\{commit\})?$")
_REAL_RUN = subprocess.run
_INSTALLED = False


def make_toy_repo(root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, text in TOY_FILES.items():
        (root / name).write_text(text)
    git = [
        "git",
        "-c",
        "user.name=golden",
        "-c",
        "user.email=golden@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "init.defaultBranch=main",
    ]
    env = {**os.environ, "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z"}
    _REAL_RUN(git + ["init", "-q"], cwd=root, check=True, env=env)
    _REAL_RUN(git + ["add", "."], cwd=root, check=True, env=env)
    _REAL_RUN(git + ["commit", "-q", "-m", "init"], cwd=root, check=True, env=env)
    return root


def _git_subcommand(argv: list[str]) -> tuple[str | None, list[str]]:
    i = 1
    while i < len(argv):
        token = argv[i]
        if token in ("-C", "-c", "--git-dir", "--work-tree"):
            i += 2
            continue
        if token.startswith("-"):
            i += 1
            continue
        return token, argv[i + 1 :]
    return None, []


def _completed(args, kwargs, returncode: int = 0, stderr: str = ""):
    textual = bool(kwargs.get("text") or kwargs.get("universal_newlines") or kwargs.get("encoding"))
    out = "" if textual else b""
    err = stderr if textual else stderr.encode()
    if kwargs.get("check") and returncode:
        raise subprocess.CalledProcessError(returncode, args, out, err)
    return subprocess.CompletedProcess(args, returncode, out, err)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    def run(args, *a, **kwargs):
        argv = [str(x) for x in args] if isinstance(args, (list, tuple)) else None
        if argv:
            program = os.path.basename(argv[0])
            if program == "git":
                sub, rest = _git_subcommand(argv)
                if sub == "clone":
                    dest = [x for x in rest if not x.startswith("-")][-1]
                    if not os.path.isabs(dest) and kwargs.get("cwd"):
                        dest = os.path.join(str(kwargs["cwd"]), dest)
                    make_toy_repo(Path(dest))
                    return _completed(args, kwargs)
                if sub == "checkout" and rest and _SHA_RE.match(rest[-1]):
                    return _completed(args, kwargs)
                if sub == "reset" and "--hard" in rest and rest and _SHA_RE.match(rest[-1]):
                    return _completed(args, kwargs)
                if sub == "rev-parse" and rest and _COMMIT_REF_RE.match(rest[-1]):
                    # The instance base commit is an alias of the toy HEAD.
                    head = argv[: argv.index("rev-parse")] + ["rev-parse", "--verify", "HEAD"]
                    return _REAL_RUN(head, *a, **kwargs)
            if program in ("singularity", "apptainer"):
                return _completed(args, kwargs, 127, "singularity is not available in the golden harness")
        return _REAL_RUN(args, *a, **kwargs)

    subprocess.run = run
