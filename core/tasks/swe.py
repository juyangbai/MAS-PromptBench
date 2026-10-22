"""SWE-bench Verified: instances, repository tools, issue briefs, patches, scoring and batch records.

An agent edits a clone of the instance repository at its base commit; the
submitted patch is ``git diff HEAD`` of that checkout. The instance is resolved
when every FAIL_TO_PASS and PASS_TO_PASS test passes after the patch is applied
inside the instance's SWE-bench image (swebench.harness.grading,
ResolvedStatus.FULL). The images are Singularity SIFs pulled from
docker://swebench on first use.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextvars import ContextVar, Token
from functools import partial
from pathlib import Path

from core import batch, cli, swe_sandbox, voting
from core.paths import RESULTS_DIR

logger = logging.getLogger(__name__)

DATASET = "swe"
HF_DATASET = "princeton-nlp/SWE-bench_Verified"
SOURCE = HF_DATASET

SHELL_TIMEOUT_S = 60
READ_CHAR_BUDGET = 20_000  # file_read output cap
MAX_SEARCH_MATCHES = 200  # Agents SDK search_repo cap
# Caps on the issue text so the first message fits the model context.
PROBLEM_CHAR_BUDGET = int(os.environ.get("SWE_PROBLEM_CHAR_BUDGET", "16000"))
HINTS_CHAR_BUDGET = int(os.environ.get("SWE_HINTS_CHAR_BUDGET", "4000"))

SIF_DIR = Path(os.environ.get("SWE_SIF_DIR", f"{Path.home()}/containers/swe")).resolve()
SWEBENCH_IMAGE = "docker://swebench/sweb.eval.x86_64.{tag}:latest"
SIF_PULL_TIMEOUT_S = 900
SIF_EVAL_TIMEOUT_S = 1800
PASSING_VERDICTS = {"PASSED", "XFAIL"}


# Data
def load_instances(
    subset: str = "test",
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
) -> list[dict]:
    """Rows of SWE-bench Verified (``subset`` split), filtered to ``only``, then ``offset`` / ``limit``."""
    from datasets import load_dataset

    rows = list(load_dataset(HF_DATASET, split=subset))
    if only:
        wanted = set(only)
        rows = [r for r in rows if r["instance_id"] in wanted]
    rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows


def instance_tests(instance: dict) -> tuple[list[str], list[str]]:
    """The instance's FAIL_TO_PASS and PASS_TO_PASS test ids (decoded when stored as JSON)."""
    f2p = instance["FAIL_TO_PASS"]
    p2p = instance["PASS_TO_PASS"]
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)
    return f2p, p2p


# Command line
DEFAULT_LIMIT = 2
EPILOG = (
    "examples:\n"
    "  %(prog)s --limit 1\n"
    "  %(prog)s --only astropy__astropy-12907\n"
    "  %(prog)s --limit 50 --eval none   # collect the patches only"
)


def add_arguments(parser: argparse.ArgumentParser, *, eval_modes: tuple[str, ...], skip_eval: bool = False) -> None:
    """SWE options of a runner command line; ``eval_modes[0]`` is the default."""
    parser.add_argument("--subset", default="test", help="HF split of SWE-bench Verified (default: test)")
    parser.add_argument(
        "--workdir-root", type=Path, default=None, metavar="DIR", help="root of the per-instance repository clones"
    )
    parser.add_argument(
        "--eval",
        dest="eval_mode",
        default=eval_modes[0],
        choices=list(eval_modes),
        help=f"how to score the patches (default: {eval_modes[0]})",
    )
    if skip_eval:
        parser.add_argument(
            "--skip-eval", dest="eval_mode", action="store_const", const="none", help="alias for --eval none"
        )
    parser.add_argument("--keep-workdirs", action="store_true", help="keep the repository clones after scoring")


def cli_main(
    argv: list[str] | None,
    *,
    description: str,
    run_batch: Callable[..., object],
    default_out_dir: Path,
    eval_modes: tuple[str, ...],
    skip_eval: bool = False,
    preflight: Callable[[], None] | None = None,
) -> int:
    """A runner's command line (:func:`core.cli.main`); the predictions default to ``default_out_dir``."""
    return cli.main(
        argv,
        description=description,
        load_instances=load_instances,
        run_batch=run_batch,
        source=SOURCE,
        predictions=default_out_dir / "predictions.jsonl",
        add_arguments=partial(add_arguments, eval_modes=eval_modes, skip_eval=skip_eval),
        default_limit=DEFAULT_LIMIT,
        epilog=EPILOG,
        preflight=preflight,
    )


# Checkouts
def env_repo_dir() -> Path:
    """``$SWE_REPO_DIR`` (default: the working directory), resolved."""
    return Path(os.environ.get("SWE_REPO_DIR", ".")).resolve()


class Workdir:
    """The repository checkout a runner's tools act on.

    A checkout is bound per context (``bind``), so concurrent instances or
    replicas each see their own; ``default`` applies where none is bound and
    ``get`` raises when there is neither. With ``sticky`` a binding also
    becomes the default, for frameworks that run tools in threads without
    copying the context. ``explain_escapes`` words the error of a path outside
    the checkout (else the ``Path.relative_to`` error is raised as is).
    """

    def __init__(self, default: Path | None = None, *, sticky: bool = False, explain_escapes: bool = True):
        self._bound: ContextVar[Path | None] = ContextVar("swe_workdir", default=None)
        self.default = default
        self.sticky = sticky
        self.explain_escapes = explain_escapes

    def get(self) -> Path:
        path = self._bound.get() or self.default
        if path is None:
            raise RuntimeError("REPO_DIR not set in this context; set it before invoking the agent")
        return path

    def bind(self, path: Path) -> Token:
        if self.sticky:
            self.default = path
        return self._bound.set(path)

    def reset(self, token: Token) -> None:
        self._bound.reset(token)

    def resolve(self, path: str) -> Path:
        """``path`` (relative to the checkout, or absolute) resolved; ValueError if it leaves the checkout."""
        repo = self.get()
        candidate = (repo / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if not self.explain_escapes:
            candidate.relative_to(repo)
            return candidate
        try:
            candidate.relative_to(repo)
        except ValueError as e:
            raise ValueError(f"path {path!r} escapes repo workdir {repo}") from e
        return candidate


# Tools. A tool's docstring is the description the model sees, so every
# wording the runners use is kept verbatim, indentation included.
FILE_READ_DOC = (
    "Read a file from the repository working directory.\n\n"
    "    Path is interpreted relative to the repo root. `offset` is a 0-indexed\n"
    "    starting line number; `limit` caps the number of lines returned.\n"
    "    Output truncated to ~20000 characters as a safety net.\n"
    "    "
)
FILE_READ_DOC_SEARCH_FIRST = (
    "Read a file from the repository working directory.\n\n"
    "    Path is interpreted relative to the repo root. `offset` is a 0-indexed\n"
    "    starting line number; `limit` caps the number of lines returned. Output\n"
    "    is also truncated to ~20000 characters as a safety net; for large files,\n"
    "    use search_repo first to locate the line range, then pass offset/limit.\n"
    "    "
)
FILE_READ_DOC_SHORT = (
    "Read a file from the repository working directory.\n\n"
    "    `offset` is a 0-indexed starting line; `limit` caps the number of\n"
    "    lines. Output is truncated to ~20000 characters as a safety net.\n"
    "    "
)
FILE_WRITE_DOC = "Overwrite (or create) a file in the repository with `content`."
FILE_WRITE_DOC_PARENTS = (
    "Overwrite (or create) a file in the repository with `content`.\n\n    Creates parent directories if needed.\n    "
)
LIST_DIR_DOC = "List entries in a directory under the repository workdir."
LIST_DIR_DOC_REPO = "List entries in a directory under the repo workdir."
SEARCH_REPO_DOC = "grep-style regex search under the repository workdir."
SEARCH_REPO_DOC_REPO = "grep-style regex search under the repo workdir."
SEARCH_REPO_DOC_GREP = (
    "grep-style search under the repository workdir.\n\n"
    "    Uses `grep -rn` with fixed-string matching disabled (so `pattern` is a\n"
    '    regex). Returns up to `max_matches` lines as "file:line:content".\n'
    "    "
)
SHELL_EXEC_DOC = "Run a shell command in the repository working directory."
SHELL_EXEC_DOC_REPO = "Run a shell command in the repo workdir (default 60s timeout)."
SHELL_EXEC_DOC_TIMEOUT = (
    "Run a shell command in the repository working directory.\n\n"
    "    Captures stdout, stderr, and exit code. Timeout defaults to 60 s.\n"
    "    "
)
SHELL_EXEC_DOC_TESTER = (
    "Run a shell command in the repository working directory.\n\n"
    "    Captures stdout, stderr, and exit code. Timeout defaults to 60 s.\n"
    '    Useful for the Tester stage to run `python -c "import module"` sanity\n'
    "    checks or `git diff` to inspect the patch.\n"
    "    "
)
STR_REPLACE_DOC_NARROW = (
    "Replace EXACTLY ONE occurrence of `old` with `new` in `path`.\n\n"
    "    Narrow-anchor edit tool: `old` must match exactly once in the file,\n"
    "    otherwise the tool errors (so no accidental catastrophic overwrite).\n"
    "    Include enough surrounding context in `old` to make the match\n"
    "    unique.\n"
    "    "
)
STR_REPLACE_DOC_UNIQUE = (
    "Replace EXACTLY ONE occurrence of `old` with `new` in `path`.\n\n"
    "    Include enough surrounding context in `old` to uniquely identify the\n"
    "    location. Returns an error if `old` is not found or appears more\n"
    "    than once.\n"
    "    "
)
_STR_REPLACE_DOC_TARGETED = (
    "Replace EXACTLY ONE occurrence of `old` with `new` in the file at `path`.\n\n"
    "    This is a targeted-edit tool: it only rewrites the matched region,\n"
    "    leaving the rest of the file untouched. Use it instead of `file_write`\n"
    "    for bug fixes {dash} `file_write` overwrites the ENTIRE file, which is\n"
    "    almost never what you want when you're changing a few lines.\n\n"
    "    Include enough surrounding context in `old` to uniquely identify the\n"
    "    location. If `old` is not found, returns an error. If `old` appears\n"
    "    more than once, returns an error listing the match count {dash} add more\n"
    "    context to disambiguate.\n\n"
    "    Args:\n"
    "        path: file path relative to the repository working directory.\n"
    "        old:  the exact substring to replace (with enough surrounding\n"
    "              lines to be unique).\n"
    '        new:  the replacement substring. Use "" to delete `old`.\n\n'
    "    Returns a success message with a short preview, or an ERROR string.\n"
    "    "
)
STR_REPLACE_DOC_TARGETED = _STR_REPLACE_DOC_TARGETED.format(dash="--")  # sequential LangGraph
STR_REPLACE_DOC_TARGETED_EM_DASH = _STR_REPLACE_DOC_TARGETED.format(dash="—")  # sequential CrewAI

# str_replace replies when `old` is missing or not unique ({path}, {count}).
NOT_FOUND = (
    "ERROR: `old` not found in {path}. Check for whitespace, line endings, or tab/space mismatch. "
    "Call file_read to inspect."
)
NOT_FOUND_READ_FIRST = (
    "ERROR: `old` not found in {path}. Check for whitespace, line endings, or tab/space mismatch. "
    "If you need to inspect the file, call file_read first."
)
NOT_FOUND_TERSE = (
    "ERROR: `old` not found in {path}. Check whitespace / line endings / tab-space mismatch. Call file_read to inspect."
)
AMBIGUOUS = (
    "ERROR: `old` appears {count} times in {path}; edit would be ambiguous. "
    "Add more surrounding lines to `old` so the match is unique."
)
AMBIGUOUS_TERSE = "ERROR: `old` appears {count} times in {path}; add more surrounding context to make the match unique."


def make_file_read(workdir: Workdir, doc: str) -> Callable[..., str]:
    """``file_read(path, offset, limit)`` on ``workdir``, documented by ``doc``."""

    def file_read(path: str, offset: int = 0, limit: int | None = None) -> str:
        try:
            content = workdir.resolve(path).read_text(errors="replace")
        except Exception as e:
            return f"ERROR: {e}"
        return _read_window(content, offset, limit)

    file_read.__doc__ = doc
    return file_read


def _read_window(content: str, offset: int, limit: int | None) -> str:
    """Lines ``offset .. offset+limit`` of ``content`` under a range header, capped at READ_CHAR_BUDGET chars."""
    total = content.count("\n") + 1
    if offset or limit is not None:
        lines = content.splitlines(keepends=True)
        start = max(0, int(offset))
        end = start + int(limit) if limit is not None else len(lines)
        content = f"[lines {start + 1}-{min(end, len(lines))} of {total}]\n" + "".join(lines[start:end])
    if len(content) > READ_CHAR_BUDGET:
        return content[:READ_CHAR_BUDGET] + f"\n... [truncated, total {len(content)} chars]"
    return content


def make_file_write(workdir: Workdir, doc: str) -> Callable[..., str]:
    """``file_write(path, content)`` on ``workdir`` (creates parent directories), documented by ``doc``."""

    def file_write(path: str, content: str) -> str:
        try:
            target = workdir.resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        except Exception as e:
            return f"ERROR: {e}"
        return f"wrote {len(content)} chars to {path}"

    file_write.__doc__ = doc
    return file_write


def make_str_replace(
    workdir: Workdir,
    doc: str,
    *,
    not_found: str = NOT_FOUND,
    ambiguous: str = AMBIGUOUS,
    preview: bool = False,
) -> Callable[..., str]:
    """``str_replace(path, old, new)`` on ``workdir``: replaces the single occurrence of ``old``.

    ``not_found`` / ``ambiguous`` are the error replies; with ``preview`` the
    success reply quotes the start of ``old`` and ``new``.
    """

    def str_replace(path: str, old: str, new: str) -> str:
        try:
            target = workdir.resolve(path)
            content = target.read_text(errors="replace")
        except Exception as e:
            return f"ERROR: {e}"
        count = content.count(old)
        if count == 0:
            return not_found.format(path=path)
        if count > 1:
            return ambiguous.format(path=path, count=count)
        try:
            target.write_text(content.replace(old, new, 1))
        except Exception as e:
            return f"ERROR: {e}"
        if not preview:
            return f"replaced 1 occurrence in {path}"
        return f"replaced 1 occurrence in {path}\n  old: {_clip(old)!r}\n  new: {_clip(new)!r}"

    str_replace.__doc__ = doc
    return str_replace


def _clip(text: str) -> str:
    return text if len(text) < 120 else text[:120] + "..."


def make_list_dir(workdir: Workdir, doc: str) -> Callable[..., str]:
    """``list_dir(path)`` on ``workdir``: directories first, then files, by name."""

    def list_dir(path: str = ".") -> str:
        try:
            target = workdir.resolve(path)
            if not target.is_dir():
                return f"ERROR: {path} is not a directory"
            repo = workdir.get()
            entries = sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name))
            return "\n".join(f"{'d' if e.is_dir() else 'f'}  {e.relative_to(repo)}" for e in entries)
        except Exception as e:
            return f"ERROR: {e}"

    list_dir.__doc__ = doc
    return list_dir


def make_search_repo(workdir: Workdir, doc: str, *, terse: bool = False) -> Callable[..., str]:
    """``search_repo(pattern, path, max_matches)`` on ``workdir``: ``grep -rn -E``, paths relative to the checkout.

    ``terse`` selects the shorter error and no-match replies.
    """

    def search_repo(pattern: str, path: str = ".", max_matches: int = 50) -> str:
        try:
            target = workdir.resolve(path)
        except Exception as e:
            return f"ERROR: {e}"
        try:
            result = subprocess.run(
                ["grep", "-rn", "-E", pattern, str(target)], capture_output=True, text=True, timeout=SHELL_TIMEOUT_S
            )
        except Exception as e:
            return f"ERROR: {e}"
        if result.returncode not in (0, 1):
            return f"ERROR: grep {'exit' if terse else 'exited'} {result.returncode}\n{result.stderr}"
        lines = result.stdout.splitlines()
        if not lines:
            return f"[no matches for {pattern!r}]" if terse else f"[no matches for {pattern!r} in {path}]"
        if len(lines) > max_matches:
            lines = lines[:max_matches] + [f"... [+{len(lines) - max_matches} more]"]
        prefix = str(workdir.get()) + "/"
        return "\n".join(line.replace(prefix, "") for line in lines)

    search_repo.__doc__ = doc
    return search_repo


def make_shell_exec(workdir: Workdir, doc: str) -> Callable[..., str]:
    """``shell_exec(command, timeout_s)`` in ``workdir``, sandboxed by :mod:`core.swe_sandbox`."""

    def shell_exec(command: str, timeout_s: int = SHELL_TIMEOUT_S) -> str:
        return swe_sandbox.shell_exec(workdir.get(), command, timeout_s)

    shell_exec.__doc__ = doc
    return shell_exec


# Issue briefs
FIX_NOTE = (
    "Use the available tools (file_read, file_write, list_dir, search_repo, "
    "shell_exec) to investigate the codebase and apply a fix. Modify files "
    "in place with file_write.\n"
    "\n"
    "Do NOT try to run the repo's own code or tests here — this workdir is "
    "only a source checkout; C extensions and test deps are NOT installed. "
    "Tests will be run separately in a prepared environment. Focus on "
    "reading source files to understand the bug, then write a fix.\n"
    "\n"
    "When done, do NOT hand-write a diff — the harness computes the patch "
    "from the git state of the workdir."
)
_NO_TESTS_NOTE = (
    "Do NOT try to run the repo's own tests here {dash} this workdir is only a "
    "source checkout; C extensions and test deps are NOT installed. Tests "
    "will be run separately in a prepared environment."
)
NO_TESTS_NOTE = _NO_TESTS_NOTE.format(dash="—")
NO_TESTS_NOTE_ASCII = _NO_TESTS_NOTE.format(dash="--")  # sequential LangGraph
PEER_CHECKOUT = "The repository is checked out on the failing commit at your peer-local workdir."


def checked_out_at(repo_dir: Path | str) -> str:
    return f"The repository is checked out at {repo_dir} on the failing commit."


def truncate(text: str, cap: int, label: str) -> str:
    """``text`` cut to ``cap`` characters with a visible truncation note."""
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n... [truncated {label}: {len(text)} -> {cap} chars]"


def issue_brief(
    problem_statement: str,
    instance_id: str | None = None,
    hints_text: str | None = None,
    *,
    checkout: str,
    note: str,
) -> str:
    """The user message of one instance: id, ``checkout`` line, issue, hints and the closing ``note``."""
    parts = []
    if instance_id:
        parts.append(f"INSTANCE: {instance_id}")
    parts.append(checkout)
    parts.append("ISSUE:\n" + truncate(problem_statement.strip(), PROBLEM_CHAR_BUDGET, "problem_statement"))
    if hints_text:
        parts.append("HINTS (from maintainers):\n" + truncate(hints_text.strip(), HINTS_CHAR_BUDGET, "hints_text"))
    parts.append(note)
    return "\n\n".join(parts)


# Centralized manager nudge
TERMINATE_NUDGE = (
    "\n\nWhen you have a patch applied to the repo and are satisfied "
    "with it, emit a short summary of the changes and immediately "
    "follow with the literal string TERMINATE on its own line so the "
    "group-chat knows to stop. The final patch is extracted from the "
    "workdir via `git diff HEAD` — you do NOT need to re-print it."
)

# Decentralized debate (LangGraph)
PEER_SUMMARIES_INTRO = (
    "These are the final summaries from other peer agents in the "
    "previous round (each peer worked in its OWN repo clone — their "
    "file changes are NOT visible in yours):"
)
PEER_REVISE = (
    "\nCompare their approach with yours. If a peer found a better fix "
    "location or caught a regression, REVISE your edits in your own "
    "workdir. Use str_replace to apply the revised fix; use shell_exec "
    "+ `git diff HEAD` to inspect the current state of your workdir.\n\n"
    "Original brief:\n"
)
RETRY_NUDGE = (
    "The previous tool call produced an invalid request. "
    "Continue with a different approach (do not repeat the same call)."
)


def peer_message(others: list[str], brief: str) -> str:
    """What a peer reads from the second round on: the other peers' final summaries, then ``brief``."""
    body = [PEER_SUMMARIES_INTRO]
    body.extend(f"\nPeer {i + 1}:\n```\n{text}\n```" for i, text in enumerate(others))
    body.append(PEER_REVISE + brief)
    return "\n".join(body)


# Patches and checkouts
def compute_patch(repo_dir: Path) -> str:
    """``git diff HEAD`` of ``repo_dir`` (an ``ERROR: ...`` string if git fails)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "diff", "HEAD"], capture_output=True, text=True, timeout=SHELL_TIMEOUT_S
        )
    except Exception as e:
        return f"ERROR: {e}"
    if result.returncode != 0:
        return f"ERROR: git diff exited {result.returncode}\n{result.stderr}"
    return result.stdout


def clone_and_checkout(repo: str, base_commit: str, workdir: Path) -> str:
    """Clone https://github.com/{repo} into ``workdir`` (replacing it) detached at ``base_commit``; "" or the error."""
    url = f"https://github.com/{repo}.git"
    workdir.parent.mkdir(parents=True, exist_ok=True)
    if workdir.exists():
        shutil.rmtree(workdir)
    r = subprocess.run(
        ["git", "clone", "--quiet", "--no-tags", url, str(workdir)], capture_output=True, text=True, timeout=900
    )
    if r.returncode != 0:
        return f"clone failed: {r.stderr.strip()}"
    r = subprocess.run(
        ["git", "-C", str(workdir), "checkout", "--quiet", "--detach", base_commit],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        return f"checkout failed: {r.stderr.strip()}"
    subprocess.run(
        ["git", "-C", str(workdir), "config", "advice.detachedHead", "false"], capture_output=True, text=True
    )
    return ""


def _git(repo: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=timeout)


def local_clone(seed: Path, target: Path, base_commit: str) -> str:
    """An independent clone of ``seed`` (objects copied, no hardlinks) detached at ``base_commit``; "" or the error."""
    try:
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["git", "clone", "--quiet", "--local", "--no-hardlinks", "--no-checkout", str(seed), str(target)],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if r.returncode != 0:
            return f"local clone failed: {r.stderr.strip()}"
        r = _git(target, "checkout", "--quiet", "--detach", base_commit, timeout=300)
        if r.returncode != 0:
            return f"checkout failed: {r.stderr.strip()}"
        _git(target, "config", "advice.detachedHead", "false")
        _git(target, "config", "core.fileMode", "false")
        head = _git(target, "rev-parse", "--verify", "HEAD").stdout.strip()
        expected = _git(target, "rev-parse", "--verify", f"{base_commit}^{{commit}}").stdout.strip()
        if not head or head != expected:
            return f"worktree HEAD {head or '?'} does not match base commit {base_commit}"
    except Exception as e:
        return f"local clone failed: {type(e).__name__}: {e}"
    return ""


def prepare_peer_worktrees(
    instance: dict, root: Path, n_peers: int, seed: Path | None = None
) -> tuple[str, str, list[Path]]:
    """Clone the repository once into ``root/seed`` (unless ``seed`` is given), then one local clone per peer.

    Returns ``(stage, error, worktrees)`` with the peers' worktrees at
    ``root/peer_<k>``; ``error`` is "" on success.
    """
    root.mkdir(parents=True, exist_ok=True)
    if seed is None:
        seed = root / "seed"
        err = clone_and_checkout(instance["repo"], instance["base_commit"], seed)
        if err:
            return "clone/seed", err, []
    worktrees: list[Path] = []
    for i in range(n_peers):
        target = root / f"peer_{i}"
        err = local_clone(seed, target, instance["base_commit"])
        if err:
            return f"clone/peer_{i}", err, []
        worktrees.append(target)
    return "", "", worktrees


def git_diff(workdir: Path) -> str:
    """``git diff HEAD`` of ``workdir`` (no external diff driver); RuntimeError if git fails."""
    r = _git(workdir, "diff", "--no-ext-diff", "HEAD", timeout=SHELL_TIMEOUT_S)
    if r.returncode != 0:
        raise RuntimeError(f"git diff exited {r.returncode}: {r.stderr.strip()[-500:]}")
    return r.stdout


class PeerWorktrees:
    """Per-peer worktrees and the SWE tools of the Agents SDK debate (peer ``peer_<k>`` acts on worktree k).

    Paths resolve inside the calling peer's own worktree (escapes and Git
    metadata are refused); shell commands run through :mod:`core.swe_sandbox`
    with only that worktree mounted.
    """

    def __init__(self, worktrees: list[Path]):
        self._repos = {f"peer_{i}": Path(p).resolve() for i, p in enumerate(worktrees)}

    def repo(self, peer: str) -> Path:
        if not re.fullmatch(r"peer_[0-9]+", peer or "") or peer not in self._repos:
            raise ValueError(f"invalid SWE peer identity: {peer!r}")
        return self._repos[peer]

    @staticmethod
    def _path(repo: Path, path: str) -> Path:
        candidate = (repo / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        try:
            relative = candidate.relative_to(repo)
        except ValueError as exc:
            raise ValueError(f"path {path!r} escapes repo workdir") from exc
        if relative.parts and relative.parts[0] == ".git":
            raise ValueError("access to Git metadata is forbidden")
        return candidate

    def file_read(self, peer: str, path: str, offset: int = 0, limit: int | None = None) -> str:
        try:
            content = self._path(self.repo(peer), path).read_text(errors="replace")
            return _read_window(content, offset, limit)
        except Exception as exc:
            return f"ERROR: {exc}"

    def str_replace(self, peer: str, path: str, old: str, new: str) -> str:
        try:
            target = self._path(self.repo(peer), path)
            content = target.read_text(errors="replace")
            count = content.count(old)
            if count == 0:
                return f"ERROR: `old` not found in {path}"
            if count > 1:
                return f"ERROR: `old` appears {count} times in {path}; edit is ambiguous"
            target.write_text(content.replace(old, new, 1))
            return f"replaced 1 occurrence in {path}"
        except Exception as exc:
            return f"ERROR: {exc}"

    def list_dir(self, peer: str, path: str = ".") -> str:
        try:
            repo = self.repo(peer)
            target = self._path(repo, path)
            if not target.is_dir():
                return f"ERROR: {path} is not a directory"
            entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name))
            return "\n".join(
                f"{'d' if item.is_dir() else 'f'}  {item.relative_to(repo)}" for item in entries if item.name != ".git"
            )
        except Exception as exc:
            return f"ERROR: {exc}"

    def search_repo(self, peer: str, pattern: str, path: str = ".", max_matches: int = 50) -> str:
        try:
            repo = self.repo(peer)
            target = self._path(repo, path)
            expression = re.compile(pattern)
            cap = min(MAX_SEARCH_MATCHES, max(1, int(max_matches)))
            matches = _grep_files(repo, target, expression, cap)
            if not matches:
                return f"[no matches for {pattern!r}]"
            if len(matches) > cap:
                matches = matches[:cap] + [f"... [+{len(matches) - cap} more]"]
            return "\n".join(matches)
        except Exception as exc:
            return f"ERROR: {exc}"

    def shell_exec(self, peer: str, command: str, timeout_s: int = SHELL_TIMEOUT_S) -> str:
        return swe_sandbox.shell_exec(self.repo(peer), command, timeout_s)


def _grep_files(repo: Path, target: Path, expression: re.Pattern, cap: int) -> list[str]:
    """``file:line:text`` matches of ``expression`` under ``target`` outside ``.git``; stops after ``cap`` + 1."""
    matches: list[str] = []
    for candidate in [target] if target.is_file() else sorted(target.rglob("*")):
        if len(matches) > cap:
            break
        if not candidate.is_file() or ".git" in candidate.relative_to(repo).parts:
            continue
        try:
            lines = candidate.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, 1):
            if expression.search(line):
                matches.append(f"{candidate.relative_to(repo)}:{number}:{line}")
                if len(matches) > cap:
                    break
    return matches


# Agents SDK tools: descriptions and JSON parameters.
AGENTS_TOOLS = {
    "file_read": (
        "Read a file from this peer's isolated repository worktree.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
            "required": ["path"],
        },
    ),
    "str_replace": (
        "Replace exactly one occurrence in a repository file.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
            "required": ["path", "old", "new"],
        },
    ),
    "list_dir": (
        "List entries below this peer's repository worktree.",
        {"type": "object", "properties": {"path": {"type": "string"}}},
    ),
    "search_repo": (
        "Regex-search text files in this peer's repository worktree.",
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "max_matches": {"type": "integer"},
            },
            "required": ["pattern"],
        },
    ),
    "shell_exec": (
        "Run a bounded shell command inside the instance's SWE image; network and host files are unavailable.",
        {
            "type": "object",
            "properties": {"command": {"type": "string"}, "timeout_s": {"type": "integer"}},
            "required": ["command"],
        },
    ),
}


def agents_input(instance: dict) -> str:
    """Task text given to every Agents SDK peer."""
    return (
        f"Repository: {instance['repo']}\nBase commit: {instance['base_commit']}\n\n"
        f"{instance['problem_statement']}\n\nReturn a unified diff patch."
    )


# Scoring in the instance image
# The running batch's log directory (``run_batch`` sets ``<out_dir>/eval_logs``), so
# cells that evaluate the same instance in parallel never share a log file.
_BATCH_EVAL_LOG_DIR: ContextVar[Path | None] = ContextVar("swe_eval_log_dir", default=None)


def eval_log_dir() -> Path:
    """Where :func:`run_tests_singularity` keeps its logs.

    ``$SWE_EVAL_LOG_DIR`` when set, else ``<out_dir>/eval_logs`` of the running
    :func:`run_batch`, else ``results/swe_eval_logs``.
    """
    override = os.environ.get("SWE_EVAL_LOG_DIR")
    if override:
        return Path(override)
    return _BATCH_EVAL_LOG_DIR.get() or RESULTS_DIR / "swe_eval_logs"


SIF_EVAL_SCRIPT = r"""
set -eo pipefail
cat > /tmp/gitcfg <<EOF
[safe]
    directory = /testbed
EOF
export GIT_CONFIG_GLOBAL=/tmp/gitcfg

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate testbed

cd /testbed
if [ -s /tmp/test.patch ]; then
    if ! git apply --verbose --recount /tmp/test.patch; then
        echo "__TEST_PATCH_APPLY_FAILED__"
        exit 3
    fi
fi
if [ -s /tmp/model.patch ]; then
    if ! git apply --verbose --recount /tmp/model.patch; then
        echo "__APPLY_FAILED__"
        exit 2
    fi
fi

python -m pytest \
    -p no:cacheprovider \
    -v --tb=no --no-header \
    -o console_output_style=classic \
    "$@"
"""
_VERDICT_RE = re.compile(r"^(?P<nodeid>\S+?)\s+(?P<verdict>PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)\b")


def ensure_sif(instance_id: str) -> Path:
    """The instance's SIF under :data:`SIF_DIR`, pulled from docker://swebench on first use.

    RuntimeError if the pull fails.
    """
    sif = SIF_DIR / f"{instance_id}.sif"
    if sif.exists():
        return sif
    SIF_DIR.mkdir(parents=True, exist_ok=True)
    docker_ref = SWEBENCH_IMAGE.format(tag=instance_id.replace("__", "_1776_"))
    logger.info("[swe] pulling %s -> %s", docker_ref, sif.name)
    result = subprocess.run(
        ["singularity", "pull", str(sif), docker_ref], capture_output=True, text=True, timeout=SIF_PULL_TIMEOUT_S
    )
    if result.returncode != 0:
        raise RuntimeError(f"singularity pull failed for {instance_id}: {result.stderr.strip()}")
    return sif


def _verdicts(output: str, test_ids: list[str]) -> dict[str, str]:
    """Verdict per test id parsed from classic pytest output (``not_run`` when absent)."""
    status: dict[str, str] = {}
    for line in output.splitlines():
        m = _VERDICT_RE.match(line.strip())
        if m:
            status[m.group("nodeid")] = m.group("verdict")
    for tid in test_ids:
        status.setdefault(tid, "not_run")
    return status


def _report(fail_to_pass: list[str], pass_to_pass: list[str], passed: Callable[[str], bool]) -> dict:
    """``{fail_to_pass, pass_to_pass: {success, failure}, f2p_rate, p2p_rate}`` (a rate is 1.0 for no tests)."""

    def bucket(ids: list[str]) -> dict:
        return {"success": [t for t in ids if passed(t)], "failure": [t for t in ids if not passed(t)]}

    f2p, p2p = bucket(fail_to_pass), bucket(pass_to_pass)
    return {
        "fail_to_pass": f2p,
        "pass_to_pass": p2p,
        "f2p_rate": (len(f2p["success"]) / len(fail_to_pass)) if fail_to_pass else 1.0,
        "p2p_rate": (len(p2p["success"]) / len(pass_to_pass)) if pass_to_pass else 1.0,
    }


def _failed_report(fail_to_pass: list[str], pass_to_pass: list[str], error: str, tail: str | None = None) -> dict:
    """Every test failed (rates 0.0) because of ``error``."""
    report = {
        "fail_to_pass": {"success": [], "failure": list(fail_to_pass)},
        "pass_to_pass": {"success": [], "failure": list(pass_to_pass)},
        "f2p_rate": 0.0,
        "p2p_rate": 0.0,
        "error": error,
    }
    if tail is not None:
        report["stderr_tail"] = tail
    return report


def run_tests_singularity(
    instance: dict,
    patch: str,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    timeout_s: int = SIF_EVAL_TIMEOUT_S,
) -> dict:
    """Apply the test patch and ``patch`` inside the instance image and run the tests.

    Returns the :func:`is_resolved` report; a patch that does not apply, or a
    timeout, fails every test and sets ``error``. The combined pytest output is
    kept in ``eval_<id>.log`` under :func:`eval_log_dir`.
    """
    iid = instance["instance_id"]
    sif = ensure_sif(iid)
    test_ids = list(fail_to_pass) + list(pass_to_pass)
    if not test_ids:
        return _report([], [], lambda _: True)  # rates 1.0
    patch_path = _temp_patch(patch or "")
    test_patch_path = _temp_patch(instance.get("test_patch") or "")
    try:
        cmd = [
            "singularity",
            "exec",
            "--writable-tmpfs",
            "--bind",
            f"{patch_path}:/tmp/model.patch:ro",
            "--bind",
            f"{test_patch_path}:/tmp/test.patch:ro",
            str(sif),
            "bash",
            "-c",
            SIF_EVAL_SCRIPT,
            "bash",  # $0 of the inline script
            *test_ids,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return _failed_report(fail_to_pass, pass_to_pass, "timeout")
    finally:
        for path in (patch_path, test_patch_path):
            Path(path).unlink(missing_ok=True)

    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    log_dir = eval_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"eval_{iid}.log").write_text(combined)
    if "__TEST_PATCH_APPLY_FAILED__" in combined:
        return _failed_report(fail_to_pass, pass_to_pass, "test_patch apply failed", combined[-500:])
    if "__APPLY_FAILED__" in combined:
        return _failed_report(fail_to_pass, pass_to_pass, "patch apply failed", combined[-500:])
    status = _verdicts(combined, test_ids)
    return _report(fail_to_pass, pass_to_pass, lambda tid: status.get(tid) in PASSING_VERDICTS)


def _temp_patch(text: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
        f.write(text)
        return f.name


def apply_test_patch(repo_dir: Path, test_patch: str) -> str:
    """Apply the instance's ``test_patch`` to ``repo_dir`` (local evaluation); "" or the error."""
    if not test_patch.strip():
        return ""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "apply", "--verbose", "--recount", "-"],
            input=test_patch,
            capture_output=True,
            text=True,
            timeout=SHELL_TIMEOUT_S,
        )
    except Exception as e:
        return f"ERROR: {e}"
    if result.returncode != 0:
        return f"ERROR: git apply exited {result.returncode}\n{result.stderr}"
    return ""


def _run_pytest(repo_dir: Path, test_ids: list[str], timeout_s: int) -> dict[str, str]:
    """Verdict per pytest node id, running pytest on the host in ``repo_dir``."""
    if not test_ids:
        return {}
    cmd = [
        "python",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--tb=no",
        "-v",
        "--no-header",
        "-o",
        "console_output_style=classic",
        *test_ids,
    ]
    try:
        result = subprocess.run(cmd, cwd=str(repo_dir), capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {tid: "timeout" for tid in test_ids}
    return _verdicts(result.stdout + "\n" + result.stderr, test_ids)


def run_tests_local(repo_dir: Path, fail_to_pass: list[str], pass_to_pass: list[str], timeout_s: int = 600) -> dict:
    """Both test groups with pytest in ``repo_dir`` on the host (fast, not equivalent to the image)."""
    f2p_status = _run_pytest(repo_dir, fail_to_pass, timeout_s)
    p2p_status = _run_pytest(repo_dir, pass_to_pass, timeout_s)

    def bucket(status: dict[str, str]) -> dict:
        return {
            "success": [t for t, v in status.items() if v in PASSING_VERDICTS],
            "failure": [t for t, v in status.items() if v not in PASSING_VERDICTS],
        }

    f2p, p2p = bucket(f2p_status), bucket(p2p_status)
    return {
        "fail_to_pass": f2p,
        "pass_to_pass": p2p,
        "f2p_rate": (len(f2p["success"]) / len(fail_to_pass)) if fail_to_pass else 1.0,
        "p2p_rate": (len(p2p["success"]) / len(pass_to_pass)) if pass_to_pass else 1.0,
    }


def is_resolved(report: dict) -> bool:
    """SWE-bench RESOLVED verdict: both rates must equal 1.0 (strict)."""
    return report.get("f2p_rate") == 1.0 and report.get("p2p_rate") == 1.0


def exact_match_score(report: dict) -> float:
    """1.0 iff the instance is resolved, else 0.0."""
    return 1.0 if is_resolved(report) else 0.0


# Ensembles: one member's patch is submitted and scored
def select_patch(patches: list[str | None]) -> int:
    """Index of the submitted patch: the vote over the whitespace-normalized patches (:mod:`core.voting`)."""
    return voting.majority_text(patches)


def score_selected(selected: dict, instance: dict) -> None:
    """Evaluate the selected member's ``patch`` in the instance image, in place.

    Sets ``report``, ``resolved`` and ``score`` (f2p_rate * p2p_rate); an empty
    patch is unresolved without a report, and an evaluation failure is the
    report's ``error``.
    """
    selected.update({"resolved": False, "score": 0.0})
    if not selected["patch"]:
        return
    f2p, p2p = instance_tests(instance)
    try:
        report = run_tests_singularity(instance, selected["patch"], f2p, p2p)
    except Exception as e:
        report = {"error": f"{type(e).__name__}: {e}"}
    selected.update(report=report, resolved=is_resolved(report))
    selected["score"] = report.get("f2p_rate", 0.0) * report.get("p2p_rate", 0.0)


# Records and artifacts
def predictions_entry(instance_id: str, patch: str, model_name: str) -> dict:
    """One line of the predictions JSONL read by ``swebench.harness.run_evaluation``."""
    return {"instance_id": instance_id, "model_patch": patch, "model_name_or_path": model_name}


def record_head(instance: dict, **fields) -> dict:
    """The fields every record starts with: instance id, repository, base commit, then ``fields``."""
    return {
        "instance_id": instance["instance_id"],
        "repo": instance["repo"],
        "base_commit": instance["base_commit"],
        **fields,
    }


def solve_in_checkout(
    instance: dict, workdir_root: Path, bind: Callable[[Path], None], solve: Callable[[], dict]
) -> tuple[dict, dict | None]:
    """Clone the instance to ``workdir_root/<id>``, register it with the sandbox, ``bind`` it and ``solve``.

    Returns the record so far (``clone_s`` and ``solve_s``, or ``error`` and
    ``stage`` on failure) and the solve output (None on failure).
    """
    iid = instance["instance_id"]
    summary = record_head(instance)
    workdir = workdir_root / iid
    start = time.time()
    error = clone_and_checkout(instance["repo"], instance["base_commit"], workdir)
    if error:
        summary.update(error=error, stage="clone")
        return summary, None
    summary["clone_s"] = round(time.time() - start, 1)
    swe_sandbox.register_worktree(workdir, lambda: ensure_sif(iid))
    bind(workdir)
    start = time.time()
    try:
        out = solve()
    except Exception as e:
        summary.update(error=f"{type(e).__name__}: {e}", stage="solve")
        return summary, None
    summary["solve_s"] = round(time.time() - start, 1)
    return summary, out


def eval_fields(eval_mode: str, report: dict | None, *, eval_s: float | None = None) -> dict:
    """Record fields of a scored patch: skipped, unresolved without a report, else rates and failing tests."""
    if eval_mode == "none":
        return {"eval": "skipped"}
    if report is None:
        return {"eval_mode": eval_mode, "resolved": False, "f2p_rate": 0.0, "p2p_rate": 0.0}
    fields = {"eval_mode": eval_mode}
    if eval_s is not None:
        fields["eval_s"] = eval_s
    fields.update(
        f2p_rate=report["f2p_rate"],
        p2p_rate=report["p2p_rate"],
        resolved=is_resolved(report),
        f2p_failures=report["fail_to_pass"]["failure"],
        p2p_failures=report["pass_to_pass"]["failure"],
    )
    if report.get("error"):
        fields["eval_error"] = report["error"]
    return fields


def winner_eval_fields(eval_mode: str, out: dict, members: list[dict], key: str) -> dict:
    """Record fields of an ensemble: the rates of the winning member (``members[i][key] == out["winner"]``)."""
    if eval_mode == "none":
        return {"eval": "skipped"}
    winner = next((m for m in members if m.get(key) == out.get("winner")), None)
    report = (winner or {}).get("report") or {}
    return {
        "eval_mode": eval_mode,
        "f2p_rate": report.get("f2p_rate", 0.0),
        "p2p_rate": report.get("p2p_rate", 0.0),
        "resolved": out.get("resolved") or False,
    }


def peer_rates(per_peer: list[dict]) -> list[dict]:
    """The per-peer record entries of a debate."""
    return [
        {
            "peer": s["peer"],
            "patch_chars": len(s.get("patch") or ""),
            "f2p_rate": (s.get("report") or {}).get("f2p_rate"),
            "p2p_rate": (s.get("report") or {}).get("p2p_rate"),
            "resolved": s.get("resolved"),
        }
        for s in per_peer
    ]


def sections(pairs) -> str:
    """Trace text: ``=== title ===`` and the text of every ``(title, text)`` pair."""
    return "".join(f"=== {title} ===\n{text}\n\n" for title, text in pairs)


def peer_trace(out: dict) -> str:
    """Trace text of a debate: the winner, then every peer's patch size and score."""
    blocks = [f"winner: peer {out.get('winner')}  resolved={out.get('resolved')}\n\n"]
    for s in out.get("per_peer") or []:
        report = s.get("report") or {}
        blocks.append(
            f"=== peer {s['peer']} ===\n"
            f"  patch_chars={len(s.get('patch') or '')}\n"
            f"  f2p_rate={report.get('f2p_rate')}  p2p_rate={report.get('p2p_rate')}\n"
            f"  resolved={s.get('resolved')}  score={s.get('score')}\n\n"
        )
    return "".join(blocks)


def write_artifacts(out_dir: Path, instance_id: str, patch: str, entry: dict, trace: str) -> None:
    """Write ``patches/<id>.diff``, append ``entry`` to ``predictions.jsonl`` and write ``traces/<id>.txt``."""
    (out_dir / "patches").mkdir(parents=True, exist_ok=True)
    (out_dir / "patches" / f"{instance_id}.diff").write_text(patch)
    with (out_dir / "predictions.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    (out_dir / "traces" / f"{instance_id}.txt").write_text(trace)


# Batch
def default_dirs(name: str) -> tuple[Path, Path]:
    """Default ``(workdir_root, out_dir)`` of a runner: ``~/swe_work_<name>`` and ``results/swe_bench_<name>``."""
    suffix = f"_{name}" if name else ""
    return Path.home() / f"swe_work{suffix}", RESULTS_DIR / f"swe_bench{suffix}"


def log_loaded(instances: list[dict], team: str = "") -> None:
    logger.info("loaded %d instance(s) from %s%s", len(instances), HF_DATASET, team)


def _summarize(records: list[dict]) -> dict:
    return {"n": len(records), "resolved": sum(1 for r in records if r.get("resolved") is True)}


def run_batch(
    instances: list[dict],
    run_one: Callable[[dict], dict],
    *,
    out_dir: Path,
    eval_mode: str,
    workdirs: Callable[[dict], list[Path]] | None = None,
    omit: str | None = None,
    predictions: Path | None = None,
) -> list[dict]:
    """Run ``run_one`` on every instance; returns the records.

    ``run_one`` appends to ``out_dir/predictions.jsonl`` (emptied first); the
    records go to ``out_dir/results.jsonl``. Progress is logged (the ``omit``
    field left out) and the end-of-batch report goes to stderr; the ``workdirs``
    of an instance are removed after it. A ``predictions`` path other than
    ``out_dir/predictions.jsonl`` receives a copy of the predictions at the end.
    The image evaluations log to ``out_dir/eval_logs`` (see :func:`eval_log_dir`).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    predictions_path.write_text("")
    results_path = out_dir / "results.jsonl"

    def row(index: int, inst: dict) -> dict:
        record = run_one(inst)
        for path in workdirs(inst) if workdirs else ():
            shutil.rmtree(path, ignore_errors=True)
        return record

    def banner(summary: dict) -> str:
        lines = [f"\ndone: predictions -> {predictions_path}", f"      results     -> {results_path}"]
        if eval_mode != "none":
            lines.append(f"      resolved ({eval_mode}): {summary['resolved']}/{summary['n']}")
        return "\n".join(lines)

    log_dir = _BATCH_EVAL_LOG_DIR.set(out_dir / "eval_logs")
    try:
        summary = batch.run_batch(
            instances,
            row,
            summarize=_summarize,
            out_path=results_path,
            stream=sys.stderr,
            header=lambda index, total, inst: (
                f"\n[{index + 1}/{total}] {inst['instance_id']}  ({inst['repo']}@{inst['base_commit'][:7]})"
            ),
            progress=lambda index,
            total,
            record,
            done: f"  -> {json.dumps({k: v for k, v in record.items() if k != omit})}",
            banner=banner,
        )
    finally:
        _BATCH_EVAL_LOG_DIR.reset(log_dir)
    if predictions is not None and Path(predictions).resolve() != predictions_path.resolve():
        Path(predictions).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(predictions_path, predictions)
    return summary["per_instance"]
