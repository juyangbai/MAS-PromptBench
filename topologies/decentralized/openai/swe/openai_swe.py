"""Decentralized debate topology specialized for SWE-bench Verified, OpenAI SDK."""

# Config
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from contextvars import ContextVar
from pathlib import Path

from topologies.output_contracts import append_output_contract_from_path

from openai import OpenAI, BadRequestError

# Shared telemetry.
_TOPO_ROOT = str(Path(__file__).resolve().parents[4])
if _TOPO_ROOT not in sys.path:
    sys.path.insert(0, _TOPO_ROOT)
from topologies.telemetry import (  # noqa: E402
    openai_sdk_telemetry, openai_sdk_accumulate, normalize,
)


_TELEM_ACC: dict = {
    "prompt_tokens": 0, "completion_tokens": 0,
    "total_tokens": 0, "n_llm_calls": 0, "n_tool_calls": 0,
}


def _reset_telem_acc() -> None:
    for k in _TELEM_ACC:
        _TELEM_ACC[k] = 0

# Infrastructure reuse from sequential/crewai/swe.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "sequential" / "crewai" / "swe"))
import crewai_swe as _crewai_swe  # noqa: E402
from crewai_swe import (  # noqa: E402
    clone_and_checkout,
    compute_patch,
    is_resolved,
    load_instances,
    run_tests_singularity,
)


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://lai:8001/v1")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")

N_AGENTS = int(os.environ.get("DECENTRALIZED_N_AGENTS", "4"))
N_ROUNDS = int(os.environ.get("DECENTRALIZED_N_ROUNDS", "2"))

_SHELL_TIMEOUT_S = 60
_READ_CHAR_BUDGET = 20_000
_PROBLEM_CHAR_BUDGET = int(os.environ.get("SWE_PROBLEM_CHAR_BUDGET", "16000"))
_HINTS_CHAR_BUDGET = int(os.environ.get("SWE_HINTS_CHAR_BUDGET", "4000"))

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROMPTS_DIR = _REPO_ROOT / "configs" / "prompts" / "decentralized" / "swe"


def _load_prompt(role: str) -> str:
    return append_output_contract_from_path((_PROMPTS_DIR / f"{role}.txt").read_text().strip(), __file__, role)


SYSTEM_PROMPT = _load_prompt("debater")


# Per-peer workdir tracking
# Each peer has its OWN clone of the repo — tools need to resolve paths
# against that peer's workdir, not a global one. ContextVar + per-turn
# bind keeps it simple.
_REPO_DIR_VAR: ContextVar[Path] = ContextVar("_REPO_DIR_VAR", default=Path("."))


def _get_repo_dir() -> Path:
    return _REPO_DIR_VAR.get()


def _repo_path(path: str) -> Path:
    repo = _get_repo_dir()
    candidate = (repo / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    candidate.relative_to(repo)  # raises if escapes
    return candidate


# Tools
def file_read(path: str, offset: int = 0, limit: int | None = None) -> str:
    try:
        p = _repo_path(path)
        content = p.read_text(errors="replace")
    except Exception as e:
        return f"ERROR: {e}"
    total = content.count("\n") + 1
    if offset or limit is not None:
        lines = content.splitlines(keepends=True)
        start = max(0, int(offset))
        end = start + int(limit) if limit is not None else len(lines)
        content = (
            f"[lines {start + 1}-{min(end, len(lines))} of {total}]\n"
            + "".join(lines[start:end])
        )
    if len(content) > _READ_CHAR_BUDGET:
        return content[:_READ_CHAR_BUDGET] + f"\n... [truncated, total {len(content)} chars]"
    return content


def str_replace(path: str, old: str, new: str) -> str:
    try:
        p = _repo_path(path)
        content = p.read_text(errors="replace")
    except Exception as e:
        return f"ERROR: {e}"
    count = content.count(old)
    if count == 0:
        return (
            f"ERROR: `old` not found in {path}. Check whitespace / line "
            "endings / tab-space mismatch. Call file_read to inspect."
        )
    if count > 1:
        return (
            f"ERROR: `old` appears {count} times in {path}; add more "
            "surrounding context to make the match unique."
        )
    try:
        p.write_text(content.replace(old, new, 1))
    except Exception as e:
        return f"ERROR: {e}"
    return f"replaced 1 occurrence in {path}"


def list_dir(path: str = ".") -> str:
    try:
        p = _repo_path(path)
        if not p.is_dir():
            return f"ERROR: {path} is not a directory"
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        repo = _get_repo_dir()
        return "\n".join(
            f"{'d' if e.is_dir() else 'f'}  {e.relative_to(repo)}" for e in entries
        )
    except Exception as e:
        return f"ERROR: {e}"


def search_repo(pattern: str, path: str = ".", max_matches: int = 50) -> str:
    try:
        target = _repo_path(path)
    except Exception as e:
        return f"ERROR: {e}"
    try:
        r = subprocess.run(
            ["grep", "-rn", "-E", pattern, str(target)],
            capture_output=True, text=True, timeout=_SHELL_TIMEOUT_S,
        )
    except Exception as e:
        return f"ERROR: {e}"
    if r.returncode not in (0, 1):
        return f"ERROR: grep exit {r.returncode}\n{r.stderr}"
    lines = r.stdout.splitlines()
    if not lines:
        return f"[no matches for {pattern!r}]"
    if len(lines) > max_matches:
        extra = len(lines) - max_matches
        lines = lines[:max_matches] + [f"... [+{extra} more]"]
    prefix = str(_get_repo_dir()) + "/"
    return "\n".join(line.replace(prefix, "") for line in lines)


def shell_exec(command: str, timeout_s: int = _SHELL_TIMEOUT_S) -> str:
    try:
        r = subprocess.run(
            command, shell=True, cwd=str(_get_repo_dir()),
            capture_output=True, text=True, timeout=timeout_s,
        )
        return f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}\nexit_code: {r.returncode}"
    except subprocess.TimeoutExpired:
        return f"ERROR: exceeded {timeout_s}s"
    except Exception as e:
        return f"ERROR: {e}"


_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read a file from the repository working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": "Replace EXACTLY ONE occurrence of `old` with `new` in `path`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List entries in a directory under the repo workdir.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_repo",
            "description": "grep-style regex search under the repo workdir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "max_matches": {"type": "integer"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": "Run a shell command in the repo workdir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_s": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
]


def _dispatch_tool(name: str, arguments: dict) -> str:
    if name == "file_read":
        return file_read(
            arguments.get("path", ""),
            arguments.get("offset", 0),
            arguments.get("limit"),
        )
    if name == "str_replace":
        return str_replace(arguments.get("path", ""), arguments.get("old", ""), arguments.get("new", ""))
    if name == "list_dir":
        return list_dir(arguments.get("path", "."))
    if name == "search_repo":
        return search_repo(
            arguments.get("pattern", ""),
            arguments.get("path", "."),
            arguments.get("max_matches", 50),
        )
    if name == "shell_exec":
        return shell_exec(arguments.get("command", ""), arguments.get("timeout_s", _SHELL_TIMEOUT_S))
    return f"ERROR: unknown tool {name!r}"


# Client
def _build_client() -> OpenAI:
    return OpenAI(base_url=VLLM_BASE_URL, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))


def _completion_kwargs() -> dict:
    return {
        "model": MODEL_ID,
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": 0,
        # SWE edits can be verbose; give headroom.
        "max_tokens": 4096,
        "extra_body": {
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def _chat_with_tools(client: OpenAI, messages: list[dict], max_tool_loops: int = 20) -> dict:
    kwargs = _completion_kwargs()
    kwargs["tools"] = _TOOL_SCHEMAS
    kwargs["tool_choice"] = "auto"
    dump: dict = {}
    for _ in range(max_tool_loops):
        try:
            resp = client.chat.completions.create(messages=messages, **kwargs)
        except BadRequestError as e:
            # vLLM sometimes rejects a follow-up request when the prior
            # assistant turn's tool_call arguments contain an unterminated
            # JSON string (truncated at max_tokens, unescaped quotes, etc).
            # Strip the last assistant turn + its tool-result messages and
            # try once more so the peer can keep working.
            if dump and messages and messages[-1].get("role") == "tool":
                while messages and messages[-1].get("role") == "tool":
                    messages.pop()
                if messages and messages[-1].get("role") == "assistant":
                    messages.pop()
                messages.append({
                    "role": "user",
                    "content": (
                        "The previous tool call produced an invalid request. "
                        "Continue with a different approach (do not repeat the "
                        "same call)."
                    ),
                })
                try:
                    resp = client.chat.completions.create(messages=messages, **kwargs)
                except BadRequestError:
                    return dump or {"role": "assistant", "content": f"ERROR: vLLM BadRequest: {e}"}
            else:
                return dump or {"role": "assistant", "content": f"ERROR: vLLM BadRequest: {e}"}
        openai_sdk_accumulate(_TELEM_ACC, resp)
        msg = resp.choices[0].message
        dump = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
        tool_calls = dump.get("tool_calls") or []
        messages.append(dump)
        if not tool_calls:
            return dump
        for tc in tool_calls:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _dispatch_tool(fn.get("name", ""), args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "content": result,
            })
    return dump


# Prompt scaffolding
def _truncate(text: str, cap: int, label: str) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n... [truncated {label}: {len(text)} -> {cap} chars]"


def format_task_brief(
    problem_statement: str,
    instance_id: str | None = None,
    hints_text: str | None = None,
) -> str:
    parts = []
    if instance_id:
        parts.append(f"INSTANCE: {instance_id}")
    parts.append(f"The repository is checked out on the failing commit at your peer-local workdir.")
    parts.append(
        "ISSUE:\n"
        + _truncate(problem_statement.strip(), _PROBLEM_CHAR_BUDGET, "problem_statement")
    )
    if hints_text:
        parts.append(
            "HINTS (from maintainers):\n"
            + _truncate(hints_text.strip(), _HINTS_CHAR_BUDGET, "hints_text")
        )
    parts.append(
        "Do NOT try to run the repo's own tests here — this workdir is only a "
        "source checkout; C extensions and test deps are NOT installed. Tests "
        "will be run separately in a prepared environment."
    )
    return "\n\n".join(parts)


# Debate loop
def _peer_injection(others_final: list[dict], brief: str) -> dict:
    body = ["These are the final summaries from other peer agents in the previous round (each peer worked in its OWN repo clone — their file changes are NOT visible in yours):"]
    for i, m in enumerate(others_final):
        body.append(f"\nPeer {i + 1}:\n```\n{m.get('content') or ''}\n```")
    body.append(
        "\nCompare their approach with yours. If a peer found a better fix "
        "location or caught a regression, REVISE your edits in your own "
        "workdir. Use str_replace to apply the revised fix; use shell_exec "
        "+ `git diff HEAD` to inspect the current state of your workdir.\n\n"
        "Original brief:\n" + brief
    )
    return {"role": "user", "content": "\n".join(body)}


def _run_peer_all_rounds(
    peer_id: int,
    workdir: Path,
    brief: str,
    peer_round_finals_ref: dict,  # {round_idx: list[dict]} shared via threading.Lock
    sync_events: list,  # [threading.Event for each round]; each peer sets its own pos + waits for all
    all_set_events: list,  # barrier per round
    client: OpenAI,
) -> list[dict]:
    """Run all R rounds for ONE peer, synchronized with other peers at
    round boundaries."""
    ctx = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": brief},
    ]
    token = _REPO_DIR_VAR.set(workdir)
    # Barrier failsafe: whatever happens (crash, early return), every round's
    # sync event for this peer MUST be set before we exit — otherwise siblings
    # hang forever in all_set_events[r].wait().
    completed_rounds: list[int] = []
    try:
        for r in range(N_ROUNDS):
            if r > 0:
                others_final = [
                    peer_round_finals_ref[r - 1][j]
                    for j in range(N_AGENTS) if j != peer_id
                ]
                ctx.append(_peer_injection(others_final, brief))
            try:
                final_msg = _chat_with_tools(client, ctx)
            except Exception as e:
                final_msg = {
                    "role": "assistant",
                    "content": f"ERROR: peer {peer_id} round {r} crashed: {type(e).__name__}: {e}",
                }
            peer_round_finals_ref[r][peer_id] = final_msg
            sync_events[r][peer_id].set()
            completed_rounds.append(r)
            all_set_events[r].wait()
        return ctx
    finally:
        # Ensure any rounds we never entered still publish a placeholder +
        # signal their event, so sibling peers can make progress.
        for r in range(N_ROUNDS):
            if r in completed_rounds:
                continue
            if peer_round_finals_ref[r][peer_id] is None:
                peer_round_finals_ref[r][peer_id] = {
                    "role": "assistant",
                    "content": f"ERROR: peer {peer_id} never reached round {r}",
                }
            sync_events[r][peer_id].set()
        _REPO_DIR_VAR.reset(token)


def _barrier_watcher(events: list, all_set: "threading.Event") -> None:
    for e in events:
        e.wait()
    all_set.set()


def run_debate(brief: str, peer_workdirs: list[Path]) -> list[list[dict]]:
    """Run N peers × R rounds concurrently.

    Each peer edits its OWN workdir; rounds are synchronized at peer
    boundaries so round r+1 injection sees all N peers' round-r finals.
    """
    assert len(peer_workdirs) == N_AGENTS, f"expected {N_AGENTS} workdirs, got {len(peer_workdirs)}"
    client = _build_client()

    # Per-round barriers: every peer signals its round-r done, every peer
    # waits for all-set before starting round r+1.
    sync_events = [
        [threading.Event() for _ in range(N_AGENTS)] for _ in range(N_ROUNDS)
    ]
    all_set_events = [threading.Event() for _ in range(N_ROUNDS)]
    round_finals = {r: [None] * N_AGENTS for r in range(N_ROUNDS)}

    for r in range(N_ROUNDS):
        threading.Thread(
            target=_barrier_watcher,
            args=(sync_events[r], all_set_events[r]),
            daemon=True,
        ).start()

    contexts: list[list[dict] | None] = [None] * N_AGENTS

    def _run(peer_id: int) -> None:
        contexts[peer_id] = _run_peer_all_rounds(
            peer_id, peer_workdirs[peer_id], brief,
            round_finals, sync_events, all_set_events, client,
        )

    threads = [threading.Thread(target=_run, args=(i,)) for i in range(N_AGENTS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return contexts  # type: ignore[return-value]


# Best-of-N over peer workdirs
def best_of_n(
    peer_workdirs: list[Path],
    instance: dict,
    eval_mode: str = "singularity",
) -> tuple[int | None, list[dict]]:
    """Run compute_patch() + SIF eval on each peer's workdir; pick the
    first resolved peer (lowest index), else max f2p × p2p."""
    f2p = instance["FAIL_TO_PASS"]
    p2p = instance["PASS_TO_PASS"]
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)

    # Phase 1: compute patches SEQUENTIALLY. Patch extraction is fast (`git
    # diff HEAD`), and keeping it serial avoids relying on helper-module repo
    # context propagation across extra worker threads.
    scored: list[dict] = []
    for i, workdir in enumerate(peer_workdirs):
        _crewai_swe._set_repo_dir(workdir)
        scored.append({
            "peer": i, "workdir": str(workdir), "patch": compute_patch(),
            "report": None, "resolved": False, "score": 0.0,
        })

    # Phase 2: run Singularity evals IN PARALLEL. Each eval spawns its own
    # child process against a per-instance SIF, so N=4 can run
    # simultaneously without sharing any Python state. Wall time per
    # instance drops from 4× SIF eval to max(4 parallel SIF evals).
    def _score_one(rec: dict) -> None:
        if not rec["patch"] or eval_mode == "none":
            return
        try:
            report = run_tests_singularity(instance, rec["patch"], f2p, p2p)
        except Exception as e:
            rec["report"] = {"error": f"{type(e).__name__}: {e}"}
            return
        rec["report"] = report
        rec["resolved"] = is_resolved(report)
        rec["score"] = report.get("f2p_rate", 0.0) * report.get("p2p_rate", 0.0)

    if eval_mode != "none":
        eval_threads = [
            threading.Thread(target=_score_one, args=(rec,)) for rec in scored
        ]
        for t in eval_threads:
            t.start()
        for t in eval_threads:
            t.join()

    perfect = [s for s in scored if s["resolved"]]
    if perfect:
        winner = min(perfect, key=lambda s: s["peer"])
    elif any(s["patch"] for s in scored):
        winner = max(scored, key=lambda s: (s["score"], -s["peer"]))
    else:
        return None, scored
    return winner["peer"], scored


# Orchestration
def solve(
    instance: dict,
    peer_workdirs: list[Path],
    eval_mode: str = "singularity",
) -> dict:
    """Run N peers × R rounds. Each peer edits ITS OWN workdir — caller is
    responsible for cloning N separate copies before calling solve()."""
    _reset_telem_acc()
    brief = format_task_brief(
        instance["problem_statement"],
        instance_id=instance.get("instance_id"),
        hints_text=instance.get("hints_text"),
    )
    contexts = run_debate(brief, peer_workdirs)

    winner_idx, scored = best_of_n(peer_workdirs, instance, eval_mode=eval_mode)
    winner = scored[winner_idx] if winner_idx is not None else None
    return {
        "patch": (winner or {}).get("patch") or "",
        "resolved": (winner or {}).get("resolved") if eval_mode != "none" else None,
        "winner": winner_idx,
        "per_peer": scored,
        "all_contexts": contexts,
        "telemetry": normalize(
            dict(_TELEM_ACC) if _TELEM_ACC["n_llm_calls"]
            else openai_sdk_telemetry(contexts)
        ),
    }


# Predictions + batch runner
def predictions_entry(
    instance_id: str, patch: str,
    model_name: str = "mas-promptbench-decentralized",
) -> dict:
    return {
        "instance_id": instance_id,
        "model_patch": patch,
        "model_name_or_path": model_name,
    }


def run_one(
    instance: dict,
    workdir_root: Path,
    out_dir: Path,
    eval_mode: str = "singularity",
) -> dict:
    """Clone N peer workdirs, run the debate, score via best-of-N, write
    artifacts. Matches single/swe's run_one pattern but with N per-peer
    checkouts under `workdir_root / <iid> / peer_<k>`."""
    iid = instance["instance_id"]
    summary: dict = {
        "instance_id": iid,
        "repo": instance["repo"],
        "base_commit": instance["base_commit"],
        "n_peers": N_AGENTS,
        "n_rounds": N_ROUNDS,
    }
    root = workdir_root / iid

    # Clone N fresh copies — one per peer. Parallelized via threads since
    # `git clone` is I/O-bound (network + disk), so GIL-free concurrency
    # gives near-linear speedup up to network bandwidth. Each clone writes
    # to its own workdir, no shared Python state.
    peer_workdirs: list[Path] = [root / f"peer_{i}" for i in range(N_AGENTS)]
    clone_errors: list[str | None] = [None] * N_AGENTS
    t0 = time.time()

    def _clone_one(i: int) -> None:
        clone_errors[i] = clone_and_checkout(
            instance["repo"], instance["base_commit"], peer_workdirs[i]
        )

    clone_threads = [
        threading.Thread(target=_clone_one, args=(i,)) for i in range(N_AGENTS)
    ]
    for t in clone_threads:
        t.start()
    for t in clone_threads:
        t.join()

    for i, err in enumerate(clone_errors):
        if err:
            summary["error"] = err
            summary["stage"] = f"clone/peer_{i}"
            return summary
    summary["clone_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    try:
        out = solve(instance, peer_workdirs, eval_mode=eval_mode)
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["stage"] = "solve"
        return summary
    summary["solve_s"] = round(time.time() - t0, 1)
    summary.update(out.get("telemetry") or {})

    patch = out.get("patch") or ""
    summary["patch_chars"] = len(patch)
    summary["winner"] = out.get("winner")

    (out_dir / "patches").mkdir(parents=True, exist_ok=True)
    (out_dir / "patches" / f"{iid}.diff").write_text(patch)
    with (out_dir / "predictions.jsonl").open("a") as f:
        f.write(json.dumps(predictions_entry(iid, patch)) + "\n")

    # Trace: per-peer patch + score.
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    with (out_dir / "traces" / f"{iid}.txt").open("w") as f:
        f.write(f"winner: peer {out.get('winner')}  "
                f"resolved={out.get('resolved')}\n\n")
        for s in out.get("per_peer") or []:
            report = s.get("report") or {}
            f.write(
                f"=== peer {s['peer']} ===\n"
                f"  patch_chars={len(s.get('patch') or '')}\n"
                f"  f2p_rate={report.get('f2p_rate')}  p2p_rate={report.get('p2p_rate')}\n"
                f"  resolved={s.get('resolved')}  score={s.get('score')}\n\n"
            )

    per_peer_rates = []
    for s in out.get("per_peer") or []:
        r = s.get("report") or {}
        per_peer_rates.append({
            "peer": s["peer"],
            "patch_chars": len(s.get("patch") or ""),
            "f2p_rate": r.get("f2p_rate"),
            "p2p_rate": r.get("p2p_rate"),
            "resolved": s.get("resolved"),
        })
    summary["per_peer"] = per_peer_rates

    if eval_mode == "none":
        summary["eval"] = "skipped"
        return summary

    winner_id = out.get("winner")
    winner = next(
        (s for s in out.get("per_peer") or [] if s.get("peer") == winner_id),
        None,
    )
    wreport = (winner or {}).get("report") or {}
    summary["eval_mode"] = eval_mode
    summary["f2p_rate"] = wreport.get("f2p_rate", 0.0)
    summary["p2p_rate"] = wreport.get("p2p_rate", 0.0)
    summary["resolved"] = out.get("resolved") or False
    return summary


def run_batch(
    subset: str = "test",
    limit: int | None = None,
    offset: int = 0,
    only: list[str] | None = None,
    workdir_root: Path | None = None,
    out_dir: Path | None = None,
    eval_mode: str = "singularity",
    keep_workdirs: bool = False,
) -> None:
    """Iterate Verified instances and run_one() each. N peer clones per
    instance → disk footprint ≈ N × single. Default keep_workdirs=False
    removes each instance's peer clones after it's scored."""
    import shutil as _sh

    workdir_root = workdir_root or Path(
        f"{os.path.expanduser('~')}/swe_work_decentralized"
    )
    _repo_root = Path(__file__).resolve().parents[4]
    out_dir = out_dir or (_repo_root / "results" / "swe_bench_decentralized")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.jsonl").write_text("")
    results_path = out_dir / "results.jsonl"
    results_path.write_text("")

    instances = load_instances(subset, limit, offset, only)
    print(f"loaded {len(instances)} instance(s) from princeton-nlp/SWE-bench_Verified "
          f"(N={N_AGENTS}, R={N_ROUNDS})", file=sys.stderr)

    for i, inst in enumerate(instances, 1):
        print(
            f"\n[{i}/{len(instances)}] {inst['instance_id']}  "
            f"({inst['repo']}@{inst['base_commit'][:7]})",
            file=sys.stderr,
        )
        summary = run_one(inst, workdir_root, out_dir, eval_mode=eval_mode)
        with results_path.open("a") as f:
            f.write(json.dumps(summary) + "\n")
        compact = {k: v for k, v in summary.items() if k != "per_peer"}
        print(f"  -> {json.dumps(compact)}", file=sys.stderr)

        if not keep_workdirs:
            _sh.rmtree(workdir_root / inst["instance_id"], ignore_errors=True)

    print(f"\ndone: predictions -> {out_dir / 'predictions.jsonl'}", file=sys.stderr)
    print(f"      results     -> {results_path}", file=sys.stderr)
    if eval_mode != "none":
        resolved = sum(
            1 for line in results_path.read_text().splitlines()
            if json.loads(line).get("resolved") is True
        )
        print(f"      resolved ({eval_mode}): {resolved}/{len(instances)}",
              file=sys.stderr)


# Demo / CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Decentralized-topology SWE-bench Verified agent (debate)."
    )
    parser.add_argument("--subset", default="test")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--only", action="append", default=None,
                        metavar="INSTANCE_ID")
    parser.add_argument(
        "--workdir-root",
        default=f"{os.path.expanduser('~')}/swe_work_decentralized",
    )
    _default_out = str(
        Path(__file__).resolve().parents[4] / "results" / "swe_bench_decentralized"
    )
    parser.add_argument("--out-dir", default=_default_out)
    parser.add_argument("--eval", dest="eval_mode", default="singularity",
                        choices=["singularity", "none"])
    parser.add_argument("--keep-workdirs", action="store_true")
    args = parser.parse_args()

    run_batch(
        subset=args.subset,
        limit=args.limit if not args.only else None,
        offset=args.offset,
        only=args.only,
        workdir_root=Path(args.workdir_root).expanduser(),
        out_dir=Path(args.out_dir).expanduser(),
        eval_mode=args.eval_mode,
        keep_workdirs=args.keep_workdirs,
    )
