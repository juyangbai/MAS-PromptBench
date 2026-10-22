"""``core.tasks.swe`` and the SWE teams: checkouts, repository tools, briefs, scoring, records and the batch."""

import importlib
import json
import logging
import subprocess
from pathlib import Path

import pytest
from langchain_core.tools import tool

from core import teams
from core.tasks import swe as task


@pytest.fixture
def repo(tmp_path):
    """A git checkout with one committed Python file."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("".join(f"line {i}\n" for i in range(10)))
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]
    subprocess.run(git + ["init", "-q"], cwd=root, check=True)
    subprocess.run(git + ["add", "."], cwd=root, check=True)
    subprocess.run(git + ["commit", "-qm", "init"], cwd=root, check=True)
    return root


def test_workdir_binds_per_context_and_guards_paths(repo):
    workdir = task.Workdir()
    with pytest.raises(RuntimeError, match="REPO_DIR not set"):
        workdir.get()
    token = workdir.bind(repo)
    assert workdir.resolve("pkg/a.py") == repo / "pkg" / "a.py"
    with pytest.raises(ValueError, match="escapes repo workdir"):
        workdir.resolve("../x")
    workdir.reset(token)
    assert workdir.default is None

    bare = task.Workdir(Path("."), explain_escapes=False)
    bare.bind(repo)
    with pytest.raises(ValueError, match="is not in the subpath|does not start with"):
        bare.resolve("/etc")

    sticky = task.Workdir(sticky=True)
    sticky.bind(repo)
    assert sticky.default == repo


def test_tool_description_is_its_docstring(repo):
    workdir = task.Workdir(repo)
    file_read = tool(task.make_file_read(workdir, task.FILE_READ_DOC))
    assert file_read.name == "file_read"
    assert file_read.description == task.FILE_READ_DOC.rstrip()
    assert file_read.invoke({"path": "pkg/a.py", "offset": 2, "limit": 2}) == "[lines 3-4 of 11]\nline 2\nline 3\n"
    assert file_read.invoke({"path": "nope.py"}).startswith("ERROR: ")


def test_edit_tools_change_the_checkout(repo):
    workdir = task.Workdir(repo)
    replace = task.make_str_replace(workdir, task.STR_REPLACE_DOC_NARROW)
    assert replace("pkg/a.py", "line", "x") == task.AMBIGUOUS.format(path="pkg/a.py", count=10)
    assert replace("pkg/a.py", "missing", "x") == task.NOT_FOUND.format(path="pkg/a.py")
    assert replace("pkg/a.py", "line 3\n", "LINE 3\n") == "replaced 1 occurrence in pkg/a.py"
    preview = task.make_str_replace(workdir, task.STR_REPLACE_DOC_TARGETED, preview=True)
    assert preview("pkg/a.py", "line 4", "L4").endswith("\n  old: 'line 4'\n  new: 'L4'")
    write = task.make_file_write(workdir, task.FILE_WRITE_DOC)
    assert write("new/b.py", "pass\n") == "wrote 5 chars to new/b.py"
    patch = task.compute_patch(repo)
    assert "+LINE 3" in patch and "+L4" in patch


def test_listing_and_search(repo):
    workdir = task.Workdir(repo)
    assert task.make_list_dir(workdir, task.LIST_DIR_DOC)(".").splitlines()[-1] == "d  pkg"
    search = task.make_search_repo(workdir, task.SEARCH_REPO_DOC)
    assert search("line [12]$", "pkg") == "pkg/a.py:2:line 1\npkg/a.py:3:line 2"
    assert search("line", "pkg", max_matches=2).endswith("... [+8 more]")
    assert search("zzz") == "[no matches for 'zzz' in .]"
    assert task.make_search_repo(workdir, task.SEARCH_REPO_DOC, terse=True)("zzz") == "[no matches for 'zzz']"


def test_peer_worktrees_refuse_git_metadata(repo):
    peers = task.PeerWorktrees([repo])
    assert peers.file_read("peer_0", ".git/HEAD") == "ERROR: access to Git metadata is forbidden"
    assert peers.file_read("peer_1", "pkg/a.py").startswith("ERROR: invalid SWE peer identity")
    assert peers.search_repo("peer_0", "line 9") == "pkg/a.py:10:line 9"


def test_issue_brief_truncates_long_issues():
    brief = task.issue_brief("x" * (task.PROBLEM_CHAR_BUDGET + 5), "a__b-1", " hint ", checkout="AT", note="NOTE")
    parts = brief.split("\n\n")
    assert parts[0] == "INSTANCE: a__b-1" and parts[1] == "AT" and parts[-1] == "NOTE"
    assert parts[2].endswith(f"[truncated problem_statement: {task.PROBLEM_CHAR_BUDGET + 5} -> 16000 chars]")
    assert "HINTS (from maintainers):\nhint" in brief


def test_instance_tests_decode_json():
    assert task.instance_tests({"FAIL_TO_PASS": '["a"]', "PASS_TO_PASS": ["b"]}) == (["a"], ["b"])


def test_singularity_report(monkeypatch, tmp_path):
    monkeypatch.setattr(task, "ensure_sif", lambda iid: tmp_path / "x.sif")
    monkeypatch.setenv("SWE_EVAL_LOG_DIR", str(tmp_path / "logs"))
    outputs = {"out": "t1 PASSED\nt2 FAILED\nt3 XFAIL\n"}

    def run(cmd, **kwargs):
        if outputs["out"] is None:
            raise subprocess.TimeoutExpired(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0, outputs["out"], "")

    monkeypatch.setattr(task.subprocess, "run", run)
    inst = {"instance_id": "a__b-1"}
    report = task.run_tests_singularity(inst, "PATCH", ["t1", "t2"], ["t3", "t4"])
    assert report["fail_to_pass"] == {"success": ["t1"], "failure": ["t2"]}
    assert (report["f2p_rate"], report["p2p_rate"]) == (0.5, 0.5)
    assert not task.is_resolved(report) and task.exact_match_score({"f2p_rate": 1.0, "p2p_rate": 1.0}) == 1.0
    outputs["out"] = "__APPLY_FAILED__"
    failed = task.run_tests_singularity(inst, "PATCH", [], ["t3"])
    assert (failed["error"], failed["f2p_rate"], failed["p2p_rate"]) == ("patch apply failed", 0.0, 0.0)
    outputs["out"] = None
    assert task.run_tests_singularity(inst, "PATCH", ["t1"], [])["error"] == "timeout"
    assert task.run_tests_singularity(inst, "PATCH", [], []) == task._report([], [], lambda _: True)


def test_eval_logs_go_to_the_batch_output_dir_unless_overridden(monkeypatch, tmp_path):
    monkeypatch.setattr(task, "ensure_sif", lambda iid: tmp_path / "x.sif")
    monkeypatch.setattr(task.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "t1 PASSED", ""))
    monkeypatch.delenv("SWE_EVAL_LOG_DIR", raising=False)
    inst = {"instance_id": "a__b-1", "repo": "a/b", "base_commit": "0123456789"}

    def run_one(instance):
        task.run_tests_singularity(instance, "PATCH", ["t1"], [])
        return task.record_head(instance)

    for cell in ("cell_a", "cell_b"):
        task.run_batch([inst], run_one, out_dir=tmp_path / cell, eval_mode="singularity")
        assert (tmp_path / cell / "eval_logs" / "eval_a__b-1.log").read_text().startswith("t1 PASSED")
    assert task.eval_log_dir() == task.RESULTS_DIR / "swe_eval_logs"  # outside a batch
    monkeypatch.setenv("SWE_EVAL_LOG_DIR", str(tmp_path / "logs"))
    assert task.eval_log_dir() == tmp_path / "logs"


def test_the_submitted_patch_is_the_vote_over_normalized_patches():
    assert task.select_patch(["p1", "p2", "p2 \n", "", None]) == 1
    assert task.select_patch(["p1", "p2"]) == 0  # a tie: the lowest member
    assert task.select_patch(["", None, "p3"]) == 2  # members without a patch abstain
    assert task.select_patch(["", None]) == 0


def test_only_the_selected_patch_is_evaluated(monkeypatch):
    evaluated = []

    def run_tests_singularity(inst, patch, f2p, p2p):
        evaluated.append(patch)
        return {"f2p_rate": 1.0, "p2p_rate": 0.5}

    monkeypatch.setattr(task, "run_tests_singularity", run_tests_singularity)
    inst = {"FAIL_TO_PASS": [], "PASS_TO_PASS": []}
    selected, empty = {"patch": "p2"}, {"patch": ""}
    task.score_selected(selected, inst)
    task.score_selected(empty, inst)
    assert evaluated == ["p2"]
    assert selected == {"patch": "p2", "report": {"f2p_rate": 1.0, "p2p_rate": 0.5}, "resolved": False, "score": 0.5}
    assert empty == {"patch": "", "resolved": False, "score": 0.0}
    monkeypatch.setattr(task, "run_tests_singularity", lambda *args: 1 / 0)
    failed = {"patch": "p"}
    task.score_selected(failed, inst)
    assert failed == {
        "patch": "p",
        "report": {"error": "ZeroDivisionError: division by zero"},
        "resolved": False,
        "score": 0.0,
    }


def test_independent_runner_evaluates_only_the_voted_replica(monkeypatch):
    runner = importlib.import_module("topologies.independent.swe.langgraph_swe")
    patches = ["resolving", "other", "other", ""]

    class Graph:
        def compile(self):
            return self

        async def ainvoke(self, state):
            return {"answers": [{"agent_id": i, "seed": i, "patch": p} for i, p in reversed(list(enumerate(patches)))]}

    evaluated = []

    def run_tests_singularity(inst, patch, f2p, p2p):
        evaluated.append(patch)
        rate = 1.0 if patch == "resolving" else 0.0
        return {"f2p_rate": rate, "p2p_rate": 1.0}

    monkeypatch.setattr(runner, "build_graph", Graph)
    monkeypatch.setattr(task, "run_tests_singularity", run_tests_singularity)
    out = runner.solve({"FAIL_TO_PASS": [], "PASS_TO_PASS": []})
    assert (out["winner"], out["patch"], out["resolved"]) == (1, "other", False)
    assert evaluated == ["other"]
    assert [a.get("resolved") for a in out["per_agent"]] == [None, False, None, None]
    assert runner.solve({}, eval_mode="none")["resolved"] is None and evaluated == ["other"]


def test_eval_fields():
    assert task.eval_fields("none", None) == {"eval": "skipped"}
    assert task.eval_fields("singularity", None) == {
        "eval_mode": "singularity",
        "resolved": False,
        "f2p_rate": 0.0,
        "p2p_rate": 0.0,
    }
    report = {
        "fail_to_pass": {"success": [], "failure": ["t1"]},
        "pass_to_pass": {"success": ["t2"], "failure": []},
        "f2p_rate": 0.0,
        "p2p_rate": 1.0,
        "error": "e",
    }
    fields = task.eval_fields("local", report, eval_s=1.5)
    assert list(fields) == [
        "eval_mode",
        "eval_s",
        "f2p_rate",
        "p2p_rate",
        "resolved",
        "f2p_failures",
        "p2p_failures",
        "eval_error",
    ]
    out = {"winner": 1, "resolved": None}
    members = [{"peer": 1, "report": {"f2p_rate": 1.0}}]
    assert task.winner_eval_fields("singularity", out, members, "peer") == {
        "eval_mode": "singularity",
        "f2p_rate": 1.0,
        "p2p_rate": 0.0,
        "resolved": False,
    }


def test_batch_writes_records_and_predictions(tmp_path, capsys, caplog):
    caplog.set_level(logging.INFO, logger="core")
    out_dir = tmp_path / "out"
    inst = {"instance_id": "a__b-1", "repo": "a/b", "base_commit": "0123456789"}

    def run_one(instance):
        entry = task.predictions_entry(instance["instance_id"], "P", "model")
        task.write_artifacts(out_dir, instance["instance_id"], "P", entry, task.sections([("STAGE", "text")]))
        return task.record_head(instance, resolved=True, per_peer=[1])

    copy = tmp_path / "copy.jsonl"
    records = task.run_batch(
        [inst], run_one, out_dir=out_dir, eval_mode="singularity", omit="per_peer", predictions=copy
    )
    assert records == [{**task.record_head(inst), "resolved": True, "per_peer": [1]}]
    assert [json.loads(line) for line in (out_dir / "results.jsonl").read_text().splitlines()] == records
    assert copy.read_text() == (out_dir / "predictions.jsonl").read_text()
    assert (out_dir / "traces" / "a__b-1.txt").read_text() == "=== STAGE ===\ntext\n\n"
    assert "[1/1] a__b-1  (a/b@0123456)" in caplog.text and "per_peer" not in caplog.text
    assert "resolved (singularity): 1/1" in capsys.readouterr().err


def test_cli_options_keep_the_runner_defaults(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(task, "load_instances", lambda subset, limit, offset, only: seen.append((limit, only)) or [])
    for argv in (["--skip-eval"], ["--only", "a__b-1"]):
        status = task.cli_main(
            argv,
            description="d",
            run_batch=dict,
            default_out_dir=tmp_path,
            eval_modes=("singularity", "none"),
            skip_eval=True,
        )
        assert status == 1
    assert seen == [(task.DEFAULT_LIMIT, None), (None, ["a__b-1"])]


@pytest.mark.parametrize("size", teams.TEAM_SIZES)
def test_team_specs(size):
    sequential = teams.spec("sequential", task.DATASET, size)
    assert all("{task_brief}" in stage.task for stage in sequential.stages)
    centralized = teams.spec("centralized", task.DATASET, size)
    assert set(centralized.manager_tools) == {"file_read", "list_dir", "search_repo"}
    tools = {worker.role: worker.tools for worker in centralized.workers}
    assert tools["patcher_worker"] == ("file_read", "str_replace")
    assert all(
        set(names) <= {"file_read", "list_dir", "search_repo", "str_replace", "shell_exec"} for names in tools.values()
    )
    suffixes = {worker.role: worker.prompt_suffix for worker in centralized.workers if worker.prompt_suffix}
    assert "patcher_worker" in suffixes and set(suffixes) <= {"patcher_worker", "tester_worker"}
    assert centralized.max_turns == (15 if size == 2 else 30)
    assert teams.spec("independent", task.DATASET, size).n_agents == size
