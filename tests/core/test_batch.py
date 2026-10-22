"""``core.batch``: the shared batch loop writes records as JSONL and summarizes them."""

import json
import logging
import sys

import pytest

from core import batch


def test_attempt_turns_a_failure_into_an_error():
    out, latency_s, error = batch.attempt(lambda: 1 / 0)
    assert out == {"answer": None} and latency_s >= 0
    assert error == "ZeroDivisionError: division by zero"
    assert batch.attempt(lambda: {"answer": 3})[0] == {"answer": 3}
    with pytest.raises(ZeroDivisionError):
        batch.attempt(lambda: 1 / 0, propagate=True)


def test_run_batch_writes_records_and_summary(tmp_path, capsys, caplog):
    caplog.set_level(logging.INFO, logger="core.batch")
    out_path = tmp_path / "nested" / "predictions.jsonl"
    summary = batch.run_batch(
        [{"id": "a"}, {"id": "b"}],
        lambda i, inst: {"id": inst["id"], "index": i},
        summarize=lambda records: {"n": len(records)},
        out_path=out_path,
        progress=lambda i, n, rec, done: f"{i + 1}/{n} {rec['id']} {len(done)}",
        banner=lambda s: f"done {s['n']}",
    )
    assert list(summary) == ["n", "total_s", "per_instance"]
    assert summary["per_instance"] == [{"id": "a", "index": 0}, {"id": "b", "index": 1}]
    assert [json.loads(line) for line in out_path.read_text().splitlines()] == summary["per_instance"]
    assert caplog.messages == ["1/2 a 1", "2/2 b 2"]  # progress is logged
    assert capsys.readouterr().out == "done 2\n"  # the report is printed


def test_run_batch_starts_its_files_afresh_and_returns_the_raw_summary(tmp_path, capsys, caplog):
    caplog.set_level(logging.INFO, logger="core.batch")
    out_path, extra = tmp_path / "results.jsonl", tmp_path / "predictions.jsonl"
    out_path.write_text("old\n")
    extra.write_text("old\n")
    summary = batch.run_batch(
        [{"id": "ü"}],
        lambda i, inst: {"id": inst["id"]},
        summarize=lambda records: {"n": len(records)},
        out_path=out_path,
        outputs=[batch.Output(extra, lambda inst, rec: {"pred": rec["id"]})],
        ensure_ascii=False,
        header=lambda i, n, inst: f"[{i + 1}/{n}] {inst['id']}",
        banner=lambda s: f"done {s['n']}",
        stream=sys.stderr,
        raw_summary=True,
    )
    assert summary == {"n": 1}
    assert out_path.read_text() == '{"id": "ü"}\n' and extra.read_text() == '{"pred": "ü"}\n'
    assert caplog.messages == ["[1/1] ü"]
    captured = capsys.readouterr()
    assert captured.err.endswith("done 1\n") and captured.out == ""


def test_run_batch_raises_after_writing_when_every_row_failed_on_infrastructure(tmp_path, capsys):
    out_path = tmp_path / "results.jsonl"
    down = {"answer": None, "error": "APIConnectionError: Connection error."}
    with pytest.raises(batch.InfrastructureFailure, match=r"every row \(2\) .* APIConnectionError"):
        batch.run_batch(
            [{"id": "a"}, {"id": "b"}],
            lambda i, inst: dict(down, id=inst["id"]),
            summarize=lambda records: {"n": len(records)},
            out_path=out_path,
            banner=lambda s: f"done {s['n']}",
        )
    assert len(out_path.read_text().splitlines()) == 2 and capsys.readouterr().out == "done 2\n"
    task_failure = {"answer": None, "error": "GraphRecursionError: Recursion limit of 12 reached"}
    rows = [down, task_failure]
    summary = batch.run_batch(rows, lambda i, row: row, summarize=lambda records: {}, verbose=False)
    assert summary["per_instance"] == rows
    assert batch.run_batch([], lambda i, row: row, summarize=lambda records: {}, raw_summary=True) == {}


def test_write_trace(tmp_path):
    path = tmp_path / "traces" / "0001.txt"
    batch.write_trace(path, [("PLANNER", "plan"), ("SOLVER", "answer")])
    assert path.read_text() == "=== PLANNER ===\nplan\n\n=== SOLVER ===\nanswer\n\n"
