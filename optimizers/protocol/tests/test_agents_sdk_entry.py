"""OpenAI Agents SDK jobs restart with the SDK first on the path and check it
before the first rollout."""

from __future__ import annotations

import pytest

from optimizers.protocol import run, runner
from optimizers.protocol.errors import JobError
from optimizers.protocol.tests.fakes import fake_cell

JOB = ["--method", "identity", "--model", "qwen", "--seed", "0", "--out", "unused"]


@pytest.mark.parametrize(
    ("dataset", "topology", "restarts"),
    [
        ("hotpotqa", "decentralized_openai_agents", 1),
        ("hotpotqa", "decentralized", 0),
        ("hotpotqa", "sequential_crewai", 0),
        ("no_such_dataset", "decentralized_openai_agents", 0),
    ],
)
def test_only_openai_agents_jobs_restart(monkeypatch, dataset, topology, restarts):
    calls = []
    monkeypatch.setattr(run, "reexec_with_sdk_first", lambda: calls.append(topology))
    run.reexec_for_sdk([*JOB, "--dataset", dataset, "--topology", topology])
    assert len(calls) == restarts


def test_unusable_sdk_stops_an_openai_agents_job_before_its_runtime_is_built(monkeypatch):
    def unavailable():
        raise RuntimeError("OpenAI Agents SDK: vendor/openai_agents is not first on PYTHONPATH.")

    monkeypatch.setattr(runner, "load_agents_sdk", unavailable)
    cell = fake_cell(task="hotpotqa", topology="decentralized", framework="openai_agents")
    with pytest.raises(JobError, match="is not first on PYTHONPATH"):
        runner.AdapterRuntime(cell)
