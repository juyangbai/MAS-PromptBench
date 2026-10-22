"""``core.teams`` loads one consistent TeamSpec per (topology, dataset, r)."""

import pytest

from core import teams


@pytest.mark.parametrize("size", teams.TEAM_SIZES)
def test_math_team_sizes(size):
    sequential = teams.spec("sequential", "math", size)
    centralized = teams.spec("centralized", "math", size)
    assert len(sequential.stages) == size
    assert len(centralized.workers) == size - 1
    assert centralized.roles[0] == centralized.manager
    for topology in ("independent", "decentralized"):
        assert teams.spec(topology, "math", size).n_agents == size


def test_default_size_is_the_topologies_runner_team():
    assert teams.spec("sequential", "math") == teams.spec("sequential", "math", teams.BASE_SIZE)
    assert teams.spec("sequential", "math").roles == ("decomposer", "computer", "checker", "verifier")


def test_delegation_note_names_the_workers():
    assert teams.spec("centralized", "math", 2).delegation_note.endswith("The only worker is: computation_worker.")
    assert teams.spec("centralized", "math", 4).delegation_note.endswith(
        "The three workers are: decomposer_worker, computation_worker, verifier_worker."
    )


def test_workers_have_their_own_tools_and_prompt_suffix():
    math = teams.spec("centralized", "math")
    assert all(worker.tools == ("calculator",) and worker.prompt_suffix == "" for worker in math.workers)
    gpqa = {worker.role: worker.tools for worker in teams.spec("centralized", "gpqa", 10).workers}
    assert gpqa["confidence_reporter_worker"] == () and gpqa["solver_worker"] == ("calculator",)
    swe = {worker.role: worker for worker in teams.spec("centralized", "swe").workers}
    assert swe["patcher_worker"].tools == ("file_read", "str_replace")
    assert swe["patcher_worker"].prompt_suffix.endswith("the manager decides when the task is done.")


def test_recursion_limit_is_the_same_at_every_size():
    for topology in ("independent", "decentralized", "sequential", "centralized"):
        limits = {teams.spec(topology, "hotpotqa", size).recursion_limit for size in teams.TEAM_SIZES}
        assert len(limits) == 1
    assert teams.spec("decentralized", "bfcl").recursion_limit is None


def test_unknown_team():
    assert not teams.defined("sequential", "no_such_dataset")
    with pytest.raises(KeyError):
        teams.spec("sequential", "no_such_dataset")
    with pytest.raises(ValueError):
        teams.spec("sequential", "math", 3)
