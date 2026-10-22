"""``core.variant`` runs a runner's code in another namespace, or as a module of its own, with preset parameters."""

import sys
import types

import pytest

from core import variant


def load(**params):
    module = types.ModuleType("teamsizes_probe", "Variant docstring.")
    variant.load(module.__dict__, "topologies.sequential.langgraph.math.langgraph_math", **params)
    return module


def test_variant_has_its_own_team_and_hooks():
    module = load(TEAM_SIZE=8)
    assert module.TEAM.size == 8
    assert module.DEFAULT_OUT_DIR.name == "math_sequential_r8"
    assert module.__doc__ == "Variant docstring."
    assert module._load_prompt.__globals__ is module.__dict__


def test_unknown_parameter():
    with pytest.raises(TypeError):
        load(COLOR="blue")


def test_parameter_values_are_validated():
    with pytest.raises(ValueError):
        load(TEAM_SIZE=3)
    with pytest.raises(ValueError):
        variant.load({}, "teamsizes.toolhop_common", TOPOLOGY="hierarchical", TEAM_SIZE=2)
    with pytest.raises(ValueError):
        variant.load({}, "topologies.sequential.langgraph.toolhop.langgraph_toolhop", STYLE="")


def test_topology_and_style_presets():
    team = types.ModuleType("toolhop_probe")
    variant.load(team.__dict__, "teamsizes.toolhop_common", TOPOLOGY="sequential", TEAM_SIZE=2)
    assert (team.ROLE, team.STYLE) == ("verifier", "sequential_toolhop_r2")
    crew = types.ModuleType("crew_probe")
    variant.load(crew.__dict__, "topologies.sequential.langgraph.toolhop.langgraph_toolhop", STYLE="sequential_crewai")
    assert crew.STYLE == "sequential_crewai" and crew.TOPOLOGY == "sequential"


def test_module_is_registered_once_per_parameters():
    runner = "topologies.sequential.langgraph.math.langgraph_math"
    eight = variant.module(runner, TEAM_SIZE=8)
    assert variant.module(runner, TEAM_SIZE=8) is eight and sys.modules[eight.__name__] is eight
    assert eight.__name__ == f"{runner}[TEAM_SIZE=8]" and eight.TEAM.size == 8
    assert variant.module(runner, TEAM_SIZE=2) is not eight
    with pytest.raises(ValueError):
        variant.module(runner, COMMUNICATION_FORMAT="yaml")
    assert f"{runner}[COMMUNICATION_FORMAT=yaml]" not in sys.modules
