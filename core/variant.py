"""Runner variants: a runner run under another module with preset parameters.

A team-size runner (``teamsizes/<topology>/<dataset>/<dataset>_r<r>.py``) is the
topologies/ LangGraph runner with a different team. Rather than import that
runner (a single shared module object), the variant module executes the runner's
source in its own namespace with ``TEAM_SIZE`` preset. The variant therefore owns
a full set of module-level hooks (``_load_prompt``, ``_build_llm``, ``N_AGENTS``,
``solve``, ``main``, ...) that callers such as the prompt optimizers can patch
or isolate exactly as for any other runner module, and running the variant as a
script runs the runner's command line.

The parameters are ``TEAM_SIZE`` (a size of :data:`core.teams.TEAM_SIZES`),
``TOPOLOGY`` (the multi-agent topology a shared runner plays, e.g. the API-Bank
and ToolHop team sizes), ``STYLE`` (the label of the records, e.g. the CrewAI
and AutoGen API-Bank / ToolHop runners) and ``COMMUNICATION_FORMAT`` (a format of
:data:`core.communication.FORMATS`: how the agents report to each other). A
runner reads a preset parameter with ``globals().get(<NAME>)`` (or
``globals()[<NAME>]`` when it is required).

:func:`load` runs a runner in a variant module's own namespace; :func:`module`
runs it as a module of its own, for a caller that keeps its own names next to
the runner's (a communications pair wraps the runner's ``solve``).
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

from core import teams
from core.communication import FORMATS

TOPOLOGIES = ("independent", "decentralized", "sequential", "centralized")
_VALID = {
    "TEAM_SIZE": lambda value: value in teams.TEAM_SIZES,
    "TOPOLOGY": lambda value: value in TOPOLOGIES,
    "STYLE": lambda value: isinstance(value, str) and bool(value),
    "COMMUNICATION_FORMAT": lambda value: value in FORMATS,
}
PARAMETERS = frozenset(_VALID)


def load(namespace: dict, runner: str, **params) -> None:
    """Execute module ``runner`` in ``namespace`` (a variant's ``globals()``) with ``params`` preset."""
    _check(params)
    spec = importlib.util.find_spec(runner)
    if spec is None or spec.origin is None or spec.loader is None:
        raise ModuleNotFoundError(runner)
    code = compile(spec.loader.get_source(runner), spec.origin, "exec", dont_inherit=True)
    doc = namespace.get("__doc__")
    namespace.update(params)
    exec(code, namespace)
    namespace["__doc__"] = doc


def module(runner: str, **params) -> ModuleType:
    """Module ``runner`` with ``params`` preset, executed once per process as a module of its own.

    It is ``sys.modules["<runner>[<NAME>=<value>, ...]"]``: registered there so that the
    runner's classes resolve their annotations (e.g. a LangGraph state), and returned
    to every later call with the same parameters, as an import would be.
    """
    _check(params)
    name = f"{runner}[{', '.join(f'{key}={value}' for key, value in sorted(params.items()))}]"
    if name not in sys.modules:
        sys.modules[name] = ModuleType(name)
        try:
            load(vars(sys.modules[name]), runner, **params)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


def _check(params: dict) -> None:
    unknown = set(params) - PARAMETERS
    if unknown:
        raise TypeError(f"unknown runner parameters: {sorted(unknown)}")
    invalid = {name: value for name, value in params.items() if not _VALID[name](value)}
    if invalid:
        raise ValueError(f"invalid runner parameters: {invalid}")
