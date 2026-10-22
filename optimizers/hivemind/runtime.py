"""Coalition execution through the protocol runner's ``optimizer_control`` hook.

The optimizer marks a bundle with ``metadata["optimizer_control"]``, the
canonical coalition request of :class:`~.topology.CoalitionGame`. The runner
(``optimizers.protocol.runner``) hands such a bundle to the registered
:class:`CoalitionExecutionHook`, which

1. validates the request: HiveMind cell inside the frozen scope (LangGraph,
   free-form, team size 4, hotpotqa/lcb/bfcl) and an exact canonical partition
   of the cell's players;
2. builds the adapter that executes the coalition: for centralized cells the
   masked subclass of the cell's own runtime class (:mod:`.coalition`),
   otherwise the native-player mask of :func:`.topology.build_topology_runner`;
3. after the rollout, verifies it: no masked worker may appear as a message
   source, and the evidence the adapter recorded while executing (bound
   delegation tools, routable workers, executed players) must equal the
   evidence expected for the request. An unmasked adapter records none.

Any violation raises ``RunnerContractError``, which the runner treats as an
uncharged infrastructure failure. The acknowledgement returned by ``verify``
is stored in ``record.metadata["runtime_metadata"]["execution_control"]``;
the optimizer additionally refuses every record without the exact
acknowledgement. An empty non-centralized coalition is the canonical
abstention and is the only execution allowed to make zero model calls.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from optimizers.protocol.adapter_output import mapping_messages
from optimizers.protocol.errors import RunnerContractError
from optimizers.protocol.runner import register_execution_hook
from optimizers.protocol.schema import content_hash

from .topology import TOPOLOGY_CONTROL_SCHEMA, build_topology_runner, coalition_game, expected_control_evidence

CONTROL_KEY = "optimizer_control"
SUPPORTED_TASKS = frozenset({"bfcl", "hotpotqa", "lcb"})
DECENTRALIZED_ROUNDS = 2


def runtime_support(cell: Any, roles: Sequence[str]) -> tuple[bool, str]:
    """Classify the explicit topology-preserving coalition adaptations."""
    if getattr(cell, "method", None) != "hivemind":
        return False, "not_hivemind"
    if (
        getattr(cell, "framework", None) != "langgraph"
        or getattr(cell, "communication", None) != "freeform"
        or getattr(cell, "team_size", None) != 4
        or getattr(cell, "task", None) not in SUPPORTED_TASKS
    ):
        return False, "outside_frozen_hivemind_scope"
    try:
        coalition_game(cell.topology, roles, cell.team_size)
    except ValueError:
        return False, "topology_runtime_role_contract_invalid"
    return True, f"{cell.topology}_native_player_coalition"


def _string_sequence(value: Any, *, field: str) -> list[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise RunnerContractError(f"HiveMind optimizer_control {field} is invalid")
    items = list(value)
    if len(items) != len(set(items)):
        raise RunnerContractError(f"HiveMind optimizer_control {field} contains duplicates")
    return items


def validated_control(cell: Any, roles: Sequence[str], raw: Any) -> dict[str, Any]:
    """The canonical control for ``raw``; anything else is a contract error."""
    if not isinstance(raw, Mapping):
        raise RunnerContractError("HiveMind optimizer_control must be a mapping")
    supported, reason = runtime_support(cell, roles)
    if not supported:
        raise RunnerContractError(f"HiveMind coalition execution is unavailable: {reason}")
    game = coalition_game(cell.topology, roles, cell.team_size)
    active = _string_sequence(raw.get("active_workers"), field="active_workers")
    masked = _string_sequence(raw.get("masked_workers"), field="masked_workers")
    workers = set(game.players)
    if set(active) & set(masked) or set(active) | set(masked) != workers:
        raise RunnerContractError("HiveMind active/masked workers are not an exact role partition")
    canonical = game.control(active)
    normalized = {**dict(raw), "active_workers": sorted(active), "masked_workers": sorted(masked)}
    if normalized != canonical:
        raise RunnerContractError("HiveMind optimizer_control is not canonical for this topology")
    return canonical


def verify_mask(value: Any, control: Mapping[str, Any]) -> None:
    """No masked worker may have produced a message."""
    if not control["masked_workers"]:
        return
    masked = set(control["masked_workers"])
    observed = {
        str(message.get("source") or message.get("role") or message.get("name")) for message in mapping_messages(value)
    }
    violations = sorted(masked & observed)
    if violations:
        raise RunnerContractError(f"HiveMind masked workers executed: {violations}")


def control_ack(
    adapter: Any, control: Mapping[str, Any], *, implementation_id: str, output: Any = None
) -> dict[str, Any]:
    """Acknowledge the installed coalition from the adapter's own execution evidence."""
    getter = getattr(adapter, "coalition_control_evidence", None)
    if not callable(getter):
        raise RunnerContractError("HiveMind runtime did not expose installed coalition evidence")
    try:
        raw = getter()
    except Exception as exc:
        raise RunnerContractError(f"HiveMind coalition evidence unavailable: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise RunnerContractError("HiveMind installed coalition evidence is not a mapping")
    expected = expected_control_evidence(control)
    if dict(raw) != expected:
        raise RunnerContractError("HiveMind installed graph/tool evidence differs from requested control")
    nested = output.get("runner_output") if isinstance(output, Mapping) else None
    abstention = (
        control["schema"] == TOPOLOGY_CONTROL_SCHEMA
        and not control["active_workers"]
        and isinstance(nested, Mapping)
        and nested.get("coalition_abstention") is True
    )
    return {
        "schema": control["schema"],
        **expected,
        "request_sha256": content_hash(dict(control)),
        "runtime_implementation_id": implementation_id,
        "allow_zero_model_calls": abstention,
    }


class CoalitionExecutionHook:
    """Build the masked adapter for one request, then verify and acknowledge it.

    ``build_adapter`` and ``verify`` of one request are paired by the request
    object; ``verify`` refuses a request this hook did not build.
    """

    def __init__(self) -> None:
        self._built: dict[int, tuple[Any, Any, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def build_adapter(self, runtime: Any, request: Any, control: Any) -> Any:
        cell = request.cell
        canonical = validated_control(cell, runtime.native_roles, control)
        kwargs = dict(getattr(runtime, "adapter_kwargs", None) or {})
        if cell.topology == "centralized":
            from .coalition import build_masked_runner

            try:
                adapter = build_masked_runner(runtime.adapter_class, frozenset(canonical["masked_workers"]), **kwargs)
            except ValueError as exc:
                raise RunnerContractError(f"HiveMind coalition execution is unavailable: {exc}") from exc
        else:
            adapter = build_topology_runner(
                runtime.adapter_class,
                canonical,
                n_agents=cell.team_size,
                n_rounds=DECENTRALIZED_ROUNDS,
                adapter_kwargs=kwargs,
            )
        if tuple(sorted(adapter.roles())) != tuple(sorted(runtime.native_roles)):
            raise RunnerContractError("HiveMind masked adapter roles differ from the canonical runtime")
        with self._lock:
            self._built = {key: entry for key, entry in self._built.items() if entry[0]() is not None}
            self._built[id(request)] = (weakref.ref(request), adapter, canonical)
        return adapter

    def verify(self, runtime: Any, request: Any, control: Any, output: Any) -> Mapping[str, Any]:
        with self._lock:
            entry = self._built.pop(id(request), None)
        if entry is None or entry[0]() is not request:
            raise RunnerContractError("HiveMind coalition rollout was not built by this execution hook")
        _, adapter, canonical = entry
        if validated_control(request.cell, runtime.native_roles, control) != canonical:
            raise RunnerContractError("HiveMind optimizer_control changed during the rollout")
        verify_mask(output, canonical)
        return control_ack(adapter, canonical, implementation_id=str(runtime.implementation_id), output=output)


@contextmanager
def coalition_execution(hook: Any = None) -> Iterator[Any]:
    """Register the coalition hook for the duration of one optimization."""
    hook = hook if hook is not None else CoalitionExecutionHook()
    register_execution_hook(CONTROL_KEY, hook)
    try:
        yield hook
    finally:
        register_execution_hook(CONTROL_KEY, None)


__all__ = [
    "CONTROL_KEY",
    "CoalitionExecutionHook",
    "SUPPORTED_TASKS",
    "coalition_execution",
    "control_ack",
    "runtime_support",
    "validated_control",
    "verify_mask",
]
