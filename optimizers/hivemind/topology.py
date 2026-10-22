"""Topology coalitions: native players, original shared prompt parameters.

Centralized has a fixed manager; other games do not invent one. Subsets retain
native order, communication, and selection. Empty noncentralized coalitions
abstain without model calls; full coalitions execute the native implementation.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

TOPOLOGY_ADAPTATION_ID = "hivemind-topology-coalitions"
CENTRALIZED_CONTROL_SCHEMA = "mas-promptbench-hivemind-centralized-coalition"
TOPOLOGY_CONTROL_SCHEMA = "mas-promptbench-hivemind-coalition"
CENTRALIZED_ACK_SCHEMA = "mas-promptbench-hivemind-centralized-coalition-ack"
TOPOLOGY_ACK_SCHEMA = "mas-promptbench-hivemind-coalition-ack"
EMPTY_COALITION = "canonical-abstention-zero-model-calls"


@dataclass(frozen=True)
class CoalitionGame:
    """Players of one topology's coalition game and the roles whose prompts they share."""

    topology: str
    manager: str | None
    players: tuple[str, ...]
    player_roles: Mapping[str, str]

    def control(self, active: Sequence[str]) -> dict[str, Any]:
        """Execution control that activates exactly ``active`` players."""
        value = {
            "schema": CENTRALIZED_CONTROL_SCHEMA,
            "manager": self.manager,
            "active_workers": sorted(active),
            "masked_workers": sorted(set(self.players) - set(active)),
        }
        if self.topology != "centralized":
            value.update(schema=TOPOLOGY_CONTROL_SCHEMA, topology=self.topology, player_roles=dict(self.player_roles))
        return value

    def state(self) -> dict[str, Any]:
        """JSON description of the game recorded in checkpoints."""
        return {
            "adaptation_id": TOPOLOGY_ADAPTATION_ID,
            "topology": self.topology,
            "manager": self.manager,
            "players": list(self.players),
            "player_roles": dict(self.player_roles),
        }


def coalition_game(topology: str, roles: Sequence[str], team_size: int = 4) -> CoalitionGame:
    """The coalition game of a grid cell's topology and roles."""
    roles = tuple(roles)
    if topology == "centralized":
        managers = [r for r in roles if r == "manager" or (r.startswith("manager_r") and r[9:].isdigit())]
        if len(managers) != 1 or len(roles) < 2:
            raise ValueError("centralized game requires exactly one manager and workers")
        manager = managers[0]
        players = tuple(sorted(r for r in roles if r != manager))
        return CoalitionGame(topology, manager, players, {p: p for p in players})
    if topology == "sequential" and len(roles) == team_size:
        players = tuple(sorted(roles))
        return CoalitionGame(topology, None, players, {p: p for p in players})
    if topology in {"independent", "decentralized"} and len(roles) == 1:
        players = tuple(f"{roles[0]}::{i}" for i in range(team_size))
        return CoalitionGame(topology, None, players, {p: roles[0] for p in players})
    raise ValueError("topology role parameterization differs from the frozen grid cell")


def expected_control_evidence(control: Mapping[str, Any]) -> dict[str, Any]:
    """The acknowledgement a runtime must return for a control value."""
    active = sorted(control["active_workers"])
    centralized = control["schema"] == CENTRALIZED_CONTROL_SCHEMA
    value = {
        "ack_schema": CENTRALIZED_ACK_SCHEMA if centralized else TOPOLOGY_ACK_SCHEMA,
        "applied": True,
        "manager": control["manager"],
        "active_workers": active,
        "masked_workers": sorted(control["masked_workers"]),
        "routable_workers": active,
        "bound_delegation_tools": [f"delegate_to_{p}" for p in active] if centralized else [],
        "enforcement": "delegation-tool-and-route-mask" if centralized else f"{control['topology']}-native-player-mask",
    }
    if not centralized:
        value.update(
            topology=control["topology"], player_roles=dict(control["player_roles"]), empty_coalition=EMPTY_COALITION
        )
    return value


def _abstention() -> dict[str, Any]:
    return {
        "model_output": [],
        "answer": None,
        "answer_text": "",
        "code": None,
        "winner": None,
        "buckets": [],
        "raw": "",
        "messages": [],
        "runner_output": {
            "messages": [],
            "by_stage": {},
            "per_agent": [],
            "per_peer": [],
            "coalition_abstention": True,
        },
    }


def build_topology_runner(
    base_cls: type,
    control: Mapping[str, Any],
    *,
    n_agents: int = 4,
    n_rounds: int = 2,
    adapter_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Install masks inside retained graphs, preserving the mutable prompt API.

    ``adapter_kwargs`` are the runtime's own constructor arguments (filtered
    by the constructor signature); ``n_agents``/``n_rounds`` and
    ``keep_messages=True`` are then set as in the native coalition runtime.
    """
    topology = control["topology"]
    active = tuple(control["active_workers"])
    masked = frozenset(control["masked_workers"])
    active_ids = tuple(int(p.rsplit("::", 1)[1]) for p in active) if topology != "sequential" else ()
    pm_name = "_patched_module" if hasattr(base_cls, "_patched_module") else "patched_module"
    base_pm = getattr(base_cls, pm_name, None)

    class TopologyMasked(base_cls):
        def __init__(self):
            signature = inspect.signature(base_cls)
            variadic = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
            kwargs = {
                name: value
                for name, value in dict(adapter_kwargs or {}).items()
                if variadic or name in signature.parameters
            }
            for name, value in (("n_agents", n_agents), ("n_rounds", n_rounds)):
                if name in signature.parameters:
                    kwargs[name] = value
            if "keep_messages" in signature.parameters:
                kwargs["keep_messages"] = True
            super().__init__(**kwargs)
            self._hive_evidence = None
            self.masked_workers = masked

        def coalition_control_evidence(self):
            if self._hive_evidence is None:
                raise RuntimeError("coalition mask has not executed")
            return dict(self._hive_evidence)

        def _make_node(self, role, seed, prior_roles):
            if role in masked:
                return lambda state: {"by_stage": {}, "messages": []}
            return super()._make_node(role, seed, prior_roles)

        def _fan_out(self, state):
            sends = super()._fan_out(state)
            for index, send in enumerate(sends):
                send.arg["agent_id"] = active_ids[index]
                send.arg["seed"] = active_ids[index]
            return sends

        def _runtime_failure_output(self, instance, exc):
            # An incomplete graph cannot attest an executed coalition. Preserve
            # the original native error for protocol classification/accounting.
            raise exc

        def run_example(self, example):
            if not active:
                self._hive_evidence = expected_control_evidence(control)
                return _abstention()
            old_n = getattr(self, "n_agents", None)
            old_factory = getattr(self, "model_factory", None)
            if masked and topology in {"independent", "decentralized"}:
                self.n_agents = len(active_ids)
                if topology == "decentralized" and old_factory is not None:
                    self.model_factory = lambda seed: old_factory(active_ids[seed])
            try:
                out = super().run_example(example)
                if masked and topology == "sequential":
                    _sequential_fallback(self, out)
                if masked and topology == "decentralized":
                    _restore_peer_ids(out, active_ids)
                self._hive_evidence = _observed_control_evidence(out, control)
                return out
            finally:
                if old_n is not None:
                    self.n_agents = old_n
                if old_factory is not None:
                    self.model_factory = old_factory

    if base_pm is not None:

        @contextmanager
        def patched(self, module):
            with base_pm(self, module):
                changes = {}

                def patch(name, value):
                    changes[name] = getattr(module, name)
                    setattr(module, name, value)

                if masked and topology == "sequential":
                    for name in ("_make_tool_node", "_make_plain_node"):
                        original = getattr(module, name)

                        def factory(role, *args, _original=original, **kwargs):
                            if role in masked:
                                return lambda state: {"by_stage": {}, "messages": []}
                            return _original(role, *args, **kwargs)

                        patch(name, factory)
                elif masked and topology == "independent":
                    original = module._fan_out

                    def fan_out(state):
                        sends = original(state)
                        for index, send in enumerate(sends):
                            send.arg["agent_id"] = active_ids[index]
                            send.arg["seed"] = active_ids[index]
                        return sends

                    patch("_fan_out", fan_out)
                try:
                    yield
                finally:
                    for name, previous in changes.items():
                        setattr(module, name, previous)

        setattr(TopologyMasked, pm_name, patched)
    return TopologyMasked()


def _observed_control_evidence(out: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    """Acknowledge the executed native player set, never an echoed request."""
    nested = out.get("runner_output", out)
    if control["topology"] == "sequential":
        observed = set(nested.get("by_stage") or out.get("by_role") or {})
    else:
        independent = control["topology"] == "independent"
        records = nested.get("per_agent" if independent else "per_peer") or []
        key = "agent_id" if independent else "peer"
        prefix = next(iter(control["player_roles"])).rsplit("::", 1)[0]
        observed = {f"{prefix}::{record[key]}" for record in records}
    all_players = set(control["player_roles"])
    if observed != set(control["active_workers"]):
        raise RuntimeError("HiveMind executed native players differ from the requested coalition")
    installed = {**control, "active_workers": sorted(observed), "masked_workers": sorted(all_players - observed)}
    return expected_control_evidence(installed)


def _sequential_fallback(adapter: Any, out: dict[str, Any]) -> None:
    """Absent stages add no output; the last active stage supplies the answer."""
    nested = out.get("runner_output", out)
    stages = nested.get("by_stage") or out.get("by_role") or {}
    raw = next((str(stages[role]) for role in reversed(adapter.roles()) if role in stages and stages[role]), "")
    if adapter.dataset == "bfcl":
        from optimizers.bridge.adapters.bfcl_common import extract_canonical

        out["model_output"] = extract_canonical(raw) or []
        out["winner"] = next((role for role in reversed(adapter.roles()) if role in stages), None)
    else:
        from optimizers.bridge.adapters.module_common import import_isolated_real_module

        module = import_isolated_real_module(adapter.module_name)
        parsed = module.extract_answer(raw) if adapter.dataset == "hotpotqa" else module.extract_code(raw)
        out.update(answer=parsed, answer_text=raw, raw=raw)
        nested.update(raw=raw)
        if adapter.dataset == "hotpotqa":
            nested["answer"] = parsed
        else:
            out["code"] = parsed
            nested["code"] = parsed


def _restore_peer_ids(out: dict[str, Any], active_ids: Sequence[int]) -> None:
    for container in (out, out.get("runner_output") or {}):
        for entry in container.get("per_peer") or []:
            for key in ("peer", "peer_id"):
                if isinstance(entry.get(key), int):
                    entry[key] = active_ids[entry[key]]
        if isinstance(container.get("winner"), int):
            container["winner"] = active_ids[container["winner"]]


__all__ = [
    "CENTRALIZED_ACK_SCHEMA",
    "CENTRALIZED_CONTROL_SCHEMA",
    "CoalitionGame",
    "EMPTY_COALITION",
    "TOPOLOGY_ACK_SCHEMA",
    "TOPOLOGY_ADAPTATION_ID",
    "TOPOLOGY_CONTROL_SCHEMA",
    "build_topology_runner",
    "coalition_game",
    "expected_control_evidence",
]
