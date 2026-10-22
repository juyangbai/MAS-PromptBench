"""Lazy registry of optimizer methods: name -> ``module:Class``.

Nothing is imported until a method is requested, so the registry lists
methods whose modules may live elsewhere or not exist yet. ``identity`` is
built in: it returns the seed bundle without spending budget.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..errors import MethodUnavailable
from ..schema import CellSpec, PromptBundle

METHODS: dict[str, str] = {
    "gepa": "optimizers.gepa.integration:GEPAOptimizer",
    "mipro": "optimizers.mipro.integration:MIPROOptimizer",
    "hivemind": "optimizers.hivemind.integration:HiveMindOptimizer",
    "mamut_gepa": "optimizers.mamut_gepa.integration:MAMUTGEPAOptimizer",
    "mapro": "optimizers.mapro.integration:MAPROOptimizer",
    "maspo": "optimizers.maspo.integration:MASPOOptimizer",
    "maspob": "optimizers.maspob.integration:MASPOBOptimizer",
    "tavo": "optimizers.tavo.integration:TAVOOptimizer",
    "identity": "optimizers.protocol.methods.identity:IdentityOptimizer",
}
# Constructor keyword that receives the frozen seed bundle (detected from the
# signature; listed for reference).
BUNDLE_PARAMETERS: dict[str, str] = {
    "gepa": "initial_bundle",
    "mipro": "initial_bundle",
    "hivemind": "seed_bundle",
    "mamut_gepa": "seed_bundle",
    "mapro": "seed_bundle",
    "maspo": "seed_bundle",
    "maspob": "initial_bundle",
    "tavo": "initial_bundle",
    "identity": "seed_bundle",
}


def method_names() -> list[str]:
    """Registered method names, sorted."""
    return sorted(METHODS)


def load_method(name: str) -> type:
    """Import and return the optimizer class registered as ``name``."""
    try:
        target = METHODS[name]
    except KeyError as exc:
        raise KeyError(f"unknown method {name!r}; choices={method_names()}") from exc
    module_name, class_name = target.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise MethodUnavailable(f"method {name!r} is registered as {target} but its module is not installed") from exc
    return getattr(module, class_name)


def build_optimizer(
    name: str,
    cell: CellSpec,
    seed_bundle: PromptBundle,
    *,
    run_dir: Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Any:
    """Construct a method with the seed bundle and the constructor kwargs it accepts.

    Passed when the constructor declares them: the bundle (``seed_bundle`` or
    ``initial_bundle``), ``run_dir`` (scratch directory inside the job),
    ``reflection_lm`` (common-policy reflection LM) and ``task_lm`` (auxiliary
    DSPy LM on the cell's task model). ``extra`` overrides any of them.
    """
    cls = load_method(name)
    parameters = inspect.signature(cls).parameters
    variadic = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    kwargs: dict[str, Any] = dict(extra or {})
    bundle_parameter = next((p for p in ("seed_bundle", "initial_bundle") if p in parameters), None)
    if bundle_parameter is None:
        if not variadic:
            raise TypeError(f"{cls.__name__} accepts neither seed_bundle nor initial_bundle")
        bundle_parameter = BUNDLE_PARAMETERS.get(name, "seed_bundle")
    kwargs.setdefault(bundle_parameter, seed_bundle)
    if run_dir is not None and "run_dir" in parameters:
        kwargs.setdefault("run_dir", run_dir)
    if "reflection_lm" in parameters and "reflection_lm" not in kwargs:
        from .dspy_bridge import build_reflection_lm

        kwargs["reflection_lm"] = build_reflection_lm()
    if "task_lm" in parameters and "task_lm" not in kwargs:
        from .dspy_bridge import build_task_lm

        kwargs["task_lm"] = build_task_lm(cell.task_model)
    return cls(**kwargs)


__all__ = ["BUNDLE_PARAMETERS", "METHODS", "MethodUnavailable", "build_optimizer", "load_method", "method_names"]
