"""Frozen protocol constants: budget, seeds, models, decoding, splits, selection.

Paper-native optimizer settings stay method-specific. Everything here is the
common experimental envelope and is identical for all methods and seeds.
"""

from __future__ import annotations

from typing import Any

# Config
PROTOCOL_ID = "mas-promptbench"
BUDGET = 600
OPTIMIZER_SEEDS = (0, 1, 2)
REQUEST_SEED_OFFSETS = {0: 0, 1: 1000, 2: 2000}
MAX_INFRASTRUCTURE_RETRIES = 2
TEAM_SIZES = (2, 4, 8, 10)
COMMUNICATIONS = ("freeform", "semi_structured", "structured_soft")
TOPOLOGIES = ("single", "independent", "sequential", "centralized", "decentralized")

TASK_MODELS = {
    "qwen": "Qwen/Qwen3.5-9B",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}
DEFAULT_TASK_MODEL = TASK_MODELS["qwen"]
REFLECTION_MODEL_ID = "Qwen/Qwen3.5-122B-A10B-FP8"

# Task decoding: optimization rollouts sample, every reported evaluation is greedy.
TASK_DECODING = {
    "optimization": {"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 32768, "thinking": False},
    "evaluation": {"temperature": 0.0, "top_p": 0.9, "max_output_tokens": 32768, "thinking": False},
}

# Reflection decoding: native call-site sampling, common thinking mode and ceiling.
REFLECTION_SAMPLING = {
    "model_id": REFLECTION_MODEL_ID,
    "default_temperature": 1.0,
    "default_top_p": 1.0,
    "max_output_tokens": 48000,
    "thinking": True,
    "sampling_policy": "benchmark_thinking_48000_native_sampling",
}

# Fixed ordered splits ship in benchmarks/<dataset>/<dataset>_splits.json.
TASKS = {
    "gpqa": {"metric": "accuracy", "train": 48, "validation": 50, "test": 100},
    "hotpotqa": {"metric": "exact_match", "train": 150, "validation": 50, "test": 100},
    "math": {"metric": "accuracy", "train": 150, "validation": 50, "test": 100},
    "lcb": {"metric": "pass_at_1", "train": 150, "validation": 50, "test": 50},
    "apps": {"metric": "pass_at_1", "train": 150, "validation": 50, "test": 50},
    "swe": {"metric": "resolved_rate", "train": 146, "validation": 50, "test": 30},
    "bfcl": {"metric": "accuracy", "train": 150, "validation": 50, "test": 100},
    "toolhop": {"metric": "accuracy", "train": 150, "validation": 50, "test": 100},
    "apibank": {"metric": "accuracy", "train": 150, "validation": 50, "test": 100},
}
TASK_ALIASES = {"livecodebench": "lcb", "swebench_verified": "swe", "swebench": "swe"}

SELECTION = {
    "data": "validation_only",
    "deploy_only_if_strictly_better": True,
    "test_exposure": "post_selection_only",
}
LEARNING_CURVE = {"start": 0, "step": 10, "stop": BUDGET, "extra_model_calls": 0}
UNCHARGED_EVALUATIONS = ("baseline", "final_validation", "test")

PROTOCOL: dict[str, Any] = {
    "protocol_id": PROTOCOL_ID,
    "budget": {
        "unit": "usable_full_MAS_task_execution",
        "maximum": BUDGET,
        "maximum_overshoot": 0,
        "force_padding": False,
        "native_early_stopping": True,
        "infrastructure_retries": MAX_INFRASTRUCTURE_RETRIES,
        "uncharged_evaluations": list(UNCHARGED_EVALUATIONS),
    },
    "optimizer_seeds": list(OPTIMIZER_SEEDS),
    "request_seed_offsets": [REQUEST_SEED_OFFSETS[seed] for seed in OPTIMIZER_SEEDS],
    "models": {"task": dict(TASK_MODELS), "reflection": dict(REFLECTION_SAMPLING)},
    "task_decoding": TASK_DECODING,
    "tasks": TASKS,
    "selection": SELECTION,
    "learning_curve": LEARNING_CURVE,
}


def protocol_hash() -> str:
    """Content hash of the frozen protocol constants."""
    from .schema import content_hash

    return content_hash(PROTOCOL)


def normalize_task(name: str) -> str:
    """Canonical dataset key (``livecodebench`` and the SWE-bench aliases map to ``lcb``/``swe``)."""
    return TASK_ALIASES.get(name, name)


def task_model_id(model: str) -> str:
    """Map ``qwen``/``llama`` (or a full model ID) to the task-model ID."""
    return TASK_MODELS.get(model, model)


__all__ = [
    "BUDGET",
    "COMMUNICATIONS",
    "DEFAULT_TASK_MODEL",
    "LEARNING_CURVE",
    "MAX_INFRASTRUCTURE_RETRIES",
    "OPTIMIZER_SEEDS",
    "PROTOCOL",
    "PROTOCOL_ID",
    "REFLECTION_MODEL_ID",
    "REFLECTION_SAMPLING",
    "REQUEST_SEED_OFFSETS",
    "SELECTION",
    "TASKS",
    "TASK_ALIASES",
    "TASK_DECODING",
    "TASK_MODELS",
    "TEAM_SIZES",
    "TOPOLOGIES",
    "UNCHARGED_EVALUATIONS",
    "normalize_task",
    "protocol_hash",
    "task_model_id",
]
