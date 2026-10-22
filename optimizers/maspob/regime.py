"""Frozen MASPOB regime: upstream pins, the surrogate settings and :class:`MASPOBSettings`.

Upstream: https://github.com/HZ1008/MASPOB (arXiv:2603.02630), pinned to
``SOURCE_COMMIT``. The four core files in ``upstream/`` are byte-identical to
that commit below their attribution header; ``CORE_SOURCE_SHA256`` pins them.
"""

from __future__ import annotations

from dataclasses import dataclass

# Provenance
ADAPTATION_ID = "maspob-official-source-adaptation"
SOURCE_URL = "https://github.com/HZ1008/MASPOB"
SOURCE_PAPER = "arXiv:2603.02630"
SOURCE_COMMIT = "0be3af42da83b9b5ab187c101ec6d2ce87a5fe04"
LICENSE_STATUS = (
    "released for research purposes (upstream README); no LICENSE/COPYING/NOTICE file in the pinned repository"
)
GENERATION_TEMPLATE_SHA256 = "d610e2871cb1fb80758b419bfb29df74dbf0c14dd4ea6b5a3be2abcb49582cd2"
# SHA-256 of each vendored file's content below its ``HEADER_END`` line.
CORE_SOURCE_SHA256 = {
    "scripts/gnn_model.py": "b182a621c2a526f37bc30782cf18efb2ce3b300d5472b1cbcc93b61d67531da3",
    "scripts/utils/training.py": "248eb0a32cff1cb3981a549d254c58d139cc555736fa61f09893338208eceb2a",
    "scripts/prompts/prompt.py": "558f3c5f8f567006534cbda862167ff1e96e8bf8f1243b2347f06235d991f889",
    "scripts/prompts/generator.py": "d783d9c258be97d420f6ba5eac93bda1997904184cb0bfbe5ff2992f896f34bf",
}
HEADER_END = b"# --- end of vendoring header ---\n"

# Prompt pool: task name inside the generation meta-prompt and the preset-priority lookup.
# The reference integration passed its own task key, "livecodebench", here.
GENERATION_TASK_NAMES = {"lcb": "livecodebench"}
DEFAULT_EMBED_BACKEND = "sentence-transformers"

# Bandit lifecycle and seeds
RNG_PROTOCOL = "independent-combo-and-row-streams"
ROW_SAMPLER_SEED_SALT = 104729
INITIALIZATION_PROTOCOL = "surrogate-mean-pretrain"
DEFAULT_SELECTION_PROTOCOL = "surrogate-mean"


@dataclass(frozen=True)
class MASPOBSettings:
    """MASPOB's knobs (no environment variables).

    ``num_prompts`` variants per role (variant 0 = seed) generated at
    ``generation_temperature``; MiniLM embeddings (replacing the paper's
    Qwen3-Embedding-8B); random/numpy/torch seeded with ``seed + seed_stride *
    optimizer seed``. Budget split: every bandit pull charges one rotating
    minibatch of ``minibatch_size`` training rows; the first
    :meth:`pretrain_pulls_for` pulls are random warm-up, the rest LinUCB,
    until the ledger is spent. Reflection: ``reflection_inflight`` concurrent
    pool requests, ``reflection_max_retries`` SDK retries.
    """

    num_prompts: int = 20
    generation_temperature: float = 0.5
    generation_top_p: float = 1.0
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_max_seq: int = 512
    seed: int = 42
    seed_stride: int = 1000
    minibatch_size: int = 5
    pretrain_pulls: int = 5
    reflection_inflight: int = 4
    reflection_max_retries: int = 3

    def pretrain_pulls_for(self, remaining: int) -> int:
        """Warm-up pulls for ``remaining`` rollouts: at most ``pretrain_pulls``, leaving one LinUCB pull."""
        return min(self.pretrain_pulls, max(1, remaining // self.minibatch_size - 1))


DEFAULTS = MASPOBSettings()

# Surrogate and UCB (upstream run.py CLI defaults)
GNN_SETTINGS = {
    "hidden_dim": 32,
    "num_gnn_layers": 1,
    "dropout": 0.05,
    "lr": 5e-3,
    "weight_decay": 1e-5,
    "epochs": 800,
    "patience": 200,
    "min_delta": 1e-10,
    "bidirectional": True,
    "use_sigmoid": True,
    "score_min": 0.0,
    "score_max": 1.0,
}
UCB_SETTINGS = {
    "type": "linear",
    "alpha": 0.2,
    "lambda_reg": 1.0,
    "fisher_coef": 10.0,
    "search_strategy": "coordinate",
}

BENCHMARK_ADAPTATIONS = (
    "pinned MiniLM embeddings replace Qwen3-Embedding-8B",
    "rotating train minibatches replace one fixed validation slice",
    "bidirectional centralized star represents the group-chat runtime",
    "final observed-combination selection uses GNN posterior mean",
)


def run_seed(optimizer_seed: int) -> int:
    """Seed of the random, numpy and torch streams of one optimizer seed."""
    return DEFAULTS.seed + DEFAULTS.seed_stride * int(optimizer_seed)


__all__ = [
    "ADAPTATION_ID",
    "BENCHMARK_ADAPTATIONS",
    "CORE_SOURCE_SHA256",
    "DEFAULTS",
    "DEFAULT_EMBED_BACKEND",
    "DEFAULT_SELECTION_PROTOCOL",
    "GENERATION_TASK_NAMES",
    "GENERATION_TEMPLATE_SHA256",
    "GNN_SETTINGS",
    "HEADER_END",
    "INITIALIZATION_PROTOCOL",
    "LICENSE_STATUS",
    "MASPOBSettings",
    "RNG_PROTOCOL",
    "ROW_SAMPLER_SEED_SALT",
    "SOURCE_COMMIT",
    "SOURCE_PAPER",
    "SOURCE_URL",
    "UCB_SETTINGS",
    "run_seed",
]
