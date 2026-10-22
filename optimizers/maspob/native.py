"""Native MASPOB components: pinned upstream core, prompt pool, embeddings, sampler, selection.

The four upstream core files in ``upstream/`` are read once, checked against
their SHA-256 pins and executed from exactly those bytes. Their ``scripts``
package name collides with this repository's ``scripts/``, so ``upstream/`` is
never put on ``sys.path``. torch, torch_geometric, sentence-transformers and
numpy are imported lazily; prompt-pool generation needs none of them.
Importing this module contacts no endpoint.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
import re
import sys
import threading
import time
import types
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from optimizers.protocol.artifacts import atomic_write_json
from optimizers.protocol.errors import MASPOBDependencyError
from optimizers.protocol.reflection import reflection_model_id

from .regime import (
    ADAPTATION_ID,
    CORE_SOURCE_SHA256,
    DEFAULT_EMBED_BACKEND,
    DEFAULTS,
    GENERATION_TEMPLATE_SHA256,
    HEADER_END,
    INITIALIZATION_PROTOCOL,
    ROW_SAMPLER_SEED_SALT,
    SOURCE_COMMIT,
)

UPSTREAM_ROOT = Path(__file__).resolve().parent / "upstream"
INSTALL_HINT = (
    "MASPOB needs numpy, torch (a CPU build is enough), torch_geometric (GATv2Conv) and "
    "sentence-transformers: pip install numpy torch torch_geometric sentence-transformers"
)


# Dependencies
def require(module: str, purpose: str) -> Any:
    """Import a heavy dependency or raise :class:`MASPOBDependencyError` with the install hint."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MASPOBDependencyError(f"MASPOB {purpose} needs the {module!r} package ({exc}). {INSTALL_HINT}") from exc


def missing_dependencies() -> tuple[str, ...]:
    """Heavy dependencies that cannot be imported (empty when MASPOB can run)."""
    missing = []
    for module in ("numpy", "torch", "torch_geometric", "sentence_transformers"):
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    return tuple(missing)


# Pinned upstream core
def upstream_source(rel_path: str) -> tuple[bytes, str]:
    """The vendored file's bytes and the SHA-256 of its upstream content.

    Everything above ``HEADER_END`` must be comment lines (the attribution
    header), so the executed code is exactly the pinned upstream content.
    """
    data = (UPSTREAM_ROOT / rel_path).read_bytes()
    header, marker, body = data.partition(HEADER_END)
    if not marker or any(line and not line.startswith(b"#") for line in header.splitlines()):
        raise RuntimeError(f"vendored MASPOB file {rel_path} lacks its comment-only attribution header")
    return data, hashlib.sha256(body).hexdigest()


def verified_core_source_hashes() -> dict[str, str]:
    """Hashes of the vendored MASPOB core sources; raises when any differs from the pin."""
    actual = {rel_path: upstream_source(rel_path)[1] for rel_path in CORE_SOURCE_SHA256}
    if actual != CORE_SOURCE_SHA256:
        mismatches = {
            rel_path: {"expected": CORE_SOURCE_SHA256[rel_path], "actual": actual[rel_path]}
            for rel_path in CORE_SOURCE_SHA256
            if actual[rel_path] != CORE_SOURCE_SHA256[rel_path]
        }
        raise RuntimeError(
            "MASPOB pinned core source integrity check failed: " + json.dumps(mismatches, sort_keys=True)
        )
    return actual


def _execute(qualname: str, rel_path: str) -> types.ModuleType:
    data, digest = upstream_source(rel_path)
    if digest != CORE_SOURCE_SHA256[rel_path]:
        raise RuntimeError(f"MASPOB pinned core source integrity check failed for {rel_path}")
    module = types.ModuleType(qualname)
    module.__file__ = str(UPSTREAM_ROOT / rel_path)
    sys.modules[qualname] = module
    try:
        exec(compile(data, module.__file__, "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(qualname, None)
        raise
    return module


_LOAD_LOCK = threading.Lock()
_PROMPT_MODULES: tuple[Any, Any] | None = None
_MODEL_MODULES: tuple[Any, Any] | None = None


def load_prompt_modules() -> tuple[Any, Any]:
    """``(styles, generator)``: upstream ``prompts/prompt.py`` and ``prompts/generator.py``."""
    global _PROMPT_MODULES
    with _LOAD_LOCK:
        if _PROMPT_MODULES is None:
            # generator.py imports ``scripts.prompts.prompt``. Register that upstream
            # namespace only while it executes, then restore the interpreter exactly:
            # leaving it installed would mask this repository's own ``scripts``.
            temporary = (
                "scripts",
                "scripts.prompts",
                "scripts.utils",
                "scripts.prompts.prompt",
                "scripts.prompts.generator",
            )
            missing = object()
            saved = {name: sys.modules.get(name, missing) for name in temporary}
            try:
                for name in ("scripts", "scripts.prompts", "scripts.utils"):
                    if name not in sys.modules:
                        package = types.ModuleType(name)
                        package.__path__ = []
                        sys.modules[name] = package
                styles = _execute("scripts.prompts.prompt", "scripts/prompts/prompt.py")
                generator = _execute("scripts.prompts.generator", "scripts/prompts/generator.py")
            finally:
                for name in reversed(temporary):
                    previous = saved[name]
                    if previous is missing:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = previous
            template = hashlib.sha256(styles.ITERATIVE_GENERATE_PROMPT.encode("utf-8")).hexdigest()
            if template != GENERATION_TEMPLATE_SHA256:
                raise RuntimeError("MASPOB generation template does not match the pinned upstream source")
            _PROMPT_MODULES = (styles, generator)
        return _PROMPT_MODULES


def load_model_modules() -> tuple[Any, Any]:
    """``(gnn, training)``: upstream ``gnn_model.py`` and ``utils/training.py`` (torch, torch_geometric)."""
    global _MODEL_MODULES
    with _LOAD_LOCK:
        if _MODEL_MODULES is None:
            require("torch", "GNN surrogate")
            require("torch_geometric", "GATv2 surrogate (GATv2Conv)")
            try:
                gnn = _execute("maspob_gnn_model", "scripts/gnn_model.py")
                training = _execute("maspob_training", "scripts/utils/training.py")
            except ImportError as exc:
                raise MASPOBDependencyError(
                    f"MASPOB GNN surrogate could not be imported ({exc}). {INSTALL_HINT}"
                ) from exc
            _MODEL_MODULES = (gnn, training)
        return _MODEL_MODULES


# Prompt pool
_PRIORITY_PRESETS = {
    # Upstream generator.py's dataset -> preset priority, keyed by the standalone
    # driver's task names; other tasks use the listed preset order.
    "math": ["DETAILED_REASONING", "VERIFY_FIRST", "SYSTEMATIC"],
    "lcb": ["CODE_RIGOROUS", "SYSTEMATIC", "PATTERN_BASED"],
    "bfcl": [],  # tool calling: no upstream preset priority
}
_KNOWN_TOOLS = {"calculator", "python_exec"}


def protocol_manifest(text: str) -> dict[str, list[str]]:
    """Exact interface tokens that a generated prompt must preserve."""
    placeholders = set(re.findall(r"(?<!\{)\{([A-Za-z_]\w*)\}(?!\})", text))
    delegations = set(re.findall(r"\bdelegate_to_[A-Za-z0-9_]+\b", text))
    roles = set(re.findall(r"\b(?:manager(?:_r\d+)?|[A-Za-z][A-Za-z0-9]*_worker)\b", text))
    backticked = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", text))
    tools = {tool for tool in _KNOWN_TOOLS if re.search(rf"\b{re.escape(tool)}\b", text)}
    xml_tags = set(re.findall(r"<\/?([A-Za-z][A-Za-z0-9_:-]*)\b[^>]*>", text))
    fence_languages = set(re.findall(r"```([A-Za-z][A-Za-z0-9_+-]*)", text))
    return {
        "placeholders": sorted(placeholders),
        "delegations": sorted(delegations),
        "roles": sorted(roles),
        "backticked_identifiers": sorted(backticked),
        "tools": sorted(tools),
        "xml_tags": sorted(xml_tags),
        "fence_languages": sorted(fence_languages),
    }


def valid_variant(text: str, seed: str) -> bool:
    """Require exact preservation of the role's executable prompt contract."""
    if not text or len(text.strip()) < 40:
        return False
    if len(text) > 4 * max(2000, len(seed)):
        return False
    return protocol_manifest(text) == protocol_manifest(seed)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def pool_cache_identity(
    reflection: Any, roles: Sequence[str], seed_prompts: Mapping[str, str], dataset: str, generation_seed: int
) -> dict[str, Any]:
    """What a cached pool must match to be reused. The endpoint is stored only as a hash."""
    endpoint = getattr(reflection, "endpoint", None)
    return {
        "adaptation_id": ADAPTATION_ID,
        "official_source_commit": SOURCE_COMMIT,
        "core_source_sha256": verified_core_source_hashes(),
        "dataset": dataset,
        "model": getattr(reflection, "model", None) or reflection_model_id(),
        "endpoint_sha256": _sha256_text(str(endpoint)) if endpoint else None,
        "generation_temperature": float(getattr(reflection, "temperature", DEFAULTS.generation_temperature)),
        "generation_template_sha256": GENERATION_TEMPLATE_SHA256,
        "roles": list(roles),
        "seed_sha256": {role: _sha256_text(seed_prompts[role]) for role in roles},
        "generation_seed": int(generation_seed),
    }


def generate_prompt_pool(
    reflection: Any,
    roles: list[str],
    seed_prompts: Mapping[str, str],
    dataset: str,
    num_prompts: int,
    refl_inflight: int,
    pool_path: Path,
    generation_seed: int = 0,
    topology_description: str = "centralized manager + workers, star topology",
    protocol_description: str = (
        "Respect the original executable collaboration/delegation protocol and exact output format."
    ),
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Create or incrementally fill a contract-validated prompt pool.

    Variant 0 of each role is the exact seed prompt. Random style draws use the
    process-global ``random`` stream (upstream behavior); the caller seeds it.
    """
    if num_prompts < 1:
        raise ValueError("num_prompts must be at least 1")
    styles, generator = load_prompt_modules()
    pool_path = Path(pool_path)
    identity = pool_cache_identity(reflection, roles, seed_prompts, dataset, generation_seed)
    pool: dict[str, list[str]] = {role: [seed_prompts[role].strip()] for role in roles}
    cache_status = "miss"
    cache_rejections = {role: 0 for role in roles}

    if pool_path.exists():
        try:
            payload = json.loads(pool_path.read_text(encoding="utf-8"))
            cached_identity = payload.get("identity") if isinstance(payload, dict) else None
            cached_pools = payload.get("pools") if isinstance(payload, dict) else None
            if cached_identity == identity and isinstance(cached_pools, dict):
                cache_status = "validated"
            else:
                cached_pools = None
                cache_status = "identity-mismatch"
            if cached_pools is not None:
                for role in roles:
                    values = cached_pools.get(role) or []
                    if not values or values[0].strip() != seed_prompts[role].strip():
                        cache_status = "seed-mismatch"
                        pool = {name: [seed_prompts[name].strip()] for name in roles}
                        break
                    seen = {seed_prompts[role].strip()}
                    valid = [seed_prompts[role].strip()]
                    for value in values[1:]:
                        text = str(value).strip()
                        if text in seen:
                            continue
                        if valid_variant(text, seed_prompts[role]):
                            valid.append(text)
                            seen.add(text)
                        else:
                            cache_rejections[role] += 1
                    pool[role] = valid[:num_prompts]
        except Exception as exc:
            cache_status = f"unreadable:{type(exc).__name__}"

    priority = _PRIORITY_PRESETS.get(dataset, [])
    preset_order = priority + [name for name in styles.PRESET_STYLE_NAMES if name not in priority]

    jobs = []  # (role, slot, meta_prompt)
    for role in roles:
        needed = max(0, num_prompts - len(pool[role]))
        # Upstream's 1.5x over-generation, so validation failures do not freeze undersized pools.
        for offset in range(max(needed, int(needed * 1.5 + 0.5))):
            k = len(pool[role]) - 1 + offset
            if k < len(preset_order):
                style = styles.generate_style_instruction(quality=preset_order[k])
            else:
                style = styles.generate_style_instruction(random_sample=True)
            manifest = protocol_manifest(seed_prompts[role])
            meta_prompt = styles.ITERATIVE_GENERATE_PROMPT.format(
                prompt_type=(
                    f"the system instruction for the '{role}' role in a "
                    f"{topology_description} multi-agent system solving {dataset} tasks"
                ),
                prompt_goal=(
                    "Achieve the same objective. "
                    + protocol_description
                    + " Demand the same final output format as this ORIGINAL instruction:\n---\n"
                    + seed_prompts[role]
                    + "\n---"
                ),
                required_placeholders=(
                    "Preserve this exact executable-interface manifest; do not remove "
                    "or invent entries:\n" + json.dumps(manifest, sort_keys=True)
                ),
                style_instruction=style,
            )
            jobs.append((role, k, meta_prompt))

    failures = {role: 0 for role in roles}

    def generate_one(job):
        role, _slot, meta_prompt = job
        for attempt in range(3):
            try:
                text = generator.clean_generated_prompt(reflection.complete(meta_prompt))
                if valid_variant(text, seed_prompts[role]):
                    return role, text
            except Exception:
                time.sleep(1.0 * (attempt + 1))
        return role, None

    if jobs:
        with ThreadPoolExecutor(max_workers=min(refl_inflight, len(jobs))) as executor:
            for role, text in executor.map(generate_one, jobs):
                if text is None:
                    failures[role] += 1
                elif text not in pool[role] and len(pool[role]) < num_prompts:
                    pool[role].append(text)

    atomic_write_json(
        pool_path, {"identity": identity, "num_prompts_target": int(num_prompts), "pools": pool}, indent=2
    )
    stats = {
        "loaded_from_cache": cache_status == "validated" and not jobs,
        "cache_status": cache_status,
        "cache_identity": identity,
        "cache_rejections": cache_rejections,
        "pool_sizes": {role: len(pool[role]) for role in roles},
        "generation_failures": failures,
        "generation_attempted": any(len(pool[role]) < num_prompts for role in roles) or bool(jobs),
        "complete": all(len(pool[role]) == num_prompts for role in roles),
    }
    if not stats["complete"]:
        raise RuntimeError(
            f"MASPOB prompt pool is incomplete (target={num_prompts}, sizes={stats['pool_sizes']}); "
            "the partial validated cache was retained for a retry"
        )
    return pool, stats


# Embeddings: sentence-transformers MiniLM on CPU (the paper used Qwen3-Embedding-8B).
def embed_pool(
    pool: Mapping[str, list[str]],
    roles: Sequence[str],
    *,
    model_name: str = DEFAULTS.embed_model,
    max_seq_length: int = DEFAULTS.embed_max_seq,
) -> tuple[list[Any], dict[str, Any]]:
    """Normalized CPU sentence embeddings of every variant, one tensor per role."""
    torch = require("torch", "embedding")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise MASPOBDependencyError(
            f"MASPOB embeddings need the 'sentence_transformers' package ({exc}). {INSTALL_HINT}"
        ) from exc
    try:
        model = SentenceTransformer(model_name, device="cpu")
        try:  # MiniLM's positional ceiling is 512; a larger value raises inside encode()
            ceiling = int(model[0].auto_model.config.max_position_embeddings)
        except Exception:
            ceiling = max_seq_length
        model.max_seq_length = min(max_seq_length, ceiling)
        embeddings = []
        for role in roles:
            vectors = model.encode(
                pool[role], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, batch_size=8
            )
            embeddings.append(torch.tensor(vectors, dtype=torch.float32))
    except Exception as exc:
        # A TF-IDF pool would be a materially different surrogate input: fail closed.
        raise RuntimeError(f"MASPOB embedding backend {model_name!r} failed; refusing a TF-IDF fallback") from exc
    try:
        truncation_counts = {}
        for role in roles:
            encoded = model.tokenizer(pool[role], add_special_tokens=True, truncation=False)
            truncation_counts[role] = sum(len(ids) > model.max_seq_length for ids in encoded["input_ids"])
    except Exception:
        truncation_counts = {role: None for role in roles}
    info = {
        "backend": DEFAULT_EMBED_BACKEND,
        "model": model_name,
        "dim": int(embeddings[0].shape[1]),
        "max_seq_length": int(getattr(model, "max_seq_length", 0) or 0),
        "prompts_exceeding_max_seq": truncation_counts,
        "note": "fallback for Qwen3-Embedding-8B",
    }
    return embeddings, info


# Minibatches, RNG streams and selection
class MinibatchSampler:
    """Rotating shuffled passes over the training rows (without replacement within a pass)."""

    def __init__(self, rows: Sequence[Any], size: int, rng: random.Random) -> None:
        self.rows = list(rows)
        self.size = max(1, int(size))
        self.rng = rng
        self._queue: list[Any] = []

    def next(self) -> list[Any]:
        """The next minibatch, drawn without replacement from a reshuffled epoch queue."""
        batch = []
        while len(batch) < self.size:
            if not self._queue:
                self._queue = list(self.rows)
                self.rng.shuffle(self._queue)
            batch.append(self._queue.pop())
        return batch


def rng_streams(run_seed: int) -> tuple[random.Random, random.Random, int]:
    """Keep prompt-combination draws independent of task-row scheduling."""
    row_sampler_seed = int(run_seed) + ROW_SAMPLER_SEED_SALT
    return random.Random(run_seed), random.Random(row_sampler_seed), row_sampler_seed


def select_best_observed(
    combo_best: Mapping[tuple[int, ...], float],
    pull_history: Sequence[Mapping[str, Any]],
    predict_mean: Callable[[tuple[int, ...]], float],
) -> tuple[list[int], float | None, str, list[dict[str, Any]]]:
    """Select among observed combinations by the fitted surrogate's posterior mean.

    Raw maxima from different rotating minibatches are not comparable; they stay
    diagnostic. The earliest first pull breaks exact ties.
    """
    if not combo_best:
        return [], None, "no combination was evaluated", []
    first_pull: dict[tuple[int, ...], int] = {}
    for entry in pull_history:
        first_pull.setdefault(tuple(entry["indices"]), int(entry["pull"]))
    predictions: list[dict[str, Any]] = []
    try:
        for key, observed in combo_best.items():
            posterior_mean = float(predict_mean(key))
            if not math.isfinite(posterior_mean):
                raise ValueError(f"non-finite posterior mean for {list(key)}: {posterior_mean}")
            predictions.append(
                {
                    "indices": list(key),
                    "posterior_mean": posterior_mean,
                    "max_observed_score": float(observed),
                    "first_pull": first_pull.get(key, 10**18),
                }
            )
    except Exception as exc:
        raise RuntimeError("MASPOB posterior-mean selection failed; refusing a raw-score fallback") from exc
    winner = max(predictions, key=lambda item: (item["posterior_mean"], -int(item["first_pull"])))
    key = tuple(winner["indices"])
    note = (
        "selected among observed combinations by GNN posterior mean; raw rotating-minibatch scores are diagnostic only"
    )
    return list(key), combo_best[key], note, predictions


def predict_posterior_mean(gnn_model: Any, combined: Any) -> float:
    """Evaluate only the fitted mean, without UCB uncertainty machinery."""
    torch = require("torch", "GNN surrogate")
    gnn_model.eval()
    with torch.no_grad():
        return float(gnn_model.unscale_score(gnn_model(combined)).item())


def initialize_from_pretrain(
    pretrain_history: Sequence[Mapping[str, Any]], predict_mean: Callable[[tuple[int, ...]], float]
) -> tuple[list[int], str, list[dict[str, Any]]]:
    """Choose the coordinate-ascent anchor on one comparable score scale."""
    combo_best: dict[tuple[int, ...], float] = {}
    for entry in pretrain_history:
        key = tuple(entry["indices"])
        combo_best[key] = max(combo_best.get(key, -1.0), float(entry["score"]))
    indices, _raw, _note, predictions = select_best_observed(combo_best, pretrain_history, predict_mean)
    return indices, INITIALIZATION_PROTOCOL, predictions


__all__ = [
    "INSTALL_HINT",
    "MinibatchSampler",
    "UPSTREAM_ROOT",
    "embed_pool",
    "generate_prompt_pool",
    "initialize_from_pretrain",
    "load_model_modules",
    "load_prompt_modules",
    "missing_dependencies",
    "pool_cache_identity",
    "predict_posterior_mean",
    "protocol_manifest",
    "require",
    "rng_streams",
    "select_best_observed",
    "upstream_source",
    "valid_variant",
    "verified_core_source_hashes",
]
