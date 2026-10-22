"""MASPOB on the shared run protocol: GATv2 surrogate + LinUCB over a prompt-variant pool.

Per role, 20 prompt variants (variant 0 = the seed prompt) are generated once
by the reflection model at MASPOB's temperature 0.5 from the upstream
10-dimension style template, then embedded with all-MiniLM-L6-v2 (384-d, CPU).
A bandit pull runs one variant combination on the next 5 rows of a rotating
shuffled pass over the training split (5 charged rollouts). After 5 random
warm-up pulls the upstream WorkflowGAT surrogate (hidden 32, one GATv2 layer,
lr 5e-3) is fitted; every later pull is chosen by coordinate-ascent LinUCB
(alpha 0.2, lambda 1, Fisher coefficient 10) and the surrogate is retrained
from scratch on all pulls, until the ledger is spent. The incumbent is the
observed combination with the highest surrogate posterior mean. The random,
numpy and torch streams are seeded with 42 + 1000 * optimizer seed.

Upstream: https://github.com/HZ1008/MASPOB (arXiv:2603.02630); the pinned core
files live in ``upstream/``. Heavy dependencies are imported lazily.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from optimizers.protocol.cells import validate_cell
from optimizers.protocol.errors import OptimizerContractError
from optimizers.protocol.journal import native_checkpoint, persist_optimizer_artifact
from optimizers.protocol.reflection import (
    REFLECTION_REQUEST_TIMEOUT_SECONDS,
    LogicalTextReflectionClient,
    ReflectionClient,
)
from optimizers.protocol.rollouts import BudgetedRunner, load_seed_bundle, native_role_order, ordered_prompt_roles
from optimizers.protocol.schema import (
    SEARCH_LAYOUT,
    CellSpec,
    LearningCurvePoint,
    OptimizerResult,
    PromptBundle,
    RunRecord,
    StopReason,
    example_id,
)
from optimizers.protocol.seeding import request_seeds

from .regime import (
    ADAPTATION_ID,
    BENCHMARK_ADAPTATIONS,
    DEFAULT_SELECTION_PROTOCOL,
    DEFAULTS,
    GENERATION_TASK_NAMES,
    GNN_SETTINGS,
    LICENSE_STATUS,
    RNG_PROTOCOL,
    SOURCE_COMMIT,
    SOURCE_PAPER,
    SOURCE_URL,
    UCB_SETTINGS,
    MASPOBSettings,
)
from .regime import (
    run_seed as maspob_run_seed,
)


def _ordered_roles(cell: CellSpec, bundle: PromptBundle) -> list[str]:
    """Native role order (the adapter's ``roles()``); centralized puts the manager first."""
    if cell.topology == "sequential":
        roles = native_role_order(cell, list(bundle.roles))
    else:
        roles = list(ordered_prompt_roles(cell, bundle))
    if cell.topology == "centralized":
        manager = next((role for role in roles if role == "manager" or role.startswith("manager_r")), None)
        if manager:
            roles = [manager] + [role for role in roles if role != manager]
    return roles


def _topology_contract(cell: CellSpec, roles: Sequence[str]) -> tuple[list[dict[str, Any]], str, str]:
    """Surrogate graph plus the topology/protocol text given to prompt generation."""
    if cell.topology == "centralized":
        graph = [{"name": role, "dependencies": [] if index == 0 else [roles[0]]} for index, role in enumerate(roles)]
        description = "centralized manager/worker star"
        protocol = (
            "Preserve the original manager delegation tools, worker return path, placeholders, "
            "and final-answer contract exactly."
        )
    elif cell.topology == "sequential":
        graph = [
            {"name": role, "dependencies": [] if index == 0 else [roles[index - 1]]} for index, role in enumerate(roles)
        ]
        description = "ordered sequential pipeline"
        protocol = (
            "Preserve the original stage order, handoff format, placeholders, and final-output "
            "contract exactly; do not invent manager delegation."
        )
    elif cell.topology == "independent":
        graph = [{"name": role, "dependencies": []} for role in roles]
        description = "independent-agent ensemble with a shared role instruction"
        protocol = (
            "Preserve independent execution and the original aggregation/output contract; "
            "do not add delegation or cross-agent communication."
        )
    elif cell.topology == "decentralized":
        graph = [{"name": role, "dependencies": []} for role in roles]
        description = "decentralized peer-to-peer team with a shared role instruction"
        protocol = (
            "Preserve peer handoffs, round limits, placeholders, and final-output contract; "
            "do not introduce a centralized manager."
        )
    else:
        raise OptimizerContractError(f"MASPOB does not support topology {cell.topology!r}")
    return graph, description, protocol


def _score_records(records: Sequence[RunRecord]) -> float:
    if not records:
        raise OptimizerContractError("MASPOB pull produced no records")
    scores = []
    for record in records:
        if not record.usable or record.score is None:
            raise OptimizerContractError(f"MASPOB pull received unusable record {record.example_id!r}")
        scores.append(float(record.score))
    return fmean(scores)


class MASPOBOptimizer:
    """Execute the hash-pinned MASPOB WorkflowGAT/Fisher/LinUCB lifecycle through the protocol runner.

    ``reflection_client`` defaults to the protocol :class:`ReflectionClient`
    (MASPOB's retry count and the reflection timeout); ``prompt_pool_factory``
    / ``embedding_factory`` replace pool generation / embedding (tests);
    ``validation_one_cycle`` runs one warm-up and one LinUCB pull on two
    minibatches and marks the result validation-only.
    """

    method = "maspob"
    native_iteration_event = "completed bandit pull/validation update"
    production_eligible = True
    implementation_kind = "native_maspob_gat_linucb_common_runner"

    def __init__(
        self,
        *,
        initial_bundle: PromptBundle | None = None,
        run_dir: Path | None = None,
        reflection_client: Any | None = None,
        prompt_pool_factory: Callable[..., Any] | None = None,
        embedding_factory: Callable[..., Any] | None = None,
        pool_path: Path | None = None,
        reflection_inflight: int = DEFAULTS.reflection_inflight,
        validation_one_cycle: bool = False,
    ) -> None:
        if reflection_inflight <= 0:
            raise ValueError("reflection_inflight must be positive")
        self.initial_bundle = initial_bundle
        self.run_dir = Path(run_dir) if run_dir is not None else None
        # The client actually used is stored here, so the job reports its usage.
        self.reflection_client = reflection_client
        self.prompt_pool_factory = prompt_pool_factory
        self.embedding_factory = embedding_factory
        self.pool_path = Path(pool_path) if pool_path is not None else None
        self.settings = MASPOBSettings(reflection_inflight=int(reflection_inflight))
        self.validation_one_cycle = bool(validation_one_cycle)

    def _pool_path(self, runner: Any) -> Path:
        if self.pool_path is not None:
            path = self.pool_path
        else:
            directory = self.run_dir if self.run_dir is not None else getattr(runner, "artifact_directory", None)
            if directory is None:
                raise OptimizerContractError("MASPOB needs pool_path, run_dir or a runner artifact directory")
            path = Path(directory) / "maspob_prompt_pool.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def optimize(
        self, cell: CellSpec, runner: Any, budget: Any, training: list[Any], validation: list[Any]
    ) -> OptimizerResult:
        """Generate and embed the prompt pool, then run bandit pulls until the ledger is spent."""
        del validation
        settings = self.settings
        validate_cell(self.method, cell)
        if not training:
            raise ValueError("native MASPOB requires a non-empty training split")

        # Fail on a missing dependency before any reflection call or rollout.
        from .native import (
            MinibatchSampler,
            embed_pool,
            generate_prompt_pool,
            initialize_from_pretrain,
            load_model_modules,
            predict_posterior_mean,
            require,
            rng_streams,
            select_best_observed,
            verified_core_source_hashes,
        )

        np = require("numpy", "seeding")
        torch = require("torch", "GNN surrogate")
        GNN, TRAINING = load_model_modules()
        if self.embedding_factory is None:
            require("sentence_transformers", "embeddings")

        seed_bundle = load_seed_bundle(cell, runner, self.initial_bundle)
        roles = _ordered_roles(cell, seed_bundle)
        seed_prompts = {role: seed_bundle.roles[role] for role in roles}
        protocol_runner = BudgetedRunner(cell=cell, runner=runner, budget=budget)
        graph, topology_description, protocol_description = _topology_contract(cell, roles)
        run_seed = maspob_run_seed(cell.optimizer_seed)
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        combo_rng, sampler_rng, row_sampler_seed = rng_streams(run_seed)
        sampler = MinibatchSampler(training, settings.minibatch_size, sampler_rng)
        self.reflection_client = self.reflection_client or ReflectionClient(
            max_retries=settings.reflection_max_retries, timeout=REFLECTION_REQUEST_TIMEOUT_SECONDS
        )
        reflection = LogicalTextReflectionClient(
            cell=cell,
            client=self.reflection_client,
            method=self.method,
            default_temperature=settings.generation_temperature,
            default_top_p=settings.generation_top_p,
        )
        generation_task = GENERATION_TASK_NAMES.get(cell.task, cell.task)
        pool_factory = self.prompt_pool_factory or generate_prompt_pool
        pool, pool_stats = pool_factory(
            reflection,
            roles,
            seed_prompts,
            generation_task,
            settings.num_prompts,
            settings.reflection_inflight,
            self._pool_path(runner),
            generation_seed=run_seed,
            topology_description=topology_description,
            protocol_description=protocol_description,
        )
        if set(pool) != set(roles) or any(len(pool[role]) != settings.num_prompts for role in roles):
            raise OptimizerContractError(
                f"MASPOB prompt pool must contain exactly {settings.num_prompts} variants per role"
            )
        embeddings_factory = self.embedding_factory or embed_pool
        all_operator_embeddings, embedding_info = embeddings_factory(pool, roles)
        if len(all_operator_embeddings) != len(roles):
            raise OptimizerContractError("MASPOB embedding list does not match the role graph")
        embedding_dim = int(all_operator_embeddings[0].shape[1])
        gnn_model = GNN.WorkflowGAT(
            embedding_dim=embedding_dim,
            num_operators=len(roles),
            hidden_dim=GNN_SETTINGS["hidden_dim"],
            num_gnn_layers=GNN_SETTINGS["num_gnn_layers"],
            dropout=GNN_SETTINGS["dropout"],
            topology=graph,
            bidirectional=GNN_SETTINGS["bidirectional"],
            use_sigmoid=GNN_SETTINGS["use_sigmoid"],
            score_min=GNN_SETTINGS["score_min"],
            score_max=GNN_SETTINGS["score_max"],
        )
        gnn_optimizer = torch.optim.Adam(
            gnn_model.parameters(), lr=GNN_SETTINGS["lr"], weight_decay=GNN_SETTINGS["weight_decay"]
        )
        total_dim = sum(int(embedding.shape[1]) for embedding in all_operator_embeddings)
        fisher_matrix = GNN.initialize_fisher(
            UCB_SETTINGS["type"],
            frozen_model=None,
            embedding_dim=total_dim,
            device=all_operator_embeddings[0].device,
            lambda_reg=UCB_SETTINGS["lambda_reg"],
        )
        initial_snapshot = dict(budget.snapshot())
        lifecycle_rollout_limit = 2 * settings.minibatch_size if self.validation_one_cycle else cell.budget
        lifecycle_start_charged = int(initial_snapshot["charged"])
        if self.validation_one_cycle and int(initial_snapshot["remaining"]) < lifecycle_rollout_limit:
            raise OptimizerContractError(
                f"MASPOB one-cycle validation requires {lifecycle_rollout_limit} rollout slots"
            )

        def lifecycle_remaining() -> int:
            snapshot = budget.snapshot()
            consumed = int(snapshot["charged"]) - lifecycle_start_charged
            return max(0, min(int(snapshot["remaining"]), lifecycle_rollout_limit - consumed))

        learning_curve = [LearningCurvePoint("initial", 0, int(initial_snapshot["charged"]), None, seed_bundle.digest)]
        checkpoints: list[Mapping[str, Any]] = []
        pull_history: list[dict[str, Any]] = []
        training_data: list[tuple[Any, float]] = []
        best_observed = float("-inf")

        def execute_pull(indices: list[int], stage: str, extras: Mapping[str, Any] | None = None):
            nonlocal best_observed
            granted = min(settings.minibatch_size, lifecycle_remaining())
            if granted <= 0:
                return None
            rows = sampler.next()[:granted]
            prompts = {role: pool[role][indices[index]] for index, role in enumerate(roles)}
            bundle = PromptBundle(roles=prompts, demos=seed_bundle.demos, metadata=seed_bundle.metadata)
            pull_number = len(pull_history) + 1
            seeds = request_seeds(cell, rows, phase=f"maspob_{stage}", iteration=pull_number, bundle=bundle)
            # The sampler can cross a pass boundary inside one pull and repeat a row
            # (always so for one-row validation). The runner requires unique IDs per
            # atomic batch: keep every occurrence and its seed, split only the dispatch.
            records: list[RunRecord] = []
            batch_start = 0
            batch_ids: set[str] = set()
            for index, row in enumerate(rows):
                row_id = example_id(row, index)
                if row_id in batch_ids:
                    records.extend(protocol_runner.run_batch(rows[batch_start:index], bundle, seeds[batch_start:index]))
                    batch_start = index
                    batch_ids.clear()
                batch_ids.add(row_id)
            records.extend(protocol_runner.run_batch(rows[batch_start:], bundle, seeds[batch_start:]))
            score = _score_records(records)
            entry = {
                "pull": pull_number,
                "stage": stage,
                "indices": list(indices),
                "n_items": len(records),
                "truncated": len(records) < settings.minibatch_size,
                "score": score,
                "items": [
                    {"id": record.example_id, "score": float(record.score), "seed": record.request_seed}
                    for record in records
                ],
                **dict(extras or {}),
            }
            pull_history.append(entry)
            best_observed = max(best_observed, score)
            learning_curve.append(
                LearningCurvePoint(
                    "completed_bandit_pull",
                    pull_number,
                    int(budget.snapshot()["charged"]),
                    best_observed,
                    bundle.digest,
                    {"stage": stage, "indices": list(indices), "pull_score": score},
                )
            )
            checkpoints.append(
                native_checkpoint(
                    method=self.method,
                    iteration=pull_number,
                    bundle=bundle,
                    budget=budget,
                    state={
                        "stage": stage,
                        "indices": list(indices),
                        "score": score,
                        "pull_history": pull_history,
                        "row_sampler_seed": row_sampler_seed,
                    },
                )
            )
            return entry

        # Warm-up: random combinations, Fisher accumulation, then the first fit.
        num_prompts_per_role = [len(pool[role]) for role in roles]
        pretrain_pulls = settings.pretrain_pulls_for(lifecycle_remaining())
        for _ in range(pretrain_pulls):
            indices = [combo_rng.randint(0, count - 1) for count in num_prompts_per_role]
            entry = execute_pull(indices, "pretrain")
            if entry is None:
                break
            combined = GNN.build_combined_embedding(all_operator_embeddings, indices)
            fisher_matrix = GNN.update_fisher(
                UCB_SETTINGS["type"], fisher_matrix, combined, UCB_SETTINGS["fisher_coef"]
            )
            training_data.append((combined.clone(), entry["score"]))

        def retrain() -> Any:
            if not training_data:
                return None
            batch_embeddings = torch.stack([embedding for embedding, _ in training_data])
            targets = torch.tensor([score for _, score in training_data], dtype=torch.float32)
            return TRAINING.train_with_early_stopping(
                gnn_model,
                gnn_optimizer,
                batch_embeddings,
                gnn_model.scale_score(targets),
                max_epochs=GNN_SETTINGS["epochs"],
                patience=GNN_SETTINGS["patience"],
                min_delta=GNN_SETTINGS["min_delta"],
                verbose=False,
            )

        def posterior_mean(key: tuple[int, ...]) -> float:
            return predict_posterior_mean(gnn_model, GNN.build_combined_embedding(all_operator_embeddings, list(key)))

        retrain()
        initialization_predictions: list[dict[str, Any]] = []
        if training_data:
            current_indices, initialization_protocol, initialization_predictions = initialize_from_pretrain(
                [entry for entry in pull_history if entry["stage"] == "pretrain"], posterior_mean
            )
        else:
            current_indices = [0] * len(roles)
            initialization_protocol = "seed-no-pretrain"
        initial_indices = list(current_indices)

        # UCB rounds: coordinate ascent, pull, update Fisher, retrain from scratch.
        while lifecycle_remaining() > 0:
            for operator_index in range(len(roles)):
                best_index, _, _, _ = GNN.select_best_prompt_for_operator(
                    UCB_SETTINGS["type"],
                    gnn_model,
                    operator_index,
                    current_indices,
                    all_operator_embeddings,
                    frozen_model=None,
                    fisher_matrix=fisher_matrix,
                    alpha=UCB_SETTINGS["alpha"],
                )
                current_indices[operator_index] = best_index
            combined = GNN.build_combined_embedding(all_operator_embeddings, current_indices)
            prediction, uncertainty, _feature = GNN.compute_prediction_and_uncertainty(
                UCB_SETTINGS["type"], gnn_model, combined, frozen_model=None, fisher_matrix=fisher_matrix
            )
            entry = execute_pull(
                current_indices,
                "ucb",
                {
                    "pred": prediction,
                    "uncertainty": uncertainty,
                    "ucb": prediction + UCB_SETTINGS["alpha"] * uncertainty,
                },
            )
            if entry is None:
                break
            training_data.append((combined.clone(), entry["score"]))
            fisher_matrix = GNN.update_fisher(
                UCB_SETTINGS["type"], fisher_matrix, combined, UCB_SETTINGS["fisher_coef"]
            )
            gnn_model.reset_parameters()
            for group in gnn_optimizer.param_groups:
                group["lr"] = GNN_SETTINGS["lr"]
                for parameter in group["params"]:
                    if parameter in gnn_optimizer.state:
                        del gnn_optimizer.state[parameter]
            retrain()

        # Final pick: observed combinations ranked by the surrogate posterior mean.
        combo_best: dict[tuple[int, ...], float] = {}
        for entry in pull_history:
            key = tuple(entry["indices"])
            combo_best[key] = max(combo_best.get(key, -1.0), float(entry["score"]))
        best_indices, best_pull_score, selection_note, selection_predictions = select_best_observed(
            combo_best, pull_history, posterior_mean
        )
        if not best_indices:
            raise OptimizerContractError("MASPOB evaluated no prompt combination")
        selected = PromptBundle(
            roles={role: pool[role][best_indices[index]] for index, role in enumerate(roles)},
            demos=seed_bundle.demos,
            metadata={
                **dict(seed_bundle.metadata),
                "optimizer": self.method,
                "selection_protocol": DEFAULT_SELECTION_PROTOCOL,
            },
        )
        checkpoints.append(
            native_checkpoint(
                method=self.method,
                iteration=len(pull_history),
                bundle=selected,
                budget=budget,
                state={
                    "complete": True,
                    "best_indices": best_indices,
                    "best_pull_score": best_pull_score,
                    "selection_predictions": selection_predictions,
                },
            )
        )
        artifact = OptimizerResult(
            layout=SEARCH_LAYOUT,
            method=self.method,
            implementation_kind=self.implementation_kind,
            production_eligible=not self.validation_one_cycle,
            cell_id=cell.cell_id,
            seed_bundle=seed_bundle,
            incumbent_bundle=selected,
            budget_snapshot=dict(budget.snapshot()),
            stop_reason=StopReason.ROLLOUT_BUDGET_SPENT,
            learning_curve=tuple(learning_curve),
            checkpoints=tuple(checkpoints),
            reflection_requests=tuple(reflection.requests),
            metadata={
                "native_backend": "MASPOB WorkflowGAT + linear Fisher/LinUCB",
                "native_optimizer_executed": True,
                "native_source_core_reused": True,
                "validation_only": self.validation_one_cycle,
                "method_provenance": {
                    "classification": "official-source adaptation",
                    "adaptation_id": ADAPTATION_ID,
                    "official_source": SOURCE_URL,
                    "paper": SOURCE_PAPER,
                    "official_source_commit": SOURCE_COMMIT,
                    "license_status": LICENSE_STATUS,
                    "core_module_sha256": verified_core_source_hashes(),
                    "benchmark_adaptations": list(BENCHMARK_ADAPTATIONS),
                },
                "effective_lifecycle": {
                    "rollout_limit": lifecycle_rollout_limit,
                    "pretrain_pulls": pretrain_pulls,
                    "ucb_pulls": sum(1 for entry in pull_history if entry["stage"] == "ucb"),
                },
                "run_seed": run_seed,
                "rng_protocol": RNG_PROTOCOL,
                "row_sampler_seed": row_sampler_seed,
                "generation_task": generation_task,
                "num_prompts_target": settings.num_prompts,
                "minibatch_size": settings.minibatch_size,
                "gnn": {**GNN_SETTINGS, "topology": graph},
                "ucb": dict(UCB_SETTINGS),
                "topology_graph": graph,
                "topology_description": topology_description,
                "pool_stats": pool_stats,
                "embedding_info": embedding_info,
                "initial_indices": initial_indices,
                "initialization_protocol": initialization_protocol,
                "initialization_predictions": initialization_predictions,
                "pull_history": pull_history,
                "selection_protocol": DEFAULT_SELECTION_PROTOCOL,
                "selection_note": selection_note,
                "selection_predictions": selection_predictions,
                "best_indices": best_indices,
                "best_pull_score": best_pull_score,
                "reflection_usage": dict(reflection.usage),
            },
        )
        persist_optimizer_artifact(runner, artifact)
        return artifact


__all__ = ["MASPOBOptimizer"]
