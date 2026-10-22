"""MAPRO's settings and the metadata describing its adaptation regime (no runner or LLM imports)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace

ADAPTATION_ID = "mapro-adapted/star-proxy+deterministic-critic+train-patience"
NODE_JUDGE_INPUTS = "candidate_prompt_and_response"


@dataclass(frozen=True)
class MAPROSettings:
    """MAPRO's knobs.

    Search: ``candidate_count`` prompts per role (seed included), at most
    ``max_iterations`` rounds, stop after ``patience`` rounds without a train
    gain above ``epsilon``. Budget split: every evaluation of an assignment
    charges one rollout per evaluation row (all training rows, or the first
    ``evaluation_batch``); judges and probes use the first ``scoring_batch``
    transcripts and blame the ``feedback_count`` worst. ``num_threads`` bounds
    concurrent rollouts and probes. Environment (read when ``native`` and
    ``src.refine.mutate`` are imported): ``MAPRO_LISTWISE=1`` and
    ``MAPRO_PAPER_ANCHOR=1`` select the paper's listwise scoring and
    latest-selection anchor (the protocol refuses both);
    ``MAPRO_INIT_MAX_TOKENS`` / ``MAPRO_MUTATE_MAX_TOKENS`` cap the native
    rewriter calls when driven by a plain client.
    """

    candidate_count: int = 5
    max_iterations: int = 8
    patience: int = 3
    epsilon: float = 0.0
    scoring_batch: int = 3
    evaluation_batch: int = 0
    feedback_count: int = 3
    num_threads: int = 12
    use_demos: bool = True
    listwise: bool = False
    paper_anchor: bool = False
    init_max_tokens: int = 1024
    mutate_max_tokens: int = 512
    probe_temperature: float = 0.2
    probe_top_p: float = 0.9
    probe_max_tokens: int = 768

    @classmethod
    def from_env(cls) -> MAPROSettings:
        """Defaults with the four ``MAPRO_*`` environment knobs applied."""
        return cls(
            listwise=os.environ.get("MAPRO_LISTWISE", "0") == "1",
            paper_anchor=os.environ.get("MAPRO_PAPER_ANCHOR", "0") == "1",
            init_max_tokens=int(os.environ.get("MAPRO_INIT_MAX_TOKENS", "1024")),
            mutate_max_tokens=int(os.environ.get("MAPRO_MUTATE_MAX_TOKENS", "512")),
        )

    def native_args(self) -> SimpleNamespace:
        """The argument namespace of ``native.optimize_mapro``."""
        return SimpleNamespace(
            K=self.candidate_count,
            max_iters=self.max_iterations,
            patience=self.patience,
            eps=self.epsilon,
            score_batch=self.scoring_batch,
            eval_batch=self.evaluation_batch,
            feedback_samples=self.feedback_count,
            num_threads=self.num_threads,
            no_demos=not self.use_demos,
        )


def regime_name(listwise: bool, latest_selection_anchor: bool) -> str:
    """Name of the scoring/anchor regime recorded in artifacts."""
    scoring = "listwise" if listwise else "pointwise"
    anchor = "latest-selection" if latest_selection_anchor else "best-so-far"
    return f"mapro-adapted/{scoring}+{anchor}-anchor"


def regime_metadata(listwise: bool, latest_selection_anchor: bool) -> dict:
    """Artifact metadata describing the adapted MAPRO regime."""
    return {
        "mapro_adaptation_id": ADAPTATION_ID,
        "mapro_fidelity": "adapted",
        "mapro_regime": regime_name(listwise, latest_selection_anchor),
        "mapro_node_judge_inputs": NODE_JUDGE_INPUTS,
        "mapro_paper_aligned_axes": {
            "listwise_reward_scoring": bool(listwise),
            "latest_selection_anchor": bool(latest_selection_anchor),
        },
        "mapro_nonpaper_axes": [
            "centralized_star_proxy",
            "deterministic_threshold_critic_without_warm_start_or_edge_demo_pools",
            "mixed_9b_122b_phase_backbones",
            "train_trajectory_patience_with_external_validation_gate",
            "single_agent_no_tool_candidate_probes",
            "rewrite_temperature_0.7_vs_paper_0.2",
        ],
    }


__all__ = ["ADAPTATION_ID", "MAPROSettings", "NODE_JUDGE_INPUTS", "regime_metadata", "regime_name"]
