#!/usr/bin/env bash
# GEPA prompt-optimization sweep.
#
# Runs GEPA on each (dataset, topology) cell via the generic pilot
# `real_runner_gepa.pilots.run_gepa_dataset` — one process per cell. The optimized
# (compiled) prompts + scores are written under results/gepa/<topology>_<dataset>/.
# Seed prompts in configs/prompts/ are read-only and never modified.
#
# Override via env: GEPA_REFL_ENDPOINT DATASETS TOPOLOGIES TRAIN_SIZE VAL_SIZE
#   MAX_FULL_EVALS REFLECTION_MINIBATCH_SIZE NUM_THREADS N_AGENTS N_ROUNDS
#   COMPONENT_SELECTOR EARLY_STOP_PATIENCE OUT_ROOT
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export GEPA_REFL_ENDPOINT="${GEPA_REFL_ENDPOINT:-http://localhost:8000/v1}"   # reflection model

DATASETS="${DATASETS:-bfcl gpqa hotpotqa math apps lcb swe apibank toolhop}"
TOPOLOGIES="${TOPOLOGIES:-single independent sequential sequential_crewai centralized centralized_autogen decentralized decentralized_openai}"
TRAIN_SIZE="${TRAIN_SIZE:-25}"
VAL_SIZE="${VAL_SIZE:-25}"
MAX_FULL_EVALS="${MAX_FULL_EVALS:-5}"
REFLECTION_MINIBATCH_SIZE="${REFLECTION_MINIBATCH_SIZE:-3}"
NUM_THREADS="${NUM_THREADS:-4}"
N_AGENTS="${N_AGENTS:-4}"
N_ROUNDS="${N_ROUNDS:-2}"
COMPONENT_SELECTOR="${COMPONENT_SELECTOR:-round_robin}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
OUT_ROOT="${OUT_ROOT:-results/gepa}"

for dataset in $DATASETS; do
  for topo in $TOPOLOGIES; do
    out="$OUT_ROOT/${topo}_${dataset}"
    echo "=== GEPA $dataset / $topo -> $out ==="
    python -m real_runner_gepa.pilots.run_gepa_dataset \
      --dataset "$dataset" --topology "$topo" \
      --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" \
      --max-full-evals "$MAX_FULL_EVALS" \
      --reflection-minibatch-size "$REFLECTION_MINIBATCH_SIZE" \
      --num-threads "$NUM_THREADS" --n-agents "$N_AGENTS" --n-rounds "$N_ROUNDS" \
      --component-selector "$COMPONENT_SELECTOR" \
      --early-stop-patience "$EARLY_STOP_PATIENCE" \
      --skip-perfect-score \
      --out "$out" || echo "  skipped/failed: $dataset/$topo"
  done
done
