#!/usr/bin/env bash
# MIPRO prompt-optimization sweep.
#
# Runs DSPy MIPROv2 on each (dataset, topology) cell via the generic pilot
# `real_runner_mipro.pilots.run_mipro_dataset` — one process per cell. The optimized
# prompts + selected demos are written under results/mipro/<topology>_<dataset>/.
# Seed prompts in configs/prompts/ are read-only and never modified.
#
# Override via env: MIPRO_REFL_ENDPOINT MIPRO_TASK_ENDPOINTS DATASETS TOPOLOGIES
#   TRAIN_SIZE VAL_SIZE NUM_CANDIDATES NUM_TRIALS NUM_THREADS OUT_ROOT
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export MIPRO_REFL_ENDPOINT="${MIPRO_REFL_ENDPOINT:-http://localhost:8000/v1}"       # reflection model
export MIPRO_TASK_ENDPOINTS="${MIPRO_TASK_ENDPOINTS:-http://localhost:8000/v1}"     # comma-separated task models

DATASETS="${DATASETS:-bfcl gpqa hotpotqa math apps lcb swe apibank toolhop}"
TOPOLOGIES="${TOPOLOGIES:-single independent sequential sequential_crewai centralized centralized_autogen decentralized decentralized_openai}"
TRAIN_SIZE="${TRAIN_SIZE:-25}"
VAL_SIZE="${VAL_SIZE:-25}"
NUM_CANDIDATES="${NUM_CANDIDATES:-3}"
NUM_TRIALS="${NUM_TRIALS:-3}"
NUM_THREADS="${NUM_THREADS:-4}"
OUT_ROOT="${OUT_ROOT:-results/mipro}"

for dataset in $DATASETS; do
  for topo in $TOPOLOGIES; do
    out="$OUT_ROOT/${topo}_${dataset}"
    echo "=== MIPRO $dataset / $topo -> $out ==="
    python -m real_runner_mipro.pilots.run_mipro_dataset \
      --dataset "$dataset" --topology "$topo" \
      --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE" \
      --num-candidates "$NUM_CANDIDATES" --num-trials "$NUM_TRIALS" \
      --num-threads "$NUM_THREADS" \
      --out "$out" || echo "  skipped/failed: $dataset/$topo"
  done
done
