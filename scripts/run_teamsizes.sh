#!/usr/bin/env bash
# Team-size baseline sweep.
#
# Runs each teamsizes cell (team-size r ∈ {2,4,8,10} × topology × dataset) as a
# standalone batch — one process per cell against a single endpoint. No sharding.
#
# Override via environment: VLLM_BASE_URL MODEL_ID RVALUES TOPOLOGIES DATASETS <DATASET>_LIMIT
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
export TOOLHOP_ALLOW_DATASET_EXEC="${TOOLHOP_ALLOW_DATASET_EXEC:-1}"

RVALUES="${RVALUES:-2 4 8 10}"
TOPOLOGIES="${TOPOLOGIES:-independent sequential centralized decentralized}"
DATASETS="${DATASETS:-gpqa hotpotqa math lcb apps bfcl swe apibank toolhop}"
declare -A LIMIT=([gpqa]=100 [hotpotqa]=100 [math]=100 [lcb]=50 [apps]=50 [bfcl]=100 [swe]=30 [apibank]=100 [toolhop]=100)

for r in $RVALUES; do
  for topo in $TOPOLOGIES; do
    for ds in $DATASETS; do
      mod="teamsizes.${topo}.${ds}.${ds}_r${r}"
      python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$mod') else 1)" 2>/dev/null || continue
      echo "=== $mod (limit=${LIMIT[$ds]:-50}) ==="
      python -m "$mod" --batch --limit "${LIMIT[$ds]:-50}" || echo "  FAILED: $mod"
    done
  done
done
