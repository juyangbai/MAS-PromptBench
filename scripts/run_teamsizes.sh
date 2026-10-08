#!/usr/bin/env bash
# Team-size baseline sweep.
#
# Runs each cell (team size r in {2,4,8,10} x topology x dataset) as its own
# batch process against one endpoint. No sharding.
#
# Env overrides: VLLM_BASE_URL MODEL_ID RVALUES TOPOLOGIES DATASETS
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
