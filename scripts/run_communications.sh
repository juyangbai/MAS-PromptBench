#!/usr/bin/env bash
# Message-format baseline sweep.
#
# Runs each msg cell as a standalone batch (one process per cell) against a
# single OpenAI-compatible endpoint. No sharding, no merge step.
#
# Override the sweep / endpoint via environment variables:
#   VLLM_BASE_URL  MODEL_ID  TOPOLOGIES  DATASETS  FORMATS  <DATASET>_LIMIT
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
export TOOLHOP_ALLOW_DATASET_EXEC="${TOOLHOP_ALLOW_DATASET_EXEC:-1}"

TOPOLOGIES="${TOPOLOGIES:-independent sequential centralized decentralized}"
DATASETS="${DATASETS:-hotpotqa lcb toolhop apibank swe}"
FORMATS="${FORMATS:-freeform semi_structured structured_soft}"
declare -A LIMIT=([hotpotqa]=100 [lcb]=50 [toolhop]=100 [apibank]=100 [swe]=30)

for topo in $TOPOLOGIES; do
  for ds in $DATASETS; do
    for fmt in $FORMATS; do
      mod="communications.${topo}.${ds}.${ds}_${fmt}"
      echo "=== $mod (limit=${LIMIT[$ds]:-50}) ==="
      python -m "$mod" --batch --limit "${LIMIT[$ds]:-50}" || echo "  FAILED: $mod"
    done
  done
done
