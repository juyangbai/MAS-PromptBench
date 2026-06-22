#!/usr/bin/env bash
# Topology baseline sweep.
#
# Runs each topology × dataset cell as a standalone batch — one process per cell
# against a single endpoint. No sharding, no merge step.
#
# Override via environment: VLLM_BASE_URL MODEL_ID DATASETS OUT_ROOT <DATASET>_LIMIT
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
export TOOLHOP_ALLOW_DATASET_EXEC="${TOOLHOP_ALLOW_DATASET_EXEC:-1}"

DATASETS="${DATASETS:-gpqa hotpotqa math lcb apps bfcl swe apibank toolhop}"
OUT_ROOT="${OUT_ROOT:-results/topologies_baseline}"
declare -A LIMIT=([gpqa]=100 [hotpotqa]=100 [math]=100 [lcb]=50 [apps]=50 [bfcl]=100 [swe]=30 [apibank]=100 [toolhop]=100)

# topology-path : runner-filename prefix  (single/independent are LangGraph-only)
TOPOS="single:langgraph independent:langgraph \
sequential/langgraph:langgraph sequential/crewai:crewai \
centralized/langgraph:langgraph centralized/autogen:autogen \
decentralized/langgraph:langgraph decentralized/openai:openai"

for spec in $TOPOS; do
  path="${spec%%:*}"; fw="${spec##*:}"
  for ds in $DATASETS; do
    mod="topologies.${path//\//.}.${ds}.${fw}_${ds}"
    python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$mod') else 1)" 2>/dev/null || continue
    tag="${path//\//_}_${ds}"
    echo "=== $mod (limit=${LIMIT[$ds]:-50}) ==="
    case "$ds" in
      # apibank/toolhop write to their own default results dir
      apibank|toolhop) python -m "$mod" --batch --limit "${LIMIT[$ds]:-50}" || echo "  FAILED: $mod" ;;
      # other runners need an explicit --out file to persist predictions
      *) python -m "$mod" --batch --limit "${LIMIT[$ds]:-50}" --out "$OUT_ROOT/${tag}/predictions.jsonl" || echo "  FAILED: $mod" ;;
    esac
  done
done
