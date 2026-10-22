#!/usr/bin/env bash
# Topology baseline sweep.
#
# Runs each topology × dataset cell as a standalone batch — one process per cell
# against a single endpoint. No sharding, no merge step.
#
# Override via environment: VLLM_BASE_URL MODEL_ID DATASETS OUT_ROOT
# BFCL and SWE-bench run exactly their frozen evaluation IDs from
# benchmarks/<dataset>/<dataset>_eval_ids.json (BFCL one AST category per call).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
export TOOLHOP_ALLOW_DATASET_EXEC="${TOOLHOP_ALLOW_DATASET_EXEC:-1}"

DATASETS="${DATASETS:-gpqa hotpotqa math lcb apps bfcl swe apibank toolhop}"
OUT_ROOT="${OUT_ROOT:-results/topologies_baseline}"
# --only flags for the frozen eval IDs of a dataset (optionally one BFCL category).
eval_only_args() {
  python - "$1" "${2:-}" <<'PY'
import json, re, sys
ds, cat = sys.argv[1], sys.argv[2]
ids = json.load(open(f"benchmarks/{ds}/{ds}_eval_ids.json"))["ids"]
if cat:
    ids = [i for i in ids if re.fullmatch(rf"{cat}_\d+", str(i))]
print(" ".join(f"--only {i}" for i in ids))
PY
}

declare -A LIMIT=([gpqa]=100 [hotpotqa]=100 [math]=100 [lcb]=50 [apps]=50 [bfcl]=100 [swe]=30 [apibank]=100 [toolhop]=100)

# topology-path : runner-filename prefix  (single/independent are LangGraph-only)
TOPOS="single:langgraph independent:langgraph \
sequential/langgraph:langgraph sequential/crewai:crewai \
centralized/langgraph:langgraph centralized/autogen:autogen \
decentralized/langgraph:langgraph decentralized/openai_agents:openai_agents"

failed=()
for spec in $TOPOS; do
  path="${spec%%:*}"; fw="${spec##*:}"
  # The OpenAI Agents SDK needs openai>=3; use its isolated install when present.
  pp="${PYTHONPATH:-}"
  [[ "$fw" == openai_agents && -d vendor/openai_agents ]] && pp="$PWD/vendor/openai_agents${PYTHONPATH:+:$PYTHONPATH}"
  for ds in $DATASETS; do
    mod="topologies.${path//\//.}.${ds}.${fw}_${ds}"
    PYTHONPATH="$pp" python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$mod') else 1)" 2>/dev/null || continue
    tag="${path//\//_}_${ds}"
    echo "=== $mod (limit=${LIMIT[$ds]:-50}) ==="
    case "$ds" in
      bfcl)
        for cat in simple multiple parallel parallel_multiple; do
          # shellcheck disable=SC2046
          PYTHONPATH="$pp" python -m "$mod" --category "$cat" $(eval_only_args bfcl "$cat") \
            --out-dir "$OUT_ROOT/${tag}/${cat}" || failed+=("$mod ($cat)")
        done ;;
      swe)
        # shellcheck disable=SC2046
        PYTHONPATH="$pp" python -m "$mod" $(eval_only_args swe) --eval singularity \
          --out-dir "$OUT_ROOT/${tag}" || failed+=("$mod") ;;
      apibank|toolhop)
        PYTHONPATH="$pp" python -m "$mod" --batch --limit "${LIMIT[$ds]:-50}" \
          --out-dir "$OUT_ROOT/${tag}" || failed+=("$mod") ;;
      # other runners need an explicit --out file to persist predictions
      *) PYTHONPATH="$pp" python -m "$mod" --batch --limit "${LIMIT[$ds]:-50}" --out "$OUT_ROOT/${tag}/predictions.jsonl" || failed+=("$mod") ;;
    esac
  done
done

if (( ${#failed[@]} )); then
  printf 'FAILED (%d):\n' "${#failed[@]}"; printf '  %s\n' "${failed[@]}"
  exit 1
fi
