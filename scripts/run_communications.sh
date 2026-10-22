#!/usr/bin/env bash
# Message-format baseline sweep.
#
# Runs each msg cell as a standalone batch (one process per cell) against a
# single OpenAI-compatible endpoint. No sharding, no merge step.
#
# Override the sweep / endpoint via environment variables:
#   VLLM_BASE_URL  MODEL_ID  TOPOLOGIES  DATASETS  FORMATS
# BFCL and SWE-bench run exactly their frozen evaluation IDs from
# benchmarks/<dataset>/<dataset>_eval_ids.json (BFCL one AST category per call).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
export TOOLHOP_ALLOW_DATASET_EXEC="${TOOLHOP_ALLOW_DATASET_EXEC:-1}"

TOPOLOGIES="${TOPOLOGIES:-independent sequential centralized decentralized}"
DATASETS="${DATASETS:-hotpotqa lcb bfcl toolhop apibank swe}"
FORMATS="${FORMATS:-freeform semi_structured structured_soft}"
declare -A LIMIT=([hotpotqa]=100 [lcb]=50 [toolhop]=100 [apibank]=100)

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

failed=()
for topo in $TOPOLOGIES; do
  for ds in $DATASETS; do
    for fmt in $FORMATS; do
      mod="communications.${topo}.${ds}.${ds}_${fmt}"
      case "$ds" in
        bfcl)
          for cat in simple multiple parallel parallel_multiple; do
            echo "=== $mod ($cat) ==="
            # shellcheck disable=SC2046
            python -m "$mod" --batch --category "$cat" $(eval_only_args bfcl "$cat") || failed+=("$mod ($cat)")
          done ;;
        swe)
          echo "=== $mod (eval ids) ==="
          # shellcheck disable=SC2046
          python -m "$mod" --batch $(eval_only_args swe) || failed+=("$mod") ;;
        *)
          echo "=== $mod (limit=${LIMIT[$ds]:-50}) ==="
          python -m "$mod" --batch --limit "${LIMIT[$ds]:-50}" || failed+=("$mod") ;;
      esac
    done
  done
done

if (( ${#failed[@]} )); then
  printf 'FAILED (%d):\n' "${#failed[@]}"; printf '  %s\n' "${failed[@]}"
  exit 1
fi
