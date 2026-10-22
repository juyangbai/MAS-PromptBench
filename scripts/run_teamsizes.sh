#!/usr/bin/env bash
# Team-size baseline sweep.
#
# Runs each teamsizes cell (team-size r ∈ {2,4,8,10} × topology × dataset) as a
# standalone batch — one process per cell against a single endpoint. No sharding.
# BFCL and SWE-bench run exactly their frozen evaluation IDs from
# benchmarks/<dataset>/<dataset>_eval_ids.json (BFCL one AST category per call).
# Exits non-zero if any cell failed.
#
# Override via environment: VLLM_BASE_URL MODEL_ID RVALUES TOPOLOGIES DATASETS OUT_ROOT
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
export TOOLHOP_ALLOW_DATASET_EXEC="${TOOLHOP_ALLOW_DATASET_EXEC:-1}"

RVALUES="${RVALUES:-2 4 8 10}"
TOPOLOGIES="${TOPOLOGIES:-independent sequential centralized decentralized}"
DATASETS="${DATASETS:-gpqa hotpotqa math lcb apps bfcl swe apibank toolhop}"
OUT_ROOT="${OUT_ROOT:-results/teamsizes}"
declare -A LIMIT=([gpqa]=100 [hotpotqa]=100 [math]=100 [lcb]=50 [apps]=50 [apibank]=100 [toolhop]=100)

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
for r in $RVALUES; do
  for topo in $TOPOLOGIES; do
    for ds in $DATASETS; do
      mod="teamsizes.${topo}.${ds}.${ds}_r${r}"
      python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$mod') else 1)" 2>/dev/null || continue
      out="$OUT_ROOT/r${r}/${topo}_${ds}"
      echo "=== $mod -> $out ==="
      case "$ds" in
        bfcl)
          for cat in simple multiple parallel parallel_multiple; do
            # shellcheck disable=SC2046
            python -m "$mod" --category "$cat" $(eval_only_args bfcl "$cat") \
              --out-dir "$out/$cat" || failed+=("$mod ($cat)")
          done ;;
        swe)
          # shellcheck disable=SC2046
          python -m "$mod" $(eval_only_args swe) --eval singularity \
            --out-dir "$out" || failed+=("$mod") ;;
        apibank|toolhop)
          python -m "$mod" --limit "${LIMIT[$ds]}" --out-dir "$out" || failed+=("$mod") ;;
        *)
          python -m "$mod" --batch --limit "${LIMIT[$ds]:-50}" \
            --out "$out/predictions.jsonl" || failed+=("$mod") ;;
      esac
    done
  done
done

if (( ${#failed[@]} )); then
  printf 'FAILED (%d):\n' "${#failed[@]}"; printf '  %s\n' "${failed[@]}"
  exit 1
fi
