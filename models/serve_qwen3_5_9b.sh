#!/bin/bash
# Serve Qwen/Qwen3.5-9B as N independent replicas, one vLLM process per GPU,
# on consecutive ports. Each replica has its own OpenAI-compatible API.
#
# GPU list, ports, model cache and memory settings can all be overridden with
# env vars. By default it starts one replica per visible GPU from port 8000.

set -e

# Conda env. vLLM lives in the project env (see environment.yml); activate it
# if needed. Override the env name with CONDA_ENV=...
CONDA_ENV=${CONDA_ENV:-mas-promptbench}
if [ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]; then
    if command -v conda >/dev/null 2>&1; then
        # shellcheck source=/dev/null
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV}"
    fi
fi

# Model cache. Point HF_HOME / MODEL_PATH at your local cache (default:
# $HOME/models).
export HF_HOME=${HF_HOME:-$HOME/models}
export MODEL_PATH=${MODEL_PATH:-$HF_HOME}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
# Linker path for flashinfer's ninja JIT (needs the libcuda.so driver stub).
export LIBRARY_PATH=${LIBRARY_PATH:+$LIBRARY_PATH:}$CONDA_PREFIX/targets/x86_64-linux/lib/stubs

# Server. Override with VLLM_HOST / VLLM_BASE_PORT: plain HOST is taken by
# conda's gcc activation scripts (they set HOST=x86_64-conda-linux-gnu).
HOST=${VLLM_HOST:-0.0.0.0}
BASE_PORT=${VLLM_BASE_PORT:-${BASE_PORT:-8000}}

# Model
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-9B}
# One replica per visible GPU by default; restrict with VLLM_GPU_LIST="0,1,2".
if command -v nvidia-smi >/dev/null 2>&1; then
    _ALL_GPUS=$(seq -s, 0 $(( $(nvidia-smi -L | wc -l) - 1 )))
else
    _ALL_GPUS=0
fi
GPU_LIST_RAW=${VLLM_GPU_LIST:-${GPU_LIST:-$_ALL_GPUS}}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.90}
# "auto" (FP16) works on any GPU. On FP8-capable GPUs (Hopper, Blackwell),
# KV_CACHE_DTYPE=fp8 roughly halves the KV cache size.
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}

IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST_RAW}"
NUM_REPLICAS=${NUM_REPLICAS:-${#GPU_IDS[@]}}

if (( NUM_REPLICAS > ${#GPU_IDS[@]} )); then
    echo "Error: NUM_REPLICAS=${NUM_REPLICAS}, but only ${#GPU_IDS[@]} GPUs in GPU list (${GPU_LIST_RAW})."
    exit 1
fi

# Every requested GPU index must be visible to NVML.
if command -v nvidia-smi >/dev/null 2>&1; then
    NVML_COUNT=$(nvidia-smi -L | wc -l)
    for gpu_id in "${GPU_IDS[@]}"; do
        if (( gpu_id < 0 || gpu_id >= NVML_COUNT )); then
            echo "Error: requested GPU ${gpu_id} is not visible (NVML reports ${NVML_COUNT} GPU(s): 0..$((NVML_COUNT - 1)))."
            exit 1
        fi
    done
fi

# Logs
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR=${LOG_DIR:-${SCRIPT_DIR}/../results/vllm_qwen3_5_9b}
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "vLLM OpenAI API server: ${MODEL_ID}"
echo "=========================================="
echo "Mode:             ${NUM_REPLICAS} replicas, 1 GPU each"
echo "GPU list:         ${GPU_LIST_RAW}"
echo "Host:             ${HOST}"
echo "Ports:            ${BASE_PORT}..$((BASE_PORT + NUM_REPLICAS - 1))"
echo "Max model len:    ${MAX_MODEL_LEN}"
echo "GPU mem util:     ${GPU_MEMORY_UTIL}"
echo "KV cache dtype:   ${KV_CACHE_DTYPE}"
echo "Endpoints:        http://${HOST}:<port>/v1"
echo "Logs:             ${LOG_DIR}/replica_<idx>.log"
echo "=========================================="

# Launch
pids=()

cleanup() {
    echo ""
    echo "Shutting down replicas: ${pids[*]}"
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

for i in $(seq 0 $((NUM_REPLICAS - 1))); do
    gpu_id="${GPU_IDS[$i]}"
    port=$((BASE_PORT + i))
    log_file="${LOG_DIR}/replica_${i}.log"

    echo "[gpu ${gpu_id}] -> port ${port}, log ${log_file}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_ID}" \
        --trust-remote-code \
        --host "${HOST}" \
        --port "${port}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
        --kv-cache-dtype "${KV_CACHE_DTYPE}" \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --download-dir "${MODEL_PATH}" \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml \
        > "${log_file}" 2>&1 &

    pids+=($!)
done

echo ""
echo "All ${NUM_REPLICAS} replicas launched. PIDs: ${pids[*]}"
echo "Tail logs with: tail -f ${LOG_DIR}/replica_*.log"
echo "Press Ctrl-C to stop all replicas."

wait "${pids[@]}"
