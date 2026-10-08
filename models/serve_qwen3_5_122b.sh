#!/bin/bash
# Serve Qwen/Qwen3.5-122B-A10B-FP8 as one tensor-parallel instance across N
# GPUs (default 4), with an OpenAI-compatible API on a single port.
#
# FP8 MoE with ~10B active params. Needs FP8-capable GPUs with enough total
# memory for the native 262144-token context. Every setting below can be
# overridden with an env var.

set -e

# Conda env
# vLLM lives in the project conda env (see environment.yml). Override with CONDA_ENV=...
CONDA_ENV=${CONDA_ENV:-mas-promptbench}
if [ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]; then
    if command -v conda >/dev/null 2>&1; then
        # shellcheck source=/dev/null
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV}"
    fi
fi

# Model cache
export HF_HOME=${HF_HOME:-$HOME/models}
export MODEL_PATH=${MODEL_PATH:-$HF_HOME}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
# Linker path for flashinfer's ninja JIT (needs the libcuda.so driver stub).
export LIBRARY_PATH=${LIBRARY_PATH:+$LIBRARY_PATH:}$CONDA_PREFIX/targets/x86_64-linux/lib/stubs

# Model and parallelism
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-122B-A10B-FP8}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
# Uses the first TENSOR_PARALLEL_SIZE GPUs unless CUDA_VISIBLE_DEVICES is set.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((TENSOR_PARALLEL_SIZE - 1)))}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}        # native context, no YaRN scaling
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.95}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}         # roughly halves the KV cache size

# Server. Override with VLLM_HOST / VLLM_PORT: plain HOST/PORT are taken by
# conda's gcc activation scripts (they set HOST=x86_64-conda-linux-gnu).
HOST=${VLLM_HOST:-0.0.0.0}
PORT=${VLLM_PORT:-8000}

# Logs
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR=${LOG_DIR:-${SCRIPT_DIR}/../results/vllm_qwen3_5_122b}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/server.log"

echo "=========================================="
echo "vLLM OpenAI API server: ${MODEL_ID}"
echo "=========================================="
echo "Mode:             single instance, TP=${TENSOR_PARALLEL_SIZE}"
echo "GPUs:             ${CUDA_VISIBLE_DEVICES}"
echo "Host:             ${HOST}"
echo "Port:             ${PORT}"
echo "Max model len:    ${MAX_MODEL_LEN}  (native, no YaRN)"
echo "GPU mem util:     ${GPU_MEMORY_UTIL}"
echo "KV cache dtype:   ${KV_CACHE_DTYPE}"
echo "Endpoint:         http://${HOST}:${PORT}/v1"
echo "Log:              ${LOG_FILE}"
echo "=========================================="

# Launch
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --trust-remote-code \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --download-dir "${MODEL_PATH}" 2>&1 | tee "${LOG_FILE}"
