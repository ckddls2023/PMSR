#!/usr/bin/env bash
set -euo pipefail

# Text embedding server for PMSR text KB retrieval.
# Override with, for example:
#   MODEL=intfloat/e5-base-v2 PORT=8006 scripts/run_text_embed_server.sh
#MODEL="${MODEL:-intfloat/e5-base-v2}"
MODEL="${MODEL:-Qwen/Qwen3-Embedding-0.6B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8006}"
DTYPE="${DTYPE:-bfloat16}"
RUNNER="${RUNNER:-pooling}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.24}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"
if [[ -z "${DATA_PARALLEL_SIZE:-}" ]]; then
  visible_devices="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -n "${visible_devices}" ]]; then
    IFS=',' read -r -a _visible_device_array <<< "${visible_devices}"
    _visible_device_count=0
    for _device in "${_visible_device_array[@]}"; do
      if [[ -n "${_device//[[:space:]]/}" ]]; then
        _visible_device_count=$((_visible_device_count + 1))
      fi
    done
    DATA_PARALLEL_SIZE="${_visible_device_count:-1}"
  else
    DATA_PARALLEL_SIZE=1
  fi
fi

VLLM_ARGS=(
  "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --runner "${RUNNER}"
  --dtype "${DTYPE}"
  --enforce-eager
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --data-parallel-size "${DATA_PARALLEL_SIZE}"
)

vllm serve "${VLLM_ARGS[@]}"
