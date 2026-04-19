#!/usr/bin/env bash
set -euo pipefail

# Qwen3-VL-Embedding pooling server for PMSR image/multimodal retrieval.
# This serves vLLM's OpenAI-compatible /v1/embeddings API, not chat completions.
MODEL="${MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8007}"
DTYPE="${DTYPE:-bfloat16}"
RUNNER="${RUNNER:-pooling}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.28}"
HF_OVERRIDES="${HF_OVERRIDES:-}"
if [[ -z "${TENSOR_PARALLEL_SIZE:-}" ]]; then
  visible_devices="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -n "${visible_devices}" ]]; then
    IFS=',' read -r -a _visible_device_array <<< "${visible_devices}"
    _visible_device_count=0
    for _device in "${_visible_device_array[@]}"; do
      if [[ -n "${_device//[[:space:]]/}" ]]; then
        _visible_device_count=$((_visible_device_count + 1))
      fi
    done
    TENSOR_PARALLEL_SIZE="${_visible_device_count:-1}"
  else
    TENSOR_PARALLEL_SIZE=1
  fi
fi

VLLM_ARGS=(
  "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --runner "${RUNNER}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --dtype "${DTYPE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
)

if [[ -n "${HF_OVERRIDES}" ]]; then
  VLLM_ARGS+=(--hf-overrides "${HF_OVERRIDES}")
fi

vllm serve "${VLLM_ARGS[@]}"
