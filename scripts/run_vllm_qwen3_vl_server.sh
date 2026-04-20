#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8004}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.98}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
SERVE_MODEL="${MODEL}"

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
  "${SERVE_MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" 
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --enable-auto-tool-choice
  --enforce-eager
  --async-scheduling
  --tool-call-parser "${TOOL_CALL_PARSER}"
  --default-chat-template-kwargs '{"enable_thinking": false}'
)

vllm serve "${VLLM_ARGS[@]}"

