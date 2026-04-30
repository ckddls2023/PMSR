#!/usr/bin/env bash
set -euo pipefail

# SigLIP2 image embedding server for PMSR FAISS image retrieval.
#
# The request shape follows vLLM's vision embedding online example:
# https://github.com/vllm-project/vllm/blob/main/examples/pooling/embed/vision_embedding_online.py
#
# SigLIP2 image embedding requires the rendered token prompt to be empty for
# image-only inputs. This template renders only text content and emits no role
# labels or whitespace for image-only messages.
#
# Hugging Face model page:
# https://huggingface.co/google/siglip2-giant-opt-patch16-384
#
# Example client payload:
# {
#   "model": "google/siglip2-giant-opt-patch16-384",
#   "messages": [
#     {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]}
#   ],
#   "encoding_format": "float"
# }

MODEL="${MODEL:-google/siglip2-giant-opt-patch16-384}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8008}"
DTYPE="${DTYPE:-bfloat16}"
RUNNER="${RUNNER:-pooling}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.20}"
HF_OVERRIDES="${HF_OVERRIDES:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${SCRIPT_DIR}/siglip2_image_template.jinja}"

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
  --chat-template "${CHAT_TEMPLATE}"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  VLLM_ARGS+=(--trust-remote-code)
fi

if [[ -n "${HF_OVERRIDES}" ]]; then
  VLLM_ARGS+=(--hf-overrides "${HF_OVERRIDES}")
fi

vllm serve "${VLLM_ARGS[@]}"
