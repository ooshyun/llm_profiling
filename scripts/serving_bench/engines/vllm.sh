#!/usr/bin/env bash
# scripts/serving_bench/engines/vllm.sh <8b|35b> [gpu-mem-util]
# vLLM in the staged Jetson container, OpenAI-compatible on :8000 (spec §6).
#
# Runs in the foreground so the caller sees startup errors; stop with Ctrl-C or
# `docker rm -f vllm_bench`.
set -euo pipefail

IMAGE=mitakad/vllm:0.22.0-r36.5.tegra-aarch64-cp312-cu129-24.04
HF_DIR="$HOME/hf"

# The models live at ~/hf/models--<org>--<name>/, i.e. ~/hf IS the hub cache
# root. HF_HOME would make it look in /hf/hub/ instead and silently re-download
# 23 GB, so set HF_HUB_CACHE. Offline mode turns a path mistake into a loud
# failure rather than a long download.
case "${1:?usage: vllm.sh <8b|35b> [gpu-mem-util]}" in
  8b)  MODEL=Qwen/Qwen3-8B;                    EXTRA=() ;;
  35b) MODEL=Qwen/Qwen3.5-35B-A3B-GPTQ-Int4;   EXTRA=(--quantization moe_wna16) ;;
  *)   echo "unknown model key: $1" >&2; exit 1 ;;
esac
UTIL="${2:-0.75}"

echo ">>> vLLM  model=$MODEL  gpu-memory-utilization=$UTIL"
# Measured 2026-09-20: on Tegra the failure mode runs OPPOSITE to a discrete
# GPU. vLLM sizes its budget from total unified memory (61 GB here), so
# util=0.5 gives a 30.5 GB budget -- the 35B GPTQ weights alone take 21 GB,
# and after activations and CUDA-graph profiling nothing is left, so startup
# dies with "No available memory for the cache blocks. Try INCREASING
# gpu_memory_utilization". 0.75 (~45.8 GB) works and still leaves ~15 GB for
# the OS and docker. So: step UP when the model is large relative to the
# budget; step down only if the host itself starts thrashing.

exec docker run --rm --name vllm_bench \
  --runtime nvidia --network host --ipc host \
  -v "$HF_DIR":/hf \
  -e HF_HUB_CACHE=/hf \
  -e HF_HUB_OFFLINE=1 \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --host 127.0.0.1 --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization "$UTIL" \
    --reasoning-parser qwen3 \
    "${EXTRA[@]}"
