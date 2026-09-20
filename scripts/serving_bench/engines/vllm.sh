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
UTIL="${2:-0.5}"

echo ">>> vLLM  model=$MODEL  gpu-memory-utilization=$UTIL"
echo ">>> Tegra shares one pool between GPU and host, so 0.9 (vLLM's default)"
echo ">>> would claim host RAM too; step down to 0.45/0.40 on OOM and record"
echo ">>> whichever value worked — it is a result, not a nuisance."

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
