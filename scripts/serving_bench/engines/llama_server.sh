#!/usr/bin/env bash
# scripts/serving_bench/engines/llama_server.sh <8b|35b> [np]
# Foreground llama-server on :8080, thinking off, ctx 8192 (spec §6).
set -euo pipefail
BIN=$HOME/llama.cpp-build/build-cuda/bin/llama-server
case "${1:?usage: llama_server.sh <8b|35b> [np]}" in
  8b)  M=$HOME/models/Qwen3-8B-Q4_K_M.gguf ;;
  35b) M=$HOME/models/Qwen3.5-35B-A3B-Q4_K_M.gguf ;;
  *)   echo "unknown model key: $1" >&2; exit 1 ;;
esac
NP="${2:-1}"
exec "$BIN" -m "$M" -ngl 99 -c 8192 -t 8 --host 127.0.0.1 --port 8080 \
     --reasoning off -np "$NP"
