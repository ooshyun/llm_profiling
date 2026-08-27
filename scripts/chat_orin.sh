#!/usr/bin/env bash
# tiny-llm-profiler — interactive chat helper for Orin
# Usage:
#   ./chat.sh <key>            interactive chat
#   ./chat.sh <key> -p "..."   one-shot prompt
#   ./chat.sh                  list available keys
#   ./chat.sh --bench <key>    quick speed test (no chat)
#
# Extra llama-cli flags (e.g. --temp 0.7 --top-p 0.9) can be appended after the key.

CLI=/tmp/llama.cpp-build/build-cuda/bin/llama-cli
M=$HOME/models

declare -A MODELS=(
  [qwen3-0.6b]="-m $M/Qwen3-0.6B-Q4_K_M.gguf -c 8192 -ngl 99 -t 8"
  [qwen3-1.7b]="-m $M/Qwen3-1.7B-Q4_K_M.gguf -c 8192 -ngl 99 -t 8"
  [qwen3-4b]="-m $M/Qwen3-4B-Q4_K_M.gguf -c 8192 -ngl 99 -t 8"
  [qwen3-8b]="-m $M/Qwen3-8B-Q4_K_M.gguf -c 8192 -ngl 99 -t 8"
  [qwen3-14b]="-m $M/Qwen3-14B-Q4_K_M.gguf -c 4096 -ngl 99 -t 8"
  [qwen3-32b]="-m $M/Qwen3-32B-Q4_K_M.gguf -c 2048 -ngl 99 -t 8"
  [qwen3-30b-a3b]="-m $M/Qwen3-30B-A3B-Q4_K_M.gguf -c 4096 -ngl 99 -t 8"
  [qwen3.5-27b]="-m $M/Qwen3.5-27B-Q4_K_M.gguf -c 2048 -ngl 99 -t 8"
  [qwen3.5-35b-a3b]="-m $M/Qwen3.5-35B-A3B-Q4_K_M.gguf -c 4096 -ngl 99 -t 8"
  [llama-3.1-8b]="-m $M/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -c 8192 -ngl 99 -t 8"
  [gemma-2-9b]="-m $M/gemma-2-9b-it-Q4_K_M.gguf -c 8192 -ngl 99 -t 8"
  [phi-3.5-mini]="-m $M/Phi-3.5-mini-instruct-Q4_K_M.gguf -c 8192 -ngl 99 -t 8"
  [mistral-7b]="-m $M/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf -c 8192 -ngl 99 -t 8"
)

# Display speeds (gen tok/s) measured 2026-04-28
declare -A SPEED=(
  [qwen3-0.6b]="~38" [qwen3-1.7b]="19.5" [qwen3-4b]="-"
  [qwen3-8b]="7.5"   [qwen3-14b]="4.5"   [qwen3-32b]="2.2"
  [qwen3-30b-a3b]="13.5" [qwen3.5-27b]="2.3" [qwen3.5-35b-a3b]="9.6"
  [phi-3.5-mini]="12.3" [mistral-7b]="8.4"
)

list_keys() {
  printf '%-20s  %-10s  %s\n' KEY 'gen t/s' COMMAND
  printf '%-20s  %-10s  %s\n' '---' '-------' '-------'
  for k in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
    printf '%-20s  %-10s  %s\n' "$k" "${SPEED[$k]:-?}" "${MODELS[$k]}"
  done
}

if [ -z "$1" ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  echo "Usage: $0 <key> [extra llama-cli args]"
  echo "       $0 --bench <key>"
  echo ""
  list_keys
  exit 0
fi

if [ "$1" = "--bench" ]; then
  KEY="$2"
  if [ -z "${MODELS[$KEY]}" ]; then
    echo "Unknown key: $KEY"
    list_keys
    exit 1
  fi
  echo ">>> bench $KEY"
  exec $CLI ${MODELS[$KEY]} -p "The capital of France is" -n 32 -no-cnv -st
fi

KEY="$1"
shift
if [ -z "${MODELS[$KEY]}" ]; then
  echo "Unknown key: $KEY"
  list_keys
  exit 1
fi

echo ">>> chatting with $KEY  (Ctrl+C or /exit to quit;  /clear to reset history)"
exec $CLI ${MODELS[$KEY]} "$@"
