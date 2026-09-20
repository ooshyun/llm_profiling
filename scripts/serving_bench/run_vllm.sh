#!/usr/bin/env bash
# vLLM S1/S2/S3 for one model. Run ON the Orin AFTER ./engines/vllm.sh has the
# server up on :8000.
#   ~/serving_bench/run_vllm.sh <8b|35b>
#
# Unlike llama.cpp, the engine is NOT restarted between scenarios here: vLLM's
# startup costs minutes (torch.compile + CUDA graph capture), and its prefix
# cache is content-addressed rather than per-slot, so S1's cold/warm split is
# still meaningful within a single run. S2 turn-1 is therefore NOT a cold
# prefill for vLLM -- it is reported as-is and the difference in protocol is
# recorded in the results doc rather than hidden.
set -uo pipefail
cd "$(dirname "$0")"

RES=results; LOGS=logs; mkdir -p "$RES" "$LOGS"
RUNTIME=cu12.6
VER=0.22.0

case "${1:?usage: run_vllm.sh <8b|35b>}" in
  8b)  LABEL=qwen3-8b;         API=Qwen/Qwen3-8B;                  FMT=bf16 ;;
  35b) LABEL=qwen3.5-35b-a3b;  API=Qwen/Qwen3.5-35B-A3B-GPTQ-Int4; FMT=gptq-int4 ;;
  *)   echo "unknown key: $1" >&2; exit 1 ;;
esac

curl -sf http://127.0.0.1:8000/v1/models >/dev/null || {
  echo "vLLM is not answering on :8000 — start ./engines/vllm.sh $1 first"; exit 1; }

for SC in S1 S2 S3; do
  OUT="$RES/vllm_${RUNTIME}_${1}_${SC}.jsonl"
  [ -f "$OUT" ] && { echo "$SC already done"; continue; }
  echo "=== $LABEL $SC -> $OUT ==="
  python3 bench.py run \
    --engine vllm --engine-version "$VER" --runtime "$RUNTIME" \
    --model "$LABEL" --fmt "$FMT" --api-model "$API" \
    --base-url http://127.0.0.1:8000 \
    --scenario "$SC" --config scenarios.yaml --out "$OUT" \
    --chat-template-kwargs '{"enable_thinking": false}' \
    >> "$LOGS/vllm_bench_${1}_${SC}.log" 2>&1
  echo "  $(wc -l < "$OUT") records, $(grep -c '"ok": true' "$OUT") ok"
done
echo "=== done $(date -Is) ==="
