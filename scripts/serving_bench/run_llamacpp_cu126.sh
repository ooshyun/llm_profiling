#!/usr/bin/env bash
# Full llama.cpp S1/S2/S3 sweep on the CUDA 12.6 build, both models.
# Run ON the Orin: ~/serving_bench/run_llamacpp_cu126.sh
#
# Follows the cold-start protocol from README.md: the engine is restarted
# before S1 and again before S2, and nothing probes it in between — a warmed
# prompt cache silently turns cold TTFT into warm TTFT, which is exactly the
# contamination that invalidated the first Phase-0 8B run.
set -uo pipefail
cd "$(dirname "$0")"

BIN=$HOME/llama.cpp-build/build-cuda/bin/llama-server
RES=results
LOGS=logs
RUNTIME=cu12.6
VER=6fdd0ac
mkdir -p "$RES" "$LOGS"

stop_server() {
  pkill -x llama-server 2>/dev/null   # -x, never -f: -f self-matches over ssh
  for _ in $(seq 1 30); do pgrep -x llama-server >/dev/null || break; sleep 1; done
  sleep 2
}

start_server() {  # $1 = 8b|35b   $2 = np
  local key="$1" np="$2" m
  case "$key" in
    8b)  m=$HOME/models/Qwen3-8B-Q4_K_M.gguf ;;
    35b) m=$HOME/models/Qwen3.5-35B-A3B-Q4_K_M.gguf ;;
  esac
  nohup "$BIN" -m "$m" -ngl 99 -c 8192 -t 8 --host 127.0.0.1 --port 8080 \
    --reasoning off -np "$np" > "$LOGS/srv_${key}_np${np}.log" 2>&1 < /dev/null &
  for _ in $(seq 1 60); do
    curl -sf http://127.0.0.1:8080/v1/models >/dev/null 2>&1 && { echo "  server up ($key np=$np)"; return 0; }
    sleep 5
  done
  echo "  SERVER FAILED TO START ($key np=$np)"; tail -5 "$LOGS/srv_${key}_np${np}.log"; return 1
}

bench() {  # $1 = model-key  $2 = model-label  $3 = scenario
  local key="$1" label="$2" sc="$3"
  local out="$RES/llamacpp_cu126_${key}_${sc}.jsonl"
  [ -f "$out" ] && { echo "  $sc already done, skipping"; return 0; }
  echo "  running $sc -> $out"
  python3 bench.py run --engine llama.cpp --engine-version "$VER" --runtime "$RUNTIME" \
    --model "$label" --fmt gguf-q4km --base-url http://127.0.0.1:8080 \
    --scenario "$sc" --config scenarios.yaml --out "$out" \
    >> "$LOGS/bench_${key}_${sc}.log" 2>&1
  echo "  $sc done: $(wc -l < "$out") records"
}

for pair in "8b:qwen3-8b" "35b:qwen3.5-35b-a3b"; do
  key="${pair%%:*}"; label="${pair##*:}"
  echo "=== $label ($key) — $(date -Is) ==="

  echo "-- S1 (cold server, np=1) --"
  stop_server; start_server "$key" 1 || exit 1
  bench "$key" "$label" S1

  echo "-- S2 (restart first: turn-1 TTFT must be cold) --"
  stop_server; start_server "$key" 1 || exit 1
  bench "$key" "$label" S2

  echo "-- S3 (np=8; ctx 8192 splits to 1024/slot) --"
  stop_server; start_server "$key" 8 || exit 1
  bench "$key" "$label" S3
done

stop_server
echo "=== ALL DONE $(date -Is) ==="
ls -la "$RES"/llamacpp_cu126_*.jsonl
