#!/usr/bin/env bash
# scripts/serving_bench/monitor.sh <logfile> — 1 Hz tegrastats with epoch prefix
set -euo pipefail
OUT="${1:?usage: monitor.sh <logfile>}"
tegrastats --interval 1000 | while IFS= read -r line; do
  echo "$(date +%s) $line"
done >> "$OUT"
