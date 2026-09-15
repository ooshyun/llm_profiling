#!/usr/bin/env bash
# scripts/serving_bench/orin_stage_downloads.sh — run ON the Orin, backgroundable.
# Stages Phase-2 models and images (spec §4 Phase 0.3/0.4). Idempotent.
set -uo pipefail
LOG=~/stage_downloads.log
{
echo "=== $(date) start ==="
python3 - <<'PY'
from huggingface_hub import snapshot_download
for repo in ["Qwen/Qwen3-8B", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"]:
    print("downloading", repo, flush=True)
    p = snapshot_download(repo, cache_dir="/home/cochl/hf")
    print("done", repo, "->", p, flush=True)
PY
docker pull mitakad/vllm:0.22.0-r36.5.tegra-aarch64-cp312-cu129-24.04
SGLANG_TAG=$(python3 - <<'PY'
import json, urllib.request
u = ("https://hub.docker.com/v2/repositories/mitakad/sglang/tags"
     "?page_size=50&ordering=last_updated")
tags = json.load(urllib.request.urlopen(u))["results"]
r36 = [t["name"] for t in tags if "r36" in t["name"] and "tegra" in t["name"]]
print(r36[0] if r36 else "")
PY
)
if [ -n "$SGLANG_TAG" ]; then
  echo "pulling mitakad/sglang:$SGLANG_TAG"
  docker pull "mitakad/sglang:$SGLANG_TAG"
else
  echo "WARNING: no mitakad/sglang r36 tegra tag found — Phase 2 must build via jetson-containers"
fi
echo "=== $(date) done ==="
} >> "$LOG" 2>&1
