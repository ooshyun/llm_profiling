#!/usr/bin/env bash
# Convert HF safetensors model → F16 GGUF → quantized GGUF (Q4_K_M default)
#
# Usage:
#   ./convert_to_gguf.sh <hf_repo> <out_basename> [quant_type]
#
# Example:
#   ./convert_to_gguf.sh meta-llama/Meta-Llama-3.1-8B-Instruct llama31_8b_inst Q4_K_M
#   ./convert_to_gguf.sh Qwen/Qwen3.5-35B-A3B qwen35_35b_a3b Q4_K_M
#
# Env overrides:
#   LLAMA_DIR     llama.cpp checkout (default /tmp/llama.cpp-build)
#   WORK_DIR      working area (default $HOME/conversion)
#   HF_TOKEN      Hugging Face token if model is gated
#   KEEP_F16      1 to keep intermediate F16 GGUF (default delete after quant)
#   KEEP_HF       1 to keep downloaded safetensors (default delete after F16)
set -euo pipefail

HF_REPO="${1:-}"
OUT_BASE="${2:-}"
QUANT="${3:-Q4_K_M}"

if [ -z "$HF_REPO" ] || [ -z "$OUT_BASE" ]; then
    cat <<EOF
Usage: $0 <hf_repo> <out_basename> [quant_type]
  hf_repo       e.g. meta-llama/Meta-Llama-3.1-8B-Instruct
  out_basename  e.g. llama31_8b_inst  (output: <out_basename>-Q4_K_M.gguf)
  quant_type    Q4_K_M (default), Q4_0, Q5_K_M, Q6_K, Q8_0, F16
EOF
    exit 1
fi

LLAMA_DIR="${LLAMA_DIR:-/tmp/llama.cpp-build}"
WORK_DIR="${WORK_DIR:-$HOME/conversion}"
HF_DIR="$WORK_DIR/hf/$OUT_BASE"
F16_GGUF="$WORK_DIR/gguf/${OUT_BASE}-F16.gguf"
OUT_GGUF="$WORK_DIR/gguf/${OUT_BASE}-${QUANT}.gguf"
KEEP_F16="${KEEP_F16:-0}"
KEEP_HF="${KEEP_HF:-0}"

mkdir -p "$HF_DIR" "$(dirname "$F16_GGUF")"

CONVERT_PY="$LLAMA_DIR/convert_hf_to_gguf.py"
QUANTIZE="$LLAMA_DIR/build-cuda/bin/llama-quantize"
[ -x "$QUANTIZE" ] || QUANTIZE="$LLAMA_DIR/build-cpu/bin/llama-quantize"

[ -f "$CONVERT_PY" ] || { echo "ERROR: convert_hf_to_gguf.py not found at $CONVERT_PY"; exit 1; }
[ -x "$QUANTIZE"   ] || { echo "ERROR: llama-quantize not found at $QUANTIZE — run build_llama_orin.sh first"; exit 1; }

echo "==========================================="
echo " GGUF conversion"
echo "  repo  : $HF_REPO"
echo "  out   : $OUT_GGUF"
echo "  work  : $WORK_DIR"
echo "==========================================="

# 1. Download safetensors from HF
echo ""
echo ">>> [1/3] downloading $HF_REPO  →  $HF_DIR"
python3 - <<PYEOF
import os
from huggingface_hub import snapshot_download
token = os.environ.get("HF_TOKEN") or None
snapshot_download(
    repo_id="$HF_REPO",
    local_dir="$HF_DIR",
    token=token,
    allow_patterns=[
        "*.safetensors", "*.json", "*.model", "*.txt", "*.bin",
        "tokenizer*", "tokenizer/**", "added_tokens*",
    ],
)
print("download ok")
PYEOF

# 2. convert HF safetensors → F16 GGUF
echo ""
echo ">>> [2/3] converting safetensors → F16 GGUF"
python3 "$CONVERT_PY" "$HF_DIR" \
    --outfile "$F16_GGUF" \
    --outtype f16

# Optionally remove HF safetensors to save disk before quantize
if [ "$KEEP_HF" != "1" ]; then
    echo "    (deleting safetensors at $HF_DIR — set KEEP_HF=1 to keep)"
    rm -rf "$HF_DIR"
fi

# 3. Quantize F16 → target
echo ""
echo ">>> [3/3] quantizing $QUANT"
"$QUANTIZE" "$F16_GGUF" "$OUT_GGUF" "$QUANT"

# Optionally remove intermediate F16
if [ "$KEEP_F16" != "1" ]; then
    echo "    (deleting F16 intermediate $F16_GGUF — set KEEP_F16=1 to keep)"
    rm -f "$F16_GGUF"
fi

echo ""
echo "==========================================="
echo " Done."
ls -lh "$OUT_GGUF"
echo "==========================================="
