#!/usr/bin/env bash
# Fast CUDA-only rebuild of llama.cpp on the Orin into ~/llama.cpp-build (the script that
# produced the 6fdd0ac build on 2026-08-27). Mirror of ~/rebuild_llama.sh on the Orin.
# Skips the CPU fallback build that build_llama_orin.sh also does.
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
LLAMA_DIR=$HOME/llama.cpp-build
JOBS=8

echo "### $(date) start"
if [ ! -d "$LLAMA_DIR/.git" ]; then
  echo "### cloning"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"
echo "### commit: $(git rev-parse --short HEAD)"

mkdir -p build-cuda && cd build-cuda
echo "### configure"
cmake -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=87 \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_CURL=OFF \
      -DGGML_NATIVE=ON \
      ..
echo "### build"
cmake --build . --target llama-cli llama-bench llama-server llama-quantize -j $JOBS
echo "### $(date) done"
ls -lh bin/ | head -20
