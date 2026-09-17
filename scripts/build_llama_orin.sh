#!/usr/bin/env bash
# Native CUDA build of llama.cpp on Jetson AGX Orin.
# Produces: build-cuda/bin/{llama-cli,llama-quantize,llama-server,llm-profiler}
# Run on Orin (NOT host). Assumes CUDA toolkit is installed.
#
# Usage:  ./build_llama_orin.sh [llama-cpp-dir]
#         default dir: $HOME/llama.cpp-build
set -euo pipefail

LLAMA_DIR="${1:-$HOME/llama.cpp-build}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
PROFILER_SRC_DIR="${PROFILER_SRC_DIR:-$(cd "$(dirname "$0")/.." && pwd)/collection/track1_ggml}"

echo "==================================================="
echo " llama.cpp Orin native CUDA build"
echo "  llama_dir : $LLAMA_DIR"
echo "  jobs      : $JOBS"
echo "  profiler  : $PROFILER_SRC_DIR (optional)"
echo "==================================================="

# 1. Clone if missing
if [ ! -d "$LLAMA_DIR" ]; then
    echo ">>> cloning llama.cpp"
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"

# 2. Configure CUDA build
mkdir -p build-cuda
cd build-cuda
if [ ! -f CMakeCache.txt ]; then
    echo ">>> configuring (CUDA enabled)"
    cmake -DGGML_CUDA=ON \
          -DCMAKE_CUDA_ARCHITECTURES=87 \
          -DCMAKE_BUILD_TYPE=Release \
          ..
fi

# 3. Build common targets
echo ">>> building llama-cli, llama-quantize, llama-server"
cmake --build . --target llama-cli       -j "$JOBS"
cmake --build . --target llama-quantize  -j "$JOBS"
cmake --build . --target llama-server    -j "$JOBS"

# 4. (Optional) build llm-profiler if profiler source is present
if [ -d "$PROFILER_SRC_DIR" ] && [ -f "$PROFILER_SRC_DIR/llm_profiler_main.cpp" ]; then
    echo ">>> building llm-profiler (custom)"
    EXAMPLE_DIR="$LLAMA_DIR/tools/llm-profiler"
    if [ ! -d "$EXAMPLE_DIR" ]; then
        mkdir -p "$EXAMPLE_DIR"
        cp "$PROFILER_SRC_DIR/llm_profiler_main.cpp" "$EXAMPLE_DIR/"
        cp "$PROFILER_SRC_DIR/CMakeLists.txt"        "$EXAMPLE_DIR/" 2>/dev/null || true
        # Wire into top-level CMake if not already
        if ! grep -q "llm-profiler" "$LLAMA_DIR/tools/CMakeLists.txt"; then
            echo "add_subdirectory(llm-profiler)" >> "$LLAMA_DIR/tools/CMakeLists.txt"
        fi
        # Re-configure
        cd "$LLAMA_DIR/build-cuda"
        cmake ..
    fi
    cmake --build . --target llm-profiler -j "$JOBS" || echo "  (llm-profiler target not registered; skipping)"
fi

# 5. CPU-only build for fallback (smaller, no CUDA dep at runtime)
cd "$LLAMA_DIR"
if [ ! -d build-cpu ]; then
    mkdir build-cpu
    cd build-cpu
    cmake -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release ..
    cmake --build . --target llama-cli      -j "$JOBS"
    cmake --build . --target llama-quantize -j "$JOBS"
fi

echo ""
echo "==================================================="
echo " Build complete. Binaries at:"
echo "   $LLAMA_DIR/build-cuda/bin/   (CUDA)"
echo "   $LLAMA_DIR/build-cpu/bin/    (CPU)"
ls -lh "$LLAMA_DIR/build-cuda/bin/" 2>/dev/null | head -10
echo "==================================================="
