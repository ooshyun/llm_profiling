#!/bin/bash
# Build llm-profiler for all target platforms using Docker
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/build/output"

mkdir -p "$OUTPUT_DIR"/{rpi4,orin-cuda,orin-cpu,android}

cd "$PROJECT_DIR"

echo "========================================"
echo "  tiny-llm-profiler: Docker Build All"
echo "========================================"

# 1. aarch64 CPU (RPi4, Pi Zero 2W) — Docker buildx QEMU arm64
echo ""
echo ">>> [1/3] Building for aarch64 CPU (RPi4) via Docker buildx..."
docker buildx build \
    --platform linux/arm64 \
    --load \
    -f build/docker/Dockerfile.aarch64-native \
    -t llm-profiler-arm64 \
    . 2>&1 | tail -10

CONTAINER_ID=$(docker create llm-profiler-arm64)
docker cp "$CONTAINER_ID:/output/." "$OUTPUT_DIR/rpi4/"
docker rm "$CONTAINER_ID" > /dev/null
echo "  ✅ RPi4: $(file "$OUTPUT_DIR/rpi4/llm-profiler" 2>/dev/null | head -1)"

# 2. Android aarch64 (NDK cross-compile) — Docker x86
echo ""
echo ">>> [2/3] Building for Android aarch64 via Docker NDK..."
docker build \
    -f build/docker/Dockerfile.android-ndk \
    -t llm-profiler-android \
    . 2>&1 | tail -10

CONTAINER_ID=$(docker create llm-profiler-android)
docker cp "$CONTAINER_ID:/output/." "$OUTPUT_DIR/android/"
docker rm "$CONTAINER_ID" > /dev/null
echo "  ✅ Android: $(file "$OUTPUT_DIR/android/llm-profiler" 2>/dev/null | head -1)"

# 3. Orin CUDA — Remote Docker build on Orin device
echo ""
echo ">>> [3/3] Building for Orin CUDA via remote Docker..."
echo "  Syncing Dockerfile + profiler source to Orin..."
ssh home.orin.local "mkdir -p /tmp/tiny-llm-build"
scp build/docker/Dockerfile.orin-cuda home.orin.local:/tmp/tiny-llm-build/
scp collection/track1_ggml/llm_profiler_main.cpp home.orin.local:/tmp/tiny-llm-build/

# Create build context on Orin
ssh home.orin.local "cd /tmp/tiny-llm-build && \
    mkdir -p collection/track1_ggml && \
    cp llm_profiler_main.cpp collection/track1_ggml/ && \
    docker build -f Dockerfile.orin-cuda -t llm-profiler-orin . 2>&1 | tail -15"

echo "  Extracting binaries from Orin..."
ssh home.orin.local "
    CONTAINER_ID=\$(docker create llm-profiler-orin) && \
    mkdir -p /tmp/orin-output && \
    docker cp \"\$CONTAINER_ID:/output/.\" /tmp/orin-output/ && \
    docker rm \$CONTAINER_ID > /dev/null
"

scp -r home.orin.local:/tmp/orin-output/cuda/* "$OUTPUT_DIR/orin-cuda/" 2>/dev/null || true
scp -r home.orin.local:/tmp/orin-output/cpu/* "$OUTPUT_DIR/orin-cpu/" 2>/dev/null || true
echo "  ✅ Orin CUDA: $(file "$OUTPUT_DIR/orin-cuda/llm-profiler" 2>/dev/null | head -1)"
echo "  ✅ Orin CPU: $(file "$OUTPUT_DIR/orin-cpu/llm-profiler" 2>/dev/null | head -1)"

echo ""
echo "========================================"
echo "  Build Complete!"
echo "========================================"
ls -lh "$OUTPUT_DIR"/*/llm-profiler 2>/dev/null
