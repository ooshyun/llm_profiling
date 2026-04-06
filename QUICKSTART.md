# Quick Start Guide

## Prerequisites

- Python 3.9+
- Docker (with buildx for cross-compilation)
- SSH access to target devices (RPi4, Jetson Orin)
- ADB for Android devices

## 1. Install

```bash
git clone git@github.com:ooshyun/llm_profiling.git
cd llm_profiling

pip install -e ".[dev]"
```

## 2. Run Tests

```bash
# Local tests (no hardware needed)
python -m pytest tests/ -m "not device" -v

# Device connectivity tests (requires hardware)
python -m pytest tests/ -m device -v
```

## 3. Build Profiler Binaries

### Option A: Use pre-built binaries

Pre-built binaries are in `build/output/`:
```
build/output/
├── rpi4/llm-profiler          # ARM64 Linux (RPi4, Pi Zero 2W)
├── android/llm-profiler       # Android ARM64 (CPU only)
├── android-opencl/llm-profiler # Android ARM64 (CPU + Adreno GPU)
└── android-qnn/llm-profiler   # Android ARM64 (QNN backend, experimental)
```

### Option B: Build from Docker

```bash
# RPi4 / Pi Zero 2W (ARM64 CPU)
docker buildx build --platform linux/arm64 --load \
    -f build/docker/Dockerfile.aarch64-native \
    -t llm-profiler-arm64 .

# Android (NDK cross-compile)
docker buildx build --platform linux/amd64 --load \
    -f build/docker/Dockerfile.android-ndk \
    -t llm-profiler-android .

# Android with OpenCL GPU (Adreno)
docker buildx build --platform linux/amd64 --load \
    -f build/docker/Dockerfile.android-opencl \
    -t llm-profiler-android-opencl .

# Jetson Orin CUDA (build ON the Orin device)
ssh orin "cd /tmp && docker build -f Dockerfile.orin-cuda -t llm-profiler-orin ."

# Extract binaries
CONTAINER_ID=$(docker create llm-profiler-arm64)
docker cp "$CONTAINER_ID:/output/." build/output/rpi4/
docker rm "$CONTAINER_ID"
```

## 4. Download a Model

```bash
pip install huggingface-hub

python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Qwen/Qwen2.5-1.5B-Instruct-GGUF',
                'qwen2.5-1.5b-instruct-q4_k_m.gguf',
                local_dir='models')
"
```

Available models:
| Model | Size | Download |
|-------|------|----------|
| Qwen2.5-1.5B Q4_K_M | 1.0GB | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` |
| Llama-3.2-1B Q4_K_M | 770MB | `bartowski/Llama-3.2-1B-Instruct-GGUF` |
| Gemma-2-2B Q4_K_M | 1.6GB | `bartowski/gemma-2-2b-it-GGUF` |
| SmolLM2-1.7B Q4_K_M | 1.0GB | `bartowski/SmolLM2-1.7B-Instruct-GGUF` |
| RWKV-6-1.6B Q4_K | 947MB | `latestissue/rwkv-6-world-1b6-gguf` |

## 5. Run Profiling

### Raspberry Pi 4

```bash
# Deploy
scp build/output/rpi4/llm-profiler home.rasp4.local:~/
scp models/qwen2.5-1.5b-instruct-q4_k_m.gguf home.rasp4.local:~/

# Profile
ssh home.rasp4.local "./llm-profiler \
    -m qwen2.5-1.5b-instruct-q4_k_m.gguf \
    -p 'Hello, how are you?' -n 16 -t 4 \
    --profiler-output profile.jsonl"

# Collect results
scp home.rasp4.local:~/profile.jsonl data/raw/rpi4/
```

### Jetson Orin (CPU)

```bash
ssh home.orin.local "/path/to/llm-profiler \
    -m /path/to/model.gguf \
    -p 'Hello' -n 16 -t 8 \
    --profiler-output profile.jsonl"
```

### Jetson Orin (GPU CUDA)

```bash
ssh home.orin.local "/path/to/llm-profiler \
    -m /path/to/model.gguf \
    -p 'Hello' -n 16 -ngl 99 \
    --profiler-output profile.jsonl"
```

### Android Phone (CPU)

```bash
adb push build/output/android/llm-profiler /data/local/tmp/
adb push build/output/android/libomp.so /data/local/tmp/
adb push models/qwen2.5-1.5b-instruct-q4_k_m.gguf /data/local/tmp/

adb shell "cd /data/local/tmp && \
    LD_LIBRARY_PATH=/data/local/tmp \
    ./llm-profiler -m qwen2.5-1.5b-instruct-q4_k_m.gguf \
    -p 'Hello' -n 16 -t 8 \
    --profiler-output profile.jsonl"

adb pull /data/local/tmp/profile.jsonl data/raw/android/
```

### Android Phone (GPU OpenCL)

```bash
adb push build/output/android-opencl/llm-profiler /data/local/tmp/llm-profiler-opencl

adb shell "cd /data/local/tmp && \
    LD_LIBRARY_PATH=/vendor/lib64:/data/local/tmp \
    ./llm-profiler-opencl -m model.gguf \
    -p 'Hello' -n 16 -t 4 -ngl 99 \
    --profiler-output profile_gpu.jsonl"
```

## 6. Process Results

```python
from collection.track1_ggml.profiler_wrapper import parse_profiler_output
from storage.db import ProfileDB
from storage.parquet_store import ParquetStore

# Parse JSONL → ProfileRecords
records = parse_profiler_output(
    'data/raw/rpi4/profile.jsonl',
    experiment_id='rpi4_qwen_001',
    device='rpi4',
    model='qwen2.5-1.5b-q4_k_m',
    architecture='qwen',
    quantization='q4_k_m',
    input_tokens=7,
)

# Store in SQLite
db = ProfileDB('data/profiles.db')
db.insert_batch(records)

# Export to Parquet
pq = ParquetStore('data/parquet/results.parquet')
pq.write(records)

# Analyze
import pandas as pd
df = pq.read()
print(df.groupby('sublayer')['latency_us'].sum().sort_values(ascending=False).head(10))
```

## 7. Visualize

### Streamlit Dashboard

```bash
streamlit run presentation/dashboard/app.py
# Open http://localhost:8501
```

### Jupyter Notebooks

```bash
cd presentation/notebooks
jupyter notebook
# Open 01_device_comparison.ipynb
```

## 8. Measure Actual Performance (without profiling overhead)

The profiler adds overhead due to per-node synchronization. To get actual inference speed:

```bash
# Use llama-cli (no cb_eval) with -no-cnv -st flags
./llama-cli -m model.gguf -p 'Hello' -n 16 -t 8 -no-cnv -st
# Look for: [ Prompt: X t/s | Generation: Y t/s ]
```

Compare with profiler results to calculate overhead factor per device.

## Troubleshooting

### Android: "library libomp.so not found"
Push the libomp.so from the NDK:
```bash
adb push build/output/android/libomp.so /data/local/tmp/
```

### Android: llama-cli hangs
Add `-no-cnv -st` flags to disable conversation mode.

### NPU not working
llama.cpp's QNN backend cannot offload GGUF tensors to Hexagon NPU. See [NPU Troubleshooting](claudedocs/npu_troubleshooting_report_20260407.md).
