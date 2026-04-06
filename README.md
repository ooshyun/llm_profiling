# tiny-llm-profiler

Per-layer LLM inference profiling across heterogeneous edge devices.

## Overview

This project profiles LLM inference at the **tensor operation level** across CPUs, GPUs, and NPUs on edge devices — from Raspberry Pi to Jetson Orin to Android phones. It measures latency, memory, FLOPS, and power per sublayer (attention, FFN, normalization, etc.) to identify bottlenecks and compare architectures.

## Key Results

### 5 Architectures × 7 Compute Targets

| Model | Type | RPi4 CPU | Orin CPU | Orin GPU | OP13 CPU | OP13 GPU | OP11 CPU | OP11 GPU |
|-------|------|---------|---------|---------|---------|---------|---------|---------|
| Qwen2.5-1.5B | Transformer (GQA) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Llama-3.2-1B | Transformer (GQA) | ✅ | ✅ | ✅ | - | - | ✅ | - |
| Gemma-2-2B | Transformer (MQA) | - | ✅ | ✅ | - | - | ✅ | ✅ |
| SmolLM2-1.7B | Transformer | - | ✅ | ✅ | - | - | ✅ | ✅ |
| RWKV-6-1.6B | RNN | - | ✅ | - | - | - | ✅ | - |

**21 profiling datasets, 300K+ per-layer records**

### Performance (cb_eval OFF — actual inference speed)

| Device | Compute | Qwen2.5-1.5B tok/s |
|--------|---------|-------------------|
| RPi 4 | CPU (Cortex-A72) | 3.1 |
| Jetson AGX Orin | CPU (Cortex-A78AE) | 7.0 |
| Jetson AGX Orin | GPU (CUDA, Ampere) | 20.9 |
| OnePlus 13 | CPU (SD 8 Gen 3) | 5.7 |
| OnePlus 13 | GPU (Adreno 750, OpenCL) | 20.9 |
| OnePlus 11 | CPU (SD 8 Gen 2) | 9.2 |

### Key Findings

1. **FFN dominates on all Transformer architectures** (40-65% of decode latency) — MatMul is the bottleneck
2. **Llama-1B has high LM head overhead** (22%) due to 128K vocabulary projection
3. **RWKV (RNN) has completely different profile** — 0% attention/FFN, 89% internal ops
4. **GPU is 3-4x faster than CPU** in actual inference, but cb_eval profiling adds 2-14x overhead
5. **NPU (Hexagon) does not work** with llama.cpp's experimental QNN backend — buffer type incompatibility

## Architecture

```
┌─ Layer 4: Presentation ──────────────────────────────┐
│  Streamlit Dashboard  │  Jupyter Notebooks (7 figures)│
└──────────────┬───────────────┬────────────────────────┘
┌─ Layer 3: Analysis ──┴───────┴────────────────────────┐
│  DeviceComparator  │  ArchitectureComparator          │
│  BottleneckAnalyzer│  StatisticsEngine                │
└──────────────────────┬────────────────────────────────┘
┌─ Layer 2: Storage ───┴────────────────────────────────┐
│  SQLite (real-time)  │  Parquet (batch analysis)      │
└───────────┬──────────────┬────────────────────────────┘
┌─ Layer 1: Collection ────┴────────────────────────────┐
│  Track 1: ggml cb_eval (C callback, per-tensor timing)│
│  Track 2: TVM profiler (Nsight, OpenCL events) [WIP]  │
│  Platform Monitor (thermal, power, freq per device)   │
└───────────────────────────────────────────────────────┘
```

## Profiling Method

Uses llama.cpp's `ggml_backend_sched_set_eval_callback()` to intercept every tensor operation during inference:

```c
// Each tensor node triggers: ask=true → compute → sync → ask=false
static bool profiler_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        record_start(t->name, ggml_time_us(), ggml_nbytes(t), t->op);
        return true;
    } else {
        record_end(ggml_time_us());  // per-node wall-clock timing
        return true;
    }
}
```

**Important**: cb_eval forces per-node synchronization, adding overhead (2-14x on GPU). Absolute speeds are not representative — use layer-level proportions for analysis.

## Devices

| Device | SoC | CPU | GPU | RAM | Connection |
|--------|-----|-----|-----|-----|------------|
| Raspberry Pi 4 | BCM2711 | Cortex-A72 4C @ 1.8GHz | - | 3.7GB | SSH |
| Jetson AGX Orin 64GB | Tegra | Cortex-A78AE 12C @ 2.2GHz | Ampere 2048 CUDA | 64GB | SSH |
| OnePlus 13 (CPH2645) | SM8650 | SD 8 Gen 3 | Adreno 750v2 | 12GB | ADB |
| OnePlus 11 (CPH2451) | SM8550 | SD 8 Gen 2 | Adreno 740v2 | 16GB | ADB |

## Project Structure

```
tiny-llm-profiler/
├── schemas/                    # Pydantic data models (ProfileRecord, PlatformSnapshot)
├── collection/
│   ├── track1_ggml/            # C profiler (cb_eval) + Python parser
│   ├── track2_tvm/             # MLC-LLM/TVM profiler (Nsight, OpenCL)
│   └── platform_monitor/      # Per-device hardware monitors (RPi, Jetson, Android)
├── transport/                  # JSONL local store + remote collection (SCP, ADB)
├── storage/                    # SQLite + Parquet dual store
├── analysis/                   # Comparators, bottleneck analyzer, statistics
├── presentation/
│   ├── dashboard/              # Streamlit real-time dashboard
│   └── notebooks/              # Jupyter notebooks for paper figures
├── build/
│   ├── docker/                 # Dockerfiles for all platforms
│   └── output/                 # Compiled binaries per target
├── configs/                    # Device, model, experiment configs
├── data/
│   ├── raw/                    # Per-device JSONL profiling output
│   └── parquet/                # Aggregated Parquet files
├── models/                     # GGUF model files (not tracked in git)
├── vendor/                     # llama.cpp, QAIRT SDK, etc.
├── claudedocs/                 # Research reports, figures, troubleshooting
└── docs/superpowers/           # Design specs, plans
```

## Tests

```bash
# Local tests (no device required)
python -m pytest tests/ -m "not device" -v

# Device tests (requires connected hardware)
python -m pytest tests/ -m device -v

# All tests
python -m pytest tests/ -v
```

**119 local tests + 10 device tests**

## Known Limitations

- **cb_eval overhead**: GPU/NPU profiling adds 2-14x synchronization overhead. Use relative proportions, not absolute speeds.
- **NPU (QNN)**: llama.cpp's experimental QNN backend cannot offload GGUF tensors to NPU — `supports_buft()` rejects CPU-mmap'd buffers. See `claudedocs/npu_troubleshooting_report_20260407.md`.
- **RWKV parser**: RWKV tensor names not fully mapped — 89% show as "internal". Needs RWKV-specific parser.

## Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Setup and first profiling run |
| [Design Spec](docs/superpowers/specs/2026-04-05-tiny-llm-profiler-design.md) | Full architecture specification |
| [llama.cpp Architecture](docs/superpowers/specs/2026-04-05-llama-cpp-architecture.md) | Internal architecture reference |
| [NPU Troubleshooting](claudedocs/npu_troubleshooting_report_20260407.md) | QNN NPU failure analysis |
| [NPU Future Plan](claudedocs/npu_future_development_plan_20260407.md) | Genie SDK / ExecuTorch paths |
| [Edge LLM Research](claudedocs/edge_llm_research_report.md) | Device/model/framework survey |

## License

MIT
