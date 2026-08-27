# Qwen3 / Qwen3.5 Model Candidates for Edge Profiling

**Date**: 2026-04-28
**Purpose**: Extend tiny-llm-profiler beyond the original 1-2B test set to find the maximum LLM size that runs on each edge device, using Qwen3 and Qwen3.5 dense + MoE families.

## Background

The initial profiling run covered five 1-2B models (Qwen2.5-1.5B, Llama-3.2-1B, Gemma-2-2B, SmolLM2-1.7B, RWKV-6-1.6B). Memory headroom on all devices was never tested. Qwen3 (released 2025) and Qwen3.5 (released Feb 2026) provide a clean parameter-count ladder from 0.6B to 397B for systematic upper-bound measurement.

## Qwen3 Family (Dense + MoE)

| Variant | Type | Q4_K_M Size | Notes |
|---|---|---:|---|
| Qwen3-0.6B | Dense | 0.40 GB | smallest dense candidate |
| Qwen3-1.7B | Dense | 1.11 GB | comparable to existing Qwen2.5-1.5B |
| Qwen3-4B | Dense | 2.50 GB | RPi4 ceiling candidate |
| Qwen3-8B | Dense | 5.03 GB | mobile mid-tier |
| Qwen3-14B | Dense | 9.00 GB | OP11 ceiling candidate |
| Qwen3-32B | Dense | 19.80 GB | Orin without swap |
| Qwen3-30B-A3B | MoE (3B active) | ~18 GB | MoE comparison |

Source: [Qwen/Qwen3-{4B,8B,14B,32B,30B-A3B}-GGUF](https://huggingface.co/Qwen) (official), [unsloth/Qwen3-{0.6B,1.7B}-GGUF](https://huggingface.co/unsloth) (community).

## Qwen3.5 Family (Dense + MoE)

Released February 2026 in two tiers — Small (0.8B / 2B / 4B / 9B) and Medium (27B / 35B-A3B MoE / 122B-A10B MoE / 397B-A17B MoE).

| Variant | Type | Q4_K_M Size | Notes |
|---|---|---:|---|
| Qwen3.5-0.8B | Dense Small | ~0.50 GB | |
| Qwen3.5-2B | Dense Small | ~1.30 GB | |
| Qwen3.5-4B | Dense Small | ~2.50 GB | |
| Qwen3.5-9B | Dense Small | ~5.60 GB | OP13 ceiling candidate |
| Qwen3.5-27B | Dense Medium | 16.70 GB | Orin candidate |
| Qwen3.5-35B-A3B | MoE (3B active) | ~21 GB | MoE — Orin |
| Qwen3.5-122B-A10B | MoE (10B active) | ~73 GB | Orin requires swap |
| Qwen3.5-397B-A17B | MoE (17B active) | ~240 GB | Skipped (impractical) |

Source: [unsloth/Qwen3.5 collection](https://huggingface.co/collections/unsloth/qwen35), [Unsloth Run-Locally docs](https://unsloth.ai/docs/models/qwen3.5).

## Per-Device Test Plan

### Raspberry Pi 4 (3.4 GB free RAM, 1.8 GB swap, 4× Cortex-A72 @ 1.8 GHz)
- Sequence: Qwen3-1.7B → Qwen3-4B → (fall back: Qwen3.5-2B / 0.8B)
- Swap-off ceiling estimated: **Qwen3-4B Q4_K_M (2.5 GB)** — tight but feasible
- Disk cleanup needed first (only 131 MB free)

### OnePlus 11 — CPH2451 (16 GB total / ~10 GB usable, SD 8 Gen 2)
- Sequence: Qwen3-8B → Qwen3.5-9B → Qwen3-14B → Qwen3.5-27B (likely fail)
- CPU ceiling: **Qwen3-14B Q4 (9 GB)** likely; 27B (16.7 GB) almost certain OOM
- GPU OpenCL via `llm-profiler-opencl -ngl 99` for working sizes

### OnePlus 13 — CPH2645 (12 GB total / ~8 GB usable, SD 8 Gen 3)
- **Not currently connected** (only OP11 attached via ADB at this run)
- Plan deferred until physical swap

### Jetson AGX Orin 64 GB (47 GB free RAM, 30 GB swap, 12× Cortex-A78AE, Ampere 2048 CUDA)
- Sequence: Qwen3-32B → Qwen3.5-27B → Qwen3-30B-A3B (MoE) → Qwen3.5-35B-A3B → Llama-3-70B → Qwen3.5-122B-A10B (swap-bound)
- Swap-off CPU ceiling: ~**Qwen3-32B (19.8 GB)** with headroom for KV cache
- Swap-off GPU CUDA ceiling: same — unified memory
- With 30 GB swap: **Qwen3.5-122B-A10B (~73 GB)** may load (active params 10B), but speed depends on swap bandwidth

### Pi Zero 2W (512 MB)
- Only **Qwen3-0.6B (397 MB)** is realistic; not connected this run

## Memory Budget Notes

- **Q4_K_M** ≈ 0.55 GB per 1B params for dense models (varies with vocab size)
- **MoE** Q4_K_M total file ≈ active+experts; runtime memory roughly equals total weights + active experts' KV cache
- **KV cache** for 4K context, 32-layer, GQA Llama-style: ~0.5-1 GB per model. Add ~1.5 GB headroom on top of file size.
- **Activation** at batch=1, seq~128: hidden_dim × layers × FP16 = 100s of MB
- **OS / runtime overhead**: 500 MB-2 GB depending on platform (Android highest)

Practical rule: **swap-off effective max ≈ 0.6 × usable RAM** (after OS/KV/activation budget).

## Profiling Method (unchanged from initial)

For each (device, model) that successfully generates ≥1 token:
1. Baseline `llama-cli -no-cnv -st -n 16` for tok/s without cb_eval overhead
2. `llm-profiler ... --profiler-output profile.jsonl` for per-tensor timing
3. Pull JSONL → parse via `collection.track1_ggml.profiler_wrapper.parse_profiler_output`
4. Store in SQLite (`storage/db.py`) + Parquet (`storage/parquet_store.py`)
5. Run analysis: sublayer breakdown, block heatmap, bottleneck analyzer

## Excluded From This Run

- **Qwen3.5-397B-A17B** (240 GB Q4) — would consume nearly all Orin disk; not justified for one data point
- **NPU (QNN)** — known broken (`supports_buft()` rejects GGUF mmap, see `npu_troubleshooting_report_20260407.md`)
- **OP13** — not connected
- **Pi Zero 2W** — not connected
- **Track 2 (TVM/MLC-LLM)** — still WIP, separate work

Sources:
- [Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF)
- [Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF)
- [Qwen3-14B-GGUF](https://huggingface.co/Qwen/Qwen3-14B-GGUF)
- [Qwen3-32B-GGUF](https://huggingface.co/Qwen/Qwen3-32B-GGUF)
- [Qwen3-30B-A3B-GGUF](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF)
- [Qwen3.5 — Unsloth Run-Locally](https://unsloth.ai/docs/models/qwen3.5)
- [Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)
- [Qwen3.5-27B-GGUF](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF)
- [Qwen3.5-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF)
- [Qwen3.5 collection](https://huggingface.co/collections/unsloth/qwen35)
