# Max LLM Size on Edge Devices — Multi-Family Sweep

**Date**: 2026-04-28 / 2026-04-29 (extended)
**Devices tested**: Raspberry Pi 4 (CPU), Jetson AGX Orin 64GB (CPU + GPU CUDA), OnePlus 11 (CPH2451, SD 8 Gen 2; CPU + GPU OpenCL), OnePlus 13 (CPH2645, SD 8 Gen 3; CPU + GPU OpenCL)
**Framework**: llama.cpp build b1-4aa962e, GGUF Q4_K_M
**Methodology**: For each (device, model), run `llama-cli -no-cnv -st -n 16 -c {context}` for baseline tok/s, then `llm-profiler` for per-tensor cb_eval profiling. Sizes pushed in increasing order until OOM / refusal / unusable speed.

## Per-Device Maxima (Qwen3 Dense)

| Device | Compute | RAM | Max usable Q4_K_M | gen tok/s | Resident | Notes |
|---|---|---|---|---:|---:|---|
| **Raspberry Pi 4** | CPU (Cortex-A72 ×4) | 3.7 GB | **Qwen3-4B** (2.5 GB) | 1.6 | 2.75 GB | `-c 512` |
| **OnePlus 13** | CPU (SD 8 Gen 3) | 12 GB / 5.7 GB free | **Qwen3-8B** (4.7 GB) | 6.8 | 5.2 GB | 14B loads (8.9 GB) but 0.2 t/s — swap-bound |
| **OnePlus 13** | GPU OpenCL (Adreno 750) | 12 GB shared | **Qwen3-8B** (4.7 GB) | 5.8 | 5.2 GB | |
| **OnePlus 11** | CPU (SD 8 Gen 2) | 16 GB / 10.3 GB free | **Qwen3-14B** (8.6 GB) | 1.7 | 9.05 GB | Qwen3.5-27B (16 GB) loads via mmap, 0.1 t/s |
| **OnePlus 11** | GPU OpenCL (Adreno 740) | 16 GB shared | **Qwen3-4B** (2.5 GB)* | 7.3 | 3.0 GB | larger sizes were not tested on GPU |
| **Jetson AGX Orin** | GPU CUDA (Ampere 2048) | 64 GB unified | **Qwen3-32B Dense** (19.8 GB) | 2.2 | 19.2 GB | full `-ngl 99` offload |
| **Jetson AGX Orin** | GPU CUDA (Ampere 2048) | 64 GB unified | **Qwen3.5-35B-A3B MoE** (21 GB) | **9.6** | 20.6 GB | full `-ngl 99` offload |
| **Jetson AGX Orin** | partial GPU (`-ngl 30`) | 64 GB unified | **Qwen3.5-122B-A10B MoE** (76.5 GB) | 1.9 | 44 GB GPU + 29 GB host | `-ngl 99` OOMs — 30/N layer split fits |

\* OP11 GPU was only sanity-tested at 1.7B / 4B in this sweep.

## Generation tok/s (no profiling overhead) — Qwen3 Dense Sweep

| Model | RPi4 CPU | OP11 CPU | OP11 GPU | OP13 CPU | OP13 GPU | Orin CPU | Orin GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 8.3 | 38.8 | — | — | — | — | — |
| Qwen3-1.7B | 3.4 | 11.4 | 16.0 | **23.4** | **23.4** | — | 19.5 |
| Qwen3-4B | **1.6** | 6.3 | 7.3 | **11.0** | **10.6** | — | — |
| Qwen3-8B | (OOM) | 3.2 | 4.1 | **6.8** | 5.8 | — | 7.5 |
| Qwen3-14B | (OOM) | **1.7** | (n/a) | 0.2† | (n/a) | — | 4.5 |
| Qwen3-32B Dense | (OOM) | (OOM) | (OOM) | (OOM) | (OOM) | — | **2.2** |
| Qwen3-30B-A3B MoE | (OOM) | (OOM) | (OOM) | (OOM) | (OOM) | — | **13.5** |
| Qwen3.5-27B Dense | (OOM) | 0.1‡ | (OOM) | (OOM) | (OOM) | — | 2.3 |
| Qwen3.5-35B-A3B MoE | (OOM) | (OOM) | (OOM) | (OOM) | (OOM) | — | **9.6** |
| Qwen3.5-122B-A10B MoE | — | — | — | — | — | 1.5 | 1.9 (`-ngl 30`) |

† swap-bound, technically loads but 0.2 t/s.
‡ mmap+swap, technically loads but 0.1 t/s.

## Cross-Family Sweep (this run added)

| Model | Params (active) | File | RPi4 CPU | OP11 CPU | OP13 CPU | Orin GPU | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Phi-3.5-mini | 3.8B | 2.4 GB | **1.7** | 8.2 | **13.7** | 12.3 | smallest non-Qwen tested |
| Llama-3.1-8B | 8B | 4.7 GB | (not pushed) | 5.3 | **9.4** | 8.0 | |
| Mistral-7B v0.3 | 7B | 4.4 GB | — | 9.6 | (not pushed) | 8.4 | classic 7B baseline |
| Gemma-2-9B | 9B | 5.4 GB | — | 4.1 | **6.9** | 6.4 | MQA architecture |

For comparison at similar size:
- **3.8-4B class**: Qwen3-4B 11.0 t/s vs Phi-3.5-mini 13.7 t/s on OP13 → **Phi slightly faster** at smaller params
- **7-9B class** on OP13: Qwen3-8B 6.8 / Llama-3.1-8B 9.4 / Gemma-2-9B 6.9 → **Llama-3.1 fastest**, Gemma-2 slower despite same params (MQA decode/prefill cost differ)
- **7-9B class** on Orin GPU: Qwen3-8B 7.5 / Llama-3.1-8B 8.0 / Mistral-7B 8.4 / Gemma-2-9B 6.4 → all close, Gemma slowest due to extra params

## Orin GPU Memory Breakdown (CUDA0 = 62,841 MiB / 61.4 GB)

Captured directly from `llama_memory_breakdown_print` per run.

| Model | File | `-ngl` | model | ctx | compute | **GPU total** | GPU % | Host | gen t/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 1.1 GB | 99 | 1,050 | 224 | 300 | **1,574 MiB / 1.5 GB** | 2.5% | — | 19.5 |
| Qwen3-8B | 4.7 GB | 99 | 4,403 | 256 | 258 | **4,917 MiB / 4.8 GB** | 7.8% | 281 | 7.5 |
| Llama-3.1-8B | 4.7 GB | 99 | 4,403 | 256 | 258 | **4,917 MiB / 4.8 GB** | 7.8% | 281 | 8.0 |
| Mistral-7B | 4.4 GB | 99 | 4,097 | 256 | 114 | **4,467 MiB / 4.4 GB** | 7.1% | 72 | 8.4 |
| Gemma-2-9B | 5.4 GB | 99 | 5,488 | 672 | 507 | **6,667 MiB / 6.5 GB** | 10.6% | 717 | 6.4 |
| Phi-3.5-mini | 2.4 GB | 99 | 2,228 | 768 | 84 | **3,081 MiB / 3.0 GB** | 4.9% | 52 | 12.3 |
| Qwen3-14B | 8.4 GB | 99 | 8,161 | 320 | 306 | **8,787 MiB / 8.6 GB** | 14.0% | — | 4.5 |
| Qwen3.5-27B | 16 GB | 99 | 15,272 | 277 | 505 | **16,054 MiB / 15.7 GB** | 25.5% | — | 2.3 |
| Qwen3-30B-A3B (MoE) | 18 GB | 99 | 17,524 | 192 | 300 | **18,016 MiB / 17.6 GB** | 28.7% | — | 13.5 |
| Qwen3-32B Dense | 19 GB | 99 | 18,423 | 512 | 306 | **19,241 MiB / 18.8 GB** | 30.6% | — | 2.2 |
| Qwen3.5-35B-A3B (MoE) | 21 GB | 99 | 20,470 | 102 | 493 | **21,065 MiB / 20.6 GB** | 33.5% | — | 9.6 |
| Qwen3.5-122B-A10B | 76.5 GB | **99** | 72,207 attempted | — | — | **OOM** | — | — | ❌ |
| Qwen3.5-122B-A10B | 76.5 GB | **30** | 43,854 | 94 | 632 | **44,580 MiB / 43.6 GB** | 71.0% | 29,223 | 1.9 |
| Qwen3.5-122B-A10B | 76.5 GB | **0** | 0 | 0 | 1,099 | **1,099 MiB / 1.1 GB** | 1.7% | 73,173 | 1.5 |

### Why partial offload is the escape valve for 122B-A10B

Tegra unified memory: CUDA "device memory" and host RAM physically share LPDDR5. With `-ngl 30` (~30/N layers on GPU), 44 GB lands in CUDA-allocator and 29 GB lands in host malloc — both still in the same physical RAM, so layer-to-layer transitions cost cache traffic, not PCIe. On a discrete-GPU system this trick is much weaker (PCIe round-trip per layer).

### ngl semantics, in brief

```
model layers (e.g. ~88 for 122B-A10B)
│
├─ -ngl 99  → all GPU   (74 GB needed → CUDA OOM at 62 GB cap)
├─ -ngl 30  → 30 GPU + (N-30) host  (44 GB GPU + 29 GB host, fits unified mem)
└─ -ngl 0   → all CPU + mmap         (73 GB host, swap-tolerant)
```

## cb_eval Profiling Overhead

| (device, model) | baseline gen t/s | cb_eval gen t/s | overhead × |
|---|---:|---:|---:|
| RPi4 / Qwen3-1.7B | 3.4 | 3.22 | 1.06× |
| RPi4 / Qwen3-4B | 1.6 | 1.41 | 1.13× |
| RPi4 / Phi-3.5-mini | 1.7 | 1.56 | 1.09× |
| OP11 CPU / Qwen3-1.7B | 11.4 | 8.81 | 1.29× |
| OP11 CPU / Phi-3.5-mini | 8.2 | 7.98 | 1.03× |
| OP11 CPU / Llama-3.1-8B | 5.3 | 3.05 | 1.74× |
| OP11 CPU / Mistral-7B | 9.6 | 2.69 | **3.6×** |
| OP11 GPU / Qwen3-1.7B | 16.0 | 1.42 | **11.3×** (OpenCL) |
| OP13 CPU / Qwen3-1.7B | 23.4 | 21.27 | 1.10× |
| OP13 CPU / Qwen3-4B | 11.0 | 8.46 | 1.30× |
| OP13 CPU / Phi-3.5-mini | 13.7 | 10.14 | 1.35× |
| OP13 CPU / Llama-3.1-8B | 9.4 | 3.61 | **2.6×** |
| Orin GPU / Qwen3-8B | 7.5 | 4.33 | 1.7× |
| Orin GPU / Qwen3-14B | 4.5 | 3.57 | 1.26× |
| Orin GPU / Qwen3-32B Dense | 2.2 | 1.49 | 1.48× |
| Orin GPU / Qwen3-30B-A3B MoE | 13.5 | 2.98 | **4.5×** |
| Orin GPU / Qwen3.5-27B Dense | 2.3 | 1.40 | 1.64× |
| Orin GPU / Qwen3.5-35B-A3B MoE | 9.6 | 2.62 | **3.7×** |
| Orin GPU / Llama-3.1-8B | 8.0 | 4.31 | 1.86× |
| Orin GPU / Phi-3.5-mini | 12.3 | 8.03 | 1.53× |
| Orin GPU / Mistral-7B | 8.4 | 4.58 | 1.83× |
| Orin GPU / Gemma-2-9B | 6.4 | 2.70 | **2.4×** |
| Orin partial-GPU / 122B-A10B | 1.9 | 0.55 | **3.5×** |

**Pattern**: dense CPU/CUDA 1.0-1.8×, MoE on CUDA 3.5-4.5× (per-expert sync amplifies cb_eval), OpenCL 11×.

## Per-Sublayer Decode Breakdown — selected entries

Full table in `data/parquet/qwen3_all.parquet` (260,678 records, 30 (device, model) combos). Figures: `claudedocs/figures/fig8_qwen3_sublayer_breakdown.png`, `fig9_qwen3_decode_latency.png`.

### FFN dominance trend (OP11 CPU, same device, just scaling Qwen3 dense)
| Model | FFN % | Attention % |
|---|---:|---:|
| Qwen3-1.7B | 51.9 | 18.9 |
| Qwen3-4B | 62.8 | 18.2 |
| Qwen3-8B | 70.1 | 15.2 |
| Qwen3-14B | **74.8** | 13.0 |

### Cross-family on OP11 CPU (similar param class)
| Model | FFN % | Attention % | Notes |
|---|---:|---:|---|
| Qwen3-8B | 70.1 | 15.2 | GQA |
| Llama-3.1-8B | 68.7 | 15.2 | GQA |
| Mistral-7B-v0.3 | 76.2 | 13.1 | GQA / Mistral arch |
| Gemma-2-9B | 65.7 | 15.0 | MQA + soft cap |
| Phi-3.5-mini | 60.2 | 2.7* | small attention share |

\* Phi-3.5's attention sublayer percentage drops because of its hidden-dim/attention-head ratio choices; the parser still classifies it correctly.

### Cross-family on OP13 CPU
| Model | FFN % | Attention % |
|---|---:|---:|
| Qwen3-8B | 63.5 | 19.6 |
| Llama-3.1-8B | 70.6 | 13.8 |
| Gemma-2-9B | 62.2 | 13.5 |
| Phi-3.5-mini | 56.2 | 4.1 |

### MoE vs dense on Orin GPU (same param class)
| Model | FFN % | Attention % | gen t/s |
|---|---:|---:|---:|
| Qwen3-32B Dense | 65.5 | 17.1 | 2.2 |
| Qwen3-30B-A3B MoE | 2.7* | 20.6 | **13.5** |
| Qwen3.5-27B Dense | 47.6 | 2.9† | 2.3 |
| Qwen3.5-35B-A3B MoE | 4.5* | 3.2† | **9.6** |

\* MoE FFN appears low because the parser doesn't yet attribute `mul_mat_id` (op 19 — expert routing) to FFN. The compute *is* mostly FFN-class; ~76-92% goes to "internal/unknown" until the parser is updated.
† Qwen3.5 renamed many tensor heads; the existing tensor_name_parser doesn't recognize them.

## Findings

### F1. FFN dominance scales with dense model size — confirmed and extended
Across the 1.7B → 14B Qwen3 dense sweep on OP11, FFN share grows monotonically (51.9 → 74.8%), consistent with the project's prior 1-2B finding. Cross-family (Llama, Mistral, Gemma) shows the same regime — FFN is 60-76% on dense ≥4B at this quantization.

### F2. MoE radically reshapes the per-layer profile and wins decisively at this scale
On Orin GPU, dense Qwen3-32B (2.2 t/s) vs MoE Qwen3-30B-A3B (13.5 t/s) — same parameter class, **6× speed-up** for MoE. The 122B-A10B test confirms it at the extreme: 76.5 GB MoE that physically does not fit in CUDA-only memory (`-ngl 99` OOMs trying to allocate 72 GB) still runs at 1.9 t/s with `-ngl 30` partial offload. Only 10B params fire per token — bandwidth, not compute, sets speed.

### F3. Partial GPU offload is the right escape valve for big MoE on unified memory
Three configs tested for 122B-A10B:
- `-ngl 99` → CUDA OOM (76 GB > 62 GB cap)
- `-ngl 30` → **1.9 t/s**, 44 GB GPU + 29 GB host, no swap thrashing
- `-ngl 0` → 1.5 t/s, mmap + 73 GB host

Fully reusable pattern on Tegra. On a discrete-GPU x86 the same trick collapses to PCIe-bound and is barely faster than CPU.

### F4. SD 8 Gen 3 (OP13) is roughly 2× SD 8 Gen 2 (OP11) on dense LLM workloads
- 1.7B: 11.4 → 23.4 t/s (+2.05×)
- 4B: 6.3 → 11.0 t/s (+1.75×)
- 8B: 3.2 → 6.8 t/s (+2.13×)
This holds across families: Llama-3.1-8B 5.3 → 9.4 (+1.77×), Gemma-2-9B 4.1 → 6.9 (+1.68×).
But OP13 has only **12 GB RAM (5.7 GB free)** vs OP11's 16 GB (10.3 GB free), so **OP13's max usable size is 8B**, while OP11 can run 14B (slowly).

### F5. Adreno 750 (OP13) shows weaker GPU lift over its CPU than Adreno 740 (OP11) shows over its CPU
- OP11: 1.7B CPU 11.4 → GPU 16.0 (+1.40×); 4B CPU 6.3 → GPU 7.3 (+1.16×)
- OP13: 1.7B CPU 23.4 → GPU 23.4 (no lift); 4B CPU 11.0 → GPU 10.6 (slightly slower)

Likely explanation: SD 8 Gen 3's CPU performance grew faster than its OpenCL throughput on small-batch LLM workloads. Full investigation would need bandwidth measurement.

### F6. Cross-family ranking by speed (similar params)
At ~7-9B class on Orin GPU: Mistral-7B 8.4 / Llama-3.1-8B 8.0 / Qwen3-8B 7.5 / Gemma-2-9B 6.4. Gemma-2-9B's 9 B params + soft-cap attention pay a real cost; Mistral's lean architecture is fastest.
At ~3.8-4B class: Phi-3.5-mini wins on every device (dramatic on OP13: 13.7 t/s).

### F7. RPi4 ceiling is 4B class, not 2B
Qwen3-4B (1.6 t/s) and Phi-3.5-mini (1.7 t/s) both run; both ~2.5-2.4 GB resident. 8B is impossible (4.7 GB > 3.4 GB free).

## Pipeline / Code Updates

- New script: `scripts/analyze_qwen3.py` — JSONL → ProfileRecord → SQLite/Parquet → sublayer summary (now covers 30 (device, model) entries).
- New script: `scripts/plot_qwen3.py` — generates `fig8_qwen3_sublayer_breakdown.png`, `fig9_qwen3_decode_latency.png`.
- New parquet: `data/parquet/qwen3_all.parquet` — **260,678 records** across all combinations.
- New raw datasets: `data/raw/{rpi4_qwen3,op11_cpu_qwen3,op13_qwen3,orin_qwen3}/*.jsonl`.

## Known Limitations / Follow-Ups

1. **`tensor_name_parser`** does not recognize Qwen3.5-renamed tensors and MoE `mul_mat_id` routing — sublayer classification accuracy degrades on those models. Raw timing/FLOP data is intact; only the categorical breakdown needs a parser update.
2. **OP11 GPU OpenCL** was tested only at 1.7B / 4B in this sweep; could push to 8B/14B for a fairer comparison vs OP13 GPU.
3. **Pi Zero 2W** not connected — Qwen3-0.6B (~400 MB) is the realistic candidate when it is.
4. **Llama-3.1-70B** Q4_K_M downloads at bartowski/lmstudio-community returned 404 across all attempted file-name patterns; 70B-class data point comes from Qwen3.5-35B-A3B and Qwen3-32B Dense instead.
5. **NPU (QNN)** binaries exist on both phones but are known broken; not exercised in this sweep.
6. **Profiling generation length** capped at `n=4-8` decode tokens to keep cb_eval runs tractable. Per-token decode latency is the relevant comparison.

## Reproduce

```bash
# Re-run analysis from collected JSONLs
python3 scripts/analyze_qwen3.py
python3 scripts/plot_qwen3.py
```

Models on Mac host (`models/`):
- Qwen3 family: 0.6B, 1.7B, 4B, 8B, 14B
- Qwen3.5: 27B
- Non-Qwen: Meta-Llama-3.1-8B-Instruct, gemma-2-9b-it, Phi-3.5-mini-instruct, Mistral-7B-Instruct-v0.3

Models on Orin (`~/models/`): all of the above + Qwen3-30B-A3B, Qwen3-32B, Qwen3.5-35B-A3B, Qwen3.5-122B-A10B (3-split).
Models on RPi4 (`~/tiny-llm-profiler/models/`): rotated due to disk constraints (15 GB SD card).
Models on OP11 (`/data/local/tmp/` on `db151d78`): Qwen3 0.6-14B, Qwen3.5-27B, Phi-3.5-mini, Llama-3.1-8B, Mistral-7B, Gemma-2-9B.
Models on OP13 (`/data/local/tmp/` on `85ea76a2`): Qwen3 1.7-14B, Phi-3.5-mini, Llama-3.1-8B, Gemma-2-9B.
