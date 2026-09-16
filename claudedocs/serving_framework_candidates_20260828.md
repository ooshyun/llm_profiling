# Serving Framework Candidates — Jetson AGX Orin 64GB

**Date**: 2026-08-28
**Target**: `home.orin.local` — Jetson AGX Orin 64GB, JetPack 6.0 (L4T R36.3.0), CUDA 12.2, Ampere sm_87 (2048 CUDA cores), 61 GB unified RAM, 152 GB free disk
**Goal**: pick a serving stack for interactive chat and agent workloads, with Qwen3.5-35B-A3B (MoE, 22 GB Q4_K_M) as the reference model

## Current state (last verified 2026-09-15 — see updates below; original 2026-08-28 rows kept where still accurate)

| Item | State |
|---|---|
| Reach the box | **`ssh home.orin.ts`** (Tailscale alias) — `home.orin.local` below only works on the LAN; see `CLAUDE.md` Devices |
| Serving process running | none (llama-server stopped after each Phase 0 run; verified 2026-09-15) |
| llama.cpp CUDA binaries | **rebuilt 2026-08-27**, no longer in `/tmp`: `~/llama.cpp-build/build-cuda/` (upstream `ggml-org/llama.cpp` @ `6fdd0ac`, `GGML_CUDA=ON`, `sm_87`). `llama-server` (OpenAI-compatible HTTP) confirmed working — see the Phase 0 measurements below |
| `~/chat.sh`, `~/build_llama_orin.sh`, `~/convert_to_gguf.sh` | present; also `~/serving_bench/` (harness copy, rsynced from `scripts/serving_bench/`) |
| GGUF models in `~/models/` | **9 files**, not 13 — `Qwen3-0.6B`, `Qwen3-32B`, `Qwen3.5-27B`, `Qwen3-30B-A3B` were deleted 2026-04-29 to free disk (see `CLAUDE.md` item 5). `Qwen3.5-35B-A3B-Q4_K_M.gguf` (22 GB) still present and is the Phase 0 35B model |
| PyTorch (base OS) | still `2.6.0+cpu` — CPU-only. Not used by Phase 0 (llama.cpp only); the staged vLLM/SGLang Docker images below bundle their own CUDA-enabled aarch64 PyTorch |
| Power mode | `MODE_30W` (mode 2) — not MAXN. All Phase 0 measurements are 30W numbers |
| Docker images | Phase-2 assets staged 2026-09-15: `mitakad/vllm:0.22.0-r36.5.tegra-aarch64-cp312-cu129-24.04` (44.7 GB), `mitakad/sglang:0.6.0-r36.5.tegra-aarch64-cp312-cu129-24.04-at-commit-d093e70` (31.6 GB). Also present: `~/hf/` HF snapshots (`Qwen/Qwen3-8B` bf16, `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`) for Phase 2. Older riva-speech/senseruntime-tvm/sense-sdk-tensorrt images (exited, unrelated to LLM) also still present |
| Disk free | **42 GB free (91% used)**, down from 152 GB free on 2026-08-28 — mostly consumed by the Phase-2 images/models above. Still enough headroom for a JetPack upgrade (spec wants ≥20 GB) but not for re-downloading `Qwen3-30B-A3B` or `Qwen3.5-122B-A10B` without cleanup |

So the effective answer to "what is the current serving framework" as of 2026-09-15 is:
**llama.cpp CUDA via `llama-server` (OpenAI-compatible HTTP), measured end-to-end on real chat/agent-loop/concurrency workloads in Phase 0.** Full results: `claudedocs/serving_framework_eval_20260915.md`.

## Candidates

### 1. llama.cpp — incumbent

- **What it is**: GGUF inference in C++/CUDA. `llama-cli` for interactive, `llama-server` for an OpenAI-compatible HTTP endpoint.
- **Fit**: proven on this exact box — every number in `max_model_size_per_device_20260428.md` came from it. Builds against CUDA 12.2 / sm_87 with `~/build_llama_orin.sh`. `-ngl` partial offload is what made 122B-A10B run at all.
- **Strengths**: lowest setup cost, widest quant support (Q4_K_M through Q2_K), runs fully offline, partial-offload escape valve for models past the CUDA cap.
- **Weaknesses**: single-stream oriented — no PagedAttention, no prefix-cache reuse across turns, so multi-turn agent loops re-prefill. MoE decode is bandwidth-bound with no expert caching. Reported to be far slower on TTFT for large MoE than purpose-built engines.
- **Verdict**: keep as the baseline and fallback. Rebuild first, then measure everything else against it.

### 2. FreeToken — strongest fit for MoE on this box

- **What it is**: edge-native MoE serving engine (UC Berkeley / FlashML, arXiv 2608.16157, Aug 2026). Bandwidth-adaptive CPU–GPU co-execution, full-layer double-buffered prefill streaming, global LRU expert cache, FTW fast weight format, graph-compatible execution.
- **Fit**: this is built for exactly our situation — a large MoE whose full weights don't comfortably sit in GPU memory, on a machine where CPU and GPU share bandwidth. Orin's unified LPDDR5 is the friendliest possible case for its CPU–GPU co-execution design, more so than the discrete-GPU desktops it targets.
- **Strengths**: reported first-token latency far below llama.cpp on large MoE (sub-44s vs 232s in their comparison). Exposes OpenAI- *and* Anthropic-compatible APIs, so Claude Code / Codex can point straight at it. Directly addresses the 122B-A10B case where `-ngl 30` currently gives 1.9 t/s.
- **Weaknesses / unknowns**: very new; their published numbers are on consumer x86 + discrete GPU, not Tegra. FTW format means a conversion step from our existing GGUFs.
- **Spike verdict (2026-09-15, `claudedocs/serving_framework_eval_20260915.md`)**: **not revisitable on this Orin, on any JetPack version.** A 2-hour-capped install spike (`~/ft-spike`, full logs in `results/serving_bench/freetoken_spike/`) found FreeToken's build hook `check_nvcc_matches_torch` aborts because the required `torch>=2.11,<2.12` wheel is linked against CUDA 13.0 while the Orin's toolchain is CUDA 12.2 (`RuntimeError: nvcc 12.2 would build kernels linking libcudart.so.12, but torch 2.11.0+cu130 ships CUDA 13.0`). An aarch64 wheel for that torch pin *does* exist (H1, "no aarch64 wheel", is refuted) — but even installed standalone, `torch.cuda.is_available()` returns `False` because the JetPack 6.0 driver (12.2) is too old to run CUDA-13-linked code. JetPack 6.x on Orin tops out at CUDA 12.6 (6.1/6.2) — no JetPack release for Orin ever reaches CUDA 13, which requires JetPack 7 / Thor-class hardware. So the blocker is not "unverified aarch64 support," it's a hard toolchain ceiling: **only a different piece of hardware (Thor) could clear it**, and even then the FP8/NVFP4 kernel-support question (H3) is untested. FreeToken is therefore removed from the actionable candidate list below and kept only as a design reference.
- **Verdict**: **eliminated by the spike above.** Its design premise (CPU–GPU co-execution on unified memory) still matches Tegra well, but the implementation is pinned to a CUDA 13 / x86 stack this Orin cannot reach.

### 3. vLLM — best if we want a real multi-user server

- **What it is**: PagedAttention + continuous batching, OpenAI-compatible server. The reference production stack.
- **Fit**: the original 2026-04-06 research report already recommended it for this device (DeepSeek-R1-Distill-Qwen-7B W4A16 at ~180 t/s is NVIDIA's own vLLM number on AGX Orin).
- **Strengths**: by far the best throughput under concurrency; mature MoE support; W4A16/AWQ/GPTQ; huge ecosystem.
- **Weaknesses**: heavyweight on Jetson. Needs a CUDA-enabled aarch64 PyTorch — the box currently has `2.6.0+cpu`, so this is a prerequisite, not a detail. Practical path is `jetson-containers` rather than a from-source build. Weakest exactly where we care most: single-user interactive chat, where continuous batching buys nothing.
- **Verdict**: right answer *if* the goal shifts to serving multiple clients or an agent fleet. Overkill for one chat session.

### 4. SGLang — best if the workload is agentic/structured

- **What it is**: RadixAttention (prefix-tree KV reuse across requests), structured/constrained decoding, OpenAI-compatible server.
- **Fit**: the win is prefix reuse. For an agent loop that resends a long, mostly-identical system prompt every turn, RadixAttention removes the re-prefill that llama.cpp pays in full — which matters more on a 2 t/s dense model than raw decode speed does.
- **Strengths**: excellent for repeated-prefix and structured-output workloads; competitive decode.
- **Weaknesses**: same CUDA-PyTorch prerequisite as vLLM. aarch64/Jetson support is **less exercised than vLLM's** — expect more friction, verify before planning around it.
- **Verdict**: revisit once there is a concrete agent workload with long shared prefixes. Not the first move.

## Comparison

| | llama.cpp | FreeToken | vLLM | SGLang |
|---|---|---|---|---|
| Verified on this box | **yes** | no | no | no |
| Setup cost | low (script exists) | unknown | high | high |
| Needs CUDA PyTorch | no | unknown | **yes** | **yes** |
| Model format | GGUF (have it) | FTW (convert) | HF safetensors / AWQ | HF safetensors / AWQ |
| MoE expert caching | no | **yes (LRU)** | partial | partial |
| Runs past GPU mem cap | yes (`-ngl` split) | **yes (co-exec)** | no | no |
| Prefix reuse across turns | no | agentic state reuse | prefix caching | **yes (RadixAttention)** |
| Concurrency | weak | unknown | **best** | strong |
| OpenAI-compatible API | yes (`llama-server`) | yes (+ Anthropic) | yes | yes |

## Recommended order

1. **Rebuild llama.cpp** (`~/build_llama_orin.sh`) — restores the known-good baseline and unblocks chat today. Move the build out of `/tmp` so a reboot stops destroying it.
2. **Switch to MAXN** (`sudo nvpmodel -m 0 && sudo jetson_clocks`) — the entire April sweep ran at MODE_30W. Every number in this repo is a 30W number; expect roughly 2× headroom unmeasured.
3. ~~Attempt a FreeToken aarch64 build.~~ **Done 2026-09-15 — eliminated.** Spike confirmed a hard CUDA 13 toolchain requirement no JetPack release for Orin can satisfy (see verdict above). Not revisitable without different hardware (Thor).
4. **vLLM or SGLang only when the workload justifies it** — vLLM for concurrency, SGLang for long shared prefixes. Both are gated behind replacing the CPU-only PyTorch.

## Open questions

- ~~Does FreeToken build on aarch64 / JetPack 6.0 at all?~~ **Answered 2026-09-15**: no — it requires a CUDA 13 toolchain (`nvcc` + driver) that no JetPack release for Orin provides; see the spike verdict above.
- Is there a CUDA-enabled PyTorch wheel for this JetPack 6.0 / CUDA 12.2 / Python combination, or does vLLM/SGLang require the container route?
- Re-measure the llama.cpp baseline at MAXN before comparing any new engine against the April numbers, otherwise the comparison is confounded by power mode.

## Sources

- [Phase 0 evaluation results — llama.cpp baseline + FreeToken spike (2026-09-15)](serving_framework_eval_20260915.md)
- [FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution (arXiv 2608.16157)](https://arxiv.org/html/2608.16157v1)
- [FlashML-org/FreeToken (GitHub)](https://github.com/FlashML-org/FreeToken)
- [FreeToken paper page (HuggingFace)](https://huggingface.co/papers/2608.16157)
- [Max-size measurement results](max_model_size_per_device_20260428.md)
- [Orin chat guide](orin_chat_guide.md)
