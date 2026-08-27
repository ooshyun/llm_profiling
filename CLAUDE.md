# tiny-llm-profiler — working context

Per-layer LLM inference profiling across edge devices. See `README.md` for what the
project is and what it measured. This file is the **working state**: what is true right
now, what is unfinished, and what bites you if you don't know it.

Last verified on-device: **2026-08-28**.

## Repo state

- Local path: `~/workplace/research/cochl/tiny_llm` (MacBook, `conans-macbook-pro`)
- Remote: `git@github.com:ooshyun/llm_profiling.git`
- Current branch: `qwen3-max-size-sweep`, pushed, **not merged into `main`**

`main` still has only the 2026-04-06 commit. The entire 2026-04-28/29 Qwen3 sweep sat
untracked on disk for four months and was committed on 2026-08-28 onto this branch.
Decide whether to merge before adding more work on top.

Untracked-by-design: `models/*.gguf` (58 GB of the repo dir), `vendor/llama.cpp/`,
`vendor/ggml/`, `vendor/qairt/`, `.claude/`.

## Devices

| Alias | What | Reach it |
|---|---|---|
| Jetson AGX Orin 64GB | primary GPU target | `ssh cochl@home-orin` (Tailscale `100.83.120.90`), or `ssh home.orin.remote`, or `home.orin.local` on LAN |
| Raspberry Pi 4 | CPU-only baseline | `home.rasp4.local` |
| OnePlus 13 | SD 8 Gen 3 | `adb:85ea76a2` |
| OnePlus 11 | SD 8 Gen 2, 16 GB | `adb:db151d78` |

Registered in `configs/devices.yaml`. Note the Orin's Tailscale name is `home-orin` but
the configs use `home.orin.local` — both reach the same box, user `cochl`.

## Orin state as of 2026-08-28 — read this before running anything

Four things are not what the April docs assume:

1. **The llama.cpp CUDA build is gone.** It lived in `/tmp/llama.cpp-build/`, and the box
   has rebooted since. `~/chat.sh` points at `/tmp/llama.cpp-build/build-cuda/bin/llama-cli`,
   which does not exist. Nothing runs until `~/build_llama_orin.sh` is re-run (20-40 min).
   **This will keep happening** — moving the build to `~/llama.cpp-build` and updating the
   path in both `chat.sh` and `build_llama_orin.sh` (and their mirrors in `scripts/`) is
   the first real fix.
2. **Power mode is `MODE_30W`, not MAXN.** Every number in
   `claudedocs/max_model_size_per_device_20260428.md` and in `chat.sh`'s speed table is a
   30W number. Before benchmarking any new engine against them, either re-measure at 30W
   or re-baseline at MAXN (`sudo nvpmodel -m 0 && sudo jetson_clocks`) — otherwise the
   comparison is confounded by power mode, not by the engine.
3. **PyTorch on the Orin is `2.6.0+cpu`** — the CPU-only wheel. vLLM and SGLang both need
   CUDA-enabled aarch64 PyTorch, so that swap (or the `jetson-containers` route) is a
   prerequisite, not a detail.
4. **`Qwen3.5-122B-A10B` was deleted** on 2026-04-29 to free disk for the 35B-A3B
   conversion. Its profiling JSONL is committed, the weights are not. Re-download from
   `unsloth/Qwen3.5-122B-A10B-GGUF` (~76 GB; 152 GB free) to reproduce.

Present and working: `~/chat.sh`, `~/build_llama_orin.sh`, `~/convert_to_gguf.sh`, and 13
GGUFs in `~/models/` including `Qwen3.5-35B-A3B-Q4_K_M.gguf` (22 GB). JetPack 6.0
(L4T R36.3.0), CUDA 12.2, sm_87, 61 GB unified RAM.

## Headline results worth not re-deriving

Orin GPU, Q4_K_M, `-ngl 99`, MODE_30W, baseline `llama-cli -no-cnv -st -n 16`:

| Model | gen tok/s | GPU resident |
|---|---:|---:|
| Qwen3-30B-A3B MoE | **13.5** | 17.6 GB |
| Qwen3.5-35B-A3B MoE | **9.6** | 20.6 GB |
| Qwen3.5-27B dense | 2.3 | 15.7 GB |
| Qwen3-32B dense | 2.2 | 18.8 GB |
| Qwen3.5-122B-A10B | 1.9 (`-ngl 30`) | 43.6 GB + 29 GB host |

**MoE beats dense ~6× at equal parameter count** — only 3B params fire per token, so
decode is bandwidth-bound. And Tegra unified memory makes `-ngl` partial offload a real
escape valve past the 62 GB CUDA cap, which does *not* transfer to discrete-GPU systems.

## Known-broken / open

- **MoE sublayer attribution is wrong.** The tensor parser doesn't map `mul_mat_id`
  (op 19, expert routing) to FFN, so MoE runs report 76-92% "internal/unknown" and FFN
  shows as 2-4%. The dense numbers are trustworthy; the MoE breakdown is not. Fixing this
  is the highest-value analysis work left. See `collection/track1_ggml/tensor_name_parser.py`.
- **RWKV parser** has the same class of gap — 89% "internal".
- **cb_eval overhead is not uniform**: dense CUDA 1.0-1.8×, MoE on CUDA 3.5-4.5×
  (per-expert sync amplifies it), OpenCL 11×. Never compare cb_eval speeds across
  backends as if they were inference speeds.
- **QNN NPU is a dead end** with llama.cpp's experimental backend — `supports_buft()`
  rejects CPU-mmap'd GGUF buffers. Full analysis in
  `claudedocs/npu_troubleshooting_report_20260407.md`.

## Next up

1. Move the llama.cpp build out of `/tmp`, rebuild, confirm `./chat.sh qwen3.5-35b-a3b`.
2. MAXN + re-baseline, so April's 30W numbers stop being the silent reference.
3. Try a FreeToken aarch64 build — best-fit candidate for the MoE case, aarch64 support
   unverified. Rationale and comparison in
   `claudedocs/serving_framework_candidates_20260828.md`.
4. Fix `mul_mat_id` attribution, then regenerate fig8/fig9 and the MoE tables.

## Doc index

| Doc | What it holds |
|---|---|
| `claudedocs/max_model_size_per_device_20260428.md` | per-device size ceilings, full tok/s sweep, Orin memory breakdown, cb_eval overhead table |
| `claudedocs/serving_framework_candidates_20260828.md` | llama.cpp / FreeToken / vLLM / SGLang evaluation + verified device state |
| `claudedocs/orin_chat_guide.md` | `chat.sh` keys, per-model commands, memory notes |
| `claudedocs/gguf_workflow.md` | safetensors → GGUF conversion |
| `claudedocs/vendor_stack_architecture_20260429.md` | ggml / llama.cpp internals reference |
| `claudedocs/edge_llm_research_report.md` | 2026-04 device/model/framework survey (pre-dates the measurements — treat its Orin numbers as estimates) |
| `claudedocs/npu_troubleshooting_report_20260407.md` | why QNN NPU fails |
| `docs/superpowers/` | original design specs and phase plans |
