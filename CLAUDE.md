# tiny-llm-profiler — working context

Per-layer LLM inference profiling across edge devices. See `README.md` for what the
project is and what it measured. This file is the **working state**: what is true right
now, what is unfinished, and what bites you if you don't know it.

Last verified on-device: **2026-08-27** (llama.cpp rebuild + 35B chat confirmed).

## Repo state

- Local path: `~/workplace/research/cochl/tiny_llm` (MacBook, `conans-macbook-pro`)
- Remote: `git@github.com:ooshyun/llm_profiling.git`
- Current branch: `qwen3-max-size-sweep`, pushed, **not merged into `main`**

`main` still has only the 2026-04-06 commit. The entire 2026-04-28/29 Qwen3 sweep sat
untracked on disk for four months and was committed on 2026-08-27 onto this branch.
Decide whether to merge before adding more work on top.

Untracked-by-design: `models/*.gguf` (58 GB of the repo dir), `vendor/llama.cpp/`,
`vendor/ggml/`, `vendor/qairt/`, `.claude/`.

## Devices

| Alias | What | Reach it |
|---|---|---|
| Jetson AGX Orin 64GB | primary GPU target | **`ssh home.orin.ts`** (Tailscale `100.83.120.90`, key `id_rsa_cochl`) |
| Raspberry Pi 4 | CPU-only baseline | `home.rasp4.local` |
| OnePlus 13 | SD 8 Gen 3 | `adb:85ea76a2` |
| OnePlus 11 | SD 8 Gen 2, 16 GB | `adb:db151d78` |

Registered in `configs/devices.yaml`. **Use the `home.orin.ts` SSH-config alias.** The
other routes in `~/.ssh/config` are unreliable: `ssh cochl@home-orin` does not resolve
(the Tailscale MagicDNS name is not an SSH host), `home.orin.remote` (asuscomm:20000)
refuses the connection, and `home.orin.local` (192.168.50.197) only works on the LAN.
Hitting `100.83.120.90` directly fails auth unless you pass `-i ~/.ssh/id_rsa_cochl`.
`configs/devices.yaml` still says `home.orin.local` — same box, user `cochl`.

## Orin state as of 2026-08-27 — read this before running anything

Five things differ from what the April docs assume (item 1 is now fixed):

1. ~~The llama.cpp CUDA build is gone.~~ **FIXED 2026-08-27.** Rebuilt at
   `~/llama.cpp-build/build-cuda/` (upstream `ggml-org/llama.cpp` @ `6fdd0ac`,
   `GGML_CUDA=ON`, `CUDA_ARCHITECTURES=87`, `-j8`, ~18 min). `chat.sh` and
   `build_llama_orin.sh` — and their `scripts/` mirrors — now point at `$HOME`, so a
   reboot no longer wipes it. Only `llama-cli/-bench/-server/-quantize` were built; the
   CPU-fallback `build-cpu/` was skipped, so `~/llama.cpp-build/build-cpu/` does not exist.
   `chat.sh` now preflights both binary and GGUF and fails with a usable message.
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
5. **Only 9 GGUFs remain in `~/models/`, not 13.** Gone since April: `Qwen3-0.6B`,
   `Qwen3-32B`, `Qwen3.5-27B`, and — most costly — **`Qwen3-30B-A3B`, the fastest model
   ever measured on this box (13.5 tok/s)**. Their `chat.sh` keys still exist but now
   print `(GGUF MISSING)`. Re-download from `unsloth/` to reproduce those rows; 152 GB free.

Present and working: `~/chat.sh`, `~/build_llama_orin.sh`, `~/convert_to_gguf.sh`, and 9
GGUFs in `~/models/` including `Qwen3.5-35B-A3B-Q4_K_M.gguf` (22 GB). JetPack 6.0
(L4T R36.3.0), CUDA 12.2, sm_87, 61 GB unified RAM (58 free), 152 GB disk free.
`nvcc` is at `/usr/local/cuda/bin` and is **not** on the default `PATH`.

## Headline results worth not re-deriving

Orin GPU, Q4_K_M, `-ngl 99`, MODE_30W, April baseline `llama-cli -no-cnv -st -n 16`
(that flag no longer exists — see Known-broken):

| Model | gen tok/s | GPU resident |
|---|---:|---:|
| Qwen3-30B-A3B MoE | **13.5** | 17.6 GB |
| Qwen3.5-35B-A3B MoE | 9.6 → **10.8** on `6fdd0ac` | 20.6 GB |
| Qwen3.5-27B dense | 2.3 | 15.7 GB |
| Qwen3-32B dense | 2.2 | 18.8 GB |
| Qwen3.5-122B-A10B | 1.9 (`-ngl 30`) | 43.6 GB + 29 GB host |

Re-measured 2026-08-27 on the new build, same MODE_30W: 35B-A3B gives **pp64 79.1 t/s,
tg32 10.8 t/s** (`llama-bench`), and 10.9 t/s in real chat. The +13% over April is the
llama.cpp version bump alone, not a power-mode change — so **10.8 is the number to beat**,
not 9.6.

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
- **`-no-cnv` no longer exists.** Upstream split raw completion out of `llama-cli`
  (now conversation-only) into a separate `llama-completion` binary, which the current
  build does not include. Every April baseline command of the form
  `llama-cli ... -n 16 -no-cnv -st` is therefore dead, including the ones in
  `QUICKSTART.md`, `claudedocs/max_model_size_per_device_20260428.md`, and the profiling
  methodology. Use `llama-bench -m M -ngl 99 -p 64 -n 32 -r 2` for speed numbers;
  `chat.sh --bench` was rewritten to do exactly that. `-st` survives.
- **Qwen3.5 thinks by default.** `chat.sh qwen3.5-35b-a3b` emits a long reasoning trace
  even for trivial prompts. Add `--reasoning off` (or `--reasoning-budget 0`) for
  latency-sensitive use — and be aware that April's tok/s numbers did not have this cost.
- **QNN NPU is a dead end** with llama.cpp's experimental backend — `supports_buft()`
  rejects CPU-mmap'd GGUF buffers. Full analysis in
  `claudedocs/npu_troubleshooting_report_20260407.md`.

## Next up

1. ~~Move the llama.cpp build out of `/tmp`, rebuild, confirm `./chat.sh qwen3.5-35b-a3b`.~~
   **Done 2026-08-27.**
2. Re-baseline on `6fdd0ac`. Two confounds now sit on top of April's table at once — the
   llama.cpp version bump (worth +13% on 35B-A3B) and MODE_30W vs MAXN. Sweep the 9
   surviving GGUFs with `llama-bench` at 30W first, *then* flip MAXN
   (`sudo nvpmodel -m 0 && sudo jetson_clocks`) and repeat, so the two effects stay separable.
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
