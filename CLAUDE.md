# tiny-llm-profiler — working context

Per-layer LLM inference profiling across edge devices. See `README.md` for what the
project is and what it measured. This file is the **working state**: what is true right
now, what is unfinished, and what bites you if you don't know it.

Last verified on-device: **2026-09-20** — JetPack upgraded to 6.2.3 and the
llama.cpp-vs-vLLM comparison completed. Phase 0 (llama.cpp @ CUDA 12.2 + FreeToken
spike): `claudedocs/serving_framework_eval_20260915.md`. Phase 1/2 (the upgrade and
the engine comparison): `claudedocs/serving_framework_eval_phase2_20260920.md`.

## Repo state

- Local path: `~/workplace/research/cochl/tiny_llm` (MacBook, `conans-macbook-pro`)
- Remote: `git@github.com:ooshyun/llm_profiling.git`
- Current branch: `main` — `qwen3-max-size-sweep` was merged via PR #1 on 2026-09-16
  (merge commit `1b6770e`); the branch still exists locally and on origin, safe to delete.

History note: `main` sat at the single 2026-04-06 commit until 2026-09-16. The 2026-04-28/29
Qwen3 sweep was untracked on disk for four months, committed onto `qwen3-max-size-sweep`
on 2026-08-27, and landed on `main` together with the serving-framework Phase 0 work.

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

## Orin state as of 2026-09-20 — read this before running anything

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
3. ~~PyTorch on the Orin is `2.6.0+cpu`.~~ **Moot as of 2026-09-20** — vLLM runs in the
   staged container (`mitakad/vllm:0.22.0-r36.5...`), which ships its own torch 2.11 and
   reports `cuda True` on sm_87. The host's CPU-only wheel is irrelevant to the serving work.
4. **`Qwen3.5-122B-A10B` was deleted** on 2026-04-29 to free disk for the 35B-A3B
   conversion. Its profiling JSONL is committed, the weights are not. Re-download from
   `unsloth/Qwen3.5-122B-A10B-GGUF` (~76 GB; 108 GB free as of 2026-09-16, see below) to
   reproduce.
5. **Only 2 GGUFs remain in `~/models/`, not 13.** `Qwen3.5-35B-A3B-Q4_K_M.gguf` (the
   serving-eval subject) and `Qwen3-8B-Q4_K_M.gguf` (its dense control) are the only ones
   left. Gone since April: `Qwen3-0.6B`, `Qwen3-32B`, `Qwen3.5-27B`,
   `Qwen3-30B-A3B` (the fastest model ever measured on this box, 13.5 tok/s). Gone as of
   **2026-09-16 cleanup** (deliberate, user-approved — sweep re-run was not planned):
   `Qwen3-14B`, `gemma-2-9b`, `Llama-3.1-8B`, `Mistral-7B`, `Qwen3-4B`, `Phi-3.5-mini`,
   `Qwen3-1.7B` (27 GB). Their profiling JSONL is committed under `data/raw/orin_qwen3/`
   (md5-verified against the on-disk copies before deletion) and unaffected. All of these
   `chat.sh` keys now print `(GGUF MISSING)`. Re-download from `unsloth/` to reproduce
   any row; item 2's re-baseline needs the full 9-model set back first.

Present and working: `~/chat.sh`, `~/build_llama_orin.sh`, `~/convert_to_gguf.sh`
(re-synced 2026-09-16 — it had drifted to the old `/tmp/llama.cpp-build` default; now
matches `scripts/convert_to_gguf.sh`), `~/rebuild_llama.sh` (mirrored into the repo as
`scripts/rebuild_llama_orin.sh` on 2026-09-16 — this is the fast CUDA-only rebuild
script that actually produced the `6fdd0ac` build, distinct from the slower
`build_llama_orin.sh` which also does a CPU-fallback build), and the 2 GGUFs above.
JetPack 6.0 (L4T R36.3.0), CUDA 12.2, sm_87, 61 GB unified RAM (58 free). `nvcc` is at
`/usr/local/cuda/bin` and is **not** on the default `PATH`.

Also present, added 2026-09-15 for the serving-framework eval: `~/serving_bench/` (rsynced
copy of `scripts/serving_bench/` — `bench.py`, `scenarios.yaml`, `engines/llama_server.sh`,
`monitor.sh`; results synced back to `results/serving_bench/` in this repo, summarized in
`claudedocs/serving_framework_eval_20260915.md`), staged Phase-2 assets (`~/hf/` HF
snapshots for `Qwen/Qwen3-8B` and `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`, 39 GB, Docker images
`mitakad/vllm:0.22.0-r36.5.tegra-aarch64-cp312-cu129-24.04` and
`mitakad/sglang:0.6.0-r36.5.tegra-aarch64-cp312-cu129-24.04-at-commit-d093e70`).
`~/ft-spike/` (the FreeToken spike venv) was **deleted 2026-09-16** — FreeToken is
permanently eliminated on this box (see item 3 under Next up), its logs are committed
under `results/serving_bench/freetoken_spike/`, nothing further needed it.

**Disk cleanup 2026-09-16: 37 GB → 108 GB free** (76% used). Removed, all md5-verified
against committed copies or confirmed unreferenced first: `~/conversion/gguf/*.gguf`
(25 GB — safetensors→GGUF workflow-test artifacts from 2026-04-29/30, superseded by the
community GGUFs in `~/models/`, procedure reproducible via `claudedocs/gguf_workflow.md`),
`~/ft-spike/` (4.4 GB), the 7 sweep-only GGUFs above (27 GB), 12 stray profiling JSONLs
in `~/models/` (all md5-matched `data/raw/orin_qwen3/`), an empty
`Meta-Llama-3.1-70B-Instruct-Q4_K_M/` dir (leftover from an April 404), stale HF
download-cache metadata, `.bak` files, `rebuild_llama.log`, the `hello-world` Docker
image, and 46 GB of Docker build cache (`docker builder prune -a` — images/containers
untouched). **Not touched in this pass**: the `riva-speech:2.19.0`/`nemo` Docker images
and `~/riva_*`/`ngc*` dirs (kept as demo assets), `~/workspace/` (Cochl SDK/ASR work,
outside this project's scope), `~/hf/` (Phase 2), `~/serving_bench/`. Prior to cleanup,
disk had fallen to 42 GB free

**Follow-up cleanup 2026-09-19: 108 GB → 144 GB free** (67% used). User-directed removal
of three more Docker images (37 GB): `sense-sdk/tensorrt-jetpack6.0.0:1.6.0-beta`
(19.1 GB) and `senseruntime-tvm:latest` (4.36 GB) — Cochl SDK images, previously left
alone as out-of-project-scope, now explicitly authorized — plus their exited/never-run
containers (`sense-sdk-tensorrt-1.6.0-beta`, `exciting_jang`, `senseruntime-tvm-container`);
and `nvcr.io/nvidia/riva/riva-speech:2.19.0-l4t-aarch64` (34.4 GB nominal, ~21.8 GB unique
after shared layers) — an unused duplicate tag with zero containers, distinct from the
`riva-speech:2.19.0` tag that IS in use and stays. `~/hf/` and `~/models/` were audited
in detail (blob/snapshot structure, checked for incomplete downloads, format duplicates,
orphaned blobs) and found to hold **no deletable candidates** — every file in both is
live-referenced by the two models actually needed for Phase 0/2. A related finding:
`~/.cache/uv/` (8.9 GB) holds the FreeToken spike's torch-CUDA-13 package cache,
orphaned now that `~/ft-spike/` is gone — flagged, not yet removed pending confirmation.
(91% used) per a `df` reading measured 2026-09-15 before the Phase-2 staging described
above pushed it that low from a 151 GB baseline (see
`claudedocs/serving_framework_eval_20260915.md` and
`claudedocs/serving_framework_candidates_20260828.md` for that figure).

**Currently serving** (since 2026-09-21): **vLLM with prefix caching forced on**,
`https://vllm1.ooshyun.cc` → cloudflared → `127.0.0.1:8000`. Restart it with:

```bash
cd ~/serving_bench && ./engines/vllm.sh 35b 0.75 --enable-prefix-caching \
  --served-model-name qwen3.5-35b-a3b /home/cochl/models/Qwen3.5-35B-A3B-Q4_K_M.gguf
```

Takes ~13 min to answer (weights 65 s + `torch.compile` 209 s + graph capture +
warmup). The second `--served-model-name` value is **an alias only** — the weights
are GPTQ-Int4; it exists because vLLM rejects unknown model names and the previous
llama-server advertised that GGUF path. To go back to llama.cpp instead:

```bash
docker rm -f vllm_bench
~/llama.cpp-build/build-cuda/bin/llama-server -m ~/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -ngl 99 -c 8192 -t 8 --host 127.0.0.1 --port 8000 --reasoning off -np 1
```

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

**llama.cpp vs vLLM on 35B-A3B** (2026-09-20, JetPack 6.2.3, MODE_30W — full detail in
`claudedocs/serving_framework_eval_phase2_20260920.md`):

| | llama.cpp Q4_K_M | vLLM 0.22 GPTQ-Int4 |
|---|---:|---:|
| decode | 10.8 tok/s | **13.6 tok/s** |
| repeated 4k prefix (S2) | **31.4×** faster | 1.0× by default; **3.0× with `--enable-prefix-caching`** |
| warm TTFT in an agent loop | **0.92 s** | 29.2 s → **9.6 s** with that flag |
| c=8 aggregate / TTFT p50 | 29.5 tok/s / **1.0 s** | **35.4 tok/s** / 16.9 s |

**vLLM decodes faster; llama.cpp answers faster.** The prefix result is the one that
decides most workloads: **Qwen3.5-35B-A3B is a hybrid model** (GDN linear attention +
mamba state), and **vLLM disables prefix caching for it automatically** — its startup
config logs `enable_prefix_caching=False` for the 35B vs `True` for the 8B, and the 35B's
hit rate stays 0.0% for the whole run. Not our misconfiguration; the engine's own decision.
The trigger is `attn_type == "hybrid"` in `ModelConfig.is_prefix_caching_supported`
(`vllm/config/model.py`) — **quantization is never consulted**, so GPTQ/AWQ/bf16 all
behave the same.

**Only the default is off.** Passing `--enable-prefix-caching` explicitly survives the
auto-disable (it is guarded by `if self.enable_prefix_caching is None`) and activates
vLLM's mamba `align` cache mode, which **works and does not change the output** (cold
prefill and cache-hit responses are byte-identical within a server). It is worth turning
on: warm TTFT 29.25 s → 9.61 s (observed cache hit rate ~73%). But it is **3.0×, not
llama.cpp's 31.8×**, because prefix reuse is block-granular and the engine floors the
attention block at 1056 tokens, so a 4.1k prompt re-prefills a ~970-token tail every
request. **Cold prefill gets 27% slower** (29.05 s → 36.95 s on an unseen prefix), so the
flag pays off from the second request onward and is a loss only if prefixes never repeat. `--mamba-block-size 512` does
not help — the block stays 1056 and the timings are unchanged. Detail and raw
data in `claudedocs/serving_framework_eval_phase2_20260920.md`.

**Scope**: this is a Qwen3.5-35B-A3B property, not a vLLM one. The user reports
`QuantTrio/Qwen3.6-35B-A3B-AWQ` caching normally, i.e. Qwen3.6 is not classified
`hybrid`.
llama.cpp's per-slot retention is indifferent to that and still gets 31×. On the
pure-attention 8B, vLLM's caching does work (27.9×) — confirming the cause is the
architecture, not our configuration. SGLang's RadixAttention is the same
content-addressed family, so expect the same limitation until measured.

⚠️ The engines run different weights (vLLM cannot load MoE GGUF): 8B is Q4_K_M vs
**bf16**, so its decode gap is mostly memory traffic, not engine. The 35B pair
(22 GB vs 21 GB) is the fair one.

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
   llama.cpp version bump (worth +13% on 35B-A3B) and MODE_30W vs MAXN. **7 of the 9
   original GGUFs were deleted in the 2026-09-16 disk cleanup** (not planned to be
   re-run at the time); re-download them first if pursuing this. Sweep with
   `llama-bench` at 30W first, *then* flip MAXN (`sudo nvpmodel -m 0 && sudo jetson_clocks`)
   and repeat, so the two effects stay separable.
3. ~~Try a FreeToken aarch64 build.~~ **Done 2026-09-15 — eliminated.** Spike (`~/ft-spike`,
   2h cap, took ~3 min) hit a hard CUDA 13 toolchain requirement: `nvcc` 12.2 vs
   `torch==2.11.0+cu130`'s CUDA 13, and even a standalone CUDA-13 torch install reports
   `cuda available: False` against the JetPack 6.0 driver. No JetPack release for Orin
   reaches CUDA 13 (6.x tops out at 12.6; CUDA 13 = JetPack 7 = Thor-class hardware only),
   so FreeToken is not revisitable on this box at all. Full verdict in
   `claudedocs/serving_framework_eval_20260915.md`; design analysis kept in
   `claudedocs/serving_framework_candidates_20260828.md`.
4. Fix `mul_mat_id` attribution, then regenerate fig8/fig9 and the MoE tables.
5. ~~Phase 1 (JetPack upgrade) and Phase 2 (vLLM/SGLang).~~ **Phase 1 + vLLM done
   2026-09-20** — see `claudedocs/serving_framework_eval_phase2_20260920.md`.
   **SGLang is the remaining gap**: its image is staged (31.6 GB) but untested.
   Worth doing specifically to check whether RadixAttention hits the same
   hybrid-model wall vLLM did — if it does, llama.cpp is the only engine on this
   box with working prefix reuse for Qwen3.5-35B-A3B, which is a strong claim
   that deserves a second data point.
6. Consider `Qwen3.6-35B-A3B`: it reuses the `qwen3_5_moe` architecture, so the
   current llama.cpp build loads it **without a rebuild** — only the GGUF download
   (~22 GB, 108 GB free). Its MTP variant plus the `--spec-draft-*` support already
   in this build could add 1.5–2× decode.

## Doc index

| Doc | What it holds |
|---|---|
| `claudedocs/max_model_size_per_device_20260428.md` | per-device size ceilings, full tok/s sweep, Orin memory breakdown, cb_eval overhead table |
| `claudedocs/serving_framework_candidates_20260828.md` | llama.cpp / FreeToken / vLLM / SGLang evaluation + verified device state |
| `claudedocs/serving_framework_eval_phase2_20260920.md` | **Phase 1/2**: JetPack 6.2.3 upgrade + llama.cpp vs vLLM on 8B and 35B-A3B; the hybrid-model prefix-caching finding; corrected unified-memory guidance |
| `claudedocs/qwen35_35b_a3b_quant_comparison_20260921.md` | **Qwen3.5-35B-A3B base vs Q4_K_M vs GPTQ-Int4**: official BF16 benchmark scores, what the GGUF quant benchmarks do and do not cover (no BF16 baseline exists), and our measured speed |
| `claudedocs/serving_framework_eval_20260915.md` | Phase 0 measured results: llama.cpp S1/S2/S3 on 8B + 35B-A3B via `scripts/serving_bench/`, prompt-cache incident/fix, FreeToken spike verdict, pending Phase 1/2 |
| `claudedocs/orin_chat_guide.md` | `chat.sh` keys, per-model commands, memory notes |
| `claudedocs/gguf_workflow.md` | safetensors → GGUF conversion |
| `claudedocs/vendor_stack_architecture_20260429.md` | ggml / llama.cpp internals reference |
| `claudedocs/edge_llm_research_report.md` | 2026-04 device/model/framework survey (pre-dates the measurements — treat its Orin numbers as estimates) |
| `claudedocs/npu_troubleshooting_report_20260407.md` | why QNN NPU fails |
| `scripts/orin_upgrade/` | JetPack upgrade runbook with built-in guards (boot-chain failure halts, DTB/NVMe warning) |
| `docs/superpowers/` | original design specs and phase plans |
