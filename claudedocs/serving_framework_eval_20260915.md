# Serving Framework Evaluation — Phase 0 Results (2026-09-15)

**Date**: 2026-09-15
**Scope**: Phase 0 only — llama.cpp baseline on the Jetson AGX Orin, plus a FreeToken aarch64
spike. vLLM and SGLang (Phase 2) were not run; see "Pending Phase 1/2" below.

Links:

- Spec (methodology, scenarios, deliverables, risks): `docs/superpowers/specs/2026-08-27-serving-framework-eval-design.md`
- Phase 0 plan: `docs/superpowers/plans/2026-09-15-serving-framework-eval-phase0.md`
- Harness: `scripts/serving_bench/` (`bench.py`, `scenarios.yaml`, `engines/llama_server.sh`, `monitor.sh`, `README.md`)
- Raw data: `results/serving_bench/` (`llamacpp_cu122_{8b,35b}_S{1,2,3}.jsonl`, `summary.md`, `s2_turns.csv`, `tegrastats_llamacpp_{8b,35b}.log`, `freetoken_spike/`)

## Methodology recap

Run on llama.cpp `6fdd0ac`, CUDA 12.2, `MODE_30W`, context 8192, Qwen3.5 thinking off,
`temperature 0`. Three scenarios (spec §5.2):

- **S1** — single-user chat, concurrency 1, three prompt lengths (short ~30 tok, code ~200 tok,
  long ~800 tok) × 3 repeats each, `max_tokens 256`.
- **S2** — agent loop, concurrency 1, fixed ~4k-token system prompt + a short user message per
  turn, 20 turns, non-accumulating history, `max_tokens 128`. Measures the prefix-cache effect.
- **S3** — concurrency sweep 1 → 2 → 4 → 8, mid-length S1 prompt × 4 requests per level,
  `max_tokens 256`. Aggregate tok/s and TTFT p50/p95.

Measured prompt sizes on the server (tokenizer-dependent, differs slightly by model):
`s1_short` 32 tok, `s1_code` 210 tok (8B) / 225 tok (35B), `s1_long` 884 tok,
`s2_system` 4062 tok (8B) / 4138 tok (35B).

**Harness validation gate** (spec §5.6): the authoritative cold rerun gives S1 8B gen tok/s
7.63, vs `llama-bench tg32` 7.63 — inside the required ±15% band. **PASS.** (The superseded
contaminated run — see the contamination incident below — gave 7.61; that figure predates the
cold rerun and should not be cited as the gate result.)

### Methodology notes: the prompt-cache contamination incident and fix

The first pass of 8B S1/S2 was run on a llama-server instance that had already been sent every
benchmark prompt text once, as `curl` probes used to calibrate `s1_long`/`s2_system` token
counts (see the calibration commit `6492470`). llama-server's prompt-prefix cache turned every
subsequent "cold" measurement of those same prompts into a cache hit, so the original 8B numbers
(TTFT ~226ms for `s1_long`, S2 turn-1 585ms, prefix speedup 1.4×) were all warm-cache values
mislabeled as cold. 35B was unaffected — it ran on a freshly loaded server that was never probed.

Fix: 8B S1 and S2 were rerun on a server restarted immediately before each scenario, with zero
probe requests sent beforehand. The contaminated originals were preserved (renamed, not deleted)
as `results/serving_bench/llamacpp_cu122_8b_S{1,2}.jsonl.contaminated`. The rerun produced
genuinely cold readings: S1 `long` rep 1 = 3822 ms (vs the contaminated 226 ms), S2 turn 1 =
18,098 ms (vs the contaminated 585 ms), prefix speedup 44.4× (vs the invalid 1.4×).

`scripts/serving_bench/metrics.py`'s `summarize()` was extended to report `ttft_first_ms` (cold,
first repeat) and `ttft_rest_mean_ms` (warm, mean of the rest) separately for S1, instead of only
the blended `ttft_ms_mean`. New methodology rule, now in `scripts/serving_bench/README.md`:
**restart the engine immediately before S1 and immediately before S2; never send probe requests
carrying benchmark prompt text to the server under test; S3 is warm-cache by design and is
unaffected.**

## Results — llama.cpp (CUDA 12.2, `6fdd0ac`)

Source: `results/serving_bench/summary.md`.

### S1 — single-user chat

| engine | runtime | model | prompt_id | n | prompt_tokens | ttft_first_ms (cold) | ttft_rest_mean_ms (warm) | tpot_ms_mean | gen_tok_s |
|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | cu12.2 | qwen3-8b | code | 3 | 210 | 998.4 | 228.4 | 132.1 | 7.6 |
| llama.cpp | cu12.2 | qwen3-8b | long | 3 | 884 | 3821.6 | 180.3 | 131.2 | 7.6 |
| llama.cpp | cu12.2 | qwen3-8b | short | 3 | 32 | 301.6 | 175.8 | 130.0 | 7.7 |
| llama.cpp | cu12.2 | qwen3.5-35b-a3b | code | 3 | 225 | 2259.1 | 295.3 | 92.4 | 10.8 |
| llama.cpp | cu12.2 | qwen3.5-35b-a3b | long | 3 | 884 | 6195.3 | 250.4 | 91.5 | 10.9 |
| llama.cpp | cu12.2 | qwen3.5-35b-a3b | short | 3 | 32 | 840.3 | 287.7 | 91.3 | 11.0 |

**Interpretation**: gen tok/s tracks the headline numbers in `CLAUDE.md` (8B untracked there
previously; 35B ≈10.8-11.0 vs the 10.8 re-baseline). Cold TTFT scales with prompt length as
expected (32 tok → ~300-840ms cold, 884 tok → 3.8-6.2s cold); warm TTFT is flat (~175-295ms)
regardless of prompt length once the prefix is cached, since llama-server's cache reuse covers
the full previous prompt. 8B's absolute TPOT (130-132 ms/tok) is slower per-token than 35B-A3B's
(91-92 ms/tok) because 35B-A3B is a 3B-active-param MoE — consistent with the "MoE beats dense"
finding already in `CLAUDE.md`.

### S2 — agent loop (4k shared system prompt, 20 turns, non-accumulating)

| engine | runtime | model | turns | ttft_turn1_ms (cold) | ttft_rest_mean_ms (warm) | prefix_speedup | tpot_ms_mean |
|---|---|---|---|---|---|---|---|
| llama.cpp | cu12.2 | qwen3-8b | 20 | 18097.5 | 407.7 | **44.4×** | 136.0 |
| llama.cpp | cu12.2 | qwen3.5-35b-a3b | 20 | 29286.3 | 977.0 | **30.0×** | 92.1 |

**Interpretation**: the speedup is **not** from llama.cpp's `--cache-reuse` flag — that flag is
never passed (`engines/llama_server.sh` does not set it, and `common/common.h` defaults
`n_cache_reuse` to 0, i.e. off). The mechanism is llama-server's per-slot prompt-prefix KV
retention: with `-np 1` the single slot keeps the previous request's KV cache, and when the next
request's prompt shares a prefix with it, that prefix is matched and reused without re-prefilling
— independent of `--cache-reuse`, which controls a different, fuzzier substring-reuse behavior.
This gives a large turn-1-to-steady-state speedup for both models even though the candidates doc
(`claudedocs/serving_framework_candidates_20260828.md`) previously listed llama.cpp as having
"no" cross-turn prefix reuse — that claim needs revision (see the doc update below). 44.4× (8B)
vs 30.0× (35B) is expected in the opposite direction of what raw compute would predict: the
35B-A3B cold prefill (29.3s) is proportionally larger than the 8B one (18.1s), but 35B's warm
TTFT (977ms) is also proportionally larger than 8B's (408ms) because MoE routing still costs
something per token even from cache, so the *ratio* comes out smaller for the bigger model. Both
numbers say the same practical thing: for an agent loop that resends a large shared system
prompt, cold-starting the server (or evicting its cache) is catastrophically expensive — tens of
seconds — while the steady state is comfortably sub-second.

### S3 — concurrency sweep

| engine | runtime | model | concurrency | n | errors | agg_tok_s | ttft_p50_ms | ttft_p95_ms |
|---|---|---|---|---|---|---|---|---|
| llama.cpp | cu12.2 | qwen3-8b | 1 | 4 | 0 | 7.5 | 224.9 | 906.5 |
| llama.cpp | cu12.2 | qwen3-8b | 2 | 8 | 0 | 13.1 | 263.2 | 448.6 |
| llama.cpp | cu12.2 | qwen3-8b | 4 | 16 | 0 | 16.0 | 364.5 | 551.2 |
| llama.cpp | cu12.2 | qwen3-8b | 8 | 32 | 0 | **17.4** | 643.4 | 2669.4 |
| llama.cpp | cu12.2 | qwen3.5-35b-a3b | 1 | 4 | 0 | 10.5 | 307.0 | 2070.6 |
| llama.cpp | cu12.2 | qwen3.5-35b-a3b | 2 | 8 | 0 | 21.7 | 434.1 | 562.8 |
| llama.cpp | cu12.2 | qwen3.5-35b-a3b | 4 | 16 | 0 | 25.8 | 882.1 | 2973.9 |
| llama.cpp | cu12.2 | qwen3.5-35b-a3b | 8 | 32 | 0 | **30.0** | 1362.2 | 1906.1 |

**Interpretation**: aggregate throughput scales sub-linearly with concurrency for both models —
8B goes 7.5 → 17.4 tok/s (2.3×) and 35B-A3B goes 10.5 → 30.0 tok/s (2.9×) from concurrency 1 to
8, not the 8× a fully parallel server would give — llama-server's `-np 8` splits the single
8192-token context into eight 1024-token slots and shares one GPU, so this measures aggregate
server throughput under a narrowed per-slot context, not 8 independent full-context streams
(spec §6 caveat, carried over from Task 7). TTFT p95 grows roughly with concurrency but noisily:
8B p95 goes 906.5 → 2669.4 ms (c=1 → c=8) while 35B-A3B's p95 is actually *lower* at c=8
(1906.1 ms) than at c=4 (2973.9 ms) or even c=1 (2070.6 ms) — the small n (4 requests per level)
makes p95 sensitive to which individual request happened to land behind a slower queue slot;
treat single-run p95 at this sample size as indicative, not definitive. 0 errors across all 120
S3 requests (both models, all four concurrency levels).

### ctx caveat (per spec §6, restated from Task 7)

S3 runs with `-np 8`, splitting the server's 8192-token context into 1024 tokens/slot; S1 and S2
run with `-np 1` (full 8192 tokens available to the single slot). S3 throughput numbers are
therefore not directly comparable to S1/S2 on a per-request-context basis.

### Truncation caveat

Several response categories hit their `max_tokens` cap rather than a natural stop: S1 `code`
hit `max_tokens=256` in all 6 runs (3 reps × 2 models), all 120 S3 responses hit `max_tokens=256`,
and 8 of the 20 8B S2 turns hit `max_tokens=128` (0 of 20 for 35B-A3B). TTFT is unaffected by
this (it's measured at first token, before any truncation could occur); TPOT is also unaffected
(it's a per-token rate, not sensitive to where generation was cut off) — but completion length,
and by extension `total_ms` and S3 `agg_tok_s`, may understate what a run without the cap would
show. `finish_reason` (which would let a future analysis distinguish `stop` from `length`
directly) is not currently recorded by the harness; this is a candidate future field.

## Memory (tegrastats)

Source: `results/serving_bench/tegrastats_llamacpp_{8b,35b}.log` (1-second samples, `RAM
used/total MB` field), verified with `grep`/`sort` rather than copied from a prior report.

All figures below use the decimal convention (1 GB = 1000 MB, matching tegrastats' own MB
field directly) applied consistently to both models.

- **8B**: 1747 samples. RAM used ranges **9860–10987 MB ≈ 9.9–11.0 GB**; the great majority of
  samples cluster in **10175–10573 MB ≈ 10.2–10.6 GB**, consistent with a steadily loaded
  8B Q4_K_M model plus KV cache and server overhead. 9860 MB is a brief low outlier (10 of 1747
  samples), most likely captured just before the model finished loading or just after it began
  unloading.
- **35B-A3B**: 1208 samples. RAM used ranges **10859–28827 MB**. The 10859 MB reading is a short
  transient (10 of 1208 samples) — the steady serving range is **24863–28827 MB ≈ 24.9–28.8 GB**,
  matching the ~22 GB GGUF file size plus KV cache and server overhead, and growing toward the
  top of that range as concurrency increases in S3 (8 parallel KV-cache slots).

**Power** (same tegrastats logs, `VDD_GPU_SOC` + `VDD_CPU_CV` instantaneous-power fields summed
per sample, computed with a small parse script over all samples — not copied from a prior
report): **8B** mean **≈7.5 W**, peak **≈10.7 W** (n=1747); **35B-A3B** mean **≈7.4 W**, peak
**≈10.3 W** (n=1208). Both are far below the 30 W `MODE_30W` power-mode cap — these two rails
account for only a quarter to a third of the budget even at peak, consistent with the box being
memory-bandwidth-bound rather than power-limited for both models at this concurrency range.

## FreeToken aarch64 spike — verdict

Full detail (committed evidence): `results/serving_bench/freetoken_spike/*.log`
(`install_base.log`, `torch_variant.log`, `torch_cuda_check.log`, `footprint.log`,
`start_time.log`, `end_time.log`).

| Step attempted | Exact failing command | First error line (verbatim) | Hypothesis |
|---|---|---|---|
| `uv pip install -e .` (base install) | `cd ~/ft-spike/src && uv pip install -e .` | `RuntimeError: nvcc 12.2 would build kernels linking libcudart.so.12, but torch 2.11.0+cu130 ships CUDA 13.0 (libcudart.so.13). Install a CUDA 13.x toolkit, or set FREETOKEN_ALLOW_CUDA_MISMATCH=1 to override.` | **H2 confirmed** (CUDA 13 toolchain required). **H1 refuted**: an aarch64 CUDA wheel for `torch>=2.11,<2.12` (`torch==2.11.0+cu130`) resolves and downloads fine — no missing-wheel problem. |
| `uv pip install "torch>=2.11,<2.12" --index-url https://pypi.org/simple` (H1-sharpening variant) | same | Succeeded — 28 packages installed, `torch==2.11.0+cu130` | Refutes H1 directly |
| `python3 -c "import torch; torch.cuda.is_available()"` (informational) | n/a | `UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12020)` → `cuda available: False` | Reinforces H2 — the JetPack 6.0 driver (CUDA 12.2) can't even run a CUDA-13-linked torch wheel, independent of the build-time `nvcc` gate |
| `[accel]` install / `ft --help` / dry-run serve | not run | n/a — base install did not succeed, so this step's precondition (per spike protocol) was not met | H3 (FP8/sm_87 kernel support) not reached |

**Recommendation**: FreeToken is **not revisitable on this Orin at all, on any JetPack version.**
JetPack 6.x for Orin ships CUDA 12.x only (6.0 = CUDA 12.2, currently installed; 6.1/6.2 = CUDA
12.6) — no JetPack upgrade on this hardware ever reaches FreeToken's required CUDA 13 toolchain.
CUDA 13 first arrives with JetPack 7, and JetPack 7 targets Thor-class hardware only — it is not
offered for Orin. The only way to satisfy FreeToken's CUDA-13 requirement is different hardware
(a Thor board), not an OS/toolkit upgrade on this box. Even on Thor, that only clears H2 — H3
(FP8/NVFP4 kernel support against the target GPU's actual tensor-core capability) remains
completely untested by this spike.

Elapsed: ~3 minutes (well under the 2-hour spike cap). Footprint left on Orin: `~/ft-spike` = 4.4
GB (`lib/` 4.3 GB — torch 2.11.0+cu130 + CUDA-13 deps, `bin/` 42 MB, `src/` 13 MB), left in place
per the no-delete rule.

## §10 quantization-difference caveat (verbatim from spec)

> GGUF Q4_K_M vs GPTQ-Int4 양자화 차이가 속도 비교를 오염 | 확실 | 해석 주의 | 결과 문서에 명시.
> 8B는 llama.cpp Q4_K_M vs bf16이라 llama.cpp에 유리 — 이것도 명시

Translation for reference: the risk is rated "certain" probability / "requires care in
interpretation" impact. Mitigation is to state it explicitly in the results document (this
section) — the 8B row compares llama.cpp's Q4_K_M GGUF against a planned bf16 HF baseline for
vLLM/SGLang, which favors llama.cpp; the 35B-A3B row compares GGUF Q4_K_M against a planned
GPTQ-Int4 baseline. **Because Phase 2 (vLLM/SGLang) has not run yet, this caveat has not yet
manifested as an actual cross-engine comparison — it applies the moment vLLM/SGLang numbers are
added to this document.**

## Phase-0 operational guidance (per workload)

Per spec §9 ("결과 문서에 워크로드별 권고가 있다"). Only llama.cpp has been measured so far
(Pending Phase 1/2 below), so this is llama.cpp-only guidance — **the cross-engine pick
(llama.cpp vs vLLM vs SGLang) is explicitly deferred to Phase 2**, once vLLM/SGLang numbers
exist to compare against.

- **1인 채팅 (single-user chat)**: 35B-A3B decodes at **10.8 t/s** vs 8B's **7.6 t/s** — the
  MoE model is faster to read despite being far larger, consistent with the "MoE beats dense"
  finding in `CLAUDE.md`. Cold TTFT for an 884-token prompt (`s1_long`) is **6.2 s** (35B-A3B)
  vs **3.8 s** (8B) — pick 8B if first-response latency on a fresh/cold server matters more than
  steady-state read speed, 35B-A3B otherwise.
- **에이전트 루프 (agent loop, one session)**: keep a single slot alive and restart-free between
  turns — that's what the measured numbers assume. After a first-turn cold prefill of **18 s**
  (8B) or **29 s** (35B-A3B) for the ~4k-token shared system prompt, steady-state turn-2+ TTFT
  drops to **≈0.4 s** (8B) / **≈1.0 s** (35B-A3B) — both are comfortably interactive once warm.
  The practical rule: never restart the server mid-loop, and never let anything else (a probe, a
  different scenario) touch the slot in between, or the next turn pays the cold cost again.
- **동시 N명 (concurrency)**: 35B-A3B aggregate throughput scales **10.5 → 30.0 tok/s** (agg,
  c=1→8) with p50 TTFT **1.4 s** at c=8; 8B scales **7.5 → 17.4 tok/s** over the same range.
  Both are sub-linear (llama-server's `-np 8` splits the 8192-token context into eight
  1024-token slots — see the ctx caveat above), so this is aggregate-server throughput under a
  narrowed per-slot context, not 8 independent full-context streams. If several people need to
  share one server concurrently, 35B-A3B gives roughly 1.7× the aggregate throughput of 8B at
  every concurrency level measured.

## Pending Phase 1/2

Only llama.cpp cells are filled. Per spec §9, 3 engines × 2 models × 3 scenarios = 18 cells; only
the 6 llama.cpp cells (8B S1/S2/S3, 35B-A3B S1/S2/S3) are populated. All vLLM and all SGLang
cells (12 of 18) are **empty** pending Phase 2:

| Engine | 8B S1 | 8B S2 | 8B S3 | 35B S1 | 35B S2 | 35B S3 |
|---|---|---|---|---|---|---|
| llama.cpp | done | done | done | done | done | done |
| vLLM | — | — | — | — | — | — |
| SGLang | — | — | — | — | — | — |

Readiness facts for Phase 1/2, as of 2026-09-15:

- Phase-2 assets are already staged on the Orin: HF snapshots under `~/hf/` (`Qwen/Qwen3-8B`
  bf16, `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`), and Docker images
  `mitakad/vllm:0.22.0-r36.5.tegra-aarch64-cp312-cu129-24.04` (44.7 GB) and
  `mitakad/sglang:0.6.0-r36.5.tegra-aarch64-cp312-cu129-24.04-at-commit-d093e70` (31.6 GB).
- Orin disk fell from 151 GB free (2026-08-28) to **42 GB free (91% used)** because of the staged
  images/models above. The spec §7.1 JetPack upgrade blocker wants ≥20 GB free — that is still
  satisfiable, but re-downloading `Qwen3-30B-A3B` or `Qwen3.5-122B-A10B` on top of the current
  footprint is not, without cleanup first.
- `~/ft-spike/` (4.4 GB, from the FreeToken spike above) is also left in place; a decision on
  whether to remove it is left to the user (no-delete-without-permission rule).
- `serial_log.py` is no longer running on the Orin, so the spec §7.1 reboot blocker for the
  JetPack 6.0 → 6.2 upgrade (Phase 1) is currently clear — Phase 1 can proceed on explicit user
  approval of timing, per the Phase 0 plan's "Not in this plan" note.
- Phase 2 (vLLM/SGLang) is planned to run after Phase 1 completes, reusing this harness unchanged
  (only `--base-url`, `--api-model`, `--fmt`, `--chat-template-kwargs` differ per spec §6).

## Orin footprint left behind (summary)

- `~/llama.cpp-build/build-cuda/` — the rebuilt llama.cpp binaries used for this run (pre-existing,
  not new).
- `~/serving_bench/` — harness copy + raw result files (rsynced back to this repo under
  `results/serving_bench/`; nothing deleted on the Orin).
- `~/hf/` — staged Phase-2 HF model snapshots (8B bf16, 35B-A3B GPTQ-Int4).
- Docker images `mitakad/vllm:...` (44.7 GB) and `mitakad/sglang:...` (31.6 GB), staged for Phase 2.
- `~/ft-spike/` — 4.4 GB FreeToken spike venv + source clone, left in place.
- Net effect: Orin disk free dropped from 151 GB (2026-08-28) to 42 GB (2026-09-15, 91% used).
