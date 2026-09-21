# Serving Framework Evaluation — Phase 1/2 Results (llama.cpp vs vLLM on Orin)

**Date**: 2026-09-20
**Device**: Jetson AGX Orin 64 GB, **JetPack 6.2.3 (L4T r36.5.2)**, CUDA 12.6.68, sm_87, **MODE_30W**
**Harness**: `scripts/serving_bench/` — S1 single-user, S2 agent loop, S3 concurrency
**Raw data**: `results/serving_bench/*.jsonl`, aggregated in `results/serving_bench/summary.md`
**Predecessor**: `claudedocs/serving_framework_eval_20260915.md` (Phase 0, llama.cpp @ CUDA 12.2)

> Every number here is MODE_30W. Nothing was measured at MAXN, so all of it is
> roughly half the board's power envelope.

---

## Headline

For **Qwen3.5-35B-A3B**, the two engines split the win cleanly, and not the way
the literature predicts:

| | llama.cpp (Q4_K_M) | vLLM 0.22 (GPTQ-Int4) |
|---|---:|---:|
| Single-user decode | 10.8 tok/s | **13.6 tok/s** |
| Repeated 4k prefix (S2 speedup) | **31.4×** | 1.0× default — **3.0× with `--enable-prefix-caching`** |
| Warm TTFT, agent loop | **0.92 s** | 29.2 s — **9.6 s** with the flag |
| Concurrency c=8, aggregate | 29.5 tok/s | **35.4 tok/s** |
| Concurrency c=8, TTFT p50 | **1.0 s** | 16.9 s |
| Cold start to first request | **~90 s** | ~7 min |

**vLLM decodes faster. llama.cpp answers faster.** Which matters depends
entirely on whether requests share a prefix, and on this model they usually do.

---

## Phase 1 — the JetPack upgrade

JetPack 6.0 (r36.3) → **6.2.3 (r36.5.2)**; CUDA 12.2 → 12.6.68, cuDNN 8.9 → 9.3,
TensorRT 8.6 → 10.3. Runbook and guards: `scripts/orin_upgrade/`.

**The upgrade changed llama.cpp performance by nothing measurable**, which is
the result that matters — it means the engine comparison below is not confounded
by the toolkit (spec D8):

| | CUDA 12.2 | CUDA 12.6 |
|---|---:|---:|
| 35B-A3B decode | 10.8 tok/s | 10.8 tok/s |
| 8B decode | 7.6 tok/s | 7.6 tok/s |
| 35B S2 speedup | 30.0× | 31.4× |
| 35B S3 c=8 | 30.0 tok/s | 29.5 tok/s |
| 35B `llama-bench` pp64 | 79.1 | 81.5 |
| 8B `llama-bench` pp64 | 174.2 | 180.7 |

Prefill gained ~3–4%; decode moved within noise, consistent with decode being
memory-bandwidth-bound rather than compute-bound.

Riva and NeMo were baselined *before* the upgrade and re-checked after: Riva
starts and sees the GPU, NeMo's torch still reports `cuda True`. No regression.
Evidence in `results/orin_upgrade/`.

---

## Phase 2 — the engine comparison

### The caveat that shapes everything below

**The two engines are not running the same weights.** vLLM cannot load MoE GGUF,
so each engine used the quantization it supports:

| Model | llama.cpp | vLLM |
|---|---|---|
| Qwen3-8B | Q4_K_M, 4.7 GB | **bf16, 16 GB** |
| Qwen3.5-35B-A3B | Q4_K_M, 22 GB | GPTQ-Int4, 21 GB |

For **8B this is not an engine comparison at all** on the decode axis — bf16
moves 3.4× more bytes per token, and decode here is bandwidth-bound. The **35B
pair is close enough to compare** (22 vs 21 GB), and that is where the
conclusions come from. The 8B numbers are still useful for concurrency
behaviour, which is a scheduling property rather than a bandwidth one.

### S1 — single user

| model | engine | fmt | cold TTFT (884 tok) | warm TTFT | decode |
|---|---|---|---:|---:|---:|
| 35B-A3B | llama.cpp | Q4_K_M | 6.10 s | **0.23 s** | 10.8 tok/s |
| 35B-A3B | vLLM | GPTQ-Int4 | 8.78 s | 8.83 s | **13.6 tok/s** |
| 8B | llama.cpp | Q4_K_M | 3.75 s | **0.18 s** | **7.6 tok/s** |
| 8B | vLLM | bf16 | **2.30 s** | 0.34 s | 6.2 tok/s |

vLLM decodes the 35B **26% faster** than llama.cpp. That is the honest
engine-vs-engine number, and it contradicts the Phase-0 expectation that
llama.cpp would hold the single-stream lead.

But look at the 35B warm TTFT: **8.83 s vs 0.23 s**. vLLM re-prefills every
request.

### S2 — agent loop (4k shared system prompt, 20 turns)

| model | engine | turn 1 | turns 2-20 | speedup |
|---|---|---:|---:|---:|
| 35B-A3B | llama.cpp | 28.8 s | **0.92 s** | **31.4×** |
| 35B-A3B | vLLM | 29.0 s | 29.2 s | **1.0×** |
| 8B | llama.cpp | 17.7 s | **0.40 s** | **44.3×** |
| 8B | vLLM | 10.9 s | 0.39 s | 27.9× |

**vLLM gets zero prefix reuse on the 35B.** Not "less" — none. Turn 20 costs
what turn 1 cost.

**Why**: Qwen3.5-35B-A3B is a *hybrid* model, and **vLLM disables prefix
caching for it by itself**. The engine config line in the startup log is
explicit, and it differs between the two models we ran:

| model | startup config | observed |
|---|---|---|
| Qwen3-8B | `enable_prefix_caching=True` | hit rate climbs to 59.5% |
| Qwen3.5-35B-A3B | **`enable_prefix_caching=False`** | `Prefix cache hit rate: 0.0%` for the whole run |

We did not pass `--enable-prefix-caching` either way; the 8B shows the default is
`True`, so the 35B's `False` is vLLM's own decision, not our configuration.

**It is the architecture, not the quantization.** The two models differ in both
(8B = bf16 + pure attention, 35B = GPTQ-Int4 + hybrid), so the measurements alone
cannot separate the causes. vLLM's source can. `ModelConfig.is_prefix_caching_supported`
(`vllm/config/model.py`) branches on `attn_type` and never consults quantization
state (`is_quantized` is an unrelated property):

```python
if attn_type == "hybrid":
    logger.debug("Hybrid models do not support prefix caching since "
                 "the feature is still experimental.")
    return False
```

Note vLLM's own wording: **"still experimental"**, not impossible — and the
default is the only thing that is off. See "Forcing it on" below: the support
exists and `--enable-prefix-caching` activates it.

**Scope — this is a property of this model, not of vLLM.** A user running
`QuantTrio/Qwen3.6-35B-A3B-AWQ` on vLLM reports prefix caching working normally
there, so Qwen3.6-35B-A3B is evidently not classified `hybrid`. Do not read the
rows above as "vLLM cannot reuse prefixes" — read them as "this 3.5 checkpoint
takes the hybrid path." An
unquantized (bf16/AWQ/FP8) build of the same hybrid model would behave
identically; a *non*-hybrid model would keep prefix caching whether quantized
or not.

The surrounding log lines are consistent with that classification - the engine
loads GDN (gated delta net) linear-attention kernels and aligns `mamba page size`
with attention page size:

```
qwen_gdn_linear_attn.py: Using Triton/FLA GDN prefill kernel
Setting attention block size to 1056 tokens to ensure that attention page size is >= mamba page size
Padding mamba page size by 0.76% ...
```

llama.cpp's mechanism - keep the previous prompt's state in the slot and reuse
the matching prefix - is indifferent to this, which is why it still gets 31x.

Evidence: `~/serving_bench/logs/vllm_35b_u075.log` vs `vllm_8b.log` on the Orin,
and `vllm/config/model.py` inside the container image.

On the pure-attention 8B, vLLM's caching does work (27.9×), confirming the
hybrid architecture is the cause rather than a misconfiguration on our side.

### Forcing prefix caching on (2026-09-21)

The auto-disable only fires when the flag is unset — `arg_utils.py` guards it
with `if self.enable_prefix_caching is None:`. Passing `--enable-prefix-caching`
explicitly survives, and vLLM 0.22 then activates a real mamba implementation
rather than a broken path:

```
WARNING config.py:355 Mamba cache mode is set to 'align' for
        Qwen3_5MoeForConditionalGeneration by default when prefix caching is enabled
INFO    config.py:375 Prefix caching in Mamba cache 'align' mode is currently
        enabled. Its support for Mamba layers is experimental.
```

**It works, and it is worth turning on — but it buys 3×, not 31×.**

| S2, 35B-A3B | turn 1 | turns 2-20 | vs no-APC |
|---|---:|---:|---:|
| vLLM, APC off (default) | 29.05 s | 29.25 s | — |
| **vLLM, `--enable-prefix-caching`** | 9.60 s † | **9.61 s** | **3.04×** |
| llama.cpp | 28.8 s | **0.92 s** | 31.8× |

† **Not a cold prefill.** `apc_probe.py` ran against this server before
`bench.py S2` and used the same `s2_system.txt`, so the cache was already
populated when turn 1 was issued. The other two turn-1 cells *are* cold. Do not
read this column as "APC made cold prefill 3× faster" — it did the opposite, see
below. Same contamination class as the Phase 0 incident; it is marked rather
than re-run because turns 2-20, the number this section is about, are unaffected.

All 20 turns landed within 9.60-9.61 s — the reuse is completely stable, just
coarse. The engine's own counter agrees with the block arithmetic: during S2 the
log reports `Prefix cache hit rate:` hovering at **~73%**, against the 3168/4138
= 76.6% predicted below.

**Cold prefill gets ~27% slower.** Measured directly on the running server with a
never-before-sent system prompt of matched length (a generated document with a
random nonce, 4,127 tokens vs the S2 prompt's 4,138):

| first request on an uncached prefix | TTFT |
|---|---:|
| vLLM, APC off (S2 turn 1, 4,138 tok) | 29.05 s |
| **vLLM, APC on (novel prefix, 4,127 tok)** | **36.95 s** |

So the flag costs ~7.9 s once and saves ~19.6 s on every subsequent request
sharing that prefix: it pays for itself on the **second** request and is a loss
only for workloads that never repeat a prefix.

**Why 3× and not more — the block-size floor.** The engine pins
`attention block size = 1056 tokens` so the attention page is at least as large
as the mamba page. Prefix reuse is block-granular, so the 4,138-token prompt
(4,088-token shared system prompt + 34-token user turn + 16 of chat template,
confirmed via `/tokenize`) reuses `floor(4138/1056) = 3` blocks = 3,168 tokens
and **re-prefills the remaining ~970 on every request**. That residual is the
9.6 s.

A corollary: vLLM's gain here depends on where the prompt falls relative to a
1056-token boundary. 4,138 is nearly the worst case (0.92 of a block wasted);
a prompt near a multiple of 1056 would reuse almost all of it. llama.cpp's
per-slot reuse is indifferent to prompt length.

`--mamba-block-size 512` does **not** fix it. The flag is accepted
(`mamba_block_size: 512` appears in non-default args) but the attention block
stays at 1056, because 1056 is a floor set by the model's mamba state size
rather than by this flag. Measured side by side the timings are identical:

| | attn block | R3 | R4 |
|---|---:|---:|---:|
| APC off | — | 38.43 s | 38.15 s |
| APC on, default block | 1056 | 19.20 s | 19.16 s |
| APC on, `--mamba-block-size 512` | 1056 | 19.16 s | 19.12 s |

So: pass `--enable-prefix-caching` and leave the block size alone — 512 buys
nothing.

⚠️ **KV cache size is not reproducible run to run on this board, so do not
read a cache-size delta as an effect of a flag.** Four starts, same
`--gpu-memory-utilization 0.75`:

| run | flags | GPU KV cache |
|---|---|---:|
| `vllm_35b_u075` | APC off | 746,216 tok |
| `vllm_35b_apc` | APC on | 589,238 tok |
| `vllm_35b_apc512` | APC on + block 512 | 369,225 tok |
| `vllm_35b_final` | APC on (**same flags as `vllm_35b_apc`**) | 369,225 tok |

The last two rows have different flags but the same cache; rows 2 and 4 have the
**same** flags and differ by 37%. vLLM sizes the cache from free memory measured
at startup, and on Tegra unified memory that reading depends on host state
(page cache, fragmentation) rather than only on configuration. An earlier draft
of this section attributed the 589k→369k drop to `--mamba-block-size 512`; that
was wrong. Even the smallest observed cache (369k tokens, 45× theoretical
concurrency at 8192 ctx) is far more headroom than this board can use.

**Correctness — the output does not change.** vLLM warns that manually enabling
an unsupported feature "may cause the engine to crash or produce incorrect
outputs", so this was verified rather than assumed. A fixed 4-request sequence
(greedy, `temperature=0`, fixed seed) was replayed against freshly started
servers; `R1` is a guaranteed miss because the cache is empty, `R3` repeats it
after the cache is populated:

| server | R1 (cold) vs R3 (served from cache) |
|---|---|
| APC off | identical — establishes the engine is deterministic at all |
| APC on, block 1056 | **identical**, 573 bytes |
| APC on, block 512 | **identical** |

`R3` is not a trivial case: at a 1056-token block it restores 3168 tokens of
attention **and mamba** state from cache and recomputes the 965-token tail, so
this exercises exactly the partial-reuse path that the warning is about.

One thing this cannot show: comparing *across* the two servers, outputs differ
slightly — but `R1` differs too, and `R1` is a cold prefill on both. The servers
have different block sizes, which changes prefill chunking and therefore
floating-point accumulation order. Cross-server comparison is confounded and
cannot isolate caching; the within-server test above is the valid one.

Raw data: `results/serving_bench/vllm_cu12.6-apc_35b_S2.jsonl`,
`apc_probe_{on,off,blk512}.json`.

### S3 — concurrency

| model | engine | c=1 | c=2 | c=4 | c=8 | TTFT p50 @ c=8 |
|---|---|---:|---:|---:|---:|---:|
| 35B-A3B | llama.cpp | 10.4 | 21.5 | 26.1 | 29.5 | **1.0 s** |
| 35B-A3B | vLLM | 10.5 | 13.5 | 23.1 | **35.4** | 16.9 s |
| 8B | llama.cpp | 7.5 | 13.1 | 16.2 | 17.5 | **0.64 s** |
| 8B | vLLM | 6.2 | 13.3 | 25.3 | **48.9** | 0.49 s |

On the **8B**, vLLM does what PagedAttention promises: llama.cpp plateaus at
17.5 tok/s (its 8 slots each get a fixed 1024-token slice of the 8192 context)
while vLLM keeps scaling to 48.9 — **2.8×** — and *still* answers faster (p50
0.49 s vs 0.64 s). This is the clean win.

On the **35B**, vLLM's throughput edge is only 1.2× and it costs **17× worse
latency** (16.9 s vs 1.0 s p50), because every one of those concurrent requests
pays a full prefill.

---

## Operational guidance

**Single-user chat, 35B**: either works; vLLM is 26% faster per token but takes
~7 minutes to start and re-prefills long prompts. For interactive use where you
resend context, llama.cpp's 0.23 s warm TTFT dominates the 2.8 tok/s decode
difference.

**Agent loop with a long shared system prompt, 35B**: **llama.cpp, decisively.**
0.92 s vs 29.2 s per turn is not a tuning gap, it is a missing feature for this
model class.

**Many concurrent users**: depends on the model. Pure-attention (8B) →
**vLLM**, 2.8× throughput at equal-or-better latency. Hybrid (35B-A3B) → vLLM
buys 1.2× throughput for 17× latency; only worth it for batch/offline work.

**Cold-start cost**: llama.cpp ~90 s for the 35B; vLLM ~7 min (75 s weights +
208 s `torch.compile` + graph capture + warmup). The 8B is 20 s vs ~3.5 min.
Matters if the service restarts often.

---

## Methodology notes

**vLLM memory on unified memory — the guidance is inverted.** The plan said to
start at `--gpu-memory-utilization 0.5` and step *down* on OOM, reasoning from
discrete-GPU habits. The actual failure was the opposite:

```
ValueError: No available memory for the cache blocks.
Try increasing `gpu_memory_utilization` ...
```

vLLM sizes its budget as a fraction of *total* memory — 61 GB here — so 0.5
gives a 30.5 GB budget from which the 21 GB of weights plus activations and
CUDA-graph profiling leave nothing for KV. **0.75 (~45.8 GB) works**, yielding a
746k-token KV cache (theoretical max concurrency 91× at 8192 ctx) and still
leaving ~15 GB for the OS and Docker. Step *up* when the model is large relative
to the budget. `engines/vllm.sh` carries the corrected note.

**Cold-start protocol differs between the engines, deliberately.** llama.cpp was
restarted before S1 and again before S2, per the Phase-0 prompt-cache rule.
vLLM was not: startup costs minutes, and its cache is content-addressed rather
than per-slot, so a fresh S2 system prompt is cold regardless. The S1 prompts
and the S2 system prompt share no content, so vLLM's S2 turn 1 is a genuine
cold prefill — **for the APC-off runs only.** In the `--enable-prefix-caching`
run the probe had already sent `s2_system.txt`, so that turn 1 is warm; it is
flagged in the table above and the true cold number was measured separately.

**vLLM flags — what we ran, and what a multi-GPU deployment would add.** Our
launch line (`engines/vllm.sh`) is deliberately minimal because the Orin has a
single integrated GPU. Compared against a typical 2-GPU `vllm serve`:

| flag | ours | 2-GPU deployment | does it affect these results? |
|---|---|---|---|
| `--tensor-parallel-size` | 1 | 1 | no — same |
| `--data-parallel-size` | 1 | 2 | **not available**: one iGPU |
| `--enable-expert-parallel` | off | on | no-op at DP=1; EP shards experts *across ranks* |
| `--api-server-count` | 1 | 2 | no — the HTTP frontend is nowhere near the bottleneck at 10–35 tok/s |
| `--gpu-memory-utilization` | 0.75 | 0.85 | no — 0.75 already yields a 746k-token KV cache (91× concurrency at 8192 ctx); we never exceeded c=8, so KV was never the binding constraint |
| `--max-model-len` | 8192 | 32768 | no — S2's shared prefix is 4k; raising the cap cannot create prefix reuse the engine has disabled |
| `enable_thinking: false` | per-request | server default | no — same effect, ours is passed by the harness on every request |
| `--trust-remote-code` | off | on | no — `qwen3_5_moe` is in-tree in vLLM 0.22 and loaded without it |

So the single-GPU constraint costs vLLM throughput scaling it would get from DP,
but **none of these flags bear on the prefix-caching result** — that is decided
by `enable_prefix_caching=False`, which vLLM sets from the model architecture.

**Truncation**: as in Phase 0, S1 `code` and all S3 responses hit
`max_tokens`; TTFT and per-token decode are unaffected but `total_ms` and
aggregate tok/s are cap-bound. `finish_reason` is still not recorded.

**`runtime` field**: records now carry `cu12.2` / `cu12.6` so the two baselines
stay separable in `summary.md` and `s2_turns.csv`.

---

## What is still open

- **SGLang is untested.** The image (`mitakad/sglang:0.6.0-r36.5...`, 31.6 GB)
  is staged. Its RadixAttention is the same content-addressed family as vLLM's
  APC, so the hybrid-model limitation above probably applies — worth confirming,
  because if it does, llama.cpp is the only engine here with working prefix
  reuse for Qwen3.5-35B-A3B.
- **MAXN.** Everything is 30W. GPU+CPU rails ran ~7.5 W mean in Phase 0, far
  under the cap, so the ceiling is likely bandwidth rather than power — but that
  is an inference, not a measurement.
- **Qwen3.6-35B-A3B** reuses the `qwen3_5_moe` architecture and would load on
  the current llama.cpp build without a rebuild; its MTP variant could add
  1.5–2× via speculative decoding.
- **`finish_reason`** in the JSONL schema, to separate "verbose" from "truncated".
