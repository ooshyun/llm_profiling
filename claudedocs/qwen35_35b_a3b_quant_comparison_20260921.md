# Qwen3.5-35B-A3B — base vs Q4_K_M (and the GPTQ-Int4 we also run)

**Date**: 2026-09-21
**Why this doc**: the serving evaluation runs llama.cpp on a Q4_K_M GGUF and vLLM
on a GPTQ-Int4 checkpoint, neither of which is the model Qwen benchmarked. This
records what is actually known about the quality gap, and — more importantly —
what is *not* known.

> **Read this first.** A rigorous "base vs Q4_K_M on major benchmarks" table does
> not exist, publicly or here. Qwen publishes scores for the **unquantized**
> model only; the community GGUF benchmarks publish perplexity/KLD **with no
> BF16 baseline row**, so they support comparisons *between quants* and not
> against the original. We measured speed, not quality. Everything below is
> labelled by source.

---

## 1. The three variants in play

| | weights | size on disk | used by | present on the Orin |
|---|---|---:|---|---|
| `Qwen/Qwen3.5-35B-A3B` | BF16 | ~70 GB | — (reference) | **no** |
| `Qwen3.5-35B-A3B-Q4_K_M.gguf` | Q4_K_M GGUF, `quantized_by: Unsloth` | **20.50 GiB** | llama.cpp | yes, `~/models/` |
| `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | GPTQ-Int4 (`--quantization moe_wna16`) | ~21 GB | vLLM | yes, `~/hf/` |

vLLM cannot load MoE GGUF, which is why the two engines run different weights at
all. Both are 4-bit and within ~1 GB of each other, so the 35B engine comparison
is reasonably fair; the 8B one (Q4_K_M vs BF16) is not.

⚠️ Our GGUF is **20.50 GiB**, but Unsloth's published benchmark table lists its
`Q4_K_M` at **18.49 GB**. Those are probably not the same file — ours may be a
dynamic (`UD-`) variant despite the filename. Metadata confirms only
`general.quantized_by = Unsloth`, `general.basename = Qwen3.5-35B-A3B`,
`general.architecture = qwen35moe`, 40 blocks. **Do not map the PPL row in §3 onto
our file without checking the exact repo it came from.**

---

## 2. Official scores — unquantized model only

Architecture: 35B total / **3B active**, 40 layers, 256 experts (8 routed + 1
shared), **hybrid Gated Delta Network + sparse MoE**, 262,144-token native
context (extensible to ~1M).

> The model card's own wording confirms independently what vLLM's startup log
> told us: this is a *hybrid* architecture. That is the reason vLLM disables
> prefix caching for it by default — see
> `serving_framework_eval_phase2_20260920.md`.

| Benchmark | Score |
|---|---:|
| MMLU-Pro | 85.3 |
| MMLU-Redux | 93.3 |
| C-Eval | 90.2 |
| SuperGPQA | 63.4 |
| IFEval | 91.9 |
| GPQA Diamond | 84.2 |
| HMMT Feb 25 | 89.0 |
| HMMT Nov 25 | 89.2 |
| SWE-bench Verified | 69.2 |
| LiveCodeBench v6 | 74.6 |
| CodeForces (Elo) | 2028 |
| MMMU | 81.4 |
| MMMU-Pro | 75.1 |
| MathVision | 83.9 |
| VideoMME (w/ sub.) | 86.6 |

**Source**: the `Qwen/Qwen3.5-35B-A3B` model card. **These are BF16 numbers.**
They are the ceiling for our deployment, not a measurement of it. Nothing here
has been reproduced locally.

---

## 3. What is published about the quantized versions

| | disk | PPL | KLD 99.9% | mean KLD | source |
|---|---:|---:|---:|---:|---|
| **BF16 baseline** | ~70 GB | **not published** | **not published** | **not published** | — |
| Unsloth `Q4_K_M` | 18.49 GB | 6.6053 | 0.5478 | — | Unsloth GGUF benchmarks |
| Unsloth `UD-Q4_K_L` | 21.3 GB | 6.586 | — | 0.015 | via APEX comparison |
| APEX "Quality" | 18.8 GB | 6.527 | — | 0.011 | mudler/APEX card |
| Unsloth `UD-Q8_K_XL` | 2–3× above | — | — | **0.0025** | Unsloth |

**The missing baseline row is the whole problem.** Perplexity ~6.6 is only
meaningful against the unquantized model's perplexity on the same corpus, and
that number is not published. Unsloth also notes its imatrix uses long-context
chat and tool-calling data rather than the usual Wiki-test/512-ctx setup, so
these figures are not comparable to PPL numbers quoted elsewhere.

**A claim we checked and withdrew.** A search summary surfaced "the ~17 GB
`Q4_K_M` matches the full model on Terminal-Bench 2.1" and an earlier draft of
this doc carried it here. Re-fetching the Qwen3.5 GGUF benchmarks page shows it
**does not mention Terminal-Bench at all**; the claim came from a Qwen3.6
discussion that the search had blended in. It says nothing about Qwen3.5 and has
been removed rather than re-attributed.

**No MMLU / GPQA / AIME / LiveCodeBench scores exist for any Q4_K_M build of this
model**, from Qwen or from the quantizers.

---

## 4. What we actually measured — speed, on this board

Jetson AGX Orin 64 GB, JetPack 6.2.3, CUDA 12.6, **MODE_30W**. S2 = 20-turn agent
loop over a **4,088-token shared system prompt** (+34-token user turn +16 of chat
template = 4,138-token prompts; no response hit the 128-token cap, so output
lengths are comparable).

| engine / weights | TTFT (turns 2-20) | total latency | output tokens | decode |
|---|---:|---:|---:|---:|
| llama.cpp, Q4_K_M | **0.92 s** | **8.69 s** | 83.3 | 10.7 tok/s |
| vLLM, GPTQ-Int4, APC off (default) | 29.25 s | 35.09 s | 80.8 | 13.8 tok/s |
| vLLM, GPTQ-Int4, `--enable-prefix-caching` | 9.61 s | 15.78 s | 82.6 | 13.4 tok/s |

Cold (first request on an unseen prefix) runs the other way: **29.05 s without
the flag, 36.95 s with it** at matched prompt length. The flag costs ~8 s once
and saves ~20 s per repeat.

**GPTQ-Int4 decodes ~25% faster per token than Q4_K_M** (13.4 vs 10.7 tok/s).
Whether that is the quantization or the engine is not separable from this data —
the format and the runtime change together.

Full detail: `serving_framework_eval_phase2_20260920.md`.

---

## 5. Not measured / open

- **Quality of either quantization, at all.** No local eval was run. The BF16
  model is not on disk (~70 GB) and 144 GB is free, so a baseline *could* be
  downloaded if a quality comparison is ever needed.
- **Q4_K_M vs GPTQ-Int4 quality against each other** — both are on disk, so this
  is the cheap version of the question: run the same eval through both servers.
  It needs no download and would tell us whether the 25% decode advantage costs
  anything.
- **Which Unsloth artifact our 20.50 GiB file actually is** (see §1).
- Everything here is MODE_30W; MAXN is unmeasured.

## Sources

- [Qwen/Qwen3.5-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) — §2 scores, architecture
- [Unsloth Qwen3.5 GGUF benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks) — §3 PPL/KLD (verified: no Terminal-Bench content)
- [mudler/Qwen3.5-35B-A3B-APEX-GGUF](https://huggingface.co/mudler/Qwen3.5-35B-A3B-APEX-GGUF) — §3 APEX comparison
- §4 is ours: `results/serving_bench/*.jsonl`
