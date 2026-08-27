# GGUF Workflow — Acquisition, Conversion, Quantization

**Date**: 2026-04-29
**Purpose**: How GGUFs were obtained for this project, how to convert HF safetensors → GGUF when no pre-built GGUF exists, and the quantization step.

## Two ways to get a GGUF

```
                     ┌────────────────────┐
                     │  Hugging Face Hub  │
                     └──┬─────────────┬───┘
                        │             │
                A) Pre-built          B) Original safetensors
                  *-GGUF repo         (model authors)
                        │             │
                  hf_hub_download     huggingface_hub.snapshot_download
                        │             │
                        │     convert_hf_to_gguf.py (F16 GGUF)
                        │             │
                        │     llama-quantize (Q4_K_M etc.)
                        │             │
                        ▼             ▼
                   *.gguf  (drop into models/ and run)
```

**Path A** is what we used for the measurement sweep — community + official `*-GGUF` repos already had Q4_K_M files.
**Path B** is needed when no GGUF repo exists, when you want a non-standard quantization, or when you need to verify the conversion is reproducible.

---

## Path A — download a pre-built GGUF

### Repos used in this project

| Family | Repo | Provider |
|---|---|---|
| Qwen3-{4B,8B,14B,32B,30B-A3B} | `Qwen/Qwen3-*-GGUF` | **Official (Alibaba/Qwen)** |
| Qwen3-{0.6B,1.7B} | `unsloth/Qwen3-*-GGUF` | Community (Unsloth) |
| Qwen3.5-{27B,35B-A3B,122B-A10B} | `unsloth/Qwen3.5-*-GGUF` | Community (Unsloth) |
| Llama-3.1-8B-Instruct | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | Community (bartowski) |
| Gemma-2-9B-it | `bartowski/gemma-2-9b-it-GGUF` | Community |
| Phi-3.5-mini-instruct | `bartowski/Phi-3.5-mini-instruct-GGUF` | Community |
| Mistral-7B-v0.3 | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` | Community |

All quantized to **Q4_K_M** for consistency.

### Single file download

```bash
pip install huggingface_hub

python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Qwen/Qwen3-8B-GGUF', 'Qwen3-8B-Q4_K_M.gguf', local_dir='models')
"
```

### Multi-split file (large MoE models)

Some GGUFs are split into pieces (e.g. Qwen3.5-122B-A10B is 3 files):

```bash
huggingface-cli download unsloth/Qwen3.5-122B-A10B-GGUF \
    Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf \
    Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00002-of-00003.gguf \
    Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00003-of-00003.gguf \
    --local-dir models
```

llama.cpp loads automatically when you point to the **first** split:
`-m Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf`

### Authentication

Some repos (Llama official, gated models) require an HF token:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
# or
huggingface-cli login
```

Without a token, you'll see `Warning: You are sending unauthenticated requests` (still works for ungated models, just rate-limited and slower).

### Why this is preferred

- **Already verified by the community** — Unsloth/bartowski use imatrix calibration (improves PPL vs naive quant)
- **Saves time and disk** — large dense models (32B) take 30-60 min to convert
- **Reproducibility** — same SHA256 for all users

### When this fails

Some repos go missing or rename files. We hit this with `bartowski/Meta-Llama-3.1-70B-Instruct-GGUF` in the sweep — got 404s on every Q4_K_M file pattern. Fall back to Path B.

---

## Path B — convert HF safetensors → GGUF yourself

Required when:
- No pre-built GGUF exists for the model
- You want a non-standard quantization (e.g. Q3_K_S, F16 baseline)
- You need to verify the entire pipeline is reproducible
- You've fine-tuned a model and need to GGUF-ize it

### Pipeline

```
HF safetensors  ──[convert_hf_to_gguf.py]──>  F16 GGUF  ──[llama-quantize]──>  Q4_K_M GGUF
   ~70 GB                                       ~70 GB                          ~21 GB
   (35B model)
```

### Step 0 — build llama.cpp tools

You need both `convert_hf_to_gguf.py` (Python) and `llama-quantize` (CMake target). Run our build script:

```bash
ssh home.orin.local
~/build_llama_orin.sh
# or with explicit dir:
~/build_llama_orin.sh /tmp/llama.cpp-build
```

This builds `llama-cli`, `llama-quantize`, `llama-server` into `build-cuda/bin/`, plus a CPU-only fallback in `build-cpu/bin/`.

### Step 1 — install Python deps

```bash
pip install --user -r /tmp/llama.cpp-build/requirements/requirements-convert_hf_to_gguf.txt
```

Pulls `torch`, `transformers`, `sentencepiece`, `protobuf`, `gguf`, etc.

### Step 2 — run the helper

We provide `scripts/convert_to_gguf.sh` (mirrored to `~/convert_to_gguf.sh` on Orin):

```bash
./convert_to_gguf.sh <hf_repo> <out_basename> [quant_type]
```

Examples:

```bash
# Llama-3.1-8B-Instruct (gated — requires HF_TOKEN)
HF_TOKEN=hf_xxx ./convert_to_gguf.sh \
    meta-llama/Meta-Llama-3.1-8B-Instruct \
    llama31_8b_inst \
    Q4_K_M

# Qwen3.5-35B-A3B MoE (ungated)
./convert_to_gguf.sh \
    Qwen/Qwen3.5-35B-A3B \
    qwen35_35b_a3b \
    Q4_K_M
```

Output:
```
~/conversion/gguf/llama31_8b_inst-Q4_K_M.gguf
```

### Disk and time budget

| Model | safetensors | F16 GGUF | Q4_K_M output | Peak disk | Wall time |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B (dense) | ~16 GB | ~16 GB | ~5 GB | ~32 GB | 5-10 min |
| Qwen3-14B (dense) | ~28 GB | ~28 GB | ~9 GB | ~56 GB | 10-15 min |
| Qwen3-32B (dense) | ~64 GB | ~64 GB | ~20 GB | ~128 GB | 25-40 min |
| Qwen3.5-35B-A3B (MoE) | ~70 GB | ~70 GB | ~21 GB | ~140 GB | 30-60 min |
| Qwen3.5-122B-A10B (MoE) | ~250 GB | ~250 GB | ~73 GB | ~500 GB | 2-4 hours |

The script defaults to **deleting the safetensors right after F16 is produced** and **deleting the F16 right after the quantized file is produced**, so the live disk peak is roughly `safetensors + F16` (~2× model size in F16). Override with `KEEP_F16=1` and `KEEP_HF=1` if you want intermediates.

### What `convert_hf_to_gguf.py` does

1. Reads `config.json`, `tokenizer.json` / `tokenizer.model`, `*.safetensors` shards
2. Maps each tensor name from HF style (`model.layers.0.self_attn.q_proj.weight`) to GGUF style (`blk.0.attn_q.weight`)
3. Reorders / transposes / fuses tensors as needed by the GGUF format
4. Writes a single `.gguf` file with the chosen `--outtype` (f16 / bf16 / f32 / q8_0)
5. **Auto-detects MoE** (Qwen3-MoE, DeepSeek-MoE, Mixtral) and handles `mul_mat_id` routing

### What `llama-quantize` does

```
llama-quantize <input-F16.gguf> <output-Q4_K_M.gguf> Q4_K_M
```

- Reads each tensor block from the F16 GGUF
- Applies the requested quantization scheme (Q4_K_M = 4-bit K-quants medium with select higher-precision tensors)
- Re-packs into a new GGUF file
- Common targets: `Q4_K_M` (best quality/size), `Q4_0` (baseline), `Q5_K_M` (more accurate), `Q6_K` (high quality), `Q8_0` (near-FP16), `Q2_K` (extreme compression)

For best quality at a given size, llama.cpp recommends `Q4_K_M` (the project default).

---

## Why use `Q4_K_M` for everything?

| Quant | Bits | Size (1B params) | PPL drop† | Notes |
|---|---:|---:|---:|---|
| F16 | 16 | 2.0 GB | baseline | reference |
| Q8_0 | 8 | 1.0 GB | ~+0.05 | near-lossless |
| Q6_K | 6 | 0.8 GB | ~+0.1 | excellent |
| **Q4_K_M** | 4 | **0.55 GB** | **~+0.5** | **project default — sweet spot** |
| Q4_0 | 4 | 0.55 GB | ~+1.5 | older naive 4-bit |
| Q3_K_M | 3 | 0.45 GB | ~+1.5 | lossy but small |
| Q2_K | 2 | 0.35 GB | ~+5 | last resort |

† perplexity delta on wikitext, rough; varies by model.

For the edge profiling sweep we standardized on **Q4_K_M** so per-device max sizes and tok/s are directly comparable across families. If you want a different quantization, run `llama-quantize` on the F16 intermediate.

---

## Sanity check after conversion

After the script produces `<basename>-Q4_K_M.gguf`, verify the model loads and generates correctly:

```bash
~/chat.sh --bench llama-3.1-8b   # if you converted to the standard filename
# or directly
/tmp/llama.cpp-build/build-cuda/bin/llama-cli \
    -m ~/conversion/gguf/llama31_8b_inst-Q4_K_M.gguf \
    -p "The capital of France is" -n 16 -ngl 99 -no-cnv -st
```

Expected: similar output and tok/s to the **pre-built community Q4_K_M** for the same model. If much worse PPL or different output style, suspect the convert step (tokenizer mismatch, wrong tensor naming, MoE expert routing).

---

## Common pitfalls

1. **Gated repos** — Llama official, Mistral-Nemo, etc. need `HF_TOKEN`. Set it in the env before calling the script.
2. **MoE models** — `convert_hf_to_gguf.py` handles Qwen3-MoE, Mixtral, DeepSeek-MoE natively. If you see "unknown architecture", you may need a newer llama.cpp.
3. **Architecture not yet supported** — for very new models, `convert_hf_to_gguf.py` may error. Update llama.cpp checkout (`cd /tmp/llama.cpp-build && git pull && bash ~/build_llama_orin.sh`).
4. **OOM during quantization** — `llama-quantize` reads the entire F16 into memory at peak. For 70B+ on 64 GB Orin, the quantize step itself can OOM. Use a swap file or quantize on a host with more RAM, then ship the file.
5. **Tokenizer mismatch** — if `--outtype f16` succeeds but `llama-cli` outputs garbage, the tokenizer wasn't found. Make sure `tokenizer.json` / `tokenizer.model` was downloaded with the safetensors.

---

## Reference: scripts in this repo

| Script | Purpose |
|---|---|
| `scripts/build_llama_orin.sh` | Build llama.cpp natively on Orin (CUDA + CPU), produces all required binaries |
| `scripts/convert_to_gguf.sh` | Download HF model → F16 GGUF → quantized GGUF |
| `scripts/chat_orin.sh` (= `~/chat.sh` on Orin) | Interactive chat with all measured models |
| `build/docker/Dockerfile.orin-cuda` | (existing) Docker-based llama.cpp build for cross-compile use |
| `build/build_all.sh` | (existing) Builds the tiny-llm-profiler binary across all targets |

Sources:
- [Hugging Face Hub docs](https://huggingface.co/docs/huggingface_hub)
- [llama.cpp convert_hf_to_gguf.py](https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py)
- [Quantization formats explained](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)
