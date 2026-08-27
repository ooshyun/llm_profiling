# Orin Chat Guide — Interactive LLM with `chat.sh`

**Date**: 2026-04-29
**Target**: Jetson AGX Orin 64GB (`home.orin.local`)
**Models loaded**: 14 (Qwen3 / Qwen3.5 / Llama-3.1 / Gemma-2 / Phi-3.5 / Mistral-7B; all GGUF Q4_K_M)
**Helper**: `~/chat.sh` on Orin

## Quick start

```bash
ssh home.orin.local
./chat.sh                  # show available model keys + measured speed
./chat.sh qwen3-8b         # interactive chat with Qwen3-8B
./chat.sh qwen3-30b-a3b    # MoE — fastest large model
./chat.sh --bench phi-3.5-mini   # one-shot speed test
```

## Available models and recommended commands

| Key | File | `-c` | `-ngl` | gen tok/s | Resident GPU |
|---|---|---:|---:|---:|---:|
| `qwen3-0.6b` | Qwen3-0.6B-Q4_K_M.gguf | 8192 | 99 | ~38 | 0.5 GB |
| `qwen3-1.7b` | Qwen3-1.7B-Q4_K_M.gguf | 8192 | 99 | 19.5 | 1.5 GB |
| `qwen3-4b` | Qwen3-4B-Q4_K_M.gguf | 8192 | 99 | (n/a) | ~2.5 GB |
| `qwen3-8b` | Qwen3-8B-Q4_K_M.gguf | 8192 | 99 | 7.5 | 4.8 GB |
| `qwen3-14b` | Qwen3-14B-Q4_K_M.gguf | 4096 | 99 | 4.5 | 8.6 GB |
| `qwen3-32b` | Qwen3-32B-Q4_K_M.gguf | 2048 | 99 | 2.2 | 18.8 GB |
| `qwen3-30b-a3b` | Qwen3-30B-A3B-Q4_K_M.gguf | 4096 | 99 | **13.5** | 17.6 GB |
| `qwen3.5-27b` | Qwen3.5-27B-Q4_K_M.gguf | 2048 | 99 | 2.3 | 15.7 GB |
| `qwen3.5-35b-a3b` | Qwen3.5-35B-A3B-Q4_K_M.gguf | 4096 | 99 | **9.6** | 20.6 GB |
| ~~`qwen3.5-122b-a10b`~~ | ~~(3-split GGUF)~~ | ~~2048~~ | ~~**30**~~ | ~~1.9~~ | **deleted 2026-04-29** to free disk for 35B-A3B conversion test; re-download from `unsloth/Qwen3.5-122B-A10B-GGUF` (~76 GB) to restore |
| `llama-3.1-8b` | Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf | 8192 | 99 | 8.0 | 4.8 GB |
| `gemma-2-9b` | gemma-2-9b-it-Q4_K_M.gguf | 8192 | 99 | 6.4 | 6.5 GB |
| `phi-3.5-mini` | Phi-3.5-mini-instruct-Q4_K_M.gguf | 8192 | 99 | **12.3** | 3.0 GB |
| `mistral-7b` | Mistral-7B-Instruct-v0.3-Q4_K_M.gguf | 8192 | 99 | 8.4 | 4.4 GB |

Speeds are baseline `llama-cli -no-cnv -st -n 16` from sweep on 2026-04-28 (see `claudedocs/max_model_size_per_device_20260428.md`).

## Per-model command (pick the right one for your goal)

### Fastest single model (most fluent within budget)
```bash
./chat.sh qwen3-1.7b      # 19.5 tok/s — small but Qwen3-tier reasoning
./chat.sh phi-3.5-mini    # 12.3 tok/s — best smaller model on this device
./chat.sh qwen3-30b-a3b   # 13.5 tok/s — best large-quality + fast (MoE)
```

### Highest-quality usable
```bash
./chat.sh qwen3.5-35b-a3b  # 9.6 tok/s — Qwen3.5 Medium, MoE-3B-active
./chat.sh qwen3-30b-a3b    # 13.5 tok/s — Qwen3 (slightly older) MoE
```

### Cross-family compare (same param class)
```bash
./chat.sh qwen3-8b        # 7.5 tok/s
./chat.sh llama-3.1-8b    # 8.0 tok/s
./chat.sh mistral-7b      # 8.4 tok/s
./chat.sh gemma-2-9b      # 6.4 tok/s
```

### The extreme — 122B-A10B (slow but works)
```bash
./chat.sh qwen3.5-122b-a10b   # 1.9 tok/s, takes ~2 min to load
```
Uses partial GPU offload (`-ngl 30`) because full offload OOMs the 62 GB CUDA cap. Splits 44 GB GPU + 29 GB host RAM (Tegra unified-memory trick).

## Slash commands inside chat

| Command | Effect |
|---|---|
| `/exit` or `Ctrl+C` | Quit |
| `/clear` | Clear conversation history |
| `/regen` | Regenerate last assistant response |
| `/read <path>` | Inject text file into conversation |
| `/glob <pat>` | Inject all files matching pattern |
| `Enter` (twice on blank) | End multi-line input |

## Passing extra `llama-cli` flags

Anything after the key is passed through:

```bash
./chat.sh qwen3-8b --temp 0.7 --top-p 0.9 --repeat-penalty 1.1 --seed 42
./chat.sh qwen3-8b -p "Explain TCP handshake" -n 256 -no-cnv -st   # one-shot
./chat.sh qwen3-8b --system-prompt "You are a terse expert."
```

## Speed-test mode (no chat)

```bash
./chat.sh --bench qwen3-30b-a3b
# runs:  llama-cli ... -p "The capital of France is" -n 32 -no-cnv -st
# prints prompt and generation tok/s
```

## Memory considerations

- **Qwen3 default context = 40K tokens** → KV cache balloons fast. The `-c` values in `chat.sh` (2048-8192) are tuned for safety on 64 GB unified memory.
- For Qwen3-32B, raising `-c` above ~4K can OOM the GPU. Watch with:
  ```bash
  watch -n 1 'tegrastats | head -1'
  ```

## Where everything lives on Orin

```
/tmp/llama.cpp-build/build-cuda/bin/llama-cli      # CUDA chat binary
/tmp/llama.cpp-build/build-cuda/bin/llama-quantize # quantization tool
/tmp/llama.cpp-build/build-cpu/bin/llama-cli       # CPU-only fallback
~/models/                                           # all GGUF files
~/chat.sh                                           # this helper
~/build_llama_orin.sh                               # rebuild script
~/convert_to_gguf.sh                                # safetensors → GGUF
```

## Reproduce the helper

The full script is in this repo: `scripts/chat_orin.sh` (mirror of `~/chat.sh`).
Re-deploy with:

```bash
scp scripts/chat_orin.sh home.orin.local:~/chat.sh
ssh home.orin.local 'chmod +x ~/chat.sh'
```

Sources for measured speeds:
- [Max-size measurement results](max_model_size_per_device_20260428.md)
- [Qwen3 / Qwen3.5 candidate inventory](qwen3_qwen35_model_candidates.md)
