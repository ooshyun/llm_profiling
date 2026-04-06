# llama.cpp Architecture & Flow Reference

> Date: 2026-04-05
> Purpose: tiny-llm-profiler 프로젝트를 위한 llama.cpp 내부 아키텍처 참조 문서

---

## 1. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    USER TOOLS LAYER                           │
│  llama-cli  |  llama-server  |  llama-bench  |  custom apps  │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│              APPLICATION LAYER (libllama)                      │
│                                                                │
│  llama_model        llama_context       llama_sampler          │
│  (weights, vocab,   (KV cache, batch,   (temperature,         │
│   hparams, arch)    graph build/exec)   top-k/p, grammar)    │
│                                                                │
│  llama_vocab        llama_batch         llama_memory_i         │
│  (BPE/SPM/WPM      (input batching,    (KV cache interface,  │
│   tokenizer)        ubatch splitting)   slot management)      │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│              COMPUTATION LAYER (ggml)                          │
│                                                                │
│  ggml_tensor       ggml_cgraph          ggml_backend_sched    │
│  (N-dim array,     (DAG of ops,         (multi-backend        │
│   quantized types) nodes + leafs)       scheduler)            │
│                                                                │
│  ggml_context      ggml_gallocr         ggml_backend_i        │
│  (metadata arena)  (graph memory        (CPU, CUDA, Metal,    │
│                     allocator)          Vulkan, OpenCL, QNN)  │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Key Source Files

| File | Role |
|------|------|
| `include/llama.h` | Public C API: model/context/sampler lifecycle |
| `src/llama-model.cpp` | Model loading, weight management, GGUF parsing |
| `src/llama-model-loader.cpp` | GGUF file parser, tensor indexing, mmap |
| `src/llama-context.cpp` | Inference state: KV cache, batching, graph build, backend sched |
| `src/llama-vocab.cpp` | Tokenizer (BPE, SentencePiece, WordPiece, RWKV) |
| `src/llama-sampling.cpp` | Sampler interface + chain + all sampler implementations |
| `src/llama-kv-cache.h/.cpp` | KV cache: cell management, slot allocation, defrag |
| `src/llama-arch.cpp` | Architecture registry (`LLM_ARCH_NAMES`), tensor name maps |
| `src/llama-batch.cpp` | Batch allocation, ubatch splitting |
| `ggml/include/ggml.h` | Core tensor library: types, operations, graph construction |
| `ggml/include/gguf.h` | GGUF file format reader/writer |
| `ggml/include/ggml-backend.h` | Backend interface, scheduler, buffer management |
| `ggml/src/ggml-alloc.c` | Graph allocator (ggml_gallocr): liveness analysis, buffer reuse |
| `ggml/src/ggml-backend.cpp` | Backend scheduler implementation |
| `ggml/src/ggml-backend-reg.cpp` | Backend registry: discovery, dynamic loading |
| `ggml/src/ggml-cpu/` | CPU backend (AVX, NEON, SVE, i8mm) |
| `ggml/src/ggml-cuda/` | CUDA backend (NVIDIA GPUs) |
| `ggml/src/ggml-vulkan/` | Vulkan backend (cross-platform GPU) |
| `ggml/src/ggml-metal/` | Metal backend (Apple Silicon) |
| `src/llama-mmap.cpp` | Platform-specific mmap/mlock wrappers |

---

## 3. End-to-End Inference Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. MODEL LOADING                                         │
│    llama_model_load(path, params)                        │
│    ├─ Open GGUF file                                     │
│    ├─ Parse header + metadata (arch, hparams, vocab)     │
│    ├─ Detect architecture (LLM_ARCH_NAMES map)           │
│    ├─ mmap() file (zero-copy, OS manages paging)         │
│    ├─ Create ggml_tensor headers → mmap region           │
│    ├─ Determine device placement (n_gpu_layers)          │
│    ├─ Allocate backend buffers for GPU layers            │
│    └─ Copy weights to GPU buffers as needed              │
├─────────────────────────────────────────────────────────┤
│ 2. CONTEXT CREATION                                      │
│    llama_context_new(model, ctx_params)                  │
│    ├─ Initialize backend scheduler (ggml_backend_sched)  │
│    ├─ Register eval callback (cb_eval) ← PROFILER HOOK  │
│    ├─ Allocate KV cache (per-layer K,V for n_ctx)        │
│    ├─ Reserve compute buffers (measure graph → allocate) │
│    └─ Initialize threading (n_threads, n_threads_batch)  │
├─────────────────────────────────────────────────────────┤
│ 3. TOKENIZATION                                          │
│    llama_tokenize(model, text, tokens)                   │
│    ├─ Detect tokenizer type from GGUF metadata           │
│    ├─ Split text → iteratively merge pairs (BPE)         │
│    ├─ Prepend BOS token if needed                        │
│    └─ Return token ID array                              │
├─────────────────────────────────────────────────────────┤
│ 4. INFERENCE LOOP                                        │
│                                                           │
│  ┌─── PREFILL (first call, n_tokens = prompt) ────┐     │
│  │ OR                                               │     │
│  └─── DECODE  (subsequent, n_tokens = 1) ─────────┘     │
│                                                           │
│    llama_decode(ctx, batch)                               │
│    ├─ Split into ubatches if needed                      │
│    ├─ FOR each ubatch:                                   │
│    │   ├─ KV cache: init_batch() + find_slot()           │
│    │   ├─ Build compute graph (Section 4)                │
│    │   ├─ Set input tensors (token IDs, positions)       │
│    │   ├─ ggml_backend_sched_graph_compute() (Section 5) │
│    │   └─ Extract logits from output tensor              │
│    └─ Return logits                                      │
├─────────────────────────────────────────────────────────┤
│ 5. SAMPLING                                              │
│    llama_sampler_sample(chain, ctx, idx)                 │
│    ├─ Retrieve logits[idx]                               │
│    ├─ Apply chain: penalties → temp → top-k → top-p     │
│    │   → min-p → grammar → greedy/dist                  │
│    └─ Return selected token ID                           │
├─────────────────────────────────────────────────────────┤
│ 6. OUTPUT                                                │
│    llama_token_to_piece(token) → string                  │
│    Check EOS → loop back to step 4                       │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Compute Graph (ggml_cgraph) — LLM Forward Pass

### 4.1 Graph Structure

```c
struct ggml_cgraph {
    struct ggml_tensor ** nodes;   // operation tensors (topological order)
    struct ggml_tensor ** leafs;   // input/weight tensors (no src[])
    int n_nodes;                   // ~487 for LLaMA-7B
    int n_leafs;                   // ~188 for LLaMA-7B
};
```

### 4.2 Per-Layer Graph (Transformer Block)

```
Token Embeddings (get_rows)
    │
    ▼
FOR each layer il = 0..N-1:
    │
    ├─── RMS Norm (attn_norm)
    │        │
    │        ▼
    │    ┌─ Q = matmul(wq, cur) ─┐
    │    │  K = matmul(wk, cur)  │  Q/K/V Projection
    │    │  V = matmul(wv, cur)  │
    │    └───────────┬───────────┘
    │                │
    │        ┌───────▼────────┐
    │        │  RoPE(Q), RoPE(K)  │  Positional Encoding
    │        └───────┬────────┘
    │                │
    │        ┌───────▼────────┐
    │        │  KV Cache:          │
    │        │  Store K,V at pos   │  Cache Management
    │        │  Read full K,V      │
    │        └───────┬────────┘
    │                │
    │        ┌───────▼────────┐
    │        │  KQ = Q @ K^T       │
    │        │  KQ /= sqrt(d_k)    │  Attention Score
    │        │  KQ = mask(KQ)      │
    │        │  KQ = softmax(KQ)   │
    │        └───────┬────────┘
    │                │
    │        ┌───────▼────────┐
    │        │  KQV = KQ @ V       │  Attention Output
    │        │  out = matmul(wo, KQV)│
    │        └───────┬────────┘
    │                │
    │        ┌───────▼────────┐
    │        │  cur = residual + out│  Residual Add
    │        └───────┬────────┘
    │                │
    ├─── RMS Norm (ffn_norm)
    │        │
    │        ▼
    │    ┌─ gate = silu(matmul(w_gate, cur)) ─┐
    │    │  up   = matmul(w_up, cur)           │  FFN (SwiGLU)
    │    │  down = matmul(w_down, gate * up)   │
    │    └─────────────┬───────────────────────┘
    │                  │
    │        ┌─────────▼────────┐
    │        │  cur = residual + down│  Residual Add
    │        └─────────┬────────┘
    │                  │
    ▼                  ▼
Final RMS Norm → matmul(output_weight, cur) → logits [n_vocab]
```

### 4.3 Tensor Naming Convention (profiler 매핑 핵심)

```
blk.{layer_idx}.{component}.weight

Examples:
  blk.0.attn_norm.weight     → block=0, sublayer=attn_norm
  blk.0.attn_q.weight        → block=0, sublayer=attention_qkv (Q)
  blk.0.attn_k.weight        → block=0, sublayer=attention_qkv (K)
  blk.0.attn_v.weight        → block=0, sublayer=attention_qkv (V)
  blk.0.attn_output.weight   → block=0, sublayer=attention_out
  blk.0.ffn_norm.weight      → block=0, sublayer=ffn_norm
  blk.0.ffn_gate.weight      → block=0, sublayer=ffn_gate
  blk.0.ffn_up.weight        → block=0, sublayer=ffn_up
  blk.0.ffn_down.weight      → block=0, sublayer=ffn_down
  output_norm.weight          → block=-1, sublayer=output_norm
  output.weight               → block=-1, sublayer=lm_head
  token_embd.weight           → block=-1, sublayer=embedding
```

### 4.4 Prefill vs Decode

| Aspect | Prefill | Decode |
|--------|---------|--------|
| Input tokens | N (full prompt) | 1 |
| Q/K/V compute | All N tokens at once | Latest token only |
| Attention matrix | N x N | 1 x (n_past+1) |
| Dominant operation | GEMM (compute-bound) | GEMV (memory-bound) |
| KV cache action | Store all N K,V | Append 1 K,V pair |
| Graph rebuilt? | Yes | Yes (view sizes change with n_past) |

---

## 5. Backend Scheduler (ggml_backend_sched)

### 5.1 Available Backends

| Backend | Hardware | Build Flag |
|---------|----------|------------|
| CPU | x86 (AVX), ARM (NEON/SVE/i8mm) | `GGML_CPU=ON` |
| CUDA | NVIDIA GPUs | `GGML_CUDA=ON` |
| Metal | Apple Silicon | `GGML_METAL=ON` |
| Vulkan | Cross-platform GPU | `GGML_VULKAN=ON` |
| HIP/ROCm | AMD GPUs | `GGML_HIP=ON` |
| SYCL | Intel GPUs | `GGML_SYCL=ON` |
| CANN | Huawei Ascend NPUs | `GGML_CANN=ON` |
| RPC | Remote backends | built-in |

### 5.2 Graph Compute Flow

```
ggml_backend_sched_graph_compute(sched, graph)
│
├─ 1. SPLIT GRAPH (5-pass algorithm)
│   ├─ Pass 1: Assign by buffer location (weight on GPU → GPU)
│   ├─ Pass 2: Propagate GPU assignment to connected ops
│   ├─ Pass 3: Upgrade to higher-priority backends
│   ├─ Pass 4: Assign remaining unassigned sources
│   └─ Pass 5: Detect backend transitions → create splits
│       Result: [Split 0: GPU nodes 0-50]
│               [Split 1: CPU nodes 51-55]
│               [Split 2: GPU nodes 56-487]
│
├─ 2. ALLOCATE GRAPH
│   └─ ggml_gallocr: liveness analysis → buffer reuse
│       (non-overlapping lifetimes share memory)
│
└─ 3. COMPUTE SPLITS
    │
    FOR each split:
    │
    ├─ Copy input tensors across backends (async or sync)
    │
    ├─ IF no cb_eval:
    │   └─ backend->graph_compute(subgraph)  ← 전체 split 한번에
    │
    └─ IF cb_eval SET:                        ← PROFILER PATH
        FOR each node in split:
        ├─ cb_eval(tensor, ask=true)          ← 실행 전 콜백
        ├─ backend->graph_compute(node)       ← 개별 노드 실행
        ├─ ggml_backend_synchronize()         ← 동기화 (타이밍 정확)
        └─ cb_eval(tensor, ask=false)         ← 실행 후 콜백
```

### 5.3 Debugging

```bash
GGML_SCHED_DEBUG=1 ./llama-cli -m model.gguf ...
# → split 경계, 텐서 복사, 백엔드 할당 상세 출력
```

---

## 6. Memory Management

### 6.1 Memory Hierarchy

```
ggml_context (metadata arena, bump allocator)
    │ tensor->data points to:
    ▼
ggml_backend_buffer (device memory: CPU heap / CUDA device / ...)
    │ allocated by:
    ▼
ggml_gallocr (graph allocator)
    ├─ Liveness analysis on ggml_cgraph
    ├─ Dead tensor memory reuse (50%+ reduction)
    └─ Offsets computed in preprocessing (not at exec time)
```

### 6.2 Model Loading (mmap)

```
GGUF File on Disk
    │
    ├─ mmap() → Virtual address space (zero-copy)
    │   ├─ tensor->data points directly into mmap region
    │   ├─ OS kernel manages paging transparently
    │   └─ Nearly instant "loading" (pages faulted on demand)
    │
    └─ GPU offload:
        ├─ Allocate ggml_backend_buffer on GPU
        ├─ Copy weights from mmap → GPU buffer
        └─ tensor->data updated to GPU address
```

### 6.3 KV Cache

```
Pre-allocated for full context window:
  Per layer: K = [n_ctx x n_embd_k_gqa]
             V = [n_ctx x n_embd_v_gqa]

Prefill (n_past=0):
  find_slot() → cells [0..n_tokens-1]
  Store K,V for ALL input tokens

Decode (n_past > 0):
  find_slot() → cell [n_past]
  Store K,V for ONLY latest token
  Attention reads [0..n_past] from cache

Quantized KV: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
```

### 6.4 Runtime Memory Categories

| Category | Managed By | Persistence | Example |
|----------|-----------|-------------|---------|
| Model weights | llama_model + mmap | Entire session | Token embeddings, layer weights |
| KV cache | llama_memory_i | Across tokens | Per-layer K,V tensors |
| Compute buffers | ggml_gallocr | Per llama_decode() | Intermediate activations (reused) |
| Metadata | ggml_context | Per graph build | Tensor headers, graph structure |

---

## 7. Eval Callback — Profiler Hook Point

### 7.1 Registration

```c
// llama.h
struct llama_context_params {
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};

// 또는 직접:
ggml_backend_sched_set_eval_callback(ctx->sched, callback, user_data);
```

### 7.2 Callback Signature

```c
typedef bool (*ggml_backend_sched_eval_callback)(
    struct ggml_tensor * t,    // 현재 텐서 노드
    bool ask,                   // true=실행 전, false=실행 후
    void * user_data
);
```

### 7.3 Accessible Data in Callback

```c
// 실행 전 (ask=true):
t->name          // "blk.5.attn_q.weight" → 레이어 매핑
t->op            // GGML_OP_MUL_MAT → op 타입
t->type          // GGML_TYPE_Q4_K_M → 양자화 타입
t->ne[4]         // dimensions → FLOPS 계산 (2*M*N*K for matmul)
t->nb[4]         // strides
t->src[0..N]     // input tensors
ggml_nbytes(t)   // 텐서 크기 (bytes)
ggml_nelements(t) // 총 원소 수

// 실행 후 (ask=false):
// 위 모든 것 + 계산 결과 데이터
ggml_backend_tensor_get(t, buf, 0, n) // GPU→CPU 데이터 복사 (선택적)
```

### 7.4 Profiler에서의 활용 (tiny-llm-profiler)

```c
static bool profiling_callback(struct ggml_tensor * t, bool ask, void * ud) {
    profiler_state_t * s = (profiler_state_t *)ud;
    
    if (ask) {
        s->cur.name       = t->name;
        s->cur.op         = t->op;
        s->cur.bytes      = ggml_nbytes(t);
        s->cur.flops      = compute_flops(t);  // 텐서 shape 기반
        s->cur.start_us   = ggml_time_us();
        return true;
    } else {
        s->cur.end_us     = ggml_time_us();
        s->cur.latency_us = s->cur.end_us - s->cur.start_us;
        parse_layer_info(s->cur.name, &s->cur.block_idx, &s->cur.sublayer);
        emit_record(&s->cur);  // → JSON/MQTT
        return true;
    }
}
```

### 7.5 Callback Execution Context

```
ggml_backend_sched_compute_splits()
│
FOR each split:
│   ├─ Copy inputs across backends
│   │
│   └─ FOR each node in split:        ← cb_eval 활성화 시
│       ├─ cb_eval(t, ask=TRUE)        ← ggml_time_us() 시작
│       ├─ backend->graph_compute(t)   ← 실제 계산
│       ├─ ggml_backend_synchronize()  ← GPU 동기화 (타이밍 보장)
│       └─ cb_eval(t, ask=FALSE)       ← ggml_time_us() 종료
│
│   동기화가 강제되므로 per-node 벽시계 시간 정확
│   오버헤드: ~10-30% (상대 비율은 유효)
```

---

## 8. GGUF File Format

```
┌──────────────────────────────────┐
│ MAGIC + VERSION                  │
├──────────────────────────────────┤
│ METADATA (key-value)             │
│  general.architecture = "llama"  │
│  llama.block_count = 32          │
│  llama.embedding_length = 4096   │
│  tokenizer.ggml.model = "llama"  │
│  tokenizer.ggml.tokens = [...]   │
├──────────────────────────────────┤
│ TENSOR DESCRIPTORS               │
│  name: "blk.0.attn_q.weight"    │
│  shape: [4096, 4096]             │
│  type: Q4_K_M                    │
│  offset: 0x1000                  │
│  ...                             │
├──────────────────────────────────┤
│ TENSOR DATA (bulk binary)        │
│  [quantized weight bytes]        │
│  (mmap-able: OS pages directly)  │
└──────────────────────────────────┘
```

---

## 9. Sampler Chain

```
logits [n_vocab]
    │
    ▼
┌─ Sampler Chain ─────────────────────┐
│  1. Penalties (repetition/presence) │
│  2. Temperature (logits /= T)       │
│  3. Softmax → probabilities         │
│  4. Top-K (keep K highest)          │
│  5. Top-P / Nucleus (cumprob >= p)  │
│  6. Min-P (prob < min_p * max)      │
│  7. Grammar (valid parse only)      │
│  8. Selection: greedy OR dist       │
└──────────────┬──────────────────────┘
               │
               ▼
         Selected token ID
```

---

## References

- [llama.cpp GitHub Repository](https://github.com/ggml-org/llama.cpp)
- [DeepWiki: llama.cpp Architecture](https://deepwiki.com/ggml-org/llama.cpp)
- [DeepWiki: GGML Tensor Library](https://deepwiki.com/ggml-org/llama.cpp/4-ggml-tensor-library)
- [Understanding LLM Inference with llama.cpp](https://www.omrimallis.com/posts/understanding-how-llm-inference-works-with-llama-cpp/)
- [llama.cpp Architecture (Mintlify)](https://www.mintlify.com/ggml-org/llama.cpp/concepts/architecture)
- [Profile llama.cpp with Arm Streamline](https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama_cpp_streamline/2_llama.cpp_intro/)
- [Optimizing llama.cpp with CUDA Graphs (NVIDIA)](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/)
- [ggml-backend.h — Scheduler API](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-backend.h)
- [Backend Scheduler Discussion #10182](https://github.com/ggml-org/llama.cpp/discussions/10182)
- [Graph Rebuild Discussion #4617](https://github.com/ggml-org/llama.cpp/discussions/4617)
- [Performance Profiling Discussion #6871](https://github.com/ggml-org/llama.cpp/discussions/6871)
