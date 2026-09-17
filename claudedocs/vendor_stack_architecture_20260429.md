# Vendor Stack Architecture — ggml / llama.cpp / llama-cpp-qnn

**Date**: 2026-04-29
**Scope**: Inventory, per-project internal architecture, and integration map of the three upstream C/C++ projects this profiler depends on.
**Audience**: Anyone joining the project who needs to know what `vendor/` contains, how each component is internally organized, and where `tiny-llm-profiler` plugs in.

---

## 1. Vendor Inventory

### 1.1 Current state of `vendor/`

| Path | Upstream | Pinned commit | Date | Form | Tracked in parent repo? |
|---|---|---|---|---|---|
| `vendor/ggml/` | `ggml-org/ggml` | `387fa29f` (v0.10.1) | 2026-04-29 | Standalone shallow clone | **No** (in `.gitignore`) |
| `vendor/llama.cpp/` | `ggml-org/llama.cpp` | `4aa962e2` | 2026-04-06 | Standalone shallow clone | **No** (in `.gitignore`) |
| `vendor/llama-cpp-qnn/` | `chraac/llama-cpp-qnn-builder` | `826a383e` (top) / `897501a7` (inner llama.cpp) | 2026-02-07 / 2025-12-26 | Standalone shallow clone | **No** (in `.gitignore`) |
| `vendor/qairt/` | Qualcomm QAIRT SDK | n/a (binary) | — | Extracted SDK files | No (license) |
| `vendor/opencl/` | Khronos / Adreno OpenCL SDK | n/a (binary) | — | Extracted SDK files | No |

### 1.2 Critical observations

1. **Not git submodules.** None of the three repos is registered as a submodule — there is no `.gitmodules` file, and `git ls-files vendor/` returns nothing. Each is a standalone shallow clone whose `.git` directory exists locally but is invisible to the parent repo.
2. **Three copies of ggml exist** because both `llama.cpp` and `llama-cpp-qnn` vendor a copy internally. Each copy is at a different point in time:
   - `vendor/ggml/src/ggml.c` — 7,760 lines, v0.10.1, 2026-04-29
   - `vendor/llama.cpp/ggml/src/ggml.c` — 7,738 lines, ~2026-04-06
   - `vendor/llama-cpp-qnn/llama.cpp/ggml/src/ggml.c` — older, ~2025-12-25
   The ABI is the same family but op enum order and minor backend interface details can drift. This is the root cause of occasional op-id-to-name mismatches in `profiler_wrapper.py`.
3. **Reproducibility hole in Docker builds.** Half of the `build/docker/Dockerfile.*` files do `COPY vendor/llama.cpp /build/llama.cpp` (uses the local pin), but the other half do `RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git` inside the container (pulls **today's** master). The README claims "build b1-4aa962e", but that statement is only true for the `COPY`-based builds.

| Dockerfile | Source of llama.cpp | Pinned? |
|---|---|---|
| `Dockerfile.aarch64-cpu` | `COPY vendor/llama.cpp` | yes |
| `Dockerfile.aarch64-native` | `git clone` inside container | **no** |
| `Dockerfile.android-ndk` | `git clone` inside container | **no** |
| `Dockerfile.android-opencl` | `git clone` + `COPY vendor/{opencl,qairt}` | partial |
| `Dockerfile.android-qnn` | `COPY vendor/llama-cpp-qnn/llama.cpp/` | yes |
| `Dockerfile.orin-cuda` | `git clone` inside container | **no** |

---

## 2. ggml — `vendor/ggml/`

### 2.1 What it is

A pure tensor / computation-graph library. It does **not** know about LLMs, GGUF, sampling, or KV caches. It exposes a uniform op set (`enum ggml_op`, 41+ ops) and a backend abstraction (`ggml_backend_t`) so that compute can run on CPU, CUDA, Metal, OpenCL, Vulkan, Hexagon NPU, etc., from the same graph.

### 2.2 Top-level layout

```
vendor/ggml/
├── CMakeLists.txt              # version 0.10.1, ~50 GGML_* options
├── ggml.pc.in                  # pkg-config template
├── README.md, LICENSE, AUTHORS
├── requirements.txt            # for examples/python
│
├── include/                     # 21 public headers — this is the API surface
│   ├── ggml.h                  # core: tensor, graph, op enum (★ canonical truth)
│   ├── ggml-alloc.h            # graph memory planner
│   ├── ggml-backend.h          # backend abstraction (★ pluggable layer)
│   ├── ggml-cpp.h              # C++ wrappers
│   ├── gguf.h                  # GGUF file format
│   ├── ggml-opt.h              # training / optimization
│   └── ggml-{cpu,cuda,metal,opencl,vulkan,hexagon,rpc,
│              blas,sycl,cann,musa,hip,openvino,virtgpu,
│              webgpu,zdnn,zendnn}.h    # per-backend init
│
├── src/                         # implementation
│   ├── ggml.c                   # ★ op kernels, tensor lifecycle (~7,760 lines)
│   ├── ggml.cpp                 # C++ glue
│   ├── gguf.cpp                 # GGUF reader/writer
│   ├── ggml-alloc.c             # graph memory planning
│   ├── ggml-quants.{c,h}        # Q4_0 / Q4_K / Q8_0 (de)quantization
│   ├── ggml-backend.cpp         # backend dispatcher
│   ├── ggml-backend-impl.h      # vtable each backend must implement
│   ├── ggml-backend-reg.cpp     # registry / device enumeration
│   ├── ggml-backend-dl.{cpp,h}  # dynamic loading of backend .so
│   ├── ggml-backend-meta.cpp    # device metadata
│   ├── ggml-threading.{cpp,h}   # thread pool
│   ├── ggml-opt.cpp             # optimizer (training)
│   │
│   └── (one subdirectory per backend, each implements the vtable)
│       ggml-cpu/      — CPU SIMD: AVX*, NEON, AMX, RVV, VSX, ...
│       ggml-cuda/     — NVIDIA CUDA
│       ggml-metal/    — Apple Metal
│       ggml-opencl/   — Adreno / Mali OpenCL
│       ggml-vulkan/   — Khronos Vulkan
│       ggml-hexagon/  — Qualcomm Hexagon NPU (FastRPC)
│       ggml-sycl/     — Intel SYCL
│       ggml-cann/     — Huawei Ascend
│       ggml-rpc/      — remote backend (distributed)
│       ggml-blas/     — CPU BLAS acceleration
│       ggml-musa, -hip, -openvino, -virtgpu, -zdnn, -zendnn,
│       ggml-webgpu/
│
├── examples/                    # standalone demos (NOT used by tiny-llm-profiler)
│   ├── gpt-2, gpt-j             # historical LLM demos pre-llama.cpp
│   ├── sam, yolo, magika, mnist # vision/classification
│   ├── simple, perf-metal       # API tutorials
│   └── python/                  # Python bindings & tests
│
├── tests/                       # 22 unit tests for the library itself
│   ├── test-backend-ops.cpp     # ★ comprehensive op correctness across backends
│   ├── test-quantize-{fns,perf}.cpp
│   └── test-{conv,pool,opt,arange,...}.cpp
│
├── docs/gguf.md                 # GGUF format spec
├── ci/                          # GitHub Actions / CI scripts
└── scripts/sync-llama.sh        # how upstream syncs ggml ↔ llama.cpp
```

### 2.3 Internal architecture — three layers

```
┌────────── User C/C++ code ──────────────────────────┐
│  ggml_init()                                         │
│  ggml_new_tensor / ggml_mul_mat / ggml_rope / ...    │
│  ggml_build_forward_expand()                         │
│  ggml_backend_sched_graph_compute()                  │
└──────────────┬───────────────────────────────────────┘
               │
┌──────────────▼ Front-end (single shared graph) ─────┐
│  struct ggml_tensor { op, ne[4], src[…], data, … }   │
│  struct ggml_cgraph { nodes[], leafs[], … }          │
│  enum ggml_op       — 41+ ops (MUL_MAT, ROPE, ADD…)  │
│  enum ggml_type     — F32/F16/Q4_K/Q8_0/…            │
└──────────────┬───────────────────────────────────────┘
               │
┌──────────────▼ Scheduler / dispatcher ──────────────┐
│  ggml_backend_sched_t                                │
│   • assigns each graph node to a backend             │
│   • inserts cross-backend tensor copies              │
│   • invokes the eval-callback hook ★ (cb_eval)       │
└─────┬──────────┬──────────┬──────────┬──────────────┘
      │          │          │          │
   ┌──▼─┐    ┌───▼──┐   ┌───▼──┐   ┌───▼───┐
   │CPU │    │ CUDA │   │Metal │   │OpenCL │  ... + Vulkan, Hexagon, ...
   └────┘    └──────┘   └──────┘   └───────┘
   Each backend:
     • implements ggml_backend_i vtable in ggml-backend-impl.h
     • declares which ops + dtypes it supports (`supports_op`)
     • declares which buffer types it accepts (`supports_buft`) ← NPU mismatch lives here
     • exposes a public init function (e.g. ggml_backend_cuda_init)
```

### 2.4 Key properties relevant to this project

- **`cb_eval` is invoked by `ggml_backend_sched`**, not by llama.cpp. That is why `tiny-llm-profiler` can profile ggml ops without modifying llama.cpp at all.
- **The `enum ggml_op` order is the source of truth**. `tiny-llm-profiler/collection/track1_ggml/profiler_wrapper.py:30-95` mirrors this list as `_OP_NAMES`. Whenever this enum changes upstream, that table goes stale.
- **`supports_buft()` mismatches** are the root cause of the QNN NPU failure documented in `claudedocs/npu_troubleshooting_report_20260407.md`.

---

## 3. llama.cpp — `vendor/llama.cpp/`

### 3.1 What it is

LLM serving runtime. It vendors ggml internally and adds: GGUF model loading, model architectures (Llama, Qwen, Gemma, Mistral, RWKV, Mamba, …), graph builders, KV cache, sampling, tokenization, chat templates, and a fleet of CLI tools.

### 3.2 Top-level layout

```
vendor/llama.cpp/
├── CMakeLists.txt              # entry; LLAMA_BUILD_{COMMON,TOOLS,EXAMPLES,SERVER,TESTS}
├── CMakePresets.json
├── Makefile                    # legacy
├── flake.nix, flake.lock       # Nix
├── pyproject.toml              # for the gguf-py package
├── poetry.lock, mypy.ini
├── README.md, AGENTS.md, CLAUDE.md, AUTHORS, CODEOWNERS, SECURITY.md
│
├── include/llama.h             # ★ public C API — 203 LLAMA_API functions
├── include/llama-cpp.h         # C++ thin wrapper
│
├── ggml/                       # ★ vendored copy of ggml (see §2)
│   ├── include/, src/, cmake/, CMakeLists.txt
│   └── (this is what links into libllama; vendor/ggml/ is NOT used here)
│
├── src/                        # ★ libllama (52 .{cpp,h}) — LLM-specific layer
│   ├── llama.cpp                       # llama_decode, llama_init_from_model
│   ├── llama-arch.{cpp,h}              # arch enum + tensor-name patterns
│   │                                   # (mirrored by our tensor_name_parser.py)
│   ├── llama-model.{cpp,h}             # weights + hparams
│   ├── llama-model-loader.{cpp,h}      # GGUF → tensor mapping
│   ├── llama-model-saver.{cpp,h}       # GGUF writer
│   ├── llama-graph.{cpp,h}             # arch → ggml_cgraph builder ★
│   ├── llama-context.{cpp,h}           # runtime container ★
│   │                                   #   (KV cache + sched + sampler)
│   ├── llama-cparams.{cpp,h}           # ctx options
│   ├── llama-batch.{cpp,h}             # input batching
│   ├── llama-hparams.{cpp,h}           # hyperparameters
│   ├── llama-impl.{cpp,h}              # internal helpers
│   ├── llama-io.{cpp,h}                # state save/load
│   ├── llama-mmap.{cpp,h}              # memory-mapped weight loading
│   │
│   ├── llama-kv-cache.{cpp,h}          # standard KV cache
│   ├── llama-kv-cache-iswa.{cpp,h}     # interleaved sliding window (Gemma)
│   ├── llama-kv-cells.h                # cell granularity
│   ├── llama-memory.{cpp,h}            # base abstraction
│   ├── llama-memory-hybrid.{cpp,h}     # hybrid (Mamba + attention)
│   ├── llama-memory-hybrid-iswa.{cpp,h}
│   ├── llama-memory-recurrent.{cpp,h}  # RWKV / Mamba state
│   │
│   ├── llama-vocab.{cpp,h}             # tokenizer dispatch
│   ├── unicode.cpp, unicode-data.cpp   # text normalization
│   ├── llama-sampler.{cpp,h}           # temp / top-p / mirostat / penalties
│   ├── llama-grammar.{cpp,h}           # GBNF constrained decoding
│   ├── llama-chat.{cpp,h}              # chat templates per arch
│   ├── llama-adapter.{cpp,h}           # LoRA adapters
│   ├── llama-quant.{cpp,h}             # quantization driver (used by tools/quantize)
│   └── llama-ext.h
│
├── common/                     # libcommon — used by tools, not by libllama
│   ├── arg.{cpp,h}             # CLI argument parser ★ (used by llm_profiler_main.cpp)
│   ├── common.{cpp,h}          # common_init_from_params, system info
│   ├── chat.{cpp,h}, chat-*    # chat template parsing
│   ├── log.{cpp,h}             # logger
│   ├── console.{cpp,h}         # interactive console
│   ├── debug.{cpp,h}           # debug helpers
│   ├── base64.hpp              # b64 utilities
│   └── build-info.cpp.in       # generated build metadata
│
├── tools/                      # ★ executables (where we patch in llm-profiler)
│   ├── cli/                    → llama-cli           (interactive / one-shot)
│   ├── server/                 → llama-server        (OpenAI-compat HTTP)
│   ├── quantize/               → llama-quantize
│   ├── bench/                  → llama-bench
│   ├── llama-bench/            → vendored bench
│   ├── batched-bench/          → batched throughput
│   ├── perplexity/             → PPL evaluation
│   ├── imatrix/                → importance matrix
│   ├── tts/                    → text-to-speech
│   ├── mtmd/                   → multimodal (vision)
│   ├── tokenize/               → tokenizer utility
│   ├── gguf-split/             → split large GGUF
│   ├── completion/             → completion test
│   ├── cvector-generator/      → control vectors
│   ├── export-lora/            → LoRA export
│   ├── fit-params/             → param fitting
│   ├── parser/                 → grammar parser
│   ├── rpc/                    → rpc-server
│   ├── results/                → result collection
│   └── (CMakeLists.txt picks them based on options)
│
├── examples/                   # ~30 demos — embedding, batched, training, etc.
│   ├── simple, simple-chat, simple-cmake-pkg
│   ├── batched, batched.swift
│   ├── speculative, speculative-simple, lookup, lookahead
│   ├── embedding, retrieval, passkey
│   ├── eval-callback                 # ← demonstrates cb_eval, similar to ours
│   ├── llama.android, llama.swiftui, llama.vim
│   ├── training, sycl, parallel
│   ├── convert_legacy_llama.py
│   ├── convert-llama2c-to-ggml/
│   └── (Python utils for grammar / json schema)
│
├── gguf-py/                    # standalone Python GGUF library
├── grammars/                   # GBNF samples
├── docs/                       # build, backend, architecture docs
├── ci/                         # GitHub Actions
├── cmake/                      # CMake helpers
├── benches/                    # microbenchmarks
├── pocs/                       # proofs-of-concept
├── models/                     # reference model configs
├── media/, licenses/           # assets
│
├── convert_hf_to_gguf.py       # HF Transformers → GGUF
├── convert_hf_to_gguf_update.py
├── convert_llama_ggml_to_gguf.py
├── convert_lora_to_gguf.py
└── build-xcframework.sh        # iOS framework build
```

### 3.3 Library / executable graph

```
            (depends on)            (depends on)
  llama-cli ─────────────► libcommon ─────────► libllama ─────────► libggml
  llama-server ────────────────┘                  ▲                    ▲
  llama-quantize  ─────────────────────────────────┘                    │
  llama-bench, perplexity, … (same shape)                               │
                                                                        │
  tiny-llm-profiler/llm-profiler  ─────► libcommon ───► libllama ───────┘
       (uses ggml.h, llama.h, common.h, arg.h)
```

CMake gates this with `LLAMA_BUILD_COMMON / TOOLS / EXAMPLES / SERVER / TESTS`. We typically build with `LLAMA_BUILD_TOOLS=OFF LLAMA_BUILD_SERVER=OFF` and `add_executable(llm-profiler ...)` patched in.

### 3.4 Decode call graph (one token)

```
llama_decode(ctx, batch)
  └─ llama_context::decode()                                  [llama-context.cpp]
       ├─ llama_graph::build(arch)                            [llama-graph.cpp]
       │    └─ ggml_new_tensor / ggml_mul_mat / ggml_rope / … [ggml.c via ggml.h]
       ├─ ggml_backend_sched_alloc_graph()                    [ggml-backend.cpp]
       │    (decides per-node which backend; inserts copies)
       ├─ ggml_backend_sched_graph_compute()                  ★ cb_eval fires
       │    └─ for each node: backend->compute(t)
       │            • CPU SIMD / CUDA kernel / Metal shader / ...
       │            • cb_eval(t, ask=true)  → our timer.start()
       │            • cb_eval(t, ask=false) → our timer.stop() + JSONL emit
       ├─ llama_kv_cache::update()                            [llama-kv-cache.cpp]
       └─ logits → caller for sampling                        [llama-sampler.cpp]
```

### 3.5 What changes between llama.cpp versions

- New model architectures land as new entries in `llama-arch.cpp` + new branches in `llama-graph.cpp`.
- New ggml ops land as new entries in `enum ggml_op` (under `ggml/include/ggml.h`).
- KV-cache variants (iSWA, hybrid, recurrent) move when new arches need them.
- The `cb_eval` ABI has been stable for a long time; our profiler has not had to change.

---

## 4. llama-cpp-qnn — `vendor/llama-cpp-qnn/`

### 4.1 What it is

Third-party project that **forks llama.cpp wholesale** and adds two NPU/GPU backends to its bundled ggml: a QNN-SDK backend and a custom Hexagon-FastRPC backend. The repo itself also ships Docker compose configs and Android push-and-test scripts so the build/test loop is automated for Snapdragon devices. Maintained by `chraac`, last commit at the top level 2026-02-07.

### 4.2 Top-level layout

```
vendor/llama-cpp-qnn/
├── README.md
├── docs/
│   ├── how-to-build.md         # Android + Windows-on-ARM build instructions
│   └── hexagon-npu.md          # Architecture of the FastRPC NPU backend
│
├── docker/                     # build-in-container infrastructure
│   ├── docker-compose-compile.yml          # default build
│   ├── docker-compose-compile-qnn.yml      # +QNN SDK
│   ├── docker-compose-run.yml              # run tests
│   ├── docker_compose_compile.sh           # entrypoint
│   ├── docker_compose_run_test.sh
│   ├── build_in_container.sh               # (mounted into container)
│   └── run_in_container.sh
│
├── scripts/                    # device push/test automation
│   ├── compile.sh                          # local cmake + build
│   ├── push_and_run_test.{sh,ps1}          # adb push + on-device test
│   ├── run_all_device_tests.{sh,ps1}
│   ├── run_device_model.{sh,ps1}
│   ├── run_local_model.sh
│   ├── batch_run_benchmarks_and_save_log.{sh,ps1}
│   ├── run_benchmark_and_save_log.sh
│   ├── log_parser.py
│   ├── start_debugger.{sh,ps1}
│   ├── tests/                              # device test fixtures
│   └── requirements.txt
│
└── llama.cpp/                  # ★ FORK of llama.cpp (NOT a submodule, full copy)
    │   (mirrors §3.2 layout exactly: src/, common/, tools/, examples/, ggml/, …)
    │   pinned at ~2025-12-26 — about 4 months behind upstream
    │
    └── ggml/src/               # the divergence is concentrated here:
        ├── ggml-qnn/  ★ ENTIRELY NEW (does not exist in upstream llama.cpp)
        │   ├── CMakeLists.txt
        │   ├── qnn/                    QNN-SDK backend
        │   │   ├── ggml-qnn.cpp        ggml_backend_t implementation
        │   │   ├── backend-ops.{cpp,hpp}   op → QNN-graph lowering
        │   │   ├── qnn-lib.{cpp,hpp}   dlopen QNN .so
        │   │   ├── qnn-types.hpp
        │   │   ├── tensor.hpp          ggml ↔ QNN tensor wrapping
        │   │   ├── buffer.hpp          QNN buffer type
        │   │   ├── graph.{cpp,hpp}     QNN graph builder
        │   │   ├── op-config{,-base,-caps,-impl}.{cpp,hpp}
        │   │   │                        per-op capability + lowering
        │   │   ├── convert.{cpp,hpp}   format conversion
        │   │   ├── event_tracer.{cpp,hpp}  internal profiler
        │   │   ├── logger.{cpp,hpp}
        │   │   └── utils.{cpp,hpp}
        │   │
        │   ├── npu/                    Hexagon FastRPC backend
        │   │   │   (bypasses QNN SDK; talks to Hexagon DSP directly)
        │   │   ├── host/               ARM-side coordination
        │   │   │   ├── host_device.cpp
        │   │   │   ├── graph.cpp       graph creation, caching
        │   │   │   └── buffer.cpp      RPC + ION buffers, zero-copy
        │   │   ├── device/             runs on the Hexagon DSP
        │   │   │   ├── device.cpp      NPU-side runtime
        │   │   │   ├── graph.cpp       4-thread parallel execution
        │   │   │   ├── op_impl.cpp     hand-written HVX intrinsics
        │   │   │   ├── quants.cpp      HVX dequantization (Q4_0/Q8_0/Q4_K)
        │   │   │   ├── thread_pool.hpp QURT-based 4-thread pool
        │   │   │   └── vtcm_mem.hpp    VTCM (high-speed local mem) RAII
        │   │   └── idl/
        │   │       └── hexagon_npu.idl FastRPC interface contract
        │   │
        │   └── shared/                 cross-backend utilities
        │       ├── common.{cpp,hpp}
        │       ├── dyn-lib-loader.hpp  dlopen wrapper
        │       ├── rpc-interface.hpp   FastRPC abstraction
        │       ├── rpc-mem.hpp         RPC memory mgmt
        │       └── profiler.hpp
        │
        └── ggml-hexagon/  ⚠ MODIFIED vs upstream (same name, different content)
            ├── ggml-hexagon.cpp        (forked variant)
            ├── htp/, htp-utils.{c,h}
            ├── CMakeLists.txt
            └── (changed CMake structure per top-level commit message)
```

### 4.3 Internal architecture — host + device split

This is fundamentally different from CUDA/Metal backends, which run on a GPU controlled from the host CPU via a kernel API. The Hexagon NPU is a **separate processor with its own RTOS (QURT)**, and the FastRPC framework is what bridges them. The QNN-SDK backend hides this; the custom NPU backend exposes it.

```
                              Snapdragon SoC
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  ┌─────────── ARM (Cortex-A) — Linux/Android ──────────────┐    │
  │  │                                                          │    │
  │  │   ┌── ggml/llama.cpp graph ────────────────────────┐     │    │
  │  │   │  ggml_backend_sched picks ggml_backend_qnn_*   │     │    │
  │  │   └────────────────┬────────────────────────────────┘     │    │
  │  │                    │                                      │    │
  │  │    ┌───────────────▼───────────────┐                      │    │
  │  │    │  ggml-qnn/qnn/  (path A)      │                      │    │
  │  │    │   • QNN SDK calls             │   ─ libQnn*.so       │    │
  │  │    │   • higher-level abstraction  │     (Qualcomm vendor)│    │
  │  │    └───────────────────────────────┘                      │    │
  │  │                                                            │    │
  │  │    ┌───────────────────────────────┐                      │    │
  │  │    │  ggml-qnn/npu/host/  (path B) │                      │    │
  │  │    │   • host_device.cpp           │                      │    │
  │  │    │   • graph.cpp coordination    │                      │    │
  │  │    │   • buffer.cpp (RPC + ION)    │                      │    │
  │  │    └───────────────┬───────────────┘                      │    │
  │  │                    │ FastRPC (hexagon_npu.idl stubs)      │    │
  │  └────────────────────┼──────────────────────────────────────┘    │
  │                       │                                            │
  │  ┌────────────────────▼──────────────────────────────────────┐    │
  │  │  Hexagon DSP — QURT RTOS                                   │    │
  │  │                                                            │    │
  │  │   ┌── ggml-qnn/npu/device/ ───────────────────────────┐  │    │
  │  │   │  device.cpp        — NPU runtime entry            │  │    │
  │  │   │  graph.cpp         — 4-thread parallel exec        │  │    │
  │  │   │  op_impl.cpp       — HVX intrinsics                │  │    │
  │  │   │  quants.cpp        — HVX dequant (Q4_0/Q8_0/Q4_K)  │  │    │
  │  │   │  thread_pool.hpp   — QURT 4-thread pool            │  │    │
  │  │   │  vtcm_mem.hpp      — VTCM RAII (high-speed local)  │  │    │
  │  │   └─────────────────────────────────────────────────────┘  │    │
  │  └────────────────────────────────────────────────────────────┘    │
  └──────────────────────────────────────────────────────────────────┘
```

Path A (QNN SDK) is the conservative, vendor-supported route. Path B (NPU FastRPC) bypasses Qualcomm's QNN graph compiler entirely and ships hand-written HVX kernels — the trade-off is more performance but a much smaller op set (`mul_mat`, `add`, RMS-norm at present).

### 4.4 Build flow (their own pipeline)

```
docker/docker_compose_compile.sh
  ↓
docker-compose-compile-qnn.yml           env: TARGET_ARCH=arm64-v8a, BUILD_HEXAGON_BACKEND=…
  ↓
chraac/llama-cpp-qnn-builder:latest      pre-built x86 image w/ NDK r27c, QNN SDK, Hexagon SDK
  ↓
build_in_container.sh                    cmake llama.cpp/ → builds tools + ggml-qnn + ggml-hexagon
  ↓
build_qnn_arm64-v8a/bin/                 main, llama-cli, etc.
  ↓
scripts/push_and_run_test.sh             adb push + on-device test
```

Our project's `Dockerfile.android-qnn` skips this and instead does `FROM chraac/llama-cpp-qnn-builder:latest` then `COPY vendor/llama-cpp-qnn/llama.cpp/` to reuse the same toolchain image but build our `llm-profiler` instead.

### 4.5 Why offload currently fails on this stack

(from `claudedocs/npu_troubleshooting_report_20260407.md`)

- llama.cpp's GGUF model loader places weights into a **CPU mmap'd buffer** (`llama-mmap.cpp`).
- The QNN backend's `ggml_backend_qnn_buffer_type::supports_buft()` does **not** accept that buffer type — it expects RPC/ION buffers it allocated itself.
- The scheduler therefore cannot place tensor nodes on the QNN backend, and silently falls back to CPU.
- This is a **buffer-type compatibility issue at the ggml backend interface layer**, not a kernel bug. It is fixable but requires intercepting the loader to allocate via the QNN buffer type, or post-load tensor copy into RPC memory.

---

## 5. Combined dependency graph (how the three connect)

```
┌──────────────────────── tiny-llm-profiler ─────────────────────────┐
│                                                                     │
│  Python   analysis / storage / transport / dashboard                │
│      ▲                                                              │
│      │ JSONL via SCP / ADB                                          │
│      │                                                              │
│  ┌───┴─────────────────────────────────────────────────────────┐   │
│  │ collection/track1_ggml/llm_profiler_main.cpp  (our main)     │   │
│  │   #include "llama.h"      ← libllama                         │   │
│  │   #include "common.h"     ← libcommon (arg parser etc.)      │   │
│  │   #include "ggml.h"       ← ggml headers (cb_eval signature) │   │
│  │   params.cb_eval = profiler_callback                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│      │ link                                                         │
└──────┼──────────────────────────────────────────────────────────────┘
       │
       ▼
  Built from one of:
       │
   ┌───┴────────────────────────────────────────┐
   │                                            │
   ▼                                            ▼
┌───── vendor/llama.cpp/ ──────┐    ┌─── vendor/llama-cpp-qnn/llama.cpp/ ───┐
│ libcommon → libllama → libggml │  │ libcommon → libllama → libggml         │
│       (same shape)             │  │       + ggml-qnn (QNN SDK)             │
│                                │  │       + ggml-qnn/npu (FastRPC)         │
│ Used by:                       │  │       + modified ggml-hexagon          │
│  • Dockerfile.aarch64-cpu      │  │ Used by:                               │
│  • Dockerfile.aarch64-native*  │  │  • Dockerfile.android-qnn              │
│  • Dockerfile.android-{ndk,    │  │                                        │
│      opencl}*                  │  │  *Note: starred files clone upstream   │
│  • Dockerfile.orin-cuda*       │  │   inside the container instead of      │
│                                │  │   using the vendored copy              │
└────────────────────────────────┘  └────────────────────────────────────────┘
       │                                            │
       │ embeds, but at older commit                │ embeds older still
       │                                            │
       ▼                                            ▼
        │
        │  reference / canonical, currently
        │  not consumed by any build:
        ▼
┌───── vendor/ggml/ ───────────────┐
│  Standalone v0.10.1               │
│  examples/ (gpt-2, sam, yolo, …) │
│  tests/ (test-backend-ops, …)    │
│  → reference for the ggml_op enum│
│    and backend interface         │
└───────────────────────────────────┘
```

### 5.1 Where `tiny-llm-profiler` plugs in

```
                      stdout: per-tensor JSONL
                              ▲
                              │ profiler_callback() writes
                              │
                    ggml_backend_sched_graph_compute()
                              │ (per-node call)
                              ▼
              for each node in the graph:
                  cb_eval(t, ask=true)   ← our timer.start()
                  backend->compute(t)    ← actual op execution
                  cb_eval(t, ask=false)  ← our timer.stop()
                              │
                              ▼
                      ggml_op enum + tensor name
                              │
                              ▼
              Python:  tensor_name_parser → (block_idx, sublayer)
                       op enum             → human-readable op name
                       → ProfileRecord     → SQLite + Parquet
```

Inference is driven by llama.cpp (`llama_decode`). Measurement is driven by ggml (`ggml_backend_sched`). Backend selection is driven by the ggml backend interface. We occupy a single C callback at the seam — that minimal contact point is what makes this stack portable across vendor versions.

---

## 6. Recommendations

1. **Register all three vendor repos as git submodules.** Currently they are gitignored local clones, so any collaborator or CI runner sees an empty `vendor/`. Use the form:
   ```
   git submodule add https://github.com/ggml-org/ggml.git           vendor/ggml
   git submodule add https://github.com/ggml-org/llama.cpp.git      vendor/llama.cpp
   git submodule add https://github.com/chraac/llama-cpp-qnn-builder.git vendor/llama-cpp-qnn
   ```
   Then `cd vendor/<name> && git checkout <pinned SHA>` and `git add` the parent repo.

2. **Eliminate the `git clone` inside Dockerfiles.** Replace `RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git` in `Dockerfile.aarch64-native`, `Dockerfile.android-ndk`, `Dockerfile.android-opencl`, `Dockerfile.orin-cuda` with `COPY vendor/llama.cpp /build/llama.cpp`. This is the only way to guarantee every device builds the same llama.cpp commit.

3. **Document the three ggml copies.** Add a `vendor/VERSIONS.md` listing the three pinned SHAs and their dates so drift is visible. When Qwen3.5 / MoE op-id parsing breaks, this is the first place to look.

4. **Auto-generate `_OP_NAMES`.** Today `profiler_wrapper.py:30-95` is hand-maintained. Replace it with a script that parses `vendor/ggml/include/ggml.h`'s `enum ggml_op` and emits the Python dict. This closes the loop on op rename / addition issues seen with Qwen3.5 and MoE `mul_mat_id`.

5. **(Optional) Move from `vendor/` to `third_party/`.** The standard convention for upstream code in C/C++ projects is `third_party/`. Either rename now (one diff, all Dockerfile `COPY` paths change) or accept `vendor/` as the project's local convention.
