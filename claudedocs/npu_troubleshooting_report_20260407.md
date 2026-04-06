# QNN NPU Troubleshooting Report

**Date**: 2026-04-07
**Devices**: OnePlus 13 (CPH2645, SM8650/V75), OnePlus 11 (CPH2451, SM8550/V73)
**QAIRT SDK**: 2.42.0.251225 (Docker 내장) + 2.39.0.250926 (로컬)
**llama.cpp fork**: chraac/llama-cpp-qnn-builder (ggml-qnn backend)

---

## 1. 증상

llama.cpp의 QNN 빌드(`GGML_QNN=ON`)로 `-ngl 99`를 지정하여 모든 레이어를 GPU/NPU에 오프로드 시도했으나, **실제로는 모든 연산이 CPU에서 실행됨.**

관측된 성능:
| Target | tok/s | 비교 기준 (CPU-only 빌드) |
|--------|-------|--------------------------|
| OP13 "NPU" (QNN 빌드) | 3.1 | OP13 CPU: 13.3 → **4.3x 느림** |
| OP11 "NPU" (QNN 빌드) | 3.0 | OP11 CPU: 9.9 → **3.3x 느림** |

→ "NPU"가 CPU보다 느린 것은 비정상. QNN backend overhead만 추가되고 실제 가속 없음.

---

## 2. 검증 방법

### 2.1 CDSP Clock Monitoring

참고: [hmx_hvx_profiling_guide_20260329.md](../../claudedocs/hmx_hvx_profiling_guide_20260329.md)

NPU(Hexagon DSP)가 실제 사용되면 CDSP clock이 idle(384MHz)에서 활성(806~1382MHz)으로 점프해야 함.

```bash
# debugfs 마운트 (OP11은 필요, OP13은 자동)
adb -s db151d78 shell "su -c 'mount -t debugfs debugfs /sys/kernel/debug'"

# CDSP stats 읽기
adb shell "su -c 'cat /sys/kernel/debug/sysmon_subsystem_stats/master_cdsp_stats'"
```

**결과:**

| 상태 | OP13 (V75) CDSP Clock | OP11 (V73) CDSP Clock | 정상 동작 시 (참고) |
|------|----------------------|----------------------|-------------------|
| IDLE | 384,000 KHz | 460,800 KHz | 384,000 KHz |
| CPU 추론 중 | 384,000 KHz | 460,800 KHz | 384,000 KHz |
| **QNN "NPU" 추론 중** | **384,000 KHz** | **460,800 KHz** | **1,382,400 KHz** |

→ **CDSP clock 변화 없음 = NPU 미사용 확인** ❌

### 2.2 GPU Busy Monitoring

QNN-GPU 백엔드가 활성화되었다면 GPU busy %가 증가해야 함.

```bash
# GPU busy ratio (active cycles / total cycles)
adb -s 85ea76a2 shell "su -c 'cat /sys/class/kgsl/kgsl-3d0/gpubusy'"
```

**결과:** QNN 추론 중 GPU busy ≈ 0% → **GPU도 미사용** ❌

### 2.3 Verbose 로그 분석

```bash
adb shell "... ./llm-profiler-qnn ... --verbose 2>&1" | grep -E "buffer size|cannot|qnn"
```

**핵심 로그 (Q4_K_M 모델):**
```
load_tensors: tensor 'token_embd.weight' (q4_K) (and 197 others) 
              cannot be used with preferred buffer type qnn-npu, using CPU instead
load_tensors:   CPU_Mapped model buffer size =  1059.88 MiB
load_tensors:      qnn-gpu model buffer size =     0.55 MiB
```

**핵심 로그 (Q8_0 모델):**
```
load_tensors: tensor 'token_embd.weight' (q8_0) (and 197 others) 
              cannot be used with preferred buffer type qnn-npu, using CPU instead
load_tensors:   CPU_Mapped model buffer size =  1801.09 MiB
load_tensors:      qnn-gpu model buffer size =     0.55 MiB
~llama_context:    qnn-npu compute buffer size is   0.0000 MiB
```

→ 198개 텐서 전부 NPU 거부 → CPU fallback

### 2.4 HMX/HVX Stats (OP11 V73만 가능)

```bash
# OP11에서 HMX/HVX 읽기 (V73은 busy 상태에서도 읽힘)
adb -s db151d78 shell "su -c 'cat /sys/kernel/debug/sysmon_subsystem_stats/master_cdsp_stats'"
```

QNN 추론 중: `HMX utilization = 0%`, `HVX utilization = 0%` → **NPU 미사용** ❌

**참고**: OP13(V75)은 DSP busy 상태에서 `error code: 1` 반환 — V75 프로덕션 펌웨어 제한. 상세: [hmx_hvx_profiling_troubleshooting_20260402.md](../../claudedocs/hmx_hvx_profiling_troubleshooting_20260402.md)

---

## 3. 원인 분석

### 3.1 1차 원인: Q4_K_M 양자화 포맷 비호환

QNN 백엔드의 `type_to_npu_type()` (npu/host/util.cpp)에서 지원하는 타입:
```cpp
case GGML_TYPE_F32:   → NPU_DATA_TYPE_FP32
case GGML_TYPE_F16:   → NPU_DATA_TYPE_FP16
case GGML_TYPE_I32:   → NPU_DATA_TYPE_I32
case GGML_TYPE_I64:   → NPU_DATA_TYPE_I64
case GGML_TYPE_Q4_K:  → NPU_DATA_TYPE_Q4_K  // 코드에는 있으나...
case GGML_TYPE_Q4_0:  → NPU_DATA_TYPE_Q4_0
case GGML_TYPE_Q8_0:  → NPU_DATA_TYPE_Q8_0
```

Q4_K는 리스트에 있지만, Q4_K_**M** (mixed precision k-quant)은 `GGML_TYPE_Q4_K`와 다를 수 있음. 어쨌든 **buffer type 체크에서 먼저 거부됨** (3.2 참조).

### 3.2 근본 원인: Buffer Type 비호환 (핵심)

`npu_device::supports_buft()` (npu/host/host_device.cpp:133):
```cpp
bool npu_device::supports_buft(ggml_backend_buffer_type_t buft) const {
    return buft && buft->device && buft->device->context == this;
}
```

**이 함수는 NPU 자체가 할당한 버퍼만 수용.** GGUF 파일에서 mmap으로 로드된 텐서는 CPU 호스트 버퍼에 있으므로 `buft->device->context != this` → **전부 거부.**

이것은 양자화 포맷과 무관한 **구조적 문제**:
- GGUF 로드 시 텐서 → CPU mmap 버퍼
- llama.cpp가 `supports_buft(cpu_buft)` 호출 → NPU가 false 반환
- 모든 텐서가 CPU fallback
- Q8_0, Q4_0, F16 어느 포맷이든 **동일하게 거부됨**

### 3.3 QNN GPU도 동일 문제

QNN GPU 백엔드도 동일한 `supports_buft()` 패턴:
- 모델 가중치 1059~1801MB가 `CPU_Mapped`에 할당
- `qnn-gpu`에는 0.55MB만 (KV cache 관련 메타데이터 추정)
- GPU compute buffer: 0MB
- 실제 matmul은 CPU에서 실행

### 3.4 395 Graph Splits의 의미

```
llama_context: graph splits = 1 (with bs=512), 395 (with bs=1)
```

decode 시 395개 split = 거의 모든 노드에서 백엔드 전환 시도. 하지만:
- 가중치가 CPU에 있으므로 실제 전환은 일어나지 않음
- 전환 시도 자체의 오버헤드가 추가됨
- CPU-only 빌드 대비 **순수 오버헤드만 증가** (3~4x 느림)

---

## 4. chraac/llama-cpp-qnn 코드 구조 분석

```
ggml/src/ggml-qnn/
├── qnn/                    # QNN SDK 기반 GPU 백엔드
│   ├── qnn-backend.cpp     # QNN GPU graph execution
│   ├── qnn-lib.hpp         # QNN SDK wrapper
│   ├── graph.cpp           # QNN graph construction
│   └── utils.cpp           # type conversion (GGML → QNN)
│
└── npu/                    # Hexagon FastRPC 기반 NPU 백엔드
    ├── host/               # 호스트(CPU) 사이드 코드
    │   ├── host_device.cpp # supports_buft(), supports_op()
    │   ├── buffer.cpp      # NPU 메모리 할당 (rpcmem)
    │   ├── graph.cpp       # NPU graph execution
    │   └── tensor.hpp      # NPU 텐서 관리
    │
    └── device/             # DSP 사이드 코드 (Hexagon에서 실행)
        ├── op/             # NPU 연산 구현 (mul_mat, rope, flash_attn)
        ├── type_traits.cpp # Q8_0/Q4_0 양자화/역양자화
        └── dma_transfer.hpp# DMA 전송
```

**NPU 백엔드 아키텍처:**
```
Host (ARM CPU)                    DSP (Hexagon NPU)
┌──────────────┐                  ┌──────────────┐
│ host_device  │  ── FastRPC ──>  │ device ops   │
│ buffer.cpp   │  ── rpcmem ──>  │ mul_mat      │
│ graph.cpp    │  ── dspqueue ─> │ rope, attn   │
└──────────────┘                  └──────────────┘
```

**문제 지점**: `host_device.cpp`의 `supports_buft()`가 GGUF mmap 버퍼를 거부하므로, graph.cpp까지 도달하지 못함. NPU device 코드는 실행되지 않음.

---

## 5. QNN GPU 실제 동작 여부 추가 검증

OpenCL 빌드 (`GGML_OPENCL=ON`)는 **GPU에서 실제로 동작함** (20.9 tok/s). QNN GPU 빌드 (`GGML_QNN=ON`)는 동작하지 않음. 차이:

| | OpenCL 빌드 | QNN GPU 빌드 |
|---|---|---|
| 라이브러리 | libOpenCL.so (디바이스 내장) | libQnnGpu.so (QAIRT SDK) |
| 버퍼 관리 | OpenCL buffer (clCreateBuffer) | QNN buffer (QnnMem) |
| 텐서 복사 | ggml이 OpenCL 버퍼로 복사 | supports_buft() 거부 → 복사 안 됨 |
| GPU 실행 | ✅ Adreno 커널 실행 | ❌ CPU fallback |
| tok/s | 20.9 | ~2.6 (CPU + overhead) |

→ **QNN 백엔드의 버퍼 관리 코드가 llama.cpp의 텐서 로딩 파이프라인과 호환되지 않음**

---

## 6. 재현 방법

### 6.1 Q4_K_M 모델 (실패)
```bash
adb shell "cd /data/local/tmp && \
    LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64 \
    ./llm-profiler-qnn -m qwen2.5-1.5b-instruct-q4_k_m.gguf \
    -p 'Hello' -n 4 -t 4 -ngl 99 -fit off --verbose \
    --profiler-output /dev/null 2>&1" | grep -E "cannot|buffer size|qnn"
```

### 6.2 Q8_0 모델 (실패)
```bash
adb shell "cd /data/local/tmp && \
    LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64 \
    ./llm-profiler-qnn -m qwen2.5-1.5b-instruct-q8_0.gguf \
    -p 'Hello' -n 4 -t 4 -ngl 99 -fit off --verbose \
    --profiler-output /dev/null 2>&1" | grep -E "cannot|buffer size|qnn"
```

### 6.3 CDSP 모니터링
```bash
# IDLE
adb shell "su -c 'cat /sys/kernel/debug/sysmon_subsystem_stats/master_cdsp_stats'" | head -5

# QNN 추론 중 (별도 터미널에서 추론 실행 후)
adb shell "su -c 'cat /sys/kernel/debug/sysmon_subsystem_stats/master_cdsp_stats'" | head -5
# Core clock이 384000에서 변하지 않으면 NPU 미사용
```

---

## 7. 관련 이슈 및 레퍼런스

| 이슈 | 링크 | 상태 |
|------|------|------|
| QNN Backend PR | [ggml-org/llama.cpp#12063](https://github.com/ggml-org/llama.cpp/pull/12063) | WIP/Draft |
| Hexagon Backend PR | [ggml-org/llama.cpp#12326](https://github.com/ggml-org/llama.cpp/pull/12326) | Research |
| Hexagon 성능 이슈 | [ggml-org/llama.cpp#18139](https://github.com/ggml-org/llama.cpp/issues/18139) | Open |
| chraac QNN Builder | [github.com/chraac/llama-cpp-qnn-builder](https://github.com/chraac/llama-cpp-qnn-builder) | Active |
| MLC-LLM NPU 요청 | [mlc-ai/mlc-llm#1689](https://github.com/mlc-ai/mlc-llm/issues/1689) | Open |

---

## 8. 결론

**llama.cpp의 QNN/NPU 백엔드(chraac fork)는 현재 GGUF 모델을 NPU/GPU에서 실행할 수 없음.**
- 양자화 포맷 문제가 아닌, 버퍼 할당 구조의 근본적 비호환
- QNN 빌드는 CPU-only 빌드 대비 **순수 오버헤드만 추가** (3~4x 느림)
- 모든 "NPU" 프로파일링 데이터는 실제로 CPU 실행 → `op13_qnn_cpu_fallback`으로 재분류 완료
