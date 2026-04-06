# Track 1 & Track 2 Feasibility Report

> Date: 2026-04-05
> Status: Confirmed
> Purpose: 투트랙 프로파일링 전략의 실현 가능성 심층 검증

---

## Track 1: llama.cpp (ggml) + QNN NPU — Feasibility: HIGH ✅

### 1.1 핵심 API: `ggml_backend_sched_set_eval_callback()`

**존재 확인됨.** `ggml/include/ggml-backend.h`에 정의.

```c
typedef bool (*ggml_backend_sched_eval_callback)(
    struct ggml_tensor * t,    // 현재 계산 중인 텐서 노드
    bool ask,                  // true="계산할까요?", false="계산 완료"
    void * user_data
);

GGML_API void ggml_backend_sched_set_eval_callback(
    ggml_backend_sched_t sched,
    ggml_backend_sched_eval_callback callback,
    void * user_data
);
```

`llama_context_params`에서 직접 등록 가능:

```c
struct llama_context_params {
    // ...
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};
```

### 1.2 동작 메커니즘

`ggml_backend_sched_compute_splits()` 내부에서 콜백 설정 시:

```c
if (sched->callback_eval) {
    for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
        struct ggml_tensor * t = split->graph.nodes[j0];
        
        // ASK phase: 실행 전
        bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);
        
        // 노드 개별 계산 + 동기화
        ggml_backend_graph_compute_async(split_backend, &split->graph);
        ggml_backend_synchronize(split_backend);
        
        // NOTIFY phase: 실행 후
        if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
            break;
        }
    }
}
```

**핵심 특성:**
- 모든 텐서 노드를 개별 인터셉트 (ask=true → 계산 → 동기화 → ask=false)
- 콜백 내에서 `t->name`, `t->op`, `t->type`, `t->ne[]`, `t->nb[]`, `t->src[]` 접근 가능
- `ggml_nbytes(t)`로 텐서 크기, `ggml_backend_tensor_get()`으로 데이터 readback 가능
- 텐서 이름이 `blk.{N}.{component}` 패턴 → 레이어 매핑 가능

### 1.3 프로파일링 코드 예시 (검증됨)

llama.cpp의 `examples/eval-callback/` 디렉토리에 실제 동작하는 예시 존재:

```c
static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    if (ask) {
        return true;  // 항상 계산 후 통지 요청
    }

    // 실행 완료 후: t->data에 계산 결과 존재
    // GPU 텐서인 경우 호스트로 복사:
    if (!ggml_backend_buffer_is_host(t->buffer)) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }
    return true;
}
```

### 1.4 메트릭별 수집 가능성

| 메트릭 | 수집 방법 | 정확도 | 비고 |
|--------|---------|--------|------|
| **(A) Latency** | `ggml_time_us()` ask=true/false 사이 | HIGH | 동기화 강제로 벽시계 시간 정확 |
| **(B) Memory** | `ggml_nbytes(t)` per tensor | MEDIUM | 근사치. `ggml_gallocr` 내부 API 미공개 |
| **(B) Memory breakdown** | `llama_memory_breakdown_print()` | HIGH | 디바이스별 model/context/compute 분류 |
| **(C) Power** | Platform Monitor (별도 스레드, 시간 기반 조인) | MEDIUM | 직접 per-layer가 아닌 시간 구간 매핑 |
| **(D) FLOPS** | 텐서 shape에서 계산 (matmul: 2×M×N×K) | HIGH | 정확한 이론적 연산량 |
| **(E) Bandwidth** | `ggml_nbytes(src+dst)` / latency | MEDIUM | 추정치 |
| **(F) Cache hit** | Linux `perf_event` (별도 수집) | MEDIUM | 플랫폼 의존, per-layer 근사 |
| **(G) Thermal** | Platform Monitor (시간 기반 조인) | MEDIUM | 100ms 샘플링 |
| **(I) Op type** | `t->op` enum 직접 접근 | HIGH | ggml_op enum으로 정확 분류 |
| **(J) GPU SM util** | Orin: tegrastats/Nsight (별도) | MEDIUM | 시간 구간 매핑 |

### 1.5 기타 ggml 프로파일링 도구

| 도구 | 용도 | 한계 |
|------|------|------|
| `llama_perf_context()` | 집계 타이밍 (총 prefill/decode 시간) | per-layer 불가 |
| `test-backend-ops perf` | 개별 op 마이크로벤치마크 (합성 텐서) | 실제 추론 시 동작 아님 |
| `GGML_SCHED_DEBUG=1/2` | 그래프 분할 디버깅 | 타이밍 없음 |
| OpenCL 백엔드 프로파일링 (PR #12442) | per-kernel 타이밍, Chrome trace 출력 | OpenCL 백엔드 전용 |

### 1.6 QNN/Hexagon NPU 프로파일링

| 구현 | 상태 | 접근법 |
|------|------|--------|
| **ggml-qnn** (PR #12063, chraac) | WIP/Draft | QNN SDK 통한 고수준 API |
| **ggml-hexagon** (PR #12326, zhouwg) | 리서치 프로토타입 | FastRPC + HVX 직접 프로그래밍 |
| `enable_perf = 1` 플래그 | PR #12326에 존재 | 기본적 타이밍 제공 |

**QNN NPU 제한사항:**
- QNN SDK는 **블랙박스** — 그래프 제출 후 결과 반환. 내부 per-op 분해 불가
- 4B 이하 모델만 (cDSP 32-bit 주소공간 한계)
- Hexagon 백엔드가 QNN SDK 대비 성능 미달 ([Issue #18139](https://github.com/ggml-org/llama.cpp/issues/18139))
- ggml 레벨에서 호출 전후 시간은 측정 가능

### 1.7 알려진 제한사항

| 제한 | 심각도 | 대응 방안 |
|------|--------|---------|
| 동기화 강제로 절대 시간 10-30% 왜곡 | MEDIUM | 상대 비율(% per layer) 유효. 콜백 유/무 비교로 오버헤드 보정 가능 |
| GPU 텐서 데이터 readback 비용 | LOW | 타이밍만 수집 시 readback 불필요 |
| QNN NPU 내부 불투명 | HIGH | ggml 레벨 타이밍만. 논문에서 "experimental, opaque NPU" 표기 |
| `ggml_gallocr` 메모리 per-node API 미공개 | MEDIUM | `ggml_nbytes(t)` 근사치 사용 |
| GGML_PERF 제거됨 (PR #8017) | N/A | cb_eval 콜백이 대체 메커니즘 |

### 1.8 플랫폼별 검증 상태

| 기기 | cb_eval 동작 | GPU 백엔드 | NPU 백엔드 |
|------|------------|-----------|-----------|
| RPi 4 (CPU) | ✅ 검증됨 | N/A | N/A |
| Orin CPU | ✅ 검증됨 | N/A | N/A |
| Orin GPU (CUDA) | ✅ 검증됨 | CUDA 동기화 | N/A |
| Android CPU | ✅ 검증됨 | N/A | N/A |
| Android GPU (OpenCL) | ✅ 검증됨 | OpenCL (Adreno 최적화, PR #12442) | N/A |
| Android NPU | ⚠️ 실험적 | N/A | QNN WIP (PR #12063, #12326) |
| Pi Zero 2 W (CPU) | ✅ 예상됨 (ARM NEON) | N/A | N/A |
| ESP32-S3 | ✅ llama2.c 커스텀 | N/A | N/A |

---

## Track 2: MLC-LLM (Apache TVM) — Feasibility: MEDIUM ⚠️

### 2.1 TVM 프로파일링 API (프레임워크 자체는 강력)

#### `tvm.runtime.profiling.Report`

```python
# per-kernel 타이밍 자동 수집
report.calls        # 각 커널 호출별 메트릭 (이름, 시간, 디바이스, 인자 형태)
report.table()      # Name | Duration (us) | Percent | Device | Count
report.csv()        # CSV 출력
report.json()       # JSON 직렬화/역직렬화
```

#### MetricCollector 구현체

| Collector | 측정 항목 | 백엔드 | 소스 |
|-----------|---------|--------|------|
| Default Timer | 벽시계 시간 via `StreamSync` | 전체 | `src/runtime/profiling.cc` |
| `CUDATimerNode` | GPU 시간 via `cudaEvent_t` | CUDA | `src/runtime/cuda/cuda_device_api.cc` |
| `OpenCLTimerNode` | GPU 시간 via `clGetEventProfilingInfo` | OpenCL | `src/runtime/opencl/opencl_device_api.cc` |
| `PAPIMetricCollectorNode` | HW 카운터 (cycles, cache, FLOPS) | CPU, CUDA | `src/runtime/contrib/papi/papi.cc` |

PAPI 활성화:
```bash
export TVM_PAPI_CPU_METRICS="PAPI_FP_OPS;PAPI_L1_DCM;PAPI_TOT_CYC"
```

#### Relax VM 프로파일링 (일반 모델에서 동작)

```python
vm = relax.VirtualMachine(ex, dev, profile=True)
report = vm.profile("forward", input_data)
print(report.table())
```

#### Vulkan 타이머 부재

Vulkan 백엔드에는 전용 타이머 미구현 — `DefaultTimerNode` (CPU-side `StreamSync` + `std::steady_clock`)으로 폴백. 동기화 오버헤드로 정확도 저하.

### 2.2 치명적 문제: MLC-LLM에서 VM Profiler 고장 ❌

**`vm.profile()`이 MLC-LLM의 LLM 추론에서 실패합니다.**

- **원인**: `mlc.create_paged_kv_cache_generic` 런타임 함수가 프로파일링 컴파일 경로에서 발견되지 않음
- **상태**: TVM Discuss 포럼에서 **미해결** (2025년 1월~)
- **Android**: "missing events" 보고됨

**레퍼런스:**
- [How to do kernel level profiling for LLMs using MLC-LLM (TVM Discuss, 미해결)](https://discuss.tvm.apache.org/t/how-to-do-kernel-level-profiling-for-llms-using-mlc-llm/18029)
- [MLC-LLM VM Profiler — partial success on M1, missing events on Android (TVM Discuss)](https://discuss.tvm.apache.org/t/mlc-llm-vm-profiler/15614)

### 2.3 RPi CPU 지원 불가 ❌

MLC-LLM은 Raspberry Pi (ARM CPU only) 환경을 **공식 지원하지 않습니다.**

- 메인테이너 직접 답변: *"MLC is not able to work on both linux-aarch64 and linux-x86, pure CPU"*
- CPU 지원은 *"not announced as officially supported yet"*

**레퍼런스:**
- [Does MLC support CPU or not? (GitHub Issue #795)](https://github.com/mlc-ai/mlc-llm/issues/795)

### 2.4 EventTraceRecorder

`mlc-llm/cpp/serve/event_trace_recorder.h`에 존재:

```cpp
class EventTraceRecorder : public ObjectRef {
public:
    static EventTraceRecorder Create();
    void AddEvent(const String& request_id, const std::string& event);
    std::string DumpJSON();  // Chrome Trace Event Format
};
```

**제한**: 서빙 엔진의 요청 레벨 이벤트만 기록 (prefill/decode 시작/종료). **per-kernel GPU 타이밍은 기록하지 않음.**

### 2.5 커널→레이어 매핑 문제

MLC-LLM 커널 이름에 레이어 인덱스 **미포함**:
```
fused_dequantize1_NT_matmul5        ← 어떤 레이어?
fused_dequantize2_NT_matmul11       ← 어떤 레이어?
softmax_with_chunked_sum
```

매핑 구축 방법:
1. `debug_dump` 활성화 → 컴파일 단계별 IR 스냅샷 저장 (`debug-phase0.py` ~ `debug-phase3.py`)
2. IR 분석으로 Relax 함수 → TIR 커널 매핑 테이블 구축
3. 또는 lm-Meter 논문의 커널 복제 기법 사용 (~3500 LOC)

### 2.6 우회 방법

| 방법 | 대상 | 노력 | 정확도 | 비고 |
|------|------|------|--------|------|
| Nsight Systems `nsys profile` | Orin (CUDA) | 낮음 | 높음 | 커널 타임라인 자동 수집. 커널→레이어 매핑 필요 |
| OpenCL 이벤트 수정 (`clGetEventProfilingInfo`) | Android (OpenCL) | 높음 (~3500 LOC) | 96-99% | lm-Meter 논문에서 검증 |
| `debug_dump` + 외부 프로파일러 | 전체 | 중간 | 높음 | 컴파일 시 매핑 → 런타임 프로파일러 매칭 |
| EventTraceRecorder | 전체 | 중간 | 낮음 | 요청 레벨만, per-kernel 아님 |

**lm-Meter 레퍼런스**: [lm-Meter: Unveiling Runtime Inference Latency for On-Device LLMs (SEC'25)](https://arxiv.org/html/2510.06126v1)

### 2.7 플랫폼 지원 현실

| 기기 | MLC-LLM 지원 | 프로파일링 가능 | 증거 |
|------|-------------|---------------|------|
| **Jetson Orin (CUDA)** | ✅ 확인됨 | ⚠️ Nsight로 우회 | [Seeed Wiki 가이드](https://wiki.seeedstudio.com/Quantized_Llama2_7B_with_MLC_LLM_on_Jetson/), jetson-containers 이미지 |
| **Android (OpenCL)** | ✅ 확인됨 | ⚠️ OpenCL 이벤트 수정 필요 | [MLC 공식 블로그](https://blog.mlc.ai/2023/05/08/bringing-hardware-accelerated-language-models-to-android-devices), [Callstack 프로파일링](https://www.callstack.com/blog/profiling-mlc-llms-opencl-backend-on-android-performance-insights) |
| **Android (Vulkan)** | ⚠️ 실험적 | ⚠️ 전용 타이머 없음 | [Issue #3372](https://github.com/mlc-ai/mlc-llm/issues/3372) |
| **RPi 4 (CPU)** | ⚠️ 초기 지원 (미검증) | ⚠️ 직접 검증 필요 | [PR #3116](https://github.com/mlc-ai/mlc-llm/pull/3116), [Issue #795](https://github.com/mlc-ai/mlc-llm/issues/795) — 아래 2.8 참조 |
| **Pi Zero 2 W** | ⚠️ 시도 가능 (고위험) | ⚠️ 직접 검증 필요 | 512MB RAM 제약. CPU 백엔드 동작 시 시도 가능 |
| **ESP32-S3** | ❌ 불가 | ❌ | MCU 환경 미지원 |

### 2.8 CPU 지원 최신 상태 (2025.02 업데이트)

**PR #3116이 2025년 2월 3일에 머지되어 CPU 백엔드 초기 지원이 추가되었습니다.**

| 시점 | 상태 | 출처 |
|------|------|------|
| 2023.08 | CPU 미지원. *"not announced as officially supported"* | [Issue #795](https://github.com/mlc-ai/mlc-llm/issues/795) |
| 2024.10 | *"we require a GPU to run models"* | [Issue #2927](https://github.com/mlc-ai/mlc-llm/issues/2927) (메인테이너 @MasterJH5574) |
| **2025.02.03** | **PR #3116 머지 — CPU 백엔드 초기 지원** (KV Cache TIR for CPU) | [PR #3116](https://github.com/mlc-ai/mlc-llm/pull/3116) (@mengshyu, reviewed by @MasterJH5574) |
| 2025.02 | *"Initial CPU support has already been merged. Next, we will be working on expanding model coverage and optimizing performance"* | [Issue #2927 코멘트](https://github.com/mlc-ai/mlc-llm/issues/2927) |
| 2025.11 | Android CPU 지원 문의 — 메인테이너 응답 없음 | [Issue #3372](https://github.com/mlc-ai/mlc-llm/issues/3372) |
| 2025.02~2026.04 | CPU 관련 후속 커밋 **없음** (14개월간 모델 추가, TVM API 마이그레이션만) | GitHub commits 분석 |

**현실 평가:**
- CPU 백엔드 초기 지원이 머지됨 (KV Cache TIR)
- 그러나 14개월간 후속 개발 없음 — "expanding model coverage" 약속 미이행
- ARM aarch64 / RPi에서 동작하는지 검증 사례 없음
- LLVM 타겟 `llvm -device=arm_cpu -mtriple=aarch64-linux-gnu`로 컴파일은 가능할 것으로 예상

**프로젝트 방침: RPi CPU 포함하되, 직접 검증/해결 필요. 동작 실패 시 Track 1로 폴백.**

---

## 투트랙 종합 비교

### 기기별 커버리지 매트릭스

| 기기 | Track 1 (ggml) 프로파일링 | Track 2 (TVM) 프로파일링 |
|------|-------------------------|------------------------|
| RPi 4 (CPU) | ✅ cb_eval | ⚠️ CPU 백엔드 초기 지원 (미검증, 직접 해결) |
| Orin CPU | ✅ cb_eval | ⚠️ CPU 백엔드 초기 지원 (미검증, 직접 해결) |
| Orin GPU (CUDA) | ✅ cb_eval + CUDA sync | ⚠️ Nsight 우회 |
| Android CPU | ✅ cb_eval | ⚠️ CPU 백엔드 초기 지원 (미검증, 직접 해결) |
| Android GPU (OpenCL) | ✅ cb_eval + OpenCL | ⚠️ OpenCL 이벤트 수정 |
| Android NPU (QNN) | ⚠️ 실험적 (블랙박스) | ❌ NPU 미지원 |
| Pi Zero 2 W (CPU) | ✅ cb_eval | ⚠️ 고위험 (512MB + 미검증 CPU 백엔드) |
| ESP32-S3 (MCU) | ✅ llama2.c 커스텀 | ❌ 불가 |

### 결론

- **Track 1 (llama.cpp)이 메인 트랙** — 전 기기 커버, `cb_eval` 메커니즘 검증됨
- **Track 2 (MLC-LLM)는 전 기기 시도하되 CPU는 직접 검증 필요** — Orin GPU + Android GPU는 확인됨, CPU 백엔드는 PR #3116 기반으로 직접 검증/해결
- Track 2의 가치: TVM 컴파일러 최적화(fusion 등) 후의 성능 특성을 비교할 수 있음 — ggml과 다른 최적화 전략이 레이어별 성능에 어떤 차이를 만드는지 분석
- CPU 백엔드 실패 시: 해당 기기는 Track 1만 사용 (폴백 전략)

---

## References

### Track 1 (llama.cpp/ggml)
- [ggml-backend.h — callback API definition (Fossies)](https://fossies.org/linux/llama.cpp/ggml/include/ggml-backend.h)
- [eval-callback example (GitLab mirror)](https://gitlab.informatik.uni-halle.de/ambcj/llama.cpp/-/blob/b3010/examples/eval-callback/eval-callback.cpp)
- [Performance profiling suggestions — Discussion #6871](https://github.com/ggml-org/llama.cpp/discussions/6871)
- [Precise decoding progress reporting — Discussion #8051](https://github.com/ggml-org/llama.cpp/discussions/8051)
- [OpenCL Profiling PR #12442 (Chrome trace format)](https://github.com/ggml-org/llama.cpp/pull/12442)
- [QNN Backend PR #12063 (chraac)](https://github.com/ggml-org/llama.cpp/pull/12063)
- [QNN/Hexagon Backend PR #12326 (zhouwg)](https://github.com/ggml-org/llama.cpp/pull/12326)
- [Hexagon backend performance issue #18139](https://github.com/ggml-org/llama.cpp/issues/18139)
- [GGML Deep Dive — Memory Management](https://medium.com/@yifeiw203/ggml-deep-dive-ii-memory-management-in-context-only-mode-part-1-8397a1055363)
- [Memory Optimization — DeepWiki](https://deepwiki.com/ggml-org/llama.cpp/3.11-memory-optimization-and-llama_params_fit)
- [Performance Tuning — llama.cpp docs](https://www.mintlify.com/ggml-org/llama.cpp/advanced/performance-tuning)
- [Backend scheduler discussion #10182](https://github.com/ggml-org/llama.cpp/discussions/10182)

### Track 2 (MLC-LLM/TVM)
- [tvm.runtime.profiling API docs](https://tvm.apache.org/docs/reference/api/python/runtime/profiling.html)
- [TVM Profiling C++ Doxygen](https://tvm.apache.org/docs/reference/api/doxygen/namespacetvm_1_1runtime_1_1profiling.html)
- [**MLC-LLM VM Profiler 고장 — kernel profiling 미해결 (TVM Discuss)**](https://discuss.tvm.apache.org/t/how-to-do-kernel-level-profiling-for-llms-using-mlc-llm/18029)
- [**MLC-LLM VM Profiler — missing events on Android (TVM Discuss)**](https://discuss.tvm.apache.org/t/mlc-llm-vm-profiler/15614)
- [**MLC-LLM CPU 미지원 — Issue #795**](https://github.com/mlc-ai/mlc-llm/issues/795)
- [Profiling MLC-LLM's OpenCL Backend on Android (Callstack)](https://www.callstack.com/blog/profiling-mlc-llms-opencl-backend-on-android-performance-insights)
- [lm-Meter: Unveiling Runtime Inference Latency for On-Device LLMs (SEC'25)](https://arxiv.org/html/2510.06126v1)
- [PAPI MetricCollector PR #7983](https://github.com/apache/tvm/pull/7983)
- [Device-specific timers PR #7472](https://github.com/apache/tvm/pull/7472)
- [Profiling interface for VM and Graph Runtime PR #7624](https://github.com/apache/tvm/pull/7624)
- [Quantized Llama2-7B with MLC LLM on Jetson (Seeed Wiki)](https://wiki.seeedstudio.com/Quantized_Llama2_7B_with_MLC_LLM_on_Jetson/)
- [MLC-LLM Android OpenCL blog post](https://blog.mlc.ai/2023/05/08/bringing-hardware-accelerated-language-models-to-android-devices)
- [MLC-LLM Android Vulkan Issue #3372](https://github.com/mlc-ai/mlc-llm/issues/3372)
- [MLC-LLM Compiler Pass Pipeline (DeepWiki)](https://deepwiki.com/mlc-ai/mlc-llm/4.2-installation-and-tvm-integration)
- [Relax VM API (TVM Unity docs)](https://mlc.ai/docs/reference/api/runtime/relax_vm.html)
