# NPU 프로파일링 — 추가 개발 계획

**Date**: 2026-04-07
**Context**: llama.cpp QNN 백엔드의 NPU 실행 실패 후, 실제 NPU 프로파일링을 달성하기 위한 대안 경로

---

## 1. 현재 상태

### 동작하는 것 (7 targets)

| Device | Compute | 빌드 | 프로파일링 | 실제 성능 (cb_eval OFF) |
|--------|---------|------|----------|----------------------|
| RPi4 | CPU (A72) | Docker arm64 | ✅ cb_eval | 3.1 tok/s |
| Orin | CPU (A78AE) | Docker JetPack | ✅ cb_eval | 7.0 tok/s |
| Orin | GPU (CUDA) | Docker JetPack | ✅ cb_eval | 20.9 tok/s |
| OP13 | CPU (SD8G3) | Docker NDK | ✅ cb_eval | 5.7 tok/s |
| OP13 | GPU (Adreno OpenCL) | Docker NDK+OpenCL | ✅ cb_eval | 20.9 tok/s |
| OP11 | CPU (SD8G2) | Docker NDK | ✅ cb_eval | 9.2 tok/s |
| OP11 | GPU (Adreno OpenCL) | Docker NDK+OpenCL | ✅ cb_eval | ~20 tok/s (추정) |

### 동작하지 않는 것

| Device | Compute | 문제 | 근본 원인 |
|--------|---------|------|----------|
| OP13 | NPU (Hexagon V75) | CPU fallback | ggml-qnn supports_buft() 비호환 |
| OP11 | NPU (Hexagon V73) | CPU fallback | 동일 |

---

## 2. NPU 실행을 위한 3가지 대안 경로

### 경로 A: Qualcomm Genie SDK (권장, 가장 현실적)

**개요**: Qualcomm의 공식 on-device LLM 추론 엔진. 사전 컴파일된 QNN Context Binary를 NPU에서 직접 실행.

**워크플로우:**
```
1. Qualcomm AI Hub에서 모델 컴파일
   qai_hub_models --model Qwen2.5-1.5B-Instruct --target sm8650 --quantize w4a16

2. QNN Context Binary (.bin) 다운로드

3. Genie SDK로 디바이스에서 실행
   Genie runtime이 HTP에 직접 오프로드

4. 프로파일링: Genie 내장 timing API 또는 CDSP clock 모니터링
```

**필요한 개발:**
| 항목 | 설명 | 난이도 | 예상 기간 |
|------|------|--------|---------|
| AI Hub 계정 + API 키 | Qualcomm AI Hub 가입 | 낮음 | 1일 |
| 모델 컴파일 | Qwen2.5-1.5B → QNN Context Binary | 중간 | 1-2일 |
| Genie SDK 통합 | Android 앱 또는 CLI 래퍼 작성 | 중간 | 3-5일 |
| 프로파일러 통합 | Genie timing → ProfileRecord 변환 | 중간 | 2-3일 |
| **총** | | | **1-2주** |

**프로파일링 방식:**
- Genie SDK는 per-layer timing을 제공하지 않음 (블랙박스)
- 대안: CDSP clock + Q6 load 모니터링 (시간 기반 조인)
- 또는: QNN optrace (QAIRT 2.39+) — chrometrace 형식의 per-op timing

**장점:**
- Qualcomm 공식 경로, 안정적
- 실제 NPU에서 실행됨 (CDSP clock 상승으로 검증 가능)
- W4A16 양자화 → NPU 최적화 포맷

**단점:**
- llama.cpp cb_eval과 다른 프로파일링 방식 → 크로스 비교 복잡
- 블랙박스 → per-layer 분해가 제한적
- AI Hub 의존성 (클라우드 컴파일)

---

### 경로 B: ExecuTorch + QNN Delegate

**개요**: Meta의 ExecuTorch 프레임워크에서 QNN delegate를 통해 NPU에 오프로드.

**워크플로우:**
```
1. PyTorch 모델을 ExecuTorch IR로 변환
   python -m executorch.export --model qwen2.5-1.5b --delegate qnn

2. QNN delegate가 HTP-compatible subgraph를 NPU에 오프로드

3. 디바이스에서 실행
   ExecuTorch runtime (50KB) + QNN delegate

4. 프로파일링: ExecuTorch profiling API + QNN delegate timing
```

**필요한 개발:**
| 항목 | 설명 | 난이도 | 예상 기간 |
|------|------|--------|---------|
| ExecuTorch 빌드 (Android) | QNN delegate 포함 빌드 | 높음 | 3-5일 |
| 모델 변환 | HuggingFace → ExecuTorch + QNN | 높음 | 3-5일 |
| 프로파일러 통합 | ET profiling → ProfileRecord | 중간 | 2-3일 |
| Android 배포 | adb push + 실행 | 낮음 | 1일 |
| **총** | | | **2-3주** |

**프로파일링 방식:**
- ExecuTorch의 `ETDumpGen` — per-operator profiling 내장
- QNN delegate가 NPU에 위임한 서브그래프 단위로 timing
- 레이어 매핑: ExecuTorch IR에서 레이어 이름 유지

**장점:**
- 오픈소스, Meta 공식 지원
- Per-operator profiling 가능
- SM8650 (SD 8 Gen 3) 공식 지원 확인 (ExecuTorch docs)

**단점:**
- 빌드 복잡도 높음
- 모델 변환 실패 가능성 (Qwen 아키텍처 호환성)
- ExecuTorch LLM 추론 최적화가 llama.cpp 대비 미성숙

**레퍼런스:**
- [ExecuTorch QNN Backend](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html)
- [Run Llama 3 3B on Android with QNN](https://docs.pytorch.org/executorch/1.0/llm/build-run-llama3-qualcomm-ai-engine-direct-backend.html)

---

### 경로 C: llama.cpp ggml-qnn 백엔드 수정 (가장 통합적이지만 고위험)

**개요**: chraac의 ggml-qnn 코드를 수정하여 GGUF mmap 버퍼 → NPU 메모리 복사를 구현.

**필요한 수정:**

#### C-1. `supports_buft()` 수정

현재:
```cpp
bool npu_device::supports_buft(ggml_backend_buffer_type_t buft) const {
    return buft && buft->device && buft->device->context == this;
}
```

수정:
```cpp
bool npu_device::supports_buft(ggml_backend_buffer_type_t buft) const {
    if (buft && buft->device && buft->device->context == this) return true;
    // Also accept host buffers — we'll copy to NPU memory at graph execution time
    if (ggml_backend_buft_is_host(buft)) return true;
    return false;
}
```

#### C-2. Buffer 복사 경로 구현

`buffer.cpp`에 CPU → NPU 메모리 복사 로직 추가:
```cpp
// rpcmem_alloc()으로 NPU 접근 가능 메모리 할당
// memcpy로 GGUF mmap 데이터 → rpcmem 버퍼 복사
// FastRPC로 DSP에 버퍼 주소 전달
```

#### C-3. Graph execution에서 dequantize 처리

NPU의 matmul 커널이 Q4_K_M/Q8_0을 직접 처리하지 못하면:
- CPU에서 dequantize → FP16
- FP16 데이터를 NPU에 전송
- NPU에서 FP16 matmul 실행

**필요한 개발:**
| 항목 | 설명 | 난이도 | 예상 기간 |
|------|------|--------|---------|
| supports_buft() 수정 | 호스트 버퍼 수용 | 낮음 | 1일 |
| rpcmem 버퍼 복사 | CPU → NPU 메모리 전송 | 높음 | 5-7일 |
| Dequantize 경로 | Q8_0 → FP16 → NPU | 높음 | 3-5일 |
| 테스트 + 디버깅 | FastRPC 이슈 해결 | 매우 높음 | 5-10일 |
| **총** | | | **3-4주** |

**장점:**
- llama.cpp 통합 → cb_eval 프로파일링 그대로 사용 가능
- GGUF 포맷 유지 → 다른 디바이스와 동일 모델
- 크로스 디바이스 비교가 가장 깔끔

**단점:**
- ggml-qnn이 실험적 코드 — 디버깅 난이도 매우 높음
- FastRPC/rpcmem은 Qualcomm 디바이스에서만 동작 → 디바이스에서 직접 디버깅 필요
- 4B 모델 한계 (cDSP 32-bit 주소공간)
- chraac의 코드가 업스트림 llama.cpp와 동기화 문제 발생 가능

**레퍼런스:**
- [ggml-qnn PR #12063](https://github.com/ggml-org/llama.cpp/pull/12063)
- [Hexagon 성능 이슈 #18139](https://github.com/ggml-org/llama.cpp/issues/18139)

---

## 3. 권장 순서

```
Phase 1 (즉시): 경로 A — Genie SDK
  ├─ AI Hub에서 Qwen2.5-1.5B QNN context binary 컴파일
  ├─ Genie SDK CLI로 OP13/OP11에서 실행
  ├─ CDSP clock 모니터링으로 NPU 동작 확인
  └─ 최소한의 프로파일링 (전체 inference time + CDSP metrics)

Phase 2 (1-2주 후): 경로 B — ExecuTorch + QNN
  ├─ ExecuTorch Llama 3 3B 예제를 기반으로 셋업
  ├─ Qwen2.5-1.5B 변환 시도
  ├─ ETDumpGen으로 per-operator profiling
  └─ ProfileRecord 통합

Phase 3 (선택적): 경로 C — ggml-qnn 수정
  ├─ supports_buft() 수정으로 빠른 확인
  ├─ 실제 NPU 오프로드가 되면 성능 측정
  └─ 안 되면 포기 (코스트 대비 효과 낮음)
```

---

## 4. 프로파일링 방식 비교

| 방식 | 레이어 분해 | 크로스 디바이스 비교 | 구현 난이도 | 정확도 |
|------|-----------|-------------------|-----------|--------|
| **cb_eval (현재)** | ✅ per-tensor | ✅ 동일 스키마 | 완료 | GPU/NPU에서 오버헤드 |
| **Genie SDK** | ❌ 블랙박스 | ⚠️ 별도 스키마 | 중간 | 높음 (실제 NPU) |
| **ExecuTorch ETDump** | ✅ per-operator | ⚠️ 별도 스키마 | 높음 | 높음 (실제 NPU) |
| **QNN optrace** | ✅ per-op chrometrace | ⚠️ 별도 스키마 | 중간 | 높음 (실제 NPU) |
| **CDSP clock polling** | ❌ 전체 NPU만 | ✅ 시간 기반 조인 | 낮음 | 간접 측정 |

---

## 5. 프로젝트 스키마 확장 필요

현재 `ProfileRecord`에 NPU 프로파일링 소스를 구분하는 필드 추가 필요:

```python
class ProfileRecord(BaseModel):
    # ... 기존 필드 ...
    
    # 추가 제안:
    profiling_method: str  # "cb_eval", "genie", "executorch_etdump", "qnn_optrace"
    actual_backend: str    # "cpu", "gpu_cuda", "gpu_opencl", "npu_htp", "cpu_fallback"
    overhead_factor: Optional[float]  # cb_eval ON/OFF 비율 (보정용)
```

이렇게 하면 다른 프로파일링 소스의 데이터도 통합 스키마에 저장 가능.

---

## 6. OpenCL GPU 프로파일링 보완 (추가 작업)

현재 OpenCL GPU 빌드에서 `GGML_OPENCL_PROFILING=ON`으로 빌드했으므로, llama.cpp의 OpenCL 백엔드가 **per-kernel GPU timing**을 Chrome trace 형식으로 출력 가능:

```bash
# OpenCL profiling trace 생성
adb shell "... ./llm-profiler-opencl ... 2>&1" 
# → OpenCL 백엔드가 clGetEventProfilingInfo로 per-kernel timing 수집
# → Chrome trace JSON 출력
```

이것은 cb_eval 오버헤드 없는 **진짜 GPU 레이어별 타이밍**을 제공할 수 있음.
→ 별도 조사 필요: `GGML_OPENCL_PROFILING` 활성화 시 출력 포맷 확인

---

## 7. OP11 cb_eval OFF 미측정 보완

현재 OP11에서 누락된 측정:

| 측정 | OP13 | OP11 |
|------|------|------|
| CPU cb_eval OFF | ✅ 5.7 tok/s | ❌ 미측정 |
| GPU OpenCL cb_eval OFF | ✅ 20.9 tok/s | ❌ 미측정 |

OP11 cb_eval OFF를 측정하여 오버헤드 테이블을 완성해야 함.

```bash
# OP11 CPU cb_eval OFF
adb -s db151d78 shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp \
    ./llama-cli -m qwen2.5-1.5b-instruct-q4_k_m.gguf \
    -p 'Hello' -n 16 -t 8 -no-cnv -st 2>&1 | tail -5"

# OP11 GPU OpenCL cb_eval OFF (llama-cli-opencl 필요)
```

---

## References

- [NPU Troubleshooting Report](./npu_troubleshooting_report_20260407.md) — 이 문서의 상세 디버깅 기록
- [Track Feasibility Report](../docs/superpowers/specs/2026-04-05-track-feasibility-report.md) — QNN 백엔드 실현가능성 조사
- [HMX/HVX Profiling Guide](../../claudedocs/hmx_hvx_profiling_guide_20260329.md) — NPU 모니터링 방법론
- [HMX/HVX Troubleshooting](../../claudedocs/hmx_hvx_profiling_troubleshooting_20260402.md) — V73/V75 프로파일링 차이
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — 모델 컴파일 서비스
- [Genie SDK README](https://aihub.qualcomm.com/apps/chatapp_android) — LLM Chat Android
- [ExecuTorch QNN Backend](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html) — Meta 공식 문서
- [ExecuTorch Llama3 QNN Guide](https://docs.pytorch.org/executorch/1.0/llm/build-run-llama3-qualcomm-ai-engine-direct-backend.html)
