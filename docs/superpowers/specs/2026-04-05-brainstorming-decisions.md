# tiny-llm-profiler: Brainstorming Decisions

> Date: 2026-04-05
> Status: Confirmed
> Context: Edge LLM per-layer profiling project across heterogeneous devices

---

## 1. Project Goal

각 플랫폼에서 LLM을 돌리며 **레이어별 리소스 사용량**을 계층적으로 프로파일링하여,
학술 논문(A), 엔지니어링 최적화 가이드(B), 프로덕트 개발 기반 데이터(D)로 활용한다.

---

## 2. Target Devices

| Device | Host | Compute Targets | RAM | Connection |
|--------|------|----------------|-----|------------|
| Raspberry Pi 4 | home.rasp4.local | CPU (Cortex-A72) | 3.7GB | SSH |
| Jetson AGX Orin 64GB | home.orin.local | **CPU (Cortex-A78AE)** + GPU (Ampere CUDA) | 64GB unified | SSH |
| OnePlus 13 (CPH2645) | adb:85ea76a2 | CPU + GPU (Adreno 750v2) + NPU (Hexagon HTP) | 12GB | ADB USB |
| OnePlus 11 (CPH2451) | adb:db151d78 | CPU + GPU (Adreno 740v2) + NPU (Hexagon HTP) | 16GB | ADB USB |
| ESP32-S3 XIAO Sense | TBD (serial/USB) | MCU dual-core (Xtensa 240MHz) | 8MB PSRAM | Serial |
| Raspberry Pi Zero 2 W | TBD (SSH) | CPU (Cortex-A53) | 512MB | SSH |

*Note: ESP32-S3와 Pi Zero 2 W는 아직 미구매. 조사 기반 후보.*

---

## 3. Profiling Metrics

### Core Metrics (A-E)

| ID | Metric | Unit | Description |
|----|--------|------|-------------|
| A | Layer latency | microseconds (us) | 레이어별 실행 시간 |
| B | Memory usage | bytes | 레이어별 메모리 사용량 (peak + resident) |
| C | Power draw | milliwatts (mW) | 레이어별 전력 소비 |
| D | FLOPS | count | 레이어별 부동소수점 연산 수 |
| E | Memory bandwidth | bytes/sec | 레이어별 메모리 대역폭 사용량 |

### Extended Metrics (F-J)

| ID | Metric | Unit | Available Devices |
|----|--------|------|-------------------|
| F | Cache hit ratio (L1/L2) | ratio (0.0~1.0) | RPi4, Orin CPU, Android CPU (perf_event) |
| G | Thermal delta | celsius | All (thermal_zone, tegrastats) |
| H | Phase separation | prefill/decode | All |
| I | Operation type ratio | percentage | All (ggml graph / TVM IR) |
| J | GPU SM/Tensor Core utilization | percentage | Orin GPU (Nsight, tegrastats) |

---

## 4. Profiling Granularity

**계층적 프로파일링 (Hierarchical, Option D):**

```
Level 1: Transformer Block (Layer 0, 1, ..., N)
  └─ Level 2: Sub-layer (RMSNorm, Q/K/V Proj, Attn Score, Softmax, FFN Up/Gate/Down, ...)
       └─ Level 3: Tensor Operation (matmul, add, rope, norm, ...)
```

- Block 단위로 요약 + Sub-layer 단위로 드릴다운
- 논문에서 가장 풍부한 데이터 제공

---

## 5. Framework Strategy: Two-Track

### Track 1: llama.cpp (ggml) + QNN NPU (experimental)

- **Coverage**: RPi4(CPU), Orin(CPU+GPU), OnePlus 13/11(CPU+GPU+NPU*), Pi Zero(CPU)
- **NPU Status**: ggml-qnn backend WIP (PR #12063, #12326). 실험적 — 논문에서 "experimental" 표기
- **Profiling**: ggml 그래프 워킹 + 커스텀 타이밍 훅 삽입 필요
- **NPU 제한**: 4B 이하 모델만 (cDSP 32-bit 주소공간)

### Track 2: MLC-LLM (Apache TVM)

- **Coverage**: RPi4(CPU/ARM), Orin(CUDA), OnePlus 13/11(OpenCL/Vulkan), Pi Zero(ARM)
- **NPU**: Qualcomm NPU 미지원 (Issue #1689)
- **Profiling**: 프레임워크 내장
  - `tvm.runtime.profiling` — per-kernel 타이밍
  - `EventTraceRecorder` — GPU 커널별 계측
  - `MetricCollector` — cache hits, FLOPS 수집
  - IR 덤프 — 컴파일 파이프라인 각 단계 저장

### Track Comparison

| Aspect | Track 1 (ggml) | Track 2 (TVM) |
|--------|---------------|---------------|
| Built-in profiling | No (커스텀 훅 필요) | Yes (내장) |
| NPU support | Experimental (QNN) | No |
| ESP32 support | Via llama2.c | No |
| Cross-validation | Primary | Secondary (검증용) |

---

## 6. Model Matrix

### Tier Structure

| Tier | Size | Target Devices |
|------|------|---------------|
| T0 | ~260K | ESP32 only |
| T1 | 135M~500M | Pi Zero ~ All |
| T2 | 1.5B~2B | RPi4 ~ All |
| T3 | 3B~3.8B | Android ~ Orin |
| T4 | 7B~8B | OnePlus 11, Orin |

### Architecture Coverage

| Tier | Qwen2.5 | Llama-3.2 | Gemma | Phi-3.5 | RWKV | SmolLM2 |
|------|---------|-----------|-------|---------|------|---------|
| T0 | - | - | - | - | - | - (TinyStories-260K) |
| T1 | 0.5B | - | 270M | - | 169M | 135M |
| T2 | 1.5B | - | 2B | - | 1.5B | 1.7B |
| T3 | 3B | 3B | 2B-IT | 3.8B | - | - |
| T4 | 7B | 8B | - | - | 7B | - |

### Architecture Comparison Points

- **Transformer (GQA)**: Qwen, Llama — Grouped Query Attention
- **Transformer (MQA)**: Gemma — Multi-Query Attention
- **Transformer (Dense)**: Phi — Dense Attention
- **Non-Transformer**: RWKV — Linear RNN, 근본적으로 다른 레이어 특성

---

## 7. Data Collection Architecture

**Hybrid approach:**

- **경량 기기** (ESP32, Pi Zero): 로컬 JSON 저장 → SCP/rsync로 수집
- **네트워크 기기** (RPi4, Orin, Android): MQTT/HTTP 실시간 전송 → 중앙 수집

---

## 8. Visualization / Reporting

**Dual output:**

- **실시간**: Streamlit 대시보드 — 기기/모델/레이어별 필터링, 드릴다운
- **논문용**: Jupyter Notebook + matplotlib/seaborn — 재현 가능한 차트, 통계 분석

데이터 포맷을 Parquet/SQLite로 통일하여 양쪽에서 공유.

---

## 9. Selected Architecture: Layered Architecture (Approach 3)

4개 레이어로 분리:

1. **Collection**: Track 1 (ggml hooks) + Track 2 (TVM profiler) + Platform Monitor (power/thermal)
2. **Storage**: 통합 Parquet/SQLite 스토어, 공통 스키마
3. **Analysis**: Pandas/NumPy 분석 엔진 (기기/아키텍처/스케일링 비교)
4. **Presentation**: Streamlit (실시간) + Jupyter (논문용)

각 레이어 독립 → 점진적 구현 가능. Platform Monitor가 프레임워크와 분리되어 power/thermal 측정 공유.

---

## References

- [Edge LLM Research Report](../../claudedocs/edge_llm_research_report.md)
- [llama.cpp ggml-qnn PR #12063](https://github.com/ggml-org/llama.cpp/pull/12063)
- [llama.cpp ggml-hexagon PR #12326](https://github.com/ggml-org/llama.cpp/pull/12326)
- [MLC-LLM NPU Request Issue #1689](https://github.com/mlc-ai/mlc-llm/issues/1689)
- [TVM Runtime Profiling API](https://mlc.ai/docs/reference/api/runtime/profiling.html)
- [Hexagon backend performance Issue #18139](https://github.com/ggml-org/llama.cpp/issues/18139)
