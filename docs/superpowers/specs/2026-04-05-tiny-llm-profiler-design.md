# tiny-llm-profiler: Design Specification

> Date: 2026-04-05
> Status: Draft → Review
> Authors: seunghyunoh
> Purpose: 이기종 엣지 디바이스에서 LLM 추론의 레이어별 리소스 프로파일링

---

## 1. Overview

### 1.1 목표

각 플랫폼에서 LLM을 실행하며 Transformer Block → Sub-layer → Tensor Operation 수준의 계층적 리소스 사용량을 측정하고, 기기/아키텍처/스케일 간 비교 분석을 수행한다.

### 1.2 활용 목적

- **(A) 학술 논문**: 정밀 수치, 재현 가능한 실험, 통계적 유의성
- **(B) 최적화 가이드**: 레이어별 병목 식별 → 모델/하드웨어 최적화
- **(D) 프로덕트 개발**: 타겟 디바이스 최적 LLM 서빙을 위한 데이터 수집

### 1.3 프레임워크 전략: 투트랙

- **Track 1 (메인)**: llama.cpp (ggml) + QNN NPU 실험적 — 전 기기 커버
- **Track 2 (보조)**: MLC-LLM (Apache TVM) — 크로스 프레임워크 검증, 컴파일러 최적화 효과 분석

---

## 2. Target Devices

| Device | Host | Compute Targets | RAM | Connection |
|--------|------|----------------|-----|------------|
| Raspberry Pi 4 | home.rasp4.local | CPU (Cortex-A72 4C @ 1.8GHz) | 3.7GB | SSH |
| Jetson AGX Orin 64GB | home.orin.local | CPU (Cortex-A78AE 8C @ 2.2GHz) + GPU (Ampere 2048 CUDA cores) | 64GB unified | SSH |
| OnePlus 13 (CPH2645) | adb:85ea76a2 | CPU (SD 8 Gen 3) + GPU (Adreno 750v2) + NPU (Hexagon 73 TOPS) | 12GB | ADB USB |
| OnePlus 11 (CPH2451) | adb:db151d78 | CPU (SD 8 Gen 2) + GPU (Adreno 740v2) + NPU (Hexagon ~37 TOPS) | 16GB | ADB USB |
| ESP32-S3 XIAO Sense | TBD (serial) | MCU dual-core (Xtensa 240MHz) | 8MB PSRAM | Serial/USB |
| Raspberry Pi Zero 2 W | TBD (SSH) | CPU (Cortex-A53 4C @ 1GHz) | 512MB | SSH |

---

## 3. Profiling Metrics

### 3.1 Core Metrics (A-E)

| ID | Metric | Unit | Collection Method |
|----|--------|------|-------------------|
| A | Layer latency | us | ggml: `ggml_time_us()` in cb_eval. TVM: `report.calls` timing |
| B | Memory usage | bytes | ggml: `ggml_nbytes(t)` per tensor. TVM: report metrics |
| C | Power draw | mW | Platform Monitor (별도 스레드, 시간 기반 조인) |
| D | FLOPS | count | 텐서 shape에서 계산 (matmul: 2*M*N*K) |
| E | Memory bandwidth | bytes/sec | `ggml_nbytes(src+dst)` / latency |

### 3.2 Extended Metrics (F-J)

| ID | Metric | Unit | Available Devices |
|----|--------|------|-------------------|
| F | Cache hit ratio (L1/L2) | ratio 0.0~1.0 | RPi4, Orin CPU, Android CPU (perf_event / PAPI) |
| G | Thermal delta | celsius | All (thermal_zone, tegrastats, dumpsys) |
| H | Phase separation | prefill/decode | All |
| I | Operation type ratio | percentage | All (ggml t->op / TVM kernel name) |
| J | GPU SM/Tensor Core util | percentage | Orin GPU (tegrastats, Nsight) |

### 3.3 Profiling Granularity

```
Level 1: Transformer Block (block_idx: 0~N)
  └─ Level 2: Sub-layer (attention_qkv, attention_score, attention_out,
  │            ffn_gate, ffn_up, ffn_down, rmsnorm, rope, softmax,
  │            embedding, lm_head)
       └─ Level 3: Tensor Operation (matmul, add, mul, rope, norm, ...)
```

---

## 4. Architecture: Layered Design

```
┌─ Layer 4: Presentation ──────────────────────────┐
│  Streamlit (실시간 대시보드)  │  Jupyter (논문용)  │
└──────────────┬───────────────┬────────────────────┘
┌─ Layer 3: Analysis ──┴───────┴────────────────────┐
│  DeviceComparator      │  ArchitectureComparator   │
│  ScalingComparator     │  FrameworkComparator       │
│  BottleneckAnalyzer    │  StatisticsEngine          │
└──────────────────────┬────────────────────────────┘
┌─ Layer 2: Storage ───┴────────────────────────────┐
│  SQLite (실시간 쿼리)  │  Parquet (대량 분석)      │
│  통합 스키마: ProfileRecord                        │
└───────────┬──────────────┬────────────────────────┘
┌─ Layer 1: Collection ────┴────────────────────────┐
│  Track 1: ggml Profiler (cb_eval callback)        │
│  Track 2: TVM Profiler (vm.profile / Nsight/OCL)  │
│  Platform Monitor (power, thermal, freq per device)│
│  Transport (local JSON → SCP | MQTT → central)    │
└───────────────────────────────────────────────────┘
```

---

## 5. Layer 1: Collection

### 5.1 Track 1 — ggml Profiler

**핵심 메커니즘**: `ggml_backend_sched_set_eval_callback()` — llama.cpp fork 불필요, `llama_context_params.cb_eval`에 등록.

```c
static bool profiling_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = (profiler_state_t *) user_data;
    if (ask) {
        state->current.name = t->name;        // "blk.5.attn_q.weight"
        state->current.op = t->op;            // GGML_OP_MUL_MAT
        state->current.bytes = ggml_nbytes(t);
        state->current.flops = compute_flops(t);
        state->current.start_us = ggml_time_us();
        return true;
    } else {
        state->current.end_us = ggml_time_us();
        emit_record(state);                    // → JSON/MQTT
        return true;
    }
}
```

**텐서 이름 → 레이어 매핑 파싱:**
- `"blk.5.attn_q.weight"` → block_idx=5, sublayer="attention_qkv"
- `"blk.5.ffn_up.weight"` → block_idx=5, sublayer="ffn_up"
- `"output_norm.weight"` → block_idx=-1, sublayer="lm_head"

**QNN NPU (실험적):**
- PR #12326의 `enable_perf = 1` 플래그로 기본 타이밍 수집
- QNN SDK 블랙박스 — ggml 레벨 호출 전후 시간만 측정 가능
- 4B 이하 모델 한정 (cDSP 32-bit 주소공간)
- 논문에서 "experimental, opaque NPU" 표기

**검증 상태:** `cb_eval` API 존재 확인, `examples/eval-callback/` 예시 검증됨. 동기화 강제로 per-node 벽시계 시간 정확. 오버헤드 10-30% (상대 비율 유효).

### 5.2 Track 2 — TVM Profiler

**확인된 기기:**
- Orin GPU (CUDA): ✅ 확인됨. Nsight Systems로 per-kernel 타이밍 수집
- Android GPU (OpenCL): ✅ 확인됨. `clGetEventProfilingInfo` 기반 수집
- RPi 4 / Orin CPU / Android CPU: ⚠️ CPU 백엔드 초기 지원 (PR #3116, 2025.02 머지). 14개월간 후속 개발 없음. 직접 검증/해결 필요. 실패 시 Track 1 폴백

**vm.profile() 이슈:**
- MLC-LLM의 paged KV cache 런타임 함수가 프로파일링 컴파일 시 미발견 → vm.profile() 실패
- 우회: Nsight (CUDA), OpenCL 이벤트 수정, `debug_dump` IR 기반 커널→레이어 매핑 구축

**커널→레이어 매핑:** MLC-LLM 커널 이름에 레이어 인덱스 미포함. `debug_dump` IR 파싱으로 매핑 테이블 자동 구축.

### 5.3 Platform Monitor

```python
class PlatformMonitor(ABC):
    def sample(self) -> PlatformSnapshot: ...

@dataclass
class PlatformSnapshot:
    timestamp: datetime
    cpu_temp_c: float
    gpu_temp_c: float | None
    soc_temp_c: float | None
    power_mw: float
    cpu_freq_mhz: list[float]
    gpu_freq_mhz: float | None
    mem_used_bytes: int
    mem_available_bytes: int
```

| Device | Temperature | Power | Frequency |
|--------|------------|-------|-----------|
| RPi 4 | `/sys/class/thermal/thermal_zone0/temp` | `vcgencmd` 기반 추정 | cpufreq sysfs |
| Orin | `tegrastats` | `tegrastats` VDD_*_POWER | `tegrastats` + `nvidia-smi` |
| Android | `/sys/class/thermal/thermal_zone*/temp` | `dumpsys battery` | cpufreq sysfs |
| ESP32 | 내부 온도 센서 API | 외부 INA219 또는 추정 | 고정 240MHz |
| Pi Zero | `/sys/class/thermal/` | 추정 (~3W) | cpufreq sysfs |

**샘플링**: 별도 스레드, 100ms 간격. 추론 프로파일러의 타임스탬프와 시간 기반 조인.

### 5.4 Transport

- **경량 기기** (ESP32, Pi Zero): 로컬 JSON → `scp`/`rsync` 수집
- **네트워크 기기** (RPi4, Orin, Android): MQTT 실시간 전송 → 중앙 수집기

---

## 6. Layer 2: Storage

### 6.1 통합 스키마

```python
@dataclass
class ProfileRecord:
    # 식별
    experiment_id: str
    timestamp: datetime
    device: str             # "rpi4", "orin_cpu", "orin_gpu", "op13_cpu", ...
    framework: str          # "ggml", "tvm"
    model: str              # "qwen2.5-1.5b-q4_k_m"
    architecture: str       # "qwen", "llama", "gemma", "phi", "rwkv"
    
    # 위치
    phase: str              # "prefill", "decode"
    block_idx: int          # 0~N, -1 for embed/lm_head
    sublayer: str           # "attention_qkv", "ffn_up", "rmsnorm", ...
    op_type: str            # "matmul", "add", "rope", "norm", ...
    
    # 메트릭 A-E
    latency_us: float
    mem_bytes: int
    mem_peak_bytes: int
    power_mw: float
    flops: int
    bandwidth_bytes_sec: int
    
    # 메트릭 F-J
    cache_l1_hit_ratio: float | None
    cache_l2_hit_ratio: float | None
    thermal_c: float
    gpu_sm_util: float | None
    gpu_tensor_util: float | None
    
    # 컨텍스트
    input_tokens: int
    output_token_idx: int
    quantization: str
    batch_size: int
```

### 6.2 듀얼 스토어

- **SQLite**: 실시간 대시보드 쿼리. 인덱스: `(experiment_id, device, model, block_idx)`
- **Parquet**: 논문 분석용 고속 처리. 파티셔닝: `experiment_id/device/model/`

### 6.3 Ingest Engine

- JSON → ProfileRecord 변환/검증 (Pydantic)
- 타임스탬프 기반 PlatformSnapshot 조인 → power_mw, thermal_c 매핑
- 중복 제거 + experiment_id 부여

---

## 7. Layer 3: Analysis

### 7.1 Comparators

| Module | 비교 축 | 핵심 질문 |
|--------|---------|---------|
| DeviceComparator | 같은 모델, 다른 기기 | Qwen2.5-1.5B가 RPi4 vs Orin에서 레이어별 latency 차이는? |
| ArchitectureComparator | 같은 기기, 다른 아키텍처 | Qwen vs Llama vs RWKV의 Attention/FFN 비율 차이는? |
| ScalingComparator | 같은 아키텍처, 다른 크기 | Qwen 0.5B→7B 스케일업 시 레이어별 latency 선형/비선형? |
| FrameworkComparator | 같은 모델+기기, 다른 프레임워크 | ggml vs TVM에서 어떤 레이어가 성능 차이나는가? |

### 7.2 BottleneckAnalyzer

- `classify_bottleneck()`: compute_bound / memory_bound / thermal_throttled 분류
- `find_hotspots()`: latency 기준 상위 K개 레이어/sublayer 추출
- `compute_roofline()`: Roofline 모델 — operational intensity vs 달성 성능

### 7.3 StatisticsEngine

- `summarize()`: mean, std, median, p95, p99, CI(95%)
- `test_significance()`: Mann-Whitney U test (비정규분포 가정)
- `compute_overhead()`: 프로파일링 오버헤드 측정 (콜백 유/무 비교)

---

## 8. Layer 4: Presentation

### 8.1 Streamlit Dashboard

| 페이지 | 내용 | 데이터 소스 |
|--------|------|-----------|
| Live Monitor | 실험 진행 중 실시간 레이어 타이밍 + 플랫폼 상태 | SQLite |
| Device Compare | 기기 간 레이어별 비교 heatmap | Parquet |
| Architecture Compare | 아키텍처별 sublayer 비율 stacked bar | Parquet |
| Scaling Analysis | 모델 크기별 스케일링 곡선 | Parquet |
| Framework Compare | Track 1 vs Track 2 검증 | Parquet |
| Roofline | Operational intensity scatter plot | Parquet |

### 8.2 Jupyter Notebooks

| Notebook | 논문 Figure |
|----------|-----------|
| 01_device_comparison | 기기별 레이어 latency heatmap |
| 02_architecture_analysis | 아키텍처별 sublayer 비율 |
| 03_scaling_analysis | 파라미터 스케일링 곡선 |
| 04_bottleneck_roofline | Roofline plot |
| 05_framework_comparison | ggml vs TVM 크로스 검증 |
| 06_thermal_power | 열/전력 vs 성능 상관관계 |
| 07_rwkv_vs_transformer | RNN vs Transformer 레이어 특성 |
| 08_profiling_overhead | 프로파일링 오버헤드 보정 |

차트: matplotlib + seaborn, 300 DPI PDF, colorblind-safe (viridis/cividis).

---

## 9. Model Matrix

### 9.1 Tier Structure

| Tier | Size Range | Target Devices |
|------|-----------|---------------|
| T0 | ~260K | ESP32 only |
| T1 | 135M~500M | Pi Zero ~ All |
| T2 | 1.5B~2B | RPi4 ~ All |
| T3 | 3B~3.8B | Android ~ Orin |
| T4 | 7B~8B | OnePlus 11, Orin |

### 9.2 Architecture × Tier

| Tier | Qwen2.5 | Llama-3.2 | Gemma | Phi-3.5 | RWKV | SmolLM2 |
|------|---------|-----------|-------|---------|------|---------|
| T0 | - | - | - | - | - | TinyStories-260K |
| T1 | 0.5B | - | 270M | - | 169M | 135M |
| T2 | 1.5B | - | 2B | - | 1.5B | 1.7B |
| T3 | 3B | 3B | 2B-IT | 3.8B | - | - |
| T4 | 7B | 8B | - | - | 7B | - |

### 9.3 Architecture Comparison Points

- **Transformer (GQA)**: Qwen, Llama — Grouped Query Attention
- **Transformer (MQA)**: Gemma — Multi-Query Attention
- **Transformer (Dense)**: Phi — Dense Attention
- **Non-Transformer**: RWKV — Linear RNN

---

## 10. Experiment Design

### 10.1 Parameters

```yaml
warmup_tokens: 32
prompt_tokens: 128
generate_tokens: 64
repetitions: 5
cooldown_sec: 30
platform_monitor_interval_ms: 100
overhead_calibration: true
```

### 10.2 Scale

- ~755 실험 (Devices × Models × Frameworks × Repetitions)
- 각 실험 ~4,600 레이어 레코드
- 총 ~3.5M 레코드 (~200-500MB Parquet)

### 10.3 Execution Phases

| Phase | Duration | Scope |
|-------|----------|-------|
| 1. 환경 구축 | 1-2주 | 빌드, 프로파일러 검증, 모델 준비 |
| 2. T1 실험 | 1주 | 전 기기 파이프라인 검증, 오버헤드 보정 |
| 3. T2-T4 실험 | 2-3주 | 전체 모델 매트릭스 실행 |
| 4. 분석 + 논문 | 2-3주 | Jupyter 분석, 대시보드, 논문 Figure |

---

## 11. Project Structure

```
tiny-llm-profiler/
├── schemas/
│   └── profile_schema.py
├── collection/
│   ├── track1_ggml/
│   │   ├── ggml_profiler.c
│   │   ├── qnn_probe.cpp
│   │   └── export.py
│   ├── track2_tvm/
│   │   ├── tvm_profiler.py
│   │   └── export.py
│   └── platform_monitor/
│       ├── base.py
│       ├── rpi.py
│       ├── jetson.py
│       ├── android.py
│       └── esp32.py
├── transport/
│   ├── local_store.py
│   ├── mqtt_sender.py
│   └── collector.py
├── storage/
│   ├── db.py
│   └── ingest.py
├── analysis/
│   ├── comparators.py
│   ├── bottleneck.py
│   └── statistics.py
├── presentation/
│   ├── dashboard/
│   │   └── app.py
│   └── notebooks/
│       ├── 01_device_comparison.ipynb
│       ├── ...
│       └── 08_profiling_overhead.ipynb
├── configs/
│   ├── devices.yaml
│   ├── models.yaml
│   └── experiments.yaml
├── scripts/
│   ├── deploy.sh
│   ├── run_experiment.py
│   └── collect_results.sh
└── docs/
    └── superpowers/specs/
```

---

## 12. Known Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| ggml cb_eval 동기화 오버헤드 10-30% | MEDIUM | 상대 비율 유효. 콜백 유/무 비교로 보정. 논문에 오버헤드 수치 명시 |
| MLC-LLM vm.profile() KV cache 이슈 | HIGH | Nsight/OpenCL 이벤트 우회. 직접 수정 시도 |
| MLC-LLM CPU 백엔드 미검증 (RPi) | MEDIUM | 직접 검증/해결. 실패 시 Track 1 폴백 |
| QNN NPU 블랙박스 | HIGH | ggml 레벨 타이밍만 수집. 논문에 "opaque" 명시 |
| Android 열 쓰로틀링 | MEDIUM | 30초 쿨다운. 열 안정화 후 실험. thermal 데이터 기록 |
| ESP32 전력 측정 한계 | LOW | 외부 INA219 모듈 또는 추정치 사용 |
| 3.5M 레코드 분석 성능 | LOW | Parquet 컬럼 스토리지 + Pandas 최적화 |

---

## References

- [Brainstorming Decisions](./2026-04-05-brainstorming-decisions.md)
- [Track Feasibility Report](./2026-04-05-track-feasibility-report.md)
- [Edge LLM Research Report](../../claudedocs/edge_llm_research_report.md)
- [ggml eval-callback example](https://gitlab.informatik.uni-halle.de/ambcj/llama.cpp/-/blob/b3010/examples/eval-callback/eval-callback.cpp)
- [TVM Runtime Profiling API](https://tvm.apache.org/docs/reference/api/python/runtime/profiling.html)
- [lm-Meter (SEC'25)](https://arxiv.org/html/2510.06126v1)
