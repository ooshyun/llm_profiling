# Serving Framework Evaluation on Jetson AGX Orin — Design

> Date: 2026-08-27
> Status: Draft, awaiting review
> Context: `claudedocs/serving_framework_candidates_20260828.md`가 후보 4개(llama.cpp / FreeToken / vLLM / SGLang)를 문헌 기준으로 비교했다. 이 스펙은 그 후보들을 **실제로 Orin에서 돌려 측정**하기 위한 설계다.
> Supersedes: 후보 문서의 "Recommended order" 섹션 (이 스펙의 §2 발견 사항으로 인해 순서와 범위가 바뀜)

---

## 1. Goal

Orin 64GB 위에서 **대화형 채팅 + 에이전트 워크로드**를 서빙할 프레임워크를 측정 데이터로 고른다.

측정 대상 워크로드 (사용자가 지정한 우선순위 순):

1. **1인 대화형 채팅** — TTFT, 생성 tok/s
2. **에이전트 루프** — 긴 공통 시스템 프롬프트를 매 턴 재전송할 때 TTFT 추이 (prefix 재사용 효과)
3. **동시 사용자 처리량** — 동시성 1/2/4/8에서 집계 tok/s, TTFT p50/p95

### Non-goals

- GPU 메모리 초과 모델 (122B-A10B) — 이번 라운드 제외. 사용자가 선택하지 않았고 76GB 재다운로드가 필요.
- MAXN 전력 모드 전환 — 사용자가 명시적으로 제외. 모든 측정은 **MODE_30W**에서 수행.
- 프로파일러(`llm-profiler`, cb_eval) 연동 — 이 라운드는 end-to-end 서빙 성능만.
- 모델 품질 비교 — 같은 모델의 다른 양자화(GGUF Q4_K_M vs GPTQ-Int4)를 쓰므로 출력 품질 차이는 존재하지만 측정 범위 밖. §10에 리스크로 기록.

---

## 2. 설계를 바꾼 발견 사항 (2026-08-27 조사)

후보 문서는 "aarch64 지원 미검증"이라고만 했다. 조사 결과 두 가지가 확정되어 계획이 갈렸다.

### 2.1 FreeToken은 Orin에서 실행 불가

| 요구사항 (FreeToken `docs/install.md`, `pyproject.toml`) | Orin 현실 |
|---|---|
| "Linux **x86_64**, NVIDIA GPU, driver r580+ (**CUDA 13**)" | aarch64, JetPack 6.x = CUDA 12.x. CUDA 13은 JetPack 7 = **Thor 전용** |
| "CUDA kernels are JIT-compiled on first use, need a CUDA 13 toolkit with `nvcc` on PATH" | `nvcc` 12.2 |
| `torch>=2.11,<2.12`, `flashinfer-python[cu13]`, `sglang-kernel` (cu130 빌드) | 전부 x86 전용 휠. aarch64 빌드 없음 |
| 지원 모델 표: Qwen3.5-35B-A3B는 **FP8 / NVFP4** 포맷만 | sm_87 Ampere는 FP8 텐서코어도 FP4도 없음. GGUF 입력은 Gemma-4 한정 |
| "native support for NVIDIA RTX 30, RTX 40, and RTX 50 series" | Jetson/Tegra/Ampere-SoC 언급 없음 |

**결론**: 설계 전제(CPU–GPU 협업 실행, 통합 메모리)는 Orin에 잘 맞지만, 구현이 x86 / CUDA 13 / RTX에 못박혀 있다. JetPack을 올려도 CUDA 13은 못 맞춘다. **정식 후보에서 제외**하고, 2시간 상한의 spike로 "실제로 어디서 막히는지" 기록만 남긴다 (§8).

### 2.2 vLLM / SGLang은 JetPack 6.0에서 실행 불가

| 사실 | 출처 |
|---|---|
| `dustynv/vllm`, `dustynv/sglang` 프리빌드 태그는 **전부 r36.4.x** (JetPack 6.1+). r36.3용 없음 | Docker Hub 태그 목록 |
| JP 6.0 (r36.3)에서 r36.4 컨테이너 실행 시 `CUDA error: device kernel image is invalid` | jetson-containers issue #1655 — 미해결 종료 |
| SGLang 공식 Jetson 문서: "JetPack **6.1 or later**" | docs.sglang.io/platforms/nvidia_jetson |
| dustynv 최신 프리빌드는 vLLM **0.9.2**, SGLang **0.4.7** (2025-06) — Qwen3.5(2026-02)를 지원하는 vLLM ≥0.13 / SGLang ≥0.5 아님 | Docker Hub |
| 커뮤니티 빌드 `mitakad/vllm:0.22.0-r36.5.tegra-aarch64-cp312-cu129-24.04` (2026-06, 13GB), `mitakad/sglang` — jetson-containers로 빌드된 최신 버전 존재 | Docker Hub |
| jetson-containers 소스 빌드는 `VLLM_VERSION` 오버라이드 가능 (기본 0.21.0) | `packages/llm/vllm/config.py` |

**결론**: **JetPack 6.0 → 6.2 업그레이드가 vLLM/SGLang의 선결 조건**이다. 업그레이드 후에도 dustynv 공식 이미지는 너무 낡아 쓸 수 없고, `mitakad/*` 커뮤니티 이미지 또는 소스 빌드가 필요하다.

### 2.3 모델 포맷은 해결됨

- `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` — 공식 양자화. vLLM/SGLang 모두 `--quantization moe_wna16`로 지원, 이 커널은 sm_80+ Ampere에서 동작. 약 20GB.
- `Qwen/Qwen3-8B` bf16 — 16GB, 모든 엔진이 확실히 지원. 하네스 검증용 대조군.
- llama.cpp는 기존 GGUF Q4_K_M 그대로.
- Qwen3.5는 **thinking 기본 ON** — 모든 엔진에서 `chat_template_kwargs: {enable_thinking: false}`로 끈다.

### 2.4 Orin 하드웨어/환경 확정 사항

| 항목 | 값 | 영향 |
|---|---|---|
| 보드 | `NVIDIA Jetson AGX Orin Developer Kit` (TNSPEC `3701-500-0005`) | apt 업그레이드 **공식 지원 대상** (devkit only) |
| 부트 스토리지 | **`TEGRA_BOOT_STORAGE nvme0n1`** — rootfs `/`가 NVMe 456GB, eMMC 59GB는 부트로더/예비 | NVIDIA 업그레이드 문서의 커널 오류 복구 절차는 `/dev/mmcblk0` 기준 → NVMe 부트에는 그대로 적용 안 될 수 있음 (§7.6) |
| L4T | r36.3.0, CUDA 12.2, cuDNN 8.9.4, TensorRT 8.6.2 | JP 6.2로 가면 CUDA 12.6 / cuDNN 9 / TRT 10 |
| 호스트 Python | 3.10, `torch 2.6.0+cpu`, `tensorrt 8.6.2` (pip) | 호스트 torch는 안 쓴다. 컨테이너 안에서만 |
| 실행 중 프로세스 | `~/cochl-board/.venv` `serial_log.py` × 2 (Arduino Nicla Voice, USB serial) — 2026-08-27 기준 3시간+ 진행 중 | **재부팅 금지 구간.** 업그레이드는 이 작업 종료 후 |
| Docker 이미지 | `sense-sdk/tensorrt-jetpack6.0.0:1.6.0-beta` (Cochl 자체 SDK, 8개월 전 마지막 실행), `riva-speech:2.19.0` (4개월 전), `llm-profiler-orin` (r36.3 베이스), `dustynv/nemo:r36.2.0` | JP 업그레이드 후 재검증 대상 (§7.5) |
| 디스크 | NVMe 151GB 여유 | 모델 36GB + 이미지 ~30GB 수용 가능 |

---

## 3. Decisions

| # | 결정 | 근거 |
|---|---|---|
| D1 | **범위: llama.cpp, vLLM, SGLang 3개 정식 비교 + FreeToken spike** | §2.1. 사용자는 "4개 전부"를 골랐고, FreeToken은 시도 자체는 하되 결과가 아니라 실패 지점을 기록하는 실험임을 명시 |
| D2 | **JetPack 6.0 → 6.2 업그레이드를 계획에 포함, 실행은 별도 승인** | §2.2. 사용자 지시: "현재 orin이 돌아가고 있어서, 올리기 위한 계획만 작성". §7이 runbook |
| D3 | **컨테이너 경로: `mitakad/*` 프리빌드를 주 경로, `jetson-containers build`를 fallback** | dustynv 공식은 Qwen3.5 미지원 버전. 프리빌드 pull 13GB vs 소스 빌드 2–4시간/엔진 |
| D4 | **모델: Qwen3.5-35B-A3B (본계) + Qwen3-8B (대조군)** | 사용자 선택. 8B는 하네스 sanity check + MoE 미지원 엔진이 나와도 비교 데이터가 남음 |
| D5 | **전력 모드 MODE_30W 고정** | 사용자 제외 지시. April 수치 및 오늘 재측정(10.8 t/s)과 직접 비교 가능 |
| D6 | **모든 엔진을 OpenAI 호환 HTTP API로 동일 하네스에서 측정** | llama.cpp는 `llama-server`, vLLM은 `vllm serve`, SGLang은 `launch_server`. CLI 경로 차이를 제거하고 S2/S3를 가능하게 함 |
| D7 | **Phase 0을 업그레이드 전에 완료** | 하네스·다운로드·llama-server 측정·FreeToken spike·이미지 pull은 재부팅이 필요 없음. 업그레이드 대기 시간을 낭비하지 않음 |
| D8 | **llama.cpp 기준선은 업그레이드 전(CUDA 12.2)과 후(12.6) 두 번 측정** | CUDA 버전 변화가 llama.cpp 성능에 주는 영향을 분리. 그렇지 않으면 vLLM 비교 시 "엔진 차이"와 "CUDA 차이"가 섞임 |

---

## 4. Phases

```
Phase 0  ── 지금, 재부팅 불필요 ──────────────────────────────────────
  0.1 벤치 하네스 작성 (scripts/serving_bench/)
  0.2 llama-server 기동 + 하네스 검증 + 35B/8B 측정  ← 첫 실데이터, CUDA 12.2 기준선
  0.3 모델 다운로드: Qwen3.5-35B-A3B-GPTQ-Int4, Qwen3-8B (bf16)
  0.4 Docker 이미지 pull: mitakad/vllm, mitakad/sglang
  0.5 FreeToken spike (2h 상한)
  0.6 결과 문서 초안 (llama.cpp 행 채움)

Phase 1  ── 업그레이드 날, ~2h + 재부팅, 별도 승인 ───────────────────
  1.1 §7 runbook 실행 (사전 점검 → 백업 → apt → 재부팅 → 검증)
  1.2 llama.cpp 재빌드 (CUDA 12.6) → 35B/8B 재측정  ← CUDA 12.6 기준선
  1.3 기존 컨테이너 기동 확인 (sense-sdk, riva, llm-profiler-orin)

Phase 2  ── 업그레이드 후 ────────────────────────────────────────────
  2.1 vLLM 기동 (8B → 35B) + 하네스 3종
  2.2 SGLang 기동 (8B → 35B) + 하네스 3종
  2.3 결과 문서 완성, 후보 문서 갱신, CLAUDE.md 갱신
```

각 Phase 종료 시 결과 문서에 해당 행이 채워져 있어야 다음 Phase로 간다. Phase 1이 지연되면 Phase 0의 산출물(하네스 + llama.cpp 수치 + FreeToken 판정)만으로도 독립적으로 가치가 있다.

---

## 5. 벤치 하네스 설계

### 5.1 위치와 구성

```
scripts/serving_bench/
  bench.py            # 하네스 본체. OpenAI 호환 /v1/chat/completions 스트리밍 클라이언트
  scenarios.yaml      # S1/S2/S3 정의 (프롬프트, 반복 수, 동시성)
  engines/
    llama_server.sh   # llama-server 기동 스크립트 (모델 키 → 명령)
    vllm.sh           # docker run ... vllm serve ...
    sglang.sh         # docker run ... sglang.launch_server ...
  monitor.sh          # tegrastats 샘플러 → CSV (RAM, GPU util, 온도, 전력)
  README.md
results/serving_bench/
  <engine>_<model>_<scenario>_<YYYYMMDD-HHMM>.jsonl   # 요청 단위 원시 기록
  summary.md                                          # 집계 표 (bench.py --summarize가 생성)
```

Orin에는 `~/serving_bench/`로 rsync해서 실행한다. 하네스는 **Orin 로컬**에서 돌린다 (Tailscale 왕복 지연이 TTFT에 섞이지 않게).

### 5.2 시나리오

공통: `temperature 0`, `max_tokens` 명시, thinking OFF, 스트리밍 ON, 측정 전 warm-up 1회 (결과 제외).

| ID | 워크로드 | 설정 | 지표 |
|---|---|---|---|
| **S1** | 1인 채팅 | 동시성 1. 프롬프트 5종 (짧은 질문 ~30 tok, 코드 설명 ~200 tok, 요약 ~800 tok) × 각 3회. `max_tokens 256` | TTFT, TPOT(=생성 tok/s), 총 지연. 프롬프트 길이별 분리 |
| **S2** | 에이전트 루프 | 동시성 1. **고정 시스템 프롬프트 ~4k tok** + 매 턴 다른 짧은 user 메시지 (~50 tok) × 20턴 순차. 대화 히스토리는 누적하지 않음 (매 요청 system + user 1개 — prefix 재사용 효과를 가장 순수하게 봄). `max_tokens 128` | 턴별 TTFT. 1턴 vs 2–20턴 평균의 비율 = prefix 캐시 효과. TPOT |
| **S3** | 동시성 | 동시성 1 → 2 → 4 → 8. 각 레벨에서 S1의 중간 길이 프롬프트를 동시성 × 4회. `max_tokens 256` | 집계 tok/s (모든 스트림 합), TTFT p50/p95, 실패/타임아웃 수 |

S2 변형 (선택, 시간 있으면): 히스토리 누적 버전 — 실제 채팅 세션에 가까움. 기본은 비누적.

### 5.3 지표 정의

| 지표 | 정의 | 측정 방법 |
|---|---|---|
| TTFT | 요청 전송 → 첫 content 토큰 수신 | 클라이언트 wall-clock. 스트리밍 첫 non-empty delta |
| TPOT | (총 생성 토큰 − 1) / (첫 토큰 → 마지막 토큰 시간) | 토큰 수는 응답 `usage.completion_tokens`; 없으면 클라이언트가 delta 수 카운트 (엔진별 토크나이저 차이 유의) |
| 집계 tok/s | S3: 전체 completion 토큰 합 / 첫 요청 시작 → 마지막 요청 종료 | |
| GPU 메모리 | `tegrastats` RAM 사용 (Orin은 통합 메모리라 GPU 전용 수치 없음). 엔진 기동 후 idle 값과 부하 중 peak | `monitor.sh` 1초 샘플 |
| 전력 | `tegrastats` VDD_GPU_SOC + VDD_CPU_CV 합, 부하 중 평균 | 참고용 (MODE_30W 상한 아래에서 얼마나 쓰는지) |

### 5.4 통제 변수

- **엔진 컨텍스트 길이**: 8192로 통일 (`-c 8192` / `--max-model-len 8192` / `--context-length 8192`). S2의 4k system + 출력이 들어가는 최소값. 더 크게 잡으면 vLLM/SGLang이 KV 캐시를 미리 잡아 메모리 비교가 왜곡됨.
- **vLLM `--gpu-memory-utilization`**: 통합 메모리라 기본 0.9는 호스트 RAM까지 다 잡는다. **0.5**부터 시작 (≈31GB), 35B GPTQ(20GB) + KV로 충분. 실패 시 조정하고 값을 기록.
- **SGLang `--mem-fraction-static`**: 같은 이유로 0.5 시작.
- **스레드**: llama-server `-t 8` (기존 chat.sh와 동일). vLLM/SGLang은 기본.
- **엔진별 최적화 옵션은 기본값 사용**. 튜닝은 이번 라운드 목표가 아님. 단 Qwen 문서가 명시하는 필수 플래그(`--quantization moe_wna16`, `--reasoning-parser qwen3`)는 적용.
- **동시에 한 엔진만 기동**. 이전 엔진 컨테이너/프로세스 종료 + 메모리 해제 확인(`free -g`) 후 다음.
- **실행 순서**: 각 (엔진, 모델) 조합마다 S1 → S2 → S3.

### 5.5 출력 형식

요청 단위 JSONL 한 줄:

```json
{"engine":"vllm","engine_version":"0.22.0","model":"qwen3.5-35b-a3b","format":"gptq-int4",
 "scenario":"S2","turn":7,"concurrency":1,"prompt_tokens":4103,"completion_tokens":128,
 "ttft_ms":412.3,"tpot_ms":91.8,"total_ms":12071.0,"ok":true,"error":null,
 "ts":"2026-09-02T14:03:11+09:00","power_mode":"MODE_30W","ctx":8192}
```

`bench.py --summarize results/serving_bench/` 가 `summary.md`를 생성: 엔진 × 모델 × 시나리오 표. S2는 턴별 TTFT 꺾은선 데이터도 CSV로.

### 5.6 검증

- **하네스 자체 검증**: llama-server 8B에 S1 1회 돌려서 TPOT이 `llama-bench tg32`(7.6 t/s)의 ±15% 안에 들어오면 통과. 벗어나면 토큰 카운팅 또는 스트리밍 파싱 버그.
- **엔진 sanity**: 각 엔진 기동 후 `curl /v1/models` + 짧은 completion으로 정상 응답 확인 후 하네스 시작.
- **thinking OFF 확인**: 응답에 `<think>`가 없고 completion_tokens가 max_tokens 근처가 아닌 짧은 답으로 끝나는지 첫 요청에서 육안 확인.

---

## 6. 엔진 × 모델 매트릭스

| 엔진 | 버전/이미지 | 8B | 35B-A3B | 기동 명령 (요지) |
|---|---|---|---|---|
| llama.cpp | `~/llama.cpp-build` @ `6fdd0ac` (Phase 0: CUDA 12.2 / Phase 1: 12.6 재빌드) | `Qwen3-8B-Q4_K_M.gguf` | `Qwen3.5-35B-A3B-Q4_K_M.gguf` | `llama-server -m M -ngl 99 -c 8192 -t 8 --port 8080 --reasoning off -np <N>` — S3에서 `-np 8` (병렬 슬롯), S1/S2는 `-np 1` |
| vLLM | `mitakad/vllm:0.22.0-r36.5.tegra-aarch64-cp312-cu129-24.04` (fallback: `jetson-containers build vllm` with `VLLM_VERSION` 최신) | `Qwen/Qwen3-8B` bf16 | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | `docker run --runtime nvidia --network host -v ~/hf:/root/.cache/huggingface IMG vllm serve MODEL --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.5 --reasoning-parser qwen3 [--quantization moe_wna16]` |
| SGLang | `mitakad/sglang` 최신 r36.x 태그 (fallback: `jetson-containers build sglang`) | 같음 | 같음 | `... python -m sglang.launch_server --model-path MODEL --port 30000 --context-length 8192 --mem-fraction-static 0.5 --reasoning-parser qwen3 [--quantization moe_wna16]` |
| FreeToken | spike only (§8) | — | — | — |

주의:
- llama-server의 `-np N`은 컨텍스트를 N등분한다. S3 `-np 8`이면 슬롯당 1024 → S3 프롬프트(~200 tok + 256 출력)는 들어가지만, S2와 S3의 컨텍스트 조건이 다르다는 점을 결과에 명시.
- llama-server는 prompt cache(`--cache-reuse`)가 기본 일부 동작하므로 S2에서 llama.cpp도 어느 정도 효과가 나올 수 있음. 기본값 그대로 두고 관찰.
- vLLM prefix caching은 v0.22 기본 ON. SGLang RadixAttention 기본 ON. 별도 플래그 없음.
- 8B에서 엔진이 정상 동작하는 것을 확인한 뒤에만 35B로 간다.

---

## 7. JetPack 6.0 → 6.2 업그레이드 Runbook

> **실행은 별도 승인 후.** 이 섹션은 실행 계획이며, 사용자가 시점을 정한다.
> 예상 소요: apt 40–60분 + 재부팅 + 검증 30분 + llama.cpp 재빌드 20분 ≈ **2시간**.
> 마지막 경로 검증: 2026-08-27. 실행 전 NVIDIA 문서에서 절차가 바뀌지 않았는지 재확인.

### 7.1 사전 조건 (하나라도 아니면 중단)

- [ ] `serial_log.py` 등 사용자 작업이 Orin에서 돌고 있지 않음 — `ps -eo pid,etime,args | grep -v grep | grep -E "python|docker"` 로 확인
- [ ] 사용자가 이 시점의 재부팅을 승인함
- [ ] Phase 0 완료 — 특히 **llama.cpp CUDA 12.2 기준선 수치가 결과 파일에 저장됨**
- [ ] NVMe 여유 ≥ 20GB (`df -h /`) — apt 캐시 + 새 CUDA
- [ ] 전원 안정 (정식 어댑터). 업그레이드 중 전원 차단은 부트 불능으로 이어질 수 있음
- [ ] 보드가 devkit인지 재확인: `cat /proc/device-tree/model` → `NVIDIA Jetson AGX Orin Developer Kit`
- [ ] **물리 접근 또는 시리얼 콘솔 확보 방안** — 재부팅 후 SSH가 안 뜨면 복구할 수단. Tailscale은 부트 후 네트워크가 올라와야 함

### 7.2 백업 (재부팅 전)

| 대상 | 방법 | 이유 |
|---|---|---|
| 패키지 목록 | `dpkg -l > ~/backup/dpkg_r36.3_$(date +%F).txt` | 롤백/비교 기준 |
| apt 소스 | `sudo cp -a /etc/apt/sources.list.d ~/backup/` | r36.3 복원용 |
| 부트 설정 | `sudo cp -a /boot/extlinux ~/backup/`; `cp /etc/nv_boot_control.conf ~/backup/` | 커널/DTB 업데이트 실패 시 참조 |
| 사용자 스크립트 | `cp ~/chat.sh ~/build_llama_orin.sh ~/convert_to_gguf.sh ~/rebuild_llama.sh ~/backup/` | 이미 리포에 미러가 있지만 로컬 사본도 |
| Docker 이미지 목록 | `docker images > ~/backup/docker_images.txt` | 재검증 체크리스트 |
| `~/models/`, `~/hf/` | **백업하지 않음** — NVMe 데이터 파티션이고 apt는 건드리지 않음. 다만 rootfs와 같은 파티션(`/`)이므로 파티션 손상 시 함께 잃는다. 재다운로드 가능한 자산으로 간주 |
| 리포 미러 | Mac 리포가 이미 스크립트/문서를 갖고 있음. 업그레이드 전 `git status` clean 확인 | |

**전체 이미지 백업은 하지 않는다.** rootfs 283GB를 덤프할 곳이 없고, devkit apt 경로는 NVIDIA 공식 지원이다. 대신 실패 시 복구 경로를 §7.6에 둔다. 사용자가 전체 백업을 원하면 외장 디스크가 필요하다 — 실행 전 확인.

### 7.3 실행

```bash
# 0. 현재 상태 기록
cat /etc/nv_tegra_release; nvpmodel -q; nvcc --version | tail -1   # nvcc: /usr/local/cuda/bin

# 1. apt 소스를 r36.4로
sudo sed -i 's/r36\.3/r36.4/g' /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
cat /etc/apt/sources.list.d/nvidia-l4t-apt-source.list   # 3줄 모두 r36.4 확인

# 2. 업그레이드 (NVIDIA JetPack 6.2 install-setup 문서 순서 그대로)
sudo apt update
sudo apt dist-upgrade            # 프롬프트: 설정 파일은 기본(N, 현재 유지) 선택. 로그 저장: | tee ~/backup/dist-upgrade.log
sudo apt install --fix-broken -o Dpkg::Options::="--force-overwrite"

# 3. (선택) JetPack 메타패키지로 CUDA/cuDNN/TRT 정합성 확보
sudo apt install nvidia-jetpack   # 현재 미설치 상태. 설치하면 12.6 스택이 한 번에 맞춰짐. 디스크 ~10GB

# 4. 재부팅 — 문서는 "physically reboot"라고 함. sudo reboot로 시작, 5분 내 SSH 안 뜨면 물리 전원 사이클
sudo reboot
```

`dist-upgrade` 중 `ERROR. Procedure for A_kernel-dtb update FAILED`가 나오면 → §7.6.

### 7.4 검증 (재부팅 후, 순서대로)

```bash
cat /etc/nv_tegra_release                 # R36 REVISION: 4.x
nvpmodel -q                               # MODE_30W 유지 확인 (업그레이드가 리셋하면 sudo nvpmodel -m 2)
/usr/local/cuda/bin/nvcc --version        # 12.6
dpkg -l | grep -E "libcudnn|tensorrt " | awk '{print $2,$3}'   # cuDNN 9.x, TRT 10.x
docker run --rm --runtime nvidia dustynv/nemo:r36.2.0 nvidia-smi 2>/dev/null || \
  docker run --rm --runtime nvidia ubuntu:22.04 true   # nvidia runtime 살아있는지
ls ~/models/*.gguf | wc -l                # 9
free -g; df -h /
```

### 7.5 업그레이드 후 작업

1. **llama.cpp 재빌드** — CUDA 12.6 링크. `rm -rf ~/llama.cpp-build/build-cuda && ~/rebuild_llama.sh` (소스는 그대로, `6fdd0ac` 유지 — 버전 변수를 하나만 바꾸기 위해 `git pull` 하지 않음). 이후 `~/chat.sh --bench qwen3.5-35b-a3b` → **CUDA 12.6 기준선** 기록.
2. **기존 컨테이너 기동 확인** — 각각 `docker run --rm --runtime nvidia IMG <간단 명령>`:
   - `sense-sdk/tensorrt-jetpack6.0.0:1.6.0-beta` — Cochl SDK. 컨테이너 내부 TRT 8.6이 호스트 드라이버 r36.4와 동작하는지. **실패하면 사용자에게 즉시 보고** — 이 이미지의 소유자/용도는 이 프로젝트 밖.
   - `riva-speech:2.19.0-l4t-aarch64`
   - `llm-profiler-orin:latest` — `/output/cuda/` 바이너리 실행
3. `pip3 list | grep tensorrt` — 호스트 pip TRT 8.6 바인딩이 시스템 TRT 10과 불일치. 이 프로젝트는 쓰지 않으므로 **건드리지 않고 기록만**.
4. CLAUDE.md "Orin state" 갱신.

### 7.6 실패 시나리오와 대응

| 증상 | 대응 |
|---|---|
| `A_kernel-dtb update FAILED` during dist-upgrade | NVIDIA 문서의 복구 명령은 `parted /dev/mmcblk0 ...` (eMMC 부트 기준). **이 Orin은 NVMe 부트** (`TEGRA_BOOT_STORAGE nvme0n1`) — 파티션 레이아웃이 다를 수 있으므로 **명령을 그대로 실행하지 말고** `lsblk`, `sudo parted /dev/mmcblk0 print`, `sudo parted /dev/nvme0n1 print` 출력을 확보한 뒤 NVIDIA 포럼 "NVMe boot dist-upgrade kernel-dtb" 사례와 대조. 재부팅하지 않은 상태에서 해결책을 찾는다 (문서: "Do NOT reboot between running these commands and retrying") |
| apt 의존성 깨짐 (`--fix-broken` 후에도) | `~/backup/dist-upgrade.log` 확보. `sudo apt --fix-broken install` 재시도. 해결 안 되면 소스를 r36.3으로 되돌리고 `apt update` — 이미 설치된 r36.4 패키지는 남지만 부팅 자체는 커널이 교체되지 않았으면 유지됨 |
| 재부팅 후 SSH 안 뜸 | 5분 대기 → 물리 전원 사이클 → 여전히 안 뜨면 시리얼 콘솔/모니터. 부트 로그에서 커널 패닉 vs 네트워크 문제 구분 |
| 부트 불능 | **SDK Manager로 재플래시** = rootfs 전체 소실 (`~/models` 포함, NVMe도 재플래시 대상이 될 수 있음). 이것이 §7.2에서 백업하지 않은 것의 최악 비용: GGUF 9개 (~58GB) + HF 모델 재다운로드 + 리포 스크립트 재배치 (~반나절). Cochl SDK 이미지는 재빌드/재수령 필요 — **재플래시 전 사용자 승인 필수** |
| CUDA 12.6인데 nvidia runtime 컨테이너 실패 | `nvidia-container-toolkit` 버전 확인, `sudo apt install --reinstall nvidia-container-toolkit` |
| sense-sdk 컨테이너 깨짐 | 이 프로젝트 범위 밖. 보고하고 진행. 롤백 근거가 되는지는 사용자 판단 |

### 7.7 롤백

apt 경로에는 **깨끗한 다운그레이드가 없다.** r36.4 → r36.3 `apt` 다운그레이드는 NVIDIA가 지원하지 않는다. 실질적 롤백은 재플래시(=전체 소실)뿐이다. 따라서 §7.1 사전 조건이 롤백 계획의 대부분이다. 실행 전 사용자가 이 비대칭을 인지해야 한다.

---

## 8. FreeToken Spike 프로토콜

목적: "안 된다"를 근거와 함께 기록. 결과물은 §9 결과 문서의 한 섹션. **상한 2시간, 초과 시 중단.**

```bash
# Orin, 호스트 Python 3.10, 격리 venv
python3 -m venv ~/ft-spike && source ~/ft-spike/bin/activate
pip install -U pip uv
git clone https://github.com/FlashML-org/FreeToken ~/ft-spike/src && cd ~/ft-spike/src
uv pip install -e .            # accel 없이. torch>=2.11 aarch64 CUDA 휠이 없어 여기서 막힐 확률 높음
# 만약 통과하면:
uv pip install -e ".[accel]"   # flashinfer[cu13] — 여기서 막힐 것이 거의 확실
# 만약 통과하면:
ft --help; ft serve --model Qwen/Qwen3.5-35B-A3B-FP8 --dry-run 2>&1 | tee ~/ft-spike/serve.log
```

기록할 것: 어느 단계에서, 어떤 에러로, 무엇이 없어서 막혔는지. 세 가지 가설(aarch64 torch 2.11 없음 / CUDA 13 필요 / FP8 커널 sm_87 미지원) 중 어디서 실제로 걸리는지. 이것이 "나중에 JetPack 7 / Thor에서 다시 볼 가치가 있는가"의 판단 근거가 된다.

venv는 `~/ft-spike/`에 격리. 호스트 Python 패키지를 건드리지 않는다. **삭제는 하지 않고** 사용자에게 보고.

---

## 9. 산출물과 성공 기준

### 산출물

| 파일 | 내용 |
|---|---|
| `scripts/serving_bench/` | 하네스 (§5.1). 재실행 가능 |
| `results/serving_bench/*.jsonl`, `summary.md` | 원시 데이터 + 집계 |
| `claudedocs/serving_framework_eval_<date>.md` | 결과 보고: 엔진×모델×시나리오 표, S2 TTFT 추이, 메모리, 셋업 비용/마찰 기록, FreeToken spike 결과, **권고** |
| `claudedocs/serving_framework_candidates_20260828.md` 갱신 | "Current state" 표 수정, FreeToken 판정 반영, 결과 문서 링크 |
| `CLAUDE.md` 갱신 | Orin state (JetPack 버전, 컨테이너 목록), Headline results, Next up |
| `docs/superpowers/plans/2026-08-xx-serving-framework-eval-phase{0,1,2}.md` | 실행 계획 (writing-plans로 작성) |

### 성공 기준

- 3개 엔진 × 2개 모델 × 3개 시나리오 = 18개 셀 중 **8B 6셀은 필수**, 35B는 엔진이 지원하는 만큼. 지원 안 되는 셀은 "왜"를 기록.
- llama.cpp는 CUDA 12.2와 12.6 두 기준선 보유.
- 결과 문서에 **워크로드별 권고**가 있다: "1인 채팅이면 X, 에이전트 루프면 Y, 동시 N명이면 Z" — 근거 수치 인용.
- FreeToken 실패 지점이 재현 가능한 로그로 기록.
- JetPack 업그레이드가 실행됐다면 §7.4 검증 전항목 통과 + §7.5 컨테이너 재검증 결과 기록.

---

## 10. 리스크

| 리스크 | 확률 | 영향 | 완화 |
|---|---|---|---|
| `mitakad/*` 이미지가 r36.4 호스트에서 안 돌거나 Qwen3.5 로딩 실패 | 중 | Phase 2 지연 | fallback `jetson-containers build` (2–4h/엔진). 하루 예산 추가 |
| vLLM/SGLang이 통합 메모리에서 OOM 또는 메모리 과점 | 중 | 35B 셀 손실 | `--gpu-memory-utilization` 0.5 → 0.4 → 0.35 단계 하향. 실패 값 기록 |
| `moe_wna16` 커널이 sm_87에서 느리거나 미지원 | 중 | 35B vLLM/SGLang 비교 불가 | 8B 결과로 엔진 오버헤드는 비교 가능. AWQ 변형 시도는 범위 외 |
| GGUF Q4_K_M vs GPTQ-Int4 양자화 차이가 속도 비교를 오염 | 확실 | 해석 주의 | 결과 문서에 명시. 8B는 llama.cpp Q4_K_M vs bf16이라 llama.cpp에 유리 — 이것도 명시 |
| JetPack 업그레이드가 Cochl SDK 컨테이너를 깨뜨림 | 저–중 | 프로젝트 외 피해 | §7.5 즉시 검증, 보고. 사전에 해당 이미지 사용 여부를 사용자에게 확인 |
| 업그레이드 중 부트 불능 → 재플래시 | 저 | 반나절 + 모델 재다운로드 | §7.1 전원/접근 조건, §7.6. 사용자가 비대칭(롤백 불가)을 인지한 상태에서 승인 |
| NVMe 부트 구성에서 kernel-dtb 업데이트 절차가 문서와 다름 | 중 | 업그레이드 중단 | §7.6 첫 행. 실행 전 NVIDIA 포럼 NVMe 사례 1건 이상 확인 |
| llama-server `-np 8`의 컨텍스트 분할이 S3 결과를 왜곡 | 저 | S3 해석 | §6 주의사항 기록. 필요시 `-c 16384 -np 8`로 재측정 |
| Qwen3.5 thinking이 어떤 엔진에서 안 꺼짐 | 저 | TPOT 왜곡 | §5.6 첫 요청 육안 확인. 안 꺼지면 `--reasoning-budget 0` 등 엔진별 대안 |

---

## 11. Open Questions (실행 전 사용자 확인)

1. **`sense-sdk/tensorrt-jetpack6.0.0` 이미지가 지금도 쓰이는가?** 쓰인다면 JetPack 업그레이드 전에 그 소유자와 조율 필요. 안 쓰이면 §7.5의 검증은 참고 수준.
2. **업그레이드 시점** — `serial_log.py` 작업이 끝나는 때. 사용자가 지정.
3. **전체 백업 여부** — §7.2는 하지 않는다고 가정. 외장 디스크를 붙여 rootfs를 덤프하길 원하면 Phase 1에 +1–2h.
4. **Qwen3-30B-A3B 재다운로드** (18GB GGUF) — April 최속 모델. 이번 비교에 넣으면 llama.cpp 행이 하나 더 생기지만 vLLM/SGLang용 GPTQ도 받아야 공정 (`Qwen/Qwen3-30B-A3B-GPTQ-Int4` 존재). 기본은 **넣지 않음**.
5. **S2 히스토리 누적 변형** 포함 여부 — 기본은 비누적만.

---

## Sources

- FreeToken: [GitHub](https://github.com/FlashML-org/FreeToken), `docs/install.md`, `docs/models.md`, `pyproject.toml` (2026-08-27 조회)
- [jetson-containers vLLM package](https://github.com/dusty-nv/jetson-containers/blob/master/packages/llm/vllm/README.md), [issue #1655 — vLLM on JetPack 6.0](https://github.com/dusty-nv/jetson-containers/issues/1655)
- [SGLang — NVIDIA Jetson Orin](https://docs.sglang.io/platforms/nvidia_jetson.html)
- [NVIDIA — JetPack 6.2 Install and Setup (apt upgrade path)](https://docs.nvidia.com/jetson/jetpack/6.2/install-setup/index.html)
- [NVIDIA forum — How to run vLLM ≥0.11 on AGX Orin](https://forums.developer.nvidia.com/t/how-to-run-vllm-0-11-0-on-jetson-agx-orin/353177)
- Docker Hub: [dustynv/vllm](https://hub.docker.com/r/dustynv/vllm/tags), [dustynv/sglang](https://hub.docker.com/r/dustynv/sglang/tags), [mitakad/vllm](https://hub.docker.com/r/mitakad/vllm/tags), [mitakad/sglang](https://hub.docker.com/r/mitakad/sglang/tags)
- [Qwen/Qwen3.5-35B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4)
- 후보 비교: `claudedocs/serving_framework_candidates_20260828.md`
- 기준선: `claudedocs/max_model_size_per_device_20260428.md`, `CLAUDE.md` Headline results
