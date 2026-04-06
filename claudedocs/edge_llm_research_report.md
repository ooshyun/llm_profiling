# Edge LLM Deployment Research Report

> Date: 2026-04-05 (Updated) | Depth: Deep | Confidence: 0.85
> Devices: Raspberry Pi 4, Jetson AGX Orin 64GB, OnePlus 13 (SD 8 Gen 3), OnePlus 11 (SD 8 Gen 2)
> Wearable Platforms: ESP32-S3 (MCU), Raspberry Pi Zero 2 W (SBC)

---

## 1. Device Profiling

### home.rasp4.local - Raspberry Pi 4 Model B Rev 1.5

| Spec | Value |
|------|-------|
| **CPU** | ARM Cortex-A72, 4 cores @ 1.8GHz |
| **Architecture** | aarch64 (ARMv8-A) |
| **RAM** | 3.7GB (MemTotal: 3,887,860 KB) |
| **Storage** | 15GB SD card (3.2GB available) |
| **GPU** | VideoCore VI (not usable for LLM) |
| **L1 Cache** | 128KB data + 192KB instruction |
| **L2 Cache** | 1MB shared |
| **OS** | Debian (kernel 6.12.47, aarch64) |
| **Temperature** | 50.6C (idle) |
| **Power** | ~8W peak during inference |

**Key Constraints**: CPU-only inference, ~3.5GB usable RAM (OS overhead), slow SD card I/O, no GPU acceleration for LLM.

---

### home.orin.local - NVIDIA Jetson AGX Orin Developer Kit

| Spec | Value |
|------|-------|
| **CPU** | ARM Cortex-A78AE, 12 cores (8 online) @ 2.2GHz |
| **GPU** | Ampere architecture, 2048 CUDA cores, 64 Tensor cores |
| **RAM** | 64GB unified (MemTotal: 64,349,340 KB) |
| **Storage** | 456GB NVMe SSD (235GB available) |
| **CUDA** | 12.2 |
| **JetPack** | 6.0 (L4T R36.3.0) |
| **Power Mode** | MODE_30W (current) |
| **GPU Clocks** | 306MHz - 1.3GHz (available) |
| **L1 Cache** | 512KB data + 512KB instruction |
| **L2/L3 Cache** | 2MB L2 + 4MB L3 |
| **AI Performance** | 275 TOPS (INT8, MAXN mode) |

**Key Strengths**: 64GB unified memory (CPU+GPU shared), CUDA 12.2, Tensor cores, NVMe storage. Can run models up to ~30B parameters comfortably.

---

### Android Phone #1 - OnePlus 13 (CPH2645)

| Spec | Value |
|------|-------|
| **SoC** | Qualcomm Snapdragon 8 Gen 3 (SM8650) |
| **CPU** | 8 cores: 2x Cortex-X4 @ 3.3GHz + 3x Cortex-A720 @ 3.15GHz + 2x Cortex-A720 @ 2.96GHz + 1x Cortex-A520 @ 2.27GHz |
| **GPU** | Adreno 750v2 (OpenCL 3.0, Vulkan 1.3) |
| **NPU** | Hexagon NPU (73 TOPS INT8) |
| **RAM** | 12GB LPDDR5X (MemTotal: 11,483,180 KB) |
| **Storage** | 219GB (204GB available) |
| **Android** | 15 (SDK 35) |
| **AI Features** | i8mm, bf16, NEON, AES, SHA, OpenCL, Vulkan, Hexagon DSP/HTP |
| **QNN Support** | HTP libs detected (via QnnHtp in vendor) |

**Key Strengths**: Top-tier mobile SoC. 73 TOPS NPU for dedicated LLM inference. Adreno 750v2 supports OpenCL backend for llama.cpp/MLC. 12GB RAM fits 3B-7B models (Q4).

**Key Constraints**: Thermal throttling under sustained load (S24 Ultra benchmark: throttle at iteration 6, GPU floor at 231MHz). 12GB shared with OS (~6-8GB available for LLM).

---

### Android Phone #2 - OnePlus 11 (CPH2451)

| Spec | Value |
|------|-------|
| **SoC** | Qualcomm Snapdragon 8 Gen 2 (SM8550) |
| **CPU** | 8 cores: 1x Cortex-X3 @ 3.19GHz + 4x Cortex-A715 @ 2.8GHz + 3x Cortex-A510 @ 2.02GHz |
| **GPU** | Adreno 740v2 @ 680MHz (OpenCL, Vulkan 1.3) |
| **NPU** | Hexagon NPU (~37 TOPS INT8) |
| **RAM** | 16GB LPDDR5X (MemTotal: 15,556,428 KB) |
| **Storage** | 218GB (204GB available) |
| **Android** | 13 (SDK 33) |
| **AI Features** | i8mm, bf16, NEON, AES, SHA, OpenCL, Vulkan, Hexagon DSP/HTP |
| **QNN Support** | QnnCpu, QnnHtp, QnnHtpPrepare, QnnSystem libs confirmed |

**Key Strengths**: 16GB RAM (more than OnePlus 13!) allows larger models. QNN libraries pre-installed and confirmed. Mature QNN HTP v73 support.

**Key Constraints**: NPU ~37 TOPS vs Gen 3's 73 TOPS (half the NPU performance). Older Android 13 may limit some frameworks.

---

### Wearable Platform #1 - Seeed XIAO ESP32-S3 Sense (추천)

| Spec | Value |
|------|-------|
| **MCU** | Espressif ESP32-S3 (Xtensa LX7 dual-core @ 240MHz) |
| **RAM** | 512KB SRAM + **8MB PSRAM** |
| **Flash** | 8MB |
| **Size** | **21 x 17.5mm** (엄지손가락 크기) |
| **Connectivity** | WiFi 2.4GHz + BLE 5.0 |
| **Sensors** | OV2640 카메라, 디지털 마이크, SD카드 슬롯 |
| **Power** | USB-C, 배터리 충전 지원 |
| **Price** | ~$14 |
| **Dev Environment** | Arduino IDE, PlatformIO, ESP-IDF (매우 쉬움) |

**왜 이 플랫폼인가**:
- 8MB PSRAM으로 ESP32 중 가장 큰 메모리 (260K~15M 파라미터 모델 구동 가능)
- Arduino/PlatformIO 지원으로 **개발 난이도 최저**
- 21x17.5mm 초소형으로 웨어러블 제품에 직접 내장 가능
- 카메라+마이크 내장으로 멀티모달 AI 프로토타이핑 가능
- TinyML (Edge Impulse) 공식 지원
- WiFi/BLE로 추론 결과 전송 또는 폰 연동 가능

**Key Constraints**: 8MB PSRAM = 최대 ~15M 파라미터 모델. 260K TinyLlama에서 19 tok/s 달성. Linux 없음 (베어메탈/RTOS).

---

### Wearable Platform #2 - Raspberry Pi Zero 2 W (추천)

| Spec | Value |
|------|-------|
| **CPU** | Broadcom BCM2710A1, Cortex-A53 quad-core @ 1GHz |
| **RAM** | **512MB** LPDDR2 |
| **Storage** | microSD (외장) |
| **Size** | **65 x 30mm** (RPi4의 ~1/3 면적) |
| **Connectivity** | WiFi 2.4GHz + BLE 4.2 |
| **GPIO** | 40-pin HAT 호환 |
| **Power** | USB micro, ~1.5W idle / ~3W active |
| **Price** | ~$15 |
| **Dev Environment** | Full Linux (Raspberry Pi OS), Python, C/C++ |

**왜 이 플랫폼인가**:
- **Full Linux** 지원으로 llama.cpp, PicoLM, Ollama 등 모든 프레임워크 사용 가능
- 512MB RAM이지만 PicoLM의 mmap 기술로 **TinyLlama 1.1B 구동 가능** (45MB RAM만 사용!)
- RPi 생태계 전체 활용 (카메라 모듈, HAT 확장보드, 방대한 문서)
- **개발 난이도 매우 낮음** (apt-get으로 패키지 설치, SSH로 원격 개발)
- WiFi로 웹 서버/API 제공 가능

**Key Constraints**: Cortex-A53 @ 1GHz는 Pi 4 대비 ~50% 느림. TinyLlama 1.1B에서 ~2 tok/s. SmolLM2-135M 권장.

---

### 플랫폼 비교표

| | **ESP32-S3 (XIAO)** | **Pi Zero 2 W** | **RPi 4 (참고)** |
|---|---|---|---|
| **크기** | 21x17.5mm | 65x30mm | 85x56mm |
| **무게** | ~3g | ~10g | ~46g |
| **RAM** | 8MB PSRAM | 512MB | 3.7GB |
| **CPU** | Xtensa 240MHz | A53 1GHz | A72 1.8GHz |
| **OS** | RTOS/Arduino | Full Linux | Full Linux |
| **최대 모델** | ~15M params | ~1.1B (mmap) | ~2B |
| **Best tok/s** | 19 (260K) | 2 (1.1B) | 3-5 (1.5B) |
| **전력** | ~0.5W | ~3W | ~8W |
| **가격** | $14 | $15 | $35+ |
| **개발 난이도** | Easy (Arduino) | Very Easy (Linux) | Very Easy (Linux) |
| **웨어러블 적합성** | **최적** | 소형 가전 | 데스크탑급 |

---

## 2. LLM Candidates - Top 3 per Device

### Raspberry Pi 4 (3.7GB RAM, CPU-only)

Target: Q4_K_M quantized models under 2.5GB RAM usage

| Rank | Model | Params | Quant Size | Expected tok/s | Why |
|------|-------|--------|-----------|----------------|-----|
| **1** | **Qwen2.5-1.5B-Instruct** | 1.5B | ~1.0GB (Q4_K_M) | 3-5 tok/s | Best balance of quality and speed in 1.5B class. Strong multilingual (Korean/English). Outperforms peers on reasoning benchmarks |
| **2** | **Gemma-2-2B-IT** | 2B | ~1.5GB (Q4_K_M) | 2-4 tok/s | Google's distilled knowledge architecture. Excellent for its size on general tasks. Tight fit for 3.7GB but feasible |
| **3** | **SmolLM2-1.7B-Instruct** | 1.7B | ~1.1GB (Q4_K_M) | 3-5 tok/s | HuggingFace's purpose-built small model. Good at instruction following. Memory efficient |

**Backup**: TinyLlama-1.1B (faster but lower quality), Phi-3.5-mini (3.8B, may be too large)

**Performance Reality Check**: Pi 4 Cortex-A72 is ~40% slower than Pi 5 Cortex-A76. Expect 1-5 tok/s for 1.5B models. Usable for chatbot/command parsing, not real-time conversation.

---

### Jetson AGX Orin 64GB (GPU + 64GB unified RAM)

Target: Maximize model quality with GPU acceleration

| Rank | Model | Params | VRAM Usage | Expected tok/s | Why |
|------|-------|--------|-----------|----------------|-----|
| **1** | **Qwen2.5-14B-Instruct** | 14B | ~10GB (Q4_K_M) | 40-60 tok/s | Excellent quality/speed ratio. Strong multilingual. vLLM W4A16 optimized for Orin |
| **2** | **Llama-3.1-8B-Instruct** | 8B | ~6GB (Q4_K_M) | 80-120 tok/s | Meta's flagship small model. Huge ecosystem. Well-optimized for TensorRT-LLM |
| **3** | **Deepseek-R1-Distill-Qwen-7B** | 7B | ~5GB (Q4_K_M) | ~180 tok/s (vLLM W4A16) | NVIDIA-validated 180.4 tok/s on AGX Orin. Best raw throughput. Strong reasoning |

**Bonus - Large Model Option**: Qwen2.5-32B (Q4) fits in 64GB unified memory at ~16 tok/s. Maximum quality for offline/batch tasks.

**Performance Notes**:
- Current power mode is 30W. Switching to MAXN (60W) will significantly boost throughput
- vLLM with W4A16 quantization is NVIDIA's recommended stack for Orin
- Deepseek-R1-Distill-Qwen-7B: 180.4 tok/s is NVIDIA's official benchmark (vLLM, W4A16, concurrency=8)

---

### OnePlus 13 - Snapdragon 8 Gen 3 (12GB RAM)

Available compute: CPU (8-core), GPU (Adreno 750v2), NPU (Hexagon 73 TOPS)

#### CPU Inference (llama.cpp ARM NEON)

| Rank | Model | Params | Quant Size | Expected tok/s | Why |
|------|-------|--------|-----------|----------------|-----|
| **1** | **Qwen2.5-3B-Instruct** | 3B | ~2.0GB (Q4_K_M) | 8-12 tok/s | Best quality in 3B class. Strong multilingual. Fits comfortably in 12GB |
| **2** | **Llama-3.2-3B-Instruct** | 3B | ~2.0GB (Q4_K_M) | 8-12 tok/s | Meta's edge-optimized model. Excellent ecosystem support |
| **3** | **Phi-3.5-mini-Instruct** | 3.8B | ~2.5GB (Q4_K_M) | 5-8 tok/s | Microsoft's strong performer. 128K context. Tight but feasible |

#### GPU Inference (MLC-LLM OpenCL / llama.cpp Adreno OpenCL)

| Rank | Model | Params | Expected tok/s | Framework | Why |
|------|-------|--------|----------------|-----------|-----|
| **1** | **Qwen2.5-1.5B-Instruct** | 1.5B | 15-25 tok/s | MLC-LLM | GPU excels at smaller models with full offload. Fast prefill |
| **2** | **Gemma-2-2B-IT** | 2B | 12-18 tok/s | MLC-LLM | Good GPU utilization. Compact architecture |
| **3** | **Llama-3.2-3B-Instruct** | 3B | 8-15 tok/s | llama.cpp OpenCL | Adreno 750v2 OpenCL backend. Q4_0 optimized |

#### NPU Inference (Qualcomm Genie / QNN HTP)

| Rank | Model | Params | Expected tok/s | Framework | Why |
|------|-------|--------|----------------|-----------|-----|
| **1** | **Llama-3.2-3B-Instruct** | 3B | 10-20 tok/s (decode) / 1000+ (prefill) | Qualcomm Genie + AI Hub | NVIDIA-class prefill via NPU. Official Qualcomm support |
| **2** | **Qwen2.5-1.5B-Instruct** | 1.5B | 15-30 tok/s | ExecuTorch QNN | ExecuTorch QNN backend validated on SM8650 |
| **3** | **Phi-3.5-mini-Instruct** | 3.8B | 5-10 tok/s | Qualcomm AI Hub | Qualcomm AI Hub compilation support. Strong reasoning |

**Key Insight**: NPU prefill is 50x faster than CPU/GPU (mllm-NPU: 1000+ tok/s for 1.5B), but decode speed is only marginally better. Best strategy: **NPU for prefill + CPU/GPU for decode** (heterogeneous computing).

**Thermal Warning**: Samsung S24 Ultra (same SoC) benchmarks show GPU frequency floor at 78.3C after ~6 sustained iterations. Plan for thermal throttling in production use.

---

### OnePlus 11 - Snapdragon 8 Gen 2 (16GB RAM)

Available compute: CPU (8-core), GPU (Adreno 740v2), NPU (Hexagon ~37 TOPS)

#### CPU Inference (llama.cpp ARM NEON)

| Rank | Model | Params | Quant Size | Expected tok/s | Why |
|------|-------|--------|-----------|----------------|-----|
| **1** | **Llama-3.2-3B-Instruct** | 3B | ~2.0GB (Q4_K_M) | 6-10 tok/s | 16GB RAM headroom. Good ARM optimization |
| **2** | **Qwen2.5-3B-Instruct** | 3B | ~2.0GB (Q4_K_M) | 6-10 tok/s | Strong multilingual. Plenty of RAM |
| **3** | **Llama-3.1-8B-Instruct** | 8B | ~5.0GB (Q4_K_M) | 3-4 tok/s | 16GB makes 8B feasible! Unique advantage over 12GB phone |

#### GPU Inference (MLC-LLM OpenCL)

| Rank | Model | Params | Expected tok/s | Framework | Why |
|------|-------|--------|----------------|-----------|-----|
| **1** | **Qwen2.5-1.5B-Instruct** | 1.5B | 12-18 tok/s | MLC-LLM | Full GPU offload, fast generation |
| **2** | **Gemma-2-2B-IT** | 2B | 10-15 tok/s | MLC-LLM | Compact, GPU-friendly |
| **3** | **Llama-3.2-3B-Instruct** | 3B | 6-12 tok/s | llama.cpp OpenCL | Adreno 740v2 OpenCL Q4_0 support |

#### NPU Inference (QNN HTP)

| Rank | Model | Params | Expected tok/s | Framework | Why |
|------|-------|--------|----------------|-----------|-----|
| **1** | **Qwen2.5-1.5B-Instruct** | 1.5B | 10-20 tok/s | QNN HTP (v73 confirmed) | QNN libs pre-installed. Validated HTP backend |
| **2** | **Llama-3.2-3B-Instruct** | 3B | 8-15 tok/s | Qualcomm Genie | Official Qualcomm support for Gen 2 |
| **3** | **Gemma-2-2B-IT** | 2B | 8-12 tok/s | ExecuTorch QNN | ExecuTorch Qualcomm backend |

**16GB Advantage**: OnePlus 11's 16GB RAM uniquely enables **8B models on CPU** (Llama-3.1-8B at 3-4 tok/s). Not fast, but functional for offline/batch tasks that need higher quality.

**QNN Library Status**: `libQnnCpu.so`, `libQnnHtp.so`, `libQnnHtpPrepare.so`, `libQnnSystem.so` all confirmed present — ready for NPU inference out of the box.

---

### ESP32-S3 (XIAO Sense, 8MB PSRAM)

Target: 베어메탈/RTOS 환경, 최대 ~15M 파라미터

| Rank | Model | Params | Memory | Expected tok/s | Framework | Why |
|------|-------|--------|--------|----------------|-----------|-----|
| **1** | **TinyStories-260K** (llama2.c) | 260K | ~1MB | **19.13 tok/s** | llama2.c (ESP-DSP 최적화) | ESP32 검증 완료. 듀얼코어 SIMD 최적화. 스토리/문장 생성 |
| **2** | **Tiny15M** (llama2.c) | 15M | ~6MB | **4.5 tok/s** (추정) | llama2.c | 8MB PSRAM에 들어가는 최대 모델. 더 풍부한 텍스트 생성 |
| **3** | **Custom TinyLLM** (30-50M, 양자화) | 30-50M | ~4-8MB (INT4) | 2-5 tok/s (추정) | SynapEdge / ONNX-to-C | ONNX→C 변환으로 최적화. 커스텀 학습 가능 |

**사용 사례**: 스마트홈 명령 파싱, 센서 데이터 요약, 간단한 대화, 키워드/의도 분류
**개발 방법**: Arduino IDE에서 `llama2.c` 포팅 → ESP-DSP 라이브러리로 SIMD 가속 → PSRAM에 모델 로드

---

### Raspberry Pi Zero 2 W (512MB RAM)

Target: Linux 환경, mmap으로 큰 모델도 가능

| Rank | Model | Params | RAM Usage | Expected tok/s | Framework | Why |
|------|-------|--------|-----------|----------------|-----------|-----|
| **1** | **SmolLM2-135M-Instruct** | 135M | ~100MB (Q4) | **5-8 tok/s** | llama.cpp | 512MB에 안정적으로 구동. Instruct 튜닝 완료. 가장 실용적 |
| **2** | **TinyLlama-1.1B** (mmap) | 1.1B | **45MB RAM** (mmap) | **~2 tok/s** | **PicoLM** | 혁신적: 638MB 모델을 디스크에서 스트리밍. 45MB RAM만 사용! |
| **3** | **RWKV-4-Pile-169M** | 169M | ~200MB | ~3 tok/s (추정) | RWKV Python | Transformer 대비 메모리 효율적 (RNN 아키텍처). 긴 컨텍스트 가능 |

**핵심 프레임워크 - PicoLM**:
- 80KB 바이너리, 외부 의존성 제로
- TinyLlama 1.1B를 45MB RAM으로 구동 (mmap으로 모델을 한 레이어씩 스트리밍)
- ARM NEON SIMD 가속
- JSON 출력 지원 (--json 플래그)
- KV 캐시 영속화 (--cache)로 반복 prefill 생략

**LLMStick 프로젝트 참고 벤치마크** (Pi Zero W, 더 느린 ARMv6):
| Model | Params | Speed |
|-------|--------|-------|
| Tiny15M | 15M | 223ms/token (~4.5 tok/s) |
| Lamini-T5-Flan-77M | 77M | 2.5s/token (~0.4 tok/s) |
| SmolLM2-136M | 136M | 2.2s/token (~0.45 tok/s) |

*Pi Zero 2 W는 ARMv8 Cortex-A53으로 위 수치의 3-5배 빠름*

---

## 3. Benchmark Data Summary

### Raspberry Pi 4 Performance Tiers (llama.cpp, Q4_K_M)

| Model Size | Expected tok/s | RAM Usage | Usability |
|-----------|----------------|-----------|-----------|
| ≤135M params | >15 tok/s | <500MB | Real-time capable |
| 135M-500M | 5-15 tok/s | 500MB-1GB | Interactive |
| 1B-1.5B | 2-5 tok/s | 1-1.5GB | Slow but usable |
| 2B-3B | 1-3 tok/s | 1.5-2.5GB | Barely usable |
| ≥7B | <1 tok/s | Needs swap | Not recommended |

### Jetson AGX Orin 64GB Performance (NVIDIA Official + Community)

| Model | Framework | Quantization | tok/s | Source |
|-------|-----------|-------------|-------|--------|
| Deepseek-R1-Distill-Qwen-7B | vLLM | W4A16 | 180.4 | NVIDIA official |
| Deepseek-R1-Distill-Qwen-32B | vLLM | W4A16 | 16.96 | NVIDIA official |
| gpt-oss-20b | vLLM | - | ~40 | Community |
| Llama2-7B | MLC | - | ~19 | Community |
| Mistral-7B | llama.cpp | - | 20-30 | Community |

### Android Phone Benchmarks (Research-Based)

#### Qwen 2.5 1.5B (Q4) on Snapdragon 8 Gen 3 (Galaxy S24 Ultra / OnePlus 13 class)

| Backend | Framework | tok/s (decode) | Prefill tok/s | Thermal | Source |
|---------|-----------|---------------|--------------|---------|--------|
| GPU (Adreno 750) | MLC-LLM OpenCL | 9.93 ± 0.79 | ~40 | Throttles at iter 6 (78.3C) | arXiv 2603.23640 |
| CPU | llama.cpp | 8-12 | ~11 | Stable | arXiv 2410.03613 |
| NPU (Hexagon HTP) | mllm-NPU | ~10-15 (decode) | **1000+** | Low power | arXiv 2407.05858 |
| NPU (Hexagon HTP) | Qualcomm Genie | ~10-20 | Fast | Low power | Qualcomm official |

#### Llama 2 7B (Q4) on Snapdragon 8 Gen 3

| Backend | Framework | tok/s (decode) | Notes |
|---------|-----------|---------------|-------|
| CPU | llama.cpp | 3-4 | Feasible on 12GB+ devices |
| NPU | Qualcomm Genie | ~5 | 8B models at 5 tok/s (Qualcomm claim) |
| Mixed CPU+GPU | PowerInfer-2 | ~11.68 (Mixtral-47B!) | Speculative, neuron-aware |

#### Key Finding: Mobile Thermal Throttling

| Device Class | Peak tok/s | Sustained tok/s | Degradation | Time to Throttle |
|-------------|-----------|-----------------|-------------|-----------------|
| SD 8 Gen 3 (GPU) | ~10 | Crash at iter 6 | -100% (OS kill) | ~5 min |
| iPhone 16 Pro (GPU) | 40.35 | 22.56 | -44% | ~8 iterations |
| Hailo-10H NPU (RPi5) | 6.914 | 6.914 | 0% | Never | 

**Critical Insight**: NPU inference is thermally stable (0% degradation). GPU inference on mobile devices suffers severe throttling. **NPU is the only viable backend for sustained mobile LLM inference.**

---

## 4. Serving Frameworks Comparison

### For Raspberry Pi 4

| Framework | Recommendation | Pros | Cons |
|-----------|---------------|------|------|
| **llama.cpp** | **Best choice** | Pure C++, no deps, ARM NEON optimized, lowest overhead, GGUF format | CLI-based, limited API |
| **Ollama** | Good for ease of use | Built on llama.cpp, REST API, model management | 10-20% overhead vs raw llama.cpp |
| **MLC LLM** | Alternative | TVM-compiled, can optimize for specific ARM | More complex setup |

**Verdict**: llama.cpp is the clear winner for Pi 4. Minimal overhead, maximum performance on constrained hardware.

### For Jetson AGX Orin 64GB

| Framework | Recommendation | Pros | Cons |
|-----------|---------------|------|------|
| **vLLM** | **Best for throughput** | NVIDIA-validated, W4A16 quant, paged KV cache, concurrent requests | Python overhead, higher memory baseline |
| **TensorRT-LLM** | Best for latency | Maximum optimization via TensorRT, INT8/FP16 kernels | Complex setup, model-specific compilation |
| **llama.cpp (CUDA)** | Best for simplicity | Easy setup, CUDA backend, good single-user perf | Lower throughput than vLLM for concurrent |
| **Ollama** | Best for prototyping | One-command setup, CUDA auto-detect | Not optimized for Orin specifically |

**Verdict**: vLLM with W4A16 quantization for production serving. llama.cpp for quick prototyping. TensorRT-LLM for maximum optimization if you need lowest latency.

### For Android Phones (Snapdragon 8 Gen 2 / Gen 3)

| Framework | Backend | Pros | Cons | Best For |
|-----------|---------|------|------|----------|
| **llama.cpp (Adreno OpenCL)** | CPU + GPU | Qualcomm-optimized OpenCL backend. Q4_0 tuned. Active development. Easy setup | Q4_0 only for GPU. Thermal throttling on GPU | **Best all-round choice** |
| **MLC-LLM** | GPU (OpenCL/Vulkan) | TVM-compiled for target GPU. Good prefill. Android APK available | Thermal throttling. Slower decode than CPU on some models | GPU-first inference |
| **Qualcomm Genie** | NPU (Hexagon HTP) | Official Qualcomm NPU engine. Lowest power. Thermally stable. Frees CPU/GPU | Requires AI Hub compilation. Limited model support. Q4 only | **Best for sustained inference** |
| **ExecuTorch + QNN** | NPU/CPU/GPU | Meta's production framework. 50KB base. Llama 3 3B validated on SD 8 Gen 3 | Complex build process. Newer framework | Production mobile apps |
| **PowerInfer-2** | CPU + NPU | 24.6x faster than LLMFlash. Runs 47B models on phone! 40% less memory | Research prototype. Complex setup | Large model experiments |
| **mllm-NPU** | NPU (Hexagon HTP) | 1000+ tok/s prefill. 3-5x faster than PowerInfer-V2-NPU | Research stage. Limited model support | Maximum NPU performance |
| **Ollama (Android)** | CPU | Familiar interface. Easy model management | No GPU/NPU support on Android. CPU-only | Quick prototyping |

**Verdict by Use Case**:
- **Quick start**: llama.cpp with Adreno OpenCL backend
- **Sustained serving (production)**: Qualcomm Genie on NPU (no thermal throttling)
- **Maximum throughput**: mllm-NPU for prefill + CPU for decode (heterogeneous)
- **App integration**: ExecuTorch + QNN backend
- **Large model on mobile**: PowerInfer-2 (experimental)

### Android Deployment Quick Start

```bash
# Option 1: llama.cpp with Adreno OpenCL (easiest)
# Build llama.cpp with GGML_OPENCL=ON for Android
# Push to device:
adb push llama-cli /data/local/tmp/
adb push qwen2.5-3b-instruct-q4_0.gguf /data/local/tmp/
adb shell "/data/local/tmp/llama-cli -m /data/local/tmp/qwen2.5-3b-instruct-q4_0.gguf -ngl 99 -t 4"

# Option 2: Qualcomm Genie (NPU, best sustained perf)
# 1. Compile model via Qualcomm AI Hub: qai_hub_models --model llama-3.2-3b --target sm8650
# 2. Deploy QNN context binary to device
# 3. Run via Genie runtime SDK

# Option 3: MLC-LLM (GPU, app-ready)
# Install MLC Chat APK from GitHub releases
# Download pre-compiled model from MLC model zoo
```

### For ESP32-S3 (Wearable MCU)

| Framework | Pros | Cons | Best For |
|-----------|------|------|----------|
| **llama2.c (ESP-DSP 최적화)** | 검증 완료 19 tok/s. 듀얼코어 SIMD. Arduino 호환 | 260K 모델만 검증. 커스텀 빌드 필요 | **최고 성능** |
| **SynapEdge Compiler** | ONNX→C 변환. 다양한 모델 지원 | 새로운 프로젝트. 문서 부족 | 커스텀 모델 배포 |
| **Edge Impulse + TFLite Micro** | GUI 기반 학습/배포. 공식 XIAO 지원 | LLM보다 분류/감지 특화 | 의도 분류, 키워드 |
| **ESP-NN (Espressif)** | 공식 Espressif 최적화. S3 SIMD 활용 | 범용 추론 엔진은 아님 | 뉴럴넷 가속 |

### For Raspberry Pi Zero 2 W (Wearable SBC)

| Framework | Pros | Cons | Best For |
|-----------|------|------|----------|
| **PicoLM** | **45MB RAM으로 1.1B 구동!** mmap 스트리밍. 80KB 바이너리. JSON 출력 | 디스크 I/O 의존. SD카드 속도 중요 | **최대 모델 크기** |
| **llama.cpp** | 범용. ARM NEON 최적화. GGUF 포맷 | 512MB RAM 한계로 작은 모델만 | 135M-500M 모델 |
| **Llamafile** | 단일 실행파일. 4x throughput vs Ollama | 큰 바이너리 | 간편 배포 |
| **picoLLM (Picovoice)** | 상용급. 다양한 모델 지원. 2/3/4비트 양자화 | 유료 라이센스. Pi Zero 공식 미지원 | 상용 제품 |

---

## 5. Wearable / Embedded-Targeted Frameworks & Models

### Ultra-Small Inference Frameworks

| Framework | Target | Footprint | Key Feature |
|-----------|--------|-----------|-------------|
| **ExecuTorch** (Meta) | MCU to smartphone | 50KB base | 12+ HW backends, production-ready (v1.0), powers Meta apps |
| **TinyChatEngine** (MIT) | Edge devices | Minimal | On-device LLM inference library, INT4/INT8 support |
| **TinyEngine** (TI NPU) | MCUs (ESP32, STM32) | HW-integrated | 2.56 GOPS, 90x latency reduction, 120x energy reduction |
| **Cactus** (YC-backed) | Mobile/wearables | Small | Sub-50ms TTFT, cross-platform, privacy-first |
| **llm.c / llm.rs** | Minimal environments | Tiny | Single-file inference, educational + production |
| **ONNX Runtime Mobile** | Cross-platform | ~1MB | Quantized model support, broad HW support |

### Ultra-Small LLMs for Wearables/Embedded

| Model | Params | Memory | tok/s (est.) | Target Use |
|-------|--------|--------|-------------|-----------|
| **MobileLLM** (Meta) | 125M | ~250MB | 50 tok/s (iPhone) | Basic tasks, command parsing |
| **Gemma-3-270M** | 270M | ~200MB (Q4) | 30+ tok/s | Extreme efficiency tasks |
| **Qwen2.5-0.5B** | 500M | ~350MB (Q4) | 15-20 tok/s | General purpose tiny |
| **BitNet b1.58** | 2B | 400MB (native) | 8+ tok/s | CPU-only, no quant needed |
| **TinyLLM custom** | 30-120M | <100MB | 20+ tok/s on MCU | IoT/sensor NLP, smart-home |
| **SmolLM2-135M** | 135M | ~100MB (Q4) | 15+ tok/s (Pi4) | Ultra-lightweight tasks |

### Additional Wearable/Embedded Frameworks (2026 Update)

| Framework | Target | Key Innovation | Status |
|-----------|--------|---------------|--------|
| **mllm-NPU** (SJTU) | Snapdragon NPU | 1000+ tok/s prefill, shadow outlier execution | Research (2024) |
| **PowerInfer-2** (SJTU) | Smartphone CPU+NPU | 47B model on phone, neuron-aware scheduling | Research (2024) |
| **LiteRT + QNN Accelerator** (Google) | Qualcomm NPU | TFLite successor, native QNN delegate | Production (2025) |
| **Qualcomm AI Hub** | Snapdragon devices | Cloud compilation → on-device QNN binary | Production (2025) |
| **WatchPrompt** (concept) | Smartwatches | Prompt signature matching, zero inference | Concept |
| **Samsung Health Companion** | Galaxy Watch | <200KB TinyML, keyword spotting + BT relay | Production |

### Ultra-Constrained Hardware Reference

| Hardware | CPU | RAM | Storage | Power | Model Limit | tok/s |
|----------|-----|-----|---------|-------|-------------|-------|
| nRF52840 ($6) | Cortex-M4 64MHz | 256KB | 1MB Flash | CR2032 8mo | <50KB model | keyword only |
| ESP32-S3 ($4) | Xtensa 240MHz | 512KB+8MB PSRAM | 16MB Flash | ~200mW | 30-50M params | 5-10 |
| RPi Pico 2 ($5) | Cortex-M33 150MHz | 520KB | 4MB Flash | ~100mW | 10-30M params | 3-5 |
| STM32H7 ($10) | Cortex-M7 480MHz | 1MB | 2MB Flash | ~500mW | 50-100M params | 5-10 |

### Wearable Deployment Stack Recommendation (Updated)

```
MCU Tier (ESP32, STM32, nRF52):
  Framework: TinyEngine NPU / ExecuTorch / TFLite Micro
  Model: TinyLLM 30-120M custom or keyword-only (<50KB)
  Use: Voice commands, sensor NLP, intent classification
  Power: <500mW, battery months

Smartwatch Tier (WearOS, Galaxy Watch):
  Framework: LiteRT (TFLite) / ExecuTorch
  Model: MobileLLM 125M or Qwen2.5-0.5B (Q4)
  Use: Smart reply, health summaries, command parsing
  Strategy: Keyword spotting local + BT relay to phone for heavy inference

Smartphone Tier (Snapdragon, Apple, Dimensity):
  Framework: Qualcomm Genie (NPU) / llama.cpp (CPU+GPU) / MLC-LLM (GPU)
  Model: Qwen2.5-3B or Llama-3.2-3B (Q4)
  Use: Full chatbot, translation, code assist, multimodal
  Critical: Use NPU for sustained inference (no thermal throttling)

SBC Tier (Raspberry Pi, Orange Pi):
  Framework: llama.cpp
  Model: Qwen2.5-1.5B or SmolLM2-1.7B (Q4_K_M)
  Use: Chatbot, code assist, translation, home automation

Edge GPU Tier (Jetson, Coral):
  Framework: vLLM / TensorRT-LLM
  Model: Qwen2.5-14B or Llama-3.1-8B
  Use: Full conversational AI, multi-modal, robotics, real-time
```

---

## 6. Recommendations

### Immediate Actions

**For home.rasp4.local (Pi 4, 3.7GB)**:
1. Install llama.cpp (build from source for ARM NEON optimization)
2. Download Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
3. Run with: `./llama-server -m qwen2.5-1.5b-instruct-q4_k_m.gguf -c 2048 -t 4`
4. Expect: ~3-5 tok/s generation

**For home.orin.local (Jetson AGX Orin 64GB)**:
1. Switch to MAXN power mode: `sudo nvpmodel -m 0 && sudo jetson_clocks`
2. Install vLLM via jetson-containers
3. Serve Deepseek-R1-Distill-Qwen-7B with W4A16 for max throughput (~180 tok/s)
4. Or serve Qwen2.5-14B for better quality (~40-60 tok/s)

**For OnePlus 13 (Snapdragon 8 Gen 3, 12GB)**:
1. Install llama.cpp built with `GGML_OPENCL=ON` for Adreno 750v2 GPU
2. Push Qwen2.5-3B-Instruct-Q4_0.gguf to device
3. Run with GPU offload: `llama-cli -m model.gguf -ngl 99 -t 4`
4. Expect: ~10-12 tok/s (CPU+GPU), with thermal throttling after ~5 min
5. For sustained use: Compile model via Qualcomm AI Hub → deploy with Genie on NPU

**For OnePlus 11 (Snapdragon 8 Gen 2, 16GB)**:
1. Same llama.cpp OpenCL setup (Adreno 740v2 supported)
2. **Unique advantage**: 16GB RAM allows Llama-3.1-8B-Instruct (Q4_K_M) on CPU
3. Push Llama-3.1-8B-Instruct-Q4_K_M.gguf (~5GB) to device
4. Run CPU-only: `llama-cli -m model.gguf -t 8` → ~3-4 tok/s
5. For NPU: QNN libs already installed (`libQnnHtp.so` confirmed) → ExecuTorch QNN or Genie ready

**For ESP32-S3 XIAO Sense (Wearable, 8MB PSRAM)**:
1. Arduino IDE 또는 PlatformIO 설치
2. [esp32-llm](https://github.com/DaveBben/esp32-llm) 프로젝트 클론
3. ESP-DSP 라이브러리 설치 (SIMD 가속)
4. TinyStories-260K 모델 빌드 → 플래시 업로드
5. Expect: **19 tok/s** (260K), ~4.5 tok/s (15M)
6. 시리얼 모니터 또는 BLE로 추론 결과 수신

**For Raspberry Pi Zero 2 W (소형 SBC, 512MB)**:
1. Raspberry Pi OS Lite (64-bit) 설치
2. PicoLM 빌드: `git clone https://github.com/RightNow-AI/picolm && make`
3. TinyLlama-1.1B GGUF 모델 다운로드 (638MB, SD카드에 저장)
4. 실행: `./picolm -m tinyllama-1.1b-q4_k.gguf -t 4`
5. Expect: **~2 tok/s** (1.1B) with only 45MB RAM usage!
6. 또는 SmolLM2-135M으로 더 빠른 응답: ~5-8 tok/s

### Power Mode Impact (Orin)

| Mode | Power | Expected Performance Boost |
|------|-------|---------------------------|
| MODE_30W (current) | 30W | Baseline |
| MODE_50W | 50W | ~1.5x |
| MAXN | 60W | ~2x |

---

## 7. All Devices Summary

| Device | Best Model | Framework | Expected tok/s | Compute | RAM Available |
|--------|-----------|-----------|---------------|---------|--------------|
| **RPi 4** | Qwen2.5-1.5B (Q4_K_M) | llama.cpp | 3-5 | CPU | ~3.5GB |
| **Jetson AGX Orin** | DS-R1-Qwen-7B (W4A16) | vLLM | ~180 | GPU CUDA | 64GB unified |
| **OnePlus 13** | Qwen2.5-3B (Q4_0) | llama.cpp OpenCL | 10-12 | CPU+GPU | ~8GB |
| **OnePlus 13** (NPU) | Llama-3.2-3B | Qualcomm Genie | 10-20 | NPU | ~8GB |
| **OnePlus 11** | Llama-3.1-8B (Q4_K_M) | llama.cpp | 3-4 | CPU | ~10GB |
| **OnePlus 11** (NPU) | Qwen2.5-1.5B | QNN HTP | 10-20 | NPU | ~10GB |
| **ESP32-S3 XIAO** | TinyStories-260K | llama2.c (ESP-DSP) | **19** | MCU dual-core | 8MB PSRAM |
| **Pi Zero 2 W** | TinyLlama-1.1B (mmap) | PicoLM | ~2 | CPU (A53) | 512MB (45MB used) |
| **Pi Zero 2 W** (fast) | SmolLM2-135M (Q4) | llama.cpp | 5-8 | CPU (A53) | 512MB |

---

## Sources

- [NVIDIA Jetson Benchmarks](https://developer.nvidia.com/embedded/jetson-benchmarks)
- [Jetson AGX Orin Token Speed (NVIDIA Forum)](https://forums.developer.nvidia.com/t/the-token-speed-of-llm-on-jetson-agx-orin/343901)
- [LLMs on Raspberry Pi 5 (ItsFOSS)](https://itsfoss.com/llms-for-raspberry-pi/)
- [LLM Evaluation on SBCs (arXiv 2511.07425)](https://arxiv.org/html/2511.07425v1)
- [On-Device LLMs: State of the Union 2026](https://v-chandra.github.io/on-device-llms/)
- [Edge LLM Deployment Guide 2025](https://medium.com/@kodekx-solutions/edge-llm-deployment-on-small-devices-the-2025-guide-2eafb7c59d07)
- [Best Open Source LLMs for Raspberry Pi 2026](https://www.siliconflow.com/articles/en/best-open-source-LLMs-for-Raspberry-Pi)
- [LLMs on Budget: RPi4 Framework Testing](https://medium.com/@thomasnahon/llms-on-a-budget-testing-serving-frameworks-on-the-raspberry-pi-4-5fc56623840e)
- [vLLM vs llama.cpp (Red Hat)](https://developers.redhat.com/articles/2025/09/30/vllm-or-llamacpp-choosing-right-llm-inference-engine-your-use-case)
- [TinyChatEngine (MIT)](https://github.com/mit-han-lab/TinyChatEngine)
- [ExecuTorch](https://executorch.ai/)
- [Cactus On-Device Inference](https://www.infoq.com/news/2025/12/cactus-on-device-inference/)
- [TI TinyEngine MCU AI](https://www.ti.com/about-ti/newsroom/news-releases/2026/2026-03-10-ti-expands-microcontroller-portfolio-and-software-ecosystem-to-enable-edge-ai-in-every-device.html)
- [Best Small LLMs for Edge 2026](https://www.siliconflow.com/articles/en/best-small-llms-for-edge-devices)
- [Running LLMs with TensorRT-LLM on Jetson AGX Orin](https://www.hackster.io/shahizat/running-llms-with-tensorrt-llm-on-nvidia-jetson-agx-orin-34372f)
- [vLLM on Jetson Deployment Guide](https://learnopencv.com/deployment-on-edge-vllm-on-jetson/)
- [LLM Inference at the Edge: Mobile NPU GPU Trade-offs (arXiv 2603.23640)](https://arxiv.org/html/2603.23640v1)
- [LLM Performance on Mobile Platforms (arXiv 2410.03613)](https://arxiv.org/html/2410.03613v3)
- [mllm-NPU: 1000 tokens/sec Prefilling (arXiv 2407.05858)](https://arxiv.org/html/2407.05858v1/)
- [PowerInfer-2: Fast LLM on Smartphone (arXiv 2406.06282)](https://arxiv.org/html/2406.06282v3)
- [Qualcomm Genie NPU Inference](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Unlocking-on-device-generative-AI-with-an-NPU-and-heterogeneous-computing.pdf)
- [llama.cpp Adreno OpenCL Backend (Qualcomm)](https://www.qualcomm.com/developer/blog/2024/11/introducing-new-opn-cl-gpu-backend-llama-cpp-for-qualcomm-adreno-gpu)
- [ExecuTorch Qualcomm QNN Backend](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html)
- [Qualcomm AI Hub - LLM Chat Android](https://aihub.qualcomm.com/apps/chatapp_android)
- [MLC-LLM OpenCL Profiling on Android](https://www.callstack.com/blog/profiling-mlc-llms-opencl-backend-on-android-performance-insights)
- [On-Device AI Chat with Qualcomm Genie, MLC, WebLLM](https://pub.towardsai.net/on-device-ai-chat-translate-on-android-qualcomm-genie-mlc-webllm-your-phone-your-llm-49594aff3b9f)
- [LiteRT QNN Accelerator for Qualcomm NPU (Google)](https://developers.googleblog.com/unlocking-peak-performance-on-qualcomm-npu-with-litert/)
- [Smallest LLMs for Wearables 2026](https://www.alibaba.com/product-insights/is-it-possible-to-run-an-open-source-llm-on-a-smartwatch-and-what-tasks-actually-work.html)
- [Best Small SLMs 2026 (BentoML)](https://www.bentoml.com/blog/the-best-open-source-small-language-models)
- [ESP32-S3 LLM - Running LLM on ESP32 (GitHub)](https://github.com/DaveBben/esp32-llm)
- [PicoLM - 1B LLM on $10 board with 256MB RAM (GitHub)](https://github.com/RightNow-AI/picolm)
- [LLMStick - LLM USB device on Pi Zero W](https://www.cnx-software.com/2025/02/20/llmstick-an-ai-and-llm-usb-device-based-on-raspberry-pi-zero-w-and-optimized-llama-cpp/)
- [llm-pi-zero - LLMs in tiny box under 3W (GitHub)](https://github.com/fxlin/llm-pi-zero)
- [Seeed XIAO ESP32S3 Sense](https://www.seeedstudio.com/XIAO-ESP32S3-Sense-p-5639.html)
- [Run Tiny Language Model on ESP32 (Hackster)](https://www.hackster.io/asadshafi5/run-tiny-language-model-genai-on-esp32-8b5dd8)
- [Google Coral NPU - Open Source IP for Wearables](https://developers.googleblog.com/en/introducing-coral-npu-a-full-stack-platform-for-edge-ai/)
- [Embedded ML Trends 2026](https://promwad.com/news/embedded-ml-trends-2026)
- [TinyML on ESP32 (XDA)](https://www.xda-developers.com/tinyml-impressive-software-esp32/)
- [picoLLM Inference Engine (Picovoice)](https://picovoice.ai/picollm/)
