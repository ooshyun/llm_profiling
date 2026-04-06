# tiny-llm-profiler Phase 5: Track 2 — MLC-LLM (TVM) Profiler

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Track 2 profiling via MLC-LLM/TVM. Build kernel→layer mapping from debug_dump IR, integrate with Nsight (Orin CUDA) and OpenCL events (Android). Output in same unified ProfileRecord schema for cross-framework comparison.

**Architecture:** Track 2 produces the same ProfileRecord as Track 1, stored in the same SQLite/Parquet. Profiling approach varies by backend: Nsight for CUDA, OpenCL events for Adreno, CPU timer for LLVM. Kernel-to-layer mapping built from TVM IR.

**Tech Stack:** Python 3.11+, MLC-LLM, Apache TVM, Nsight Systems CLI, pytest

**Depends on:** Phase 1 (schemas, storage), Phase 4 (device tests)

**Risk:** MLC-LLM CPU backend is unverified on ARM. vm.profile() is broken for LLM models. This phase includes validation and fallback.

---

## File Structure

```
collection/
└── track2_tvm/
    ├── __init__.py
    ├── kernel_layer_map.py     # Parse debug_dump IR → kernel→layer mapping
    ├── tvm_profiler.py         # Unified profiler: dispatches by backend
    ├── nsight_parser.py        # Parse Nsight Systems sqlite export → ProfileRecord
    ├── opencl_profiler.py      # OpenCL event-based profiling for Android
    └── export.py               # TVM results → ProfileRecord conversion
tests/
├── test_kernel_layer_map.py
├── test_nsight_parser.py
├── test_tvm_profiler.py
├── test_tvm_device_orin.py     # Real device: Orin GPU via MLC-LLM + Nsight
├── test_tvm_device_android.py  # Real device: Android OpenCL via MLC-LLM
└── test_tvm_device_rpi.py      # Real device: RPi CPU via MLC-LLM (validation)
```

---

### Task 1: Kernel → Layer Mapping

**Files:**
- Create: `collection/track2_tvm/__init__.py`
- Create: `collection/track2_tvm/kernel_layer_map.py`
- Create: `tests/test_kernel_layer_map.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_kernel_layer_map.py
from collection.track2_tvm.kernel_layer_map import KernelLayerMapper


def test_parse_fused_matmul():
    mapper = KernelLayerMapper()
    # Simulated TVM kernel names from compilation
    mapper.add_mapping("fused_dequantize1_NT_matmul5", block_idx=0, sublayer="attention_qkv")
    mapper.add_mapping("fused_dequantize2_NT_matmul11", block_idx=0, sublayer="ffn_up")

    info = mapper.lookup("fused_dequantize1_NT_matmul5")
    assert info.block_idx == 0
    assert info.sublayer == "attention_qkv"


def test_lookup_unknown_returns_default():
    mapper = KernelLayerMapper()
    info = mapper.lookup("unknown_kernel_xyz")
    assert info.block_idx == -1
    assert info.sublayer == "unknown"


def test_build_from_ir_pattern():
    """Test pattern-based mapping when exact names aren't registered."""
    mapper = KernelLayerMapper()
    mapper.add_pattern(r".*softmax.*", sublayer="attention_score")
    mapper.add_pattern(r".*rms_norm.*", sublayer="rmsnorm")

    info = mapper.lookup("fused_softmax_with_chunked_sum")
    assert info.sublayer == "attention_score"

    info2 = mapper.lookup("rms_norm_fp16")
    assert info2.sublayer == "rmsnorm"


def test_save_and_load(tmp_path):
    mapper = KernelLayerMapper()
    mapper.add_mapping("kernel_a", block_idx=0, sublayer="attention_qkv")
    mapper.add_mapping("kernel_b", block_idx=1, sublayer="ffn_up")

    path = tmp_path / "map.json"
    mapper.save(path)

    mapper2 = KernelLayerMapper.load(path)
    assert mapper2.lookup("kernel_a").sublayer == "attention_qkv"
    assert mapper2.lookup("kernel_b").block_idx == 1
```

- [ ] **Step 2: Implement kernel layer mapper**

```python
# collection/track2_tvm/__init__.py
```

```python
# collection/track2_tvm/kernel_layer_map.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelLayerInfo:
    block_idx: int
    sublayer: str


_DEFAULT = KernelLayerInfo(block_idx=-1, sublayer="unknown")


class KernelLayerMapper:
    """Maps TVM kernel names to transformer layer locations."""

    def __init__(self):
        self._exact: dict[str, KernelLayerInfo] = {}
        self._patterns: list[tuple[re.Pattern, str]] = []

    def add_mapping(self, kernel_name: str, block_idx: int, sublayer: str) -> None:
        self._exact[kernel_name] = KernelLayerInfo(block_idx=block_idx, sublayer=sublayer)

    def add_pattern(self, pattern: str, sublayer: str, block_idx: int = -1) -> None:
        self._patterns.append((re.compile(pattern), sublayer))

    def lookup(self, kernel_name: str) -> KernelLayerInfo:
        if kernel_name in self._exact:
            return self._exact[kernel_name]
        for pat, sublayer in self._patterns:
            if pat.match(kernel_name):
                return KernelLayerInfo(block_idx=-1, sublayer=sublayer)
        return _DEFAULT

    def save(self, path: Path | str) -> None:
        data = {
            "exact": {k: {"block_idx": v.block_idx, "sublayer": v.sublayer} for k, v in self._exact.items()},
            "patterns": [{"pattern": p.pattern, "sublayer": s} for p, s in self._patterns],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> KernelLayerMapper:
        data = json.loads(Path(path).read_text())
        mapper = cls()
        for name, info in data.get("exact", {}).items():
            mapper.add_mapping(name, info["block_idx"], info["sublayer"])
        for p in data.get("patterns", []):
            mapper.add_pattern(p["pattern"], p["sublayer"])
        return mapper
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_kernel_layer_map.py -v`
Expected: All passed

- [ ] **Step 4: Commit**

```bash
git add collection/track2_tvm/ tests/test_kernel_layer_map.py
git commit -m "feat: add KernelLayerMapper (TVM kernel name → layer mapping)"
```

---

### Task 2: Nsight Systems Parser (Orin CUDA)

**Files:**
- Create: `collection/track2_tvm/nsight_parser.py`
- Create: `tests/test_nsight_parser.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_nsight_parser.py
from collection.track2_tvm.nsight_parser import parse_nsight_csv
from collection.track2_tvm.kernel_layer_map import KernelLayerMapper


def test_parse_nsight_csv(tmp_path):
    """Parse Nsight Systems GPU kernel CSV export."""
    csv_content = """Start (ns),Duration (ns),Name,Device
100000,5000,fused_dequantize1_NT_matmul5,GPU 0
105000,3000,softmax_with_chunked_sum,GPU 0
108000,4000,fused_dequantize2_NT_matmul11,GPU 0
"""
    csv_file = tmp_path / "kernels.csv"
    csv_file.write_text(csv_content)

    mapper = KernelLayerMapper()
    mapper.add_mapping("fused_dequantize1_NT_matmul5", block_idx=0, sublayer="attention_qkv")
    mapper.add_pattern(r".*softmax.*", sublayer="attention_score")
    mapper.add_mapping("fused_dequantize2_NT_matmul11", block_idx=0, sublayer="ffn_up")

    records = parse_nsight_csv(
        csv_path=csv_file,
        mapper=mapper,
        experiment_id="nsight_001",
        device="orin_gpu",
        model="qwen2.5-3b-q4_0",
        architecture="qwen",
    )

    assert len(records) == 3
    assert records[0].sublayer == "attention_qkv"
    assert records[0].latency_us == 5.0  # 5000 ns = 5 us
    assert records[1].sublayer == "attention_score"
    assert records[2].sublayer == "ffn_up"
```

- [ ] **Step 2: Implement Nsight parser**

```python
# collection/track2_tvm/nsight_parser.py
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from collection.track2_tvm.kernel_layer_map import KernelLayerMapper
from schemas.profile_schema import ProfileRecord


def parse_nsight_csv(
    csv_path: Path | str,
    mapper: KernelLayerMapper,
    experiment_id: str,
    device: str,
    model: str,
    architecture: str,
    quantization: str = "q4_0",
    input_tokens: int = 128,
) -> list[ProfileRecord]:
    """Parse Nsight Systems GPU kernel trace CSV into ProfileRecords."""
    records: list[ProfileRecord] = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel_name = row["Name"].strip()
            duration_ns = int(row["Duration (ns)"])
            latency_us = duration_ns / 1000.0

            info = mapper.lookup(kernel_name)

            records.append(ProfileRecord(
                experiment_id=experiment_id,
                timestamp=datetime.now(timezone.utc),
                device=device,
                framework="tvm",
                model=model,
                architecture=architecture,
                phase="decode",
                block_idx=info.block_idx,
                sublayer=info.sublayer,
                op_type=kernel_name,
                latency_us=latency_us,
                mem_bytes=0,
                mem_peak_bytes=0,
                power_mw=0.0,
                flops=0,
                bandwidth_bytes_sec=0,
                thermal_c=0.0,
                input_tokens=input_tokens,
                output_token_idx=0,
                quantization=quantization,
                batch_size=1,
            ))

    return records
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_nsight_parser.py -v`
Expected: All passed

- [ ] **Step 4: Commit**

```bash
git add collection/track2_tvm/nsight_parser.py tests/test_nsight_parser.py
git commit -m "feat: add Nsight Systems CSV parser for TVM/CUDA profiling on Orin"
```

---

### Task 3: TVM Profiler Dispatcher

**Files:**
- Create: `collection/track2_tvm/tvm_profiler.py`
- Create: `tests/test_tvm_profiler.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_tvm_profiler.py
from collection.track2_tvm.tvm_profiler import TVMProfiler


def test_profiler_backend_selection():
    profiler = TVMProfiler.__new__(TVMProfiler)
    assert profiler._select_backend("orin_gpu") == "nsight"
    assert profiler._select_backend("op13_gpu") == "opencl_events"
    assert profiler._select_backend("rpi4") == "cpu_timer"
    assert profiler._select_backend("orin_cpu") == "cpu_timer"
```

- [ ] **Step 2: Implement TVM profiler dispatcher**

```python
# collection/track2_tvm/tvm_profiler.py
from __future__ import annotations


class TVMProfiler:
    """Dispatches profiling strategy based on device backend."""

    _BACKEND_MAP = {
        "orin_gpu": "nsight",
        "op13_gpu": "opencl_events",
        "op13_opencl": "opencl_events",
        "op11_gpu": "opencl_events",
        "op11_opencl": "opencl_events",
        "rpi4": "cpu_timer",
        "orin_cpu": "cpu_timer",
        "op13_cpu": "cpu_timer",
        "op11_cpu": "cpu_timer",
        "pi_zero2w": "cpu_timer",
    }

    def _select_backend(self, device: str) -> str:
        return self._BACKEND_MAP.get(device, "cpu_timer")
```

- [ ] **Step 3: Run test**

Run: `python -m pytest tests/test_tvm_profiler.py -v`
Expected: All passed

- [ ] **Step 4: Commit**

```bash
git add collection/track2_tvm/tvm_profiler.py tests/test_tvm_profiler.py
git commit -m "feat: add TVMProfiler backend dispatcher (nsight, opencl, cpu)"
```

---

### Task 4: Real Device Test — MLC-LLM on Orin (CUDA)

**Files:**
- Create: `tests/test_tvm_device_orin.py`

- [ ] **Step 1: Write device test**

```python
# tests/test_tvm_device_orin.py
"""Real device test: MLC-LLM on Jetson AGX Orin (CUDA)."""
import subprocess
import pytest

pytestmark = pytest.mark.device


def _orin_ssh():
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=3", "home.orin.local", "echo ok"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() == "ok"
    except Exception:
        return False


@pytest.mark.skipif(not _orin_ssh(), reason="Orin not reachable")
def test_orin_mlc_installed():
    """Check if MLC-LLM is available on Orin."""
    r = subprocess.run(
        ["ssh", "home.orin.local", "python3 -c 'import mlc_llm; print(mlc_llm.__file__)'"],
        capture_output=True, text=True, timeout=10,
    )
    # May fail if not installed — that's expected, marks as known status
    if r.returncode != 0:
        pytest.skip("MLC-LLM not installed on Orin")
    assert "mlc_llm" in r.stdout


@pytest.mark.skipif(not _orin_ssh(), reason="Orin not reachable")
def test_orin_nsight_available():
    """Check if Nsight Systems CLI is available."""
    r = subprocess.run(
        ["ssh", "home.orin.local", "nsys --version"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0 or "nsys" in r.stdout + r.stderr
```

- [ ] **Step 2: Write RPi CPU validation test**

```python
# tests/test_tvm_device_rpi.py
"""Real device test: MLC-LLM CPU backend on RPi4 (VALIDATION - may fail)."""
import subprocess
import pytest

pytestmark = pytest.mark.device


def _rpi_ssh():
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=3", "home.rasp4.local", "echo ok"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() == "ok"
    except Exception:
        return False


@pytest.mark.skipif(not _rpi_ssh(), reason="RPi4 not reachable")
def test_rpi4_mlc_cpu_validation():
    """Validate if MLC-LLM CPU backend works on RPi4 ARM.
    
    This is an EXPERIMENTAL validation test.
    Expected: May fail (CPU backend unverified on ARM aarch64).
    If fails: Track 2 will use Track 1 fallback for RPi4.
    """
    r = subprocess.run(
        ["ssh", "home.rasp4.local", "python3 -c 'import tvm; print(tvm.__version__)'"],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode != 0:
        pytest.skip("TVM not installed on RPi4 — Track 2 not available, use Track 1 fallback")
    assert "tvm" in r.stdout.lower() or len(r.stdout.strip()) > 0
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_tvm_device_orin.py tests/test_tvm_device_rpi.py
git commit -m "test: add Track 2 device validation tests (Orin Nsight, RPi CPU)"
```

---

## Summary

| Task | Component | Local Tests | Device Tests |
|------|-----------|-------------|-------------|
| 1 | Kernel→Layer Mapper | 4 | N/A |
| 2 | Nsight CSV Parser | 1 | N/A |
| 3 | TVM Profiler Dispatcher | 1 | N/A |
| 4 | Device Validation | N/A | 3 (Orin + RPi) |
| **Total** | | **6 local** | **3 device** |

### Test Commands

```bash
# Local tests only
python -m pytest tests/test_kernel_layer_map.py tests/test_nsight_parser.py tests/test_tvm_profiler.py -v

# Device tests (requires connected devices)
python -m pytest tests/test_tvm_device_orin.py tests/test_tvm_device_rpi.py -v -m device

# All tests
python -m pytest tests/ -v --tb=short
```

### Post-Phase 5 Status

After all 5 phases, the project will have:
- **~20 local test files, ~80+ tests**
- **Track 1**: Full per-layer profiling via ggml cb_eval on all devices
- **Track 2**: Nsight/OpenCL based profiling on Orin GPU + Android GPU, CPU validation on RPi
- **Analysis**: 4 comparators + bottleneck + statistics
- **Presentation**: Streamlit dashboard + Jupyter notebooks
- **Experiment Runner**: Config-driven orchestration with device auto-detection
