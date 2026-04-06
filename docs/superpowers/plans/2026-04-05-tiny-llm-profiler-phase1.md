# tiny-llm-profiler Phase 1: Core Infrastructure

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the core data pipeline — schema, storage, platform monitors, transport, and the ggml profiler — so that we can run a single profiling experiment on RPi4 and collect per-layer data into SQLite/Parquet.

**Architecture:** Layered design with 4 layers (Collection → Transport → Storage → query). This phase covers Layers 1-2. Track 1 (ggml cb_eval) is the profiling mechanism. Platform Monitor runs in a separate thread. Transport handles local JSON + central collection.

**Tech Stack:** Python 3.11+, C (ggml callback), Pydantic, SQLite, PyArrow/Parquet, pytest, llama.cpp (submodule or system install)

**Spec:** `docs/superpowers/specs/2026-04-05-tiny-llm-profiler-design.md`
**Architecture Ref:** `docs/superpowers/specs/2026-04-05-llama-cpp-architecture.md`

---

## File Structure

```
tiny-llm-profiler/
├── pyproject.toml                          # Project config, dependencies
├── schemas/
│   ├── __init__.py
│   └── profile_schema.py                   # Pydantic models: ProfileRecord, PlatformSnapshot
├── collection/
│   ├── __init__.py
│   ├── track1_ggml/
│   │   ├── CMakeLists.txt                  # Build config for C profiler
│   │   ├── ggml_profiler.c                 # cb_eval callback + JSON export
│   │   ├── ggml_profiler.h                 # Header for callback types
│   │   └── tensor_name_parser.py           # "blk.5.attn_q.weight" → (block_idx, sublayer)
│   └── platform_monitor/
│       ├── __init__.py
│       ├── base.py                         # ABC: PlatformMonitor
│       ├── rpi.py                          # RPi 4 / Pi Zero implementation
│       └── jetson.py                       # Jetson Orin implementation
├── transport/
│   ├── __init__.py
│   ├── local_store.py                      # Write ProfileRecords to local JSON
│   └── collector.py                        # SCP/rsync collection from remote devices
├── storage/
│   ├── __init__.py
│   ├── db.py                               # SQLite read/write
│   └── parquet_store.py                    # Parquet read/write
├── tests/
│   ├── __init__.py
│   ├── test_schema.py
│   ├── test_tensor_name_parser.py
│   ├── test_platform_monitor.py
│   ├── test_local_store.py
│   ├── test_db.py
│   └── test_parquet_store.py
└── configs/
    ├── devices.yaml
    └── experiments.yaml
```

---

### Task 1: Project Scaffold + Schema

**Files:**
- Create: `pyproject.toml`
- Create: `schemas/__init__.py`
- Create: `schemas/profile_schema.py`
- Create: `tests/__init__.py`
- Create: `tests/test_schema.py`

- [ ] **Step 1: Initialize git repo and pyproject.toml**

```bash
cd /Users/seunghyunoh/workplace/research/tiny_llm
git init
```

```toml
# pyproject.toml
[project]
name = "tiny-llm-profiler"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "pyarrow>=14.0",
    "pandas>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write failing test for ProfileRecord**

```python
# tests/test_schema.py
from datetime import datetime, timezone
from schemas.profile_schema import ProfileRecord, PlatformSnapshot


def test_profile_record_creation():
    record = ProfileRecord(
        experiment_id="exp_001",
        timestamp=datetime.now(timezone.utc),
        device="rpi4",
        framework="ggml",
        model="qwen2.5-1.5b-q4_k_m",
        architecture="qwen",
        phase="decode",
        block_idx=5,
        sublayer="attention_qkv",
        op_type="matmul",
        latency_us=1234.5,
        mem_bytes=4096,
        mem_peak_bytes=8192,
        power_mw=4200.0,
        flops=16777216,
        bandwidth_bytes_sec=1000000000,
        cache_l1_hit_ratio=None,
        cache_l2_hit_ratio=None,
        thermal_c=52.3,
        gpu_sm_util=None,
        gpu_tensor_util=None,
        input_tokens=128,
        output_token_idx=10,
        quantization="q4_k_m",
        batch_size=1,
    )
    assert record.device == "rpi4"
    assert record.block_idx == 5
    assert record.latency_us == 1234.5


def test_profile_record_to_dict():
    record = ProfileRecord(
        experiment_id="exp_001",
        timestamp=datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc),
        device="orin_gpu",
        framework="ggml",
        model="qwen2.5-3b-q4_0",
        architecture="qwen",
        phase="prefill",
        block_idx=0,
        sublayer="ffn_up",
        op_type="matmul",
        latency_us=500.0,
        mem_bytes=2048,
        mem_peak_bytes=2048,
        power_mw=30000.0,
        flops=33554432,
        bandwidth_bytes_sec=2000000000,
        cache_l1_hit_ratio=0.68,
        cache_l2_hit_ratio=0.12,
        thermal_c=45.0,
        gpu_sm_util=0.85,
        gpu_tensor_util=0.42,
        input_tokens=128,
        output_token_idx=0,
        quantization="q4_0",
        batch_size=1,
    )
    d = record.model_dump()
    assert d["device"] == "orin_gpu"
    assert d["gpu_sm_util"] == 0.85
    assert isinstance(d["timestamp"], datetime)


def test_platform_snapshot_creation():
    snap = PlatformSnapshot(
        timestamp=datetime.now(timezone.utc),
        cpu_temp_c=52.3,
        gpu_temp_c=None,
        soc_temp_c=None,
        power_mw=4200.0,
        cpu_freq_mhz=[1800.0, 1800.0, 1800.0, 1800.0],
        gpu_freq_mhz=None,
        mem_used_bytes=2_000_000_000,
        mem_available_bytes=1_700_000_000,
    )
    assert snap.cpu_temp_c == 52.3
    assert snap.gpu_temp_c is None
    assert len(snap.cpu_freq_mhz) == 4
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/seunghyunoh/workplace/research/tiny_llm && python -m pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'schemas'`

- [ ] **Step 4: Implement schema**

```python
# schemas/__init__.py
```

```python
# schemas/profile_schema.py
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel


class ProfileRecord(BaseModel):
    # Identification
    experiment_id: str
    timestamp: datetime
    device: str
    framework: str
    model: str
    architecture: str

    # Location
    phase: str
    block_idx: int
    sublayer: str
    op_type: str

    # Metrics A-E
    latency_us: float
    mem_bytes: int
    mem_peak_bytes: int
    power_mw: float
    flops: int
    bandwidth_bytes_sec: int

    # Metrics F-J
    cache_l1_hit_ratio: float | None = None
    cache_l2_hit_ratio: float | None = None
    thermal_c: float = 0.0
    gpu_sm_util: float | None = None
    gpu_tensor_util: float | None = None

    # Context
    input_tokens: int
    output_token_idx: int
    quantization: str
    batch_size: int


class PlatformSnapshot(BaseModel):
    timestamp: datetime
    cpu_temp_c: float
    gpu_temp_c: float | None = None
    soc_temp_c: float | None = None
    power_mw: float
    cpu_freq_mhz: list[float]
    gpu_freq_mhz: float | None = None
    mem_used_bytes: int
    mem_available_bytes: int
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_schema.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml schemas/ tests/
git commit -m "feat: add project scaffold and core Pydantic schemas (ProfileRecord, PlatformSnapshot)"
```

---

### Task 2: Tensor Name Parser

**Files:**
- Create: `collection/__init__.py`
- Create: `collection/track1_ggml/__init__.py`
- Create: `collection/track1_ggml/tensor_name_parser.py`
- Create: `tests/test_tensor_name_parser.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tensor_name_parser.py
from collection.track1_ggml.tensor_name_parser import parse_tensor_name, LayerInfo


def test_parse_attention_q():
    info = parse_tensor_name("blk.5.attn_q.weight")
    assert info.block_idx == 5
    assert info.sublayer == "attention_qkv"


def test_parse_attention_k():
    info = parse_tensor_name("blk.12.attn_k.weight")
    assert info.block_idx == 12
    assert info.sublayer == "attention_qkv"


def test_parse_attention_v():
    info = parse_tensor_name("blk.0.attn_v.weight")
    assert info.block_idx == 0
    assert info.sublayer == "attention_qkv"


def test_parse_attention_output():
    info = parse_tensor_name("blk.3.attn_output.weight")
    assert info.block_idx == 3
    assert info.sublayer == "attention_out"


def test_parse_ffn_gate():
    info = parse_tensor_name("blk.7.ffn_gate.weight")
    assert info.block_idx == 7
    assert info.sublayer == "ffn_gate"


def test_parse_ffn_up():
    info = parse_tensor_name("blk.7.ffn_up.weight")
    assert info.block_idx == 7
    assert info.sublayer == "ffn_up"


def test_parse_ffn_down():
    info = parse_tensor_name("blk.7.ffn_down.weight")
    assert info.block_idx == 7
    assert info.sublayer == "ffn_down"


def test_parse_attn_norm():
    info = parse_tensor_name("blk.0.attn_norm.weight")
    assert info.block_idx == 0
    assert info.sublayer == "attn_norm"


def test_parse_ffn_norm():
    info = parse_tensor_name("blk.2.ffn_norm.weight")
    assert info.block_idx == 2
    assert info.sublayer == "ffn_norm"


def test_parse_output_norm():
    info = parse_tensor_name("output_norm.weight")
    assert info.block_idx == -1
    assert info.sublayer == "output_norm"


def test_parse_output_weight():
    info = parse_tensor_name("output.weight")
    assert info.block_idx == -1
    assert info.sublayer == "lm_head"


def test_parse_token_embd():
    info = parse_tensor_name("token_embd.weight")
    assert info.block_idx == -1
    assert info.sublayer == "embedding"


def test_parse_rope():
    info = parse_tensor_name("blk.5.attn_q-rope")
    assert info.block_idx == 5
    assert info.sublayer == "rope"


def test_parse_unknown():
    info = parse_tensor_name("some_random_tensor")
    assert info.block_idx == -1
    assert info.sublayer == "unknown"


def test_parse_ggml_op_node():
    """ggml operation nodes may have names like 'norm-0' or 'KQ_soft_max-5'"""
    info = parse_tensor_name("KQ_soft_max-5")
    assert info.sublayer == "attention_score"


def test_parse_result_node():
    info = parse_tensor_name("result_output-0")
    assert info.block_idx == -1
    assert info.sublayer == "lm_head"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tensor_name_parser.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement tensor name parser**

```python
# collection/__init__.py
```

```python
# collection/track1_ggml/__init__.py
```

```python
# collection/track1_ggml/tensor_name_parser.py
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LayerInfo:
    block_idx: int
    sublayer: str


# blk.{N}.{component}[.weight]
_BLK_PATTERN = re.compile(r"^blk\.(\d+)\.([a-z_]+)")

# Component name → canonical sublayer name
_COMPONENT_MAP: dict[str, str] = {
    "attn_q": "attention_qkv",
    "attn_k": "attention_qkv",
    "attn_v": "attention_qkv",
    "attn_qkv": "attention_qkv",
    "attn_output": "attention_out",
    "attn_norm": "attn_norm",
    "ffn_gate": "ffn_gate",
    "ffn_up": "ffn_up",
    "ffn_down": "ffn_down",
    "ffn_norm": "ffn_norm",
}

# Top-level (non-block) tensor names
_TOPLEVEL_MAP: dict[str, str] = {
    "output_norm": "output_norm",
    "output": "lm_head",
    "token_embd": "embedding",
}

# ggml operation node name patterns (e.g., "KQ_soft_max-5", "result_output-0")
_OP_NODE_PATTERNS: list[tuple[str, str, int]] = [
    ("KQ_soft_max", "attention_score", -1),
    ("KQ_mask", "attention_score", -1),
    ("result_output", "lm_head", -1),
]


def parse_tensor_name(name: str) -> LayerInfo:
    """Parse ggml tensor name into block index and sublayer category."""
    # Try blk.N.component pattern
    m = _BLK_PATTERN.match(name)
    if m:
        block_idx = int(m.group(1))
        component = m.group(2)

        # Check for rope suffix
        if "rope" in name:
            return LayerInfo(block_idx=block_idx, sublayer="rope")

        sublayer = _COMPONENT_MAP.get(component, component)
        return LayerInfo(block_idx=block_idx, sublayer=sublayer)

    # Try top-level names
    for prefix, sublayer in _TOPLEVEL_MAP.items():
        if name.startswith(prefix):
            return LayerInfo(block_idx=-1, sublayer=sublayer)

    # Try ggml operation node patterns
    for pattern, sublayer, block_idx in _OP_NODE_PATTERNS:
        if pattern in name:
            return LayerInfo(block_idx=block_idx, sublayer=sublayer)

    return LayerInfo(block_idx=-1, sublayer="unknown")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tensor_name_parser.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add collection/ tests/test_tensor_name_parser.py
git commit -m "feat: add tensor name parser (blk.N.component → block_idx, sublayer)"
```

---

### Task 3: Platform Monitor — Base + RPi

**Files:**
- Create: `collection/platform_monitor/__init__.py`
- Create: `collection/platform_monitor/base.py`
- Create: `collection/platform_monitor/rpi.py`
- Create: `tests/test_platform_monitor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_platform_monitor.py
from datetime import datetime, timezone
from unittest.mock import patch, mock_open
from collection.platform_monitor.base import PlatformMonitor
from collection.platform_monitor.rpi import RpiMonitor
from schemas.profile_schema import PlatformSnapshot


def test_base_is_abstract():
    """PlatformMonitor cannot be instantiated directly."""
    import pytest
    with pytest.raises(TypeError):
        PlatformMonitor()


def test_rpi_parse_thermal():
    """RPi thermal zone file contains millidegrees (e.g., 52300 = 52.3C)."""
    monitor = RpiMonitor.__new__(RpiMonitor)
    assert monitor._parse_thermal("52300\n") == 52.3
    assert monitor._parse_thermal("45000\n") == 45.0


def test_rpi_parse_meminfo():
    meminfo = (
        "MemTotal:        3887860 kB\n"
        "MemFree:         2796576 kB\n"
        "MemAvailable:    3491776 kB\n"
    )
    monitor = RpiMonitor.__new__(RpiMonitor)
    used, available = monitor._parse_meminfo(meminfo)
    # used = total - available = 3887860 - 3491776 = 396084 kB
    assert used == 396084 * 1024
    assert available == 3491776 * 1024


def test_rpi_parse_cpufreq():
    monitor = RpiMonitor.__new__(RpiMonitor)
    # cpufreq files contain kHz values
    assert monitor._parse_cpufreq("1800000\n") == 1800.0
    assert monitor._parse_cpufreq("600000\n") == 600.0


def test_rpi_sample_returns_snapshot(tmp_path):
    """Integration test with mocked sysfs files."""
    thermal = tmp_path / "thermal"
    thermal.write_text("50600\n")

    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:        3887860 kB\n"
        "MemFree:         2796576 kB\n"
        "MemAvailable:    3491776 kB\n"
    )

    freq_dir = tmp_path / "cpu0" / "cpufreq"
    freq_dir.mkdir(parents=True)
    (freq_dir / "scaling_cur_freq").write_text("1800000\n")

    monitor = RpiMonitor(
        thermal_path=str(thermal),
        meminfo_path=str(meminfo),
        cpufreq_pattern=str(tmp_path / "cpu{}" / "cpufreq" / "scaling_cur_freq"),
        n_cpus=1,
    )
    snap = monitor.sample()

    assert isinstance(snap, PlatformSnapshot)
    assert snap.cpu_temp_c == 50.6
    assert snap.gpu_temp_c is None
    assert snap.power_mw == 0.0  # RPi: no direct power reading
    assert snap.cpu_freq_mhz == [1800.0]
    assert snap.mem_used_bytes == 396084 * 1024
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_platform_monitor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement base monitor**

```python
# collection/platform_monitor/__init__.py
```

```python
# collection/platform_monitor/base.py
from __future__ import annotations

from abc import ABC, abstractmethod

from schemas.profile_schema import PlatformSnapshot


class PlatformMonitor(ABC):
    """Abstract base for platform-specific hardware monitoring."""

    @abstractmethod
    def sample(self) -> PlatformSnapshot:
        """Take a snapshot of current hardware state."""
        ...
```

- [ ] **Step 4: Implement RPi monitor**

```python
# collection/platform_monitor/rpi.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from collection.platform_monitor.base import PlatformMonitor
from schemas.profile_schema import PlatformSnapshot

_DEFAULT_THERMAL = "/sys/class/thermal/thermal_zone0/temp"
_DEFAULT_MEMINFO = "/proc/meminfo"
_DEFAULT_CPUFREQ = "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq"


class RpiMonitor(PlatformMonitor):
    """Platform monitor for Raspberry Pi 4 and Pi Zero 2 W."""

    def __init__(
        self,
        thermal_path: str = _DEFAULT_THERMAL,
        meminfo_path: str = _DEFAULT_MEMINFO,
        cpufreq_pattern: str = _DEFAULT_CPUFREQ,
        n_cpus: int = 4,
    ):
        self._thermal_path = thermal_path
        self._meminfo_path = meminfo_path
        self._cpufreq_pattern = cpufreq_pattern
        self._n_cpus = n_cpus

    def sample(self) -> PlatformSnapshot:
        cpu_temp = self._read_thermal()
        mem_used, mem_available = self._read_meminfo()
        cpu_freqs = self._read_cpufreqs()

        return PlatformSnapshot(
            timestamp=datetime.now(timezone.utc),
            cpu_temp_c=cpu_temp,
            gpu_temp_c=None,
            soc_temp_c=None,
            power_mw=0.0,  # RPi4 has no direct power sensor
            cpu_freq_mhz=cpu_freqs,
            gpu_freq_mhz=None,
            mem_used_bytes=mem_used,
            mem_available_bytes=mem_available,
        )

    def _read_thermal(self) -> float:
        text = Path(self._thermal_path).read_text()
        return self._parse_thermal(text)

    def _read_meminfo(self) -> tuple[int, int]:
        text = Path(self._meminfo_path).read_text()
        return self._parse_meminfo(text)

    def _read_cpufreqs(self) -> list[float]:
        freqs = []
        for i in range(self._n_cpus):
            path = Path(self._cpufreq_pattern.format(i))
            if path.exists():
                freqs.append(self._parse_cpufreq(path.read_text()))
        return freqs

    @staticmethod
    def _parse_thermal(text: str) -> float:
        """Parse millidegree value (e.g., '52300' → 52.3)."""
        return int(text.strip()) / 1000.0

    @staticmethod
    def _parse_meminfo(text: str) -> tuple[int, int]:
        """Parse /proc/meminfo, return (used_bytes, available_bytes)."""
        values: dict[str, int] = {}
        for line in text.strip().splitlines():
            parts = line.split()
            key = parts[0].rstrip(":")
            val_kb = int(parts[1])
            values[key] = val_kb
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", 0)
        used = total - available
        return used * 1024, available * 1024

    @staticmethod
    def _parse_cpufreq(text: str) -> float:
        """Parse kHz value (e.g., '1800000' → 1800.0 MHz)."""
        return int(text.strip()) / 1000.0
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_platform_monitor.py -v`
Expected: All passed

- [ ] **Step 6: Commit**

```bash
git add collection/platform_monitor/ tests/test_platform_monitor.py
git commit -m "feat: add PlatformMonitor base + RpiMonitor (thermal, meminfo, cpufreq)"
```

---

### Task 4: Platform Monitor — Jetson Orin

**Files:**
- Create: `collection/platform_monitor/jetson.py`
- Modify: `tests/test_platform_monitor.py` (append)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_platform_monitor.py (append)

from collection.platform_monitor.jetson import JetsonMonitor


def test_jetson_parse_tegrastats_line():
    """Parse a tegrastats output line."""
    line = (
        "04-05-2026 18:00:00 "
        "RAM 2200/62835MB (lfb 14520x4MB) "
        "CPU [25%@2201,30%@2201,20%@2201,22%@2201,0%@0,0%@0,0%@0,0%@0] "
        "GR3D_FREQ 50% "
        "VDD_GPU_SOC 4500mW "
        "VDD_CPU_CV 3200mW "
        "SOC 45000C "
        "GPU 42000C "
        "tj 48000C"
    )
    monitor = JetsonMonitor.__new__(JetsonMonitor)
    snap = monitor._parse_tegrastats(line)
    assert snap.mem_used_bytes == 2200 * 1024 * 1024
    assert snap.mem_available_bytes == (62835 - 2200) * 1024 * 1024
    assert snap.power_mw == 4500.0 + 3200.0  # VDD_GPU_SOC + VDD_CPU_CV
    assert len(snap.cpu_freq_mhz) == 8


def test_jetson_parse_tegrastats_gpu_temp():
    line = (
        "04-05-2026 18:00:00 "
        "RAM 2200/62835MB (lfb 14520x4MB) "
        "CPU [25%@2201,30%@2201,0%@0,0%@0,0%@0,0%@0,0%@0,0%@0] "
        "GR3D_FREQ 80% "
        "VDD_GPU_SOC 5000mW "
        "VDD_CPU_CV 3000mW "
        "SOC 50000C "
        "GPU 55000C "
        "tj 58000C"
    )
    monitor = JetsonMonitor.__new__(JetsonMonitor)
    snap = monitor._parse_tegrastats(line)
    assert snap.gpu_temp_c == 55.0
    assert snap.soc_temp_c == 50.0
    assert snap.cpu_temp_c == 58.0  # tj = junction temp
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_platform_monitor.py::test_jetson_parse_tegrastats_line -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement Jetson monitor**

```python
# collection/platform_monitor/jetson.py
from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone

from collection.platform_monitor.base import PlatformMonitor
from schemas.profile_schema import PlatformSnapshot


class JetsonMonitor(PlatformMonitor):
    """Platform monitor for NVIDIA Jetson AGX Orin (tegrastats-based)."""

    def __init__(self, tegrastats_cmd: str = "tegrastats --interval 100"):
        self._cmd = tegrastats_cmd
        self._process: subprocess.Popen | None = None

    def sample(self) -> PlatformSnapshot:
        """Run tegrastats once and parse output."""
        result = subprocess.run(
            ["tegrastats", "--interval", "100", "--count", "1"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return self._parse_tegrastats(result.stdout.strip())

    def _parse_tegrastats(self, line: str) -> PlatformSnapshot:
        """Parse a single tegrastats output line into PlatformSnapshot."""
        # RAM used/total: "RAM 2200/62835MB"
        ram_match = re.search(r"RAM (\d+)/(\d+)MB", line)
        ram_used_mb = int(ram_match.group(1)) if ram_match else 0
        ram_total_mb = int(ram_match.group(2)) if ram_match else 0

        # CPU frequencies: "CPU [25%@2201,30%@2201,...]"
        cpu_match = re.search(r"CPU \[([^\]]+)\]", line)
        cpu_freqs: list[float] = []
        if cpu_match:
            for entry in cpu_match.group(1).split(","):
                freq_match = re.search(r"@(\d+)", entry)
                cpu_freqs.append(float(freq_match.group(1)) if freq_match else 0.0)

        # GPU frequency: "GR3D_FREQ 50%"
        gr3d_match = re.search(r"GR3D_FREQ (\d+)%", line)
        gpu_util = float(gr3d_match.group(1)) if gr3d_match else 0.0

        # Power: VDD_GPU_SOC and VDD_CPU_CV in mW
        power_mw = 0.0
        for power_match in re.finditer(r"VDD_\w+ (\d+)mW", line):
            power_mw += float(power_match.group(1))

        # Temperatures: SOC, GPU, tj (junction) in millidegrees
        def _extract_temp(label: str) -> float:
            m = re.search(rf"{label} (\d+)C", line)
            return int(m.group(1)) / 1000.0 if m else 0.0

        return PlatformSnapshot(
            timestamp=datetime.now(timezone.utc),
            cpu_temp_c=_extract_temp("tj"),
            gpu_temp_c=_extract_temp("GPU"),
            soc_temp_c=_extract_temp("SOC"),
            power_mw=power_mw,
            cpu_freq_mhz=cpu_freqs,
            gpu_freq_mhz=None,  # GR3D_FREQ is utilization %, not MHz
            mem_used_bytes=ram_used_mb * 1024 * 1024,
            mem_available_bytes=(ram_total_mb - ram_used_mb) * 1024 * 1024,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_platform_monitor.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add collection/platform_monitor/jetson.py tests/test_platform_monitor.py
git commit -m "feat: add JetsonMonitor (tegrastats parsing for Orin)"
```

---

### Task 5: Local Store (JSON Transport)

**Files:**
- Create: `transport/__init__.py`
- Create: `transport/local_store.py`
- Create: `tests/test_local_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_local_store.py
import json
from datetime import datetime, timezone
from pathlib import Path

from schemas.profile_schema import ProfileRecord
from transport.local_store import LocalStore


def _make_record(**overrides) -> ProfileRecord:
    defaults = dict(
        experiment_id="exp_001",
        timestamp=datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc),
        device="rpi4",
        framework="ggml",
        model="qwen2.5-1.5b-q4_k_m",
        architecture="qwen",
        phase="decode",
        block_idx=0,
        sublayer="attention_qkv",
        op_type="matmul",
        latency_us=100.0,
        mem_bytes=4096,
        mem_peak_bytes=4096,
        power_mw=0.0,
        flops=1000,
        bandwidth_bytes_sec=500000,
        cache_l1_hit_ratio=None,
        cache_l2_hit_ratio=None,
        thermal_c=50.0,
        gpu_sm_util=None,
        gpu_tensor_util=None,
        input_tokens=128,
        output_token_idx=0,
        quantization="q4_k_m",
        batch_size=1,
    )
    defaults.update(overrides)
    return ProfileRecord(**defaults)


def test_local_store_write_single(tmp_path):
    store = LocalStore(output_dir=tmp_path)
    record = _make_record()
    store.write(record)

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    line = files[0].read_text().strip()
    data = json.loads(line)
    assert data["device"] == "rpi4"
    assert data["block_idx"] == 0


def test_local_store_write_multiple(tmp_path):
    store = LocalStore(output_dir=tmp_path)
    for i in range(5):
        store.write(_make_record(block_idx=i))

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().strip().splitlines()
    assert len(lines) == 5


def test_local_store_read_back(tmp_path):
    store = LocalStore(output_dir=tmp_path)
    store.write(_make_record(block_idx=3, sublayer="ffn_up"))
    store.write(_make_record(block_idx=4, sublayer="ffn_down"))

    records = store.read_all()
    assert len(records) == 2
    assert records[0].block_idx == 3
    assert records[1].sublayer == "ffn_down"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_local_store.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement local store**

```python
# transport/__init__.py
```

```python
# transport/local_store.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from schemas.profile_schema import ProfileRecord


class LocalStore:
    """Append-only JSONL file store for ProfileRecords."""

    def __init__(self, output_dir: Path | str, filename: str | None = None):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            filename = f"profile_{ts}.jsonl"
        self._filepath = self._output_dir / filename

    def write(self, record: ProfileRecord) -> None:
        """Append a single record as a JSON line."""
        with open(self._filepath, "a") as f:
            f.write(record.model_dump_json() + "\n")

    def read_all(self) -> list[ProfileRecord]:
        """Read all records from the JSONL file."""
        records: list[ProfileRecord] = []
        with open(self._filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(ProfileRecord.model_validate_json(line))
        return records

    @property
    def filepath(self) -> Path:
        return self._filepath
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_local_store.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add transport/ tests/test_local_store.py
git commit -m "feat: add LocalStore (JSONL append-only transport)"
```

---

### Task 6: SQLite Storage

**Files:**
- Create: `storage/__init__.py`
- Create: `storage/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_db.py
from datetime import datetime, timezone

from schemas.profile_schema import ProfileRecord
from storage.db import ProfileDB


def _make_record(**overrides) -> ProfileRecord:
    defaults = dict(
        experiment_id="exp_001",
        timestamp=datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc),
        device="rpi4",
        framework="ggml",
        model="qwen2.5-1.5b-q4_k_m",
        architecture="qwen",
        phase="decode",
        block_idx=0,
        sublayer="attention_qkv",
        op_type="matmul",
        latency_us=100.0,
        mem_bytes=4096,
        mem_peak_bytes=4096,
        power_mw=0.0,
        flops=1000,
        bandwidth_bytes_sec=500000,
        cache_l1_hit_ratio=None,
        cache_l2_hit_ratio=None,
        thermal_c=50.0,
        gpu_sm_util=None,
        gpu_tensor_util=None,
        input_tokens=128,
        output_token_idx=0,
        quantization="q4_k_m",
        batch_size=1,
    )
    defaults.update(overrides)
    return ProfileRecord(**defaults)


def test_db_insert_and_query(tmp_path):
    db = ProfileDB(tmp_path / "test.db")
    db.insert(_make_record(block_idx=0))
    db.insert(_make_record(block_idx=1))
    db.insert(_make_record(block_idx=2))

    rows = db.query(experiment_id="exp_001")
    assert len(rows) == 3
    assert rows[0].block_idx == 0
    assert rows[2].block_idx == 2


def test_db_query_by_device(tmp_path):
    db = ProfileDB(tmp_path / "test.db")
    db.insert(_make_record(device="rpi4", block_idx=0))
    db.insert(_make_record(device="orin_cpu", block_idx=0))
    db.insert(_make_record(device="rpi4", block_idx=1))

    rows = db.query(experiment_id="exp_001", device="rpi4")
    assert len(rows) == 2
    assert all(r.device == "rpi4" for r in rows)


def test_db_query_by_model(tmp_path):
    db = ProfileDB(tmp_path / "test.db")
    db.insert(_make_record(model="qwen2.5-1.5b-q4_k_m"))
    db.insert(_make_record(model="llama-3.2-3b-q4_0"))

    rows = db.query(experiment_id="exp_001", model="llama-3.2-3b-q4_0")
    assert len(rows) == 1
    assert rows[0].model == "llama-3.2-3b-q4_0"


def test_db_insert_batch(tmp_path):
    db = ProfileDB(tmp_path / "test.db")
    records = [_make_record(block_idx=i) for i in range(100)]
    db.insert_batch(records)

    rows = db.query(experiment_id="exp_001")
    assert len(rows) == 100


def test_db_count(tmp_path):
    db = ProfileDB(tmp_path / "test.db")
    db.insert_batch([_make_record(block_idx=i) for i in range(50)])
    assert db.count(experiment_id="exp_001") == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement SQLite storage**

```python
# storage/__init__.py
```

```python
# storage/db.py
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from schemas.profile_schema import ProfileRecord

_COLUMNS = [
    "experiment_id", "timestamp", "device", "framework", "model", "architecture",
    "phase", "block_idx", "sublayer", "op_type",
    "latency_us", "mem_bytes", "mem_peak_bytes", "power_mw", "flops",
    "bandwidth_bytes_sec",
    "cache_l1_hit_ratio", "cache_l2_hit_ratio", "thermal_c",
    "gpu_sm_util", "gpu_tensor_util",
    "input_tokens", "output_token_idx", "quantization", "batch_size",
]

_CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS profiles (
    {', '.join(f'{c} TEXT' if c in ('experiment_id','timestamp','device','framework','model','architecture','phase','sublayer','op_type','quantization') else f'{c} INTEGER' if c in ('mem_bytes','mem_peak_bytes','flops','bandwidth_bytes_sec','input_tokens','output_token_idx','batch_size','block_idx') else f'{c} REAL' for c in _COLUMNS)}
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_main ON profiles(experiment_id, device, model, block_idx);
CREATE INDEX IF NOT EXISTS idx_arch ON profiles(architecture, block_idx, sublayer);
"""


class ProfileDB:
    """SQLite-backed profile record store."""

    def __init__(self, db_path: Path | str):
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute(_CREATE_TABLE)
        for stmt in _CREATE_INDEX.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt)
        self._conn.commit()

    def insert(self, record: ProfileRecord) -> None:
        d = record.model_dump()
        d["timestamp"] = d["timestamp"].isoformat()
        placeholders = ", ".join("?" for _ in _COLUMNS)
        values = [d[c] for c in _COLUMNS]
        self._conn.execute(
            f"INSERT INTO profiles ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
            values,
        )
        self._conn.commit()

    def insert_batch(self, records: list[ProfileRecord]) -> None:
        placeholders = ", ".join("?" for _ in _COLUMNS)
        rows = []
        for r in records:
            d = r.model_dump()
            d["timestamp"] = d["timestamp"].isoformat()
            rows.append([d[c] for c in _COLUMNS])
        self._conn.executemany(
            f"INSERT INTO profiles ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
            rows,
        )
        self._conn.commit()

    def query(
        self,
        experiment_id: str,
        device: str | None = None,
        model: str | None = None,
    ) -> list[ProfileRecord]:
        conditions = ["experiment_id = ?"]
        params: list[str] = [experiment_id]
        if device:
            conditions.append("device = ?")
            params.append(device)
        if model:
            conditions.append("model = ?")
            params.append(model)
        where = " AND ".join(conditions)
        cursor = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM profiles WHERE {where} ORDER BY block_idx",
            params,
        )
        results = []
        for row in cursor.fetchall():
            d = dict(zip(_COLUMNS, row))
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
            results.append(ProfileRecord(**d))
        return results

    def count(self, experiment_id: str) -> int:
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM profiles WHERE experiment_id = ?",
            (experiment_id,),
        )
        return cursor.fetchone()[0]

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_db.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add storage/ tests/test_db.py
git commit -m "feat: add ProfileDB (SQLite storage with query/batch insert)"
```

---

### Task 7: Parquet Storage

**Files:**
- Create: `storage/parquet_store.py`
- Create: `tests/test_parquet_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_parquet_store.py
from datetime import datetime, timezone

import pandas as pd

from schemas.profile_schema import ProfileRecord
from storage.parquet_store import ParquetStore


def _make_record(**overrides) -> ProfileRecord:
    defaults = dict(
        experiment_id="exp_001",
        timestamp=datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc),
        device="rpi4",
        framework="ggml",
        model="qwen2.5-1.5b-q4_k_m",
        architecture="qwen",
        phase="decode",
        block_idx=0,
        sublayer="attention_qkv",
        op_type="matmul",
        latency_us=100.0,
        mem_bytes=4096,
        mem_peak_bytes=4096,
        power_mw=0.0,
        flops=1000,
        bandwidth_bytes_sec=500000,
        cache_l1_hit_ratio=None,
        cache_l2_hit_ratio=None,
        thermal_c=50.0,
        gpu_sm_util=None,
        gpu_tensor_util=None,
        input_tokens=128,
        output_token_idx=0,
        quantization="q4_k_m",
        batch_size=1,
    )
    defaults.update(overrides)
    return ProfileRecord(**defaults)


def test_parquet_write_and_read(tmp_path):
    store = ParquetStore(tmp_path / "profiles.parquet")
    records = [_make_record(block_idx=i) for i in range(10)]
    store.write(records)

    df = store.read()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert list(df["block_idx"]) == list(range(10))


def test_parquet_append(tmp_path):
    store = ParquetStore(tmp_path / "profiles.parquet")
    store.write([_make_record(block_idx=0)])
    store.append([_make_record(block_idx=1)])

    df = store.read()
    assert len(df) == 2


def test_parquet_from_db(tmp_path):
    from storage.db import ProfileDB

    db = ProfileDB(tmp_path / "test.db")
    db.insert_batch([_make_record(block_idx=i) for i in range(20)])

    store = ParquetStore(tmp_path / "export.parquet")
    rows = db.query(experiment_id="exp_001")
    store.write(rows)

    df = store.read()
    assert len(df) == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_parquet_store.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement Parquet store**

```python
# storage/parquet_store.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from schemas.profile_schema import ProfileRecord


class ParquetStore:
    """Parquet file store for bulk ProfileRecord analysis."""

    def __init__(self, filepath: Path | str):
        self._filepath = Path(filepath)

    def write(self, records: list[ProfileRecord]) -> None:
        """Write records to a new Parquet file (overwrites existing)."""
        df = self._records_to_df(records)
        df.to_parquet(self._filepath, index=False)

    def append(self, records: list[ProfileRecord]) -> None:
        """Append records to existing Parquet file."""
        new_df = self._records_to_df(records)
        if self._filepath.exists():
            existing = pd.read_parquet(self._filepath)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_parquet(self._filepath, index=False)

    def read(self) -> pd.DataFrame:
        """Read entire Parquet file as DataFrame."""
        return pd.read_parquet(self._filepath)

    @staticmethod
    def _records_to_df(records: list[ProfileRecord]) -> pd.DataFrame:
        rows = [r.model_dump() for r in records]
        return pd.DataFrame(rows)

    @property
    def filepath(self) -> Path:
        return self._filepath
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_parquet_store.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add storage/parquet_store.py tests/test_parquet_store.py
git commit -m "feat: add ParquetStore (write/append/read for bulk analysis)"
```

---

### Task 8: ggml Profiler (C Callback + Python Wrapper)

**Files:**
- Create: `collection/track1_ggml/ggml_profiler.h`
- Create: `collection/track1_ggml/ggml_profiler.c`
- Create: `collection/track1_ggml/CMakeLists.txt`
- Create: `collection/track1_ggml/profiler_wrapper.py`
- Create: `tests/test_ggml_profiler_wrapper.py`

This task creates the C callback that hooks into llama.cpp's `cb_eval` and a Python wrapper that reads the output JSON.

- [ ] **Step 1: Write the C profiler header**

```c
// collection/track1_ggml/ggml_profiler.h
#ifndef GGML_PROFILER_H
#define GGML_PROFILER_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

// Forward declarations (from ggml.h)
struct ggml_tensor;

#define PROFILER_MAX_RECORDS 65536
#define PROFILER_NAME_LEN 128

typedef struct {
    char     name[PROFILER_NAME_LEN];
    int      op;           // ggml_op enum value
    int64_t  ne[4];        // tensor dimensions
    size_t   bytes;        // ggml_nbytes(t)
    double   start_us;
    double   end_us;
    int64_t  flops;        // computed from shape
} profiler_record_t;

typedef struct {
    profiler_record_t records[PROFILER_MAX_RECORDS];
    int               n_records;
    profiler_record_t current;
    const char *      output_path;  // JSONL output file
    FILE *            output_file;
    int               phase;        // 0=prefill, 1=decode
} profiler_state_t;

// Initialize profiler state
void profiler_init(profiler_state_t * state, const char * output_path);

// The callback to register with llama_context_params.cb_eval
bool profiler_callback(struct ggml_tensor * t, bool ask, void * user_data);

// Set phase (call before each llama_decode)
void profiler_set_phase(profiler_state_t * state, int phase);

// Flush buffered records to JSONL file
void profiler_flush(profiler_state_t * state);

// Cleanup
void profiler_free(profiler_state_t * state);

#endif // GGML_PROFILER_H
```

- [ ] **Step 2: Write the C profiler implementation**

```c
// collection/track1_ggml/ggml_profiler.c
#include "ggml_profiler.h"
#include <string.h>
#include <time.h>

// We need ggml headers for ggml_nbytes, ggml_time_us, etc.
// These are provided by llama.cpp's ggml installation.
#include "ggml.h"

static int64_t compute_flops(struct ggml_tensor * t) {
    // For MUL_MAT: flops = 2 * M * N * K
    if (t->op == GGML_OP_MUL_MAT && t->src[0] && t->src[1]) {
        int64_t M = t->ne[1];  // rows of output
        int64_t N = t->ne[0];  // cols of output
        int64_t K = t->src[0]->ne[0];  // inner dimension
        return 2 * M * N * K;
    }
    return 0;
}

void profiler_init(profiler_state_t * state, const char * output_path) {
    memset(state, 0, sizeof(profiler_state_t));
    state->output_path = output_path;
    state->output_file = fopen(output_path, "w");
    state->phase = 0;
}

void profiler_set_phase(profiler_state_t * state, int phase) {
    state->phase = phase;
}

bool profiler_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    profiler_state_t * state = (profiler_state_t *)user_data;

    if (ask) {
        // Pre-execution: record metadata and start time
        strncpy(state->current.name, t->name, PROFILER_NAME_LEN - 1);
        state->current.name[PROFILER_NAME_LEN - 1] = '\0';
        state->current.op = (int)t->op;
        for (int i = 0; i < 4; i++) {
            state->current.ne[i] = t->ne[i];
        }
        state->current.bytes = ggml_nbytes(t);
        state->current.flops = compute_flops(t);
        state->current.start_us = ggml_time_us();
        return true;
    } else {
        // Post-execution: record end time, emit JSONL line
        state->current.end_us = ggml_time_us();

        if (state->output_file && state->n_records < PROFILER_MAX_RECORDS) {
            fprintf(state->output_file,
                "{\"name\":\"%s\",\"op\":%d,"
                "\"ne\":[%lld,%lld,%lld,%lld],"
                "\"bytes\":%zu,\"flops\":%lld,"
                "\"latency_us\":%.1f,\"phase\":%d}\n",
                state->current.name,
                state->current.op,
                (long long)state->current.ne[0],
                (long long)state->current.ne[1],
                (long long)state->current.ne[2],
                (long long)state->current.ne[3],
                state->current.bytes,
                (long long)state->current.flops,
                state->current.end_us - state->current.start_us,
                state->phase
            );

            state->records[state->n_records] = state->current;
            state->n_records++;
        }
        return true;
    }
}

void profiler_flush(profiler_state_t * state) {
    if (state->output_file) {
        fflush(state->output_file);
    }
}

void profiler_free(profiler_state_t * state) {
    if (state->output_file) {
        fclose(state->output_file);
        state->output_file = NULL;
    }
}
```

- [ ] **Step 3: Write CMakeLists.txt**

```cmake
# collection/track1_ggml/CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(ggml_profiler C)

# llama.cpp must be installed or available
# Set LLAMA_CPP_DIR to the llama.cpp root directory
if(NOT DEFINED LLAMA_CPP_DIR)
    message(FATAL_ERROR "Set LLAMA_CPP_DIR to your llama.cpp directory")
endif()

add_library(ggml_profiler STATIC
    ggml_profiler.c
)

target_include_directories(ggml_profiler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/include
)
```

- [ ] **Step 4: Write Python wrapper for parsing C profiler output**

```python
# collection/track1_ggml/profiler_wrapper.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from collection.track1_ggml.tensor_name_parser import parse_tensor_name
from schemas.profile_schema import ProfileRecord

# ggml_op enum values (subset)
_GGML_OP_NAMES: dict[int, str] = {
    0: "none",
    6: "add",
    8: "mul",
    18: "mul_mat",
    30: "scale",
    33: "reshape",
    37: "rope",
    42: "norm",
    43: "rms_norm",
    46: "soft_max",
    50: "get_rows",
}


def parse_profiler_output(
    jsonl_path: Path | str,
    experiment_id: str,
    device: str,
    model: str,
    architecture: str,
    quantization: str,
    input_tokens: int,
    batch_size: int = 1,
) -> list[ProfileRecord]:
    """Parse C profiler JSONL output into ProfileRecord list."""
    records: list[ProfileRecord] = []
    output_token_idx = 0

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            layer_info = parse_tensor_name(raw["name"])
            phase = "prefill" if raw["phase"] == 0 else "decode"

            if phase == "decode":
                output_token_idx += 1

            op_name = _GGML_OP_NAMES.get(raw["op"], f"op_{raw['op']}")
            latency = raw["latency_us"]
            mem_bytes = raw["bytes"]
            flops = raw["flops"]
            bandwidth = int(mem_bytes / (latency / 1_000_000)) if latency > 0 else 0

            records.append(ProfileRecord(
                experiment_id=experiment_id,
                timestamp=datetime.now(timezone.utc),
                device=device,
                framework="ggml",
                model=model,
                architecture=architecture,
                phase=phase,
                block_idx=layer_info.block_idx,
                sublayer=layer_info.sublayer,
                op_type=op_name,
                latency_us=latency,
                mem_bytes=mem_bytes,
                mem_peak_bytes=mem_bytes,
                power_mw=0.0,
                flops=flops,
                bandwidth_bytes_sec=bandwidth,
                cache_l1_hit_ratio=None,
                cache_l2_hit_ratio=None,
                thermal_c=0.0,
                gpu_sm_util=None,
                gpu_tensor_util=None,
                input_tokens=input_tokens,
                output_token_idx=output_token_idx,
                quantization=quantization,
                batch_size=batch_size,
            ))

    return records
```

- [ ] **Step 5: Write tests for Python wrapper**

```python
# tests/test_ggml_profiler_wrapper.py
import json
from pathlib import Path

from collection.track1_ggml.profiler_wrapper import parse_profiler_output


def test_parse_single_record(tmp_path):
    jsonl = tmp_path / "profile.jsonl"
    jsonl.write_text(json.dumps({
        "name": "blk.0.attn_q.weight",
        "op": 18,
        "ne": [4096, 4096, 1, 1],
        "bytes": 4096,
        "flops": 33554432,
        "latency_us": 1500.0,
        "phase": 0,
    }) + "\n")

    records = parse_profiler_output(
        jsonl_path=jsonl,
        experiment_id="test_001",
        device="rpi4",
        model="qwen2.5-1.5b-q4_k_m",
        architecture="qwen",
        quantization="q4_k_m",
        input_tokens=128,
    )

    assert len(records) == 1
    r = records[0]
    assert r.block_idx == 0
    assert r.sublayer == "attention_qkv"
    assert r.op_type == "mul_mat"
    assert r.latency_us == 1500.0
    assert r.flops == 33554432
    assert r.phase == "prefill"
    assert r.bandwidth_bytes_sec > 0


def test_parse_multiple_phases(tmp_path):
    lines = [
        json.dumps({"name": "blk.0.attn_q.weight", "op": 18, "ne": [4096, 4096, 1, 1], "bytes": 4096, "flops": 100, "latency_us": 100.0, "phase": 0}),
        json.dumps({"name": "blk.0.ffn_up.weight", "op": 18, "ne": [4096, 11008, 1, 1], "bytes": 8192, "flops": 200, "latency_us": 200.0, "phase": 0}),
        json.dumps({"name": "blk.0.attn_q.weight", "op": 18, "ne": [4096, 4096, 1, 1], "bytes": 4096, "flops": 50, "latency_us": 50.0, "phase": 1}),
    ]
    jsonl = tmp_path / "profile.jsonl"
    jsonl.write_text("\n".join(lines) + "\n")

    records = parse_profiler_output(
        jsonl_path=jsonl,
        experiment_id="test_002",
        device="rpi4",
        model="qwen2.5-1.5b-q4_k_m",
        architecture="qwen",
        quantization="q4_k_m",
        input_tokens=128,
    )

    assert len(records) == 3
    assert records[0].phase == "prefill"
    assert records[1].phase == "prefill"
    assert records[2].phase == "decode"


def test_parse_output_norm(tmp_path):
    jsonl = tmp_path / "profile.jsonl"
    jsonl.write_text(json.dumps({
        "name": "output_norm.weight",
        "op": 43,
        "ne": [4096, 1, 1, 1],
        "bytes": 16384,
        "flops": 0,
        "latency_us": 50.0,
        "phase": 1,
    }) + "\n")

    records = parse_profiler_output(
        jsonl_path=jsonl,
        experiment_id="test_003",
        device="orin_cpu",
        model="qwen2.5-3b-q4_0",
        architecture="qwen",
        quantization="q4_0",
        input_tokens=64,
    )

    assert records[0].block_idx == -1
    assert records[0].sublayer == "output_norm"
    assert records[0].op_type == "rms_norm"
```

- [ ] **Step 6: Run all tests**

Run: `python -m pytest tests/test_ggml_profiler_wrapper.py -v`
Expected: All passed

- [ ] **Step 7: Commit**

```bash
git add collection/track1_ggml/ tests/test_ggml_profiler_wrapper.py
git commit -m "feat: add ggml profiler (C callback + Python JSONL parser)"
```

---

### Task 9: Device Config + Integration Test

**Files:**
- Create: `configs/devices.yaml`
- Create: `configs/experiments.yaml`
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write device config**

```yaml
# configs/devices.yaml
devices:
  rpi4:
    host: home.rasp4.local
    connection: ssh
    compute_targets: [cpu]
    cpu: "Cortex-A72 4C @ 1.8GHz"
    ram_gb: 3.7
    n_cpus: 4

  orin_cpu:
    host: home.orin.local
    connection: ssh
    compute_targets: [cpu]
    cpu: "Cortex-A78AE 8C @ 2.2GHz"
    ram_gb: 64
    n_cpus: 8

  orin_gpu:
    host: home.orin.local
    connection: ssh
    compute_targets: [gpu]
    gpu: "Ampere 2048 CUDA cores"
    ram_gb: 64

  op13_cpu:
    host: "adb:85ea76a2"
    connection: adb
    compute_targets: [cpu]
    cpu: "Snapdragon 8 Gen 3"
    ram_gb: 12
    n_cpus: 8

  op11_cpu:
    host: "adb:db151d78"
    connection: adb
    compute_targets: [cpu]
    cpu: "Snapdragon 8 Gen 2"
    ram_gb: 16
    n_cpus: 8
```

```yaml
# configs/experiments.yaml
experiment:
  warmup_tokens: 32
  prompt_tokens: 128
  generate_tokens: 64
  repetitions: 5
  cooldown_sec: 30

profiling:
  platform_monitor_interval_ms: 100
  overhead_calibration: true
```

- [ ] **Step 2: Write integration test (full pipeline: record → store → db → parquet)**

```python
# tests/test_integration.py
"""Integration test: ProfileRecord → LocalStore → ProfileDB → ParquetStore."""
from datetime import datetime, timezone

from schemas.profile_schema import ProfileRecord
from transport.local_store import LocalStore
from storage.db import ProfileDB
from storage.parquet_store import ParquetStore


def _make_record(block_idx: int, sublayer: str, latency: float) -> ProfileRecord:
    return ProfileRecord(
        experiment_id="int_test_001",
        timestamp=datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc),
        device="rpi4",
        framework="ggml",
        model="qwen2.5-1.5b-q4_k_m",
        architecture="qwen",
        phase="decode",
        block_idx=block_idx,
        sublayer=sublayer,
        op_type="matmul",
        latency_us=latency,
        mem_bytes=4096,
        mem_peak_bytes=4096,
        power_mw=4200.0,
        flops=16777216,
        bandwidth_bytes_sec=1000000000,
        cache_l1_hit_ratio=None,
        cache_l2_hit_ratio=None,
        thermal_c=52.0,
        gpu_sm_util=None,
        gpu_tensor_util=None,
        input_tokens=128,
        output_token_idx=5,
        quantization="q4_k_m",
        batch_size=1,
    )


def test_full_pipeline(tmp_path):
    # 1. Create records simulating a 4-layer model
    records = [
        _make_record(0, "attention_qkv", 1200.0),
        _make_record(0, "ffn_up", 800.0),
        _make_record(1, "attention_qkv", 1100.0),
        _make_record(1, "ffn_up", 900.0),
        _make_record(2, "attention_qkv", 1150.0),
        _make_record(2, "ffn_up", 850.0),
        _make_record(3, "attention_qkv", 1250.0),
        _make_record(3, "ffn_up", 750.0),
    ]

    # 2. Write to local JSONL
    local = LocalStore(tmp_path / "local", filename="test.jsonl")
    for r in records:
        local.write(r)

    # 3. Read back from JSONL
    read_back = local.read_all()
    assert len(read_back) == 8

    # 4. Insert into SQLite
    db = ProfileDB(tmp_path / "profiles.db")
    db.insert_batch(read_back)
    assert db.count(experiment_id="int_test_001") == 8

    # 5. Query SQLite
    queried = db.query(experiment_id="int_test_001")
    assert len(queried) == 8
    assert queried[0].block_idx == 0

    # 6. Export to Parquet
    pq_store = ParquetStore(tmp_path / "export.parquet")
    pq_store.write(queried)

    # 7. Read Parquet and verify
    df = pq_store.read()
    assert len(df) == 8
    assert df["latency_us"].sum() > 0
    assert set(df["sublayer"].unique()) == {"attention_qkv", "ffn_up"}

    # 8. Verify round-trip data integrity
    attn_rows = df[df["sublayer"] == "attention_qkv"]
    assert len(attn_rows) == 4
    assert attn_rows["latency_us"].mean() > 1000

    db.close()
```

- [ ] **Step 3: Run integration test**

Run: `python -m pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests passed

- [ ] **Step 5: Commit**

```bash
git add configs/ tests/test_integration.py
git commit -m "feat: add device/experiment configs and full pipeline integration test"
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | Schema + Scaffold | `pyproject.toml`, `schemas/` | 3 tests |
| 2 | Tensor Name Parser | `collection/track1_ggml/tensor_name_parser.py` | 15 tests |
| 3 | Platform Monitor (RPi) | `collection/platform_monitor/base.py`, `rpi.py` | 5 tests |
| 4 | Platform Monitor (Jetson) | `collection/platform_monitor/jetson.py` | 2 tests |
| 5 | Local Store | `transport/local_store.py` | 3 tests |
| 6 | SQLite Storage | `storage/db.py` | 5 tests |
| 7 | Parquet Storage | `storage/parquet_store.py` | 3 tests |
| 8 | ggml Profiler (C + Python) | `ggml_profiler.c/.h`, `profiler_wrapper.py` | 3 tests |
| 9 | Config + Integration | `configs/`, `tests/test_integration.py` | 1 test |
| **Total** | | **~20 files** | **~40 tests** |

Phase 1 완료 후: RPi4에서 llama.cpp + cb_eval로 모델을 돌리고, per-layer JSONL → SQLite → Parquet 파이프라인이 동작하는 상태.
