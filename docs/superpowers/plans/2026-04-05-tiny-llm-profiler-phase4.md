# tiny-llm-profiler Phase 4: Android Monitor + Experiment Runner

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Android platform monitor (ADB-based), experiment orchestrator, and end-to-end experiment capability across all connected devices.

**Architecture:** Android monitor uses `adb shell` to read sysfs. Experiment runner reads configs, deploys profiler, runs inference, collects results. Remote collection via SCP for SSH devices, `adb pull` for Android.

**Tech Stack:** Python 3.11+, subprocess (adb/ssh), PyYAML, pytest

**Depends on:** Phase 1 (collection, storage, transport), Phase 2 (analysis)

---

## File Structure

```
collection/
└── platform_monitor/
    └── android.py              # ADB-based thermal/power/freq monitor
transport/
└── collector.py                # Remote result collection (SCP + adb pull)
scripts/
├── run_experiment.py           # Experiment orchestrator
├── deploy.sh                   # Deploy profiler binaries to devices
└── collect_results.sh          # Collect JSONL from all devices
configs/
├── devices.yaml                # (exists from Phase 1, extended)
└── models.yaml                 # Model matrix definition
tests/
├── test_android_monitor.py
├── test_collector.py
├── test_experiment_runner.py
├── test_device_rpi4.py         # Real device test: RPi4
├── test_device_orin.py         # Real device test: Orin
└── test_device_android.py      # Real device test: Android phones
```

---

### Task 1: Android Platform Monitor

**Files:**
- Create: `collection/platform_monitor/android.py`
- Create: `tests/test_android_monitor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_android_monitor.py
from collection.platform_monitor.android import AndroidMonitor


def test_parse_thermal_zones():
    monitor = AndroidMonitor.__new__(AndroidMonitor)
    # Simulated output: type\ntemp pairs
    raw = "30000\n"  # millidegrees
    assert monitor._parse_thermal(raw) == 30.0


def test_parse_battery_power():
    monitor = AndroidMonitor.__new__(AndroidMonitor)
    dumpsys = """Current Battery Service state:
  AC powered: false
  USB powered: true
  level: 85
  voltage: 4200
  current now: -500000
  temperature: 310
"""
    voltage_mv, current_ua = monitor._parse_battery(dumpsys)
    assert voltage_mv == 4200
    assert current_ua == -500000


def test_parse_cpufreq_all():
    monitor = AndroidMonitor.__new__(AndroidMonitor)
    raw = "2265600\n"
    assert monitor._parse_cpufreq(raw) == 2265.6


def test_estimate_power():
    monitor = AndroidMonitor.__new__(AndroidMonitor)
    # P = V * I = 4.2V * 0.5A = 2100 mW
    power = monitor._estimate_power(voltage_mv=4200, current_ua=-500000)
    assert abs(power - 2100.0) < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_android_monitor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Android monitor**

```python
# collection/platform_monitor/android.py
from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone

from collection.platform_monitor.base import PlatformMonitor
from schemas.profile_schema import PlatformSnapshot


class AndroidMonitor(PlatformMonitor):
    """ADB-based platform monitor for Android phones."""

    def __init__(self, serial: str, n_cpus: int = 8):
        self._serial = serial
        self._n_cpus = n_cpus

    def sample(self) -> PlatformSnapshot:
        cpu_temp = self._read_thermal()
        voltage, current = self._read_battery()
        power = self._estimate_power(voltage, current)
        freqs = self._read_cpufreqs()
        mem_used, mem_available = self._read_meminfo()

        return PlatformSnapshot(
            timestamp=datetime.now(timezone.utc),
            cpu_temp_c=cpu_temp,
            gpu_temp_c=None,
            soc_temp_c=None,
            power_mw=power,
            cpu_freq_mhz=freqs,
            gpu_freq_mhz=None,
            mem_used_bytes=mem_used,
            mem_available_bytes=mem_available,
        )

    def _adb(self, cmd: str) -> str:
        result = subprocess.run(
            ["adb", "-s", self._serial, "shell", cmd],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()

    def _read_thermal(self) -> float:
        raw = self._adb("cat /sys/class/thermal/thermal_zone0/temp")
        return self._parse_thermal(raw)

    def _read_battery(self) -> tuple[int, int]:
        raw = self._adb("dumpsys battery")
        return self._parse_battery(raw)

    def _read_cpufreqs(self) -> list[float]:
        freqs = []
        for i in range(self._n_cpus):
            try:
                raw = self._adb(f"cat /sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq")
                freqs.append(self._parse_cpufreq(raw))
            except Exception:
                freqs.append(0.0)
        return freqs

    def _read_meminfo(self) -> tuple[int, int]:
        raw = self._adb("cat /proc/meminfo")
        values: dict[str, int] = {}
        for line in raw.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                values[parts[0].rstrip(":")] = int(parts[1])
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", 0)
        return (total - available) * 1024, available * 1024

    @staticmethod
    def _parse_thermal(raw: str) -> float:
        return int(raw.strip()) / 1000.0

    @staticmethod
    def _parse_battery(dumpsys: str) -> tuple[int, int]:
        voltage = 0
        current = 0
        for line in dumpsys.splitlines():
            line = line.strip()
            if line.startswith("voltage:"):
                voltage = int(line.split(":")[1].strip())
            elif line.startswith("current now:"):
                current = int(line.split(":")[1].strip())
        return voltage, current

    @staticmethod
    def _parse_cpufreq(raw: str) -> float:
        return int(raw.strip()) / 1000.0

    @staticmethod
    def _estimate_power(voltage_mv: int, current_ua: int) -> float:
        return abs(voltage_mv / 1000.0 * current_ua / 1e6) * 1000.0  # mW
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_android_monitor.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add collection/platform_monitor/android.py tests/test_android_monitor.py
git commit -m "feat: add AndroidMonitor (ADB-based thermal, battery, cpufreq)"
```

---

### Task 2: Remote Collector

**Files:**
- Create: `transport/collector.py`
- Create: `tests/test_collector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_collector.py
from pathlib import Path
from transport.collector import RemoteCollector


def test_collector_local_copy(tmp_path):
    """Test collecting from a 'local' source (simulating remote)."""
    src_dir = tmp_path / "remote"
    src_dir.mkdir()
    (src_dir / "profile.jsonl").write_text('{"test": 1}\n')

    dst_dir = tmp_path / "collected"
    collector = RemoteCollector(output_dir=dst_dir)
    collector.collect_local(src_dir, device_name="rpi4")

    result = dst_dir / "rpi4" / "profile.jsonl"
    assert result.exists()
    assert result.read_text() == '{"test": 1}\n'


def test_collector_creates_device_dirs(tmp_path):
    dst_dir = tmp_path / "collected"
    collector = RemoteCollector(output_dir=dst_dir)

    for device in ["rpi4", "orin", "op13"]:
        src = tmp_path / device
        src.mkdir()
        (src / "data.jsonl").write_text(f'{{"device": "{device}"}}\n')
        collector.collect_local(src, device_name=device)

    assert (dst_dir / "rpi4" / "data.jsonl").exists()
    assert (dst_dir / "orin" / "data.jsonl").exists()
    assert (dst_dir / "op13" / "data.jsonl").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_collector.py -v`
Expected: FAIL

- [ ] **Step 3: Implement remote collector**

```python
# transport/collector.py
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class RemoteCollector:
    """Collect profiling results from remote devices."""

    def __init__(self, output_dir: Path | str):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def collect_local(self, src_dir: Path, device_name: str) -> Path:
        """Copy local directory to output_dir/device_name/."""
        dst = self._output_dir / device_name
        dst.mkdir(parents=True, exist_ok=True)
        for f in Path(src_dir).iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
        return dst

    def collect_ssh(self, host: str, remote_path: str, device_name: str) -> Path:
        """SCP files from SSH host."""
        dst = self._output_dir / device_name
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["scp", "-r", f"{host}:{remote_path}/*", str(dst)],
            check=True, timeout=60,
        )
        return dst

    def collect_adb(self, serial: str, remote_path: str, device_name: str) -> Path:
        """Pull files from Android device via ADB."""
        dst = self._output_dir / device_name
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["adb", "-s", serial, "pull", remote_path, str(dst)],
            check=True, timeout=60,
        )
        return dst
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_collector.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add transport/collector.py tests/test_collector.py
git commit -m "feat: add RemoteCollector (local copy, SCP, ADB pull)"
```

---

### Task 3: Model Config

**Files:**
- Create: `configs/models.yaml`

- [ ] **Step 1: Write model matrix config**

```yaml
# configs/models.yaml
tiers:
  t0:
    size_range: "~260K"
    models:
      - name: "tinystories-260k"
        architecture: "tinyllama"
        quantization: "f32"
        huggingface_id: null  # custom trained
    target_devices: ["esp32"]

  t1:
    size_range: "135M-500M"
    models:
      - name: "smollm2-135m-instruct"
        architecture: "smollm"
        quantization: "q4_k_m"
        huggingface_id: "HuggingFaceTB/SmolLM2-135M-Instruct"
      - name: "qwen2.5-0.5b-instruct"
        architecture: "qwen"
        quantization: "q4_k_m"
        huggingface_id: "Qwen/Qwen2.5-0.5B-Instruct"
      - name: "gemma-3-270m"
        architecture: "gemma"
        quantization: "q4_k_m"
        huggingface_id: "google/gemma-3-270m"
      - name: "rwkv-4-169m"
        architecture: "rwkv"
        quantization: "q4_k_m"
        huggingface_id: "BlinkDL/rwkv-4-pile-169m"
    target_devices: ["pi_zero2w", "rpi4", "orin_cpu", "orin_gpu", "op13_cpu", "op11_cpu"]

  t2:
    size_range: "1.5B-2B"
    models:
      - name: "qwen2.5-1.5b-instruct"
        architecture: "qwen"
        quantization: "q4_k_m"
        huggingface_id: "Qwen/Qwen2.5-1.5B-Instruct"
      - name: "gemma-2-2b-it"
        architecture: "gemma"
        quantization: "q4_k_m"
        huggingface_id: "google/gemma-2-2b-it"
      - name: "rwkv-4-1.5b"
        architecture: "rwkv"
        quantization: "q4_k_m"
        huggingface_id: "BlinkDL/rwkv-4-pile-1b5"
      - name: "smollm2-1.7b-instruct"
        architecture: "smollm"
        quantization: "q4_k_m"
        huggingface_id: "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    target_devices: ["rpi4", "orin_cpu", "orin_gpu", "op13_cpu", "op11_cpu"]

  t3:
    size_range: "3B-3.8B"
    models:
      - name: "qwen2.5-3b-instruct"
        architecture: "qwen"
        quantization: "q4_k_m"
        huggingface_id: "Qwen/Qwen2.5-3B-Instruct"
      - name: "llama-3.2-3b-instruct"
        architecture: "llama"
        quantization: "q4_k_m"
        huggingface_id: "meta-llama/Llama-3.2-3B-Instruct"
      - name: "gemma-2-2b-it"
        architecture: "gemma"
        quantization: "q4_k_m"
        huggingface_id: "google/gemma-2-2b-it"
      - name: "phi-3.5-mini-instruct"
        architecture: "phi"
        quantization: "q4_k_m"
        huggingface_id: "microsoft/Phi-3.5-mini-instruct"
    target_devices: ["orin_cpu", "orin_gpu", "op13_cpu", "op11_cpu"]

  t4:
    size_range: "7B-8B"
    models:
      - name: "qwen2.5-7b-instruct"
        architecture: "qwen"
        quantization: "q4_k_m"
        huggingface_id: "Qwen/Qwen2.5-7B-Instruct"
      - name: "llama-3.1-8b-instruct"
        architecture: "llama"
        quantization: "q4_k_m"
        huggingface_id: "meta-llama/Llama-3.1-8B-Instruct"
      - name: "rwkv-4-7b"
        architecture: "rwkv"
        quantization: "q4_k_m"
        huggingface_id: "BlinkDL/rwkv-4-pile-7b"
    target_devices: ["orin_cpu", "orin_gpu", "op11_cpu"]
```

- [ ] **Step 2: Commit**

```bash
git add configs/models.yaml
git commit -m "feat: add model matrix config (T0-T4 tiers, 6 architectures)"
```

---

### Task 4: Real Device Tests

**Files:**
- Create: `tests/test_device_rpi4.py`
- Create: `tests/test_device_orin.py`
- Create: `tests/test_device_android.py`

These tests require actual device connectivity and are marked with `pytest.mark.device`.

- [ ] **Step 1: Write RPi4 device test**

```python
# tests/test_device_rpi4.py
"""Real device test: Raspberry Pi 4 (requires SSH to home.rasp4.local)."""
import subprocess
import pytest
from collection.platform_monitor.rpi import RpiMonitor

pytestmark = pytest.mark.device


def _ssh_available():
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=3", "home.rasp4.local", "echo ok"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() == "ok"
    except Exception:
        return False


@pytest.mark.skipif(not _ssh_available(), reason="RPi4 not reachable")
def test_rpi4_ssh_connection():
    r = subprocess.run(
        ["ssh", "home.rasp4.local", "uname -m"],
        capture_output=True, text=True, timeout=5,
    )
    assert "aarch64" in r.stdout


@pytest.mark.skipif(not _ssh_available(), reason="RPi4 not reachable")
def test_rpi4_platform_monitor_remote():
    """Run RpiMonitor on remote device via SSH-mounted sysfs (or remote exec)."""
    # Read thermal remotely
    r = subprocess.run(
        ["ssh", "home.rasp4.local", "cat /sys/class/thermal/thermal_zone0/temp"],
        capture_output=True, text=True, timeout=5,
    )
    temp = int(r.stdout.strip()) / 1000.0
    assert 20.0 < temp < 90.0  # reasonable range
```

- [ ] **Step 2: Write Orin device test**

```python
# tests/test_device_orin.py
"""Real device test: Jetson AGX Orin (requires SSH to home.orin.local)."""
import subprocess
import pytest

pytestmark = pytest.mark.device


def _ssh_available():
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=3", "home.orin.local", "echo ok"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() == "ok"
    except Exception:
        return False


@pytest.mark.skipif(not _ssh_available(), reason="Orin not reachable")
def test_orin_ssh_connection():
    r = subprocess.run(
        ["ssh", "home.orin.local", "uname -m"],
        capture_output=True, text=True, timeout=5,
    )
    assert "aarch64" in r.stdout


@pytest.mark.skipif(not _ssh_available(), reason="Orin not reachable")
def test_orin_nvidia_smi():
    r = subprocess.run(
        ["ssh", "home.orin.local", "nvidia-smi"],
        capture_output=True, text=True, timeout=10,
    )
    assert "Orin" in r.stdout


@pytest.mark.skipif(not _ssh_available(), reason="Orin not reachable")
def test_orin_tegrastats_single():
    r = subprocess.run(
        ["ssh", "home.orin.local", "tegrastats --interval 100 --count 1"],
        capture_output=True, text=True, timeout=10,
    )
    assert "RAM" in r.stdout
    assert "CPU" in r.stdout
```

- [ ] **Step 3: Write Android device test**

```python
# tests/test_device_android.py
"""Real device test: Android phones (requires ADB connection)."""
import subprocess
import pytest

pytestmark = pytest.mark.device


def _adb_devices():
    try:
        r = subprocess.run(["adb", "devices", "-l"], capture_output=True, text=True, timeout=5)
        lines = [l for l in r.stdout.strip().splitlines()[1:] if "device" in l and "offline" not in l]
        return lines
    except Exception:
        return []


@pytest.mark.skipif(len(_adb_devices()) == 0, reason="No Android devices connected")
def test_android_device_detected():
    devices = _adb_devices()
    assert len(devices) >= 1


@pytest.mark.skipif(len(_adb_devices()) == 0, reason="No Android devices connected")
def test_android_thermal_readable():
    r = subprocess.run(
        ["adb", "shell", "cat /sys/class/thermal/thermal_zone0/temp"],
        capture_output=True, text=True, timeout=5,
    )
    temp = int(r.stdout.strip()) / 1000.0
    assert 15.0 < temp < 100.0


@pytest.mark.skipif(len(_adb_devices()) == 0, reason="No Android devices connected")
def test_android_cpufreq_readable():
    r = subprocess.run(
        ["adb", "shell", "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"],
        capture_output=True, text=True, timeout=5,
    )
    freq_khz = int(r.stdout.strip())
    assert freq_khz > 100000  # > 100 MHz
```

- [ ] **Step 4: Configure pytest markers**

Add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "device: tests requiring real device connectivity (skip with -m 'not device')",
]
```

- [ ] **Step 5: Run local tests (no devices)**

Run: `python -m pytest tests/ -v -m "not device" --tb=short`
Expected: All local tests pass

- [ ] **Step 6: Run device tests**

Run: `python -m pytest tests/ -v -m device --tb=short`
Expected: Device tests pass (or skip if device unreachable)

- [ ] **Step 7: Commit**

```bash
git add tests/test_device_rpi4.py tests/test_device_orin.py tests/test_device_android.py pyproject.toml
git commit -m "test: add real device tests (RPi4, Orin, Android) with pytest device marker"
```

---

### Task 5: Experiment Runner (Orchestrator)

**Files:**
- Create: `scripts/run_experiment.py`
- Create: `tests/test_experiment_runner.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_experiment_runner.py
from scripts.run_experiment import ExperimentConfig, ExperimentRunner


def test_config_loads(tmp_path):
    config_yaml = tmp_path / "experiment.yaml"
    config_yaml.write_text("""
experiment:
  warmup_tokens: 32
  prompt_tokens: 128
  generate_tokens: 64
  repetitions: 3
  cooldown_sec: 10
""")
    config = ExperimentConfig.from_yaml(config_yaml)
    assert config.warmup_tokens == 32
    assert config.repetitions == 3


def test_generate_experiment_matrix():
    runner = ExperimentRunner.__new__(ExperimentRunner)
    devices = ["rpi4", "orin_cpu"]
    models = [{"name": "qwen2.5-1.5b", "architecture": "qwen", "quantization": "q4_k_m"}]
    matrix = runner._generate_matrix(devices, models, repetitions=2)
    assert len(matrix) == 4  # 2 devices * 1 model * 2 reps
    assert matrix[0]["device"] == "rpi4"
    assert matrix[0]["repetition"] == 0
```

- [ ] **Step 2: Implement experiment runner**

```python
# scripts/run_experiment.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    warmup_tokens: int
    prompt_tokens: int
    generate_tokens: int
    repetitions: int
    cooldown_sec: int

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentConfig:
        with open(path) as f:
            data = yaml.safe_load(f)["experiment"]
        return cls(**data)


class ExperimentRunner:
    """Orchestrates profiling experiments across devices."""

    def __init__(self, config: ExperimentConfig):
        self._config = config

    def _generate_matrix(
        self,
        devices: list[str],
        models: list[dict],
        repetitions: int,
    ) -> list[dict]:
        matrix = []
        for device in devices:
            for model in models:
                for rep in range(repetitions):
                    matrix.append({
                        "device": device,
                        "model": model["name"],
                        "architecture": model["architecture"],
                        "quantization": model["quantization"],
                        "repetition": rep,
                    })
        return matrix
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_experiment_runner.py -v`
Expected: All passed

- [ ] **Step 4: Commit**

```bash
git add scripts/run_experiment.py tests/test_experiment_runner.py
git commit -m "feat: add ExperimentRunner with config loading and matrix generation"
```

---

## Summary

| Task | Component | Local Tests | Device Tests |
|------|-----------|-------------|-------------|
| 1 | Android Monitor | 4 (parsing) | via test_device_android.py |
| 2 | Remote Collector | 2 | N/A |
| 3 | Model Config | N/A (YAML) | N/A |
| 4 | Real Device Tests | N/A | 8 (RPi4 + Orin + Android) |
| 5 | Experiment Runner | 2 | N/A |
| **Total** | | **8 local** | **8 device** |

### Test Commands

```bash
# Local tests only (no devices needed)
python -m pytest tests/ -v -m "not device" --tb=short

# Device tests (requires connected devices)
python -m pytest tests/ -v -m device --tb=short

# All tests
python -m pytest tests/ -v --tb=short
```
