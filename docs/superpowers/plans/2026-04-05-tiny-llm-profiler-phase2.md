# tiny-llm-profiler Phase 2: Analysis Engine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the analysis layer — Comparators (device/architecture/scaling/framework), BottleneckAnalyzer (roofline, hotspot), StatisticsEngine — so that collected profile data can be queried and analyzed programmatically.

**Architecture:** All analysis modules consume from ProfileDB/ParquetStore and return DataFrames. Pure Python, no device dependencies. Stateless functions operating on data.

**Tech Stack:** Python 3.11+, Pandas, NumPy, SciPy (Mann-Whitney), pytest

**Depends on:** Phase 1 (schemas, storage)

---

## File Structure

```
analysis/
├── __init__.py
├── comparators.py          # DeviceComparator, ArchitectureComparator, ScalingComparator, FrameworkComparator
├── bottleneck.py           # BottleneckAnalyzer (classify, hotspots, roofline)
├── statistics.py           # StatisticsEngine (summarize, significance, overhead)
└── device_specs.py         # Hardware theoretical peaks (FLOPS, bandwidth) per device
tests/
├── test_comparators.py
├── test_bottleneck.py
├── test_statistics.py
└── fixtures/
    └── sample_profiles.py  # Shared test data factory
```

---

### Task 1: Test Data Factory + Device Specs

**Files:**
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/sample_profiles.py`
- Create: `analysis/__init__.py`
- Create: `analysis/device_specs.py`
- Create: `tests/test_device_specs.py`

- [ ] **Step 1: Write test data factory**

```python
# tests/fixtures/__init__.py
```

```python
# tests/fixtures/sample_profiles.py
"""Shared test data factory for analysis tests."""
from datetime import datetime, timezone
from schemas.profile_schema import ProfileRecord


def make_record(
    device: str = "rpi4",
    model: str = "qwen2.5-1.5b-q4_k_m",
    architecture: str = "qwen",
    framework: str = "ggml",
    phase: str = "decode",
    block_idx: int = 0,
    sublayer: str = "attention_qkv",
    op_type: str = "matmul",
    latency_us: float = 100.0,
    mem_bytes: int = 4096,
    flops: int = 1000,
    thermal_c: float = 50.0,
    **overrides,
) -> ProfileRecord:
    defaults = dict(
        experiment_id="exp_001",
        timestamp=datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc),
        device=device,
        framework=framework,
        model=model,
        architecture=architecture,
        phase=phase,
        block_idx=block_idx,
        sublayer=sublayer,
        op_type=op_type,
        latency_us=latency_us,
        mem_bytes=mem_bytes,
        mem_peak_bytes=mem_bytes,
        power_mw=4200.0,
        flops=flops,
        bandwidth_bytes_sec=int(mem_bytes / (latency_us / 1e6)) if latency_us > 0 else 0,
        cache_l1_hit_ratio=None,
        cache_l2_hit_ratio=None,
        thermal_c=thermal_c,
        gpu_sm_util=None,
        gpu_tensor_util=None,
        input_tokens=128,
        output_token_idx=0,
        quantization="q4_k_m",
        batch_size=1,
    )
    defaults.update(overrides)
    return ProfileRecord(**defaults)


def make_layer_set(
    device: str, model: str, architecture: str, n_layers: int = 4,
    base_attn_latency: float = 1000.0, base_ffn_latency: float = 800.0,
) -> list[ProfileRecord]:
    """Generate a complete set of sublayer records for n_layers."""
    records = []
    for i in range(n_layers):
        for sublayer, lat in [
            ("attention_qkv", base_attn_latency * (1 + i * 0.05)),
            ("attention_score", base_attn_latency * 0.3),
            ("attention_out", base_attn_latency * 0.4),
            ("ffn_gate", base_ffn_latency * 0.3),
            ("ffn_up", base_ffn_latency * 0.35),
            ("ffn_down", base_ffn_latency * 0.35),
            ("attn_norm", 50.0),
            ("ffn_norm", 50.0),
        ]:
            records.append(make_record(
                device=device, model=model, architecture=architecture,
                block_idx=i, sublayer=sublayer, latency_us=lat,
                flops=33554432 if "matmul" in sublayer or "ffn" in sublayer else 0,
            ))
    return records
```

- [ ] **Step 2: Write device specs**

```python
# analysis/device_specs.py
"""Theoretical hardware peak specs for bottleneck classification."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    peak_gflops: float       # theoretical peak GFLOPS (FP16/INT8 where applicable)
    peak_bandwidth_gb_s: float  # theoretical memory bandwidth GB/s
    tdp_watts: float          # thermal design power


DEVICE_SPECS: dict[str, DeviceSpec] = {
    "rpi4": DeviceSpec(
        name="Raspberry Pi 4",
        peak_gflops=13.5,        # Cortex-A72 4C @ 1.8GHz, ~2 FLOP/cycle/core * 4 * 1.8G (NEON)
        peak_bandwidth_gb_s=4.0,  # LPDDR4 @ 3200MT/s, 32-bit = ~4 GB/s
        tdp_watts=8.0,
    ),
    "orin_cpu": DeviceSpec(
        name="Jetson AGX Orin CPU",
        peak_gflops=70.0,        # Cortex-A78AE 8C @ 2.2GHz
        peak_bandwidth_gb_s=204.8, # LPDDR5 256-bit @ 3200MHz
        tdp_watts=60.0,
    ),
    "orin_gpu": DeviceSpec(
        name="Jetson AGX Orin GPU",
        peak_gflops=5300.0,      # 275 TOPS INT8 ≈ 5.3 TFLOPS FP16 (approx)
        peak_bandwidth_gb_s=204.8,
        tdp_watts=60.0,
    ),
    "op13_cpu": DeviceSpec(
        name="OnePlus 13 CPU (SD 8 Gen 3)",
        peak_gflops=40.0,        # Cortex-X4 + A720 mixed
        peak_bandwidth_gb_s=51.2, # LPDDR5X
        tdp_watts=10.0,
    ),
    "op11_cpu": DeviceSpec(
        name="OnePlus 11 CPU (SD 8 Gen 2)",
        peak_gflops=35.0,        # Cortex-X3 + A715 mixed
        peak_bandwidth_gb_s=44.8,
        tdp_watts=10.0,
    ),
    "pi_zero2w": DeviceSpec(
        name="Raspberry Pi Zero 2 W",
        peak_gflops=6.0,         # Cortex-A53 4C @ 1GHz
        peak_bandwidth_gb_s=2.0,  # LPDDR2
        tdp_watts=3.0,
    ),
}
```

- [ ] **Step 3: Write tests for device specs**

```python
# tests/test_device_specs.py
from analysis.device_specs import DEVICE_SPECS, DeviceSpec


def test_all_devices_present():
    expected = {"rpi4", "orin_cpu", "orin_gpu", "op13_cpu", "op11_cpu", "pi_zero2w"}
    assert set(DEVICE_SPECS.keys()) == expected


def test_orin_gpu_highest_gflops():
    max_device = max(DEVICE_SPECS.values(), key=lambda d: d.peak_gflops)
    assert max_device.name == "Jetson AGX Orin GPU"


def test_spec_values_positive():
    for name, spec in DEVICE_SPECS.items():
        assert spec.peak_gflops > 0, f"{name} gflops"
        assert spec.peak_bandwidth_gb_s > 0, f"{name} bandwidth"
        assert spec.tdp_watts > 0, f"{name} tdp"
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_device_specs.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add analysis/ tests/fixtures/ tests/test_device_specs.py
git commit -m "feat: add device specs and test data factory for analysis"
```

---

### Task 2: Comparators

**Files:**
- Create: `analysis/comparators.py`
- Create: `tests/test_comparators.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_comparators.py
import pandas as pd
from tests.fixtures.sample_profiles import make_record, make_layer_set
from analysis.comparators import (
    DeviceComparator,
    ArchitectureComparator,
    ScalingComparator,
    FrameworkComparator,
)


def test_device_comparator():
    records = (
        make_layer_set("rpi4", "qwen2.5-1.5b", "qwen", n_layers=2, base_attn_latency=1000)
        + make_layer_set("orin_cpu", "qwen2.5-1.5b", "qwen", n_layers=2, base_attn_latency=200)
    )
    comp = DeviceComparator()
    df = comp.compare(records, metric="latency_us")

    assert isinstance(df, pd.DataFrame)
    assert "device" in df.columns
    assert set(df["device"].unique()) == {"rpi4", "orin_cpu"}
    # RPi4 should have higher latency
    rpi_mean = df[df["device"] == "rpi4"]["latency_us"].mean()
    orin_mean = df[df["device"] == "orin_cpu"]["latency_us"].mean()
    assert rpi_mean > orin_mean


def test_architecture_comparator():
    records = (
        make_layer_set("orin_gpu", "qwen2.5-3b", "qwen", n_layers=2)
        + make_layer_set("orin_gpu", "llama-3.2-3b", "llama", n_layers=2)
    )
    comp = ArchitectureComparator()
    df = comp.compare(records, metric="latency_us")

    assert set(df["architecture"].unique()) == {"qwen", "llama"}


def test_scaling_comparator():
    records = (
        make_layer_set("orin_gpu", "qwen2.5-0.5b", "qwen", n_layers=2, base_attn_latency=200)
        + make_layer_set("orin_gpu", "qwen2.5-1.5b", "qwen", n_layers=4, base_attn_latency=600)
        + make_layer_set("orin_gpu", "qwen2.5-3b", "qwen", n_layers=8, base_attn_latency=1200)
    )
    comp = ScalingComparator()
    df = comp.compare(records, metric="latency_us", group_by="model")

    assert len(df["model"].unique()) == 3


def test_framework_comparator():
    records_ggml = [make_record(framework="ggml", block_idx=i, latency_us=100 + i * 10) for i in range(4)]
    records_tvm = [make_record(framework="tvm", block_idx=i, latency_us=90 + i * 12) for i in range(4)]
    comp = FrameworkComparator()
    df = comp.compare(records_ggml + records_tvm, metric="latency_us")

    assert set(df["framework"].unique()) == {"ggml", "tvm"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_comparators.py -v`
Expected: FAIL

- [ ] **Step 3: Implement comparators**

```python
# analysis/comparators.py
from __future__ import annotations

import pandas as pd
from schemas.profile_schema import ProfileRecord


def _records_to_df(records: list[ProfileRecord]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in records])


class DeviceComparator:
    """Compare same model across different devices."""

    def compare(self, records: list[ProfileRecord], metric: str) -> pd.DataFrame:
        df = _records_to_df(records)
        return df.groupby(["device", "block_idx", "sublayer"])[metric].mean().reset_index()


class ArchitectureComparator:
    """Compare different architectures on same device."""

    def compare(self, records: list[ProfileRecord], metric: str) -> pd.DataFrame:
        df = _records_to_df(records)
        return df.groupby(["architecture", "block_idx", "sublayer"])[metric].mean().reset_index()


class ScalingComparator:
    """Compare different model sizes of same architecture."""

    def compare(
        self, records: list[ProfileRecord], metric: str, group_by: str = "model"
    ) -> pd.DataFrame:
        df = _records_to_df(records)
        return df.groupby([group_by, "block_idx", "sublayer"])[metric].mean().reset_index()


class FrameworkComparator:
    """Compare ggml vs TVM on same model and device."""

    def compare(self, records: list[ProfileRecord], metric: str) -> pd.DataFrame:
        df = _records_to_df(records)
        return df.groupby(["framework", "block_idx", "sublayer"])[metric].mean().reset_index()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_comparators.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add analysis/comparators.py tests/test_comparators.py
git commit -m "feat: add comparators (device, architecture, scaling, framework)"
```

---

### Task 3: Bottleneck Analyzer

**Files:**
- Create: `analysis/bottleneck.py`
- Create: `tests/test_bottleneck.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_bottleneck.py
import pandas as pd
from tests.fixtures.sample_profiles import make_record
from analysis.bottleneck import BottleneckAnalyzer
from analysis.device_specs import DEVICE_SPECS


def test_classify_compute_bound():
    analyzer = BottleneckAnalyzer()
    record = make_record(
        device="orin_gpu",
        latency_us=100.0,
        flops=300_000_000_000,  # 300 GFLOPS — high relative to Orin GPU peak
        mem_bytes=4096,
    )
    result = analyzer.classify(record)
    assert result == "compute_bound"


def test_classify_memory_bound():
    analyzer = BottleneckAnalyzer()
    record = make_record(
        device="rpi4",
        latency_us=1000.0,
        flops=1000,             # very low compute
        mem_bytes=4_000_000,    # 4MB in 1ms → 4 GB/s = 100% of RPi4 bandwidth
    )
    result = analyzer.classify(record)
    assert result == "memory_bound"


def test_find_hotspots():
    analyzer = BottleneckAnalyzer()
    records = [
        make_record(block_idx=0, sublayer="attention_qkv", latency_us=5000.0),
        make_record(block_idx=0, sublayer="ffn_up", latency_us=100.0),
        make_record(block_idx=1, sublayer="attention_qkv", latency_us=4500.0),
        make_record(block_idx=1, sublayer="ffn_up", latency_us=200.0),
    ]
    df = analyzer.find_hotspots(records, top_k=2)
    assert len(df) == 2
    assert df.iloc[0]["sublayer"] == "attention_qkv"
    assert df.iloc[0]["latency_us"] == 5000.0


def test_compute_roofline():
    analyzer = BottleneckAnalyzer()
    records = [
        make_record(device="rpi4", latency_us=1000.0, flops=1_000_000, mem_bytes=100_000),
        make_record(device="rpi4", latency_us=500.0, flops=5_000_000, mem_bytes=50_000),
    ]
    df = analyzer.compute_roofline(records)
    assert "operational_intensity" in df.columns
    assert "achieved_gflops" in df.columns
    assert len(df) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bottleneck.py -v`
Expected: FAIL

- [ ] **Step 3: Implement bottleneck analyzer**

```python
# analysis/bottleneck.py
from __future__ import annotations

import pandas as pd
from schemas.profile_schema import ProfileRecord
from analysis.device_specs import DEVICE_SPECS, DeviceSpec


class BottleneckAnalyzer:
    """Classify per-layer bottlenecks and find hotspots."""

    def classify(self, record: ProfileRecord) -> str:
        """Classify a single record as compute_bound, memory_bound, or unknown."""
        spec = DEVICE_SPECS.get(record.device)
        if spec is None:
            return "unknown"

        latency_sec = record.latency_us / 1e6
        if latency_sec <= 0:
            return "unknown"

        achieved_gflops = (record.flops / 1e9) / latency_sec
        achieved_bw_gb = (record.mem_bytes / 1e9) / latency_sec

        compute_util = achieved_gflops / spec.peak_gflops if spec.peak_gflops > 0 else 0
        bw_util = achieved_bw_gb / spec.peak_bandwidth_gb_s if spec.peak_bandwidth_gb_s > 0 else 0

        if compute_util > bw_util and compute_util > 0.1:
            return "compute_bound"
        elif bw_util > 0.1:
            return "memory_bound"
        return "unknown"

    def find_hotspots(
        self, records: list[ProfileRecord], top_k: int = 10
    ) -> pd.DataFrame:
        """Find top-K sublayers by latency."""
        df = pd.DataFrame([r.model_dump() for r in records])
        return df.nlargest(top_k, "latency_us")[
            ["block_idx", "sublayer", "op_type", "latency_us", "flops", "mem_bytes"]
        ].reset_index(drop=True)

    def compute_roofline(self, records: list[ProfileRecord]) -> pd.DataFrame:
        """Compute operational intensity and achieved performance for roofline plot."""
        rows = []
        for r in records:
            latency_sec = r.latency_us / 1e6
            if latency_sec <= 0:
                continue
            oi = r.flops / r.mem_bytes if r.mem_bytes > 0 else 0
            achieved = (r.flops / 1e9) / latency_sec
            rows.append({
                "block_idx": r.block_idx,
                "sublayer": r.sublayer,
                "operational_intensity": oi,
                "achieved_gflops": achieved,
                "device": r.device,
            })
        return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_bottleneck.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add analysis/bottleneck.py tests/test_bottleneck.py
git commit -m "feat: add BottleneckAnalyzer (classify, hotspots, roofline)"
```

---

### Task 4: Statistics Engine

**Files:**
- Create: `analysis/statistics.py`
- Create: `tests/test_statistics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_statistics.py
import pandas as pd
from tests.fixtures.sample_profiles import make_record
from analysis.statistics import StatisticsEngine


def test_summarize_basic():
    engine = StatisticsEngine()
    records = [make_record(block_idx=0, latency_us=100 + i * 10) for i in range(10)]
    df = engine.summarize(records, group_by=["block_idx"], metric="latency_us")

    assert "mean" in df.columns
    assert "std" in df.columns
    assert "median" in df.columns
    assert "p95" in df.columns
    assert "ci_95_lower" in df.columns
    assert len(df) == 1  # one group (block_idx=0)


def test_summarize_by_sublayer():
    engine = StatisticsEngine()
    records = (
        [make_record(sublayer="attention_qkv", latency_us=1000 + i) for i in range(5)]
        + [make_record(sublayer="ffn_up", latency_us=500 + i) for i in range(5)]
    )
    df = engine.summarize(records, group_by=["sublayer"], metric="latency_us")
    assert len(df) == 2


def test_significance_different():
    engine = StatisticsEngine()
    group_a = [make_record(latency_us=100 + i) for i in range(20)]
    group_b = [make_record(latency_us=500 + i) for i in range(20)]
    result = engine.test_significance(group_a, group_b, metric="latency_us")

    assert result.p_value < 0.05
    assert result.significant is True


def test_significance_same():
    engine = StatisticsEngine()
    group_a = [make_record(latency_us=100.0) for _ in range(20)]
    group_b = [make_record(latency_us=100.0) for _ in range(20)]
    result = engine.test_significance(group_a, group_b, metric="latency_us")

    assert result.significant is False


def test_compute_overhead():
    engine = StatisticsEngine()
    with_cb = [make_record(latency_us=120.0) for _ in range(10)]
    without_cb = [make_record(latency_us=100.0) for _ in range(10)]
    overhead = engine.compute_overhead(with_cb, without_cb, metric="latency_us")

    assert abs(overhead - 20.0) < 0.01  # 20% overhead
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_statistics.py -v`
Expected: FAIL

- [ ] **Step 3: Implement statistics engine**

```python
# analysis/statistics.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from schemas.profile_schema import ProfileRecord


@dataclass
class SignificanceResult:
    statistic: float
    p_value: float
    significant: bool  # p < 0.05


class StatisticsEngine:
    """Statistical analysis for profiling data."""

    def summarize(
        self,
        records: list[ProfileRecord],
        group_by: list[str],
        metric: str,
    ) -> pd.DataFrame:
        """Compute mean, std, median, p95, p99, CI(95%) grouped by specified columns."""
        df = pd.DataFrame([r.model_dump() for r in records])

        def _agg(group: pd.Series) -> pd.Series:
            values = group.values
            n = len(values)
            mean = np.mean(values)
            std = np.std(values, ddof=1) if n > 1 else 0.0
            se = std / np.sqrt(n) if n > 1 else 0.0
            ci_margin = 1.96 * se
            return pd.Series({
                "mean": mean,
                "std": std,
                "median": np.median(values),
                "p95": np.percentile(values, 95),
                "p99": np.percentile(values, 99),
                "ci_95_lower": mean - ci_margin,
                "ci_95_upper": mean + ci_margin,
                "count": n,
            })

        return df.groupby(group_by)[metric].apply(_agg).unstack().reset_index()

    def test_significance(
        self,
        group_a: list[ProfileRecord],
        group_b: list[ProfileRecord],
        metric: str,
        alpha: float = 0.05,
    ) -> SignificanceResult:
        """Mann-Whitney U test (non-parametric, no normality assumption)."""
        values_a = [getattr(r, metric) for r in group_a]
        values_b = [getattr(r, metric) for r in group_b]
        stat, p_value = sp_stats.mannwhitneyu(
            values_a, values_b, alternative="two-sided"
        )
        return SignificanceResult(
            statistic=stat, p_value=p_value, significant=p_value < alpha
        )

    def compute_overhead(
        self,
        with_callback: list[ProfileRecord],
        without_callback: list[ProfileRecord],
        metric: str,
    ) -> float:
        """Compute profiling overhead as percentage increase."""
        mean_with = np.mean([getattr(r, metric) for r in with_callback])
        mean_without = np.mean([getattr(r, metric) for r in without_callback])
        if mean_without == 0:
            return 0.0
        return ((mean_with - mean_without) / mean_without) * 100.0
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_statistics.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add analysis/statistics.py tests/test_statistics.py
git commit -m "feat: add StatisticsEngine (summarize, significance, overhead)"
```

---

### Task 5: Analysis Integration Test

**Files:**
- Create: `tests/test_analysis_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_analysis_integration.py
"""Integration: generate profiles → store → analyze → verify insights."""
from tests.fixtures.sample_profiles import make_layer_set
from storage.parquet_store import ParquetStore
from analysis.comparators import DeviceComparator, ArchitectureComparator
from analysis.bottleneck import BottleneckAnalyzer
from analysis.statistics import StatisticsEngine


def test_full_analysis_pipeline(tmp_path):
    # 1. Generate synthetic data: RPi4 slow, Orin fast
    rpi_records = make_layer_set("rpi4", "qwen2.5-1.5b", "qwen", n_layers=4, base_attn_latency=2000)
    orin_records = make_layer_set("orin_cpu", "qwen2.5-1.5b", "qwen", n_layers=4, base_attn_latency=400)
    all_records = rpi_records + orin_records

    # 2. Store to parquet
    store = ParquetStore(tmp_path / "test.parquet")
    store.write(all_records)

    # 3. Device comparison
    comp = DeviceComparator()
    df = comp.compare(all_records, "latency_us")
    rpi_mean = df[df["device"] == "rpi4"]["latency_us"].mean()
    orin_mean = df[df["device"] == "orin_cpu"]["latency_us"].mean()
    assert rpi_mean > orin_mean * 2  # RPi should be >2x slower

    # 4. Bottleneck analysis
    analyzer = BottleneckAnalyzer()
    hotspots = analyzer.find_hotspots(rpi_records, top_k=3)
    assert hotspots.iloc[0]["sublayer"] == "attention_qkv"

    # 5. Significance test
    stats = StatisticsEngine()
    result = stats.test_significance(rpi_records, orin_records, "latency_us")
    assert result.significant is True
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/test_analysis_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_analysis_integration.py
git commit -m "test: add analysis layer integration test"
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Test factory + Device specs | 3 |
| 2 | Comparators (4 types) | 4 |
| 3 | BottleneckAnalyzer | 4 |
| 4 | StatisticsEngine | 5 |
| 5 | Integration test | 1 |
| **Total** | | **17 tests** |
