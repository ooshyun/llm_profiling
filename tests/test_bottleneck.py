import pytest

from analysis.bottleneck import BottleneckAnalyzer
from analysis.device_specs import DEVICE_SPECS
from tests.fixtures.sample_profiles import make_record, make_layer_set


def _compute_bound_record():
    """Manufacture a record whose achieved GFLOP/s exceeds 60% of rpi4 peak."""
    spec = DEVICE_SPECS["rpi4"]
    peak_gflops = spec.peak_gflops  # e.g. 13.5
    # Target 80% of peak → achieved = 0.80 * 13.5 = 10.8 GFLOP/s
    target_gflops = 0.80 * peak_gflops
    latency_us = 1_000_000.0  # 1 second
    flops = int(target_gflops * 1e9 * (latency_us / 1_000_000.0))
    # Keep bandwidth low so it doesn't trigger memory_bound first
    bandwidth_bytes_sec = int(0.01 * spec.peak_bandwidth_gb_s * 1e9)
    return make_record(
        device="rpi4",
        latency_us=latency_us,
        flops=flops,
        bandwidth_bytes_sec=bandwidth_bytes_sec,
    )


def _memory_bound_record():
    """Manufacture a record whose achieved BW exceeds 60% of rpi4 peak."""
    spec = DEVICE_SPECS["rpi4"]
    peak_bw_gb_s = spec.peak_bandwidth_gb_s  # e.g. 4.0
    # Target 80% of peak BW → 3.2 GB/s
    target_bw_gb_s = 0.80 * peak_bw_gb_s
    bandwidth_bytes_sec = int(target_bw_gb_s * 1e9)
    # Keep FLOP count low so compute_ratio is below threshold
    flops = int(0.01 * spec.peak_gflops * 1e9 * 1.0)
    latency_us = 1_000_000.0
    return make_record(
        device="rpi4",
        latency_us=latency_us,
        flops=flops,
        bandwidth_bytes_sec=bandwidth_bytes_sec,
    )


class TestBottleneckAnalyzer:
    def test_classify_compute_bound(self):
        record = _compute_bound_record()
        result = BottleneckAnalyzer().classify(record)
        assert result == "compute_bound"

    def test_classify_memory_bound(self):
        record = _memory_bound_record()
        result = BottleneckAnalyzer().classify(record)
        assert result == "memory_bound"

    def test_find_hotspots_returns_top_k(self):
        records = make_layer_set("rpi4", "llama-7b", "llama", n_layers=4)
        # Spike one record's latency to make it a clear hotspot
        records[0] = make_record(latency_us=999_999.0)
        df = BottleneckAnalyzer().find_hotspots(records, top_k=3)
        assert len(df) == 3
        assert df.iloc[0]["latency_us"] == 999_999.0

    def test_compute_roofline_adds_columns(self):
        records = make_layer_set("rpi4", "llama-7b", "llama", n_layers=2)
        df = BottleneckAnalyzer().compute_roofline(records)
        assert "operational_intensity" in df.columns
        assert "achieved_gflops" in df.columns
        assert len(df) == len(records)
        # All values should be non-negative
        assert (df["operational_intensity"] >= 0).all()
        assert (df["achieved_gflops"] >= 0).all()
