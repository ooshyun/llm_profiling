import pytest

from analysis.statistics import StatisticsEngine
from tests.fixtures.sample_profiles import make_record, make_layer_set


def _fast_records(n: int = 20) -> list:
    """Records with low latency (around 100 µs)."""
    return [make_record(latency_us=100.0 + i * 0.1, sublayer="attention_qkv") for i in range(n)]


def _slow_records(n: int = 20) -> list:
    """Records with high latency (around 2000 µs)."""
    return [make_record(latency_us=2000.0 + i * 0.1, sublayer="attention_qkv") for i in range(n)]


class TestStatisticsEngine:
    def test_summarize_basic(self):
        records = make_layer_set("rpi4", "llama-7b", "llama", n_layers=4)
        engine = StatisticsEngine()
        df = engine.summarize(records, group_by=["device"], metric="latency_us")
        # One row per device
        assert len(df) == 1
        row = df.iloc[0]
        assert "mean" in df.columns
        assert "std" in df.columns
        assert "median" in df.columns
        assert "p95" in df.columns
        assert "p99" in df.columns
        assert "ci_95_lower" in df.columns
        assert "ci_95_upper" in df.columns
        assert "count" in df.columns
        assert row["count"] == len(records)
        assert row["ci_95_lower"] <= row["mean"] <= row["ci_95_upper"]

    def test_summarize_by_sublayer(self):
        records = make_layer_set("rpi4", "llama-7b", "llama", n_layers=4)
        engine = StatisticsEngine()
        df = engine.summarize(records, group_by=["sublayer"], metric="latency_us")
        # 8 sublayers defined in make_layer_set
        assert len(df) == 8
        assert set(df.columns) >= {"sublayer", "mean", "count"}

    def test_significance_different_groups(self):
        """Clearly different distributions should be flagged as significant."""
        fast = _fast_records(30)
        slow = _slow_records(30)
        engine = StatisticsEngine()
        result = engine.test_significance(fast, slow, metric="latency_us", alpha=0.05)
        assert result.significant is True
        assert 0.0 <= result.p_value <= 1.0
        assert result.statistic >= 0

    def test_significance_same_group(self):
        """Identical distributions should not be flagged as significant."""
        records = _fast_records(30)
        engine = StatisticsEngine()
        result = engine.test_significance(records, records, metric="latency_us", alpha=0.05)
        # Same data → p_value will be 1.0 (or near it), not significant
        assert result.significant is False

    def test_compute_overhead(self):
        without = [make_record(latency_us=100.0) for _ in range(10)]
        with_cb = [make_record(latency_us=110.0) for _ in range(10)]
        engine = StatisticsEngine()
        overhead = engine.compute_overhead(with_cb, without, metric="latency_us")
        assert abs(overhead - 10.0) < 1e-6  # exactly 10% overhead
