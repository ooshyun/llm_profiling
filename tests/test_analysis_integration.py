"""Integration test: generate data → store → compare → bottleneck → significance."""
import tempfile
from pathlib import Path

import pytest

from analysis.bottleneck import BottleneckAnalyzer
from analysis.comparators import DeviceComparator
from analysis.statistics import StatisticsEngine
from storage.db import ProfileDB
from tests.fixtures.sample_profiles import make_layer_set, make_record


def test_full_analysis_pipeline():
    """End-to-end: generate records, store in DB, compare, find hotspots, significance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "integration.db"
        db = ProfileDB(db_path)

        # 1. Generate records for two devices
        rpi4_records = make_layer_set(
            "rpi4", "llama-7b", "llama", n_layers=4, base_attn_latency=1000.0
        )
        orin_records = make_layer_set(
            "orin_cpu", "llama-7b", "llama", n_layers=4, base_attn_latency=300.0
        )
        all_records = rpi4_records + orin_records

        # 2. Store in DB
        db.insert_batch(all_records)
        assert db.count(rpi4_records[0].experiment_id) > 0

        # 3. Compare devices
        comparator = DeviceComparator()
        df_compare = comparator.compare(all_records, "latency_us")
        assert set(df_compare["device"]) == {"rpi4", "orin_cpu"}
        # orin_cpu should be faster (lower latency) than rpi4
        rpi4_mean = df_compare[df_compare["device"] == "rpi4"]["latency_us"].mean()
        orin_mean = df_compare[df_compare["device"] == "orin_cpu"]["latency_us"].mean()
        assert orin_mean < rpi4_mean

        # 4. Find hotspots across all records
        analyzer = BottleneckAnalyzer()
        hotspots = analyzer.find_hotspots(all_records, top_k=5)
        assert len(hotspots) == 5
        # Hotspots should be sorted by latency descending
        latencies = hotspots["latency_us"].tolist()
        assert latencies == sorted(latencies, reverse=True)

        # 5. Roofline
        roofline_df = analyzer.compute_roofline(all_records)
        assert "operational_intensity" in roofline_df.columns
        assert "achieved_gflops" in roofline_df.columns

        # 6. Significance test between the two device groups
        engine = StatisticsEngine()
        result = engine.test_significance(
            rpi4_records, orin_records, metric="latency_us", alpha=0.05
        )
        # rpi4 is slower → should be significantly different
        assert result.significant is True
        assert 0.0 <= result.p_value <= 1.0

        db.close()
