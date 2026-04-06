"""
Unit tests for the Nsight Systems CSV parser.
"""
import csv
import tempfile
from pathlib import Path

import pytest

from collection.track2_tvm.kernel_layer_map import KernelLayerMapper
from collection.track2_tvm.nsight_parser import parse_nsight_csv


class TestNsightParser:
    def test_parse_nsight_csv(self):
        """Three kernel rows are parsed into three ProfileRecords with correct fields."""
        # Build a mapper with two known kernels; the third will be unknown.
        mapper = KernelLayerMapper()
        mapper.add_mapping("fused_matmul_0", block_idx=0, sublayer="attention_qkv")
        mapper.add_mapping("fused_softmax_0", block_idx=0, sublayer="attention_score")

        rows = [
            {
                "Start (ns)": "1000",
                "Duration (ns)": "500000",   # 500 µs
                "Name": "fused_matmul_0",
                "Device": "GPU 0",
            },
            {
                "Start (ns)": "2000",
                "Duration (ns)": "250000",   # 250 µs
                "Name": "fused_softmax_0",
                "Device": "GPU 0",
            },
            {
                "Start (ns)": "3000",
                "Duration (ns)": "100000",   # 100 µs
                "Name": "some_unknown_kernel",
                "Device": "GPU 0",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = str(Path(tmp_dir) / "trace.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["Start (ns)", "Duration (ns)", "Name", "Device"]
                )
                writer.writeheader()
                writer.writerows(rows)

            records = parse_nsight_csv(
                csv_path=csv_path,
                mapper=mapper,
                experiment_id="exp_001",
                device="orin_gpu",
                model="llama-7b",
                architecture="llama",
                quantization="q4_0",
                input_tokens=128,
            )

        assert len(records) == 3

        # First record: fused_matmul_0
        r0 = records[0]
        assert r0.sublayer == "attention_qkv"
        assert r0.block_idx == 0
        assert abs(r0.latency_us - 500.0) < 1e-6
        assert r0.framework == "tvm"
        assert r0.device == "orin_gpu"
        assert r0.experiment_id == "exp_001"

        # Second record: fused_softmax_0
        r1 = records[1]
        assert r1.sublayer == "attention_score"
        assert abs(r1.latency_us - 250.0) < 1e-6

        # Third record: unknown kernel
        r2 = records[2]
        assert r2.sublayer == "unknown"
        assert r2.block_idx == -1
        assert abs(r2.latency_us - 100.0) < 1e-6
