"""
tests/test_ggml_profiler_wrapper.py

Unit tests for collection.track1_ggml.profiler_wrapper.parse_profiler_output.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from collection.track1_ggml.profiler_wrapper import parse_profiler_output
from schemas.profile_schema import ProfileRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMMON_KWARGS = dict(
    experiment_id="test-exp-001",
    device="test-device",
    model="llama-3-8b",
    architecture="llama",
    quantization="q4_0",
    input_tokens=128,
    batch_size=1,
)


def _write_jsonl(path: Path, lines: List[dict]) -> str:
    """Write a list of dicts as JSONL and return the file path string."""
    with path.open("w") as fh:
        for record in lines:
            fh.write(json.dumps(record) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# Test 1: single record – attention QKV weight, MUL_MAT, prefill
# ---------------------------------------------------------------------------

def test_parse_single_record(tmp_path: Path) -> None:
    """Parse one JSONL line and verify key fields are correctly mapped."""
    jsonl_file = _write_jsonl(
        tmp_path / "single.jsonl",
        [
            {
                "name": "blk.0.attn_q.weight",
                "op": 18,           # GGML_OP_MUL_MAT
                "ne": [4096, 4096, 1, 1],
                "bytes": 4096,
                "flops": 33554432,
                "latency_us": 1500.0,
                "phase": 0,         # prefill
            }
        ],
    )

    records: List[ProfileRecord] = parse_profiler_output(jsonl_file, **_COMMON_KWARGS)

    assert len(records) == 1, "Expected exactly one record"
    rec = records[0]

    # Layer location
    assert rec.block_idx == 0, f"Expected block_idx=0, got {rec.block_idx}"
    assert rec.sublayer == "attention_qkv", f"Expected sublayer='attention_qkv', got {rec.sublayer!r}"

    # Op and phase
    assert rec.op_type == "mul_mat", f"Expected op_type='mul_mat', got {rec.op_type!r}"
    assert rec.phase == "prefill", f"Expected phase='prefill', got {rec.phase!r}"

    # Numeric fields passed through
    assert rec.latency_us == pytest.approx(1500.0)
    assert rec.mem_bytes == 4096
    assert rec.flops == 33554432

    # Framework tag
    assert rec.framework == "ggml"


# ---------------------------------------------------------------------------
# Test 2: multiple lines, two phases
# ---------------------------------------------------------------------------

def test_parse_multiple_phases(tmp_path: Path) -> None:
    """Write 2 prefill + 1 decode records and verify phase labels and count."""
    lines = [
        {
            "name": "blk.0.attn_q.weight",
            "op": 18,
            "ne": [4096, 4096, 1, 1],
            "bytes": 8192,
            "flops": 33554432,
            "latency_us": 1200.0,
            "phase": 0,   # prefill
        },
        {
            "name": "blk.1.ffn_gate.weight",
            "op": 18,
            "ne": [4096, 11008, 1, 1],
            "bytes": 16384,
            "flops": 90177536,
            "latency_us": 800.0,
            "phase": 0,   # prefill
        },
        {
            "name": "blk.0.attn_q.weight",
            "op": 18,
            "ne": [4096, 4096, 1, 1],
            "bytes": 8192,
            "flops": 33554432,
            "latency_us": 950.0,
            "phase": 1,   # decode
        },
    ]
    jsonl_file = _write_jsonl(tmp_path / "multi.jsonl", lines)

    records: List[ProfileRecord] = parse_profiler_output(jsonl_file, **_COMMON_KWARGS)

    assert len(records) == 3, f"Expected 3 records, got {len(records)}"

    phases = [r.phase for r in records]
    assert phases.count("prefill") == 2, f"Expected 2 prefill records, got {phases}"
    assert phases.count("decode") == 1, f"Expected 1 decode record, got {phases}"

    # Verify ordering is preserved
    assert records[0].phase == "prefill"
    assert records[1].phase == "prefill"
    assert records[2].phase == "decode"

    # Sublayer check for ffn_gate
    assert records[1].sublayer == "ffn_gate"
    assert records[1].block_idx == 1


# ---------------------------------------------------------------------------
# Test 3: output_norm tensor (non-block global tensor, rms_norm op)
# ---------------------------------------------------------------------------

def test_parse_output_norm(tmp_path: Path) -> None:
    """output_norm.weight should map to block_idx=-1, sublayer='output_norm'."""
    jsonl_file = _write_jsonl(
        tmp_path / "output_norm.jsonl",
        [
            {
                "name": "output_norm.weight",
                "op": 43,           # GGML_OP_RMS_NORM
                "ne": [4096, 1, 1, 1],
                "bytes": 16384,
                "flops": 0,
                "latency_us": 50.0,
                "phase": 0,         # prefill
            }
        ],
    )

    records: List[ProfileRecord] = parse_profiler_output(jsonl_file, **_COMMON_KWARGS)

    assert len(records) == 1, "Expected exactly one record"
    rec = records[0]

    # Global (non-block) tensor
    assert rec.block_idx == -1, f"Expected block_idx=-1, got {rec.block_idx}"
    assert rec.sublayer == "output_norm", f"Expected sublayer='output_norm', got {rec.sublayer!r}"

    # Op
    assert rec.op_type == "rms_norm", f"Expected op_type='rms_norm', got {rec.op_type!r}"

    # Phase
    assert rec.phase == "prefill"

    # Bandwidth computed from bytes / latency
    expected_bw = int(16384 / (50.0 / 1_000_000))
    assert rec.bandwidth_bytes_sec == expected_bw, (
        f"Expected bandwidth_bytes_sec={expected_bw}, got {rec.bandwidth_bytes_sec}"
    )
