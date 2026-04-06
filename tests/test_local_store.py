import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from schemas.profile_schema import ProfileRecord
from transport.local_store import LocalStore

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "experiment_id": "exp-001",
    "timestamp": datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
    "device": "cpu",
    "framework": "pytorch",
    "model": "gpt2",
    "architecture": "transformer",
    "phase": "prefill",
    "block_idx": 0,
    "sublayer": "attn",
    "op_type": "matmul",
    "latency_us": 123.4,
    "mem_bytes": 1024,
    "mem_peak_bytes": 2048,
    "power_mw": 5.0,
    "flops": 10000,
    "bandwidth_bytes_sec": 500000,
    "thermal_c": 45.0,
    "input_tokens": 128,
    "output_token_idx": 0,
    "quantization": "fp32",
    "batch_size": 1,
}


def _make_record(**overrides: Any) -> ProfileRecord:
    data = {**_DEFAULTS, **overrides}
    return ProfileRecord(**data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_local_store_write_single(tmp_path: Path) -> None:
    store = LocalStore(tmp_path, filename="out.jsonl")
    record = _make_record()

    store.write(record)

    jsonl_files = list(tmp_path.glob("*.jsonl"))
    assert len(jsonl_files) == 1

    lines = jsonl_files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    parsed = json.loads(lines[0])
    assert parsed["experiment_id"] == "exp-001"
    assert parsed["device"] == "cpu"
    assert parsed["block_idx"] == 0


def test_local_store_write_multiple(tmp_path: Path) -> None:
    store = LocalStore(tmp_path, filename="out.jsonl")

    for i in range(5):
        store.write(_make_record(block_idx=i))

    jsonl_files = list(tmp_path.glob("*.jsonl"))
    assert len(jsonl_files) == 1

    lines = jsonl_files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5


def test_local_store_read_back(tmp_path: Path) -> None:
    store = LocalStore(tmp_path, filename="out.jsonl")

    r0 = _make_record(block_idx=0, sublayer="attn")
    r1 = _make_record(block_idx=3, sublayer="ffn")

    store.write(r0)
    store.write(r1)

    records = store.read_all()
    assert len(records) == 2

    assert records[0].block_idx == 0
    assert records[0].sublayer == "attn"

    assert records[1].block_idx == 3
    assert records[1].sublayer == "ffn"
