"""Full pipeline integration test: ProfileRecord -> LocalStore -> ProfileDB -> ParquetStore."""
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from schemas.profile_schema import ProfileRecord
from storage.db import ProfileDB
from storage.parquet_store import ParquetStore
from transport.local_store import LocalStore

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_FIXED_TS = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

_DEFAULTS: dict = {
    "experiment_id": "int_test_001",
    "timestamp": _FIXED_TS,
    "device": "rpi4",
    "framework": "ggml",
    "model": "llama-3b",
    "architecture": "transformer",
    "phase": "decode",
    "block_idx": 0,
    "sublayer": "attention_qkv",
    "op_type": "matmul",
    "latency_us": 1000.0,
    "mem_bytes": 4096,
    "mem_peak_bytes": 8192,
    "power_mw": 3.5,
    "flops": 50000,
    "bandwidth_bytes_sec": 1_000_000,
    "thermal_c": 55.0,
    "input_tokens": 128,
    "output_token_idx": 0,
    "quantization": "q4_0",
    "batch_size": 1,
}


def _make_record(**overrides: Any) -> ProfileRecord:
    data = {**_DEFAULTS, **overrides}
    return ProfileRecord(**data)


# ---------------------------------------------------------------------------
# Full pipeline integration test
# ---------------------------------------------------------------------------

def test_full_pipeline(tmp_path: Path) -> None:
    # 1. Create 8 records simulating a 4-layer model (2 sublayers each)
    records = [
        _make_record(block_idx=0, sublayer="attention_qkv", latency_us=1200.0),
        _make_record(block_idx=0, sublayer="ffn_up", latency_us=800.0),
        _make_record(block_idx=1, sublayer="attention_qkv", latency_us=1100.0),
        _make_record(block_idx=1, sublayer="ffn_up", latency_us=900.0),
        _make_record(block_idx=2, sublayer="attention_qkv", latency_us=1150.0),
        _make_record(block_idx=2, sublayer="ffn_up", latency_us=850.0),
        _make_record(block_idx=3, sublayer="attention_qkv", latency_us=1250.0),
        _make_record(block_idx=3, sublayer="ffn_up", latency_us=750.0),
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
