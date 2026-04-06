from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from schemas.profile_schema import ProfileRecord
from storage.db import ProfileDB
from storage.parquet_store import ParquetStore


def _make_record(**overrides: Any) -> ProfileRecord:
    defaults: dict = {
        "experiment_id": "exp_default",
        "timestamp": datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        "device": "cpu",
        "framework": "pytorch",
        "model": "llama-7b",
        "architecture": "transformer",
        "phase": "prefill",
        "block_idx": 0,
        "sublayer": "attn",
        "op_type": "matmul",
        "latency_us": 100.0,
        "mem_bytes": 1024,
        "mem_peak_bytes": 2048,
        "power_mw": 5.0,
        "flops": 1_000_000,
        "bandwidth_bytes_sec": 100_000_000,
        "cache_l1_hit_ratio": None,
        "cache_l2_hit_ratio": None,
        "thermal_c": 40.0,
        "gpu_sm_util": None,
        "gpu_tensor_util": None,
        "input_tokens": 128,
        "output_token_idx": 0,
        "quantization": "fp16",
        "batch_size": 1,
    }
    defaults.update(overrides)
    return ProfileRecord(**defaults)


def test_parquet_write_and_read(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "records.parquet")
    records = [_make_record(block_idx=i) for i in range(10)]

    store.write(records)
    df = store.read()

    assert len(df) == 10
    assert sorted(df["block_idx"].tolist()) == list(range(10))


def test_parquet_append(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "records.parquet")

    store.write([_make_record(block_idx=0)])
    store.append([_make_record(block_idx=1)])
    df = store.read()

    assert len(df) == 2


def test_parquet_from_db(tmp_path: Path) -> None:
    db = ProfileDB(tmp_path / "test.db")
    records = [_make_record(experiment_id="exp_parquet", block_idx=i) for i in range(20)]
    db.insert_batch(records)

    queried = db.query("exp_parquet")
    db.close()

    store = ParquetStore(tmp_path / "from_db.parquet")
    store.write(queried)
    df = store.read()

    assert len(df) == 20
