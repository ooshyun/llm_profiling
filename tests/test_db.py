import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from schemas.profile_schema import ProfileRecord
from storage.db import ProfileDB


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


@pytest.fixture()
def db(tmp_path: Path) -> ProfileDB:
    database = ProfileDB(tmp_path / "test.db")
    yield database
    database.close()


def test_db_insert_and_query(db: ProfileDB) -> None:
    records = [
        _make_record(experiment_id="exp1", block_idx=2),
        _make_record(experiment_id="exp1", block_idx=0),
        _make_record(experiment_id="exp1", block_idx=1),
    ]
    for r in records:
        db.insert(r)

    results = db.query("exp1")

    assert len(results) == 3
    # Verify ORDER BY block_idx ascending
    assert results[0].block_idx == 0
    assert results[1].block_idx == 1
    assert results[2].block_idx == 2


def test_db_query_by_device(db: ProfileDB) -> None:
    db.insert_batch([
        _make_record(experiment_id="exp2", device="cpu", block_idx=0),
        _make_record(experiment_id="exp2", device="cuda", block_idx=1),
        _make_record(experiment_id="exp2", device="cpu", block_idx=2),
    ])

    cpu_results = db.query("exp2", device="cpu")
    cuda_results = db.query("exp2", device="cuda")

    assert len(cpu_results) == 2
    assert all(r.device == "cpu" for r in cpu_results)
    assert len(cuda_results) == 1
    assert cuda_results[0].device == "cuda"


def test_db_query_by_model(db: ProfileDB) -> None:
    db.insert_batch([
        _make_record(experiment_id="exp3", model="llama-7b", block_idx=0),
        _make_record(experiment_id="exp3", model="mistral-7b", block_idx=1),
        _make_record(experiment_id="exp3", model="llama-7b", block_idx=2),
        _make_record(experiment_id="exp3", model="mistral-7b", block_idx=3),
    ])

    llama_results = db.query("exp3", model="llama-7b")
    mistral_results = db.query("exp3", model="mistral-7b")

    assert len(llama_results) == 2
    assert all(r.model == "llama-7b" for r in llama_results)
    assert len(mistral_results) == 2
    assert all(r.model == "mistral-7b" for r in mistral_results)


def test_db_insert_batch(db: ProfileDB) -> None:
    records = [
        _make_record(experiment_id="exp4", block_idx=i) for i in range(100)
    ]
    db.insert_batch(records)

    results = db.query("exp4")
    assert len(results) == 100


def test_db_count(db: ProfileDB) -> None:
    records = [
        _make_record(experiment_id="exp5", block_idx=i) for i in range(50)
    ]
    db.insert_batch(records)

    assert db.count("exp5") == 50
    # Unrelated experiment should return 0
    assert db.count("exp_nonexistent") == 0
