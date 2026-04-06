from datetime import datetime, timezone

import pytest

from schemas.profile_schema import PlatformSnapshot, ProfileRecord


FIXED_TS = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _make_profile_record(**overrides) -> ProfileRecord:
    defaults = dict(
        experiment_id="exp-001",
        timestamp=FIXED_TS,
        device="nvidia-a100",
        framework="pytorch",
        model="llama-7b",
        architecture="transformer",
        phase="prefill",
        block_idx=3,
        sublayer="attention",
        op_type="matmul",
        latency_us=123.45,
        mem_bytes=1_048_576,
        mem_peak_bytes=2_097_152,
        power_mw=250.0,
        flops=1_000_000,
        bandwidth_bytes_sec=500_000_000,
        input_tokens=512,
        output_token_idx=0,
        quantization="fp16",
        batch_size=1,
    )
    defaults.update(overrides)
    return ProfileRecord(**defaults)


def test_profile_record_creation():
    record = _make_profile_record(
        cache_l1_hit_ratio=0.95,
        cache_l2_hit_ratio=0.80,
        thermal_c=72.5,
        gpu_sm_util=0.88,
        gpu_tensor_util=0.75,
    )

    assert record.experiment_id == "exp-001"
    assert record.timestamp == FIXED_TS
    assert record.device == "nvidia-a100"
    assert record.framework == "pytorch"
    assert record.model == "llama-7b"
    assert record.architecture == "transformer"
    assert record.phase == "prefill"
    assert record.block_idx == 3
    assert record.sublayer == "attention"
    assert record.op_type == "matmul"
    assert record.latency_us == pytest.approx(123.45)
    assert record.mem_bytes == 1_048_576
    assert record.mem_peak_bytes == 2_097_152
    assert record.power_mw == pytest.approx(250.0)
    assert record.flops == 1_000_000
    assert record.bandwidth_bytes_sec == 500_000_000
    assert record.cache_l1_hit_ratio == pytest.approx(0.95)
    assert record.cache_l2_hit_ratio == pytest.approx(0.80)
    assert record.thermal_c == pytest.approx(72.5)
    assert record.gpu_sm_util == pytest.approx(0.88)
    assert record.gpu_tensor_util == pytest.approx(0.75)
    assert record.input_tokens == 512
    assert record.output_token_idx == 0
    assert record.quantization == "fp16"
    assert record.batch_size == 1


def test_profile_record_to_dict():
    record = _make_profile_record()
    data = record.model_dump()

    assert isinstance(data, dict)

    expected_keys = {
        "experiment_id", "timestamp", "device", "framework", "model",
        "architecture", "phase", "block_idx", "sublayer", "op_type",
        "latency_us", "mem_bytes", "mem_peak_bytes", "power_mw", "flops",
        "bandwidth_bytes_sec", "cache_l1_hit_ratio", "cache_l2_hit_ratio",
        "thermal_c", "gpu_sm_util", "gpu_tensor_util", "input_tokens",
        "output_token_idx", "quantization", "batch_size",
    }
    assert expected_keys.issubset(data.keys())

    assert data["experiment_id"] == "exp-001"
    assert data["latency_us"] == pytest.approx(123.45)
    assert data["mem_bytes"] == 1_048_576
    assert data["cache_l1_hit_ratio"] is None
    assert data["thermal_c"] == pytest.approx(0.0)
    assert data["quantization"] == "fp16"
    assert data["batch_size"] == 1


def test_platform_snapshot_creation():
    snapshot = PlatformSnapshot(
        timestamp=FIXED_TS,
        cpu_temp_c=65.0,
        power_mw=180.0,
        cpu_freq_mhz=[3200.0, 3100.0, 3050.0, 3200.0],
        mem_used_bytes=8_589_934_592,
        mem_available_bytes=25_769_803_776,
    )

    assert snapshot.timestamp == FIXED_TS
    assert snapshot.cpu_temp_c == pytest.approx(65.0)
    assert snapshot.gpu_temp_c is None
    assert snapshot.soc_temp_c is None
    assert snapshot.power_mw == pytest.approx(180.0)
    assert snapshot.cpu_freq_mhz == [3200.0, 3100.0, 3050.0, 3200.0]
    assert snapshot.gpu_freq_mhz is None
    assert snapshot.mem_used_bytes == 8_589_934_592
    assert snapshot.mem_available_bytes == 25_769_803_776

    snapshot_with_gpu = PlatformSnapshot(
        timestamp=FIXED_TS,
        cpu_temp_c=70.0,
        gpu_temp_c=85.0,
        soc_temp_c=68.0,
        power_mw=350.0,
        cpu_freq_mhz=[3400.0],
        gpu_freq_mhz=1800.0,
        mem_used_bytes=16_000_000_000,
        mem_available_bytes=16_000_000_000,
    )

    assert snapshot_with_gpu.gpu_temp_c == pytest.approx(85.0)
    assert snapshot_with_gpu.soc_temp_c == pytest.approx(68.0)
    assert snapshot_with_gpu.gpu_freq_mhz == pytest.approx(1800.0)
