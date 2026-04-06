"""Unit tests for ExperimentConfig and ExperimentRunner (no device required)."""
from pathlib import Path

import pytest
import yaml

from scripts.run_experiment import ExperimentConfig, ExperimentRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict) -> None:  # type: ignore[type-arg]
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_config_loads(tmp_path: Path) -> None:
    """ExperimentConfig.from_yaml should parse all expected fields correctly."""
    config_file = tmp_path / "experiment.yaml"
    _write_yaml(
        config_file,
        {
            "experiment": {
                "warmup_tokens": 16,
                "prompt_tokens": 64,
                "generate_tokens": 32,
                "repetitions": 3,
                "cooldown_sec": 15,
            },
            "profiling": {
                "platform_monitor_interval_ms": 200,
                "overhead_calibration": False,
            },
        },
    )

    cfg = ExperimentConfig.from_yaml(config_file)

    assert cfg.warmup_tokens == 16
    assert cfg.prompt_tokens == 64
    assert cfg.generate_tokens == 32
    assert cfg.repetitions == 3
    assert cfg.cooldown_sec == 15
    assert cfg.platform_monitor_interval_ms == 200
    assert cfg.overhead_calibration is False


def test_generate_experiment_matrix(tmp_path: Path) -> None:
    """2 devices × 1 model × 2 repetitions should yield exactly 4 matrix entries."""
    config_file = tmp_path / "experiment.yaml"
    _write_yaml(
        config_file,
        {
            "experiment": {
                "warmup_tokens": 32,
                "prompt_tokens": 128,
                "generate_tokens": 64,
                "repetitions": 2,
                "cooldown_sec": 10,
            },
            "profiling": {
                "platform_monitor_interval_ms": 100,
                "overhead_calibration": True,
            },
        },
    )

    cfg = ExperimentConfig.from_yaml(config_file)
    runner = ExperimentRunner(cfg)

    devices = ["rpi4", "orin_cpu"]
    models = ["SmolLM2-135M"]
    matrix = runner._generate_matrix(devices=devices, models=models, repetitions=2)

    assert len(matrix) == 4  # 2 devices × 1 model × 2 reps

    # Every entry must have the required keys
    for entry in matrix:
        assert "run_id" in entry
        assert "device" in entry
        assert "model" in entry
        assert "repetition" in entry

    # Devices and model names should match inputs
    seen_devices = {e["device"] for e in matrix}
    seen_models = {e["model"] for e in matrix}
    seen_reps = {e["repetition"] for e in matrix}

    assert seen_devices == {"rpi4", "orin_cpu"}
    assert seen_models == {"SmolLM2-135M"}
    assert seen_reps == {1, 2}
