import pytest

from analysis.device_specs import DEVICE_SPECS

_EXPECTED_DEVICES = {"rpi4", "orin_cpu", "orin_gpu", "op13_cpu", "op11_cpu", "pi_zero2w"}


def test_all_expected_devices_present():
    """All six devices must be registered in DEVICE_SPECS."""
    assert _EXPECTED_DEVICES <= set(DEVICE_SPECS.keys())


def test_orin_gpu_has_highest_gflops():
    """orin_gpu should be the highest-compute device in the spec table."""
    all_gflops = {k: v.peak_gflops for k, v in DEVICE_SPECS.items()}
    max_device = max(all_gflops, key=all_gflops.__getitem__)
    assert max_device == "orin_gpu"


def test_all_values_positive():
    """Every numeric field in every DeviceSpec must be strictly positive."""
    for name, spec in DEVICE_SPECS.items():
        assert spec.peak_gflops > 0, f"{name}: peak_gflops must be > 0"
        assert spec.peak_bandwidth_gb_s > 0, f"{name}: peak_bandwidth_gb_s must be > 0"
        assert spec.tdp_watts > 0, f"{name}: tdp_watts must be > 0"
