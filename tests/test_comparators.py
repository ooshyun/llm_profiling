import pytest

from analysis.comparators import (
    ArchitectureComparator,
    DeviceComparator,
    FrameworkComparator,
    ScalingComparator,
)
from tests.fixtures.sample_profiles import make_layer_set, make_record


def _multi_device_records():
    """Records from two devices with the same model and architecture."""
    a = make_layer_set("rpi4", "llama-7b", "llama", n_layers=2)
    b = make_layer_set("orin_cpu", "llama-7b", "llama", n_layers=2)
    return a + b


def _multi_arch_records():
    """Records from two architectures."""
    a = make_layer_set("rpi4", "llama-7b", "llama", n_layers=2)
    b = make_layer_set("rpi4", "mistral-7b", "mistral", n_layers=2)
    return a + b


def _multi_model_records():
    """Records from two models of the same architecture."""
    a = make_layer_set("rpi4", "llama-7b", "llama", n_layers=2)
    b = make_layer_set("rpi4", "llama-13b", "llama", n_layers=2)
    return a + b


def _multi_framework_records():
    r1 = [make_record(framework="ggml", sublayer="attention_qkv", block_idx=0)]
    r2 = [make_record(framework="mnn", sublayer="attention_qkv", block_idx=0)]
    return r1 + r2


class TestDeviceComparator:
    def test_compare_returns_expected_columns(self):
        records = _multi_device_records()
        df = DeviceComparator().compare(records, "latency_us")
        assert set(df.columns) >= {"device", "block_idx", "sublayer", "latency_us"}

    def test_compare_has_two_devices(self):
        records = _multi_device_records()
        df = DeviceComparator().compare(records, "latency_us")
        assert set(df["device"]) == {"rpi4", "orin_cpu"}


class TestArchitectureComparator:
    def test_compare_returns_expected_columns(self):
        records = _multi_arch_records()
        df = ArchitectureComparator().compare(records, "latency_us")
        assert set(df.columns) >= {"architecture", "block_idx", "sublayer", "latency_us"}

    def test_compare_has_two_architectures(self):
        records = _multi_arch_records()
        df = ArchitectureComparator().compare(records, "latency_us")
        assert set(df["architecture"]) == {"llama", "mistral"}


class TestScalingComparator:
    def test_compare_returns_expected_columns(self):
        records = _multi_model_records()
        df = ScalingComparator().compare(records, "latency_us")
        assert set(df.columns) >= {"model", "block_idx", "sublayer", "latency_us"}

    def test_compare_custom_group_by(self):
        records = _multi_device_records()
        df = ScalingComparator().compare(records, "latency_us", group_by="device")
        assert "device" in df.columns
        assert set(df["device"]) == {"rpi4", "orin_cpu"}


class TestFrameworkComparator:
    def test_compare_returns_expected_columns(self):
        records = _multi_framework_records()
        df = FrameworkComparator().compare(records, "latency_us")
        assert set(df.columns) >= {"framework", "block_idx", "sublayer", "latency_us"}

    def test_compare_has_two_frameworks(self):
        records = _multi_framework_records()
        df = FrameworkComparator().compare(records, "latency_us")
        assert set(df["framework"]) == {"ggml", "mnn"}
