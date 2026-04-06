"""Unit tests for RemoteCollector (local copy paths only; no SSH/ADB)."""
import os
from pathlib import Path

import pytest

from transport.collector import RemoteCollector


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_collector_local_copy(tmp_path: Path) -> None:
    """collect_local should copy source files into output_dir/device_name/."""
    # Build a small source tree
    src = tmp_path / "src_data"
    src.mkdir()
    (src / "results.json").write_text('{"latency_us": 1234}')
    (src / "subdir").mkdir()
    (src / "subdir" / "trace.bin").write_bytes(b"\x00\x01\x02")

    output_dir = tmp_path / "collected"
    collector = RemoteCollector(output_dir=output_dir)

    dest = collector.collect_local(src_dir=src, device_name="rpi4")

    # Destination directory should exist under output_dir/rpi4/
    assert dest == output_dir / "rpi4"
    assert dest.is_dir()

    # Files must be present in the destination
    assert (dest / "results.json").exists()
    assert (dest / "subdir" / "trace.bin").exists()
    assert (dest / "results.json").read_text() == '{"latency_us": 1234}'


def test_collector_creates_device_dirs(tmp_path: Path) -> None:
    """collect_local called for three devices should create three subdirectories."""
    output_dir = tmp_path / "multi_device"
    collector = RemoteCollector(output_dir=output_dir)

    device_names = ["rpi4", "orin_cpu", "op13_cpu"]

    for name in device_names:
        src = tmp_path / f"src_{name}"
        src.mkdir()
        (src / "data.txt").write_text(f"device={name}")
        collector.collect_local(src_dir=src, device_name=name)

    # All three device directories must exist
    for name in device_names:
        device_dir = output_dir / name
        assert device_dir.is_dir(), f"Missing device directory: {device_dir}"
        assert (device_dir / "data.txt").read_text() == f"device={name}"
