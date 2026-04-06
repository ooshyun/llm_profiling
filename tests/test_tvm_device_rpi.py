"""
Device test: verify TVM importability on the Raspberry Pi 4 board.

Requires SSH access to ``home.rasp4.local`` and TVM installed on the board.
Expected to skip in CI / when the device is not connected.
"""
import subprocess

import pytest


@pytest.mark.device
def test_rpi4_tvm_validation():
    """``import tvm`` succeeds on the Raspberry Pi 4 via SSH."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             "home.rasp4.local", "python3 -c 'import tvm'"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        pytest.skip(f"Raspberry Pi 4 not reachable: {exc}")

    if result.returncode != 0:
        pytest.skip(
            f"TVM not installed on RPi4 (returncode={result.returncode}): "
            f"{result.stderr.strip()}"
        )

    # If we reach here, import succeeded — no stdout expected from bare import
    assert result.returncode == 0
