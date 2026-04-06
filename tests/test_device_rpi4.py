"""Device tests for Raspberry Pi 4 (require real SSH connectivity).

Run only when targeting an actual RPi4:
    pytest -m device tests/test_device_rpi4.py
"""
import subprocess

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RPI4_HOST = "home.rasp4.local"
SSH_TIMEOUT = 10


def _rpi4_reachable() -> bool:
    """Return True if the RPi4 SSH host is reachable."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             RPI4_HOST, "echo ok"],
            capture_output=True,
            text=True,
            timeout=SSH_TIMEOUT,
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Skip condition (evaluated once at collection time)
# ---------------------------------------------------------------------------

_RPI4_AVAILABLE = _rpi4_reachable()
skip_if_no_rpi4 = pytest.mark.skipif(
    not _RPI4_AVAILABLE,
    reason=f"RPi4 not reachable at {RPI4_HOST}",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.device
@skip_if_no_rpi4
def test_rpi4_ssh_connection() -> None:
    """Verify that the RPi4 host runs a 64-bit ARM kernel (aarch64)."""
    result = subprocess.run(
        ["ssh", RPI4_HOST, "uname -m"],
        capture_output=True,
        text=True,
        timeout=SSH_TIMEOUT,
        check=True,
    )
    arch = result.stdout.strip()
    assert arch == "aarch64", f"Expected aarch64, got '{arch}'"


@pytest.mark.device
@skip_if_no_rpi4
def test_rpi4_thermal_readable() -> None:
    """Read the SoC thermal zone and assert a plausible temperature range."""
    result = subprocess.run(
        ["ssh", RPI4_HOST, "cat /sys/class/thermal/thermal_zone0/temp"],
        capture_output=True,
        text=True,
        timeout=SSH_TIMEOUT,
        check=True,
    )
    raw = result.stdout.strip()
    assert raw.lstrip("-").isdigit(), f"Non-numeric thermal output: '{raw}'"

    temp_c = int(raw) / 1000.0
    assert 20.0 <= temp_c <= 90.0, (
        f"Temperature {temp_c}°C is outside the expected 20–90°C range"
    )
