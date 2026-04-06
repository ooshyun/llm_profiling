"""Device tests for Android phones via ADB (require a connected device).

Run only when an Android device is attached:
    pytest -m device tests/test_device_android.py
"""
import subprocess
from typing import List

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ADB_TIMEOUT = 15


def _connected_serials() -> List[str]:
    """Return serial numbers of ADB-connected devices (not emulators)."""
    try:
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            timeout=ADB_TIMEOUT,
        )
        serials: List[str] = []
        for line in result.stdout.splitlines()[1:]:
            line = line.strip()
            if line.endswith("device"):
                serial = line.split()[0]
                if not serial.startswith("emulator"):
                    serials.append(serial)
        return serials
    except Exception:
        return []


def _adb(serial: str, cmd: str, timeout: int = ADB_TIMEOUT) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    return subprocess.run(
        ["adb", "-s", serial, "shell", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Skip condition (evaluated once at collection time)
# ---------------------------------------------------------------------------

_SERIALS = _connected_serials()
_FIRST_SERIAL = _SERIALS[0] if _SERIALS else ""

skip_if_no_android = pytest.mark.skipif(
    not _SERIALS,
    reason="No ADB-connected Android device found",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.device
@skip_if_no_android
def test_android_device_detected() -> None:
    """Verify that at least one real Android device is visible via ADB."""
    assert len(_SERIALS) >= 1, "Expected at least one connected Android device"


@pytest.mark.device
@skip_if_no_android
def test_android_thermal_readable() -> None:
    """Find the first thermal zone with a plausible reading (15–100°C).

    Not all thermal zones are active on every device; zone0 may always report 0
    on some SoCs.  We scan zones 0-30 and assert that at least one reports a
    temperature in the expected idle/load range.
    """
    found_valid = False
    for zone_idx in range(31):
        result = _adb(
            _FIRST_SERIAL,
            f"cat /sys/class/thermal/thermal_zone{zone_idx}/temp 2>/dev/null",
        )
        raw = result.stdout.strip()
        if not raw.lstrip("-").isdigit():
            continue
        temp_c = int(raw) / 1000.0
        if 15.0 <= temp_c <= 100.0:
            found_valid = True
            break

    assert found_valid, (
        "No thermal zone in range 0-30 reported a temperature between 15-100°C"
    )


@pytest.mark.device
@skip_if_no_android
def test_android_cpufreq_readable() -> None:
    """Read CPU0 current frequency and assert it is >100 MHz."""
    result = _adb(
        _FIRST_SERIAL,
        "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
    )
    assert result.returncode == 0, "Failed to read cpu0 cpufreq"

    raw = result.stdout.strip()
    assert raw.isdigit(), f"Non-numeric cpufreq output: '{raw}'"

    freq_mhz = int(raw) / 1000.0
    assert freq_mhz > 100.0, (
        f"CPU frequency {freq_mhz} MHz is unexpectedly low (expected >100 MHz)"
    )
