"""Device tests for NVIDIA Jetson Orin (require real SSH connectivity).

Run only when targeting an actual Orin board:
    pytest -m device tests/test_device_orin.py
"""
import subprocess

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ORIN_HOST = "home.orin.local"
SSH_TIMEOUT = 15


def _orin_reachable() -> bool:
    """Return True if the Orin SSH host is reachable."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             ORIN_HOST, "echo ok"],
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

_ORIN_AVAILABLE = _orin_reachable()
skip_if_no_orin = pytest.mark.skipif(
    not _ORIN_AVAILABLE,
    reason=f"Orin not reachable at {ORIN_HOST}",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.device
@skip_if_no_orin
def test_orin_ssh_connection() -> None:
    """Verify that the Orin host runs a 64-bit ARM kernel (aarch64)."""
    result = subprocess.run(
        ["ssh", ORIN_HOST, "uname -m"],
        capture_output=True,
        text=True,
        timeout=SSH_TIMEOUT,
        check=True,
    )
    arch = result.stdout.strip()
    assert arch == "aarch64", f"Expected aarch64, got '{arch}'"


@pytest.mark.device
@skip_if_no_orin
def test_orin_nvidia_smi() -> None:
    """Verify that nvidia-smi reports an Orin GPU."""
    result = subprocess.run(
        ["ssh", ORIN_HOST, "nvidia-smi --query-gpu=name --format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=SSH_TIMEOUT,
        check=True,
    )
    output = result.stdout.strip()
    assert "Orin" in output, (
        f"Expected 'Orin' in nvidia-smi output, got: '{output}'"
    )


@pytest.mark.device
@skip_if_no_orin
def test_orin_tegrastats_single() -> None:
    """Capture a single tegrastats sample and verify expected fields.

    tegrastats does not have a built-in "capture one sample and exit" mode on
    all L4T versions, so we wrap it with the shell ``timeout`` utility to
    collect at least one line and then terminate.
    """
    result = subprocess.run(
        ["ssh", ORIN_HOST, "timeout 3 tegrastats --interval 1000 2>&1 | head -n 1"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout.strip()
    assert "RAM" in output, f"Expected 'RAM' in tegrastats output, got: '{output}'"
    assert "CPU" in output, f"Expected 'CPU' in tegrastats output, got: '{output}'"
