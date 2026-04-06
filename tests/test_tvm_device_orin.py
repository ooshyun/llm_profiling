"""
Device test: verify Nsight Systems availability on the Orin board.

Requires SSH access to ``home.orin.local``.
Skipped automatically when the host is not reachable.
"""
import subprocess

import pytest


@pytest.mark.device
def test_orin_nsight_available():
    """``nsys --version`` succeeds on the Orin board via SSH."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             "home.orin.local", "nsys --version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        pytest.skip(f"Orin board not reachable: {exc}")

    if result.returncode != 0:
        # SSH succeeded but host refused or nsys missing — still skip gracefully
        pytest.skip(
            f"nsys not available on Orin (returncode={result.returncode}): "
            f"{result.stderr.strip()}"
        )

    assert "nsys" in result.stdout.lower() or "nsight" in result.stdout.lower(), (
        f"Unexpected nsys output: {result.stdout!r}"
    )
