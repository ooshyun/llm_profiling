"""Unit tests for AndroidMonitor parsing helpers (no device required)."""
from unittest.mock import MagicMock, patch

import pytest

from collection.platform_monitor.android import AndroidMonitor


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def monitor() -> AndroidMonitor:
    """Return an AndroidMonitor wired to a fake serial; no ADB calls are made."""
    return AndroidMonitor(serial="test_serial_0000", n_cpus=8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parse_thermal(monitor: AndroidMonitor) -> None:
    """Millidegrees Celsius string '30000\n' should convert to 30.0°C."""
    result = monitor._parse_thermal("30000\n")
    assert result == pytest.approx(30.0)


def test_parse_battery(monitor: AndroidMonitor) -> None:
    """Extract voltage and current from a realistic dumpsys battery excerpt."""
    dumpsys = (
        "Current Battery Service state:\n"
        "  AC powered: false\n"
        "  USB powered: true\n"
        "  Wireless powered: false\n"
        "  status: 2\n"
        "  health: 2\n"
        "  present: true\n"
        "  level: 87\n"
        "  scale: 100\n"
        "  voltage: 4200\n"
        "  current now: -500000\n"
        "  temperature: 280\n"
        "  technology: Li-ion\n"
    )
    voltage_mv, current_ua = monitor._parse_battery(dumpsys)
    assert voltage_mv == 4200
    assert current_ua == -500000


def test_parse_cpufreq(monitor: AndroidMonitor) -> None:
    """kHz string '2265600\n' should convert to 2265.6 MHz."""
    result = monitor._parse_cpufreq("2265600\n")
    assert result == pytest.approx(2265.6)


def test_estimate_power(monitor: AndroidMonitor) -> None:
    """4200 mV * 500000 uA (absolute) should yield 2100.0 mW."""
    # current_ua is negative (charging); _estimate_power uses abs()
    result = monitor._estimate_power(voltage_mv=4200, current_ua=-500000)
    assert result == pytest.approx(2100.0)
