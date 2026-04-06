import subprocess
from datetime import datetime
from typing import List, Optional, Tuple

from schemas.profile_schema import PlatformSnapshot
from collection.platform_monitor.base import PlatformMonitor


class AndroidMonitor(PlatformMonitor):
    """ADB-based platform monitor for Android devices."""

    def __init__(self, serial: str, n_cpus: int = 8) -> None:
        self._serial = serial
        self._n_cpus = n_cpus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self) -> PlatformSnapshot:
        thermal_raw = self._adb("cat /sys/class/thermal/thermal_zone0/temp")
        battery_raw = self._adb("dumpsys battery")
        mem_raw = self._adb("cat /proc/meminfo")

        cpu_freqs: List[float] = []
        for cpu_idx in range(self._n_cpus):
            freq_raw = self._adb(
                f"cat /sys/devices/system/cpu/cpu{cpu_idx}/cpufreq/scaling_cur_freq"
            )
            try:
                cpu_freqs.append(self._parse_cpufreq(freq_raw))
            except ValueError:
                cpu_freqs.append(0.0)

        avg_freq = sum(cpu_freqs) / len(cpu_freqs) if cpu_freqs else 0.0

        voltage_mv, current_ua = self._parse_battery(battery_raw)
        power_mw = self._estimate_power(voltage_mv, current_ua)

        mem_used, mem_available = self._parse_meminfo(mem_raw)

        return PlatformSnapshot(
            timestamp=datetime.utcnow(),
            cpu_temp_c=self._parse_thermal(thermal_raw),
            gpu_temp_c=None,
            soc_temp_c=None,
            power_mw=power_mw,
            cpu_freq_mhz=cpu_freqs if cpu_freqs else [avg_freq],
            gpu_freq_mhz=None,
            mem_used_bytes=mem_used,
            mem_available_bytes=mem_available,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adb(self, cmd: str) -> str:
        """Run an adb shell command and return stdout as a string."""
        result = subprocess.run(
            ["adb", "-s", self._serial, "shell", cmd],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout

    def _parse_thermal(self, raw: str) -> float:
        """Convert millidegrees Celsius string to degrees Celsius."""
        return int(raw.strip()) / 1000.0

    def _parse_battery(self, dumpsys: str) -> Tuple[int, int]:
        """Extract voltage (mV) and current (uA) from dumpsys battery output.

        Returns (voltage_mv, current_ua). Current may be negative (charging);
        the sign is preserved and handled in _estimate_power.
        """
        voltage_mv = 0
        current_ua = 0

        for line in dumpsys.splitlines():
            line = line.strip()
            if line.startswith("voltage:"):
                try:
                    voltage_mv = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("current now:"):
                try:
                    current_ua = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass

        return voltage_mv, current_ua

    def _parse_cpufreq(self, raw: str) -> float:
        """Convert kHz string to MHz."""
        return int(raw.strip()) / 1000.0

    def _estimate_power(self, voltage_mv: int, current_ua: int) -> float:
        """Compute power in mW from voltage (mV) and current (uA).

        Uses absolute value of current so the result is always positive
        regardless of charging/discharging direction reported by the kernel.
        """
        voltage_v = voltage_mv / 1000.0
        current_a = abs(current_ua) / 1_000_000.0
        return voltage_v * current_a * 1000.0  # convert W to mW

    def _parse_meminfo(self, raw: str) -> Tuple[int, int]:
        """Parse /proc/meminfo and return (mem_used_bytes, mem_available_bytes)."""
        mem_total = 0
        mem_available = 0

        for line in raw.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0].rstrip(":")
            try:
                value_kb = int(parts[1])
            except ValueError:
                continue
            if key == "MemTotal":
                mem_total = value_kb * 1024
            elif key == "MemAvailable":
                mem_available = value_kb * 1024

        mem_used = max(0, mem_total - mem_available)
        return mem_used, mem_available
