from datetime import datetime, timezone
from typing import List, Tuple

from schemas.profile_schema import PlatformSnapshot
from collection.platform_monitor.base import PlatformMonitor


class RpiMonitor(PlatformMonitor):
    def __init__(
        self,
        thermal_path: str = "/sys/class/thermal/thermal_zone0/temp",
        meminfo_path: str = "/proc/meminfo",
        cpufreq_pattern: str = "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq",
        n_cpus: int = 4,
    ) -> None:
        self._thermal_path = thermal_path
        self._meminfo_path = meminfo_path
        self._cpufreq_pattern = cpufreq_pattern
        self._n_cpus = n_cpus

    def sample(self) -> PlatformSnapshot:
        with open(self._thermal_path, "r") as f:
            cpu_temp_c = self._parse_thermal(f.read())

        with open(self._meminfo_path, "r") as f:
            mem_used_bytes, mem_available_bytes = self._parse_meminfo(f.read())

        cpu_freq_mhz: List[float] = []
        for i in range(self._n_cpus):
            path = self._cpufreq_pattern.format(i)
            with open(path, "r") as f:
                cpu_freq_mhz.append(self._parse_cpufreq(f.read()))

        return PlatformSnapshot(
            timestamp=datetime.now(tz=timezone.utc),
            cpu_temp_c=cpu_temp_c,
            gpu_temp_c=None,
            soc_temp_c=None,
            power_mw=0.0,
            cpu_freq_mhz=cpu_freq_mhz,
            gpu_freq_mhz=None,
            mem_used_bytes=mem_used_bytes,
            mem_available_bytes=mem_available_bytes,
        )

    def _parse_thermal(self, text: str) -> float:
        """Parse millidegree Celsius string to degrees Celsius.

        Example: "52300\\n" -> 52.3
        """
        return int(text.strip()) / 1000.0

    def _parse_meminfo(self, text: str) -> Tuple[int, int]:
        """Parse /proc/meminfo text.

        Returns (used_bytes, available_bytes).
        used = (MemTotal - MemAvailable) * 1024
        """
        mem_total_kb: int = 0
        mem_available_kb: int = 0

        for line in text.splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                mem_available_kb = int(line.split()[1])

        mem_available_bytes = mem_available_kb * 1024
        mem_used_bytes = (mem_total_kb - mem_available_kb) * 1024
        return mem_used_bytes, mem_available_bytes

    def _parse_cpufreq(self, text: str) -> float:
        """Parse kHz string to MHz.

        Example: "1800000\\n" -> 1800.0
        """
        return int(text.strip()) / 1000.0
