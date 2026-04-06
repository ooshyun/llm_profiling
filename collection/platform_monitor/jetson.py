import re
from datetime import datetime
from typing import List, Optional

from schemas.profile_schema import PlatformSnapshot
from collection.platform_monitor.base import PlatformMonitor


class JetsonMonitor(PlatformMonitor):
    """Platform monitor for Jetson Orin devices using tegrastats output."""

    def __init__(self) -> None:
        pass

    def sample(self) -> PlatformSnapshot:
        raise NotImplementedError(
            "Live sampling requires tegrastats subprocess; use _parse_tegrastats directly."
        )

    def _parse_tegrastats(self, line: str) -> PlatformSnapshot:
        """Parse a single tegrastats output line into a PlatformSnapshot.

        Args:
            line: A single line of tegrastats output.

        Returns:
            PlatformSnapshot populated with parsed values.
        """
        # --- Memory ---
        mem_match = re.search(r"RAM (\d+)/(\d+)MB", line)
        if mem_match:
            mem_used_mb = int(mem_match.group(1))
            mem_total_mb = int(mem_match.group(2))
        else:
            mem_used_mb = 0
            mem_total_mb = 0

        mem_used_bytes: int = mem_used_mb * 1024 * 1024
        mem_available_bytes: int = (mem_total_mb - mem_used_mb) * 1024 * 1024

        # --- CPU frequencies ---
        # Format: CPU [0%@729,3%@729,...,off,off]
        cpu_freq_mhz: List[float] = []
        cpu_match = re.search(r"CPU \[([^\]]+)\]", line)
        if cpu_match:
            entries = cpu_match.group(1).split(",")
            for entry in entries:
                freq_match = re.search(r"@(\d+)", entry)
                if freq_match:
                    cpu_freq_mhz.append(float(freq_match.group(1)))
                # "off" cores are skipped (not appended)

        # --- Power: sum all VDD_* mW values ---
        # Format: VDD_GPU_SOC 2388mW/2388mW or VDD_GPU_SOC 2388mW
        vdd_values = re.findall(r"VDD_\w+ (\d+)mW", line)
        power_mw: float = sum(float(v) for v in vdd_values)

        # --- Temperatures ---
        # Real tegrastats uses two formats:
        #   New (JetPack 6+): tj@45.625C, cpu@45.625C, soc0@42.843C
        #   Legacy: tj 45000C, GPU 42000C, SOC 45000C (millidegrees)

        def _extract_temp(label: str) -> Optional[float]:
            # Try new format first: label@value.valueC
            m = re.search(rf"{label}@([\d.]+)C", line)
            if m:
                return float(m.group(1))
            # Try legacy format: label valueC (millidegrees)
            m = re.search(rf"{label} (\d+)C", line)
            if m:
                return int(m.group(1)) / 1000.0
            return None

        cpu_temp_c: float = _extract_temp("tj") or 0.0
        gpu_temp_c: Optional[float] = _extract_temp("GPU")
        # SoC: try soc0 (new format) then SOC (legacy)
        soc_temp_c: Optional[float] = _extract_temp("soc0") or _extract_temp("SOC")

        # --- GPU utilization (GR3D_FREQ) — captured but not stored in snapshot ---
        # gr3d_match = re.search(r"GR3D_FREQ (\d+)%", line)

        return PlatformSnapshot(
            timestamp=datetime.utcnow(),
            cpu_temp_c=cpu_temp_c,
            gpu_temp_c=gpu_temp_c,
            soc_temp_c=soc_temp_c,
            power_mw=power_mw,
            cpu_freq_mhz=cpu_freq_mhz,
            gpu_freq_mhz=None,
            mem_used_bytes=mem_used_bytes,
            mem_available_bytes=mem_available_bytes,
        )
