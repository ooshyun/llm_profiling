from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    peak_gflops: float          # Peak FP32 GFLOP/s (or equivalent)
    peak_bandwidth_gb_s: float  # Peak memory bandwidth in GB/s
    tdp_watts: float            # Thermal Design Power in Watts


DEVICE_SPECS: Dict[str, DeviceSpec] = {
    "rpi4": DeviceSpec(
        name="Raspberry Pi 4",
        peak_gflops=13.5,        # Cortex-A72 quad-core ~13.5 GFLOP/s
        peak_bandwidth_gb_s=4.0, # LPDDR4-3200 dual-channel
        tdp_watts=7.5,
    ),
    "orin_cpu": DeviceSpec(
        name="NVIDIA Jetson AGX Orin (CPU)",
        peak_gflops=115.2,       # Cortex-A78AE 12-core
        peak_bandwidth_gb_s=51.2,# Shared LPDDR5
        tdp_watts=15.0,          # CPU portion of 60W module
    ),
    "orin_gpu": DeviceSpec(
        name="NVIDIA Jetson AGX Orin (GPU)",
        peak_gflops=5632.0,      # 2048 CUDA cores @ ~1.3 GHz, FP32
        peak_bandwidth_gb_s=204.8,# LPDDR5 full bandwidth
        tdp_watts=45.0,          # GPU portion
    ),
    "op13_cpu": DeviceSpec(
        name="OnePlus 13 (Snapdragon 8 Elite CPU)",
        peak_gflops=614.4,       # Oryon cores estimated peak
        peak_bandwidth_gb_s=77.0,# LPDDR5X
        tdp_watts=8.0,
    ),
    "op11_cpu": DeviceSpec(
        name="OnePlus 11 (Snapdragon 8 Gen 2 CPU)",
        peak_gflops=297.6,       # Cortex-X3 + cluster estimated
        peak_bandwidth_gb_s=51.2,# LPDDR5X
        tdp_watts=6.5,
    ),
    "pi_zero2w": DeviceSpec(
        name="Raspberry Pi Zero 2W",
        peak_gflops=4.0,         # Cortex-A53 quad-core ~4 GFLOP/s
        peak_bandwidth_gb_s=1.6, # LPDDR2
        tdp_watts=2.5,
    ),
}
