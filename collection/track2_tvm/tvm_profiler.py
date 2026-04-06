"""
TVMProfiler — selects the correct profiling backend for a given device.

Backend mapping
---------------
orin_gpu                 → "nsight"
op13_gpu, op13_opencl   → "opencl_events"
op11_gpu, op11_opencl   → "opencl_events"
everything else          → "cpu_timer"
"""
from __future__ import annotations

_NSIGHT_DEVICES = {"orin_gpu"}
_OPENCL_DEVICES = {"op13_gpu", "op13_opencl", "op11_gpu", "op11_opencl"}


class TVMProfiler:
    """Dispatcher that resolves the profiling backend for a target device."""

    def _select_backend(self, device: str) -> str:
        """Return the backend name for *device*.

        Args:
            device: Target device identifier (e.g. ``"orin_gpu"``).

        Returns:
            One of ``"nsight"``, ``"opencl_events"``, or ``"cpu_timer"``.
        """
        if device in _NSIGHT_DEVICES:
            return "nsight"
        if device in _OPENCL_DEVICES:
            return "opencl_events"
        return "cpu_timer"
