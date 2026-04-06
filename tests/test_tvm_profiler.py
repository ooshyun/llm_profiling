"""
Unit tests for TVMProfiler backend selection.
"""
import pytest

from collection.track2_tvm.tvm_profiler import TVMProfiler


class TestTVMProfilerBackendSelection:
    def setup_method(self):
        self.profiler = TVMProfiler()

    def test_profiler_backend_selection(self):
        """Backend is selected correctly for all device classes."""
        # Nsight backend
        assert self.profiler._select_backend("orin_gpu") == "nsight"

        # OpenCL events backend (op13 variants)
        assert self.profiler._select_backend("op13_gpu") == "opencl_events"
        assert self.profiler._select_backend("op13_opencl") == "opencl_events"

        # OpenCL events backend (op11 variants)
        assert self.profiler._select_backend("op11_gpu") == "opencl_events"
        assert self.profiler._select_backend("op11_opencl") == "opencl_events"

        # CPU timer fallback — Raspberry Pi 4 and generic CPU
        assert self.profiler._select_backend("rpi4") == "cpu_timer"
        assert self.profiler._select_backend("orin_cpu") == "cpu_timer"
        assert self.profiler._select_backend("x86_linux") == "cpu_timer"
        assert self.profiler._select_backend("unknown_device") == "cpu_timer"
