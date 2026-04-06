from typing import List

import pandas as pd

from analysis.device_specs import DEVICE_SPECS
from schemas.profile_schema import ProfileRecord

# Fallback peak values when device is not in DEVICE_SPECS
_DEFAULT_PEAK_GFLOPS = 10.0
_DEFAULT_PEAK_BW_GB_S = 10.0

# Threshold fractions: a record is classified as bound if its achieved
# utilisation exceeds this fraction of the theoretical peak.
_COMPUTE_BOUND_THRESHOLD = 0.6
_MEMORY_BOUND_THRESHOLD = 0.6


def _get_peak_gflops(device: str) -> float:
    spec = DEVICE_SPECS.get(device)
    return spec.peak_gflops if spec is not None else _DEFAULT_PEAK_GFLOPS


def _get_peak_bw_gb_s(device: str) -> float:
    spec = DEVICE_SPECS.get(device)
    return spec.peak_bandwidth_gb_s if spec is not None else _DEFAULT_PEAK_BW_GB_S


class BottleneckAnalyzer:
    """Classify operations as compute-bound or memory-bound using roofline metrics."""

    def classify(self, record: ProfileRecord) -> str:
        """Return ``"compute_bound"``, ``"memory_bound"``, or ``"unknown"``.

        Classification logic
        --------------------
        1. Compute achieved GFLOP/s from flops and latency_us.
        2. Compute achieved bandwidth in GB/s from bandwidth_bytes_sec.
        3. If achieved GFLOP/s / peak GFLOP/s >= threshold → compute_bound.
        4. Elif achieved BW / peak BW >= threshold → memory_bound.
        5. Else → unknown.
        """
        if record.latency_us <= 0:
            return "unknown"

        latency_s = record.latency_us / 1_000_000.0

        # Achieved GFLOP/s
        achieved_gflops = (record.flops / 1e9) / latency_s

        # Achieved bandwidth in GB/s
        achieved_bw_gb_s = record.bandwidth_bytes_sec / 1e9

        peak_gflops = _get_peak_gflops(record.device)
        peak_bw_gb_s = _get_peak_bw_gb_s(record.device)

        compute_ratio = achieved_gflops / peak_gflops if peak_gflops > 0 else 0.0
        bw_ratio = achieved_bw_gb_s / peak_bw_gb_s if peak_bw_gb_s > 0 else 0.0

        if compute_ratio >= _COMPUTE_BOUND_THRESHOLD:
            return "compute_bound"
        if bw_ratio >= _MEMORY_BOUND_THRESHOLD:
            return "memory_bound"
        return "unknown"

    def find_hotspots(
        self, records: List[ProfileRecord], top_k: int = 10
    ) -> pd.DataFrame:
        """Return the top-K records by latency_us as a DataFrame."""
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame([r.model_dump() for r in records])
        top = df.nlargest(top_k, "latency_us").reset_index(drop=True)
        return top

    def compute_roofline(self, records: List[ProfileRecord]) -> pd.DataFrame:
        """Compute operational intensity and achieved GFLOP/s per record.

        Adds two columns to a per-record DataFrame:
        - ``operational_intensity``: FLOP / byte  (flops / bandwidth_bytes_sec * latency_s)
        - ``achieved_gflops``: GFLOP/s
        """
        if not records:
            return pd.DataFrame()

        rows = []
        for r in records:
            latency_s = r.latency_us / 1_000_000.0 if r.latency_us > 0 else 1e-9

            achieved_gflops = (r.flops / 1e9) / latency_s

            # Operational intensity = FLOP / Byte
            # bandwidth_bytes_sec is bytes/s; bytes moved in this op = bw * latency_s
            bytes_moved = r.bandwidth_bytes_sec * latency_s
            if bytes_moved > 0:
                operational_intensity = r.flops / bytes_moved
            else:
                operational_intensity = 0.0

            row = r.model_dump()
            row["operational_intensity"] = operational_intensity
            row["achieved_gflops"] = achieved_gflops
            rows.append(row)

        return pd.DataFrame(rows)
