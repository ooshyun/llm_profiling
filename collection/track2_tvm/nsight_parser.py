"""
Parse Nsight Systems CSV exports into ProfileRecord lists.

Expected CSV columns (subset):
    "Start (ns)", "Duration (ns)", "Name", "Device"

Rows where Duration is missing or non-positive are silently skipped.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from schemas.profile_schema import ProfileRecord
from collection.track2_tvm.kernel_layer_map import KernelLayerMapper


def parse_nsight_csv(
    csv_path: str,
    mapper: KernelLayerMapper,
    experiment_id: str,
    device: str,
    model: str,
    architecture: str,
    quantization: str = "q4_0",
    input_tokens: int = 128,
) -> List[ProfileRecord]:
    """Parse an Nsight Systems CSV export and return a list of ProfileRecord.

    Args:
        csv_path:      Path to the ``.csv`` file exported by ``nsys stats``.
        mapper:        Pre-configured :class:`KernelLayerMapper` instance.
        experiment_id: Unique identifier for the profiling run.
        device:        Target device string (e.g. ``"orin_gpu"``).
        model:         Model name (e.g. ``"llama-7b"``).
        architecture:  Architecture name (e.g. ``"llama"``).
        quantization:  Weight quantisation format (default ``"q4_0"``).
        input_tokens:  Number of prompt tokens used in this run (default 128).

    Returns:
        A list of :class:`ProfileRecord` — one per valid kernel row.
    """
    records: List[ProfileRecord] = []
    ts_now = datetime.now(tz=timezone.utc)

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # --- Duration ---------------------------------------------------
            raw_duration = row.get("Duration (ns)", "").strip()
            if not raw_duration:
                continue
            try:
                duration_ns = float(raw_duration)
            except ValueError:
                continue
            if duration_ns <= 0:
                continue

            latency_us = duration_ns / 1000.0

            # --- Kernel name ------------------------------------------------
            kernel_name = row.get("Name", "").strip()

            # --- Layer mapping ----------------------------------------------
            layer_info = mapper.lookup(kernel_name)

            records.append(
                ProfileRecord(
                    # Identification
                    experiment_id=experiment_id,
                    timestamp=ts_now,
                    device=device,
                    framework="tvm",
                    model=model,
                    architecture=architecture,
                    # Location
                    phase="decode",
                    block_idx=layer_info.block_idx,
                    sublayer=layer_info.sublayer,
                    op_type=kernel_name,
                    # Metrics — only latency is available from Nsight CSV
                    latency_us=latency_us,
                    mem_bytes=0,
                    mem_peak_bytes=0,
                    power_mw=0.0,
                    flops=0,
                    bandwidth_bytes_sec=0,
                    # Context
                    input_tokens=input_tokens,
                    output_token_idx=0,
                    quantization=quantization,
                    batch_size=1,
                )
            )

    return records
