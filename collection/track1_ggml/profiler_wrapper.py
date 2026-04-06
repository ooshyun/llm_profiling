"""
profiler_wrapper.py

Parse the JSONL output produced by the C ggml_profiler callback and return a
list of ProfileRecord objects ready for storage / analysis.

Each line in the JSONL file has the shape:
    {"name":"blk.0.attn_q.weight","op":18,"ne":[4096,4096,1,1],
     "bytes":4096,"flops":33554432,"latency_us":1500.0,"phase":0}
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from collection.track1_ggml.tensor_name_parser import parse_tensor_name
from schemas.profile_schema import ProfileRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Op-code → human-readable name
# These values mirror ggml_op in ggml.h (as of llama.cpp ~b3000).
# Only the most common ops encountered during LLM inference are listed; the
# rest fall back to "op_<int>".
# ---------------------------------------------------------------------------
_OP_NAMES: Dict[int, str] = {
    0:  "none",
    1:  "dup",
    2:  "add",
    3:  "add1",
    4:  "acc",
    5:  "sub",
    6:  "mul",
    7:  "div",
    8:  "sqr",
    9:  "sqrt",
    10: "log",
    11: "sum",
    12: "sum_rows",
    13: "mean",
    14: "argmax",
    15: "repeat",
    16: "repeat_back",
    17: "concat",
    18: "mul_mat",
    19: "mul_mat_id",
    20: "out_prod",
    21: "scale",
    22: "set",
    23: "cpy",
    24: "cont",
    25: "reshape",
    26: "view",
    27: "permute",
    28: "transpose",
    29: "get_rows",
    30: "get_rows_back",
    31: "diag",
    32: "diag_mask_inf",
    33: "diag_mask_zero",
    34: "soft_max",
    35: "soft_max_back",
    36: "rope",
    37: "rope_back",
    38: "alibi",
    39: "clamp",
    40: "conv_transpose_1d",
    41: "im2col",
    42: "conv_transpose_2d",
    43: "rms_norm",
    44: "rms_norm_back",
    45: "group_norm",
    46: "soft_max",   # duplicate alias kept for op=46 in some builds
    47: "norm",
    48: "flash_attn",
    49: "flash_attn_back",
    50: "flash_ff",
    51: "win_part",
    52: "win_unpart",
    53: "unary",
    54: "map_unary",
    55: "map_binary",
    56: "map_custom1_f32",
    57: "map_custom2_f32",
    58: "map_custom3_f32",
    59: "map_custom1",
    60: "map_custom2",
    61: "map_custom3",
    62: "cross_entropy_loss",
    63: "cross_entropy_loss_back",
}

# Explicit overrides for op codes that have well-known GGML names
_OP_NAMES[43] = "rms_norm"
_OP_NAMES[46] = "soft_max"

_PHASE_MAP: Dict[int, str] = {
    0: "prefill",
    1: "decode",
}


def _op_name(op_int: int) -> str:
    """Map a ggml_op integer to a human-readable string."""
    return _OP_NAMES.get(op_int, f"op_{op_int}")


def _phase_name(phase_int: int) -> str:
    """Map a phase integer to a human-readable string."""
    return _PHASE_MAP.get(phase_int, f"phase_{phase_int}")


def parse_profiler_output(
    jsonl_path: str,
    experiment_id: str,
    device: str,
    model: str,
    architecture: str,
    quantization: str,
    input_tokens: int,
    batch_size: int = 1,
) -> List[ProfileRecord]:
    """Parse a JSONL file produced by ggml_profiler.c into ProfileRecord list.

    Parameters
    ----------
    jsonl_path:    Path to the JSONL file written by the C callback.
    experiment_id: Unique identifier for this profiling run.
    device:        Human-readable device label (e.g. "rpi5", "m2_macbook").
    model:         Model identifier (e.g. "llama-3-8b").
    architecture:  Architecture family (e.g. "llama").
    quantization:  Quantisation format (e.g. "q4_0", "f16").
    input_tokens:  Number of prompt tokens.
    batch_size:    Batch size (default 1).

    Returns
    -------
    List of ProfileRecord instances, one per valid JSONL line.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Profiler output not found: {jsonl_path}")

    records: List[ProfileRecord] = []
    now = datetime.now(tz=timezone.utc)

    output_token_idx = 0  # increments each time the phase is "decode"
    last_phase: Optional[str] = None

    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Line %d: JSON parse error – %s", lineno, exc)
                continue

            # ----------------------------------------------------------------
            # Required fields
            # ----------------------------------------------------------------
            try:
                tensor_name: str  = raw["name"]
                op_int: int       = int(raw["op"])
                ne: List[int]     = [int(x) for x in raw["ne"]]
                mem_bytes: int    = int(raw["bytes"])
                flops: int        = int(raw["flops"])
                latency_us: float = float(raw["latency_us"])
                phase_int: int    = int(raw["phase"])
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Line %d: missing / malformed field – %s", lineno, exc)
                continue

            # Pad ne to length 4
            while len(ne) < 4:
                ne.append(1)

            # ----------------------------------------------------------------
            # Derived fields
            # ----------------------------------------------------------------
            phase_str = _phase_name(phase_int)
            op_str    = _op_name(op_int)

            # Track output_token_idx: bumps whenever we transition into decode
            if phase_str == "decode" and last_phase != "decode":
                output_token_idx += 1
            last_phase = phase_str

            layer_info = parse_tensor_name(tensor_name)

            # Memory bandwidth: bytes / seconds
            bandwidth: int
            if latency_us > 0.0:
                bandwidth = int(mem_bytes / (latency_us / 1_000_000))
            else:
                bandwidth = 0

            # ----------------------------------------------------------------
            # Build record
            # ----------------------------------------------------------------
            record = ProfileRecord(
                experiment_id=experiment_id,
                timestamp=now,
                device=device,
                framework="ggml",
                model=model,
                architecture=architecture,
                phase=phase_str,
                block_idx=layer_info.block_idx,
                sublayer=layer_info.sublayer,
                op_type=op_str,
                latency_us=latency_us,
                mem_bytes=mem_bytes,
                mem_peak_bytes=mem_bytes,  # C profiler does not track peak separately
                power_mw=0.0,
                flops=flops,
                bandwidth_bytes_sec=bandwidth,
                input_tokens=input_tokens,
                output_token_idx=output_token_idx,
                quantization=quantization,
                batch_size=batch_size,
            )
            records.append(record)

    return records
