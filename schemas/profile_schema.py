from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class ProfileRecord(BaseModel):
    # Identification
    experiment_id: str
    timestamp: datetime
    device: str
    framework: str
    model: str
    architecture: str

    # Location
    phase: str
    block_idx: int
    sublayer: str
    op_type: str

    # Metrics A-E
    latency_us: float
    mem_bytes: int
    mem_peak_bytes: int
    power_mw: float
    flops: int
    bandwidth_bytes_sec: int

    # Metrics F-J
    cache_l1_hit_ratio: Optional[float] = None
    cache_l2_hit_ratio: Optional[float] = None
    thermal_c: float = 0.0
    gpu_sm_util: Optional[float] = None
    gpu_tensor_util: Optional[float] = None

    # Context
    input_tokens: int
    output_token_idx: int
    quantization: str
    batch_size: int


class PlatformSnapshot(BaseModel):
    timestamp: datetime
    cpu_temp_c: float
    gpu_temp_c: Optional[float] = None
    soc_temp_c: Optional[float] = None
    power_mw: float
    cpu_freq_mhz: List[float]
    gpu_freq_mhz: Optional[float] = None
    mem_used_bytes: int
    mem_available_bytes: int
