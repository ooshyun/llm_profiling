from datetime import datetime
from typing import Any, Dict, List

from schemas.profile_schema import ProfileRecord

_DEFAULTS: Dict[str, Any] = {
    "experiment_id": "exp_test_001",
    "timestamp": datetime(2024, 1, 1, 12, 0, 0),
    "device": "rpi4",
    "framework": "ggml",
    "model": "llama-7b",
    "architecture": "llama",
    "phase": "decode",
    "block_idx": 0,
    "sublayer": "attention_qkv",
    "op_type": "matmul",
    "latency_us": 1000.0,
    "mem_bytes": 1024 * 1024,
    "mem_peak_bytes": 2 * 1024 * 1024,
    "power_mw": 5000.0,
    "flops": 1_000_000,
    "bandwidth_bytes_sec": 10_000_000_000,
    "cache_l1_hit_ratio": 0.85,
    "cache_l2_hit_ratio": 0.70,
    "thermal_c": 45.0,
    "gpu_sm_util": None,
    "gpu_tensor_util": None,
    "input_tokens": 128,
    "output_token_idx": 0,
    "quantization": "q4_0",
    "batch_size": 1,
}

_SUBLAYER_SEQUENCE = [
    "attention_qkv",
    "attention_score",
    "attention_out",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "attn_norm",
    "ffn_norm",
]


def make_record(**overrides: Any) -> ProfileRecord:
    """Create a ProfileRecord with sensible defaults, any field overridable."""
    fields = dict(_DEFAULTS)
    fields.update(overrides)
    return ProfileRecord(**fields)


def make_layer_set(
    device: str,
    model: str,
    architecture: str,
    n_layers: int = 4,
    base_attn_latency: float = 1000.0,
    base_ffn_latency: float = 800.0,
) -> List[ProfileRecord]:
    """Generate ProfileRecords for n_layers, each with all standard sublayers."""
    _LATENCY_MAP = {
        "attention_qkv": base_attn_latency * 1.0,
        "attention_score": base_attn_latency * 0.8,
        "attention_out": base_attn_latency * 0.6,
        "ffn_gate": base_ffn_latency * 1.0,
        "ffn_up": base_ffn_latency * 1.2,
        "ffn_down": base_ffn_latency * 0.9,
        "attn_norm": base_attn_latency * 0.1,
        "ffn_norm": base_ffn_latency * 0.1,
    }

    records: List[ProfileRecord] = []
    for layer_idx in range(n_layers):
        for sublayer in _SUBLAYER_SEQUENCE:
            latency = _LATENCY_MAP[sublayer] * (1.0 + layer_idx * 0.01)
            records.append(
                make_record(
                    device=device,
                    model=model,
                    architecture=architecture,
                    block_idx=layer_idx,
                    sublayer=sublayer,
                    latency_us=latency,
                    experiment_id=f"exp_{device}_{model}_{layer_idx}_{sublayer}",
                )
            )
    return records
