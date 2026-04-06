"""
Parser for ggml tensor names → (block_idx, sublayer) pairs.

Handles TWO naming conventions:
1. Weight names: "blk.N.component.weight" (from GGUF model file)
2. Runtime op names: "component-N" or "component-N (view)" (from ggml compute graph)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class LayerInfo:
    block_idx: int  # -1 for non-block tensors
    sublayer: str   # canonical sublayer name


# === Runtime op names: "component-N" pattern ===
# These are the actual names seen in cb_eval during inference.
# Format: {name}-{layer_index} or {name}-{layer_index} (view) etc.

_RUNTIME_COMPONENT_MAP: Dict[str, str] = {
    # Attention
    "Qcur":      "attention_qkv",
    "Kcur":      "attention_qkv",
    "Vcur":      "attention_qkv",
    "kqv_out":   "attention_out",
    "__fattn__":  "flash_attention",

    # FFN
    "ffn_gate":  "ffn_gate",
    "ffn_up":    "ffn_up",
    "ffn_out":   "ffn_down",     # ffn_out = output of FFN down projection
    "ffn_swiglu": "ffn_activation",
    "ffn_inp":   "ffn_residual",

    # Norms
    "attn_norm": "attn_norm",
    "ffn_norm":  "ffn_norm",
    "norm":      "rmsnorm",

    # Layer output
    "l_out":     "layer_output",

    # KV cache
    "cache_k":   "kv_cache_k",
    "cache_v":   "kv_cache_v",
}

# Pattern: {component}-{number} optionally followed by modifiers
_RUNTIME_RE = re.compile(r"^([A-Za-z_]+)-(\d+)")

# Pattern: cache_k_l{number} or cache_v_l{number}
_CACHE_RE = re.compile(r"^cache_([kv])_l(\d+)")

# === Weight names: "blk.N.component.weight" pattern ===
_WEIGHT_COMPONENT_MAP: Dict[str, str] = {
    "attn_q":      "attention_qkv",
    "attn_k":      "attention_qkv",
    "attn_v":      "attention_qkv",
    "attn_qkv":    "attention_qkv",
    "attn_output": "attention_out",
    "attn_norm":   "attn_norm",
    "ffn_gate":    "ffn_gate",
    "ffn_up":      "ffn_up",
    "ffn_down":    "ffn_down",
    "ffn_norm":    "ffn_norm",
}

_BLK_RE = re.compile(r"^blk\.(\d+)\.([^.]+)")

# === Global / special names ===
_GLOBAL_MAP: Dict[str, LayerInfo] = {
    "embd":          LayerInfo(block_idx=-1, sublayer="embedding"),
    "result_output": LayerInfo(block_idx=-1, sublayer="lm_head"),
    "result_norm":   LayerInfo(block_idx=-1, sublayer="output_norm"),
    # Weight-style names
    "output_norm.weight":  LayerInfo(block_idx=-1, sublayer="output_norm"),
    "output.weight":       LayerInfo(block_idx=-1, sublayer="lm_head"),
    "token_embd.weight":   LayerInfo(block_idx=-1, sublayer="embedding"),
}

# Op-node prefix patterns (fallback)
_OP_PATTERNS: List[Tuple[str, str]] = [
    ("KQ_soft_max", "attention_score"),
    ("result_output", "lm_head"),
    ("result_norm", "output_norm"),
    ("attn_inp", "attention_mask"),
    ("node_", "internal"),
]


def parse_tensor_name(name: str) -> LayerInfo:
    """Convert a ggml tensor name to a LayerInfo(block_idx, sublayer) pair.

    Handles both weight names (blk.N.component.weight) and runtime op names
    (component-N, component-N (view), etc.)
    """
    # Strip modifiers like " (view)", " (reshaped)", " (permuted)", " (copy)"
    base_name = re.split(r"\s*\(", name)[0].strip()

    # 1. Exact global lookup
    if base_name in _GLOBAL_MAP:
        return _GLOBAL_MAP[base_name]

    # 2. Runtime op names: "component-N"
    m = _RUNTIME_RE.match(base_name)
    if m:
        component = m.group(1)
        block_idx = int(m.group(2))
        sublayer = _RUNTIME_COMPONENT_MAP.get(component)
        if sublayer:
            return LayerInfo(block_idx=block_idx, sublayer=sublayer)

    # 3. KV cache: "cache_k_lN" or "cache_v_lN"
    m = _CACHE_RE.match(base_name)
    if m:
        kv = m.group(1)
        block_idx = int(m.group(2))
        sublayer = f"kv_cache_{kv}"
        return LayerInfo(block_idx=block_idx, sublayer=sublayer)

    # 4. Weight names: "blk.N.component.weight"
    m = _BLK_RE.match(name)
    if m:
        block_idx = int(m.group(1))
        component = m.group(2)
        if "rope" in name:
            return LayerInfo(block_idx=block_idx, sublayer="rope")
        sublayer = _WEIGHT_COMPONENT_MAP.get(component, component)
        return LayerInfo(block_idx=block_idx, sublayer=sublayer)

    # 5. Rope anywhere
    if "rope" in name:
        return LayerInfo(block_idx=-1, sublayer="rope")

    # 6. Op-node prefix patterns
    for prefix, sublayer in _OP_PATTERNS:
        if base_name.startswith(prefix):
            return LayerInfo(block_idx=-1, sublayer=sublayer)

    # 7. Unknown
    return LayerInfo(block_idx=-1, sublayer="unknown")
