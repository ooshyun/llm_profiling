"""Tests for tensor name parser — covers both weight and runtime op names."""
from collection.track1_ggml.tensor_name_parser import parse_tensor_name


# === Weight-style names (blk.N.component.weight) ===

def test_attn_q_maps_to_attention_qkv():
    info = parse_tensor_name("blk.5.attn_q.weight")
    assert info.block_idx == 5
    assert info.sublayer == "attention_qkv"


def test_attn_k_maps_to_attention_qkv():
    info = parse_tensor_name("blk.12.attn_k.weight")
    assert info.block_idx == 12
    assert info.sublayer == "attention_qkv"


def test_attn_v_maps_to_attention_qkv():
    info = parse_tensor_name("blk.0.attn_v.weight")
    assert info.block_idx == 0
    assert info.sublayer == "attention_qkv"


def test_attn_qkv_fused_maps_to_attention_qkv():
    info = parse_tensor_name("blk.3.attn_qkv.weight")
    assert info.block_idx == 3
    assert info.sublayer == "attention_qkv"


def test_attn_output_maps_to_attention_out():
    info = parse_tensor_name("blk.3.attn_output.weight")
    assert info.block_idx == 3
    assert info.sublayer == "attention_out"


def test_attn_norm_maps_to_attn_norm():
    info = parse_tensor_name("blk.0.attn_norm.weight")
    assert info.block_idx == 0
    assert info.sublayer == "attn_norm"


def test_ffn_norm_maps_to_ffn_norm():
    info = parse_tensor_name("blk.2.ffn_norm.weight")
    assert info.block_idx == 2
    assert info.sublayer == "ffn_norm"


def test_ffn_gate_maps_correctly():
    info = parse_tensor_name("blk.7.ffn_gate.weight")
    assert info.block_idx == 7
    assert info.sublayer == "ffn_gate"


def test_ffn_up_maps_correctly():
    info = parse_tensor_name("blk.7.ffn_up.weight")
    assert info.block_idx == 7
    assert info.sublayer == "ffn_up"


def test_ffn_down_maps_correctly():
    info = parse_tensor_name("blk.7.ffn_down.weight")
    assert info.block_idx == 7
    assert info.sublayer == "ffn_down"


def test_rope_in_name_maps_to_rope():
    info = parse_tensor_name("blk.5.attn_q_rope.weight")
    assert info.block_idx == 5
    assert info.sublayer == "rope"


def test_output_norm_weight():
    info = parse_tensor_name("output_norm.weight")
    assert info.block_idx == -1
    assert info.sublayer == "output_norm"


def test_output_weight_maps_to_lm_head():
    info = parse_tensor_name("output.weight")
    assert info.block_idx == -1
    assert info.sublayer == "lm_head"


def test_token_embd_weight_maps_to_embedding():
    info = parse_tensor_name("token_embd.weight")
    assert info.block_idx == -1
    assert info.sublayer == "embedding"


def test_kq_soft_max_op_node():
    info = parse_tensor_name("KQ_soft_max-5")
    assert info.sublayer == "attention_score"


def test_result_output_op_node():
    info = parse_tensor_name("result_output-0")
    assert info.block_idx == -1
    assert info.sublayer == "lm_head"


def test_unknown_tensor_name_returns_unknown():
    info = parse_tensor_name("some_random_tensor")
    assert info.block_idx == -1
    assert info.sublayer == "unknown"


# === Runtime op names (component-N) — actual profiling output ===

def test_runtime_qcur():
    info = parse_tensor_name("Qcur-0")
    assert info.block_idx == 0
    assert info.sublayer == "attention_qkv"


def test_runtime_qcur_with_view():
    info = parse_tensor_name("Qcur-5 (reshaped)")
    assert info.block_idx == 5
    assert info.sublayer == "attention_qkv"


def test_runtime_qcur_with_double_modifier():
    info = parse_tensor_name("Qcur-12 (view) (permuted)")
    assert info.block_idx == 12
    assert info.sublayer == "attention_qkv"


def test_runtime_kcur():
    info = parse_tensor_name("Kcur-3")
    assert info.block_idx == 3
    assert info.sublayer == "attention_qkv"


def test_runtime_vcur():
    info = parse_tensor_name("Vcur-7 (view)")
    assert info.block_idx == 7
    assert info.sublayer == "attention_qkv"


def test_runtime_flash_attention():
    info = parse_tensor_name("__fattn__-15")
    assert info.block_idx == 15
    assert info.sublayer == "flash_attention"


def test_runtime_kqv_out():
    info = parse_tensor_name("kqv_out-0")
    assert info.block_idx == 0
    assert info.sublayer == "attention_out"


def test_runtime_attn_norm():
    info = parse_tensor_name("attn_norm-27")
    assert info.block_idx == 27
    assert info.sublayer == "attn_norm"


def test_runtime_ffn_gate():
    info = parse_tensor_name("ffn_gate-10")
    assert info.block_idx == 10
    assert info.sublayer == "ffn_gate"


def test_runtime_ffn_up():
    info = parse_tensor_name("ffn_up-20")
    assert info.block_idx == 20
    assert info.sublayer == "ffn_up"


def test_runtime_ffn_out():
    info = parse_tensor_name("ffn_out-5")
    assert info.block_idx == 5
    assert info.sublayer == "ffn_down"


def test_runtime_ffn_swiglu():
    info = parse_tensor_name("ffn_swiglu-8")
    assert info.block_idx == 8
    assert info.sublayer == "ffn_activation"


def test_runtime_ffn_norm():
    info = parse_tensor_name("ffn_norm-3")
    assert info.block_idx == 3
    assert info.sublayer == "ffn_norm"


def test_runtime_norm():
    info = parse_tensor_name("norm-14")
    assert info.block_idx == 14
    assert info.sublayer == "rmsnorm"


def test_runtime_layer_output():
    info = parse_tensor_name("l_out-0")
    assert info.block_idx == 0
    assert info.sublayer == "layer_output"


def test_runtime_cache_k():
    info = parse_tensor_name("cache_k_l5 (view)")
    assert info.block_idx == 5
    assert info.sublayer == "kv_cache_k"


def test_runtime_cache_v():
    info = parse_tensor_name("cache_v_l12 (view) (permuted)")
    assert info.block_idx == 12
    assert info.sublayer == "kv_cache_v"


def test_runtime_embd():
    info = parse_tensor_name("embd")
    assert info.block_idx == -1
    assert info.sublayer == "embedding"


def test_runtime_result_output():
    info = parse_tensor_name("result_output")
    assert info.block_idx == -1
    assert info.sublayer == "lm_head"


def test_runtime_result_norm():
    info = parse_tensor_name("result_norm")
    assert info.block_idx == -1
    assert info.sublayer == "output_norm"


def test_runtime_attn_inp():
    info = parse_tensor_name("attn_inp_kq_mask (copy)")
    assert info.sublayer == "attention_mask"


def test_runtime_node_internal():
    info = parse_tensor_name("node_129")
    assert info.sublayer == "internal"


def test_runtime_ffn_inp():
    info = parse_tensor_name("ffn_inp-0")
    assert info.block_idx == 0
    assert info.sublayer == "ffn_residual"
