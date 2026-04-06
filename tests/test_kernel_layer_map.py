"""
Unit tests for KernelLayerMapper.
"""
import json
import tempfile
from pathlib import Path

import pytest

from collection.track2_tvm.kernel_layer_map import KernelLayerInfo, KernelLayerMapper


class TestKernelLayerMapper:
    def test_parse_fused_matmul(self):
        """Exact mapping lookup returns the registered KernelLayerInfo."""
        mapper = KernelLayerMapper()
        mapper.add_mapping("fused_matmul_0", block_idx=3, sublayer="attention_qkv")

        result = mapper.lookup("fused_matmul_0")

        assert result == KernelLayerInfo(block_idx=3, sublayer="attention_qkv")

    def test_lookup_unknown_returns_default(self):
        """A name with no registered rule returns the unknown fallback."""
        mapper = KernelLayerMapper()

        result = mapper.lookup("totally_unknown_kernel_xyz")

        assert result.block_idx == -1
        assert result.sublayer == "unknown"

    def test_build_from_ir_pattern(self):
        """Regex patterns match expected kernel name substrings."""
        mapper = KernelLayerMapper()
        mapper.add_pattern(r"softmax", sublayer="attention_score", block_idx=0)
        mapper.add_pattern(r"rms_norm", sublayer="rms_norm", block_idx=0)

        softmax_result = mapper.lookup("fused_nn_softmax_prim_func_0")
        rms_result = mapper.lookup("fused_rms_norm_add_cast_1")

        assert softmax_result.sublayer == "attention_score"
        assert rms_result.sublayer == "rms_norm"

    def test_save_and_load(self):
        """Round-trip through JSON serialisation preserves all mappings."""
        mapper = KernelLayerMapper()
        mapper.add_mapping("fused_dense_0", block_idx=1, sublayer="ffn_gate")
        mapper.add_pattern(r"softmax", sublayer="attention_score", block_idx=-1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = str(Path(tmp_dir) / "mapper.json")
            mapper.save(path)

            loaded = KernelLayerMapper.load(path)

        # Exact mapping survives
        assert loaded.lookup("fused_dense_0") == KernelLayerInfo(
            block_idx=1, sublayer="ffn_gate"
        )
        # Regex pattern survives
        assert loaded.lookup("fused_nn_softmax_0").sublayer == "attention_score"
        # Unknown still returns default
        assert loaded.lookup("no_match_kernel").sublayer == "unknown"
