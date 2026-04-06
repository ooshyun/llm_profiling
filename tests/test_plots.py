"""Tests for presentation.plots — verify each function returns plt.Figure."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from presentation.plots import plot_layer_heatmap, plot_sublayer_stacked_bar, plot_roofline


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def layer_df() -> pd.DataFrame:
    """Minimal DataFrame suitable for plot_layer_heatmap."""
    rng = np.random.default_rng(42)
    rows = []
    for block_idx in range(4):
        for sublayer in ["attn", "ffn", "norm"]:
            rows.append(
                {
                    "block_idx": block_idx,
                    "sublayer": sublayer,
                    "latency_us": rng.uniform(100, 500),
                    "mem_bytes": rng.integers(1_000, 10_000),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def group_df() -> pd.DataFrame:
    """Minimal DataFrame suitable for plot_sublayer_stacked_bar."""
    rng = np.random.default_rng(7)
    rows = []
    for arch in ["llama", "mistral", "falcon"]:
        for sublayer in ["attn", "ffn", "norm"]:
            rows.append(
                {
                    "architecture": arch,
                    "sublayer": sublayer,
                    "latency_us": rng.uniform(50, 600),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def roofline_df() -> pd.DataFrame:
    """Minimal DataFrame suitable for plot_roofline."""
    rng = np.random.default_rng(13)
    n = 20
    return pd.DataFrame(
        {
            "flops": rng.integers(1_000_000, 100_000_000, n),
            "latency_us": rng.uniform(50, 2000, n),
            "bandwidth_bytes_sec": rng.integers(1_000_000_000, 50_000_000_000, n),
            "sublayer": rng.choice(["attn", "ffn", "norm"], n),
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPlotLayerHeatmap:
    def test_returns_figure(self, layer_df: pd.DataFrame) -> None:
        fig = plot_layer_heatmap(layer_df, value_col="latency_us", title="Test heatmap")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_value_col(self, layer_df: pd.DataFrame) -> None:
        fig = plot_layer_heatmap(layer_df, value_col="mem_bytes")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_title_uses_default(self, layer_df: pd.DataFrame) -> None:
        fig = plot_layer_heatmap(layer_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotSublayerStackedBar:
    def test_returns_figure(self, group_df: pd.DataFrame) -> None:
        fig = plot_sublayer_stacked_bar(group_df, group_col="architecture", value_col="latency_us")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_group(self) -> None:
        df = pd.DataFrame(
            {"architecture": ["llama"] * 3, "sublayer": ["attn", "ffn", "norm"], "latency_us": [100.0, 200.0, 50.0]}
        )
        fig = plot_sublayer_stacked_bar(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_default_group_col(self, group_df: pd.DataFrame) -> None:
        fig = plot_sublayer_stacked_bar(group_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotRoofline:
    def test_returns_figure(self, roofline_df: pd.DataFrame) -> None:
        fig = plot_roofline(roofline_df, peak_gflops=100.0, peak_bw_gb=900.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_title(self, roofline_df: pd.DataFrame) -> None:
        fig = plot_roofline(roofline_df, peak_gflops=50.0, peak_bw_gb=500.0, title="My Roofline")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_df_still_returns_figure(self) -> None:
        empty = pd.DataFrame(columns=["flops", "latency_us", "bandwidth_bytes_sec", "sublayer"])
        fig = plot_roofline(empty, peak_gflops=10.0, peak_bw_gb=100.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
