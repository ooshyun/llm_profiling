"""Shared plotting utilities for the tiny-llm-profiler presentation layer.

All public functions return a ``matplotlib.figure.Figure`` so callers control
when (and whether) the figure is displayed or saved.
"""

from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Global matplotlib defaults
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
    }
)


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------


def plot_layer_heatmap(
    df: pd.DataFrame,
    value_col: str = "latency_us",
    title: str = "",
) -> plt.Figure:
    """Return a heatmap of *value_col* pivoted on block_idx (rows) x sublayer (cols).

    Parameters
    ----------
    df:
        DataFrame containing at least the columns ``block_idx``, ``sublayer``,
        and *value_col*.  When multiple rows share the same (block_idx, sublayer)
        pair their values are averaged before pivoting.
    value_col:
        Numeric column to visualise.
    title:
        Optional figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    agg = (
        df.groupby(["block_idx", "sublayer"])[value_col]
        .mean()
        .reset_index()
    )
    pivot = agg.pivot(index="block_idx", columns="sublayer", values=value_col)

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.2), max(4, len(pivot) * 0.5)))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="cividis",
        linewidths=0.3,
        linecolor="white",
        annot=len(pivot) * len(pivot.columns) <= 100,
        fmt=".0f",
        cbar_kws={"label": value_col},
    )
    ax.set_title(title or f"Layer Heatmap — {value_col}")
    ax.set_xlabel("Sublayer")
    ax.set_ylabel("Block index")
    fig.tight_layout()
    return fig


def plot_sublayer_stacked_bar(
    df: pd.DataFrame,
    group_col: str = "architecture",
    value_col: str = "latency_us",
) -> plt.Figure:
    """Return a stacked-bar chart of *value_col* grouped by *group_col* and sublayer.

    Parameters
    ----------
    df:
        DataFrame containing *group_col*, ``sublayer``, and *value_col*.
    group_col:
        Column used as the x-axis category (e.g. ``"architecture"`` or ``"device"``).
    value_col:
        Numeric column to sum into each bar segment.

    Returns
    -------
    matplotlib.figure.Figure
    """
    agg = (
        df.groupby([group_col, "sublayer"])[value_col]
        .mean()
        .reset_index()
    )
    pivot = agg.pivot(index=group_col, columns="sublayer", values=value_col).fillna(0)

    n_sublayers = len(pivot.columns)
    cmap = plt.cm.get_cmap("viridis")
    colors = [cmap(i / max(n_sublayers - 1, 1)) for i in range(n_sublayers)]

    fig, ax = plt.subplots(figsize=(max(6, len(pivot) * 1.5), 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_title(f"Sublayer breakdown by {group_col} — {value_col}")
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    ax.legend(title="Sublayer", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_roofline(
    df: pd.DataFrame,
    peak_gflops: float,
    peak_bw_gb: float,
    title: str = "",
) -> plt.Figure:
    """Return a roofline model chart in log-log space.

    The chart overlays:
    * The compute-bound and memory-bound roofline ceilings.
    * Scatter points for each row in *df*, coloured by ``sublayer``.

    Parameters
    ----------
    df:
        DataFrame with columns ``flops``, ``latency_us``, ``bandwidth_bytes_sec``,
        and ``sublayer``.  Rows missing any of these are silently dropped.
    peak_gflops:
        Theoretical peak compute throughput in GFLOP/s.
    peak_bw_gb:
        Theoretical peak memory bandwidth in GB/s.
    title:
        Optional figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    required = {"flops", "latency_us", "bandwidth_bytes_sec", "sublayer"}
    df = df.dropna(subset=list(required & set(df.columns)))

    # Derived metrics
    latency_s = df["latency_us"] * 1e-6
    achieved_gflops = df["flops"] / latency_s / 1e9
    # arithmetic intensity: FLOP / byte
    bw_bytes = df["bandwidth_bytes_sec"] * latency_s  # bytes moved
    arithmetic_intensity = df["flops"] / bw_bytes.replace(0, np.nan)

    # Ridge point: where memory and compute ceilings intersect
    ridge_intensity = peak_gflops / peak_bw_gb  # FLOP/byte

    # Build roofline ceiling over a log-spaced intensity range
    x_min = max(arithmetic_intensity.dropna().min() * 0.1, 1e-3) if not arithmetic_intensity.dropna().empty else 1e-3
    x_max = max(arithmetic_intensity.dropna().max() * 10, ridge_intensity * 10) if not arithmetic_intensity.dropna().empty else ridge_intensity * 10
    intensities = np.logspace(np.log10(x_min), np.log10(x_max), 300)
    roof = np.minimum(peak_bw_gb * intensities, peak_gflops)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(intensities, roof, "k-", linewidth=2, label="Roofline ceiling")
    ax.axvline(ridge_intensity, color="gray", linestyle="--", linewidth=1, label=f"Ridge point ({ridge_intensity:.1f} FLOP/B)")

    sublayers = df["sublayer"].unique()
    cmap = plt.cm.get_cmap("viridis")
    for idx, sublayer in enumerate(sublayers):
        mask = df["sublayer"] == sublayer
        ax.scatter(
            arithmetic_intensity[mask],
            achieved_gflops[mask],
            label=sublayer,
            color=cmap(idx / max(len(sublayers) - 1, 1)),
            alpha=0.75,
            s=40,
            zorder=3,
        )

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Achieved Performance (GFLOP/s)")
    ax.set_title(title or "Roofline Model")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    return fig
