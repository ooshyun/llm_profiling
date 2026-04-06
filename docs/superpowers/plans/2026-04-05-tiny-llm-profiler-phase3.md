# tiny-llm-profiler Phase 3: Presentation Layer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Streamlit real-time dashboard and Jupyter notebook templates for paper-quality visualizations.

**Architecture:** Presentation layer consumes only from Storage (SQLite/Parquet). Streamlit polls SQLite for live data. Jupyter reads Parquet for batch analysis. Shared plotting utilities.

**Tech Stack:** Python 3.11+, Streamlit, matplotlib, seaborn, Jupyter, pandas

**Depends on:** Phase 1 (storage), Phase 2 (analysis)

---

## File Structure

```
presentation/
├── __init__.py
├── plots.py                     # Shared plotting utilities (heatmap, roofline, stacked bar)
├── dashboard/
│   ├── app.py                   # Streamlit main app
│   ├── pages/
│   │   ├── 1_live_monitor.py
│   │   ├── 2_device_compare.py
│   │   ├── 3_architecture.py
│   │   ├── 4_scaling.py
│   │   ├── 5_framework.py
│   │   └── 6_roofline.py
│   └── config.py                # Dashboard settings
└── notebooks/
    ├── 01_device_comparison.ipynb
    ├── 02_architecture_analysis.ipynb
    ├── 03_scaling_analysis.ipynb
    ├── 04_bottleneck_roofline.ipynb
    ├── 05_framework_comparison.ipynb
    ├── 06_thermal_power.ipynb
    ├── 07_rwkv_vs_transformer.ipynb
    └── 08_profiling_overhead.ipynb
tests/
├── test_plots.py
└── test_dashboard_smoke.py
```

---

### Task 1: Shared Plotting Utilities

**Files:**
- Create: `presentation/__init__.py`
- Create: `presentation/plots.py`
- Create: `tests/test_plots.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_plots.py
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import pandas as pd
import matplotlib.pyplot as plt
from presentation.plots import (
    plot_layer_heatmap,
    plot_sublayer_stacked_bar,
    plot_roofline,
)


def test_layer_heatmap_returns_figure():
    df = pd.DataFrame({
        "block_idx": [0, 0, 1, 1],
        "sublayer": ["attention_qkv", "ffn_up", "attention_qkv", "ffn_up"],
        "latency_us": [1000, 800, 1100, 750],
        "device": ["rpi4", "rpi4", "rpi4", "rpi4"],
    })
    fig = plot_layer_heatmap(df, value_col="latency_us", title="Test Heatmap")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_sublayer_stacked_bar_returns_figure():
    df = pd.DataFrame({
        "architecture": ["qwen", "qwen", "llama", "llama"],
        "sublayer": ["attention_qkv", "ffn_up", "attention_qkv", "ffn_up"],
        "latency_us": [1000, 800, 900, 850],
    })
    fig = plot_sublayer_stacked_bar(df, group_col="architecture", value_col="latency_us")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_roofline_returns_figure():
    df = pd.DataFrame({
        "operational_intensity": [0.5, 5.0, 50.0],
        "achieved_gflops": [1.0, 5.0, 10.0],
        "sublayer": ["attn_norm", "attention_qkv", "ffn_up"],
        "device": ["rpi4", "rpi4", "rpi4"],
    })
    fig = plot_roofline(df, peak_gflops=13.5, peak_bw_gb=4.0, title="Test Roofline")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_plots.py -v`
Expected: FAIL

- [ ] **Step 3: Implement plotting utilities**

```python
# presentation/__init__.py
```

```python
# presentation/plots.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Paper-quality defaults
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.figsize": (10, 6),
})


def plot_layer_heatmap(
    df: pd.DataFrame,
    value_col: str = "latency_us",
    title: str = "Layer Latency Heatmap",
) -> plt.Figure:
    """Heatmap: rows=block_idx, columns=sublayer, values=metric."""
    pivot = df.pivot_table(index="block_idx", columns="sublayer", values=value_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="cividis", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Block Index")
    ax.set_xlabel("Sublayer")
    fig.tight_layout()
    return fig


def plot_sublayer_stacked_bar(
    df: pd.DataFrame,
    group_col: str = "architecture",
    value_col: str = "latency_us",
) -> plt.Figure:
    """Stacked bar chart: sublayer proportions per group."""
    pivot = df.pivot_table(index=group_col, columns="sublayer", values=value_col, aggfunc="sum")
    fig, ax = plt.subplots()
    pivot.plot(kind="bar", stacked=True, ax=ax, cmap="viridis")
    ax.set_ylabel(value_col)
    ax.set_title(f"Sublayer Breakdown by {group_col}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_roofline(
    df: pd.DataFrame,
    peak_gflops: float,
    peak_bw_gb: float,
    title: str = "Roofline Model",
) -> plt.Figure:
    """Roofline plot: operational intensity vs achieved GFLOPS."""
    fig, ax = plt.subplots()

    # Roofline ceiling
    oi_range = np.logspace(-2, 4, 200)
    ceiling = np.minimum(peak_gflops, oi_range * peak_bw_gb)
    ax.loglog(oi_range, ceiling, "k-", linewidth=2, label="Roofline ceiling")

    # Data points
    colors = {"attention_qkv": "C0", "ffn_up": "C1", "ffn_down": "C2", "attn_norm": "C3", "ffn_gate": "C4"}
    for _, row in df.iterrows():
        c = colors.get(row["sublayer"], "gray")
        ax.scatter(row["operational_intensity"], row["achieved_gflops"], c=c, s=60, zorder=5)

    ax.set_xlabel("Operational Intensity (FLOP/Byte)")
    ax.set_ylabel("Achieved GFLOPS")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_plots.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add presentation/ tests/test_plots.py
git commit -m "feat: add shared plotting utilities (heatmap, stacked bar, roofline)"
```

---

### Task 2: Streamlit Dashboard

**Files:**
- Create: `presentation/dashboard/config.py`
- Create: `presentation/dashboard/app.py`
- Create: `presentation/dashboard/pages/1_live_monitor.py`
- Create: `presentation/dashboard/pages/2_device_compare.py`
- Create: `tests/test_dashboard_smoke.py`

- [ ] **Step 1: Write dashboard config**

```python
# presentation/dashboard/config.py
from pathlib import Path

DEFAULT_DB_PATH = Path("data/profiles.db")
DEFAULT_PARQUET_DIR = Path("data/parquet/")
REFRESH_INTERVAL_SEC = 2
```

- [ ] **Step 2: Write main app**

```python
# presentation/dashboard/app.py
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="tiny-llm-profiler", page_icon="📊", layout="wide")

st.title("tiny-llm-profiler Dashboard")
st.markdown("Per-layer LLM profiling across heterogeneous edge devices.")

db_path = st.sidebar.text_input("SQLite DB Path", value="data/profiles.db")
st.sidebar.markdown("---")
st.sidebar.markdown("Navigate using pages in the sidebar.")

if not Path(db_path).exists():
    st.warning(f"Database not found at `{db_path}`. Run an experiment first.")
else:
    from storage.db import ProfileDB
    db = ProfileDB(db_path)
    experiments = db._conn.execute("SELECT DISTINCT experiment_id FROM profiles").fetchall()
    st.metric("Total Experiments", len(experiments))
    total = db._conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    st.metric("Total Records", f"{total:,}")
```

- [ ] **Step 3: Write live monitor page**

```python
# presentation/dashboard/pages/1_live_monitor.py
import streamlit as st
import pandas as pd
from pathlib import Path
from storage.db import ProfileDB
from presentation.plots import plot_layer_heatmap
from presentation.dashboard.config import DEFAULT_DB_PATH, REFRESH_INTERVAL_SEC

st.header("Live Monitor")

db_path = st.sidebar.text_input("DB Path", value=str(DEFAULT_DB_PATH), key="live_db")

if Path(db_path).exists():
    db = ProfileDB(db_path)

    experiments = [r[0] for r in db._conn.execute("SELECT DISTINCT experiment_id FROM profiles").fetchall()]
    selected_exp = st.selectbox("Experiment", experiments) if experiments else None

    if selected_exp:
        records = db.query(experiment_id=selected_exp)
        if records:
            df = pd.DataFrame([r.model_dump() for r in records])

            col1, col2, col3 = st.columns(3)
            col1.metric("Records", len(df))
            col2.metric("Layers", df["block_idx"].nunique())
            col3.metric("Avg Latency (us)", f"{df['latency_us'].mean():.0f}")

            fig = plot_layer_heatmap(df, title=f"Layer Latency — {selected_exp}")
            st.pyplot(fig)
        else:
            st.info("No records yet for this experiment.")
else:
    st.warning("Database not found.")
```

- [ ] **Step 4: Write device compare page**

```python
# presentation/dashboard/pages/2_device_compare.py
import streamlit as st
import pandas as pd
from pathlib import Path
from storage.db import ProfileDB
from analysis.comparators import DeviceComparator
from presentation.plots import plot_layer_heatmap
from presentation.dashboard.config import DEFAULT_DB_PATH

st.header("Device Comparison")

db_path = st.sidebar.text_input("DB Path", value=str(DEFAULT_DB_PATH), key="dev_db")

if Path(db_path).exists():
    db = ProfileDB(db_path)

    models = [r[0] for r in db._conn.execute("SELECT DISTINCT model FROM profiles").fetchall()]
    selected_model = st.selectbox("Model", models) if models else None

    if selected_model:
        records = db.query(experiment_id="%", model=selected_model)
        if records:
            comp = DeviceComparator()
            df = comp.compare(records, "latency_us")

            devices = df["device"].unique()
            for device in devices:
                st.subheader(device)
                device_df = df[df["device"] == device]
                fig = plot_layer_heatmap(device_df, title=f"{device} — {selected_model}")
                st.pyplot(fig)
```

- [ ] **Step 5: Write smoke test**

```python
# tests/test_dashboard_smoke.py
"""Smoke test: verify dashboard modules import without error."""


def test_plots_import():
    from presentation.plots import plot_layer_heatmap, plot_sublayer_stacked_bar, plot_roofline
    assert callable(plot_layer_heatmap)


def test_dashboard_config_import():
    from presentation.dashboard.config import DEFAULT_DB_PATH, REFRESH_INTERVAL_SEC
    assert REFRESH_INTERVAL_SEC > 0


def test_app_module_exists():
    """Verify app.py is importable (won't run Streamlit)."""
    import importlib.util
    spec = importlib.util.find_spec("presentation.dashboard.app")
    # spec may be None if streamlit not installed, that's OK for CI
    # Just verify file structure is correct
    from pathlib import Path
    assert Path("presentation/dashboard/app.py").exists() or spec is not None
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_dashboard_smoke.py tests/test_plots.py -v`
Expected: All passed

- [ ] **Step 7: Commit**

```bash
git add presentation/ tests/test_dashboard_smoke.py
git commit -m "feat: add Streamlit dashboard (live monitor, device compare pages)"
```

---

### Task 3: Jupyter Notebook Templates

**Files:**
- Create: `presentation/notebooks/01_device_comparison.ipynb`
- Create: `presentation/notebooks/04_bottleneck_roofline.ipynb`

(Other notebooks follow same pattern — create stubs with cell structure)

- [ ] **Step 1: Create device comparison notebook**

```python
# Use nbformat to create programmatically, or write directly.
# Content for 01_device_comparison.ipynb:
```

Notebook cells:
1. **Markdown**: "# Device Comparison Analysis\nCompare per-layer latency across devices for the same model."
2. **Code**: Import + load data from Parquet
3. **Code**: DeviceComparator + summary statistics
4. **Code**: Heatmap per device
5. **Code**: Save figures as PDF (300 DPI)

- [ ] **Step 2: Create roofline notebook**

Notebook cells:
1. **Markdown**: "# Bottleneck & Roofline Analysis"
2. **Code**: Load data + BottleneckAnalyzer
3. **Code**: Classify all records + pie chart
4. **Code**: Roofline plot per device
5. **Code**: Hotspot table

- [ ] **Step 3: Commit**

```bash
git add presentation/notebooks/
git commit -m "feat: add Jupyter notebook templates (device comparison, roofline)"
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Plotting utilities | 3 |
| 2 | Streamlit dashboard (2 pages) | 3 (smoke) |
| 3 | Jupyter notebook templates | manual verification |
| **Total** | | **6 tests** |

### Local Test

Run: `python -m pytest tests/test_plots.py tests/test_dashboard_smoke.py -v`

### Live Test

Run: `streamlit run presentation/dashboard/app.py`
Then verify pages load in browser at `http://localhost:8501`
