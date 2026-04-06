"""Device Comparison page — model selector + per-device heatmaps."""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from presentation.dashboard.config import DEFAULT_DB_PATH
from presentation.plots import plot_layer_heatmap

st.set_page_config(page_title="Device Comparison", layout="wide")
st.title("Device Comparison")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Data Source")
db_path = Path(
    st.sidebar.text_input("Database path", value=str(DEFAULT_DB_PATH))
)

metric = st.sidebar.selectbox(
    "Metric",
    options=["latency_us", "mem_bytes", "flops"],
    index=0,
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if not db_path.exists():
    st.warning(f"Database not found: {db_path}")
    st.stop()

con = sqlite3.connect(str(db_path))
models: list = [
    row[0]
    for row in con.execute(
        "SELECT DISTINCT model FROM profile_records ORDER BY model"
    ).fetchall()
]

if not models:
    st.info("No models found in the database.")
    con.close()
    st.stop()

selected_model = st.selectbox("Model", options=models)

df = pd.read_sql_query(
    "SELECT * FROM profile_records WHERE model = ?",
    con,
    params=(selected_model,),
)
con.close()

# ---------------------------------------------------------------------------
# Render one heatmap per device
# ---------------------------------------------------------------------------
devices = df["device"].unique().tolist() if not df.empty else []

if not devices:
    st.warning("No data available for the selected model.")
    st.stop()

st.subheader(f"Per-device layer heatmaps — {metric}")
cols = st.columns(max(len(devices), 1))
for col, device in zip(cols, devices):
    device_df = df[df["device"] == device]
    with col:
        st.markdown(f"**{device}**")
        if metric not in device_df.columns or device_df.empty:
            st.warning("No data.")
        else:
            fig = plot_layer_heatmap(
                device_df,
                value_col=metric,
                title=f"{device} — {metric}",
            )
            st.pyplot(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
