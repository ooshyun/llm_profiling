"""Live Monitor page — experiment selector + layer latency heatmap."""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from presentation.dashboard.config import DEFAULT_DB_PATH
from presentation.plots import plot_layer_heatmap

st.set_page_config(page_title="Live Monitor", layout="wide")
st.title("Live Monitor")

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
experiments: list = [
    row[0]
    for row in con.execute(
        "SELECT DISTINCT experiment_id FROM profile_records ORDER BY experiment_id"
    ).fetchall()
]

if not experiments:
    st.info("No experiments found in the database.")
    con.close()
    st.stop()

selected_exp = st.selectbox("Experiment", options=experiments)

df = pd.read_sql_query(
    "SELECT * FROM profile_records WHERE experiment_id = ?",
    con,
    params=(selected_exp,),
)
con.close()

# ---------------------------------------------------------------------------
# Render heatmap
# ---------------------------------------------------------------------------
st.subheader(f"Layer heatmap — {metric}")
if df.empty or metric not in df.columns:
    st.warning("No data available for the selected experiment and metric.")
else:
    fig = plot_layer_heatmap(df, value_col=metric, title=f"{selected_exp} — {metric}")
    st.pyplot(fig)
    import matplotlib.pyplot as plt
    plt.close(fig)
