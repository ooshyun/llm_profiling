"""Streamlit entry-point for the tiny-llm-profiler dashboard.

Run with:
    streamlit run presentation/dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

st.set_page_config(page_title="tiny-llm-profiler", page_icon="📊", layout="wide")

st.title("📊 tiny-llm-profiler Dashboard")
st.markdown("Per-layer LLM profiling across heterogeneous edge devices")

# Load data
PARQUET_DIR = Path("data/parquet")
parquet_files = sorted(PARQUET_DIR.glob("*.parquet")) if PARQUET_DIR.exists() else []

if not parquet_files:
    st.warning("No profiling data found in `data/parquet/`. Run experiments first.")
    st.stop()

selected_file = st.sidebar.selectbox("Dataset", parquet_files, format_func=lambda x: x.stem)

@st.cache_data
def load_data(path):
    return pd.read_parquet(path)

df = load_data(selected_file)

# Sidebar filters
st.sidebar.markdown("### Filters")
devices = sorted(df['device'].unique())
selected_devices = st.sidebar.multiselect("Devices", devices, default=devices)

models_available = sorted(df['model'].unique())
selected_models = st.sidebar.multiselect("Models", models_available, default=models_available)

phases = sorted(df['phase'].unique())
selected_phase = st.sidebar.selectbox("Phase", phases, index=phases.index('decode') if 'decode' in phases else 0)

# Filter
mask = (df['device'].isin(selected_devices)) & (df['model'].isin(selected_models)) & (df['phase'] == selected_phase)
filtered = df[mask]

MAIN_SUBLAYERS = ['attention_qkv', 'flash_attention', 'attention_out', 'attn_norm',
                  'ffn_gate', 'ffn_up', 'ffn_down', 'ffn_activation', 'ffn_norm',
                  'rmsnorm', 'lm_head', 'embedding', 'layer_output']
df_main = filtered[filtered['sublayer'].isin(MAIN_SUBLAYERS)]

# ── Metrics ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("Records", f"{len(filtered):,}")
col2.metric("Devices", len(filtered['device'].unique()))
col3.metric("Models", len(filtered['model'].unique()))
col4.metric("Max Layers", int(filtered['block_idx'].max()) + 1 if len(filtered) > 0 else 0)

st.markdown("---")

# ── Tab Layout ──
tab1, tab2, tab3, tab4 = st.tabs(["Sublayer Breakdown", "Per-Block Heatmap", "Device Comparison", "Raw Data"])

with tab1:
    st.subheader("Sublayer Latency Distribution")
    if len(df_main) > 0:
        pivot = df_main.groupby(['device', 'sublayer'])['latency_us'].sum().unstack(fill_value=0)
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        st.bar_chart(pivot_pct)

        st.subheader("Top 20 Sublayers by Total Latency")
        top = df_main.groupby(['device', 'model', 'sublayer']).agg(
            total_ms=('latency_us', lambda x: x.sum() / 1000),
            avg_us=('latency_us', 'mean'),
            count=('latency_us', 'count')
        ).reset_index().sort_values('total_ms', ascending=False).head(20)
        top['avg_us'] = top['avg_us'].round(0)
        top['total_ms'] = top['total_ms'].round(1)
        st.dataframe(top, use_container_width=True)

with tab2:
    st.subheader("Per-Block Latency Heatmap")
    if len(df_main) > 0 and len(selected_devices) > 0:
        device_for_heatmap = st.selectbox("Device", selected_devices, key="hm_device")
        dev_data = df_main[df_main['device'] == device_for_heatmap]
        if len(dev_data) > 0:
            pivot_block = dev_data.pivot_table(
                index='block_idx', columns='sublayer', values='latency_us', aggfunc='sum'
            ).fillna(0).round(0)
            st.dataframe(
                pivot_block.style.background_gradient(cmap='YlOrRd', axis=None),
                use_container_width=True, height=600
            )

with tab3:
    st.subheader("Device Comparison")
    if len(df_main) > 0 and len(selected_devices) >= 2:
        metric = st.selectbox("Metric", ['latency_us', 'mem_bytes', 'flops'], key="cmp_metric")
        comparison = df_main.groupby(['device', 'sublayer'])[metric].sum().unstack(fill_value=0)
        st.dataframe(comparison.style.background_gradient(cmap='Blues', axis=0), use_container_width=True)

        # Performance summary
        st.subheader("Performance Summary")
        summary = []
        for device in selected_devices:
            dev = df_main[df_main['device'] == device]
            total = dev['latency_us'].sum()
            summary.append({
                'Device': device,
                'Total (ms)': round(total / 1000, 1),
                'Avg per op (us)': round(dev['latency_us'].mean(), 0),
                'Ops': len(dev),
                'Top bottleneck': dev.groupby('sublayer')['latency_us'].sum().idxmax() if len(dev) > 0 else '-',
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True)

with tab4:
    st.subheader("Raw Profiling Data")
    st.dataframe(filtered.head(1000), use_container_width=True)
    st.caption(f"Showing first 1000 of {len(filtered):,} records")
