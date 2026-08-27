"""Plot sublayer breakdown and decode latency for Qwen3 measurement runs."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from collection.track1_ggml.profiler_wrapper import parse_profiler_output


JOBS = [
    ("data/raw/rpi4_qwen3/rpi4_qwen3_1.7b.jsonl", "RPi4 CPU / Qwen3-1.7B"),
    ("data/raw/rpi4_qwen3/rpi4_qwen3_4b.jsonl", "RPi4 CPU / Qwen3-4B"),
    ("data/raw/op11_cpu_qwen3/op11_qwen3_1.7b.jsonl", "OP11 CPU / Qwen3-1.7B"),
    ("data/raw/op11_cpu_qwen3/op11_qwen3_4b.jsonl", "OP11 CPU / Qwen3-4B"),
    ("data/raw/op11_cpu_qwen3/op11_qwen3_8b.jsonl", "OP11 CPU / Qwen3-8B"),
    ("data/raw/op11_cpu_qwen3/op11_qwen3_14b.jsonl", "OP11 CPU / Qwen3-14B"),
    ("data/raw/op11_cpu_qwen3/op11_qwen3_1.7b_gpu.jsonl", "OP11 GPU / Qwen3-1.7B"),
    ("data/raw/op11_cpu_qwen3/op11_qwen3_4b_gpu.jsonl", "OP11 GPU / Qwen3-4B"),
    ("data/raw/orin_qwen3/orin_cpu_qwen3_1.7b.jsonl", "Orin CPU / Qwen3-1.7B"),
    ("data/raw/orin_qwen3/orin_gpu_qwen3_8b.jsonl", "Orin GPU / Qwen3-8B"),
    ("data/raw/orin_qwen3/orin_gpu_qwen3_14b.jsonl", "Orin GPU / Qwen3-14B"),
    ("data/raw/orin_qwen3/orin_gpu_qwen3_32b.jsonl", "Orin GPU / Qwen3-32B"),
    ("data/raw/orin_qwen3/orin_gpu_qwen3_30b_a3b.jsonl", "Orin GPU / Qwen3-30B-A3B (MoE)"),
    ("data/raw/orin_qwen3/orin_gpu_qwen35_27b.jsonl", "Orin GPU / Qwen3.5-27B"),
    ("data/raw/orin_qwen3/orin_gpu_qwen35_35b_a3b.jsonl", "Orin GPU / Qwen3.5-35B-A3B (MoE)"),
    ("data/raw/orin_qwen3/orin_gpu_qwen35_122b_a10b.jsonl", "Orin partial GPU / Qwen3.5-122B-A10B (MoE)"),
]

# Which sublayers to plot (in order). Others go to "other".
KEEP = [
    "ffn_down", "ffn_up", "ffn_gate", "ffn_activation",
    "attention_qkv", "flash_attention", "kv_cache_k", "kv_cache_v",
    "rmsnorm", "attn_norm", "ffn_norm", "layer_output", "lm_head",
]

COLORS = plt.cm.tab20.colors


def collect():
    rows = []
    for jsonl, label in JOBS:
        p = ROOT / jsonl
        if not p.exists():
            continue
        recs = parse_profiler_output(
            str(p), experiment_id=label, device=label.split(" / ")[0],
            model=label.split(" / ")[1], architecture="qwen3",
            quantization="q4_k_m", input_tokens=5,
        )
        for r in recs:
            rows.append({
                "label": label,
                "phase": r.phase,
                "sublayer": r.sublayer,
                "latency_us": r.latency_us,
            })
    return pd.DataFrame(rows)


def plot_breakdown(df: pd.DataFrame, out: Path):
    decode = df[df.phase == "decode"]
    pivot = decode.groupby(["label", "sublayer"])["latency_us"].sum().unstack(fill_value=0)
    pivot["other"] = pivot.drop(columns=[c for c in KEEP if c in pivot.columns]).sum(axis=1)
    pivot = pivot[[c for c in KEEP if c in pivot.columns] + ["other"]]
    # Normalize to %
    pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    # Order by total latency (slowest first → top)
    order = pivot.sum(axis=1).sort_values(ascending=True).index
    pct = pct.loc[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    pct.plot(kind="barh", stacked=True, ax=ax, color=COLORS, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("decode latency share (%)")
    ax.set_title("Per-sublayer share of decode time — Qwen3/3.5 across edge devices\n(cb_eval profile, lower-is-faster ordering)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print(f"saved {out}")


def plot_decode_speed(df: pd.DataFrame, out: Path):
    decode = df[df.phase == "decode"]
    g = decode.groupby("label")["latency_us"].sum().sort_values()
    # ms per token assuming 2 decode tokens (n=8 with 6 generated, but we capture per-decode-call)
    # Use raw cb_eval total / decode-tokens not derivable here; use per-record sum
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(g.index, g.values / 1000, color="steelblue", edgecolor="black")
    ax.set_xlabel("total decode latency (ms, cb_eval)")
    ax.set_title("Decode latency under cb_eval profiling\n(absolute speeds reflect synchronization overhead, not real inference)")
    for bar, v in zip(bars, g.values / 1000):
        ax.text(v, bar.get_y() + bar.get_height() / 2, f" {v:.0f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print(f"saved {out}")


def main():
    df = collect()
    out_dir = ROOT / "claudedocs/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_breakdown(df, out_dir / "fig8_qwen3_sublayer_breakdown.png")
    plot_decode_speed(df, out_dir / "fig9_qwen3_decode_latency.png")


if __name__ == "__main__":
    main()
