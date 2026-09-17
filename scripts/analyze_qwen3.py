"""Parse Qwen3 JSONL files into ProfileRecords, store, and run sublayer analysis."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from collection.track1_ggml.profiler_wrapper import parse_profiler_output
from storage.parquet_store import ParquetStore


JOBS = [
    # (jsonl_path, experiment_id, device, model, architecture, quantization, input_tokens)
    ("data/raw/rpi4_qwen3/rpi4_qwen3_1.7b.jsonl",
     "rpi4_qwen3_1.7b", "rpi4", "qwen3-1.7b", "qwen3", "q4_k_m", 5),
    ("data/raw/rpi4_qwen3/rpi4_qwen3_4b.jsonl",
     "rpi4_qwen3_4b", "rpi4", "qwen3-4b", "qwen3", "q4_k_m", 5),
    ("data/raw/op11_cpu_qwen3/op11_qwen3_1.7b.jsonl",
     "op11cpu_qwen3_1.7b", "op11_cpu", "qwen3-1.7b", "qwen3", "q4_k_m", 5),
    ("data/raw/op11_cpu_qwen3/op11_qwen3_4b.jsonl",
     "op11cpu_qwen3_4b", "op11_cpu", "qwen3-4b", "qwen3", "q4_k_m", 5),
    ("data/raw/op11_cpu_qwen3/op11_qwen3_8b.jsonl",
     "op11cpu_qwen3_8b", "op11_cpu", "qwen3-8b", "qwen3", "q4_k_m", 5),
    ("data/raw/op11_cpu_qwen3/op11_qwen3_14b.jsonl",
     "op11cpu_qwen3_14b", "op11_cpu", "qwen3-14b", "qwen3", "q4_k_m", 5),
    ("data/raw/op11_cpu_qwen3/op11_qwen3_1.7b_gpu.jsonl",
     "op11gpu_qwen3_1.7b", "op11_gpu", "qwen3-1.7b", "qwen3", "q4_k_m", 5),
    ("data/raw/op11_cpu_qwen3/op11_qwen3_4b_gpu.jsonl",
     "op11gpu_qwen3_4b", "op11_gpu", "qwen3-4b", "qwen3", "q4_k_m", 5),
    ("data/raw/orin_qwen3/orin_cpu_qwen3_1.7b.jsonl",
     "orincpu_qwen3_1.7b", "orin_cpu", "qwen3-1.7b", "qwen3", "q4_k_m", 5),
    ("data/raw/orin_qwen3/orin_gpu_qwen3_8b.jsonl",
     "oringpu_qwen3_8b", "orin_gpu", "qwen3-8b", "qwen3", "q4_k_m", 5),
    ("data/raw/orin_qwen3/orin_gpu_qwen3_14b.jsonl",
     "oringpu_qwen3_14b", "orin_gpu", "qwen3-14b", "qwen3", "q4_k_m", 5),
    ("data/raw/orin_qwen3/orin_gpu_qwen3_32b.jsonl",
     "oringpu_qwen3_32b", "orin_gpu", "qwen3-32b", "qwen3", "q4_k_m", 5),
    ("data/raw/orin_qwen3/orin_gpu_qwen3_30b_a3b.jsonl",
     "oringpu_qwen3_30b_a3b", "orin_gpu", "qwen3-30b-a3b", "qwen3-moe", "q4_k_m", 5),
    ("data/raw/orin_qwen3/orin_gpu_qwen35_27b.jsonl",
     "oringpu_qwen35_27b", "orin_gpu", "qwen3.5-27b", "qwen3.5", "q4_k_m", 5),
    ("data/raw/orin_qwen3/orin_gpu_qwen35_35b_a3b.jsonl",
     "oringpu_qwen35_35b_a3b", "orin_gpu", "qwen3.5-35b-a3b", "qwen3.5-moe", "q4_k_m", 5),
    ("data/raw/orin_qwen3/orin_gpu_qwen35_122b_a10b.jsonl",
     "oringpu_qwen35_122b_a10b", "orin_gpu_part", "qwen3.5-122b-a10b", "qwen3.5-moe", "q4_k_m", 5),
    # OP13 Qwen3 sweep
    ("data/raw/op13_qwen3/op13_qwen3_1.7b.jsonl",
     "op13_qwen3_1.7b", "op13_cpu", "qwen3-1.7b", "qwen3", "q4_k_m", 5),
    ("data/raw/op13_qwen3/op13_qwen3_4b.jsonl",
     "op13_qwen3_4b", "op13_cpu", "qwen3-4b", "qwen3", "q4_k_m", 5),
    ("data/raw/op13_qwen3/op13_qwen3_8b.jsonl",
     "op13_qwen3_8b", "op13_cpu", "qwen3-8b", "qwen3", "q4_k_m", 5),
    # Non-Qwen on Orin
    ("data/raw/orin_qwen3/orin_gpu_llama31_8b.jsonl",
     "orin_llama31_8b", "orin_gpu", "llama-3.1-8b", "llama3", "q4_k_m", 5),
    ("data/raw/orin_qwen3/orin_gpu_gemma2_9b.jsonl",
     "orin_gemma2_9b", "orin_gpu", "gemma-2-9b", "gemma2", "q4_k_m", 5),
    ("data/raw/orin_qwen3/orin_gpu_phi35_mini.jsonl",
     "orin_phi35_mini", "orin_gpu", "phi-3.5-mini", "phi3", "q4_k_m", 5),
    ("data/raw/orin_qwen3/orin_gpu_mistral7b.jsonl",
     "orin_mistral7b", "orin_gpu", "mistral-7b-v0.3", "mistral", "q4_k_m", 5),
    # Non-Qwen on OP11 / OP13 / RPi4
    ("data/raw/op11_cpu_qwen3/op11_phi35_mini.jsonl",
     "op11_phi35_mini", "op11_cpu", "phi-3.5-mini", "phi3", "q4_k_m", 5),
    ("data/raw/op11_cpu_qwen3/op11_llama31_8b.jsonl",
     "op11_llama31_8b", "op11_cpu", "llama-3.1-8b", "llama3", "q4_k_m", 5),
    ("data/raw/op11_cpu_qwen3/op11_mistral7b.jsonl",
     "op11_mistral7b", "op11_cpu", "mistral-7b-v0.3", "mistral", "q4_k_m", 5),
    ("data/raw/op11_cpu_qwen3/op11_gemma2_9b.jsonl",
     "op11_gemma2_9b", "op11_cpu", "gemma-2-9b", "gemma2", "q4_k_m", 5),
    ("data/raw/op13_qwen3/op13_phi35_mini.jsonl",
     "op13_phi35_mini", "op13_cpu", "phi-3.5-mini", "phi3", "q4_k_m", 5),
    ("data/raw/op13_qwen3/op13_llama31_8b.jsonl",
     "op13_llama31_8b", "op13_cpu", "llama-3.1-8b", "llama3", "q4_k_m", 5),
    ("data/raw/op13_qwen3/op13_gemma2_9b.jsonl",
     "op13_gemma2_9b", "op13_cpu", "gemma-2-9b", "gemma2", "q4_k_m", 5),
    ("data/raw/rpi4_qwen3/rpi4_phi35.jsonl",
     "rpi4_phi35_mini", "rpi4", "phi-3.5-mini", "phi3", "q4_k_m", 5),
]


def to_df(records):
    rows = []
    for r in records:
        rows.append({
            "device": r.device,
            "model": r.model,
            "architecture": r.architecture,
            "block_idx": r.block_idx,
            "sublayer": r.sublayer,
            "op_type": r.op_type,
            "phase": r.phase,
            "latency_us": r.latency_us,
            "bytes": r.mem_bytes,
            "flops": r.flops,
            "output_token_idx": r.output_token_idx,
        })
    return pd.DataFrame(rows)


def sublayer_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    decode = df[df.phase == "decode"]
    total = decode["latency_us"].sum()
    by = decode.groupby("sublayer")["latency_us"].sum().sort_values(ascending=False)
    pct = (by / total * 100).round(2)
    return pd.DataFrame({"latency_us": by, "pct": pct})


def main():
    out_dir = ROOT / "data/parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    summary = []

    for jsonl, exp_id, device, model, arch, quant, prompt_tokens in JOBS:
        p = ROOT / jsonl
        if not p.exists():
            print(f"[skip] {jsonl} not found")
            continue
        try:
            recs = parse_profiler_output(
                str(p), experiment_id=exp_id, device=device, model=model,
                architecture=arch, quantization=quant, input_tokens=prompt_tokens,
            )
        except Exception as e:
            print(f"[err] {jsonl}: {e}")
            continue
        all_records.extend(recs)
        df = to_df(recs)
        print(f"\n=== {device} / {model} ({len(recs)} records) ===")
        sb = sublayer_breakdown(df)
        print(sb.head(15).to_string())

        decode = df[df.phase == "decode"]
        total_us = decode["latency_us"].sum()
        n_decode_tokens = (decode["output_token_idx"].max() + 1) if len(decode) else 0
        summary.append({
            "device": device,
            "model": model,
            "records": len(recs),
            "decode_total_us": total_us,
            "decode_tokens": int(n_decode_tokens),
            "decode_us_per_token": (total_us / n_decode_tokens) if n_decode_tokens else 0.0,
            "ffn_pct": float(sb.loc[sb.index.str.contains("ffn", case=False, na=False), "pct"].sum()) if len(sb) else 0.0,
            "attn_pct": float(sb.loc[sb.index.str.contains("attn|attention", case=False, na=False, regex=True), "pct"].sum()) if len(sb) else 0.0,
        })

    if all_records:
        out = out_dir / "qwen3_all.parquet"
        store = ParquetStore(str(out))
        store.write(all_records)
        print(f"\n[parquet] wrote {len(all_records)} records to {out}")

    if summary:
        sdf = pd.DataFrame(summary)
        print("\n========= SUMMARY =========")
        print(sdf.to_string(index=False))
        sdf.to_csv(out_dir / "qwen3_summary.csv", index=False)


if __name__ == "__main__":
    main()
