import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from serving_bench.metrics import (
    RequestRecord, tpot_ms, percentile, load_jsonl, append_jsonl,
    summarize, render_markdown,
)


def rec(**kw):
    base = dict(
        engine="llama.cpp", engine_version="6fdd0ac", model="qwen3-8b",
        fmt="gguf-q4km", scenario="S1", prompt_id="short", turn=0,
        concurrency=1, prompt_tokens=30, completion_tokens=256,
        ttft_ms=200.0, tpot_ms=130.0, total_ms=33000.0,
        start_ms=1000.0, end_ms=34000.0, ok=True, error=None,
        ts="2026-09-15T12:00:00+09:00", power_mode="MODE_30W", ctx=8192,
        runtime="cu12.2", reasoning_deltas=0,
    )
    base.update(kw)
    return RequestRecord(**base)


def test_tpot_ms():
    # 255 gaps over 25.5s -> 100ms/token
    assert abs(tpot_ms(10.0, 35.5, 256) - 100.0) < 1e-6
    assert tpot_ms(10.0, 10.0, 1) == 0.0   # single token: undefined -> 0


def test_percentile():
    vals = [10.0, 20.0, 30.0, 40.0]
    assert percentile(vals, 50) == 25.0
    assert percentile(vals, 95) == 38.5
    assert percentile([7.0], 95) == 7.0


def test_jsonl_roundtrip(tmp_path):
    p = tmp_path / "x.jsonl"
    append_jsonl(p, rec())
    append_jsonl(p, rec(prompt_id="code"))
    back = load_jsonl(p)
    assert len(back) == 2 and back[1].prompt_id == "code"
    assert back[0].completion_tokens == 256


def test_load_jsonl_defaults_runtime_for_old_records(tmp_path):
    # Old records written before the `runtime` field existed have no such
    # key in their JSON; load_jsonl must still parse them, defaulting to
    # "unknown" rather than raising.
    p = tmp_path / "old.jsonl"
    old = dict(
        engine="llama.cpp", engine_version="6fdd0ac", model="qwen3-8b",
        fmt="gguf-q4km", scenario="S1", prompt_id="short", turn=0,
        concurrency=1, prompt_tokens=30, completion_tokens=256,
        ttft_ms=200.0, tpot_ms=130.0, total_ms=33000.0,
        start_ms=1000.0, end_ms=34000.0, ok=True, error=None,
        ts="2026-08-27T12:00:00+09:00", power_mode="MODE_30W", ctx=8192,
    )
    p.write_text(json.dumps(old) + "\n")
    back = load_jsonl(p)
    assert len(back) == 1
    assert back[0].runtime == "unknown"
    assert back[0].reasoning_deltas == 0


def test_key_separates_runtimes_that_would_otherwise_collide():
    # Same engine/engine_version/model/fmt but different CUDA runtime must
    # not be merged into one summary row (the JetPack-upgrade collision bug).
    recs = [rec(runtime="cu12.2", tpot_ms=100.0),
            rec(runtime="cu12.6", tpot_ms=50.0)]
    rows = summarize(recs)["s1"]
    short_rows = [r for r in rows if r["prompt_id"] == "short"]
    assert len(short_rows) == 2
    runtimes = {r["runtime"] for r in short_rows}
    assert runtimes == {"cu12.2", "cu12.6"}
    for r in short_rows:
        assert r["n"] == 1


def test_summarize_s1_groups_by_prompt():
    rows = summarize([rec(tpot_ms=100.0), rec(tpot_ms=120.0),
                      rec(prompt_id="code", tpot_ms=200.0)])["s1"]
    short = next(r for r in rows if r["prompt_id"] == "short")
    assert short["n"] == 2
    assert abs(short["tpot_ms_mean"] - 110.0) < 1e-6
    assert abs(short["gen_tok_s"] - 1000.0 / 110.0) < 1e-6


def test_summarize_s1_cold_warm_split():
    rows = summarize([rec(ttft_ms=5000.0), rec(ttft_ms=200.0),
                      rec(ttft_ms=210.0)])["s1"]
    short = next(r for r in rows if r["prompt_id"] == "short")
    assert short["ttft_first_ms"] == 5000.0
    assert abs(short["ttft_rest_mean_ms"] - 205.0) < 1e-6
    # ttft_ms_mean kept for compatibility: mean of all three
    assert abs(short["ttft_ms_mean"] - (5000.0 + 200.0 + 210.0) / 3.0) < 1e-6


def test_summarize_s2_prefix_ratio():
    recs = [rec(scenario="S2", prompt_id="turn", turn=1, ttft_ms=8000.0)]
    recs += [rec(scenario="S2", prompt_id="turn", turn=t, ttft_ms=400.0)
             for t in range(2, 21)]
    row = summarize(recs)["s2"][0]
    assert row["ttft_turn1_ms"] == 8000.0
    assert abs(row["ttft_rest_mean_ms"] - 400.0) < 1e-6
    assert abs(row["prefix_speedup"] - 20.0) < 1e-6
    turns = summarize(recs)["s2_turns"]
    assert len(turns) == 20 and turns[0]["turn"] == 1


def test_summarize_s3_aggregate_tok_s():
    # 2 workers, wall span 0->10s, 100 tokens each -> 20 tok/s aggregate
    recs = [
        rec(scenario="S3", concurrency=2, completion_tokens=100,
            start_ms=0.0, end_ms=9000.0, ttft_ms=300.0),
        rec(scenario="S3", concurrency=2, completion_tokens=100,
            start_ms=1000.0, end_ms=10000.0, ttft_ms=500.0),
    ]
    row = summarize(recs)["s3"][0]
    assert row["concurrency"] == 2
    assert abs(row["agg_tok_s"] - 20.0) < 1e-6
    assert row["ttft_p50_ms"] == 400.0
    assert row["errors"] == 0


def test_summarize_skips_failed_records_but_counts_them():
    recs = [rec(scenario="S3", concurrency=1),
            rec(scenario="S3", concurrency=1, ok=False, error="timeout",
                completion_tokens=0)]
    row = summarize(recs)["s3"][0]
    assert row["errors"] == 1 and row["n"] == 1


def test_render_markdown_has_three_tables():
    md = render_markdown(summarize([rec()]))
    assert "## S1" in md and "## S2" in md and "## S3" in md
    assert "qwen3-8b" in md
    assert "runtime" in md and "cu12.2" in md
