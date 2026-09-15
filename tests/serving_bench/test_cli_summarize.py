import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "scripts" / "serving_bench" / "bench.py"


def test_summarize_cli(tmp_path):
    row = dict(
        engine="llama.cpp", engine_version="6fdd0ac", model="qwen3-8b",
        fmt="gguf-q4km", scenario="S1", prompt_id="short", turn=0,
        concurrency=1, prompt_tokens=30, completion_tokens=256,
        ttft_ms=200.0, tpot_ms=130.0, total_ms=33000.0,
        start_ms=0.0, end_ms=33000.0, ok=True, error=None,
        ts="t", power_mode="MODE_30W", ctx=8192)
    (tmp_path / "a.jsonl").write_text(json.dumps(row) + "\n")
    out_md = tmp_path / "summary.md"
    r = subprocess.run([sys.executable, str(BENCH), "summarize",
                        str(tmp_path), "--out", str(out_md)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = out_md.read_text()
    assert "## S1" in text and "qwen3-8b" in text
    # s2_turns CSV emitted next to summary
    assert (tmp_path / "s2_turns.csv").exists()
