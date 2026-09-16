import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "scripts" / "serving_bench" / "bench.py"


def test_run_refuses_to_start_if_out_already_exists(tmp_path):
    out = tmp_path / "existing.jsonl"
    out.write_text('{"already": "here"}\n')
    r = subprocess.run(
        [sys.executable, str(BENCH), "run",
         "--engine", "llama.cpp", "--engine-version", "6fdd0ac",
         "--runtime", "cu12.2", "--model", "qwen3-8b", "--fmt", "gguf-q4km",
         # base-url deliberately unreachable — the guard must fire before
         # any network call is attempted, so this must never be hit.
         "--base-url", "http://127.0.0.1:1",
         "--scenario", "S1", "--config", str(tmp_path / "no-such.yaml"),
         "--out", str(out)],
        capture_output=True, text=True, timeout=30)
    assert r.returncode != 0
    assert str(out) in (r.stdout + r.stderr)
    assert "already exists" in (r.stdout + r.stderr)
    # untouched — the guard must fire before anything overwrites it
    assert out.read_text() == '{"already": "here"}\n'


def test_run_append_flag_allows_existing_out(tmp_path):
    out = tmp_path / "existing.jsonl"
    out.write_text('{"already": "here"}\n')
    cfg = tmp_path / "no-such.yaml"
    r = subprocess.run(
        [sys.executable, str(BENCH), "run",
         "--engine", "llama.cpp", "--engine-version", "6fdd0ac",
         "--runtime", "cu12.2", "--model", "qwen3-8b", "--fmt", "gguf-q4km",
         "--base-url", "http://127.0.0.1:1",
         "--scenario", "S1", "--config", str(cfg),
         "--out", str(out), "--append"],
        capture_output=True, text=True, timeout=30)
    # Passes the --out guard with --append, then fails downstream (missing
    # config file) rather than on the "already exists" check.
    assert "already exists" not in (r.stdout + r.stderr)
