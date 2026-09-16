"""Serving benchmark driver. Runs ON the target box against localhost.

  bench.py run --engine llama.cpp --engine-version 6fdd0ac \
      --model qwen3-8b --fmt gguf-q4km --api-model default \
      --base-url http://127.0.0.1:8080 --scenario S1 \
      --config scenarios.yaml --out results/llamacpp_8b_S1.jsonl
  bench.py summarize <dir> --out summary.md

Dependencies: httpx, pyyaml, stdlib. Python >=3.9.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serving_bench.metrics import (RequestRecord, append_jsonl, load_jsonl,
                                   render_markdown, summarize, tpot_ms)
from serving_bench.sse import (delta_content, delta_reasoning,
                               parse_sse_line, usage_of)

REQUEST_TIMEOUT_S = 900.0


class StreamResult:
    def __init__(self):
        self.start_s = 0.0
        self.first_content_s: Optional[float] = None
        self.last_delta_s = 0.0
        self.end_s = 0.0
        self.n_content_deltas = 0
        self.n_reasoning_deltas = 0
        self.reasoning_chars = 0
        self.usage: Optional[dict] = None
        self.head = ""            # first 120 chars, for eyeballing
        self.error: Optional[str] = None


def stream_once(client, base_url: str, payload: dict) -> StreamResult:
    import httpx  # imported here so summarize works without httpx
    res = StreamResult()
    res.start_s = time.monotonic()
    timeout = httpx.Timeout(connect=10.0, read=REQUEST_TIMEOUT_S,
                            write=60.0, pool=10.0)
    try:
        with client.stream("POST", base_url.rstrip("/")
                           + "/v1/chat/completions", json=payload,
                           timeout=timeout) as resp:
            if resp.status_code != 200:
                body = resp.read().decode("utf-8", "replace")[:300]
                res.error = f"HTTP {resp.status_code}: {body}"
                res.end_s = time.monotonic()
                return res
            for line in resp.iter_lines():
                chunk = parse_sse_line(line)
                if chunk is None:
                    continue
                if chunk.get("_done"):
                    break
                now = time.monotonic()
                u = usage_of(chunk)
                if u:
                    res.usage = u
                content = delta_content(chunk)
                if content:
                    if res.first_content_s is None:
                        res.first_content_s = now
                    res.n_content_deltas += 1
                    if len(res.head) < 120:
                        res.head += content
                    res.last_delta_s = now
                reasoning = delta_reasoning(chunk)
                if reasoning:
                    res.n_reasoning_deltas += 1
                    res.reasoning_chars += len(reasoning)
    except httpx.HTTPError as e:
        res.error = f"{type(e).__name__}: {e}"
    res.end_s = time.monotonic()
    return res


def make_payload(args, messages: List[dict], max_tokens: int) -> dict:
    payload = {
        "model": args.api_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.chat_template_kwargs:
        payload["chat_template_kwargs"] = json.loads(
            args.chat_template_kwargs)
    return payload


def to_record(args, scenario, prompt_id, turn, concurrency, res,
              wall_start_ms) -> RequestRecord:
    completion = (res.usage or {}).get("completion_tokens",
                                       res.n_content_deltas)
    prompt_toks = (res.usage or {}).get("prompt_tokens", -1)
    ttft = ((res.first_content_s - res.start_s) * 1000.0
            if res.first_content_s else -1.0)
    tp = (tpot_ms(res.first_content_s, res.last_delta_s, completion)
          if res.first_content_s else 0.0)
    return RequestRecord(
        engine=args.engine, engine_version=args.engine_version,
        model=args.model, fmt=args.fmt, scenario=scenario,
        prompt_id=prompt_id, turn=turn, concurrency=concurrency,
        prompt_tokens=prompt_toks, completion_tokens=completion,
        ttft_ms=ttft, tpot_ms=tp,
        total_ms=(res.end_s - res.start_s) * 1000.0,
        start_ms=wall_start_ms,
        end_ms=wall_start_ms + (res.end_s - res.start_s) * 1000.0,
        ok=res.error is None and res.first_content_s is not None,
        error=res.error,
        ts=datetime.datetime.now().astimezone().isoformat(),
        power_mode=args.power_mode, ctx=args.ctx,
        runtime=args.runtime, reasoning_deltas=res.n_reasoning_deltas)


def read_prompt(cfg_dir: Path, rel: str) -> str:
    return (cfg_dir / rel).read_text(encoding="utf-8")


def warmup(client, args):
    payload = make_payload(args, [{"role": "user", "content": "ping"}], 16)
    res = stream_once(client, args.base_url, payload)
    status = "ok" if res.error is None else f"FAILED: {res.error}"
    print(f"[warmup] {status}  head={res.head!r}", flush=True)
    if res.error:
        sys.exit(f"warmup failed, aborting: {res.error}")
    if "<think>" in res.head:
        sys.exit("warmup response contains <think> — thinking is ON; "
                 "fix engine flags before benchmarking")
    if res.n_reasoning_deltas:
        sys.exit(f"warmup response contained {res.n_reasoning_deltas} "
                 f"reasoning_content delta(s) ({res.reasoning_chars} chars) "
                 "— thinking is ON; fix engine flags before benchmarking")


def run_s1(client, args, cfg, cfg_dir, out):
    sc = cfg["s1"]
    for p in sc["prompts"]:
        text = read_prompt(cfg_dir, p["file"])
        for i in range(sc["repeats"]):
            wall = time.time() * 1000.0
            res = stream_once(client, args.base_url, make_payload(
                args, [{"role": "user", "content": text}],
                sc["max_tokens"]))
            r = to_record(args, "S1", p["id"], 0, 1, res, wall)
            append_jsonl(out, r)
            print(f"[S1 {p['id']} #{i+1}] ttft={r.ttft_ms:.0f}ms "
                  f"tpot={r.tpot_ms:.0f}ms ok={r.ok}", flush=True)


def run_s2(client, args, cfg, cfg_dir, out):
    sc = cfg["s2"]
    system = read_prompt(cfg_dir, sc["system_file"])
    for t in range(1, sc["turns"] + 1):
        user = sc["user_template"].format(i=t)
        wall = time.time() * 1000.0
        res = stream_once(client, args.base_url, make_payload(
            args, [{"role": "system", "content": system},
                   {"role": "user", "content": user}], sc["max_tokens"]))
        r = to_record(args, "S2", "turn", t, 1, res, wall)
        append_jsonl(out, r)
        print(f"[S2 turn {t:02d}] ttft={r.ttft_ms:.0f}ms ok={r.ok}",
              flush=True)


def run_s3(client_factory, args, cfg, cfg_dir, out):
    sc = cfg["s3"]
    text = read_prompt(cfg_dir, sc["prompt_file"])
    lock = threading.Lock()
    for level in sc["levels"]:
        results = []

        def worker():
            import httpx
            with httpx.Client() as c:
                for _ in range(sc["requests_per_worker"]):
                    wall = time.time() * 1000.0
                    res = stream_once(c, args.base_url, make_payload(
                        args, [{"role": "user", "content": text}],
                        sc["max_tokens"]))
                    with lock:
                        results.append((res, wall))

        threads = [threading.Thread(target=worker) for _ in range(level)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        for res, wall in results:
            append_jsonl(out, to_record(args, "S3", "code", 0, level,
                                        res, wall))
        n_ok = sum(1 for res, _ in results if res.error is None)
        print(f"[S3 c={level}] {n_ok}/{len(results)} ok", flush=True)


def cmd_run(args):
    out_path = Path(args.out)
    if out_path.exists() and not args.append:
        sys.exit(f"--out {args.out} already exists; pass --append to "
                 "append to it, or choose a different --out path")
    import httpx
    import yaml
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg_dir = cfg_path.parent
    with httpx.Client() as client:
        warmup(client, args)
        if args.scenario == "S1":
            run_s1(client, args, cfg, cfg_dir, args.out)
        elif args.scenario == "S2":
            run_s2(client, args, cfg, cfg_dir, args.out)
        elif args.scenario == "S3":
            run_s3(None, args, cfg, cfg_dir, args.out)
        else:
            sys.exit(f"unknown scenario {args.scenario}")
    print(f"done -> {args.out}")


def cmd_summarize(args):
    records = []
    for p in sorted(Path(args.dir).glob("*.jsonl")):
        records.extend(load_jsonl(p))
    summ = summarize(records)
    Path(args.out).write_text(render_markdown(summ), encoding="utf-8")
    turns_csv = Path(args.dir) / "s2_turns.csv"
    with open(turns_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["engine", "model", "fmt", "turn", "ttft_ms"])
        w.writeheader()
        w.writerows(summ["s2_turns"])
    print(f"wrote {args.out} and {turns_csv} "
          f"({len(records)} records)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--engine", required=True)
    r.add_argument("--engine-version", required=True)
    r.add_argument("--runtime", required=True,
                   help="e.g. cu12.2 / cu12.6 — disambiguates baselines "
                        "taken under different CUDA/JetPack runtimes")
    r.add_argument("--model", required=True)      # our short key
    r.add_argument("--fmt", required=True)        # gguf-q4km | gptq-int4 | bf16
    r.add_argument("--api-model", default="default")  # name sent to the API
    r.add_argument("--base-url", required=True)
    r.add_argument("--scenario", required=True)
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--append", action="store_true",
                   help="allow appending to an existing --out file "
                        "(default: refuse to start if it already exists)")
    r.add_argument("--ctx", type=int, default=8192)
    r.add_argument("--power-mode", default="MODE_30W")
    r.add_argument("--chat-template-kwargs", default=None,
                   help='e.g. {"enable_thinking": false} for vLLM/SGLang')
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("summarize")
    s.add_argument("dir")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_summarize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
