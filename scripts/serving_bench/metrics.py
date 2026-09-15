"""Records and aggregation for the serving benchmark.

Stdlib only: runs on Mac (py3.9, tests) and Orin (py3.10, runtime).
Schema = spec §5.5 plus additive prompt_id/start_ms/end_ms.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RequestRecord:
    engine: str
    engine_version: str
    model: str
    fmt: str
    scenario: str            # "S1" | "S2" | "S3"
    prompt_id: str           # S1: short|code|long; S2: "turn"; S3: "code"
    turn: int                # S2 1-based turn; 0 otherwise
    concurrency: int         # S3 level; 1 for S1/S2
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    tpot_ms: float
    total_ms: float
    start_ms: float          # epoch ms when request was sent
    end_ms: float            # epoch ms when stream closed
    ok: bool
    error: Optional[str]
    ts: str
    power_mode: str
    ctx: int


def tpot_ms(first_token_s: float, last_token_s: float,
            completion_tokens: int) -> float:
    if completion_tokens < 2:
        return 0.0
    return (last_token_s - first_token_s) * 1000.0 / (completion_tokens - 1)


def percentile(values: List[float], p: float) -> float:
    if not values:
        raise ValueError("percentile of empty list")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def append_jsonl(path, record: RequestRecord) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_jsonl(path) -> List[RequestRecord]:
    names = {f.name for f in fields(RequestRecord)}
    out: List[RequestRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(RequestRecord(**{k: v for k, v in d.items()
                                        if k in names}))
    return out


def _key(r: RequestRecord):
    return (r.engine, r.engine_version, r.model, r.fmt)


def summarize(records: List[RequestRecord]) -> Dict[str, list]:
    s1_rows, s2_rows, s3_rows, s2_turns = [], [], [], []

    s1 = [r for r in records if r.scenario == "S1"]
    for key in sorted({(_key(r), r.prompt_id) for r in s1}):
        (eng, ver, model, fmt), pid = key
        grp = [r for r in s1 if _key(r) == (eng, ver, model, fmt)
               and r.prompt_id == pid and r.ok]
        if not grp:
            continue
        tp = statistics.mean(r.tpot_ms for r in grp)
        s1_rows.append({
            "engine": eng, "engine_version": ver, "model": model, "fmt": fmt,
            "prompt_id": pid, "n": len(grp),
            "prompt_tokens": grp[0].prompt_tokens,
            "ttft_ms_mean": statistics.mean(r.ttft_ms for r in grp),
            "tpot_ms_mean": tp,
            "gen_tok_s": (1000.0 / tp) if tp > 0 else 0.0,
        })

    s2 = [r for r in records if r.scenario == "S2" and r.ok]
    for key in sorted({_key(r) for r in s2}):
        grp = sorted([r for r in s2 if _key(r) == key], key=lambda r: r.turn)
        first = [r for r in grp if r.turn == 1]
        rest = [r for r in grp if r.turn >= 2]
        if not first or not rest:
            continue
        t1 = first[0].ttft_ms
        trest = statistics.mean(r.ttft_ms for r in rest)
        eng, ver, model, fmt = key
        s2_rows.append({
            "engine": eng, "engine_version": ver, "model": model, "fmt": fmt,
            "turns": len(grp), "ttft_turn1_ms": t1,
            "ttft_rest_mean_ms": trest,
            "prefix_speedup": (t1 / trest) if trest > 0 else 0.0,
            "tpot_ms_mean": statistics.mean(r.tpot_ms for r in grp),
        })
        for r in grp:
            s2_turns.append({"engine": eng, "model": model, "fmt": fmt,
                             "turn": r.turn, "ttft_ms": r.ttft_ms})

    s3 = [r for r in records if r.scenario == "S3"]
    for key in sorted({(_key(r), r.concurrency) for r in s3}):
        (eng, ver, model, fmt), conc = key
        grp = [r for r in s3 if _key(r) == (eng, ver, model, fmt)
               and r.concurrency == conc]
        okg = [r for r in grp if r.ok]
        row = {
            "engine": eng, "engine_version": ver, "model": model, "fmt": fmt,
            "concurrency": conc, "n": len(okg),
            "errors": sum(1 for r in grp if not r.ok),
        }
        if okg:
            span_s = (max(r.end_ms for r in okg)
                      - min(r.start_ms for r in okg)) / 1000.0
            row["agg_tok_s"] = (sum(r.completion_tokens for r in okg) / span_s
                                if span_s > 0 else 0.0)
            row["ttft_p50_ms"] = percentile([r.ttft_ms for r in okg], 50)
            row["ttft_p95_ms"] = percentile([r.ttft_ms for r in okg], 95)
        s3_rows.append(row)

    return {"s1": s1_rows, "s2": s2_rows, "s3": s3_rows,
            "s2_turns": s2_turns}


def _table(rows: List[dict], cols: List[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:.1f}" if isinstance(v, float) else str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


def render_markdown(summary: Dict[str, list]) -> str:
    parts = ["# serving_bench summary", ""]
    parts += ["## S1 — single-user chat", "",
              _table(summary["s1"], ["engine", "engine_version", "model",
                                     "fmt", "prompt_id", "n", "prompt_tokens",
                                     "ttft_ms_mean", "tpot_ms_mean",
                                     "gen_tok_s"]), ""]
    parts += ["## S2 — agent loop (4k shared system prompt)", "",
              _table(summary["s2"], ["engine", "engine_version", "model",
                                     "fmt", "turns", "ttft_turn1_ms",
                                     "ttft_rest_mean_ms", "prefix_speedup",
                                     "tpot_ms_mean"]), ""]
    parts += ["## S3 — concurrency", "",
              _table(summary["s3"], ["engine", "engine_version", "model",
                                     "fmt", "concurrency", "n", "errors",
                                     "agg_tok_s", "ttft_p50_ms",
                                     "ttft_p95_ms"]), ""]
    return "\n".join(parts)
