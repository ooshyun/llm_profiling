"""Minimal OpenAI-style SSE chunk handling shared by all engines."""
from __future__ import annotations

import json
from typing import Optional


def parse_sse_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return {"_done": True}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def delta_content(chunk: dict) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return delta.get("content") or ""


def usage_of(chunk: dict) -> Optional[dict]:
    return chunk.get("usage") or None
