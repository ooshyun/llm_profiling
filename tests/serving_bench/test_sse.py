from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from serving_bench.sse import (parse_sse_line, delta_content,
                               delta_reasoning, usage_of)


def test_parse_sse_line():
    assert parse_sse_line("") is None
    assert parse_sse_line(": keepalive") is None
    assert parse_sse_line("data: [DONE]") == {"_done": True}
    d = parse_sse_line('data: {"choices":[{"delta":{"content":"hi"}}]}')
    assert d["choices"][0]["delta"]["content"] == "hi"


def test_delta_content_normal_and_empty():
    assert delta_content(
        {"choices": [{"delta": {"content": "abc"}}]}) == "abc"
    assert delta_content({"choices": [{"delta": {}}]}) == ""
    assert delta_content({"choices": []}) == ""          # usage-only chunk
    assert delta_content({"usage": {}}) == ""


def test_delta_content_ignores_reasoning():
    # thinking leak must not count as first content token
    chunk = {"choices": [{"delta": {"reasoning_content": "hmm"}}]}
    assert delta_content(chunk) == ""


def test_delta_reasoning():
    chunk = {"choices": [{"delta": {"reasoning_content": "hmm"}}]}
    assert delta_reasoning(chunk) == "hmm"
    assert delta_reasoning(
        {"choices": [{"delta": {"content": "hi"}}]}) == ""
    assert delta_reasoning({"choices": [{"delta": {}}]}) == ""
    assert delta_reasoning({"choices": []}) == ""


def test_usage_of():
    assert usage_of({"usage": {"completion_tokens": 5}}) == \
        {"completion_tokens": 5}
    assert usage_of({"choices": []}) is None
