"""Generate the prompt files deterministically. Rerunnable; output committed.

Targets (see spec §5.2): s1_short ~30 tok, s1_code ~200 tok,
s1_long ~800 tok, s2_system ~4000 tok.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "prompts"

S1_SHORT = ("Explain the difference between kernel space and user space "
            "in an operating system, in exactly three sentences.")

CODE_BLOCK = '''\
def lru_cache_decorator(maxsize=128):
    """A minimal LRU cache decorator using an OrderedDict."""
    from collections import OrderedDict
    from functools import wraps

    def deco(fn):
        cache = OrderedDict()

        @wraps(fn)
        def wrapper(*args):
            if args in cache:
                cache.move_to_end(args)
                return cache[args]
            result = fn(*args)
            cache[args] = result
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return result

        wrapper.cache = cache
        return wrapper

    return deco


@lru_cache_decorator(maxsize=2)
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)
'''

S1_CODE = ("Review the following Python code. Explain what it does, "
           "point out the bug that makes it incorrect for recursive "
           "functions with a small maxsize, and propose a fix.\n\n"
           "```python\n" + CODE_BLOCK + "```\n")

LONG_PARA = (
    "Edge inference of large language models trades peak throughput for "
    "locality, privacy, and predictable cost. On unified-memory systems "
    "such as NVIDIA's Tegra line, the CPU and GPU contend for the same "
    "LPDDR bandwidth, which makes decode throughput track memory speed "
    "far more closely than compute capability. Mixture-of-experts "
    "architectures shift this balance: only a small subset of parameters "
    "activates per token, so a 35-billion-parameter model can decode at "
    "the speed of a much smaller dense network while retaining most of "
    "its quality. Quantization to four bits further reduces the resident "
    "footprint, at a modest and task-dependent accuracy cost. ")

S1_LONG = ("Summarize the following text in exactly five bullet points, "
           "each under twenty words.\n\n" + LONG_PARA * 7)

TOOL_TMPL = (
    "### tool_{i}: {name}\n"
    "Description: {desc} Accepts a JSON object with fields `target` "
    "(string, required), `options` (object, optional), and `timeout_s` "
    "(number, default 30). Returns a JSON object with `status`, "
    "`payload`, and `elapsed_ms`. Fails with E_TIMEOUT if the deadline "
    "is exceeded, E_PERM if the sandbox denies access, and E_ARG on "
    "malformed input. Retries are the caller's responsibility; at most "
    "two retries with exponential backoff are recommended.\n")

TOOLS = [
    ("read_file", "Reads a UTF-8 text file from the project sandbox."),
    ("write_file", "Writes or overwrites a text file inside the sandbox."),
    ("list_dir", "Lists directory entries with size and mtime."),
    ("grep", "Searches file contents with an RE2-compatible pattern."),
    ("http_get", "Fetches an allow-listed HTTPS URL and returns the body."),
    ("run_tests", "Runs the project test suite and returns failures."),
    ("git_diff", "Returns the unified diff of the working tree."),
    ("git_commit", "Creates a commit from staged changes with a message."),
    ("search_docs", "Semantic search over the internal documentation set."),
    ("open_ticket", "Files an issue in the tracker with title and body."),
    ("query_db", "Runs a read-only SQL query against the metrics store."),
    ("render_chart", "Renders a chart spec to PNG and stores it."),
]

RULES = [
    "Never modify files outside the sandbox root.",
    "Prefer reading existing code before proposing edits.",
    "Cite the file path and line range for every claim about code.",
    "Ask for confirmation before any destructive operation.",
    "Keep answers under 200 words unless the user asks for detail.",
    "Use run_tests after every code change and report the outcome.",
    "Do not fabricate tool outputs; report tool errors verbatim.",
    "When a tool fails twice, stop and summarize the failure.",
    "Respond in the user's language.",
    "Never include secrets or tokens in any output.",
]


def build_system() -> str:
    parts = ["You are DevAgent, an autonomous coding assistant operating "
             "inside a sandboxed repository. Follow every rule below and "
             "use only the declared tools.\n\n## Tools\n"]
    for i, (name, desc) in enumerate(TOOLS, 1):
        parts.append(TOOL_TMPL.format(i=i, name=name, desc=desc))
    parts.append("\n## Rules\n")
    parts += [f"{i}. {r}\n" for i, r in enumerate(RULES, 1)]
    parts.append("\n## Worked example\nUser: Fix the failing test in "
                 "tests/test_io.py.\nAssistant: (calls read_file on "
                 "tests/test_io.py, inspects the assertion, calls grep for "
                 "the symbol, edits src/io.py via write_file, calls "
                 "run_tests, reports the diff and the passing result.)\n")
    text = "".join(parts)
    # tile the tools section until ~17k chars (~4k tokens; calibrated
    # against llama-server actual tokenization, task-6 validation)
    while len(text) < 17000:
        extra_idx = (len(text) // 700) % len(TOOLS)
        name, desc = TOOLS[extra_idx]
        text += TOOL_TMPL.format(i=extra_idx + 100, name=name + "_v2",
                                 desc="(extended variant) " + desc)
    return text


def main():
    OUT.mkdir(exist_ok=True)
    files = {"s1_short.txt": S1_SHORT, "s1_code.txt": S1_CODE,
             "s1_long.txt": S1_LONG, "s2_system.txt": build_system()}
    for name, text in files.items():
        (OUT / name).write_text(text, encoding="utf-8")
        print(f"{name}: {len(text)} chars (~{len(text)//4} tok)")


if __name__ == "__main__":
    main()
