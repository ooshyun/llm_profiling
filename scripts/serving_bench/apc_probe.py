#!/usr/bin/env python3
"""Replayable probe: fixed request sequence against a FRESHLY STARTED vLLM.

Run once with --enable-prefix-caching and once without, then diff the JSON.
Greedy decoding (temp 0, seed fixed) makes the outputs comparable, so any text
difference between the two runs is attributable to prefix caching.

Sequence (order matters -- R1 is a guaranteed miss on a fresh server):
  R1  [P] + Q2   cold prefill, nothing cached yet
  R2  [P] + Q1   shares the 4k prefix P with R1  -> PARTIAL hit when APC is on
  R3  [P] + Q2   identical to R1                 -> FULL hit when APC is on
  R4  [P] + Q3   shares P only                   -> PARTIAL hit
"""
import json, sys, time, urllib.request

BASE, MODEL = "http://127.0.0.1:8000", "qwen3.5-35b-a3b"
SYS = open("prompts/s2_system.txt").read()
Q1 = "Task 1: list which tools you would call, in order, to rename a function across the repository and verify nothing broke. Answer in at most five numbered steps."
Q2 = "Task 2: describe how you would locate the slowest test in the suite and confirm the cause. Answer in at most five numbered steps."
Q3 = "Task 3: outline how you would bisect a regression introduced somewhere in the last fifty commits. Answer in at most five numbered steps."

def ask(user):
    payload = {"model": MODEL,
               "messages": [{"role": "system", "content": SYS},
                            {"role": "user", "content": user}],
               "temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 1234,
               "max_tokens": 128, "stream": False,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        body = json.loads(r.read())
    el = time.time() - t0
    u = body.get("usage", {})
    d = u.get("prompt_tokens_details") or {}
    return {"text": body["choices"][0]["message"].get("content") or "",
            "elapsed_s": round(el, 3),
            "prompt_tokens": u.get("prompt_tokens"),
            "cached_tokens": d.get("cached_tokens"),
            "completion_tokens": u.get("completion_tokens")}

SEQ = [("R1_Q2_cold", Q2), ("R2_Q1_prefixhit", Q1),
       ("R3_Q2_fullhit", Q2), ("R4_Q3_prefixhit", Q3)]

out = {"system_prompt_chars": len(SYS), "requests": {}}
for tag, q in SEQ:
    r = ask(q)
    out["requests"][tag] = r
    print(f"{tag:18s} {r['elapsed_s']:7.2f}s  prompt={r['prompt_tokens']:>5} "
          f"cached={r['cached_tokens']}  out={r['completion_tokens']}")

path = sys.argv[1]
json.dump(out, open(path, "w"), indent=2)
print("wrote", path)
