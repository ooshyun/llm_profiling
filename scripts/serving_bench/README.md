# serving_bench

Drives any OpenAI-compatible chat endpoint with three scenarios and writes
one JSONL line per request. Spec:
`docs/superpowers/specs/2026-08-27-serving-framework-eval-design.md` §5.

## On the Orin

    rsync -a scripts/serving_bench/ home.orin.ts:~/serving_bench/
    ssh home.orin.ts
    cd ~/serving_bench

    # terminal A — engine (one at a time; free -g must be back to idle first)
    ./engines/llama_server.sh 8b          # S1/S2 (np=1)
    ./engines/llama_server.sh 8b 8       # S3 (np=8; ctx splits to 1024/slot)

    # terminal B — monitor
    ./monitor.sh results/tegrastats_llamacpp_8b.log &

    # terminal B — scenarios (always MODE_30W; check: nvpmodel -q)
    python3 bench.py run --engine llama.cpp --engine-version 6fdd0ac \
      --runtime cu12.2 \
      --model qwen3-8b --fmt gguf-q4km --base-url http://127.0.0.1:8080 \
      --scenario S1 --config scenarios.yaml \
      --out results/llamacpp_cu122_8b_S1.jsonl

    # vLLM/SGLang additionally need:
    #   --api-model <HF id> --chat-template-kwargs '{"enable_thinking": false}'

`--runtime` is required (e.g. `cu12.2`, `cu12.6`) and disambiguates baselines
taken under different CUDA/JetPack runtimes — it's part of the summary
grouping key alongside `--engine`/`--engine-version`/`--model`/`--fmt`, so a
post-JetPack-upgrade CUDA 12.6 rerun doesn't silently merge into the CUDA
12.2 rows.

`--out` refuses to start if the file already exists, to avoid silently
mixing runs into one JSONL. Pass `--append` to opt into appending to an
existing file on purpose.

### Phase 2

`engines/vllm.sh` and `engines/sglang.sh` are added in Phase 2, once the
CUDA-enabled PyTorch / container prerequisites are in place (see
`claudedocs/serving_framework_candidates_20260828.md`). Until then only
`engines/llama_server.sh` exists.

### Orin ops notes

- A detached background launch of the form `ssh home.orin.ts 'nohup ... < /dev/null &'`
  can leave the *local* `ssh` process hung waiting on the connection even
  though the remote child keeps running fine detached — Ctrl-C the local
  ssh, don't assume the remote job died.
- `pkill -f llama-server` run *over ssh* matches the ssh command line itself
  (it contains the string `llama-server`), so it can self-match and the ssh
  session exits with status 255 — the target process still dies as
  intended; a nonzero exit here is not a failure signal.

`scripts/serving_bench/orin_stage_downloads.sh` stages the Phase-2 HF model
snapshots and Docker images ahead of time (idempotent, backgroundable via
`nohup ... &` on the Orin).

Note: when summarizing results captured *on* the Orin rather than synced
back to this repo, the results directory is `results/` (not
`results/serving_bench/`) — pass the right path to `bench.py summarize`.

## Summarize (Mac or Orin)

    python3 bench.py summarize results/serving_bench \
      --out results/serving_bench/summary.md

Warm-up (1 request, excluded) runs automatically and aborts if the reply
contains `<think>` — fix the engine's reasoning flags, don't benchmark.

## Methodology: prompt cache

llama-server (and vLLM/SGLang prefix caching) keep prompt prefixes cached
across requests on a running server instance, so TTFT depends on that
server's request history, not just on the prompt being measured — a
prompt that was sent even once before (by a curl probe, a prior scenario,
or a leftover process) will come back "warm" (cache hit) and understate
true cold-start latency by 10-50x. Rules:

- **Restart the engine (or otherwise clear its cache) immediately before
  S1 and immediately before S2.** A server that served any other
  scenario, or that was probed for any reason, is contaminated for TTFT
  purposes even if the model and prompts are unchanged.
- **Never send probe/curl requests carrying benchmark prompt text to the
  server under test.** If you need to measure token counts for prompt
  calibration (see Task 6), do it against a throwaway server instance, or
  do it and then restart the server before running S1/S2 for real —
  measuring `usage.prompt_tokens` on the server you are about to
  benchmark contaminates its cache for every prompt you probed.
- **S1 reports cold (first repeat) and warm (rest) TTFT separately** —
  `ttft_first_ms` (the first, i.e. cold, repeat of each prompt on a
  freshly (re)started server) and `ttft_rest_mean_ms` (mean of the
  remaining repeats, which hit the cache from the first). `ttft_ms_mean`
  is kept for backward compatibility but averages cold into warm and
  should not be used to judge cache behavior.
- **S3 is warm-cache by design** for every engine: it repeats the same
  prompt at increasing concurrency to measure aggregate throughput, not
  cold-start latency, so its numbers are unaffected by this rule.
