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
      --model qwen3-8b --fmt gguf-q4km --base-url http://127.0.0.1:8080 \
      --scenario S1 --config scenarios.yaml \
      --out results/llamacpp_cu122_8b_S1.jsonl

    # vLLM/SGLang additionally need:
    #   --api-model <HF id> --chat-template-kwargs '{"enable_thinking": false}'

## Summarize (Mac or Orin)

    python3 bench.py summarize results/serving_bench \
      --out results/serving_bench/summary.md

Warm-up (1 request, excluded) runs automatically and aborts if the reply
contains `<think>` — fix the engine's reasoning flags, don't benchmark.
