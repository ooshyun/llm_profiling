# serving_bench summary

## S1 — single-user chat

| engine | engine_version | runtime | model | fmt | prompt_id | n | prompt_tokens | ttft_ms_mean | ttft_first_ms | ttft_rest_mean_ms | tpot_ms_mean | gen_tok_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 6fdd0ac | cu12.2 | qwen3-8b | gguf-q4km | code | 3 | 210 | 485.0 | 998.4 | 228.4 | 132.1 | 7.6 |
| llama.cpp | 6fdd0ac | cu12.2 | qwen3-8b | gguf-q4km | long | 3 | 884 | 1394.0 | 3821.6 | 180.3 | 131.2 | 7.6 |
| llama.cpp | 6fdd0ac | cu12.2 | qwen3-8b | gguf-q4km | short | 3 | 32 | 217.8 | 301.6 | 175.8 | 130.0 | 7.7 |
| llama.cpp | 6fdd0ac | cu12.2 | qwen3.5-35b-a3b | gguf-q4km | code | 3 | 225 | 949.9 | 2259.1 | 295.3 | 92.4 | 10.8 |
| llama.cpp | 6fdd0ac | cu12.2 | qwen3.5-35b-a3b | gguf-q4km | long | 3 | 884 | 2232.0 | 6195.3 | 250.4 | 91.5 | 10.9 |
| llama.cpp | 6fdd0ac | cu12.2 | qwen3.5-35b-a3b | gguf-q4km | short | 3 | 32 | 471.9 | 840.3 | 287.7 | 91.3 | 11.0 |

## S2 — agent loop (4k shared system prompt)

| engine | engine_version | runtime | model | fmt | turns | ttft_turn1_ms | ttft_rest_mean_ms | prefix_speedup | tpot_ms_mean |
|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 6fdd0ac | cu12.2 | qwen3-8b | gguf-q4km | 20 | 18097.5 | 407.7 | 44.4 | 136.0 |
| llama.cpp | 6fdd0ac | cu12.2 | qwen3.5-35b-a3b | gguf-q4km | 20 | 29286.3 | 977.0 | 30.0 | 92.1 |

## S3 — concurrency

| engine | engine_version | runtime | model | fmt | concurrency | n | errors | agg_tok_s | ttft_p50_ms | ttft_p95_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 6fdd0ac | cu12.2 | qwen3-8b | gguf-q4km | 1 | 4 | 0 | 7.5 | 224.9 | 906.5 |
| llama.cpp | 6fdd0ac | cu12.2 | qwen3-8b | gguf-q4km | 2 | 8 | 0 | 13.1 | 263.2 | 448.6 |
| llama.cpp | 6fdd0ac | cu12.2 | qwen3-8b | gguf-q4km | 4 | 16 | 0 | 16.0 | 364.5 | 551.2 |
| llama.cpp | 6fdd0ac | cu12.2 | qwen3-8b | gguf-q4km | 8 | 32 | 0 | 17.4 | 643.4 | 2669.4 |
| llama.cpp | 6fdd0ac | cu12.2 | qwen3.5-35b-a3b | gguf-q4km | 1 | 4 | 0 | 10.5 | 307.0 | 2070.6 |
| llama.cpp | 6fdd0ac | cu12.2 | qwen3.5-35b-a3b | gguf-q4km | 2 | 8 | 0 | 21.7 | 434.1 | 562.8 |
| llama.cpp | 6fdd0ac | cu12.2 | qwen3.5-35b-a3b | gguf-q4km | 4 | 16 | 0 | 25.8 | 882.1 | 2973.9 |
| llama.cpp | 6fdd0ac | cu12.2 | qwen3.5-35b-a3b | gguf-q4km | 8 | 32 | 0 | 30.0 | 1362.2 | 1906.1 |
