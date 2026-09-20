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
| llama.cpp | 6fdd0ac | cu12.6 | qwen3-8b | gguf-q4km | code | 3 | 210 | 482.6 | 1007.2 | 220.3 | 132.0 | 7.6 |
| llama.cpp | 6fdd0ac | cu12.6 | qwen3-8b | gguf-q4km | long | 3 | 884 | 1369.6 | 3745.8 | 181.5 | 130.9 | 7.6 |
| llama.cpp | 6fdd0ac | cu12.6 | qwen3-8b | gguf-q4km | short | 3 | 32 | 211.5 | 292.6 | 171.0 | 129.9 | 7.7 |
| llama.cpp | 6fdd0ac | cu12.6 | qwen3.5-35b-a3b | gguf-q4km | code | 3 | 225 | 923.2 | 2239.9 | 264.8 | 93.6 | 10.7 |
| llama.cpp | 6fdd0ac | cu12.6 | qwen3.5-35b-a3b | gguf-q4km | long | 3 | 884 | 2185.9 | 6102.7 | 227.5 | 92.8 | 10.8 |
| llama.cpp | 6fdd0ac | cu12.6 | qwen3.5-35b-a3b | gguf-q4km | short | 3 | 32 | 419.7 | 762.1 | 248.5 | 92.5 | 10.8 |
| vllm | 0.22.0 | cu12.6 | qwen3-8b | bf16 | code | 3 | 210 | 394.1 | 805.7 | 188.4 | 161.9 | 6.2 |
| vllm | 0.22.0 | cu12.6 | qwen3-8b | bf16 | long | 3 | 884 | 993.6 | 2296.8 | 342.1 | 160.7 | 6.2 |
| vllm | 0.22.0 | cu12.6 | qwen3-8b | bf16 | short | 3 | 32 | 421.5 | 579.9 | 342.2 | 158.7 | 6.3 |
| vllm | 0.22.0 | cu12.6 | qwen3.5-35b-a3b | gptq-int4 | code | 3 | 225 | 5983.5 | 6018.4 | 5966.1 | 73.0 | 13.7 |
| vllm | 0.22.0 | cu12.6 | qwen3.5-35b-a3b | gptq-int4 | long | 3 | 884 | 8816.5 | 8782.2 | 8833.6 | 73.4 | 13.6 |
| vllm | 0.22.0 | cu12.6 | qwen3.5-35b-a3b | gptq-int4 | short | 3 | 32 | 1051.2 | 1787.9 | 682.9 | 70.0 | 14.3 |

## S2 — agent loop (4k shared system prompt)

| engine | engine_version | runtime | model | fmt | turns | ttft_turn1_ms | ttft_rest_mean_ms | prefix_speedup | tpot_ms_mean |
|---|---|---|---|---|---|---|---|---|---|
| llama.cpp | 6fdd0ac | cu12.2 | qwen3-8b | gguf-q4km | 20 | 18097.5 | 407.7 | 44.4 | 136.0 |
| llama.cpp | 6fdd0ac | cu12.2 | qwen3.5-35b-a3b | gguf-q4km | 20 | 29286.3 | 977.0 | 30.0 | 92.1 |
| llama.cpp | 6fdd0ac | cu12.6 | qwen3-8b | gguf-q4km | 20 | 17732.8 | 400.3 | 44.3 | 133.1 |
| llama.cpp | 6fdd0ac | cu12.6 | qwen3.5-35b-a3b | gguf-q4km | 20 | 28843.3 | 919.5 | 31.4 | 93.2 |
| vllm | 0.22.0 | cu12.6 | qwen3-8b | bf16 | 20 | 10877.2 | 390.0 | 27.9 | 167.6 |
| vllm | 0.22.0 | cu12.6 | qwen3.5-35b-a3b | gptq-int4 | 20 | 29049.7 | 29245.0 | 1.0 | 72.4 |

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
| llama.cpp | 6fdd0ac | cu12.6 | qwen3-8b | gguf-q4km | 1 | 4 | 0 | 7.5 | 216.8 | 884.2 |
| llama.cpp | 6fdd0ac | cu12.6 | qwen3-8b | gguf-q4km | 2 | 8 | 0 | 13.1 | 247.0 | 441.8 |
| llama.cpp | 6fdd0ac | cu12.6 | qwen3-8b | gguf-q4km | 4 | 16 | 0 | 16.2 | 366.2 | 581.3 |
| llama.cpp | 6fdd0ac | cu12.6 | qwen3-8b | gguf-q4km | 8 | 32 | 0 | 17.5 | 642.7 | 2604.3 |
| llama.cpp | 6fdd0ac | cu12.6 | qwen3.5-35b-a3b | gguf-q4km | 1 | 4 | 0 | 10.4 | 261.5 | 2006.8 |
| llama.cpp | 6fdd0ac | cu12.6 | qwen3.5-35b-a3b | gguf-q4km | 2 | 8 | 0 | 21.5 | 452.1 | 567.5 |
| llama.cpp | 6fdd0ac | cu12.6 | qwen3.5-35b-a3b | gguf-q4km | 4 | 16 | 0 | 26.1 | 528.2 | 767.2 |
| llama.cpp | 6fdd0ac | cu12.6 | qwen3.5-35b-a3b | gguf-q4km | 8 | 32 | 0 | 29.5 | 1037.0 | 1337.3 |
| vllm | 0.22.0 | cu12.6 | qwen3-8b | bf16 | 1 | 4 | 0 | 6.2 | 184.7 | 255.6 |
| vllm | 0.22.0 | cu12.6 | qwen3-8b | bf16 | 2 | 8 | 0 | 13.3 | 328.0 | 338.5 |
| vllm | 0.22.0 | cu12.6 | qwen3-8b | bf16 | 4 | 16 | 0 | 25.3 | 491.5 | 496.4 |
| vllm | 0.22.0 | cu12.6 | qwen3-8b | bf16 | 8 | 32 | 0 | 48.9 | 492.7 | 532.9 |
| vllm | 0.22.0 | cu12.6 | qwen3.5-35b-a3b | gptq-int4 | 1 | 4 | 0 | 10.5 | 6266.8 | 6281.4 |
| vllm | 0.22.0 | cu12.6 | qwen3.5-35b-a3b | gptq-int4 | 2 | 8 | 0 | 13.5 | 9413.6 | 12554.7 |
| vllm | 0.22.0 | cu12.6 | qwen3.5-35b-a3b | gptq-int4 | 4 | 16 | 0 | 23.1 | 13477.6 | 13529.7 |
| vllm | 0.22.0 | cu12.6 | qwen3.5-35b-a3b | gptq-int4 | 8 | 32 | 0 | 35.4 | 16930.8 | 17000.7 |
