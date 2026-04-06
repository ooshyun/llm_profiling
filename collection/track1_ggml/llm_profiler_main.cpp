// llm_profiler_main.cpp — Per-layer profiling tool using llama.cpp cb_eval
// Outputs JSONL with per-tensor timing, op type, dimensions, and FLOPS.
// Usage: ./llm-profiler -m model.gguf -p "prompt text" -n 64 --profiler-output profile.jsonl

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

// ---------- Profiler State ----------

#define PROFILER_NAME_LEN 256

struct profiler_record {
    char     name[PROFILER_NAME_LEN];
    int      op;
    int64_t  ne[4];
    size_t   bytes;
    int64_t  flops;
    double   start_us;
    double   end_us;
};

struct profiler_state {
    FILE *   output_file;
    int      phase;          // 0=prefill, 1=decode
    int      token_idx;      // current decode token index
    profiler_record current;
    int64_t  total_records;
};

static int64_t compute_flops(const struct ggml_tensor * t) {
    if (t->op == GGML_OP_MUL_MAT && t->src[0] && t->src[1]) {
        int64_t M = t->ne[1];
        int64_t N = t->ne[0];
        int64_t K = t->src[0]->ne[0];
        return 2LL * M * N * K;
    }
    return 0;
}

static bool profiler_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = (profiler_state *)user_data;

    if (ask) {
        strncpy(state->current.name, t->name, PROFILER_NAME_LEN - 1);
        state->current.name[PROFILER_NAME_LEN - 1] = '\0';
        state->current.op = (int)t->op;
        for (int i = 0; i < 4; i++) {
            state->current.ne[i] = t->ne[i];
        }
        state->current.bytes = ggml_nbytes(t);
        state->current.flops = compute_flops(t);
        state->current.start_us = (double)ggml_time_us();
        return true;
    } else {
        state->current.end_us = (double)ggml_time_us();
        double latency = state->current.end_us - state->current.start_us;

        if (state->output_file) {
            fprintf(state->output_file,
                "{\"name\":\"%s\",\"op\":%d,"
                "\"ne\":[%lld,%lld,%lld,%lld],"
                "\"bytes\":%zu,\"flops\":%lld,"
                "\"latency_us\":%.1f,\"phase\":%d,\"token_idx\":%d}\n",
                state->current.name,
                state->current.op,
                (long long)state->current.ne[0], (long long)state->current.ne[1],
                (long long)state->current.ne[2], (long long)state->current.ne[3],
                state->current.bytes,
                (long long)state->current.flops,
                latency,
                state->phase,
                state->token_idx
            );
        }
        state->total_records++;
        return true;
    }
}

// ---------- Main ----------

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    // Parse --profiler-output from argv before common_params_parse
    std::string profiler_output = "profile_output.jsonl";
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--profiler-output") == 0) {
            profiler_output = argv[i + 1];
            // Remove these args so common_params_parse doesn't see them
            for (int j = i; j < argc - 2; j++) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            break;
        }
    }

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        LOG_ERR("Failed to parse params\n");
        return 1;
    }

    // Setup profiler state
    profiler_state state = {};
    state.output_file = fopen(profiler_output.c_str(), "w");
    if (!state.output_file) {
        LOG_ERR("Failed to open profiler output: %s\n", profiler_output.c_str());
        return 1;
    }
    state.phase = 0;
    state.token_idx = 0;
    state.total_records = 0;

    // Register callback
    params.cb_eval = profiler_callback;
    params.cb_eval_user_data = &state;
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    // Init model and context
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model || !ctx) {
        LOG_ERR("Failed to init model/context\n");
        fclose(state.output_file);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    LOG_INF("Profiler output: %s\n", profiler_output.c_str());
    LOG_INF("System info: %s\n", common_params_get_system_info(params).c_str());

    // Tokenize prompt
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    LOG_INF("Prompt tokens: %zu\n", tokens.size());

    if (tokens.empty()) {
        LOG_ERR("No input tokens\n");
        fclose(state.output_file);
        return 1;
    }

    // Phase 0: Prefill
    state.phase = 0;
    state.token_idx = 0;
    LOG_INF("=== PREFILL (%zu tokens) ===\n", tokens.size());
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("Prefill failed\n");
        fclose(state.output_file);
        return 1;
    }

    // Phase 1: Decode (generate tokens)
    state.phase = 1;
    int n_gen = params.n_predict > 0 ? params.n_predict : 32;
    LOG_INF("=== DECODE (%d tokens) ===\n", n_gen);

    llama_token eos = llama_vocab_eos(vocab);
    for (int i = 0; i < n_gen; i++) {
        state.token_idx = i;

        // Greedy sampling from logits
        auto * logits = llama_get_logits_ith(ctx, -1);
        int n_vocab = llama_vocab_n_tokens(vocab);

        llama_token best = 0;
        float best_logit = logits[0];
        for (int j = 1; j < n_vocab; j++) {
            if (logits[j] > best_logit) {
                best_logit = logits[j];
                best = j;
            }
        }

        if (best == eos) {
            LOG_INF("EOS at token %d\n", i);
            break;
        }

        // Decode next token
        if (llama_decode(ctx, llama_batch_get_one(&best, 1))) {
            LOG_ERR("Decode failed at token %d\n", i);
            break;
        }
    }

    // Summary
    fflush(state.output_file);
    fclose(state.output_file);

    LOG_INF("\n=== PROFILING COMPLETE ===\n");
    LOG_INF("Total records: %lld\n", (long long)state.total_records);
    LOG_INF("Output: %s\n", profiler_output.c_str());

    llama_perf_context_print(ctx);
    llama_backend_free();

    return 0;
}
