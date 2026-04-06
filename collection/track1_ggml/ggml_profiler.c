/**
 * ggml_profiler.c
 *
 * Implementation of the ggml callback profiler.
 *
 * NOTE: This file must be compiled within a llama.cpp build tree where
 *       ggml.h (and the symbols it references) are available.  It will NOT
 *       compile standalone.
 */

#include "ggml_profiler.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Internal helpers                                                    */
/* ------------------------------------------------------------------ */

/**
 * now_us
 *
 * Returns the current wall-clock time in microseconds.
 * Uses clock_gettime(CLOCK_MONOTONIC) where available; falls back to
 * gettimeofday on platforms that lack CLOCK_MONOTONIC.
 */
static double now_us(void) {
#if defined(_POSIX_TIMERS) && _POSIX_TIMERS > 0
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
#else
    /* Fallback: use time() with microsecond precision lost */
    return (double)time(NULL) * 1e6;
#endif
}

/**
 * compute_flops
 *
 * Estimates the number of floating-point operations for supported op types.
 *
 * For GGML_OP_MUL_MAT with output shape [N, M, ...] and src1 shape [K, M]:
 *   flops = 2 * M * N * K   (multiply-accumulate counted as 2 ops)
 *
 * Returns 0 for op types where estimation is not implemented.
 *
 * @param t  The output tensor of the op.
 * @return   Estimated FLOPs, or 0.
 */
static int64_t compute_flops(const struct ggml_tensor *t) {
    if (t->op != GGML_OP_MUL_MAT) {
        return 0;
    }

    /*
     * ggml MUL_MAT: result = src0 x src1
     *   src0 shape: [K, N, ...]   (weight matrix, column-major in ggml terms)
     *   src1 shape: [K, M, ...]   (input activations)
     *   out  shape: [N, M, ...]
     *
     * ne[0] = N (output rows / hidden dim)
     * ne[1] = M (batch / sequence dimension)
     * src0->ne[0] = K (inner / contracted dimension)
     */
    if (t->src[0] == NULL) {
        return 0;
    }

    const int64_t N = t->ne[0];
    const int64_t M = t->ne[1];
    const int64_t K = t->src[0]->ne[0];

    return (int64_t)2 * M * N * K;
}

/**
 * write_record_jsonl
 *
 * Serialises a completed profiler_record_t as a single JSON object followed
 * by a newline and writes it to the output file.
 *
 * JSON fields:
 *   name        string
 *   op          integer (ggml_op enum)
 *   ne          array of 4 integers  [ne0, ne1, ne2, ne3]
 *   bytes       integer
 *   flops       integer
 *   latency_us  float
 *   phase       integer  (0 = prefill, 1 = decode)
 */
static void write_record_jsonl(FILE *f, const profiler_record_t *rec, int phase) {
    if (f == NULL) {
        return;
    }

    const double latency_us = rec->end_us - rec->start_us;

    /* Escape the name – conservative approach: replace '"' and '\' */
    char safe_name[PROFILER_NAME_LEN * 2];
    size_t si = 0;
    for (size_t ni = 0; ni < PROFILER_NAME_LEN && rec->name[ni] != '\0'; ++ni) {
        char c = rec->name[ni];
        if (c == '"' || c == '\\') {
            safe_name[si++] = '\\';
        }
        safe_name[si++] = c;
    }
    safe_name[si] = '\0';

    fprintf(f,
        "{\"name\":\"%s\","
        "\"op\":%d,"
        "\"ne\":[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "],"
        "\"bytes\":%zu,"
        "\"flops\":%" PRId64 ","
        "\"latency_us\":%.3f,"
        "\"phase\":%d}\n",
        safe_name,
        rec->op,
        rec->ne[0], rec->ne[1], rec->ne[2], rec->ne[3],
        rec->bytes,
        rec->flops,
        latency_us,
        phase);
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

int profiler_init(profiler_state_t *state, const char *output_path) {
    if (state == NULL || output_path == NULL) {
        return -1;
    }

    memset(state, 0, sizeof(profiler_state_t));

    strncpy(state->output_path, output_path, sizeof(state->output_path) - 1);
    state->output_path[sizeof(state->output_path) - 1] = '\0';

    state->output_file = fopen(output_path, "w");
    if (state->output_file == NULL) {
        fprintf(stderr, "[ggml_profiler] ERROR: cannot open output file: %s\n",
                output_path);
        return -1;
    }

    state->phase       = PROFILER_PHASE_PREFILL;
    state->n_records   = 0;
    state->has_current = 0;

    return 0;
}

bool profiler_callback(struct ggml_tensor *t, bool ask, void *user_data) {
    profiler_state_t *state = (profiler_state_t *)user_data;
    if (state == NULL || t == NULL) {
        return true;
    }

    if (ask) {
        /* ---- Pre-eval: record start time and tensor metadata ---- */
        memset(&state->current, 0, sizeof(profiler_record_t));

        const char *raw_name = t->name;
        if (raw_name == NULL || raw_name[0] == '\0') {
            /* Unnamed tensor: use op number as fallback name */
            snprintf(state->current.name, PROFILER_NAME_LEN,
                     "op_%d", (int)t->op);
        } else {
            strncpy(state->current.name, raw_name, PROFILER_NAME_LEN - 1);
            state->current.name[PROFILER_NAME_LEN - 1] = '\0';
        }

        state->current.op      = (int)t->op;
        state->current.ne[0]   = t->ne[0];
        state->current.ne[1]   = t->ne[1];
        state->current.ne[2]   = t->ne[2];
        state->current.ne[3]   = t->ne[3];
        state->current.bytes   = ggml_nbytes(t);
        state->current.flops   = compute_flops(t);
        state->current.start_us = now_us();

        state->has_current = 1;

    } else {
        /* ---- Post-eval: record end time, write JSONL ---- */
        if (!state->has_current) {
            return true;  /* spurious post-eval with no matching pre-eval */
        }

        state->current.end_us = now_us();
        state->has_current = 0;

        /* Buffer the record */
        if (state->n_records < PROFILER_MAX_RECORDS) {
            state->records[state->n_records] = state->current;
            state->n_records++;
        }

        /* Write immediately to avoid large in-memory buffers */
        write_record_jsonl(state->output_file, &state->current, state->phase);
    }

    return true;
}

void profiler_set_phase(profiler_state_t *state, int phase) {
    if (state == NULL) {
        return;
    }
    state->phase = phase;
}

void profiler_flush(profiler_state_t *state) {
    if (state == NULL || state->output_file == NULL) {
        return;
    }
    fflush(state->output_file);
    state->n_records = 0;
}

void profiler_free(profiler_state_t *state) {
    if (state == NULL) {
        return;
    }
    if (state->output_file != NULL) {
        fflush(state->output_file);
        fclose(state->output_file);
        state->output_file = NULL;
    }
    memset(state, 0, sizeof(profiler_state_t));
}
