/**
 * ggml_profiler.h
 *
 * C callback profiler for llama.cpp's cb_eval hook.
 * Writes per-op JSONL records to a file for post-processing by the Python
 * wrapper (profiler_wrapper.py).
 *
 * NOTE: This header references ggml.h types.  It is compiled on-device where
 *       llama.cpp (and therefore ggml.h) is present.  Do not attempt to build
 *       it without those headers.
 */

#ifndef GGML_PROFILER_H
#define GGML_PROFILER_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* ggml.h is available when compiled inside a llama.cpp build tree. */
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

#define PROFILER_MAX_RECORDS  65536
#define PROFILER_NAME_LEN     128

/* Inference phase identifiers – kept in sync with profiler_wrapper.py */
#define PROFILER_PHASE_PREFILL  0
#define PROFILER_PHASE_DECODE   1

/* ------------------------------------------------------------------ */
/* Structures                                                          */
/* ------------------------------------------------------------------ */

/**
 * profiler_record_t
 *
 * One record per ggml op invocation.
 * ne[4] mirrors the shape of the *output* tensor (ggml convention: ne[0] is
 * the fastest-varying dimension).
 */
typedef struct {
    char     name[PROFILER_NAME_LEN]; /* tensor / op name (NUL-terminated)  */
    int      op;                      /* ggml_op enum value                  */
    int64_t  ne[4];                   /* output tensor shape                 */
    size_t   bytes;                   /* output tensor size in bytes         */
    double   start_us;                /* wall-clock start (microseconds)     */
    double   end_us;                  /* wall-clock end   (microseconds)     */
    int64_t  flops;                   /* estimated FLOPs (0 if not computed) */
} profiler_record_t;

/**
 * profiler_state_t
 *
 * Singleton state object.  Initialise once with profiler_init(); pass the
 * pointer to profiler_callback() as the userdata argument.
 */
typedef struct {
    profiler_record_t  records[PROFILER_MAX_RECORDS];
    int                n_records;          /* number of completed records    */
    profiler_record_t  current;            /* record being built (pre-eval)  */
    int                has_current;        /* 1 while current is in-flight   */

    char               output_path[512];   /* path to the JSONL output file  */
    FILE              *output_file;        /* open file handle (NULL = err)  */

    int                phase;             /* PROFILER_PHASE_PREFILL / DECODE */
} profiler_state_t;

/* ------------------------------------------------------------------ */
/* API                                                                 */
/* ------------------------------------------------------------------ */

/**
 * profiler_init
 *
 * Allocates (or resets) the profiler state and opens output_path for writing.
 * Must be called before attaching profiler_callback to llama_context.
 *
 * @param state        Caller-allocated profiler_state_t (stack or static).
 * @param output_path  Path of the JSONL file to create / truncate.
 * @return             0 on success, non-zero on error (e.g. file open failure).
 */
int profiler_init(profiler_state_t *state, const char *output_path);

/**
 * profiler_callback
 *
 * The cb_eval callback to register with llama_context.
 * Signature matches llama_eval_callback:
 *   void cb(struct ggml_tensor *t, bool ask, void *user_data)
 *
 * When ask == true  : called before the op executes – record start time.
 * When ask == false : called after  the op executes – record end time, flush.
 *
 * @param t         The tensor whose op is about to execute / just executed.
 * @param ask       true = pre-eval query, false = post-eval notification.
 * @param user_data Pointer to profiler_state_t.
 * @return          true (always – tells llama.cpp to keep the tensor in graph).
 */
bool profiler_callback(struct ggml_tensor *t, bool ask, void *user_data);

/**
 * profiler_set_phase
 *
 * Switch between prefill and decode phases.  Call this whenever the inference
 * stage changes (e.g. between prompt ingestion and autoregressive generation).
 *
 * @param state  The active profiler state.
 * @param phase  PROFILER_PHASE_PREFILL or PROFILER_PHASE_DECODE.
 */
void profiler_set_phase(profiler_state_t *state, int phase);

/**
 * profiler_flush
 *
 * Write all buffered records to the JSONL file and reset n_records to 0.
 * Called automatically by profiler_callback after each op but may also be
 * called explicitly to force a flush.
 *
 * @param state  The active profiler state.
 */
void profiler_flush(profiler_state_t *state);

/**
 * profiler_free
 *
 * Close the output file and zero out the state.
 * Always call this when profiling is complete.
 *
 * @param state  The active profiler state.
 */
void profiler_free(profiler_state_t *state);

#ifdef __cplusplus
}
#endif

#endif /* GGML_PROFILER_H */
