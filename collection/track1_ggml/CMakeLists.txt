cmake_minimum_required(VERSION 3.14)
project(ggml_profiler C)

# ---------------------------------------------------------------------------
# Required: path to a llama.cpp source tree that contains ggml.h
# Usage:  cmake -DLLAMA_CPP_DIR=/path/to/llama.cpp ..
# ---------------------------------------------------------------------------
if(NOT DEFINED LLAMA_CPP_DIR)
    message(FATAL_ERROR
        "LLAMA_CPP_DIR is not set.\n"
        "Re-run cmake with:  -DLLAMA_CPP_DIR=/path/to/llama.cpp")
endif()

# Resolve to an absolute path and verify the header exists
get_filename_component(LLAMA_CPP_DIR "${LLAMA_CPP_DIR}" ABSOLUTE)
if(NOT EXISTS "${LLAMA_CPP_DIR}/ggml.h")
    message(FATAL_ERROR
        "ggml.h not found in LLAMA_CPP_DIR=${LLAMA_CPP_DIR}\n"
        "Make sure LLAMA_CPP_DIR points to the llama.cpp root.")
endif()

# ---------------------------------------------------------------------------
# Static library: ggml_profiler
# ---------------------------------------------------------------------------
add_library(ggml_profiler STATIC
    ggml_profiler.c
)

target_include_directories(ggml_profiler
    PUBLIC
        # This directory (for ggml_profiler.h consumers)
        "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE
        # llama.cpp root (for ggml.h)
        "${LLAMA_CPP_DIR}"
)

target_compile_features(ggml_profiler PRIVATE c_std_99)

target_compile_options(ggml_profiler PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-g -O0>
)

# POSIX_TIMERS feature-test macro so clock_gettime is available
target_compile_definitions(ggml_profiler PRIVATE
    _POSIX_C_SOURCE=200809L
)

# ---------------------------------------------------------------------------
# Install rules (optional – useful when embedding in a larger build)
# ---------------------------------------------------------------------------
install(TARGETS ggml_profiler
    ARCHIVE DESTINATION lib
)
install(FILES ggml_profiler.h
    DESTINATION include
)
