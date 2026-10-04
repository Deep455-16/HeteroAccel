// src/engine/ExecutionContext.h
// Phase 11: Backend-independent execution context descriptor.
//
// An ExecutionContext carries the parameters that govern a single
// inference session.  HeteroAccel's Execution Policy Engine (Phase 9)
// fills this in; the execution engine reads from it.
//
// Phase 13 (Execution Planner) will extend this with device-placement
// decisions. Phase 11 only defines the structural contract.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace agr {

/// Requested compute backend for an execution session.
/// "AUTO" means HeteroAccel chooses; the choice is not surfaced here
/// in Phase 11 (Phase 13 will handle planning).
enum class ExecutionBackendHint {
    AUTO,    ///< Let HeteroAccel decide (default)
    CPU,     ///< Prefer CPU path
    VULKAN,  ///< Prefer Vulkan GPU path
    CUDA,    ///< Prefer CUDA GPU path
};

/// Per-session execution parameters passed to IExecutionEngine::execute().
///
/// CONCURRENCY SEMANTICS (Phase 11):
///   - A single IExecutionEngine instance serialises concurrent calls
///     internally when the underlying backend (e.g. llama.cpp) is not
///     thread-safe. Independent engine instances may run concurrently.
///   - cancel_flag: caller-owned atomic; set to true to request abort.
///   - request_id: used by WorkloadRegistry for cancellation routing.
struct ExecutionContext {
    // ---------------------------------------------------------------
    // Prompt / input
    // ---------------------------------------------------------------
    std::string prompt;         ///< Input text prompt

    // ---------------------------------------------------------------
    // Generation constraints
    // ---------------------------------------------------------------
    int      max_tokens   = 256;   ///< Maximum tokens to generate
    float    temperature  = 0.7f;  ///< Sampling temperature
    uint32_t seed         = 42;    ///< RNG seed for reproducibility
    int      n_ctx        = 2048;  ///< Context window size (tokens)
    int      n_batch      = 512;   ///< Logical batch size for prompt eval
    int      n_threads    = 0;     ///< Worker threads (0 = auto)

    // ---------------------------------------------------------------
    // Resource constraints / hints
    // ---------------------------------------------------------------
    int  n_gpu_layers        = 99;   ///< Layers to offload to GPU (99 = as many as fit)
    bool cpu_only            = false; ///< Force CPU-only execution
    ExecutionBackendHint backend_hint = ExecutionBackendHint::AUTO;

    // ---------------------------------------------------------------
    // Scheduling / workload metadata
    // ---------------------------------------------------------------
    uint64_t request_id      = 0;    ///< Unique ID used by WorkloadRegistry
    int      priority        = 0;    ///< 0=BACKGROUND, 1=NORMAL, 2=HIGH, 3=CRITICAL
    int      workload_class  = 0;    ///< Maps to WorkloadClass enum

    // ---------------------------------------------------------------
    // Cancellation
    // ---------------------------------------------------------------
    /// Caller-owned flag; safe to poll from inference loop.
    /// Engine MUST honour this within a bounded number of tokens.
    std::atomic<bool>* cancel_flag = nullptr;

    // ---------------------------------------------------------------
    // Streaming
    // ---------------------------------------------------------------
    bool streaming = false;  ///< If true, engine emits per-token callbacks
};

} // namespace agr
