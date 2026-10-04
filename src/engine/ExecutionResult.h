// src/engine/ExecutionResult.h
// Phase 11: Backend-independent execution result.
//
// Returned by IExecutionEngine::execute(). Contains:
//  - success/failure flag
//  - generated text
//  - error message
//  - execution telemetry compatible with the existing InferenceTelemetry
//
// Extensible: future phases may add embedding vectors, image output, etc.
// without changing the base contract.
#pragma once

#include <cstdint>
#include <string>

namespace agr {

/// Engine-reported backend type used in telemetry.
/// Deliberately independent of llama.cpp enumerations.
enum class ExecutionBackendType {
    UNKNOWN,
    CPU,
    VULKAN,
    CUDA,
    NPU,
};

inline const char* toString(ExecutionBackendType t) {
    switch (t) {
        case ExecutionBackendType::CPU:    return "CPU";
        case ExecutionBackendType::VULKAN: return "Vulkan";
        case ExecutionBackendType::CUDA:   return "CUDA";
        case ExecutionBackendType::NPU:    return "NPU";
        default:                           return "Unknown";
    }
}

/// Telemetry produced by one execution run.
/// Mirrors the existing InferenceTelemetry so that the existing
/// Profiler/AutoTuner can consume it without modification.
struct ExecutionTelemetry {
    // Backend identification
    ExecutionBackendType backend_type   = ExecutionBackendType::UNKNOWN;
    std::string          backend_name;    ///< e.g. "LlamaCppBackend"
    std::string          device_name;     ///< e.g. "Intel Iris Xe"
    bool                 gpu_used       = false;
    int                  gpu_layers     = 0;

    // Token counts
    int prompt_tokens  = 0;
    int output_tokens  = 0;

    // Timing (milliseconds)
    double model_load_ms   = 0.0;  ///< 0 if model was already resident
    double prompt_eval_ms  = 0.0;  ///< Time-to-first-token proxy
    double generation_ms   = 0.0;  ///< Token-generation phase
    double total_ms        = 0.0;  ///< prompt_eval + generation

    // Derived
    double ttft_ms        = 0.0;  ///< == prompt_eval_ms
    double tokens_per_sec = 0.0;  ///< output_tokens / (generation_ms / 1000)

    // Scheduler metadata
    std::string scheduler_reason;  ///< Why this backend was chosen
    int         fallbacks = 0;     ///< Number of backend fallbacks this run
};

/// Result returned from IExecutionEngine::execute().
struct ExecutionResult {
    bool        success = false;
    std::string output;   ///< Generated text (UTF-8)
    std::string error;    ///< Non-empty if success == false

    ExecutionTelemetry telemetry;
};

} // namespace agr
