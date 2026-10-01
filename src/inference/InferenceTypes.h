// src/inference/InferenceTypes.h
// Phase 7: Engine-agnostic types for inference requests and results.
// No llama.cpp types exposed here.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace agr {

/// Options controlling token generation.
struct GenerationOptions {
    int      max_tokens  = 256;
    float    temperature = 0.7f;
    uint32_t seed        = 42;
    int      n_ctx       = 2048;
    int      n_batch     = 512;
    int      n_threads   = 4;
};

/// Lifecycle state of a loaded model.
enum class ModelState {
    DISCOVERED,  ///< Path known, nothing loaded.
    LOADING,     ///< llama.cpp loading in progress.
    LOADED,      ///< Model weights are in memory.
    READY,       ///< Context created, ready for inference.
    EXECUTING,   ///< Inference in progress.
    FAILED,      ///< A non-recoverable error occurred.
    RELEASED,    ///< All resources freed.
};

inline const char* toString(ModelState s) {
    switch (s) {
        case ModelState::DISCOVERED: return "DISCOVERED";
        case ModelState::LOADING:    return "LOADING";
        case ModelState::LOADED:     return "LOADED";
        case ModelState::READY:      return "READY";
        case ModelState::EXECUTING:  return "EXECUTING";
        case ModelState::FAILED:     return "FAILED";
        case ModelState::RELEASED:   return "RELEASED";
        default:                     return "UNKNOWN";
    }
}

/// Metadata exposed by a loaded model. No llama.cpp types.
struct ModelInfo {
    std::string path;
    std::string architecture;   ///< e.g. "llama", "qwen2"
    std::string quantization;   ///< e.g. "Q4_K_M"
    int64_t     n_params  = 0;  ///< Parameter count (0 if not queryable)
    int32_t     n_ctx_max = 0;  ///< Maximum supported context length
    int32_t     n_layers  = 0;  ///< Number of transformer layers
    size_t      size_bytes = 0; ///< Approximate file size in bytes
};

/// A single inference request.
struct InferenceRequest {
    std::string       prompt;
    GenerationOptions options;
    bool              prefer_gpu    = true;
    bool              allow_cpu_fallback = true;
};

/// Telemetry for one completed inference.
struct InferenceTelemetry {
    std::string backend;       ///< "Vulkan" | "CPU" | "CUDA"
    std::string device_name;   ///< e.g. "Intel Iris Xe"
    bool        vulkan_used   = false;
    int         gpu_layers    = 0;

    int    prompt_tokens  = 0;
    int    output_tokens  = 0;

    /// Timing in milliseconds.
    double model_load_ms   = 0.0; ///< 0 if model was already loaded
    double prompt_eval_ms  = 0.0; ///< Prompt processing (TTFT proxy)
    double generation_ms   = 0.0; ///< Token generation
    double total_ms        = 0.0; ///< prompt_eval + generation

    double ttft_ms         = 0.0; ///< Time to first token ≈ prompt_eval_ms
    double tokens_per_sec  = 0.0; ///< output_tokens / (generation_ms / 1000)

    /// Scheduler decision info.
    std::string scheduler_reason; ///< Why this backend was chosen
    int         fallbacks = 0;    ///< Number of backend fallbacks
};

/// Result of one inference.
struct InferenceResult {
    bool        success = false;
    std::string text;            ///< Generated text
    std::string error;           ///< Populated on failure
    InferenceTelemetry telemetry;
};

/// Callback type for streaming token generation.
/// Called with each piece of text as it is generated.
using TokenCallback = std::function<void(const std::string& piece)>;

} // namespace agr
