#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace agr {

// Result of one LLM inference run, returned by LlamaCppEngine::infer().
struct LlmResult {
    bool        success      = false;
    std::string output;               // Generated text
    std::string error;                // Populated on failure

    // --- Backend / device ---
    std::string backend;              // "CUDA" | "Vulkan" | "CPU"
    std::string device_name;          // e.g. "Intel(R) Iris(R) Xe Graphics" (Vulkan only)
    int         gpu_layers_requested = 0;
    int         gpu_layers_actual    = 0;  // layers llama.cpp placed on GPU
    bool        vulkan_confirmed     = false;  // true if log intercept saw ggml_vulkan
    bool        cuda_confirmed       = false;  // true if CUDA device was selected and layers offloaded
    size_t      device_total_bytes   = 0;      // selected GPU memory after load (0 if unknown)
    size_t      device_free_bytes    = 0;

    // --- Timings (ms) ---
    double  model_load_ms   = 0.0;    // Time to load + mmap model file
    double  prompt_eval_ms  = 0.0;    // Time to evaluate prompt tokens
    double  generation_ms   = 0.0;    // Time to generate output tokens
    double  total_ms        = 0.0;    // model_load + prompt_eval + generation

    // --- Token ids / numerics (Phase 26) ---
    std::vector<int>   output_token_ids;     // generated token ids (vocabulary ids)
    std::vector<float> first_token_logits;   // logits for the first generated token (only when LlmConfig::capture_first_logits)

    // --- Token counts ---
    int     prompt_tokens  = 0;
    int     output_tokens  = 0;
    double  tokens_per_sec = 0.0;     // output_tokens / (generation_ms / 1000)

    // Phase 33: why generation ended ("eos", "max_tokens", "stop_sequence",
    // "time_budget", "cancelled", "context_limit").
    std::string finish_reason;
};

} // namespace agr
