#pragma once
#include <cstdint>
#include <string>

namespace agr {

// Configuration for a single LLM inference run.
// Passed to LlamaCppEngine::initialize().
struct LlmConfig {
    std::string model_path;           // Absolute path to GGUF model file

    // GPU offload
    int  n_gpu_layers  = 99;          // 99 = "offload as many layers as fit"
    bool cpu_only      = false;       // Force n_gpu_layers = 0 (CPU reference path)

    // Context / generation
    int      n_ctx         = 2048;    // Context length (tokens)
    int      n_batch       = 512;     // Batch size for prompt eval
    int      max_new_tokens = 256;    // Maximum tokens to generate
    float    temperature    = 0.7f;   // Sampling temperature
    uint32_t seed           = 42;     // RNG seed for reproducibility

    // Helpers
    int effective_gpu_layers() const {
        return cpu_only ? 0 : n_gpu_layers;
    }
};

} // namespace agr
