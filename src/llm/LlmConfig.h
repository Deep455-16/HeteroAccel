#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace agr {

// Configuration for a single LLM inference run.
// Passed to LlamaCppEngine::initialize().
struct LlmConfig {
    std::string model_path;           // Absolute path to GGUF model file

    // GPU offload
    int  n_gpu_layers  = 99;          // 99 = "offload as many layers as fit"
    bool cpu_only      = false;       // Force n_gpu_layers = 0 (CPU reference path)

    // Explicit backend selection (HeteroAccel Phase 26). Empty / "auto" keeps the
    // legacy behaviour (llama.cpp decides across every compiled-in backend).
    // "cpu" | "cuda" | "vulkan": restrict llama.cpp to exactly that device kind;
    // initialize() FAILS (with a reason) when that kind is unavailable, so callers
    // can fall back deliberately instead of silently running elsewhere.
    std::string backend_preference;
    int         gpu_device_index = 0;  // index among devices of the preferred kind

    // Context / generation
    int      n_ctx         = 2048;    // Context length (tokens)
    int      n_batch       = 512;     // Batch size for prompt eval
    int      max_new_tokens = 256;    // Maximum tokens to generate
    float    temperature    = 0.7f;   // Sampling temperature
    uint32_t seed           = 42;     // RNG seed for reproducibility
    int      n_threads      = 0;      // 0 = auto-select
    bool     capture_first_logits = false; // copy first-token logits into LlmResult (numerical validation)

    // Phase 33 generation controls (0 / empty = unbounded / disabled)
    int      time_budget_ms = 0;            // wall-clock limit for the generation loop
    std::vector<std::string> stop_sequences; // generation stops before any of these
    bool     parse_special = true;          // treat <|...|> control markers in prompt as special tokens
    float    repeat_penalty = 1.0f;         // 1.0 = disabled (preserves pure greedy equivalence)

    // Helpers
    int effective_gpu_layers() const {
        return cpu_only ? 0 : n_gpu_layers;
    }
};

} // namespace agr
