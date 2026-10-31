// src/orchestration/MemoryEstimator.h
// Phase 24: Context-Aware Memory Estimation for HeteroAccel

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

namespace agr {
namespace orchestration {

struct ModelMemoryEstimate {
    size_t weights_bytes = 0;       // dequantized float32 model weights
    size_t kv_cache_bytes = 0;      // KV cache for max_seq_len
    size_t activations_bytes = 0;   // per-token scratch (hidden, normed, q/k/v, ffn bufs)
    size_t overhead_bytes = 0;      // safety headroom
    size_t total_estimated = 0;     // sum of above
    bool   unified_memory = false;  // true = GPU and CPU share the same pool
    std::string notes;              // human-readable explanation
};

class MemoryEstimator {
public:
    // Estimate memory for a HydroXL model given its hyperparameters
    static ModelMemoryEstimate estimate(
        uint32_t n_layers,
        uint32_t n_embd,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t n_ff,
        uint32_t vocab_size,
        uint32_t max_seq_len,
        float    safety_factor = 1.1f
    );

    // Query available system RAM
    static size_t available_ram_bytes();

    // Check if model fits with safety headroom
    static bool fits_in_memory(const ModelMemoryEstimate& est,
                               size_t available_bytes,
                               bool unified_memory);
};

} // namespace orchestration
} // namespace agr
