// src/orchestration/MemoryEstimator.cpp
// Phase 24: Context-Aware Memory Estimation for HeteroAccel

#define _CRT_SECURE_NO_WARNINGS
#include "orchestration/MemoryEstimator.h"
#include "orchestration/HardwareDetector.h"

#include <sstream>
#include <algorithm>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace agr {
namespace orchestration {

ModelMemoryEstimate MemoryEstimator::estimate(
    uint32_t n_layers,
    uint32_t n_embd,
    uint32_t n_heads,
    uint32_t n_kv_heads,
    uint32_t n_ff,
    uint32_t vocab_size,
    uint32_t max_seq_len,
    float    safety_factor
) {
    ModelMemoryEstimate est;
    if (n_layers == 0 || n_embd == 0 || n_heads == 0 || n_kv_heads == 0 || vocab_size == 0) {
        est.notes = "Invalid parameters: dimensions must be non-zero";
        return est;
    }

    uint32_t head_dim = n_embd / n_heads;
    if (head_dim == 0) head_dim = 64;
    if (max_seq_len == 0) max_seq_len = 512;

    // 1. Layer weights (all dequantized to float32):
    // attn_norm: n_embd
    // attn_q: n_embd * (n_heads * head_dim) + (n_heads * head_dim bias)
    // attn_k: n_embd * (n_kv_heads * head_dim) + (n_kv_heads * head_dim bias)
    // attn_v: n_embd * (n_kv_heads * head_dim) + (n_kv_heads * head_dim bias)
    // attn_out: (n_heads * head_dim) * n_embd
    // ffn_norm: n_embd
    // ffn_gate: n_embd * n_ff
    // ffn_up: n_embd * n_ff
    // ffn_down: n_ff * n_embd
    size_t layer_weights = (size_t)2 * n_embd
                         + (size_t)n_embd * (n_heads * head_dim)
                         + (size_t)n_embd * (n_kv_heads * head_dim)
                         + (size_t)n_embd * (n_kv_heads * head_dim)
                         + (size_t)(n_heads * head_dim) * n_embd
                         + (size_t)n_embd * n_ff
                         + (size_t)n_embd * n_ff
                         + (size_t)n_ff * n_embd
                         + (size_t)(n_heads * head_dim + 2 * n_kv_heads * head_dim);

    size_t total_layer_weights = (size_t)n_layers * layer_weights;
    // Token embedding + output norm + lm_head
    size_t non_layer_weights = (size_t)vocab_size * n_embd + (size_t)n_embd + (size_t)vocab_size * n_embd;
    est.weights_bytes = (total_layer_weights + non_layer_weights) * sizeof(float);

    // 2. KV Cache:
    // 2 (K and V) * n_layers * n_kv_heads * head_dim * max_seq_len * sizeof(float)
    est.kv_cache_bytes = (size_t)2 * n_layers * n_kv_heads * head_dim * max_seq_len * sizeof(float);

    // 3. Activations (per-token scratch buffers):
    // hidden, normed, q, k, v, attn_out, proj_out, gate, up, ffn_buf, ffn_out, attn_scores
    size_t act_elems = (size_t)10 * n_embd + (size_t)3 * n_ff + max_seq_len;
    est.activations_bytes = act_elems * sizeof(float);

    if (safety_factor < 1.0f) safety_factor = 1.0f;
    size_t base_bytes = est.weights_bytes + est.kv_cache_bytes + est.activations_bytes;
    est.overhead_bytes = static_cast<size_t>(base_bytes * (safety_factor - 1.0f));
    est.total_estimated = base_bytes + est.overhead_bytes;

    const auto* gpu = HardwareDetector::instance().best_gpu();
    bool is_unified = true;
    if (gpu) {
        if ((gpu->vendor == DeviceInfo::Vendor::NVIDIA || gpu->vendor == DeviceInfo::Vendor::AMD) &&
            gpu->vram_bytes > 0) {
            is_unified = false;
        }
    }
    est.unified_memory = is_unified;

    std::ostringstream ss;
    ss << "HydroXL estimate: layers=" << n_layers << ", embd=" << n_embd
       << ", max_seq=" << max_seq_len
       << " -> weights: " << (est.weights_bytes / (1024 * 1024)) << " MB, "
       << "KV: " << (est.kv_cache_bytes / (1024 * 1024)) << " MB, "
       << "total: " << (est.total_estimated / (1024 * 1024)) << " MB";
    if (est.unified_memory) ss << " (unified memory GPU/CPU pool)";
    est.notes = ss.str();

    return est;
}

size_t MemoryEstimator::available_ram_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo)) {
        return static_cast<size_t>(memInfo.ullAvailPhys);
    }
#endif
    return 16ULL * 1024 * 1024 * 1024;
}

bool MemoryEstimator::fits_in_memory(const ModelMemoryEstimate& est,
                                     size_t available_bytes,
                                     bool unified_memory) {
    if (available_bytes == 0) return true;
    size_t threshold = unified_memory ? static_cast<size_t>(available_bytes * 0.90) : available_bytes;
    return est.total_estimated <= threshold;
}

} // namespace orchestration
} // namespace agr
