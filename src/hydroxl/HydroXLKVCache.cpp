// src/hydroxl/HydroXLKVCache.cpp
// Phase 17: KV cache implementation.

#include "hydroxl/HydroXLKVCache.h"

#include <cstring>
#include <stdexcept>

namespace agr {
namespace hydroxl {

bool HydroXLKVCache::allocate(uint32_t n_layers, uint32_t n_kv_heads,
                               uint32_t max_seq_len, uint32_t head_dim) {
    if (n_layers == 0 || n_kv_heads == 0 || max_seq_len == 0 || head_dim == 0)
        return false;

    // Safety: reject combinations that would exceed ~4 GB
    uint64_t per_layer_floats = static_cast<uint64_t>(n_kv_heads) * max_seq_len * head_dim;
    uint64_t total_bytes = static_cast<uint64_t>(n_layers) * 2 * per_layer_floats * 4;
    if (total_bytes > (static_cast<uint64_t>(4) << 30)) return false;

    n_layers_    = n_layers;
    n_kv_heads_  = n_kv_heads;
    max_seq_len_ = max_seq_len;
    head_dim_    = head_dim;
    seq_len_     = 0;

    size_t per_layer = static_cast<size_t>(per_layer_floats);
    try {
        k_data_.assign(n_layers, std::vector<float>(per_layer, 0.0f));
        v_data_.assign(n_layers, std::vector<float>(per_layer, 0.0f));
    } catch (const std::bad_alloc&) {
        allocated_ = false;
        return false;
    }

    allocated_ = true;
    return true;
}

size_t HydroXLKVCache::memoryBytes() const {
    if (!allocated_) return 0;
    size_t per_layer = static_cast<size_t>(n_kv_heads_) * max_seq_len_ * head_dim_;
    return 2 * n_layers_ * per_layer * sizeof(float);
}

void HydroXLKVCache::reset() {
    seq_len_ = 0;
    // Zero the cache data (preserves allocation)
    for (auto& v : k_data_) std::fill(v.begin(), v.end(), 0.0f);
    for (auto& v : v_data_) std::fill(v.begin(), v.end(), 0.0f);
}

bool HydroXLKVCache::writeKV(uint32_t layer, uint32_t pos,
                              const float* k_vec, const float* v_vec) {
    if (!allocated_)                    return false;
    if (layer >= n_layers_)             return false;
    if (pos >= max_seq_len_)            return false;
    if (!k_vec || !v_vec)              return false;

    size_t vec_len = static_cast<size_t>(n_kv_heads_) * head_dim_;
    size_t offset  = static_cast<size_t>(pos) * n_kv_heads_ * head_dim_;

    // Check bounds
    if (offset + vec_len > k_data_[layer].size()) return false;

    std::memcpy(k_data_[layer].data() + offset, k_vec, vec_len * sizeof(float));
    std::memcpy(v_data_[layer].data() + offset, v_vec, vec_len * sizeof(float));
    return true;
}

bool HydroXLKVCache::advanceSeq() {
    if (!allocated_ || seq_len_ >= max_seq_len_) return false;
    ++seq_len_;
    return true;
}

const float* HydroXLKVCache::kLayer(uint32_t layer) const {
    if (!allocated_ || layer >= n_layers_) return nullptr;
    return k_data_[layer].data();
}

const float* HydroXLKVCache::vLayer(uint32_t layer) const {
    if (!allocated_ || layer >= n_layers_) return nullptr;
    return v_data_[layer].data();
}

} // namespace hydroxl
} // namespace agr
