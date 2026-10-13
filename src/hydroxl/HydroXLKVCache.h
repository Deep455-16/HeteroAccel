// src/hydroxl/HydroXLKVCache.h
// Phase 17: Key-Value cache management for HydroXL.
//
// KV cache stores pre-computed K and V tensors for all past token positions
// in all transformer layers. This allows incremental decoding (no full
// re-computation of previous tokens on each new token).
//
// Layout:
//   layer l, head h, position t, dimension d:
//     k_cache[l][h * max_seq * head_dim + t * head_dim + d]
//     v_cache[l][h * max_seq * head_dim + t * head_dim + d]
//
// Memory: 2 * n_layers * n_kv_heads * max_seq_len * head_dim * 4 bytes

#pragma once

#include "hydroxl/HydroXLTypes.h"

#include <cstdint>
#include <vector>

namespace agr {
namespace hydroxl {

class HydroXLKVCache {
public:
    HydroXLKVCache() = default;
    ~HydroXLKVCache() = default;

    // Non-copyable, movable
    HydroXLKVCache(const HydroXLKVCache&) = delete;
    HydroXLKVCache& operator=(const HydroXLKVCache&) = delete;
    HydroXLKVCache(HydroXLKVCache&&) = default;
    HydroXLKVCache& operator=(HydroXLKVCache&&) = default;

    /// Allocate the cache for a model with given dimensions.
    /// Returns false if allocation fails or parameters are invalid.
    bool allocate(uint32_t n_layers, uint32_t n_kv_heads,
                  uint32_t max_seq_len, uint32_t head_dim);

    bool isAllocated() const { return allocated_; }

    uint32_t nLayers()     const { return n_layers_; }
    uint32_t nKVHeads()    const { return n_kv_heads_; }
    uint32_t maxSeqLen()   const { return max_seq_len_; }
    uint32_t headDim()     const { return head_dim_; }
    uint32_t currentSeqLen() const { return seq_len_; }

    /// Memory used in bytes.
    size_t memoryBytes() const;

    /// Reset sequence position to 0 (keeps allocated memory).
    void reset();

    /// Write K and V vectors for a token at position `pos` in layer `layer`.
    /// k_vec and v_vec: [n_kv_heads * head_dim] floats each.
    bool writeKV(uint32_t layer, uint32_t pos,
                 const float* k_vec, const float* v_vec);

    /// Advance sequence length by 1 (called after writeKV for current token).
    /// Returns false if would exceed max_seq_len.
    bool advanceSeq();

    /// K cache pointer for layer `layer` (read access for attention).
    /// Layout: [n_kv_heads * max_seq_len * head_dim]
    const float* kLayer(uint32_t layer) const;

    /// V cache pointer for layer `layer`.
    const float* vLayer(uint32_t layer) const;

private:
    uint32_t n_layers_    = 0;
    uint32_t n_kv_heads_  = 0;
    uint32_t max_seq_len_ = 0;
    uint32_t head_dim_    = 0;
    uint32_t seq_len_     = 0;  ///< Current filled position count
    bool     allocated_   = false;

    // k_data_[layer] and v_data_[layer] each store:
    // [n_kv_heads * max_seq_len * head_dim] floats
    std::vector<std::vector<float>> k_data_;
    std::vector<std::vector<float>> v_data_;
};

} // namespace hydroxl
} // namespace agr
