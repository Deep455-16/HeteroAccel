// src/hydroxl/HydroXLForwardPass.h
// Phase 17: Qwen2 transformer forward pass for HydroXL.
//
// Implements incremental (single-token) forward pass using the loaded model
// weights and the KV cache. Produces logits over the vocabulary.
//
// Pipeline per token:
//   1. Embedding lookup
//   2. For each layer:
//      a. attn_norm (RMSNorm)
//      b. QKV projection + bias
//      c. RoPE on Q and K
//      d. Write K, V to KV cache
//      e. Causal attention (GQA)
//      f. Output projection
//      g. Residual add
//      h. ffn_norm (RMSNorm)
//      i. FFN: gate + up projections, SwiGLU, down projection
//      j. Residual add
//   3. Final RMSNorm
//   4. LM head projection → logits [vocab_size]

#pragma once

#include "hydroxl/HydroXLModel.h"
#include "hydroxl/HydroXLKVCache.h"

#include <cstdint>
#include <vector>

namespace agr {
namespace hydroxl {

class HydroXLForwardPass {
public:
    HydroXLForwardPass() = default;

    /// Attach model and KV cache (non-owning pointers).
    /// model and kv_cache must outlive this object.
    void attach(const HydroXLModel* model, HydroXLKVCache* kv_cache);

    bool isReady() const { return model_ && kv_cache_ && kv_cache_->isAllocated(); }

    /// Run one forward pass for the given token at position pos.
    /// Fills logits[0..vocab_size-1].
    /// Returns false on error. lastError() contains the reason.
    bool forward(uint32_t token_id, uint32_t pos,
                 std::vector<float>& logits);

    std::string lastError() const { return last_error_; }

private:
    const HydroXLModel* model_    = nullptr;
    HydroXLKVCache*     kv_cache_ = nullptr;

    mutable std::string last_error_;

    // Scratch buffers (reused across tokens to avoid repeated allocation)
    std::vector<float> hidden_;     ///< [n_embd]
    std::vector<float> normed_;     ///< [n_embd]
    std::vector<float> q_buf_;      ///< [n_heads * head_dim]
    std::vector<float> k_buf_;      ///< [n_kv_heads * head_dim]
    std::vector<float> v_buf_;      ///< [n_kv_heads * head_dim]
    std::vector<float> attn_out_;   ///< [n_heads * head_dim]
    std::vector<float> proj_out_;   ///< [n_embd]
    std::vector<float> gate_buf_;   ///< [n_ff]
    std::vector<float> up_buf_;     ///< [n_ff]
    std::vector<float> ffn_buf_;    ///< [n_ff]
    std::vector<float> ffn_out_;    ///< [n_embd]
    std::vector<float> attn_scores_;///< [max_seq_len] scratch for causal attention
};

} // namespace hydroxl
} // namespace agr
