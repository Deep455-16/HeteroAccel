// src/hydroxl/HydroXLForwardPass.cpp
// Phase 17: Qwen2 transformer forward pass — incremental decoding.

#include "hydroxl/HydroXLForwardPass.h"
#include "hydroxl/HydroXLOps.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace agr {
namespace hydroxl {

void HydroXLForwardPass::attach(const HydroXLModel* model,
                                 HydroXLKVCache*    kv_cache) {
    model_    = model;
    kv_cache_ = kv_cache;
}

bool HydroXLForwardPass::forward(uint32_t token_id, uint32_t pos,
                                  std::vector<float>& logits) {
    if (!model_ || !kv_cache_) {
        last_error_ = "Forward pass not attached to model/cache";
        return false;
    }
    if (!model_->isLoaded()) {
        last_error_ = "Model is not loaded";
        return false;
    }
    if (!kv_cache_->isAllocated()) {
        last_error_ = "KV cache not allocated";
        return false;
    }

    const Qwen2HParams& hp = model_->hparams();
    uint32_t n_embd    = hp.n_embd;
    uint32_t n_heads   = hp.n_heads;
    uint32_t n_kv      = hp.n_kv_heads;
    uint32_t head_dim  = hp.n_head_dim;
    uint32_t n_ff      = hp.n_ff;
    uint32_t vocab     = hp.vocab_size;
    float    rope_base = hp.rope_freq_base;
    float    attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // ---- Allocate / resize scratch buffers (first call only) ----
    hidden_.resize(n_embd);
    normed_.resize(n_embd);
    q_buf_.resize(n_heads * head_dim);
    k_buf_.resize(n_kv   * head_dim);
    v_buf_.resize(n_kv   * head_dim);
    attn_out_.resize(n_heads * head_dim);
    proj_out_.resize(n_embd);
    gate_buf_.resize(n_ff);
    up_buf_.resize(n_ff);
    ffn_buf_.resize(n_ff);
    ffn_out_.resize(n_embd);
    attn_scores_.resize(kv_cache_->maxSeqLen());
    logits.resize(vocab);

    // ---- Step 1: Embedding lookup ----
    if (!embed_lookup(model_->tokenEmbedding().data(), token_id,
                      vocab, n_embd, hidden_.data())) {
        last_error_ = "Invalid token_id=" + std::to_string(token_id);
        return false;
    }

    // ---- Steps 2–...: Transformer layers ----
    for (uint32_t l = 0; l < hp.n_layers; ++l) {
        const Qwen2LayerWeights& w = model_->layer(l);

        // --- 2a. Attention RMSNorm ---
        if (!rms_norm(hidden_.data(), w.attn_norm.data(), normed_.data(),
                      n_embd, hp.rms_norm_eps)) {
            last_error_ = "rms_norm failed at layer " + std::to_string(l);
            return false;
        }

        // --- 2b. QKV projections ---
        matvec(w.attn_q.data(), normed_.data(), q_buf_.data(),
               n_heads * head_dim, n_embd);
        matvec(w.attn_k.data(), normed_.data(), k_buf_.data(),
               n_kv   * head_dim, n_embd);
        matvec(w.attn_v.data(), normed_.data(), v_buf_.data(),
               n_kv   * head_dim, n_embd);

        // Add biases if present (Qwen2 has QKV biases)
        if (!w.attn_q_bias.empty()) {
            for (uint32_t i = 0; i < n_heads * head_dim; ++i)
                q_buf_[i] += w.attn_q_bias[i];
        }
        if (!w.attn_k_bias.empty()) {
            for (uint32_t i = 0; i < n_kv * head_dim; ++i)
                k_buf_[i] += w.attn_k_bias[i];
        }
        if (!w.attn_v_bias.empty()) {
            for (uint32_t i = 0; i < n_kv * head_dim; ++i)
                v_buf_[i] += w.attn_v_bias[i];
        }

        // --- 2c. RoPE ---
        rope_apply(q_buf_.data(), n_heads,
                   k_buf_.data(), n_kv,
                   head_dim, pos, rope_base, 1.0f);

        // --- 2d. Write K, V to cache ---
        if (!kv_cache_->writeKV(l, pos, k_buf_.data(), v_buf_.data())) {
            last_error_ = "KV cache write failed at layer " + std::to_string(l);
            return false;
        }

        // --- 2e. Causal attention (GQA) ---
        uint32_t seq_len = pos + 1; // current position is already written
        if (!causal_attention(q_buf_.data(),
                              kv_cache_->kLayer(l),
                              kv_cache_->vLayer(l),
                              attn_out_.data(),
                              seq_len,
                              n_heads, n_kv, head_dim,
                              kv_cache_->maxSeqLen(),
                              attn_scale,
                              attn_scores_.data())) {
            last_error_ = "causal_attention failed at layer " + std::to_string(l);
            return false;
        }

        // --- 2f. Output projection ---
        // attn_out_ is [n_heads * head_dim]; project to [n_embd]
        matvec(w.attn_out.data(), attn_out_.data(), proj_out_.data(),
               n_embd, n_heads * head_dim);

        // --- 2g. Residual add (attention) ---
        residual_add(hidden_.data(), proj_out_.data(), n_embd);

        // --- 2h. FFN RMSNorm ---
        if (!rms_norm(hidden_.data(), w.ffn_norm.data(), normed_.data(),
                      n_embd, hp.rms_norm_eps)) {
            last_error_ = "ffn rms_norm failed at layer " + std::to_string(l);
            return false;
        }

        // --- 2i. SwiGLU FFN ---
        // gate = W_gate * normed   [n_ff]
        matvec(w.ffn_gate.data(), normed_.data(), gate_buf_.data(), n_ff, n_embd);
        // up   = W_up   * normed   [n_ff]
        matvec(w.ffn_up.data(),   normed_.data(), up_buf_.data(),   n_ff, n_embd);
        // ffn_buf = silu(gate) * up
        swiglu(gate_buf_.data(), up_buf_.data(), ffn_buf_.data(), n_ff);
        // down = W_down * ffn_buf  [n_embd]
        matvec(w.ffn_down.data(), ffn_buf_.data(), ffn_out_.data(), n_embd, n_ff);

        // --- 2j. Residual add (FFN) ---
        residual_add(hidden_.data(), ffn_out_.data(), n_embd);
    }

    // ---- Step 3: Final RMSNorm ----
    if (!rms_norm(hidden_.data(), model_->outputNorm().data(), normed_.data(),
                  n_embd, hp.rms_norm_eps)) {
        last_error_ = "Final rms_norm failed";
        return false;
    }

    // ---- Step 4: LM head → logits ----
    matvec(model_->lmHead().data(), normed_.data(), logits.data(), vocab, n_embd);

    return true;
}

} // namespace hydroxl
} // namespace agr
