// src/hydroxl/HydroXLOps.h
// Phase 17 / Phase 27: Native transformer operators for HydroXL.
//
// All operators work on FLOAT32 data (post-dequantization).
//
// Supported architectures: Llama/Llama 3, Mistral, Qwen2/Qwen2.5, Phi-3, Gemma, Gemma 2
//   - RMSNorm with epsilon
//   - GEMM: mat-vec product (weight [out, in] × input [in]) → output [out]
//   - Embedding lookup
//   - RoPE (both NEOX split-half and NORM interleaved-pair variants, partial rope_dim)
//   - SiLU / SwiGLU FFN: silu(gate) * up
//   - GELU / GeGLU FFN: gelu(gate) * up (Gemma / Gemma 2)
//   - Tanh logit softcapping (Gemma 2 attention & final logits)
//   - Softmax (causal masked, optional sliding window)
//   - Causal attention (multi-head, grouped-query)
//   - Residual add (in-place)

#pragma once

#include <cstddef>
#include <cstdint>

namespace agr {
namespace hydroxl {

/// RMS normalisation: out[i] = x[i] / rms(x) * weight[i]
/// x and out may alias. weight is the learned norm weights (size n).
bool rms_norm(const float* x, const float* weight, float* out,
              uint32_t n, float eps) noexcept;

/// Matrix-vector product: out[m] = A[m,k] * x[k]  (BLAS sgemv variant)
/// A is row-major [m_rows × k_cols], x is [k_cols], out is [m_rows].
bool matvec(const float* A, const float* x, float* out,
            uint32_t m_rows, uint32_t k_cols) noexcept;

/// Embedding lookup: dst[0..n_embd-1] = table[token_id * n_embd .. (token_id+1)*n_embd - 1]
bool embed_lookup(const float* table, uint32_t token_id,
                  uint32_t vocab_size, uint32_t n_embd,
                  float* dst) noexcept;

/// In-place RoPE for a single token at position pos (NEOX split-half default).
bool rope_apply(float* q, uint32_t n_heads,
                float* k, uint32_t n_kv_heads,
                uint32_t head_dim, uint32_t pos,
                float freq_base, float freq_scale) noexcept;

/// Extended in-place RoPE supporting both NEOX (split-half) and NORM (interleaved)
/// styles as well as partial rotary dimensions (rope_dim <= head_dim).
bool rope_apply_ext(float* q, uint32_t n_heads,
                    float* k, uint32_t n_kv_heads,
                    uint32_t head_dim, uint32_t rope_dim,
                    uint32_t pos,
                    float freq_base, float freq_scale,
                    bool neox_style) noexcept;

/// SiLU: out[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i]))
void silu_inplace(float* x, uint32_t n) noexcept;

/// SwiGLU combine: out[i] = silu(gate[i]) * up[i]
void swiglu(const float* gate, const float* up, float* out, uint32_t n) noexcept;

/// GeGLU combine (Gemma / Gemma 2): out[i] = gelu(gate[i]) * up[i]
void geglu(const float* gate, const float* up, float* out, uint32_t n) noexcept;

/// In-place tanh logit softcapping (Gemma 2): x[i] = tanh(x[i] / cap) * cap
void softcap_inplace(float* x, uint32_t n, float cap) noexcept;

/// Residual add: a[i] += b[i]
void residual_add(float* a, const float* b, uint32_t n) noexcept;

/// Softmax in-place over x[0..n-1].
void softmax_inplace(float* x, uint32_t n) noexcept;

/// Causal self-attention for a single query token at position pos.
bool causal_attention(const float* q,
                      const float* k_cache,
                      const float* v_cache,
                      float* out,
                      uint32_t seq_len,
                      uint32_t n_heads,
                      uint32_t n_kv_heads,
                      uint32_t head_dim,
                      uint32_t max_seq_len,
                      float scale,
                      float* scratch_scores = nullptr,
                      float attn_logit_softcap = 0.0f,
                      uint32_t sliding_window = 0) noexcept;

} // namespace hydroxl
} // namespace agr
