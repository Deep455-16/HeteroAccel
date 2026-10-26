// src/hydroxl/HydroXLOps.h
// Phase 17: Native transformer operators for HydroXL.
//
// All operators work on FLOAT32 data (post-dequantization).
// No SIMD extensions: plain scalar C++ for portability and correctness.
//
// Supported architecture: Qwen2 (qwen2)
//   - RMSNorm with epsilon
//   - GEMM: mat-vec product (weight [out, in] × input [in]) → output [out]
//   - Embedding lookup
//   - RoPE (rotary positional encoding, Qwen2 variant with freq_base=1e6)
//   - SiLU activation (for SwiGLU / SiLU-FFN)
//   - SwiGLU FFN step: silu(gate) * up
//   - Softmax (causal masked)
//   - Causal attention (multi-head, grouped-query)
//   - Residual add (in-place)
//
// All functions validate inputs and return false on error.

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

/// In-place RoPE for a single token at position pos.
/// q/k: [n_heads, head_dim] and [n_kv_heads, head_dim] respectively.
/// head_dim must be even.
bool rope_apply(float* q, uint32_t n_heads,
                float* k, uint32_t n_kv_heads,
                uint32_t head_dim, uint32_t pos,
                float freq_base, float freq_scale) noexcept;

/// SiLU: out[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i]))
void silu_inplace(float* x, uint32_t n) noexcept;

/// SwiGLU combine: out[i] = silu(gate[i]) * up[i]
void swiglu(const float* gate, const float* up, float* out, uint32_t n) noexcept;

/// Residual add: a[i] += b[i]
void residual_add(float* a, const float* b, uint32_t n) noexcept;

/// Softmax in-place over x[0..n-1].
void softmax_inplace(float* x, uint32_t n) noexcept;

/// Causal self-attention for a single query token at position pos.
///
/// q          : [n_heads, head_dim] — query for current token
/// k_cache    : [n_kv_heads, max_seq, head_dim] — key cache (all prior tokens incl. current)
/// v_cache    : [n_kv_heads, max_seq, head_dim] — value cache
/// out        : [n_heads, head_dim] — attention output
/// seq_len    : number of valid tokens in cache (including current)
/// n_heads    : query heads
/// n_kv_heads : key/value heads (≤ n_heads, must divide evenly)
/// head_dim   : dimension per head
/// scale      : attention scale (typically 1/sqrt(head_dim))
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
                      float* scratch_scores = nullptr) noexcept;

} // namespace hydroxl
} // namespace agr
