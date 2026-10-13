// src/hydroxl/HydroXLOps.cpp
// Phase 17: Scalar FLOAT32 transformer operator implementations.

#include "hydroxl/HydroXLOps.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace agr {
namespace hydroxl {

// ---------------------------------------------------------------------------
// RMS Normalisation
// ---------------------------------------------------------------------------
bool rms_norm(const float* x, const float* weight, float* out,
              uint32_t n, float eps) noexcept {
    if (!x || !weight || !out || n == 0) return false;

    // Compute mean of squares
    double sum_sq = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        sum_sq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    float rms_inv = 1.0f / std::sqrt(static_cast<float>(sum_sq / n) + eps);

    for (uint32_t i = 0; i < n; ++i) {
        out[i] = x[i] * rms_inv * weight[i];
    }
    return true;
}

// ---------------------------------------------------------------------------
// Matrix-vector product (row-major weight matrix)
// out[m] = sum_k A[m*k_cols + k] * x[k]
// ---------------------------------------------------------------------------
bool matvec(const float* A, const float* x, float* out,
            uint32_t m_rows, uint32_t k_cols) noexcept {
    if (!A || !x || !out || m_rows == 0 || k_cols == 0) return false;

    for (uint32_t m = 0; m < m_rows; ++m) {
        double acc = 0.0;
        const float* row = A + static_cast<size_t>(m) * k_cols;
        for (uint32_t k = 0; k < k_cols; ++k) {
            acc += static_cast<double>(row[k]) * static_cast<double>(x[k]);
        }
        out[m] = static_cast<float>(acc);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Embedding lookup
// ---------------------------------------------------------------------------
bool embed_lookup(const float* table, uint32_t token_id,
                  uint32_t vocab_size, uint32_t n_embd,
                  float* dst) noexcept {
    if (!table || !dst || n_embd == 0) return false;
    if (token_id >= vocab_size) return false;

    const float* src = table + static_cast<size_t>(token_id) * n_embd;
    std::memcpy(dst, src, n_embd * sizeof(float));
    return true;
}

// ---------------------------------------------------------------------------
// RoPE (Rotary Positional Encoding)
// Applies to Q and K tensors in-place.
// Each head pair (q[2i], q[2i+1]) is rotated by angle θ_i * pos
// where θ_i = 1 / (freq_base ^ (2i / head_dim)).
// Qwen2 uses freq_base = 1,000,000.
// ---------------------------------------------------------------------------
bool rope_apply(float* q, uint32_t n_heads,
                float* k, uint32_t n_kv_heads,
                uint32_t head_dim, uint32_t pos,
                float freq_base, float freq_scale) noexcept {
    if (!q || !k || head_dim == 0 || head_dim % 2 != 0) return false;
    if (n_heads == 0 || n_kv_heads == 0) return false;

    uint32_t half = head_dim / 2;

    // Apply to query heads
    for (uint32_t h = 0; h < n_heads; ++h) {
        float* qh = q + static_cast<size_t>(h) * head_dim;
        for (uint32_t i = 0; i < half; ++i) {
            // θ_i = freq_scale / (freq_base ^ (2i / head_dim))
            float theta = freq_scale / std::pow(freq_base, static_cast<float>(2 * i) / static_cast<float>(head_dim));
            float angle = static_cast<float>(pos) * theta;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);

            float q0 = qh[i];
            float q1 = qh[i + half];
            qh[i]        = q0 * cos_a - q1 * sin_a;
            qh[i + half] = q0 * sin_a + q1 * cos_a;
        }
    }

    // Apply to key heads
    for (uint32_t h = 0; h < n_kv_heads; ++h) {
        float* kh = k + static_cast<size_t>(h) * head_dim;
        for (uint32_t i = 0; i < half; ++i) {
            float theta = freq_scale / std::pow(freq_base, static_cast<float>(2 * i) / static_cast<float>(head_dim));
            float angle = static_cast<float>(pos) * theta;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);

            float k0 = kh[i];
            float k1 = kh[i + half];
            kh[i]        = k0 * cos_a - k1 * sin_a;
            kh[i + half] = k0 * sin_a + k1 * cos_a;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// SiLU activation
// ---------------------------------------------------------------------------
void silu_inplace(float* x, uint32_t n) noexcept {
    for (uint32_t i = 0; i < n; ++i) {
        float v = x[i];
        x[i] = v / (1.0f + std::exp(-v));
    }
}

// ---------------------------------------------------------------------------
// SwiGLU: out[i] = silu(gate[i]) * up[i]
// ---------------------------------------------------------------------------
void swiglu(const float* gate, const float* up, float* out, uint32_t n) noexcept {
    for (uint32_t i = 0; i < n; ++i) {
        float g = gate[i];
        float activated = g / (1.0f + std::exp(-g));
        out[i] = activated * up[i];
    }
}

// ---------------------------------------------------------------------------
// Residual add: a += b
// ---------------------------------------------------------------------------
void residual_add(float* a, const float* b, uint32_t n) noexcept {
    for (uint32_t i = 0; i < n; ++i) {
        a[i] += b[i];
    }
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------
void softmax_inplace(float* x, uint32_t n) noexcept {
    if (n == 0) return;
    float mx = *std::max_element(x, x + n);
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    if (sum > 0.0f) {
        for (uint32_t i = 0; i < n; ++i) x[i] /= sum;
    }
}

// ---------------------------------------------------------------------------
// Causal attention
// Grouped Query Attention (GQA): n_kv_heads divides n_heads evenly.
// Each KV head is shared by (n_heads / n_kv_heads) query heads.
//
// k_cache layout: [n_kv_heads * max_seq_len * head_dim]
//   stored as: head → position → dim
// v_cache layout: same
// ---------------------------------------------------------------------------
bool causal_attention(const float* q,
                      const float* k_cache,
                      const float* v_cache,
                      float* out,
                      uint32_t seq_len,
                      uint32_t n_heads,
                      uint32_t n_kv_heads,
                      uint32_t head_dim,
                      uint32_t max_seq_len,
                      float scale) noexcept {
    if (!q || !k_cache || !v_cache || !out) return false;
    if (seq_len == 0 || n_heads == 0 || n_kv_heads == 0 || head_dim == 0) return false;
    if (n_heads % n_kv_heads != 0) return false;
    if (seq_len > max_seq_len) return false;

    uint32_t heads_per_kv = n_heads / n_kv_heads;

    // Allocate a temporary score buffer on the stack for small seq_len.
    // For longer sequences, we use a static thread-local buffer.
    // Phase 17 uses a simple heap allocation per head.
    std::vector<float> scores(seq_len);

    for (uint32_t h = 0; h < n_heads; ++h) {
        const float* qh = q + static_cast<size_t>(h) * head_dim;
        float* outh     = out + static_cast<size_t>(h) * head_dim;

        // Corresponding KV head
        uint32_t kv_h = h / heads_per_kv;

        // Compute attention scores: scores[t] = dot(q_h, k_cache[kv_h, t]) * scale
        for (uint32_t t = 0; t < seq_len; ++t) {
            const float* kt = k_cache +
                static_cast<size_t>(kv_h) * max_seq_len * head_dim +
                static_cast<size_t>(t) * head_dim;

            double dot = 0.0;
            for (uint32_t d = 0; d < head_dim; ++d) {
                dot += static_cast<double>(qh[d]) * static_cast<double>(kt[d]);
            }
            scores[t] = static_cast<float>(dot) * scale;
        }

        // Causal mask is implicitly satisfied: seq_len only includes past + current tokens.
        softmax_inplace(scores.data(), seq_len);

        // Weighted sum of values: out_h = sum_t scores[t] * v_cache[kv_h, t]
        std::fill(outh, outh + head_dim, 0.0f);
        for (uint32_t t = 0; t < seq_len; ++t) {
            float s = scores[t];
            const float* vt = v_cache +
                static_cast<size_t>(kv_h) * max_seq_len * head_dim +
                static_cast<size_t>(t) * head_dim;
            for (uint32_t d = 0; d < head_dim; ++d) {
                outh[d] += s * vt[d];
            }
        }
    }
    return true;
}

} // namespace hydroxl
} // namespace agr
