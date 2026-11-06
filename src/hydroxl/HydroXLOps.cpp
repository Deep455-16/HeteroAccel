// src/hydroxl/HydroXLOps.cpp
// Phase 17 / Phase 27: FLOAT32 transformer operator implementations.

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
        const float* row = A + static_cast<size_t>(m) * k_cols;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        uint32_t k = 0;
        for (; k + 4 <= k_cols; k += 4) {
            acc0 += row[k + 0] * x[k + 0];
            acc1 += row[k + 1] * x[k + 1];
            acc2 += row[k + 2] * x[k + 2];
            acc3 += row[k + 3] * x[k + 3];
        }
        float acc = (acc0 + acc1) + (acc2 + acc3);
        for (; k < k_cols; ++k) {
            acc += row[k] * x[k];
        }
        out[m] = acc;
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
// RoPE (Rotary Positional Encoding) — supports NEOX (split-half) and NORM
// (interleaved consecutive pairs) styles and partial rope_dim <= head_dim.
// ---------------------------------------------------------------------------
bool rope_apply_ext(float* q, uint32_t n_heads,
                    float* k, uint32_t n_kv_heads,
                    uint32_t head_dim, uint32_t rope_dim,
                    uint32_t pos,
                    float freq_base, float freq_scale,
                    bool neox_style) noexcept {
    if (!q || !k || head_dim == 0 || head_dim % 2 != 0) return false;
    if (rope_dim == 0 || rope_dim > head_dim || rope_dim % 2 != 0) return false;
    if (n_heads == 0 || n_kv_heads == 0) return false;

    uint32_t half = rope_dim / 2;

    auto rotate_heads = [&](float* buf, uint32_t count) {
        for (uint32_t h = 0; h < count; ++h) {
            float* hd = buf + static_cast<size_t>(h) * head_dim;
            for (uint32_t i = 0; i < half; ++i) {
                float theta = freq_scale / std::pow(freq_base, static_cast<float>(2 * i) / static_cast<float>(rope_dim));
                float angle = static_cast<float>(pos) * theta;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                if (neox_style) {
                    float v0 = hd[i];
                    float v1 = hd[i + half];
                    hd[i]        = v0 * cos_a - v1 * sin_a;
                    hd[i + half] = v0 * sin_a + v1 * cos_a;
                } else {
                    float v0 = hd[2 * i];
                    float v1 = hd[2 * i + 1];
                    hd[2 * i]     = v0 * cos_a - v1 * sin_a;
                    hd[2 * i + 1] = v0 * sin_a + v1 * cos_a;
                }
            }
        }
    };

    rotate_heads(q, n_heads);
    rotate_heads(k, n_kv_heads);
    return true;
}

bool rope_apply(float* q, uint32_t n_heads,
                float* k, uint32_t n_kv_heads,
                uint32_t head_dim, uint32_t pos,
                float freq_base, float freq_scale) noexcept {
    return rope_apply_ext(q, n_heads, k, n_kv_heads,
                          head_dim, head_dim, pos,
                          freq_base, freq_scale, true);
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
// GeGLU (Gemma / Gemma 2): out[i] = gelu(gate[i]) * up[i]
// Matches ggml_fp32_to_fp32 GELU reference in ggml/src/ggml-cpu/vec.h.
// ---------------------------------------------------------------------------
void geglu(const float* gate, const float* up, float* out, uint32_t n) noexcept {
    static constexpr float SQRT_2_OVER_PI = 0.79788456080286535588f;
    static constexpr float GELU_COEF_A    = 0.044715f;
    for (uint32_t i = 0; i < n; ++i) {
        float x = gate[i];
        float gelu_x = 0.5f * x * (1.0f + std::tanh(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
        out[i] = gelu_x * up[i];
    }
}

// ---------------------------------------------------------------------------
// Tanh logit softcapping (Gemma 2)
// ---------------------------------------------------------------------------
void softcap_inplace(float* x, uint32_t n, float cap) noexcept {
    if (!x || cap <= 0.0f) return;
    for (uint32_t i = 0; i < n; ++i) {
        x[i] = std::tanh(x[i] / cap) * cap;
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
// Causal attention (Grouped Query Attention, optional softcapping & window)
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
                      float scale,
                      float* scratch_scores,
                      float attn_logit_softcap,
                      uint32_t sliding_window) noexcept {
    if (!q || !k_cache || !v_cache || !out) return false;
    if (seq_len == 0 || n_heads == 0 || n_kv_heads == 0 || head_dim == 0) return false;
    if (n_heads % n_kv_heads != 0) return false;
    if (seq_len > max_seq_len) return false;

    uint32_t heads_per_kv = n_heads / n_kv_heads;
    uint32_t start_t = (sliding_window > 0 && seq_len > sliding_window)
                           ? (seq_len - sliding_window)
                           : 0u;
    uint32_t active_len = seq_len - start_t;

    std::vector<float> local_scores;
    float* scores_ptr = scratch_scores;
    if (!scores_ptr) {
        local_scores.resize(seq_len);
        scores_ptr = local_scores.data();
    }

    for (uint32_t h = 0; h < n_heads; ++h) {
        const float* qh = q + static_cast<size_t>(h) * head_dim;
        float* outh     = out + static_cast<size_t>(h) * head_dim;

        uint32_t kv_h = h / heads_per_kv;

        for (uint32_t t = start_t; t < seq_len; ++t) {
            const float* kt = k_cache +
                static_cast<size_t>(kv_h) * max_seq_len * head_dim +
                static_cast<size_t>(t) * head_dim;

            double dot = 0.0;
            for (uint32_t d = 0; d < head_dim; ++d) {
                dot += static_cast<double>(qh[d]) * static_cast<double>(kt[d]);
            }
            float s = static_cast<float>(dot) * scale;
            if (attn_logit_softcap > 0.0f) {
                s = std::tanh(s / attn_logit_softcap) * attn_logit_softcap;
            }
            scores_ptr[t - start_t] = s;
        }

        softmax_inplace(scores_ptr, active_len);

        std::fill(outh, outh + head_dim, 0.0f);
        for (uint32_t t = start_t; t < seq_len; ++t) {
            float s = scores_ptr[t - start_t];
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
