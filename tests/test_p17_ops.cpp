// tests/test_p17_ops.cpp
// Phase 17: Unit tests for transformer operator correctness.

#include "mini_test.h"
#include "hydroxl/HydroXLOps.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace agr::hydroxl;

int main() {

    // === RMS Norm ===
    {   // identity weights
        float x[4]   = {1.f, 2.f, 3.f, 4.f};
        float w[4]   = {1.f, 1.f, 1.f, 1.f};
        float out[4] = {};
        AGR_CHECK(rms_norm(x, w, out, 4, 1e-6f));
        float expected_scale = 1.0f / std::sqrt(7.5f + 1e-6f);
        for (int i = 0; i < 4; ++i)
            AGR_CHECK(std::fabs(out[i] - x[i] * expected_scale) < 1e-5f);
    }
    {   // zero input stays finite
        float x[4]   = {};
        float w[4]   = {1.f, 1.f, 1.f, 1.f};
        float out[4] = {};
        AGR_CHECK(rms_norm(x, w, out, 4, 1e-6f));
        for (int i = 0; i < 4; ++i) AGR_CHECK(std::isfinite(out[i]));
    }
    {   // null check
        AGR_CHECK(!rms_norm(nullptr, nullptr, nullptr, 4, 1e-6f));
    }
    {   // scaled weights: x=[1,1], w=[2,3] → out ≈ [2,3]
        float x[2]   = {1.f, 1.f};
        float w[2]   = {2.f, 3.f};
        float out[2] = {};
        AGR_CHECK(rms_norm(x, w, out, 2, 1e-6f));
        AGR_CHECK(std::fabs(out[0] - 2.0f) < 0.01f);
        AGR_CHECK(std::fabs(out[1] - 3.0f) < 0.01f);
    }

    // === Matrix-vector product ===
    {   // 1×1: 3 * 2 = 6
        float A[1] = {3.f}, x[1] = {2.f}, out[1] = {};
        AGR_CHECK(matvec(A, x, out, 1, 1));
        AGR_CHECK(std::fabs(out[0] - 6.f) < 1e-5f);
    }
    {   // 2×2: [[1,2],[3,4]] * [1,1] = [3,7]
        float A[4] = {1.f,2.f,3.f,4.f}, x[2] = {1.f,1.f}, out[2] = {};
        AGR_CHECK(matvec(A, x, out, 2, 2));
        AGR_CHECK(std::fabs(out[0] - 3.f) < 1e-5f);
        AGR_CHECK(std::fabs(out[1] - 7.f) < 1e-5f);
    }
    {   // zero matrix
        float A[4] = {}, x[2] = {1.f,2.f}, out[2] = {};
        AGR_CHECK(matvec(A, x, out, 2, 2));
        AGR_CHECK(out[0] == 0.f && out[1] == 0.f);
    }
    {   // null check
        AGR_CHECK(!matvec(nullptr, nullptr, nullptr, 2, 2));
    }

    // === Embedding lookup ===
    {   // vocab=3, n_embd=2, pick token 1
        float table[6] = {1.f,2.f,3.f,4.f,5.f,6.f};
        float dst[2]   = {};
        AGR_CHECK(embed_lookup(table, 1, 3, 2, dst));
        AGR_CHECK(dst[0] == 3.f && dst[1] == 4.f);
    }
    {   // out of bounds token
        float table[4] = {1.f,2.f,3.f,4.f};
        float dst[2]   = {};
        AGR_CHECK(!embed_lookup(table, 2, 2, 2, dst));
    }
    {   // null check
        AGR_CHECK(!embed_lookup(nullptr, 0, 2, 2, nullptr));
    }

    // === RoPE ===
    {   // at pos=0, angle=0 → no rotation
        float q[4] = {1.f,2.f,3.f,4.f}, k[4] = {5.f,6.f,7.f,8.f};
        float q0[4], k0[4];
        std::memcpy(q0, q, 16); std::memcpy(k0, k, 16);
        AGR_CHECK(rope_apply(q, 1, k, 1, 4, 0, 1000000.f, 1.f));
        for (int i = 0; i < 4; ++i) {
            AGR_CHECK(std::fabs(q[i]-q0[i]) < 1e-4f);
            AGR_CHECK(std::fabs(k[i]-k0[i]) < 1e-4f);
        }
    }
    {   // rotation preserves L2 norm
        float q[4] = {1.f,0.f,0.f,1.f}, k[4] = {0.f,1.f,1.f,0.f};
        auto norm4 = [](const float* v) {
            return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]+v[3]*v[3]);
        };
        float qn0 = norm4(q), kn0 = norm4(k);
        AGR_CHECK(rope_apply(q, 1, k, 1, 4, 5, 1000000.f, 1.f));
        AGR_CHECK(std::fabs(norm4(q) - qn0) < 1e-4f);
        AGR_CHECK(std::fabs(norm4(k) - kn0) < 1e-4f);
    }
    {   // null check
        AGR_CHECK(!rope_apply(nullptr, 1, nullptr, 1, 4, 0, 1e6f, 1.f));
    }
    {   // odd head_dim rejected
        float q[4] = {}, k[4] = {};
        AGR_CHECK(!rope_apply(q, 1, k, 1, 3, 0, 1e6f, 1.f));
    }

    // === SiLU ===
    {   float x = 0.f; silu_inplace(&x, 1); AGR_CHECK(x == 0.f); }
    {   float x[2] = {1.f, 2.f}; silu_inplace(x, 2);
        AGR_CHECK(std::fabs(x[0] - 0.7310586f) < 1e-4f);
    }

    // === SwiGLU ===
    {   float gate[2]={0.f,0.f}, up[2]={5.f,5.f}, out[2]={};
        swiglu(gate, up, out, 2);
        AGR_CHECK(out[0] == 0.f && out[1] == 0.f);
    }
    {   float gate[1]={1.f}, up[1]={2.f}, out[1]={};
        swiglu(gate, up, out, 1);
        AGR_CHECK(std::fabs(out[0] - 0.7310586f * 2.f) < 1e-4f);
    }

    // === Softmax ===
    {   float x[4] = {1.f,2.f,3.f,4.f}; softmax_inplace(x, 4);
        float s = x[0]+x[1]+x[2]+x[3];
        AGR_CHECK(std::fabs(s - 1.f) < 1e-5f);
    }
    {   float x[3] = {0.f,0.f,0.f}; softmax_inplace(x, 3);
        AGR_CHECK(std::fabs(x[0] - 1.f/3.f) < 1e-5f);
    }
    {   float x[1] = {7.f}; softmax_inplace(x, 1);
        AGR_CHECK(std::fabs(x[0] - 1.f) < 1e-5f);
    }

    // === Residual add ===
    {   float a[3]={1.f,2.f,3.f}, b[3]={4.f,5.f,6.f};
        residual_add(a, b, 3);
        AGR_CHECK(a[0]==5.f && a[1]==7.f && a[2]==9.f);
    }

    // === Causal attention (GQA smoke test) ===
    {   // 1 head, 1 KV head, head_dim=4, seq_len=1, max_seq=4
        // Single-token: softmax([score])=[1.0], out = v0
        float q[4]          = {1.f,0.f,0.f,0.f};
        float k_cache[16]   = {};
        float v_cache[16]   = {};
        float k0[4]={1.f,0.f,0.f,0.f}, v0[4]={3.f,1.f,4.f,1.f};
        std::memcpy(k_cache, k0, 16);
        std::memcpy(v_cache, v0, 16);
        float out[4] = {};
        float scale = 1.0f / std::sqrt(4.0f);
        AGR_CHECK(causal_attention(q, k_cache, v_cache, out, 1, 1, 1, 4, 4, scale));
        AGR_CHECK(std::fabs(out[0]-3.f) < 1e-4f);
        AGR_CHECK(std::fabs(out[1]-1.f) < 1e-4f);
        AGR_CHECK(std::fabs(out[2]-4.f) < 1e-4f);
        AGR_CHECK(std::fabs(out[3]-1.f) < 1e-4f);
    }
    {   // null check
        AGR_CHECK(!causal_attention(nullptr, nullptr, nullptr, nullptr, 1, 1, 1, 4, 4, 1.f));
    }
    {   // mismatched heads (3 % 2 != 0)
        float buf[16] = {};
        AGR_CHECK(!causal_attention(buf, buf, buf, buf, 1, 3, 2, 4, 4, 1.f));
    }

    AGR_TEST_MAIN_END()
}
