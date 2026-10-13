// tests/test_p17_dequant.cpp
// Phase 17: Unit tests for Q4_K_M and Q8_0 dequantization.

#include "mini_test.h"
#include "hydroxl/HydroXLDequant.h"

#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <iostream>

using namespace agr::hydroxl;

int main() {
    // ------------------------------------------------------------------
    // FP16 → FP32 conversion
    // ------------------------------------------------------------------
    {   // +0.0
        float f = fp16_to_float(0x0000u);
        AGR_CHECK(f == 0.0f);
    }
    {   // 1.0 → 0x3C00
        float f = fp16_to_float(0x3C00u);
        AGR_CHECK(std::fabs(f - 1.0f) < 1e-4f);
    }
    {   // -1.0 → 0xBC00
        float f = fp16_to_float(0xBC00u);
        AGR_CHECK(std::fabs(f + 1.0f) < 1e-4f);
    }
    {   // 0.5 → 0x3800
        float f = fp16_to_float(0x3800u);
        AGR_CHECK(std::fabs(f - 0.5f) < 1e-4f);
    }
    {   // 2.0 → 0x4000
        float f = fp16_to_float(0x4000u);
        AGR_CHECK(std::fabs(f - 2.0f) < 1e-4f);
    }

    // ------------------------------------------------------------------
    // Q8_0 dequantization
    // ------------------------------------------------------------------
    {   // Zero block (scale=0, all values=0)
        uint8_t block[Q8_0_BLOCK_BYTES] = {};
        float out[Q8_0_BLOCK_SIZE] = {};
        dequant_q8_0_block(block, out);
        for (int i = 0; i < Q8_0_BLOCK_SIZE; ++i)
            AGR_CHECK(out[i] == 0.0f);
    }
    {   // scale=1.0 (FP16 0x3C00), q[0]=1, q[1]=2, q[2]=-1
        uint8_t block[Q8_0_BLOCK_BYTES] = {};
        block[0] = 0x00u; block[1] = 0x3Cu;
        reinterpret_cast<int8_t*>(block + 2)[0] = 1;
        reinterpret_cast<int8_t*>(block + 2)[1] = 2;
        reinterpret_cast<int8_t*>(block + 2)[2] = -1;
        float out[Q8_0_BLOCK_SIZE] = {};
        dequant_q8_0_block(block, out);
        AGR_CHECK(std::fabs(out[0] -  1.0f) < 1e-4f);
        AGR_CHECK(std::fabs(out[1] -  2.0f) < 1e-4f);
        AGR_CHECK(std::fabs(out[2] - -1.0f) < 1e-4f);
        AGR_CHECK(out[3] == 0.0f);
    }
    {   // null/bad args return false
        AGR_CHECK(!dequant_q8_0(nullptr, nullptr, 0));
        float dummy[32] = {};
        uint8_t raw[34] = {};
        AGR_CHECK(!dequant_q8_0(raw, dummy, 31)); // not divisible by 32
    }
    {   // Full one-block Q8_0 tensor
        uint8_t raw[Q8_0_BLOCK_BYTES] = {};
        raw[0] = 0x00u; raw[1] = 0x3Cu; // scale=1.0
        int8_t* q = reinterpret_cast<int8_t*>(raw + 2);
        for (int i = 0; i < Q8_0_BLOCK_SIZE; ++i) q[i] = static_cast<int8_t>(i - 16);
        float out[Q8_0_BLOCK_SIZE] = {};
        AGR_CHECK(dequant_q8_0(raw, out, Q8_0_BLOCK_SIZE));
        AGR_CHECK(std::fabs(out[0]  - -16.0f) < 1e-3f);
        AGR_CHECK(std::fabs(out[16] -   0.0f) < 1e-3f);
        AGR_CHECK(std::fabs(out[31] -  15.0f) < 1e-3f);
    }

    // ------------------------------------------------------------------
    // Q4_K block dequantization
    // ------------------------------------------------------------------
    {   // null/bad args
        AGR_CHECK(!dequant_q4k(nullptr, nullptr, 0));
        std::vector<uint8_t> raw(Q4K_RAW_BLOCK_BYTES, 0);
        float out[Q4K_BLOCK_SIZE] = {};
        AGR_CHECK(!dequant_q4k(raw.data(), out, 128)); // 128 not divisible by 256
    }
    {   // All-zero block → all outputs = 0
        std::vector<uint8_t> raw(Q4K_RAW_BLOCK_BYTES, 0);
        std::vector<float> out(Q4K_BLOCK_SIZE, 99.0f);
        AGR_CHECK(dequant_q4k(raw.data(), out.data(), Q4K_BLOCK_SIZE));
        for (float v : out) AGR_CHECK(v == 0.0f);
    }
    {   // d=1.0, scales[0..3]=1, all nibbles=4 → outputs must be finite
        std::vector<uint8_t> raw(Q4K_RAW_BLOCK_BYTES, 0);
        raw[0] = 0x00u; raw[1] = 0x3Cu; // d=1.0 FP16
        for (int i = 4; i < 8; ++i) raw[i] = 0x01u;
        for (int i = 16; i < 144; ++i) raw[i] = 0x44u;
        std::vector<float> out(Q4K_BLOCK_SIZE);
        AGR_CHECK(dequant_q4k(raw.data(), out.data(), Q4K_BLOCK_SIZE));
        for (float v : out) {
            AGR_CHECK(v == v);          // NaN check
            AGR_CHECK(std::isfinite(v));
        }
    }

    AGR_TEST_MAIN_END()
}
