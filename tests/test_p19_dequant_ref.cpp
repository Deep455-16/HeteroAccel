// tests/test_p19_dequant_ref.cpp
// Phase 19: independent numerical regression tests for HydroXL dequantizers
// against the bundled GGML reference implementation.
//
// Reference source: ggml/src/ggml-quants.{h,c} from the llama.cpp dependency
// fetched by the project's CMake (linked directly; no transcription).
//
// Procedure per format:
//   1. Generate deterministic pseudo-random input floats (std::mt19937).
//   2. Quantize them with the real GGML quantize_row_*_ref().
//   3. Decode the same blocks twice: with the real GGML dequantize_row_*()
//      (reference output) and with the HydroXL decoder under test.
//   4. Require max |hydroxl - reference| <= 1e-6 * max(1, max|reference|).
//      Both implementations perform the same float operations in the same
//      order (fp16->fp32 conversion is exact), so in practice the diff is 0;
//      the epsilon only guards against compiler-level FMA contraction.
//
// Structural assertions (block sizes) are checked against static_asserted
// GGML structs, so a layout drift in either side fails the test.

#include "mini_test.h"

#include "ggml-quants.h" // bundled GGML reference (extern "C", GGML_API)

#include "hydroxl/HydroXLDequant.h"
#include "hydroxl/HydroXLTypes.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace agr::hydroxl;

namespace {

struct DiffResult {
    double max_abs;
    double max_rel_scale; // max_abs relative to max|ref|
    bool   exact;
};

DiffResult compare(const std::vector<float>& ref, const std::vector<float>& got) {
    DiffResult d{0.0, 0.0, true};
    double max_ref = 0.0;
    for (float v : ref) max_ref = std::max(max_ref, std::fabs((double)v));
    for (size_t i = 0; i < ref.size(); ++i) {
        double diff = std::fabs((double)ref[i] - (double)got[i]);
        if (diff > d.max_abs) d.max_abs = diff;
        if (diff != 0.0) d.exact = false;
    }
    d.max_rel_scale = d.max_abs / std::max(1.0, max_ref);
    return d;
}

std::vector<float> makeInput(size_t n, uint32_t seed, float lo, float hi) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> x(n);
    for (float& v : x) v = dist(rng);
    return x;
}

bool withinTolerance(const DiffResult& d) {
    return d.max_rel_scale <= 1e-6;
}

} // namespace

int main() {
    // ------------------------------------------------------------------
    // Struct-size agreement between GGML reference and HydroXL constants
    // ------------------------------------------------------------------
    AGR_CHECK(sizeof(block_q4_K) == Q4K_RAW_BLOCK_BYTES); // 144
    AGR_CHECK(sizeof(block_q3_K) == Q3K_RAW_BLOCK_BYTES); // 110
    AGR_CHECK(sizeof(block_q5_0) == 22);
    AGR_CHECK(sizeof(block_q5_1) == 24);
    AGR_CHECK(sizeof(block_q6_K) == 210);
    AGR_CHECK(sizeof(block_q8_0) == Q8_0_BLOCK_BYTES);   // 34

    // ==================================================================
    // Q4_K  (type 12; Q4_K_M/Q4_K_S models store Q4_K and Q6_K tensors)
    // ==================================================================
    {
        const int k = 1024; // 4 blocks
        std::vector<float> x = makeInput(k, 1001u, -2.0f, 3.0f);
        std::vector<block_q4_K> blocks(k / QK_K);
        quantize_row_q4_K_ref(x.data(), blocks.data(), k);

        std::vector<float> ref(k), got(k);
        dequantize_row_q4_K(blocks.data(), ref.data(), k);
        AGR_CHECK(dequant_q4k(blocks.data(), got.data(), k));

        DiffResult d = compare(ref, got);
        std::cout << "  Q4_K k=1024: max_abs=" << d.max_abs
                  << " exact=" << (d.exact ? "yes" : "no") << "\n";
        AGR_CHECK(withinTolerance(d));

        // Deterministic single block, zero input
        std::vector<float> zeros(QK_K, 0.0f);
        std::vector<block_q4_K> zblock(1);
        quantize_row_q4_K_ref(zeros.data(), zblock.data(), QK_K);
        std::vector<float> zref(QK_K), zgot(QK_K);
        dequantize_row_q4_K(zblock.data(), zref.data(), QK_K);
        dequant_q4k(zblock.data(), zgot.data(), QK_K);
        DiffResult dz = compare(zref, zgot);
        AGR_CHECK(withinTolerance(dz));

        // Constant input exercises non-trivial mins
        std::vector<float> c(QK_K, 0.5f);
        std::vector<block_q4_K> cblock(1);
        quantize_row_q4_K_ref(c.data(), cblock.data(), QK_K);
        std::vector<float> cref(QK_K), cgot(QK_K);
        dequantize_row_q4_K(cblock.data(), cref.data(), QK_K);
        dequant_q4k(cblock.data(), cgot.data(), QK_K);
        DiffResult dc = compare(cref, cgot);
        AGR_CHECK(withinTolerance(dc));

        // First/last element of each block explicitly
        for (int b = 0; b < 4; ++b) {
            AGR_CHECK(got[b*QK_K + 0]   == ref[b*QK_K + 0]   || withinTolerance(
                compare({ref[b*QK_K+0]}, {got[b*QK_K+0]})));
            AGR_CHECK(withinTolerance(compare({ref[b*QK_K+255]}, {got[b*QK_K+255]})));
        }

        // Invalid sizes rejected
        AGR_CHECK(!dequant_q4k(blocks.data(), got.data(), 255));
        AGR_CHECK(!dequant_q4k(blocks.data(), got.data(), 0));
        AGR_CHECK(!dequant_q4k(nullptr, got.data(), k));
    }

    // ==================================================================
    // Q3_K  (type 11)
    // ==================================================================
    {
        const int k = 768; // 3 blocks
        std::vector<float> x = makeInput(k, 2002u, -1.0f, 1.0f);
        std::vector<block_q3_K> blocks(k / QK_K);
        quantize_row_q3_K_ref(x.data(), blocks.data(), k);

        std::vector<float> ref(k), got(k);
        dequantize_row_q3_K(blocks.data(), ref.data(), k);
        AGR_CHECK(dequant_q3k(blocks.data(), got.data(), k));

        DiffResult d = compare(ref, got);
        std::cout << "  Q3_K k=768: max_abs=" << d.max_abs
                  << " exact=" << (d.exact ? "yes" : "no") << "\n";
        AGR_CHECK(withinTolerance(d));

        AGR_CHECK(!dequant_q3k(blocks.data(), got.data(), 100)); // not %256
        AGR_CHECK(!dequant_q3k(blocks.data(), got.data(), 0));
    }

    // ==================================================================
    // Q5_0  (type 6)
    // ==================================================================
    {
        const int k = 1024; // 32 blocks
        std::vector<float> x = makeInput(k, 3003u, -1.0f, 1.0f);
        std::vector<block_q5_0> blocks(k / 32);
        quantize_row_q5_0_ref(x.data(), blocks.data(), k);

        std::vector<float> ref(k), got(k);
        dequantize_row_q5_0(blocks.data(), ref.data(), k);
        AGR_CHECK(dequant_q5_0(blocks.data(), got.data(), k));

        DiffResult d = compare(ref, got);
        std::cout << "  Q5_0 k=1024: max_abs=" << d.max_abs
                  << " exact=" << (d.exact ? "yes" : "no") << "\n";
        AGR_CHECK(withinTolerance(d));

        // Zero and extreme inputs
        std::vector<float> zeros(32, 0.0f);
        std::vector<block_q5_0> zb(1);
        quantize_row_q5_0_ref(zeros.data(), zb.data(), 32);
        std::vector<float> zr(32), zg(32);
        dequantize_row_q5_0(zb.data(), zr.data(), 32);
        dequant_q5_0(zb.data(), zg.data(), 32);
        AGR_CHECK(withinTolerance(compare(zr, zg)));

        std::vector<float> amp(96, 4.5f); // 3 blocks, constant
        std::vector<block_q5_0> ab(3);
        quantize_row_q5_0_ref(amp.data(), ab.data(), 96);
        std::vector<float> ar(96), ag(96);
        dequantize_row_q5_0(ab.data(), ar.data(), 96);
        dequant_q5_0(ab.data(), ag.data(), 96);
        DiffResult da = compare(ar, ag);
        AGR_CHECK(withinTolerance(da));

        AGR_CHECK(!dequant_q5_0(blocks.data(), got.data(), 33));
        AGR_CHECK(!dequant_q5_0(blocks.data(), got.data(), 0));
    }

    // ==================================================================
    // Q5_1  (type 7)
    // ==================================================================
    {
        const int k = 1024;
        // shifted distribution so the additive minimum is non-trivial
        std::vector<float> x = makeInput(k, 4004u, 1.0f, 5.0f);
        std::vector<block_q5_1> blocks(k / 32);
        quantize_row_q5_1_ref(x.data(), blocks.data(), k);

        std::vector<float> ref(k), got(k);
        dequantize_row_q5_1(blocks.data(), ref.data(), k);
        AGR_CHECK(dequant_q5_1(blocks.data(), got.data(), k));

        DiffResult d = compare(ref, got);
        std::cout << "  Q5_1 k=1024: max_abs=" << d.max_abs
                  << " exact=" << (d.exact ? "yes" : "no") << "\n";
        AGR_CHECK(withinTolerance(d));

        AGR_CHECK(!dequant_q5_1(blocks.data(), got.data(), 16));
        AGR_CHECK(!dequant_q5_1(blocks.data(), got.data(), 0));
    }

    // ==================================================================
    // Q6_K  (type 14)
    // ==================================================================
    {
        const int k = 1024; // 4 blocks
        std::vector<float> x = makeInput(k, 5005u, -3.0f, 2.0f);
        std::vector<block_q6_K> blocks(k / QK_K);
        quantize_row_q6_K_ref(x.data(), blocks.data(), k);

        std::vector<float> ref(k), got(k);
        dequantize_row_q6_K(blocks.data(), ref.data(), k);
        AGR_CHECK(dequant_q6_k(blocks.data(), got.data(), k));

        DiffResult d = compare(ref, got);
        std::cout << "  Q6_K k=1024: max_abs=" << d.max_abs
                  << " exact=" << (d.exact ? "yes" : "no") << "\n";
        AGR_CHECK(withinTolerance(d));

        // Single block and negative-skewed input
        std::vector<float> neg(QK_K);
        std::mt19937 rng(6006u);
        std::uniform_real_distribution<float> dn(-8.0f, -1.0f);
        for (float& v : neg) v = dn(rng);
        std::vector<block_q6_K> nb(1);
        quantize_row_q6_K_ref(neg.data(), nb.data(), QK_K);
        std::vector<float> nr(QK_K), ng(QK_K);
        dequantize_row_q6_K(nb.data(), nr.data(), QK_K);
        dequant_q6_k(nb.data(), ng.data(), QK_K);
        DiffResult dn2 = compare(nr, ng);
        AGR_CHECK(withinTolerance(dn2));

        AGR_CHECK(!dequant_q6_k(blocks.data(), got.data(), 255));
        AGR_CHECK(!dequant_q6_k(blocks.data(), got.data(), 0));
    }

    // ==================================================================
    // Q8_0 regression (type 8) and F16 spot-check
    // ==================================================================
    {
        const int k = 512;
        std::vector<float> x = makeInput(k, 7007u, -1.5f, 1.5f);
        std::vector<block_q8_0> blocks(k / 32);
        quantize_row_q8_0_ref(x.data(), blocks.data(), k);

        std::vector<float> ref(k), got(k);
        dequantize_row_q8_0(blocks.data(), ref.data(), k);
        AGR_CHECK(dequant_q8_0(blocks.data(), got.data(), k));

        DiffResult d = compare(ref, got);
        std::cout << "  Q8_0 k=512: max_abs=" << d.max_abs
                  << " exact=" << (d.exact ? "yes" : "no") << "\n";
        AGR_CHECK(withinTolerance(d));
    }
    {
        // F16: compare against an independent software reference written
        // with a different formulation (ldexpf-based) than HydroXL's
        // bit-shift implementation.
        auto refHalfToFloat = [](uint16_t h) -> float {
            const bool  sign = (h >> 15) != 0;
            const uint32_t exp = (h >> 10) & 0x1F;
            const uint32_t man = h & 0x3FF;
            float v;
            if (exp == 0) {
                v = std::ldexpf(static_cast<float>(man), -24); // subnormal
            } else if (exp == 31) {
                v = (man == 0) ? std::numeric_limits<float>::infinity()
                               : std::numeric_limits<float>::quiet_NaN();
            } else {
                v = std::ldexpf(static_cast<float>(man + 0x400), static_cast<int>(exp) - 25);
            }
            return sign ? -v : v;
        };

        uint16_t raw[256];
        for (int i = 0; i < 256; ++i) raw[i] = static_cast<uint16_t>(i * 257);
        float got[256];
        dequant_f16(raw, got, 256);
        int mismatches = 0;
        for (int i = 0; i < 256; ++i) {
            float expected = refHalfToFloat(raw[i]);
            if (got[i] != expected &&
                !(std::isnan(got[i]) && std::isnan(expected))) {
                ++mismatches;
            }
        }
        AGR_CHECK(mismatches == 0);
    }

    AGR_TEST_MAIN_END()
}
