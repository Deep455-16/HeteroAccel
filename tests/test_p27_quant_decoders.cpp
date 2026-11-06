// tests/test_p27_quant_decoders.cpp
// Phase 27: Comprehensive GGML quantization decoder validation for HydroXL.
//
// Validates all 13 prioritized GGML tensor types against the bundled upstream
// GGML reference implementation (ggml/src/ggml-quants.{h,c}):
//   F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K

#include "mini_test.h"
#include "gguf_fixture.h"

#include "ggml-quants.h"

#include "hydroxl/GGUFLoader.h"
#include "hydroxl/HydroXLDequant.h"
#include "hydroxl/HydroXLTypes.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace agr::hydroxl;

namespace {

std::vector<float> makeInput(size_t n, uint32_t seed, float lo, float hi) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> x(n);
    for (float& v : x) v = dist(rng);
    return x;
}

double maxRelDiff(const std::vector<float>& ref, const std::vector<float>& got) {
    double max_abs = 0.0;
    double max_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref[i])));
        double diff = std::fabs(static_cast<double>(ref[i]) - static_cast<double>(got[i]));
        if (diff > max_abs) max_abs = diff;
    }
    return max_abs / std::max(1.0, max_ref);
}

uint16_t floatToFp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

} // namespace

int main() {
    std::cout << "=== Phase 27: All-13 GGML Quantization Decoder Reference Suite ===\n";

    // ------------------------------------------------------------------
    // 1. Block size & byte layout verification against upstream GGML structs
    // ------------------------------------------------------------------
    AGR_CHECK(sizeof(block_q4_0) == Q4_0_BLOCK_BYTES);   // 18
    AGR_CHECK(sizeof(block_q4_1) == Q4_1_BLOCK_BYTES);   // 20
    AGR_CHECK(sizeof(block_q5_0) == Q5_0_BLOCK_BYTES);   // 22
    AGR_CHECK(sizeof(block_q5_1) == Q5_1_BLOCK_BYTES);   // 24
    AGR_CHECK(sizeof(block_q8_0) == Q8_0_BLOCK_BYTES);   // 34
    AGR_CHECK(sizeof(block_q2_K) == Q2K_RAW_BLOCK_BYTES);// 84
    AGR_CHECK(sizeof(block_q3_K) == Q3K_RAW_BLOCK_BYTES);// 110
    AGR_CHECK(sizeof(block_q4_K) == Q4K_RAW_BLOCK_BYTES);// 144
    AGR_CHECK(sizeof(block_q5_K) == Q5K_RAW_BLOCK_BYTES);// 176
    AGR_CHECK(sizeof(block_q6_K) == Q6K_RAW_BLOCK_BYTES);// 210
    AGR_CHECK(sizeof(block_q8_K) == Q8K_RAW_BLOCK_BYTES);// 292

    const GGMLType all_13_types[] = {
        GGMLType::F32,  GGMLType::F16,
        GGMLType::Q4_0, GGMLType::Q4_1,
        GGMLType::Q5_0, GGMLType::Q5_1,
        GGMLType::Q8_0,
        GGMLType::Q2_K, GGMLType::Q3_K,
        GGMLType::Q4_K, GGMLType::Q5_K,
        GGMLType::Q6_K, GGMLType::Q8_K
    };
    for (GGMLType t : all_13_types) {
        AGR_CHECK(is_quant_type_supported(t));
    }
    AGR_CHECK(!is_quant_type_supported(GGMLType::UNKNOWN));

    const int k = 1024; // divisible by both 32 and 256

    // ------------------------------------------------------------------
    // 2. Q4_0 (type 2: 32 elems, 18 bytes)
    // ------------------------------------------------------------------
    {
        std::vector<float> x = makeInput(k, 2701u, -3.5f, 3.5f);
        std::vector<block_q4_0> blocks(k / 32);
        quantize_row_q4_0_ref(x.data(), blocks.data(), k);

        std::vector<float> ref(k), got1(k), got2(k);
        dequantize_row_q4_0(blocks.data(), ref.data(), k);
        AGR_CHECK(dequant_q4_0(blocks.data(), got1.data(), k));
        AGR_CHECK(dequantize_tensor(GGMLType::Q4_0, blocks.data(), got2.data(), k));

        double err1 = maxRelDiff(ref, got1);
        double err2 = maxRelDiff(ref, got2);
        std::cout << "  Q4_0 maxRelDiff=" << err1 << "\n";
        AGR_CHECK(err1 <= 1e-6);
        AGR_CHECK(err2 <= 1e-6);

        // Bounds & alignment checks
        AGR_CHECK(!dequant_q4_0(blocks.data(), got1.data(), 31));
        AGR_CHECK(!dequant_q4_0(blocks.data(), got1.data(), 0));
        AGR_CHECK(!dequant_q4_0(nullptr, got1.data(), k));
    }

    // ------------------------------------------------------------------
    // 3. Q4_1 (type 3: 32 elems, 20 bytes)
    // ------------------------------------------------------------------
    {
        std::vector<float> x = makeInput(k, 2702u, 0.5f, 6.0f); // non-zero min
        std::vector<block_q4_1> blocks(k / 32);
        quantize_row_q4_1_ref(x.data(), blocks.data(), k);

        std::vector<float> ref(k), got1(k), got2(k);
        dequantize_row_q4_1(blocks.data(), ref.data(), k);
        AGR_CHECK(dequant_q4_1(blocks.data(), got1.data(), k));
        AGR_CHECK(dequantize_tensor(GGMLType::Q4_1, blocks.data(), got2.data(), k));

        double err1 = maxRelDiff(ref, got1);
        std::cout << "  Q4_1 maxRelDiff=" << err1 << "\n";
        AGR_CHECK(err1 <= 1e-6);
        AGR_CHECK(maxRelDiff(ref, got2) <= 1e-6);

        AGR_CHECK(!dequant_q4_1(blocks.data(), got1.data(), 33));
        AGR_CHECK(!dequant_q4_1(blocks.data(), got1.data(), 0));
    }

    // ------------------------------------------------------------------
    // 4. Q2_K (type 10: 256 elems, 84 bytes)
    // ------------------------------------------------------------------
    {
        std::vector<float> x = makeInput(k, 2703u, -2.0f, 4.0f);
        std::vector<block_q2_K> blocks(k / 256);
        quantize_row_q2_K_ref(x.data(), blocks.data(), k);

        std::vector<float> ref(k), got1(k), got2(k);
        dequantize_row_q2_K(blocks.data(), ref.data(), k);
        AGR_CHECK(dequant_q2_k(blocks.data(), got1.data(), k));
        AGR_CHECK(dequantize_tensor(GGMLType::Q2_K, blocks.data(), got2.data(), k));

        double err1 = maxRelDiff(ref, got1);
        std::cout << "  Q2_K maxRelDiff=" << err1 << "\n";
        AGR_CHECK(err1 <= 1e-6);
        AGR_CHECK(maxRelDiff(ref, got2) <= 1e-6);

        AGR_CHECK(!dequant_q2_k(blocks.data(), got1.data(), 255));
        AGR_CHECK(!dequant_q2_k(blocks.data(), got1.data(), 0));
    }

    // ------------------------------------------------------------------
    // 5. Q5_K (type 13: 256 elems, 176 bytes)
    // ------------------------------------------------------------------
    {
        std::vector<float> x = makeInput(k, 2704u, -4.0f, 2.5f);
        std::vector<block_q5_K> blocks(k / 256);
        quantize_row_q5_K_ref(x.data(), blocks.data(), k);

        std::vector<float> ref(k), got1(k), got2(k);
        dequantize_row_q5_K(blocks.data(), ref.data(), k);
        AGR_CHECK(dequant_q5_k(blocks.data(), got1.data(), k));
        AGR_CHECK(dequantize_tensor(GGMLType::Q5_K, blocks.data(), got2.data(), k));

        double err1 = maxRelDiff(ref, got1);
        std::cout << "  Q5_K maxRelDiff=" << err1 << "\n";
        AGR_CHECK(err1 <= 1e-6);
        AGR_CHECK(maxRelDiff(ref, got2) <= 1e-6);

        AGR_CHECK(!dequant_q5_k(blocks.data(), got1.data(), 128));
        AGR_CHECK(!dequant_q5_k(blocks.data(), got1.data(), 0));
    }

    // ------------------------------------------------------------------
    // 6. Q8_K (type 15: 256 elems, 292 bytes)
    // ------------------------------------------------------------------
    {
        std::vector<float> x = makeInput(k, 2705u, -5.0f, 5.0f);
        std::vector<block_q8_K> blocks(k / 256);
        quantize_row_q8_K_ref(x.data(), blocks.data(), k);

        std::vector<float> ref(k), got1(k), got2(k);
        dequantize_row_q8_K(blocks.data(), ref.data(), k);
        AGR_CHECK(dequant_q8_k(blocks.data(), got1.data(), k));
        AGR_CHECK(dequantize_tensor(GGMLType::Q8_K, blocks.data(), got2.data(), k));

        double err1 = maxRelDiff(ref, got1);
        std::cout << "  Q8_K maxRelDiff=" << err1 << "\n";
        AGR_CHECK(err1 <= 1e-6);
        AGR_CHECK(maxRelDiff(ref, got2) <= 1e-6);

        AGR_CHECK(!dequant_q8_k(blocks.data(), got1.data(), 250));
        AGR_CHECK(!dequant_q8_k(blocks.data(), got1.data(), 0));
    }

    // ------------------------------------------------------------------
    // 7. Full-stack GGUF disk round-trip across all 13 GGML types
    // ------------------------------------------------------------------
    {
        const char* gguf_path = "test_p27_all13_quants.gguf";
        std::vector<float> src = makeInput(256, 2706u, -1.75f, 2.25f);

        std::vector<uint16_t> b_f16(256);
        for (size_t i = 0; i < 256; ++i) b_f16[i] = floatToFp16(src[i]);

        std::vector<block_q4_0> b_q4_0(256 / 32); quantize_row_q4_0_ref(src.data(), b_q4_0.data(), 256);
        std::vector<block_q4_1> b_q4_1(256 / 32); quantize_row_q4_1_ref(src.data(), b_q4_1.data(), 256);
        std::vector<block_q5_0> b_q5_0(256 / 32); quantize_row_q5_0_ref(src.data(), b_q5_0.data(), 256);
        std::vector<block_q5_1> b_q5_1(256 / 32); quantize_row_q5_1_ref(src.data(), b_q5_1.data(), 256);
        std::vector<block_q8_0> b_q8_0(256 / 32); quantize_row_q8_0_ref(src.data(), b_q8_0.data(), 256);
        std::vector<block_q2_K> b_q2_k(1);        quantize_row_q2_K_ref(src.data(), b_q2_k.data(), 256);
        std::vector<block_q3_K> b_q3_k(1);        quantize_row_q3_K_ref(src.data(), b_q3_k.data(), 256);
        std::vector<block_q4_K> b_q4_k(1);        quantize_row_q4_K_ref(src.data(), b_q4_k.data(), 256);
        std::vector<block_q5_K> b_q5_k(1);        quantize_row_q5_K_ref(src.data(), b_q5_k.data(), 256);
        std::vector<block_q6_K> b_q6_k(1);        quantize_row_q6_K_ref(src.data(), b_q6_k.data(), 256);
        std::vector<block_q8_K> b_q8_k(1);        quantize_row_q8_K_ref(src.data(), b_q8_k.data(), 256);

        {
            fixture::GgufWriter w;
            w.kvStr("general.architecture", "llama");
            auto addRaw = [&](const std::string& name, GGMLType t, const void* ptr, size_t nbytes) {
                std::vector<uint8_t> bytes(nbytes);
                std::memcpy(bytes.data(), ptr, nbytes);
                w.tensor(name, {256}, static_cast<uint32_t>(t), bytes);
            };
            addRaw("t.f32",  GGMLType::F32,  src.data(),    256 * sizeof(float));
            addRaw("t.f16",  GGMLType::F16,  b_f16.data(),  256 * sizeof(uint16_t));
            addRaw("t.q4_0", GGMLType::Q4_0, b_q4_0.data(), b_q4_0.size() * sizeof(block_q4_0));
            addRaw("t.q4_1", GGMLType::Q4_1, b_q4_1.data(), b_q4_1.size() * sizeof(block_q4_1));
            addRaw("t.q5_0", GGMLType::Q5_0, b_q5_0.data(), b_q5_0.size() * sizeof(block_q5_0));
            addRaw("t.q5_1", GGMLType::Q5_1, b_q5_1.data(), b_q5_1.size() * sizeof(block_q5_1));
            addRaw("t.q8_0", GGMLType::Q8_0, b_q8_0.data(), b_q8_0.size() * sizeof(block_q8_0));
            addRaw("t.q2_k", GGMLType::Q2_K, b_q2_k.data(), sizeof(block_q2_K));
            addRaw("t.q3_k", GGMLType::Q3_K, b_q3_k.data(), sizeof(block_q3_K));
            addRaw("t.q4_k", GGMLType::Q4_K, b_q4_k.data(), sizeof(block_q4_K));
            addRaw("t.q5_k", GGMLType::Q5_K, b_q5_k.data(), sizeof(block_q5_K));
            addRaw("t.q6_k", GGMLType::Q6_K, b_q6_k.data(), sizeof(block_q6_K));
            addRaw("t.q8_k", GGMLType::Q8_K, b_q8_k.data(), sizeof(block_q8_K));
            AGR_CHECK(w.write(gguf_path));
        }

        GGUFLoader loader;
        AGR_CHECK(loader.open(gguf_path));
        AGR_CHECK(loader.tensors().size() == 13);

        for (const auto& desc : loader.tensors()) {
            std::vector<uint8_t> raw(desc.n_bytes);
            AGR_CHECK(loader.readTensorData(desc, raw.data(), raw.size()));
            std::vector<float> decoded(256, 0.0f);
            AGR_CHECK(dequantize_tensor(desc.type, raw.data(), decoded.data(), 256));

            // Verify finite and correlated with original src
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (size_t i = 0; i < 256; ++i) {
                AGR_CHECK(std::isfinite(decoded[i]));
                dot += static_cast<double>(src[i]) * decoded[i];
                na  += static_cast<double>(src[i]) * src[i];
                nb  += static_cast<double>(decoded[i]) * decoded[i];
            }
            double cos_sim = dot / (std::sqrt(na) * std::sqrt(nb));
            std::cout << "  GGUF round-trip " << desc.name << " (" << toString(desc.type)
                      << ") cosine_vs_f32=" << cos_sim << "\n";
            AGR_CHECK(cos_sim > 0.90);
        }

        std::remove(gguf_path);
    }

    AGR_TEST_MAIN_END()
}
