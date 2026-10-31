// tests/test_p24_memory_est.cpp
// Phase 24: Context-Aware Memory Estimation Tests

#define _CRT_SECURE_NO_WARNINGS
#include "mini_test.h"
#include "orchestration/MemoryEstimator.h"
#include <heteroaccel/heteroaccel.h>
#include <heteroaccel/HeteroAccel.hpp>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>

using namespace agr::orchestration;

int main() {
    std::cout << "--- Phase 24: MemoryEstimator Tests ---\n";

    // Test 1: Calculation for typical 0.5B model hyperparameters
    // Qwen2.5-0.5B: layers=24, embd=896, heads=14, kv_heads=2, ff=4864, vocab=151936, ctx=2048
    auto est = MemoryEstimator::estimate(
        24,     // n_layers
        896,    // n_embd
        14,     // n_heads
        2,      // n_kv_heads
        4864,   // n_ff
        151936, // vocab_size
        2048,   // max_seq_len
        1.10f   // safety_factor
    );

    AGR_CHECK(est.weights_bytes > 0);
    AGR_CHECK(est.kv_cache_bytes > 0);
    AGR_CHECK(est.activations_bytes > 0);
    AGR_CHECK(est.overhead_bytes > 0);
    AGR_CHECK(est.total_estimated > est.weights_bytes);
    AGR_CHECK(est.total_estimated == est.weights_bytes + est.kv_cache_bytes +
                                   est.activations_bytes + est.overhead_bytes);
    AGR_CHECK(!est.notes.empty());
    AGR_CHECK(est.notes.find("weights:") != std::string::npos || est.notes.find("Weights:") != std::string::npos);

    std::cout << "  0.5B Total estimated: " << (est.total_estimated / (1024 * 1024)) << " MB\n";
    std::cout << "  Weights: " << (est.weights_bytes / (1024 * 1024)) << " MB\n";
    std::cout << "  KV Cache: " << (est.kv_cache_bytes / (1024 * 1024)) << " MB\n";
    std::cout << "  Activations: " << (est.activations_bytes / (1024 * 1024)) << " MB\n";

    // Test 2: Edge cases: zero layers or zero embedding
    auto est_zero = MemoryEstimator::estimate(0, 0, 0, 0, 0, 0, 0);
    AGR_CHECK(est_zero.total_estimated == 0);

    // Test 3: System RAM detection
    size_t ram = MemoryEstimator::available_ram_bytes();
    AGR_CHECK(ram > 0);
    std::cout << "  Detected Available RAM: " << (ram / (1024 * 1024)) << " MB\n";

    // Test 4: Memory fitting checks
    // Model should fit in 64 GB
    AGR_CHECK(MemoryEstimator::fits_in_memory(est, (size_t)64 * 1024 * 1024 * 1024ULL, false));
    // Model should NOT fit in 1 MB
    AGR_CHECK(!MemoryEstimator::fits_in_memory(est, (size_t)1024 * 1024, false));

    // Unified memory safety check (needs headroom for OS)
    AGR_CHECK(MemoryEstimator::fits_in_memory(est, (size_t)16 * 1024 * 1024 * 1024ULL, true));

    // Test 5: SDK API error handling
    HaMemoryEstimate sdk_est = {};
    AGR_CHECK(ha_model_memory_estimate(nullptr, 2048, &sdk_est) == HA_ERR_INVALID_ARG);

    HaRuntime rt = ha_runtime_create();
    AGR_CHECK(rt != nullptr);

    // Test 6: Model memory estimate with real model if available
    const char* model_env = std::getenv("HETEROACCEL_MODEL_PATH");
    std::string model_path = model_env ? model_env : "";
    if (model_path.empty() || !std::filesystem::exists(model_path)) {
        std::string fallback = "D:/HeteroAccel/Models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";
        if (std::filesystem::exists(fallback)) {
            model_path = fallback;
        }
    }

    if (!model_path.empty() && std::filesystem::exists(model_path)) {
        HaError err = HA_OK;
        HaModel model = ha_model_load(rt, model_path.c_str(), &err);
        AGR_CHECK(model != nullptr);
        AGR_CHECK(err == HA_OK);

        AGR_CHECK(ha_model_memory_estimate(model, 2048, nullptr) == HA_ERR_INVALID_ARG);

        HaMemoryEstimate m_est = {};
        AGR_CHECK(ha_model_memory_estimate(model, 2048, &m_est) == HA_OK);
        AGR_CHECK(m_est.total_estimated > 0);
        AGR_CHECK(m_est.weights_bytes > 0);
        AGR_CHECK(m_est.notes != nullptr);
        AGR_CHECK(std::strlen(m_est.notes) > 0);

        std::cout << "  Model total estimate via SDK: " << (m_est.total_estimated / (1024 * 1024)) << " MB\n";
        std::cout << "  Notes: " << m_est.notes << "\n";

        // C++ wrapper check
        {
            ha::Model cpp_model(model);
            ha::MemoryEstimate cpp_est = cpp_model.memory_estimate(2048);
            AGR_CHECK(cpp_est.total_estimated > 0);
            AGR_CHECK(cpp_est.weights_bytes > 0);
            AGR_CHECK(!cpp_est.notes.empty());
            cpp_model.release(); // Keep underlying C handle alive for ha_model_unload
        }

        ha_model_unload(model);
    } else {
        std::cout << "  [INFO] Skipping real model estimation test (model file not found)\n";
    }

    ha_runtime_destroy(rt);

    AGR_TEST_MAIN_END();
}
