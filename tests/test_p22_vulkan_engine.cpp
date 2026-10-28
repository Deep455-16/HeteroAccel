// tests/test_p22_vulkan_engine.cpp
// Phase 22: HydroXL Vulkan Engine Adapter Tests

#include "mini_test.h"
#include "orchestration/HydroXLEngineAdapter.h"
#include "orchestration/HardwareDetector.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

using namespace agr;
using namespace agr::orchestration;

int main() {
    auto& hw = HardwareDetector::instance();
    hw.detect();
    if (!hw.has_vulkan()) {
        std::cout << "[SKIP] Vulkan not available on this platform\n";
        return 77;
    }

    const char* model_env = std::getenv("HETEROACCEL_MODEL_PATH");
    std::string model_path = model_env ? model_env : "";
    if (model_path.empty()) {
        std::cout << "[SKIP] HETEROACCEL_MODEL_PATH environment variable not set\n";
        return 77;
    }

    std::ifstream mf(model_path, std::ios::binary);
    if (!mf.is_open()) {
        std::cout << "[SKIP] Model file not accessible: " << model_path << "\n";
        return 77;
    }
    mf.close();

    std::cout << "--- Phase 22: HydroXLVulkanEngineAdapter Test Suite ---\n";
    std::cout << "[INFO] Model path: " << model_path << "\n";

    HydroXLVulkanEngineAdapter vk_engine;
    std::string err;

    // Test 1: initialize() succeeds with real model
    bool init_ok = vk_engine.initialize(model_path, err);
    if (!init_ok) {
        std::cerr << "Vulkan engine initialization error: " << err << "\n";
    }
    AGR_CHECK(init_ok);
    AGR_CHECK(vk_engine.is_ready());

    // Test 4: backend_id() returns "hydroxl_vulkan"
    AGR_CHECK(vk_engine.backend_id() == "hydroxl_vulkan");

    // Test 2: run() returns success=true, gpu_used=true, output_tokens > 0
    InferenceRequest req;
    // prompt_token_ids left empty — engine defaults to tokenizing "The capital of France is"
    req.max_new_tokens = 8;
    req.temperature = 0.0f; // greedy

    auto res1 = vk_engine.run(req);
    std::cout << "[INFO] Vulkan output 1: '" << res1.output_text << "' (tokens=" << res1.output_tokens
              << ", tok/s=" << res1.tokens_per_second << ")\n";
    AGR_CHECK(res1.success);
    AGR_CHECK(res1.gpu_used == true);
    AGR_CHECK(res1.backend_used == "hydroxl_vulkan");
    AGR_CHECK(res1.output_tokens > 0);
    AGR_CHECK(!res1.output_text.empty());

    // Test 3: Greedy output is deterministic (run twice, compare)
    auto res2 = vk_engine.run(req);
    std::cout << "[INFO] Vulkan output 2: '" << res2.output_text << "' (tokens=" << res2.output_tokens << ")\n";
    AGR_CHECK(res2.success);
    AGR_CHECK(res1.output_token_ids == res2.output_token_ids);
    AGR_CHECK(res1.output_text == res2.output_text);

    // Test 5: Compare Vulkan output tokens with CPU output tokens
    {
        HydroXLCpuEngineAdapter cpu_engine;
        std::string cpu_err;
        bool cpu_init = cpu_engine.initialize(model_path, cpu_err);
        AGR_CHECK(cpu_init);
        if (cpu_init) {
            auto cpu_res = cpu_engine.run(req);
            std::cout << "[INFO] CPU output: '" << cpu_res.output_text << "' (tokens=" << cpu_res.output_tokens
                      << ", tok/s=" << cpu_res.tokens_per_second << ")\n";
            AGR_CHECK(cpu_res.success);
            AGR_CHECK(cpu_res.gpu_used == false);
            bool tokens_match = (cpu_res.output_token_ids == res1.output_token_ids);
            std::cout << "[INFO] CPU vs Vulkan tokens match: " << (tokens_match ? "YES" : "NO") << "\n";
            AGR_CHECK(tokens_match);
            cpu_engine.shutdown();
        }
    }

    // Test 6: shutdown() then is_ready() returns false
    vk_engine.shutdown();
    AGR_CHECK(!vk_engine.is_ready());

    AGR_TEST_MAIN_END();
}
