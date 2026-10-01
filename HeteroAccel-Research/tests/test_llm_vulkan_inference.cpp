// test_llm_vulkan_inference.cpp
//
// Verifies that Vulkan-offloaded inference succeeds.
// Skips with code 77 if no model is provided OR if Vulkan is not available.

#include "mini_test.h"
#include "llm/LlamaCppEngine.h"
#include "llm/LlmConfig.h"
#include "hardware/HardwareDetector.h"

#include <cstdlib>

int main() {
    std::cout << "== test_llm_vulkan_inference ==\n";

    // 1. Check Vulkan availability first
    agr::HardwareInfo info = agr::HardwareDetector::detectAll();
    if (!info.vulkan.available) {
        std::cout << "  SKIP: Vulkan not available on this system.\n";
        return 77;
    }

    // 2. Check model path
    const char* model_env = std::getenv("HETEROACCEL_MODEL_PATH");
    if (!model_env || std::string(model_env).empty()) {
        std::cout << "  SKIP: HETEROACCEL_MODEL_PATH is not set.\n";
        return 77;
    }

    agr::LlmConfig cfg;
    cfg.model_path = model_env;
    cfg.n_gpu_layers = 99; // offload all
    cfg.max_new_tokens = 10;
    cfg.seed = 42;

    agr::LlamaCppEngine engine;
    bool ok = engine.initialize(cfg);
    if (!ok) {
        std::cerr << "  initialize() failed: " << engine.lastError() << "\n";
        return 1;
    }

    std::cout << "  Model loaded.\n";
    std::cout << "  Vulkan confirmed in logs: " << (engine.vulkanConfirmed() ? "YES" : "NO") << "\n";
    std::cout << "  Device: " << engine.detectedVulkanDevice() << "\n";
    std::cout << "  GPU layers loaded: " << engine.gpuLayersActual() << "\n";

    agr::LlmResult r = engine.infer("The quick brown fox jumps over the lazy");
    
    std::cout << "  infer() success: " << (r.success ? "true" : "false") << "\n";
    if (!r.success) {
        std::cerr << "  Error: " << r.error << "\n";
    }

    AGR_CHECK(r.success);
    AGR_CHECK(r.backend == "Vulkan");
    AGR_CHECK(r.vulkan_confirmed);
    AGR_CHECK(r.gpu_layers_actual > 0);
    AGR_CHECK(r.output_tokens > 0);
    AGR_CHECK(r.total_ms > 0.0);
    AGR_CHECK(!r.output.empty());
    
    std::cout << "  Output: " << r.output << "\n";
    std::cout << "  Time: " << r.total_ms << " ms\n";
    
    AGR_TEST_MAIN_END();
}
