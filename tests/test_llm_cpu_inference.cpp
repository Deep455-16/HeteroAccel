// test_llm_cpu_inference.cpp
//
// Verifies that a real CPU-only inference succeeds.
// Skips with code 77 if no model is provided via HETEROACCEL_MODEL_PATH.

#include "mini_test.h"
#include "llm/LlamaCppEngine.h"
#include "llm/LlmConfig.h"

#include <cstdlib>

int main() {
    std::cout << "== test_llm_cpu_inference ==\n";

    const char* model_env = std::getenv("HETEROACCEL_MODEL_PATH");
    if (!model_env || std::string(model_env).empty()) {
        std::cout << "  SKIP: HETEROACCEL_MODEL_PATH is not set.\n";
        return 77;
    }

    agr::LlmConfig cfg;
    cfg.model_path = model_env;
    cfg.cpu_only   = true;
    cfg.max_new_tokens = 10;
    cfg.seed = 42;

    agr::LlamaCppEngine engine;
    bool ok = engine.initialize(cfg);
    if (!ok) {
        std::cerr << "  initialize() failed: " << engine.lastError() << "\n";
        // Do not skip here -- if a model path is provided but fails to load, it's a test failure.
        return 1;
    }

    std::cout << "  Model loaded.\n";
    std::cout << "  Backend: " << engine.detectedVulkanDevice() << " (Expected CPU-only for this test)\n";

    agr::LlmResult r = engine.infer("The quick brown fox jumps over the lazy");
    
    std::cout << "  infer() success: " << (r.success ? "true" : "false") << "\n";
    if (!r.success) {
        std::cerr << "  Error: " << r.error << "\n";
    }

    AGR_CHECK(r.success);
    AGR_CHECK(r.backend == "CPU");
    AGR_CHECK(r.gpu_layers_actual == 0);
    AGR_CHECK(r.output_tokens > 0);
    AGR_CHECK(r.total_ms > 0.0);
    AGR_CHECK(!r.output.empty());
    
    std::cout << "  Output: " << r.output << "\n";
    std::cout << "  Time: " << r.total_ms << " ms\n";
    
    AGR_TEST_MAIN_END();
}
