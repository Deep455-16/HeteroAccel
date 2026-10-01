// test_llm_mode_distinction.cpp
//
// Verifies that the runtime accurately distinguishes and reports
// CPU vs Vulkan modes when using the same model.
// Skips if no model OR no Vulkan device.

#include "mini_test.h"
#include "llm/LlamaCppEngine.h"
#include "llm/LlmConfig.h"
#include "hardware/HardwareDetector.h"

#include <cstdlib>

int main() {
    std::cout << "== test_llm_mode_distinction ==\n";

    agr::HardwareInfo info = agr::HardwareDetector::detectAll();
    if (!info.vulkan.available) {
        std::cout << "  SKIP: Vulkan not available.\n";
        return 77;
    }

    const char* model_env = std::getenv("HETEROACCEL_MODEL_PATH");
    if (!model_env || std::string(model_env).empty()) {
        std::cout << "  SKIP: HETEROACCEL_MODEL_PATH not set.\n";
        return 77;
    }

    // 1. CPU Mode
    agr::LlmResult cpuResult;
    {
        agr::LlmConfig cpuCfg;
        cpuCfg.model_path = model_env;
        cpuCfg.cpu_only = true;
        cpuCfg.max_new_tokens = 2;

        agr::LlamaCppEngine engine;
        AGR_CHECK(engine.initialize(cpuCfg));
        cpuResult = engine.infer("Hello");
        AGR_CHECK(cpuResult.success);
    }

    // 2. Vulkan Mode
    agr::LlmResult gpuResult;
    {
        agr::LlmConfig gpuCfg;
        gpuCfg.model_path = model_env;
        gpuCfg.n_gpu_layers = 99;
        gpuCfg.max_new_tokens = 2;

        agr::LlamaCppEngine engine;
        AGR_CHECK(engine.initialize(gpuCfg));
        gpuResult = engine.infer("Hello");
        AGR_CHECK(gpuResult.success);
    }

    std::cout << "  CPU run:  backend=" << cpuResult.backend << ", gpu_layers=" << cpuResult.gpu_layers_actual << "\n";
    std::cout << "  GPU run:  backend=" << gpuResult.backend << ", gpu_layers=" << gpuResult.gpu_layers_actual << "\n";

    // 3. Distinction assertions
    AGR_CHECK(cpuResult.backend == "CPU");
    AGR_CHECK(cpuResult.gpu_layers_actual == 0);
    AGR_CHECK(cpuResult.vulkan_confirmed == false);

    AGR_CHECK(gpuResult.backend == "Vulkan");
    AGR_CHECK(gpuResult.gpu_layers_actual > 0);
    AGR_CHECK(gpuResult.vulkan_confirmed == true);

    AGR_TEST_MAIN_END();
}
