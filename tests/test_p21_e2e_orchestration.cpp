// tests/test_p21_e2e_orchestration.cpp
// Phase 21: End-to-End Orchestrated Inference with Real Model

#include "mini_test.h"
#include "orchestration/EngineOrchestrator.h"
#include "orchestration/OrchestratorInit.h"
#include "orchestration/HardwareDetector.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>

using namespace agr::orchestration;

static const char* DEFAULT_MODEL = "D:\\HeteroAccel\\Models\\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";

static std::string getModelPath() {
    const char* env = std::getenv("HETEROACCEL_MODEL_PATH");
    if (env && *env) return env;
    return DEFAULT_MODEL;
}

int main() {
    std::cout << "[TEST] Running test_p21_e2e_orchestration...\n";

    std::string model_path = getModelPath();
    if (!std::filesystem::exists(model_path)) {
        std::cout << "[SKIP] Model file not found: " << model_path << "\n";
        std::cout << "       Set HETEROACCEL_MODEL_PATH to run real-model e2e test.\n";
        return 77;
    }

    std::cout << "[INFO] Real Model: " << model_path << "\n";
    size_t model_bytes = std::filesystem::file_size(model_path);
    std::cout << "[INFO] Model size on disk: " << (model_bytes / (1024 * 1024)) << " MB\n";

    // 1. Detect hardware
    auto& hw = HardwareDetector::instance();
    hw.detect();
    std::cout << "[INFO] Hardware detected: Vulkan=" << (hw.has_vulkan() ? "YES" : "NO")
              << ", Available RAM=" << (hw.available_system_ram() / (1024 * 1024)) << " MB\n";

    // 2. Initialize orchestrator and register backends
    auto& orch = EngineOrchestrator::instance();
    orch.reset();
    register_default_backends(orch, hw);

    // 3. Select backend for real model (Qwen2, Q4_K_M, ~491 MB)
    EngineOrchestrator::SelectionRequest req;
    req.model_path = model_path;
    req.model_arch = "qwen2";
    req.quant_type = 12; // GGML_TYPE_Q4_K = 12
    req.model_size_bytes = model_bytes;
    req.n_ctx = 512;
    req.prefer_latency = false;
    req.allow_cpu_fallback = true;

    auto t_sel0 = std::chrono::steady_clock::now();
    auto sel1 = orch.select_backend(req);
    auto t_sel1 = std::chrono::steady_clock::now();
    double sel_us = std::chrono::duration<double, std::micro>(t_sel1 - t_sel0).count();

    std::cout << "[INFO] Orchestrator selected backend: " << sel1.selected_backend << "\n";
    std::cout << "[INFO] Reason: " << sel1.reason << "\n";
    std::cout << "[INFO] Selection latency: " << sel_us << " microseconds\n";

    AGR_CHECK(!sel1.selected_backend.empty());
    AGR_CHECK(sel1.selected_backend == "hydroxl_cpu"); // Q4_K is native to CPU

    // 4. Create and initialize engine
    std::string error_out;
    auto t_init0 = std::chrono::steady_clock::now();
    auto engine = orch.create_engine(sel1, req, error_out);
    auto t_init1 = std::chrono::steady_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t_init1 - t_init0).count();

    if (!engine) {
        std::cerr << "[FAIL] Failed to create engine: " << error_out << "\n";
    }
    AGR_CHECK(engine != nullptr);
    if (!engine) return 1;

    AGR_CHECK(engine->is_ready());
    std::cout << "[INFO] Engine initialized in " << init_ms << " ms\n";

    // 5. Run first inference pass
    InferenceRequest inf_req;
    inf_req.max_new_tokens = 8;
    inf_req.temperature = 0.0f; // greedy
    inf_req.top_p = 0.95f;

    auto res1 = engine->run(inf_req);
    std::cout << "[INFO] Pass 1 inference result:\n";
    std::cout << "       Success:    " << (res1.success ? "YES" : "NO") << "\n";
    std::cout << "       Prompt tok: " << res1.prompt_tokens << "\n";
    std::cout << "       Output tok: " << res1.output_tokens << "\n";
    std::cout << "       Tok/sec:    " << res1.tokens_per_second << "\n";
    std::cout << "       Backend:    " << res1.backend_used << "\n";
    std::cout << "       Output:     '" << res1.output_text << "'\n";

    AGR_CHECK(res1.success);
    AGR_CHECK(!res1.output_text.empty());
    AGR_CHECK(res1.output_tokens > 0);
    AGR_CHECK(res1.tokens_per_second > 0.0f);

    // 6. Record performance to orchestrator
    orch.record_performance(res1.backend_used, res1.tokens_per_second, 128 * 1024 * 1024);

    // 7. Second selection pass: verify benchmark history is utilized
    auto sel2 = orch.select_backend(req);
    std::cout << "[INFO] Pass 2 selection: " << sel2.selected_backend
              << " (used_history=" << (sel2.used_benchmark_history ? "true" : "false") << ")\n";
    AGR_CHECK(sel2.selected_backend == "hydroxl_cpu");
    AGR_CHECK(sel2.used_benchmark_history);

    // 8. Run second inference pass
    auto res2 = engine->run(inf_req);
    AGR_CHECK(res2.success);
    AGR_CHECK(res2.output_tokens > 0);
    std::cout << "[INFO] Pass 2 output: '" << res2.output_text << "'\n";
    AGR_CHECK(res2.output_text == res1.output_text); // greedy must be deterministic

    // 9. Diagnostics printout
    orch.print_diagnostics(std::cout);

    engine->shutdown();
    AGR_CHECK(!engine->is_ready());

    std::cout << "[SUCCESS] Phase 21 e2e orchestration passed cleanly.\n";
    AGR_TEST_MAIN_END();
}
