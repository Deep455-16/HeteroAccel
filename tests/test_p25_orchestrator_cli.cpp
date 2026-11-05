// tests/test_p25_orchestrator_cli.cpp
// Phase 25: EngineOrchestrator backend selection — matches what the CLI `infer` command uses

#include "mini_test.h"
#include "orchestration/EngineOrchestrator.h"
#include "orchestration/OrchestratorInit.h"
#include "orchestration/HardwareDetector.h"
#include <iostream>
#include <string>

using namespace agr::orchestration;

int main() {
    std::cout << "--- Phase 25: EngineOrchestrator CLI Backend Selection ---\n";

    // Reset orchestrator so we start from a clean state
    EngineOrchestrator& orch = EngineOrchestrator::instance();
    orch.reset();

    // Use singleton hardware detector (private constructor)
    HardwareDetector& hw = HardwareDetector::instance();
    hw.detect();
    register_default_backends(orch, hw);

    // Should always have at least the CPU backend
    auto backends = orch.backends();
    AGR_CHECK(!backends.empty());
    std::cout << "  Registered backends: " << backends.size() << "\n";
    for (const auto& b : backends) {
        std::cout << "    - " << b.backend_id << "\n";
    }

    // === Test 1: auto-select for a small model (must resolve to something) ===
    EngineOrchestrator::SelectionRequest req;
    req.model_path         = "dummy.gguf";
    req.model_arch         = "qwen2";
    req.quant_type         = 0;              // F32 — all backends accept
    req.model_size_bytes   = 100 * 1024 * 1024;  // 100 MB
    req.n_ctx              = 512;
    req.allow_cpu_fallback = true;

    EngineOrchestrator::SelectionResult sel = orch.select_backend(req);
    AGR_CHECK(!sel.selected_backend.empty());
    std::cout << "  Auto-selected: " << sel.selected_backend << "\n";
    std::cout << "  Reason:        " << sel.reason << "\n";

    // === Test 2: forced backend override — must honour the request ===
    req.forced_backend = "hydroxl_cpu";
    EngineOrchestrator::SelectionResult forced = orch.select_backend(req);
    AGR_CHECK(forced.selected_backend == "hydroxl_cpu");
    std::cout << "  Forced cpu:    " << forced.selected_backend << "\n";
    req.forced_backend.clear();

    // === Test 3: performance history influences selection ===
    // Record a very high CPU score
    orch.record_performance("hydroxl_cpu", 999.9f, 0);
    EngineOrchestrator::SelectionResult hist = orch.select_backend(req);
    AGR_CHECK(!hist.selected_backend.empty());
    std::cout << "  After perf history: " << hist.selected_backend << "\n";
    if (hist.used_benchmark_history)
        std::cout << "  (used benchmark history)\n";

    // === Test 4: diagnostics does not crash ===
    orch.print_diagnostics(std::cout);

    std::cout << "  PASS: test_p25_orchestrator_cli\n";
    AGR_TEST_MAIN_END();
}
