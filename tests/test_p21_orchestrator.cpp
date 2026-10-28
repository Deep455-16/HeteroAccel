// tests/test_p21_orchestrator.cpp
// Phase 21: Engine Orchestrator Unit Tests

#include "mini_test.h"
#include "orchestration/EngineOrchestrator.h"
#include "orchestration/OrchestratorInit.h"
#include "orchestration/HardwareDetector.h"

#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

using namespace agr::orchestration;

// Mock engine that intentionally fails initialize()
class FailingMockEngine : public IInferenceEngine {
public:
    bool initialize(const std::string&, std::string& error) override {
        error = "FailingMockEngine intentional failure";
        return false;
    }
    InferenceResult run(const InferenceRequest&) override { return {}; }
    void shutdown() override {}
    std::string backend_id() const override { return "failing_mock"; }
    bool is_ready() const override { return false; }
};

// Mock CPU engine for fallback verification
class SuccessMockCpuEngine : public IInferenceEngine {
public:
    bool initialize(const std::string&, std::string&) override {
        ready_ = true;
        return true;
    }
    InferenceResult run(const InferenceRequest&) override {
        InferenceResult r;
        r.success = true;
        r.backend_used = "hydroxl_cpu";
        return r;
    }
    void shutdown() override { ready_ = false; }
    std::string backend_id() const override { return "hydroxl_cpu"; }
    bool is_ready() const override { return ready_; }
private:
    bool ready_ = false;
};

int main() {
    std::cout << "[TEST] Running test_p21_orchestrator...\n";

    auto& hw = HardwareDetector::instance();
    hw.detect();

    auto& orch = EngineOrchestrator::instance();
    orch.reset();
    register_default_backends(orch, hw);

    // 1. Test selection with CPU-only constraints (quant 12 = Q4_K only on CPU)
    {
        EngineOrchestrator::SelectionRequest req;
        req.model_arch = "qwen2";
        req.quant_type = 12; // Q4_K
        req.model_size_bytes = 100 * 1024 * 1024;
        req.allow_cpu_fallback = true;

        auto res = orch.select_backend(req);
        std::cout << "[INFO] CPU-only quant selection: " << res.selected_backend
                  << " (reason: " << res.reason << ")\n";
        AGR_CHECK(res.selected_backend == "hydroxl_cpu");
        AGR_CHECK(!res.is_fallback);
    }

    // 2. Test selection with Vulkan available + Vulkan-compatible quant (quant 8 = Q8_0)
    if (hw.has_vulkan()) {
        EngineOrchestrator::SelectionRequest req;
        req.model_arch = "qwen2";
        req.quant_type = 8; // Q8_0 supported by Vulkan and CPU, Vulkan throughput=1.5
        req.model_size_bytes = 100 * 1024 * 1024;
        req.allow_cpu_fallback = true;

        auto res = orch.select_backend(req);
        std::cout << "[INFO] Vulkan-compatible quant selection: " << res.selected_backend
                  << " (reason: " << res.reason << ")\n";
        AGR_CHECK(res.selected_backend == "hydroxl_vulkan");
    }

    // 3. Test forced_backend override
    {
        EngineOrchestrator::SelectionRequest req;
        req.model_arch = "qwen2";
        req.quant_type = 8;
        req.forced_backend = "hydroxl_cpu";

        auto res = orch.select_backend(req);
        std::cout << "[INFO] Forced backend result: " << res.selected_backend << "\n";
        AGR_CHECK(res.selected_backend == "hydroxl_cpu");
    }

    // 4. Test rejection when backend doesn't support quant type
    {
        EngineOrchestrator::SelectionRequest req;
        req.model_arch = "qwen2";
        req.quant_type = 9999; // unknown quant type
        req.allow_cpu_fallback = false;

        auto res = orch.select_backend(req);
        std::cout << "[INFO] Unknown quant selection result: '" << res.selected_backend << "'\n";
        AGR_CHECK(res.selected_backend.empty());
        AGR_CHECK(!res.rejected_backends.empty());
    }

    // 5. Test rejection when insufficient RAM
    {
        EngineOrchestrator::SelectionRequest req;
        req.model_arch = "qwen2";
        req.quant_type = 12;
        req.model_size_bytes = 2000ULL * 1024ULL * 1024ULL * 1024ULL; // 2000 GB
        req.allow_cpu_fallback = false;

        auto res = orch.select_backend(req);
        std::cout << "[INFO] Excessive RAM request result: '" << res.selected_backend << "'\n";
        AGR_CHECK(res.selected_backend.empty());
        AGR_CHECK(!res.rejected_backends.empty());
    }

    // 6. Test fallback: if best backend fails init, falls back to CPU
    {
        BackendCapabilities fail_cap;
        fail_cap.backend_id = "failing_gpu";
        fail_cap.supported_quant_types = {0, 1, 2, 12};
        fail_cap.supported_arch = {"qwen2"};
        fail_cap.requires_cuda = false;
        fail_cap.requires_vulkan = false;
        fail_cap.relative_throughput = 20.0f; // top scorer
        fail_cap.priority = 50;
        orch.register_backend(fail_cap);

        orch.register_engine_factory("failing_gpu", []() {
            return std::make_shared<FailingMockEngine>();
        });
        orch.register_engine_factory("hydroxl_cpu", []() {
            return std::make_shared<SuccessMockCpuEngine>();
        });

        EngineOrchestrator::SelectionRequest req;
        req.model_arch = "qwen2";
        req.quant_type = 12;
        req.model_size_bytes = 10 * 1024 * 1024;
        req.allow_cpu_fallback = true;

        auto sel = orch.select_backend(req);
        AGR_CHECK(sel.selected_backend == "failing_gpu");

        std::string err;
        auto engine = orch.create_engine(sel, req, err);
        AGR_CHECK(engine != nullptr);
        if (engine) {
            std::cout << "[INFO] Created engine ID after fallback: " << engine->backend_id() << "\n";
            AGR_CHECK(engine->backend_id() == "hydroxl_cpu");
            AGR_CHECK(engine->is_ready());
        }
    }

    // 7. Test record_performance updates future selections
    {
        // Re-register clean default factories
        orch.reset();
        register_default_backends(orch, hw);

        // Before record_performance: quant 8 -> hydroxl_vulkan (if vulkan available)
        // Record very high throughput for hydroxl_cpu
        orch.record_performance("hydroxl_cpu", 999.0f, 256 * 1024 * 1024);

        EngineOrchestrator::SelectionRequest req;
        req.model_arch = "qwen2";
        req.quant_type = 8;
        req.model_size_bytes = 100 * 1024 * 1024;
        req.allow_cpu_fallback = true;

        auto res = orch.select_backend(req);
        std::cout << "[INFO] Post-benchmark-recording selection: " << res.selected_backend
                  << " (used_history: " << res.used_benchmark_history << ")\n";
        AGR_CHECK(res.selected_backend == "hydroxl_cpu");
        AGR_CHECK(res.used_benchmark_history);
    }

    // 8. Test concurrent selection requests (10 threads)
    {
        std::vector<std::thread> threads;
        std::atomic<int> success_count{0};

        for (int t = 0; t < 10; ++t) {
            threads.emplace_back([&orch, &success_count, t]() {
                for (int iter = 0; iter < 50; ++iter) {
                    EngineOrchestrator::SelectionRequest req;
                    req.model_arch = (iter % 2 == 0) ? "qwen2" : "llama";
                    req.quant_type = (iter % 3 == 0) ? 12 : 8;
                    req.model_size_bytes = (iter + 1) * 10 * 1024 * 1024;
                    req.prefer_latency = (iter % 2 == 1);
                    req.allow_cpu_fallback = true;

                    auto r = orch.select_backend(req);
                    if (!r.selected_backend.empty()) {
                        success_count.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            });
        }

        for (auto& th : threads) {
            th.join();
        }

        std::cout << "[INFO] Concurrent selections completed: " << success_count.load() << "/500\n";
        AGR_CHECK(success_count.load() == 500);
    }

    // 9. Diagnostics output non-empty
    {
        std::ostringstream oss;
        orch.print_diagnostics(oss);
        std::string diag = oss.str();
        AGR_CHECK(!diag.empty());
        AGR_CHECK(diag.find("Registered Backends") != std::string::npos);
        AGR_CHECK(diag.find("Observed Performance History") != std::string::npos);
        std::cout << "[INFO] Diagnostics output verification passed.\n";
    }

    AGR_TEST_MAIN_END();
}
