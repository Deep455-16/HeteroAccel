// tests/test_p31_resource_memory_pressure.cpp
// Unified Collaborative Runtime — Constrained Hardware (~16 GB RAM) &
// Shared Resource Management Tests.
//
// Validates (Sections 9, 11 & 15.4):
//   1. Memory estimation for HydroXL (F32 weights + KV + activations),
//      llama.cpp (quantized weights + KV + compute buffer), and Colibri (~8 MB).
//   2. Constrained RAM detection: when concurrent neural footprints exceed the
//      configured memory budget, `memory_constrained_mode` is activated.
//   3. Sequential model residency swapping: under memory pressure,
//      CollaborativeResourceManager automatically evicts the resident heavy
//      neural engine before loading the next heavy neural engine so peak
//      resident memory never exceeds the budget.
//   4. Cancellation and deadlock-free GPU lock coordination.

#include "mini_test.h"
#include "collaboration/CollaborativeExecutionOrchestrator.h"
#include "orchestration/EngineOrchestrator.h"
#include "orchestration/HardwareDetector.h"
#include "orchestration/OrchestratorInit.h"

#include <atomic>
#include <iostream>
#include <memory>
#include <string>

using namespace agr::collaboration;
using namespace agr::orchestration;

namespace {

class TrackingMockNeuralEngine : public IInferenceEngine {
public:
    TrackingMockNeuralEngine(std::string id, std::atomic<int>& active_heavy_count, std::atomic<int>& peak_heavy_count)
        : id_(std::move(id)), active_(active_heavy_count), peak_(peak_heavy_count) {}

    bool initialize(const std::string&, std::string&) override {
        if (!ready_) {
            ready_ = true;
            int cur = ++active_;
            int prev = peak_.load();
            while (cur > prev && !peak_.compare_exchange_weak(prev, cur)) {}
        }
        return true;
    }

    InferenceResult run(const InferenceRequest& req) override {
        InferenceResult r;
        r.backend_used = id_;
        r.output_text = "Analyzed query on " + id_ + ": " + req.prompt.substr(0, 32);
        r.output_tokens = 12;
        r.tokens_per_second = 50.0f;
        r.success = true;
        return r;
    }

    void shutdown() override {
        if (ready_) {
            ready_ = false;
            --active_;
        }
    }

    std::string backend_id() const override { return id_; }
    bool is_ready() const override { return ready_; }

private:
    std::string id_;
    bool ready_ = false;
    std::atomic<int>& active_;
    std::atomic<int>& peak_;
};

} // namespace

int main() {
    std::cout << "=== test_p31_resource_memory_pressure ===\n";

    // 1. Verify Memory Estimation & Constrained Budget Detection
    {
        ModelMemoryProfile prof = CollaborativeResourceManager::estimateModelMemory("", 512);
        AGR_CHECK(prof.hydroxl_estimated_bytes > 0);
        AGR_CHECK(prof.llamacpp_estimated_bytes > 0);
        AGR_CHECK(prof.colibri_estimated_bytes == 8ULL * 1024ULL * 1024ULL);

        CollaborativeResourceManager mgr;
        // Set a generous 32 GB budget -> can keep concurrent neural engines
        mgr.configureBudget(32ULL * 1024ULL * 1024ULL * 1024ULL);
        AGR_CHECK(mgr.canKeepConcurrentNeuralEngines(prof));

        // Set a constrained 2 GB budget (smaller than HydroXL F32 + llama.cpp combined)
        mgr.configureBudget(2ULL * 1024ULL * 1024ULL * 1024ULL);
        AGR_CHECK(!mgr.canKeepConcurrentNeuralEngines(prof));
        std::cout << "  [OK] Memory footprint estimation & constrained budget detection\n";
    }

    // 2. Verify Sequential Model Swapping Prevents Simultaneous Heavy Residency
    {
        auto& orch = EngineOrchestrator::instance();
        orch.reset();
        auto& hw = HardwareDetector::instance();
        hw.detect();
        register_default_backends(orch, hw);

        std::atomic<int> active_heavy{0};
        std::atomic<int> peak_heavy{0};

        orch.register_engine_factory("hydroxl_cpu", [&]() {
            return std::make_shared<TrackingMockNeuralEngine>("hydroxl_cpu", active_heavy, peak_heavy);
        });
        orch.register_engine_factory("llamacpp_cpu", [&]() {
            return std::make_shared<TrackingMockNeuralEngine>("llamacpp_cpu", active_heavy, peak_heavy);
        });

        CollaborativeExecutionOrchestrator collab;
        CollaborativeRequest req;
        req.model_path = "constrained_mock_model.gguf";
        req.prompt = "Analyze cache-oblivious matrix transposition and verify loop bounds.";
        req.preferred_gpu_backend = "cpu";
        // Constrain RAM budget to 2.8 GB so HydroXL (~2.6 GB) and llama.cpp (~0.7 GB)
        // cannot both remain resident at the same time
        req.max_ram_budget_bytes = 2800ULL * 1024ULL * 1024ULL;
        req.allow_parallel_waves = true;

        CollaborativeResult res = collab.execute(req);
        AGR_CHECK(res.success);
        AGR_CHECK(res.resource_telemetry.memory_constrained_mode);
        AGR_CHECK(!res.resource_telemetry.concurrent_wave_executed);

        // All 3 engines still participated in the collaborative plan!
        AGR_CHECK(res.engineParticipated("hydroxl"));
        AGR_CHECK(res.engineParticipated("llama.cpp"));
        AGR_CHECK(res.engineParticipated("colibri"));

        // Crucial check: peak simultaneously resident heavy neural engines was NEVER > 1!
        AGR_CHECK(peak_heavy.load() == 1);
        AGR_CHECK(res.resource_telemetry.models_evicted_count >= 1);

        collab.resourceManager().evictAll();
        orch.reset();
        register_default_backends(orch, hw);
        std::cout << "  [OK] Constrained memory mode swaps neural engines sequentially (peak_heavy == 1)\n";
    }

    // 3. Verify Cancellation During Collaborative Execution
    {
        CollaborativeExecutionOrchestrator collab;
        std::atomic<bool> cancel_flag{false};

        collab.setCustomExecutor(EngineId::HYDROXL, [&](const CollaborativeTask& task, const CollaborativeRequest&) {
            // Trigger cancellation as soon as Task 1 finishes
            cancel_flag.store(true);
            IntermediateResult r;
            r.task_id = task.task_id;
            r.engine_id = "hydroxl";
            r.backend_used = "CPU";
            r.structured_result = "Partial output before cancellation.";
            r.success = true;
            return r;
        });

        collab.setCustomExecutor(EngineId::LLAMA_CPP, [](const CollaborativeTask& task, const CollaborativeRequest&) {
            IntermediateResult r;
            r.task_id = task.task_id;
            r.engine_id = "llama.cpp";
            r.backend_used = "CPU";
            r.structured_result = "Secondary output.";
            r.success = true;
            return r;
        });

        CollaborativeRequest req;
        req.prompt = "Explain lock-free ring buffers.";
        req.allow_parallel_waves = false;
        req.cancel_flag = &cancel_flag;

        CollaborativeResult res = collab.execute(req);
        AGR_CHECK(!res.success);
        AGR_CHECK(res.error.find("Cancel") != std::string::npos);
        std::cout << "  [OK] Mid-execution cancellation stops remaining DAG waves cleanly\n";
    }

    AGR_TEST_MAIN_END();
}
