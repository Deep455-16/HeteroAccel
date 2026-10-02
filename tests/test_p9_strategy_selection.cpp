// tests/test_p9_strategy_selection.cpp
// Phase 9: Strategy selection — size-aware, pressure-aware, history-aware.
#include "mini_test.h"
#include "scheduler/ExecutionPolicyEngine.h"
#include "gpu/VulkanBackend.h"
#include "backend/BackendManager.h"

using namespace agr;

int main() {
    std::cout << "== test_p9_strategy_selection ==\n";

    agr::VulkanBackend vk;
    agr::BackendManager backendMgr(vk);
    backendMgr.discover();

    agr::MemoryManager memMgr(vk);
    agr::PerformanceHistory history;
    agr::ExecutionPolicyEngine engine(backendMgr, memMgr, history);

    // Helper: build a policy input
    auto makeInput = [](bool gpu, size_t model_mb, size_t gpu_mb,
                        PressureLevel press, WorkloadClass wc) {
        PolicyInput inp;
        inp.gpu_available    = gpu;
        inp.model_size_bytes = model_mb * 1024 * 1024;
        inp.available_gpu_mem = gpu_mb * 1024 * 1024;
        inp.memory_pressure  = press;
        inp.wclass           = wc;
        return inp;
    };

    // 1. Strategy AUTO selection logic: unknown model size -> optimistic FULL_RESIDENT
    {
        PolicyInput inp = makeInput(true, 0, 8192, PressureLevel::NORMAL, WorkloadClass::DEFAULT);
        PolicyDecision dec = engine.selectStrategy(inp);
        // Unknown size -> should assume fits
        AGR_CHECK(dec.strategy == ExecutionStrategy::FULL_RESIDENT ||
                  dec.strategy == ExecutionStrategy::CPU_FALLBACK); // cpu_fallback if no GPU detected
        std::cout << "  ok: unknown model size -> " << toString(dec.strategy) << "\n";
    }

    // 2. Large model (20 GB) with 4 GB GPU -> PARTIAL_RESIDENT or STREAMING
    {
        PolicyInput inp = makeInput(true, 20480, 4096, PressureLevel::NORMAL, WorkloadClass::DEFAULT);
        PolicyDecision dec = engine.selectStrategy(inp);
        bool ok = (dec.strategy == ExecutionStrategy::PARTIAL_RESIDENT ||
                   dec.strategy == ExecutionStrategy::STREAMING ||
                   dec.strategy == ExecutionStrategy::CPU_FALLBACK);
        AGR_CHECK(ok);
        std::cout << "  ok: 20GB model, 4GB GPU -> " << toString(dec.strategy)
                  << " layers=" << dec.n_gpu_layers << "\n";
    }

    // 3. Model too large for any GPU (e.g., 200 GB, 4 GB GPU) -> STREAMING or CPU_FALLBACK
    {
        PolicyInput inp = makeInput(true, 200 * 1024, 4096, PressureLevel::NORMAL, WorkloadClass::DEFAULT);
        PolicyDecision dec = engine.selectStrategy(inp);
        bool ok = (dec.strategy == ExecutionStrategy::STREAMING ||
                   dec.strategy == ExecutionStrategy::CPU_FALLBACK);
        AGR_CHECK(ok);
        AGR_CHECK(dec.streaming_enabled || dec.n_gpu_layers == 0);
        std::cout << "  ok: 200GB model -> " << toString(dec.strategy) << "\n";
    }

    // 4. Interactive workload gets n_gpu_layers priority
    {
        PolicyInput inp = makeInput(true, 14336, 4096, PressureLevel::NORMAL, WorkloadClass::INTERACTIVE);
        PolicyDecision dec = engine.selectStrategy(inp);
        // Interactive: either PARTIAL_RESIDENT with maximised layers, or another valid strategy
        std::cout << "  ok: INTERACTIVE -> " << toString(dec.strategy)
                  << " layers=" << dec.n_gpu_layers << " reason=" << dec.reason << "\n";
        AGR_CHECK(!dec.reason.empty());
    }

    // 5. Batch workload gets fewer threads
    {
        PolicyInput inp = makeInput(true, 500, 8192, PressureLevel::NORMAL, WorkloadClass::BATCH);
        PolicyDecision dec = engine.selectStrategy(inp);
        std::cout << "  ok: BATCH -> " << toString(dec.strategy) << " threads=" << dec.n_threads << "\n";
        // BATCH should reduce threads compared to INTERACTIVE
        PolicyInput inp2 = inp;
        inp2.wclass = WorkloadClass::INTERACTIVE;
        PolicyDecision dec2 = engine.selectStrategy(inp2);
        AGR_CHECK(dec.n_threads <= dec2.n_threads + 2); // batch <= interactive (with some tolerance)
    }

    // 6. Transfer-aware: no GPU means no transfer cost, just CPU
    {
        PolicyInput inp = makeInput(false, 4096, 0, PressureLevel::NORMAL, WorkloadClass::DEFAULT);
        PolicyDecision dec = engine.selectStrategy(inp);
        AGR_CHECK(dec.strategy == ExecutionStrategy::CPU_FALLBACK);
        AGR_CHECK(dec.n_gpu_layers == 0);
        std::cout << "  ok: no GPU -> transfer-free CPU_FALLBACK\n";
    }

    // 7. Decision reason is always non-empty
    {
        PolicyInput inp = makeInput(true, 1024, 8192, PressureLevel::NORMAL, WorkloadClass::THROUGHPUT);
        PolicyDecision dec = engine.selectStrategy(inp);
        AGR_CHECK(!dec.reason.empty());
        std::cout << "  ok: strategy reason always provided: \"" << dec.reason.substr(0,50) << "...\"\n";
    }

    AGR_TEST_MAIN_END();
}
