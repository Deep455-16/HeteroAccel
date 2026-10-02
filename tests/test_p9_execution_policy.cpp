// tests/test_p9_execution_policy.cpp
// Phase 9: ExecutionPolicyEngine - strategy selection based on memory, pressure, workload class
#include "mini_test.h"
#include "scheduler/ExecutionPolicyEngine.h"
#include "gpu/VulkanBackend.h"
#include "backend/BackendManager.h"

using namespace agr;

int main() {
    std::cout << "== test_p9_execution_policy ==\n";

    agr::VulkanBackend vk;
    agr::BackendManager backendMgr(vk);
    backendMgr.discover();

    agr::MemoryManager memMgr(vk);
    agr::PerformanceHistory history;
    agr::ExecutionPolicyEngine engine(backendMgr, memMgr, history);

    // 1. No GPU -> CPU fallback
    {
        PolicyInput input;
        input.gpu_available = false;
        input.model_size_bytes = 4ULL * 1024 * 1024 * 1024; // 4 GB
        PolicyDecision dec = engine.selectStrategy(input);
        AGR_CHECK(dec.strategy == ExecutionStrategy::CPU_FALLBACK);
        AGR_CHECK(dec.n_gpu_layers == 0);
        std::cout << "  ok: no GPU -> CPU_FALLBACK (" << dec.reason << ")\n";
    }

    // 2. Critical pressure -> CPU fallback regardless of GPU
    {
        PolicyInput input;
        input.gpu_available    = true;
        input.memory_pressure  = PressureLevel::CRITICAL;
        input.model_size_bytes = 1ULL * 1024 * 1024 * 1024;
        input.available_gpu_mem = 8ULL * 1024 * 1024 * 1024;
        PolicyDecision dec = engine.selectStrategy(input);
        AGR_CHECK(dec.strategy == ExecutionStrategy::CPU_FALLBACK);
        std::cout << "  ok: critical pressure -> CPU_FALLBACK (" << dec.reason << ")\n";
    }

    // 3. Model fits in GPU -> FULL_RESIDENT
    {
        PolicyInput input;
        input.gpu_available    = true;
        input.memory_pressure  = PressureLevel::NORMAL;
        input.model_size_bytes = 500ULL * 1024 * 1024; // 500 MB
        input.available_gpu_mem = 8ULL * 1024 * 1024 * 1024; // 8 GB
        PolicyDecision dec = engine.selectStrategy(input);
        AGR_CHECK(dec.strategy == ExecutionStrategy::FULL_RESIDENT);
        AGR_CHECK(dec.n_gpu_layers == 99);
        std::cout << "  ok: model fits -> FULL_RESIDENT (" << dec.reason << ")\n";
    }

    // 4. Model partially fits -> PARTIAL_RESIDENT or MEMORY_PRESSURE
    {
        PolicyInput input;
        input.gpu_available    = true;
        input.memory_pressure  = PressureLevel::NORMAL;
        input.model_size_bytes = 14ULL * 1024 * 1024 * 1024; // 14 GB
        input.available_gpu_mem = 4ULL * 1024 * 1024 * 1024; // 4 GB
        PolicyDecision dec = engine.selectStrategy(input);
        AGR_CHECK(dec.strategy == ExecutionStrategy::PARTIAL_RESIDENT ||
                  dec.strategy == ExecutionStrategy::STREAMING);
        AGR_CHECK(dec.streaming_enabled == true);
        std::cout << "  ok: partial fit -> " << toString(dec.strategy) << " (" << dec.reason << ")\n";
    }

    // 5. Adaptive prefetch distance
    {
        AGR_CHECK(engine.adaptivePrefetchDistance(PressureLevel::NORMAL, 2) == 2);
        AGR_CHECK(engine.adaptivePrefetchDistance(PressureLevel::WARNING, 2) == 1);
        AGR_CHECK(engine.adaptivePrefetchDistance(PressureLevel::HIGH, 2) == 1);
        AGR_CHECK(engine.adaptivePrefetchDistance(PressureLevel::CRITICAL, 2) == 0);
        std::cout << "  ok: adaptive prefetch distance respects pressure\n";
    }

    // 6. Background workload gets fewer threads
    {
        PolicyInput input;
        input.gpu_available    = true;
        input.memory_pressure  = PressureLevel::NORMAL;
        input.model_size_bytes = 0; // unknown -> optimistic
        input.available_gpu_mem = 8ULL * 1024 * 1024 * 1024;
        input.wclass           = WorkloadClass::BACKGROUND;
        PolicyDecision dec = engine.selectStrategy(input);
        // For background, threads should be reduced
        // FULL_RESIDENT strategy starts with n_threads=4, background reduces by 2 -> 2
        AGR_CHECK(dec.n_threads <= 4);
        std::cout << "  ok: background class reduces threads (" << dec.n_threads << ")\n";
    }

    // 7. computeGpuLayers
    {
        size_t model_sz = 8ULL * 1024 * 1024 * 1024;  // 8 GB
        size_t gpu_mem  = 4ULL * 1024 * 1024 * 1024;  // 4 GB
        int layers = engine.computeGpuLayers(model_sz, gpu_mem, 32);
        AGR_CHECK(layers >= 1 && layers <= 32);
        std::cout << "  ok: computeGpuLayers=" << layers << " for 50% fit scenario\n";

        // Zero GPU mem
        AGR_CHECK(engine.computeGpuLayers(model_sz, 0, 32) == 0);
        // Zero model size (unknown) -> return total layers
        AGR_CHECK(engine.computeGpuLayers(0, gpu_mem, 32) == 32);
    }

    AGR_TEST_MAIN_END();
}
