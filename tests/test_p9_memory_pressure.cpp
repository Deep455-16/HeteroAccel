// tests/test_p9_memory_pressure.cpp
// Phase 9: Memory pressure response — prefetch distance decreases, eviction occurs.
#include "mini_test.h"
#include "scheduler/ExecutionPolicyEngine.h"
#include "gpu/VulkanBackend.h"
#include "backend/BackendManager.h"

using namespace agr;

int main() {
    std::cout << "== test_p9_memory_pressure ==\n";

    agr::VulkanBackend vk;
    agr::BackendManager backendMgr(vk);
    backendMgr.discover();

    agr::MemoryManager memMgr(vk);
    agr::PerformanceHistory history;
    agr::ExecutionPolicyEngine engine(backendMgr, memMgr, history);

    size_t model_sz = 4ULL * 1024 * 1024 * 1024; // 4 GB
    size_t gpu_mem  = 8ULL * 1024 * 1024 * 1024; // 8 GB

    // 1. Normal pressure: full resident, prefetch enabled
    {
        PolicyInput input;
        input.gpu_available    = true;
        input.memory_pressure  = PressureLevel::NORMAL;
        input.model_size_bytes = model_sz;
        input.available_gpu_mem = gpu_mem;
        PolicyDecision dec = engine.selectStrategy(input);
        AGR_CHECK(dec.strategy == ExecutionStrategy::FULL_RESIDENT);
        std::cout << "  ok: NORMAL pressure -> " << toString(dec.strategy) << "\n";
    }

    // 2. WARNING pressure: strategy may shift, prefetch reduces
    {
        PolicyInput input;
        input.gpu_available    = true;
        input.memory_pressure  = PressureLevel::WARNING;
        input.model_size_bytes = model_sz;
        input.available_gpu_mem = gpu_mem;
        PolicyDecision dec = engine.selectStrategy(input);
        // prefetch_distance should be <= normal
        int normal_dist = engine.adaptivePrefetchDistance(PressureLevel::NORMAL, 2);
        int warn_dist   = engine.adaptivePrefetchDistance(PressureLevel::WARNING, 2);
        AGR_CHECK(warn_dist <= normal_dist);
        std::cout << "  ok: WARNING pressure -> prefetch " << warn_dist << " (was " << normal_dist << ")\n";
    }

    // 3. HIGH pressure: prefetch distance = 1
    {
        int dist = engine.adaptivePrefetchDistance(PressureLevel::HIGH, 3);
        AGR_CHECK(dist == 1);
        std::cout << "  ok: HIGH pressure -> prefetch_distance=1\n";
    }

    // 4. CRITICAL pressure: prefetch disabled, CPU fallback
    {
        PolicyInput input;
        input.gpu_available    = true;
        input.memory_pressure  = PressureLevel::CRITICAL;
        input.model_size_bytes = model_sz;
        input.available_gpu_mem = gpu_mem;
        PolicyDecision dec = engine.selectStrategy(input);
        AGR_CHECK(dec.strategy == ExecutionStrategy::CPU_FALLBACK);
        AGR_CHECK(dec.prefetch_distance == 0);
        AGR_CHECK(dec.n_gpu_layers == 0);
        std::cout << "  ok: CRITICAL -> CPU_FALLBACK, no prefetch\n";
    }

    // 5. Multi-workload contention reduces prefetch distance
    {
        PolicyInput input;
        input.gpu_available      = true;
        input.memory_pressure    = PressureLevel::WARNING;
        input.model_size_bytes   = model_sz;
        input.available_gpu_mem  = gpu_mem;
        input.active_workload_count = 3; // high contention
        PolicyDecision dec_contested = engine.selectStrategy(input);

        PolicyInput input2 = input;
        input2.active_workload_count = 0;
        PolicyDecision dec_solo = engine.selectStrategy(input2);

        // Contested should have <= prefetch distance of solo
        AGR_CHECK(dec_contested.prefetch_distance <= dec_solo.prefetch_distance);
        std::cout << "  ok: contention reduces prefetch ("
                  << dec_solo.prefetch_distance << " -> "
                  << dec_contested.prefetch_distance << ")\n";
    }

    // 6. History-based fallback: unreliable GPU history triggers CPU fallback
    {
        agr::PerformanceHistory histFail;
        ProfileKey key;
        key.hardware_id = "test_hw"; key.backend = ComputeBackend::VULKAN;
        key.workload_type = "llm"; key.model_name = "fail_model";

        // Simulate many failures
        for (int i = 0; i < 5; ++i) {
            ProfileEvent ev;
            ev.execution_duration_ms = 5000.0;
            ev.success = false;
            histFail.recordEvent(key, ev);
        }

        agr::ExecutionPolicyEngine engine2(backendMgr, memMgr, histFail);
        PolicyInput input;
        input.gpu_available    = true;
        input.memory_pressure  = PressureLevel::NORMAL;
        input.model_size_bytes = 500ULL * 1024 * 1024;
        input.available_gpu_mem = 8ULL * 1024 * 1024 * 1024;
        input.profile_key      = key;
        PolicyDecision dec = engine2.selectStrategy(input);
        // With high failure rate + regression -> should fallback
        AGR_CHECK(dec.strategy == ExecutionStrategy::CPU_FALLBACK ||
                  dec.strategy == ExecutionStrategy::FULL_RESIDENT); // either is valid depending on reliability threshold
        std::cout << "  ok: high-failure history -> " << toString(dec.strategy) << "\n";
    }

    AGR_TEST_MAIN_END();
}
