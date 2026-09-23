#include "mini_test.h"
#include "scheduler/CostModel.h"
#include "backend/BackendManager.h"
#include "gpu/VulkanBackend.h"

int main() {
    std::cout << "== test_sched_residency_bias ==\n";
    agr::VulkanBackend vk;
    agr::BackendManager backendMgr(vk);
    backendMgr.discover();
    agr::PerformanceHistory history;
    agr::CostModel costModel(backendMgr, history);

    agr::Workload w;
    w.input_bytes = 100ULL * 1024 * 1024; // 100 MB
    w.cpu_execute = [](){ return true; };
    w.vulkan_execute = [](){ return true; };

    // Fake that data is already on GPU
    w.input_block.id       = 1; // must be non-zero for isValid() to return true
    w.input_block.location = agr::MemoryLocation::ACCELERATOR;
    w.input_block.size     = w.input_bytes;

    agr::ExecutionPlan plan = costModel.evaluate(w);
    
    // CPU should have an input transfer cost, GPU shouldn't.
    // Note: evaluateCandidate explicitly returns the plan per backend.
    
    const agr::ComputeDevice* cpu = backendMgr.getDevice(agr::ComputeBackend::CPU);
    const agr::ComputeDevice* gpu = backendMgr.getDevice(agr::ComputeBackend::VULKAN);
    
    if (cpu && gpu && gpu->is_available) {
        auto cpuPlan = costModel.evaluateCandidate(w, *cpu);
        auto gpuPlan = costModel.evaluateCandidate(w, *gpu);
        
        AGR_CHECK(cpuPlan.requires_input_transfer == true);
        AGR_CHECK(gpuPlan.requires_input_transfer == false);
        AGR_CHECK(cpuPlan.estimated_transfer_ms > 0.0);
        AGR_CHECK(gpuPlan.estimated_transfer_ms == 0.0);
    }

    AGR_TEST_MAIN_END();
}
