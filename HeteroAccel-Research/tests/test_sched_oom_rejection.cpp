#include "mini_test.h"
#include "scheduler/CostModel.h"
#include "backend/BackendManager.h"
#include "gpu/VulkanBackend.h"

int main() {
    std::cout << "== test_sched_oom_rejection ==\n";
    agr::VulkanBackend vk;
    agr::BackendManager backendMgr(vk);
    backendMgr.discover();
    agr::PerformanceHistory history;
    agr::CostModel costModel(backendMgr, history);

    agr::Workload w;
    // Ask for 100 Terabytes
    w.input_bytes = 100ULL * 1024 * 1024 * 1024 * 1024;
    w.output_bytes = 1024;
    w.cpu_execute = [](){ return true; };
    w.vulkan_execute = [](){ return true; };
    w.cuda_execute = [](){ return true; };

    agr::ExecutionPlan plan = costModel.evaluate(w);
    
    // It should fallback to CPU and assign a massive penalty since no device has 100TB
    AGR_CHECK(plan.selected_backend == agr::ComputeBackend::CPU);
    AGR_CHECK(plan.total_cost_ms >= 999999.0);

    AGR_TEST_MAIN_END();
}
