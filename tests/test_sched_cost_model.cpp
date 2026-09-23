#include "mini_test.h"
#include "scheduler/AdaptiveScheduler.h"
#include "scheduler/CostModel.h"

int main() {
    std::cout << "== test_sched_cost_model ==\n";
    agr::VulkanBackend vk;
    agr::BackendManager backendMgr(vk);
    backendMgr.discover();
    agr::PerformanceHistory history;
    agr::CostModel costModel(backendMgr, history);

    agr::Workload w;
    w.input_bytes = 1024 * 1024 * 100; // 100 MB
    w.output_bytes = 1024 * 1024 * 100; // 100 MB
    w.compute_ops_estimate = 1000000;
    w.cpu_execute = [](){ return true; };
    w.vulkan_execute = [](){ return true; };
    w.cuda_execute = [](){ return true; };

    agr::ExecutionPlan plan = costModel.evaluate(w);
    
    std::cout << "Cost Model Selected: " << agr::toString(plan.selected_backend) << "\n";
    std::cout << "Total Cost: " << plan.total_cost_ms << " ms\n";

    AGR_CHECK(plan.total_cost_ms < 999999.0);

    AGR_TEST_MAIN_END();
}
