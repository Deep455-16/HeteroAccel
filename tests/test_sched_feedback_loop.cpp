#include "mini_test.h"
#include "scheduler/AdaptiveScheduler.h"
#include <thread>
#include <chrono>

int main() {
    std::cout << "== test_sched_feedback_loop ==\n";
    agr::VulkanBackend vk;
    agr::BackendManager backendMgr(vk);
    backendMgr.discover();
    agr::MemoryManager memMgr(vk);
    agr::AdaptiveScheduler scheduler(backendMgr, memMgr);

    // Get initial estimate for CPU
    double initial_ops = scheduler.history().getEstimatedOpsPerMs(agr::ComputeBackend::CPU);
    
    agr::Workload w;
    w.compute_ops_estimate = 1000000;
    w.cpu_execute = []() {
        // Sleep 10ms to simulate slow compute.
        // 1,000,000 ops in 10ms = 100,000 ops/ms.
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return true;
    };

    agr::TaskHandle handle = scheduler.scheduleToBackend(w, agr::ComputeBackend::CPU);
    scheduler.wait(handle);

    // Verify history updated
    double updated_ops = scheduler.history().getEstimatedOpsPerMs(agr::ComputeBackend::CPU);
    
    std::cout << "Initial Ops/ms: " << initial_ops << "\n";
    std::cout << "Updated Ops/ms: " << updated_ops << "\n";
    
    // Since alpha is 0.2, the estimate should have moved towards 100,000.
    AGR_CHECK(initial_ops != updated_ops);

    AGR_TEST_MAIN_END();
}
