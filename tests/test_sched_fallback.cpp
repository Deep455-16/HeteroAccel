#include "mini_test.h"
#include "scheduler/AdaptiveScheduler.h"

int main() {
    std::cout << "== test_sched_fallback ==\n";
    agr::VulkanBackend vk;
    agr::BackendManager backendMgr(vk);
    backendMgr.discover();
    agr::MemoryManager memMgr(vk);
    agr::AdaptiveScheduler scheduler(backendMgr, memMgr);
    
    agr::Workload w;
    w.vulkan_execute = []() {
        throw std::runtime_error("Simulated Vulkan crash");
        return false;
    };

    // Force schedule to Vulkan. It should fail safely without bringing down the process.
    agr::TaskHandle handle = scheduler.scheduleToBackend(w, agr::ComputeBackend::VULKAN);
    agr::TaskResult result = scheduler.wait(handle);

    AGR_CHECK(result.success == false);
    AGR_CHECK(result.error_message == "Simulated Vulkan crash");

    AGR_TEST_MAIN_END();
}
