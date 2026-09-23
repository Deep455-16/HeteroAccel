#include "mini_test.h"
#include "scheduler/AdaptiveScheduler.h"
#include <thread>
#include <chrono>

int main() {
    std::cout << "== test_sched_cpu_only ==\n";
    agr::VulkanBackend vk;
    agr::BackendManager backendMgr(vk);
    backendMgr.discover();
    agr::MemoryManager memMgr(vk);
    agr::AdaptiveScheduler scheduler(backendMgr, memMgr);

    agr::Workload w;
    w.input_bytes = 1024;
    w.output_bytes = 1024;
    w.compute_ops_estimate = 5000;
    
    // CPU-only setup
    w.cpu_execute = []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        return true;
    };
    // No Vulkan/CUDA execute provided, forces CPU only!

    agr::TaskHandle handle = scheduler.schedule(w);
    agr::TaskResult result = scheduler.wait(handle);

    AGR_CHECK(result.success);
    // Since only CPU callback provided, it MUST select CPU or fail.
    // Actually our status tracking doesn't expose the backend chosen in TaskResult yet, 
    // but if it succeeded, it ran on CPU.
    
    AGR_TEST_MAIN_END();
}
