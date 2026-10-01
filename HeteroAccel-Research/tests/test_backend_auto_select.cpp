// tests/test_backend_auto_select.cpp
// Verifies automatic backend selection: no user configuration needed.
#include "mini_test.h"
#include "gpu/VulkanBackend.h"
#include "backend/BackendManager.h"
#include "backend/DeviceSelector.h"

int main() {
    std::cout << "== test_backend_auto_select ==\n";
    agr::VulkanBackend vk;
    agr::BackendManager mgr(vk);
    mgr.discover();
    agr::DeviceSelector selector(mgr);

    // GPU-preferring compute workload
    agr::WorkloadHint hint;
    hint.prefer_gpu       = true;
    hint.compute_intensive = true;
    agr::ComputeDevice selected = selector.selectDevice(hint);

    AGR_CHECK(selected.is_available);
    std::cout << "  Auto-selected: " << agr::toString(selected.backend)
              << " (" << selected.name << ") score=" << selected.compute_score << "\n";

    // On this machine: CUDA unavailable, Vulkan available -> Vulkan must win over CPU
    bool cudaAvail   = mgr.isBackendAvailable(agr::ComputeBackend::CUDA);
    bool vulkanAvail = mgr.isBackendAvailable(agr::ComputeBackend::VULKAN);
    if (!cudaAvail && vulkanAvail) {
        AGR_CHECK(selected.backend == agr::ComputeBackend::VULKAN);
        std::cout << "  Correctly selected Vulkan (CUDA unavailable)\n";
    }

    AGR_TEST_MAIN_END();
}
