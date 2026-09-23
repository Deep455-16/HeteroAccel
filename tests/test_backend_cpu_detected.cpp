// tests/test_backend_cpu_detected.cpp
#include "mini_test.h"
#include "gpu/VulkanBackend.h"
#include "backend/BackendManager.h"

int main() {
    std::cout << "== test_backend_cpu_detected ==\n";
    agr::VulkanBackend vk;
    agr::BackendManager mgr(vk);
    mgr.discover();

    AGR_CHECK(mgr.isBackendAvailable(agr::ComputeBackend::CPU));
    const agr::ComputeDevice* cpu = mgr.getDevice(agr::ComputeBackend::CPU);
    AGR_CHECK(cpu != nullptr);
    AGR_CHECK(cpu->memory_capacity > 0);
    AGR_CHECK(cpu->compute_units > 0);
    AGR_CHECK(!cpu->name.empty());

    std::cout << "  CPU: " << cpu->name << " (" << cpu->compute_units << " cores)\n";
    AGR_TEST_MAIN_END();
}
