// tests/test_backend_vulkan_detected.cpp
#include "mini_test.h"
#include "gpu/VulkanBackend.h"
#include "backend/BackendManager.h"

int main() {
    std::cout << "== test_backend_vulkan_detected ==\n";
    agr::VulkanBackend vk;
    agr::BackendManager mgr(vk);
    mgr.discover();

    const agr::ComputeDevice* vd = mgr.getDevice(agr::ComputeBackend::VULKAN);
    AGR_CHECK(vd != nullptr); // always present even if unavailable
    if (!vd->is_available) {
        std::cout << "  SKIP: Vulkan unavailable: " << vd->unavailable_reason << "\n";
        return 77;
    }
    AGR_CHECK(!vd->name.empty());
    AGR_CHECK(vd->memory_capacity > 0);
    std::cout << "  Vulkan: " << vd->name << " (" << (vd->memory_capacity/1024/1024/1024) << " GB)\n";
    AGR_TEST_MAIN_END();
}
