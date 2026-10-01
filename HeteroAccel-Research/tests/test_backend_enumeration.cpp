// tests/test_backend_enumeration.cpp
#include "mini_test.h"
#include "gpu/VulkanBackend.h"
#include "backend/BackendManager.h"

int main() {
    std::cout << "== test_backend_enumeration ==\n";
    agr::VulkanBackend vk;
    agr::BackendManager mgr(vk);
    mgr.discover();

    // Must always enumerate exactly 3 backends
    AGR_CHECK(mgr.allDevices().size() == 3);
    // CPU is always available
    AGR_CHECK(mgr.isBackendAvailable(agr::ComputeBackend::CPU));
    // Every device returned by availableDevices() must actually be available
    for (const auto& d : mgr.availableDevices()) {
        AGR_CHECK(d.is_available);
    }
    std::cout << "  Available: " << mgr.availableDevices().size() << "/3 backends\n";
    AGR_TEST_MAIN_END();
}
