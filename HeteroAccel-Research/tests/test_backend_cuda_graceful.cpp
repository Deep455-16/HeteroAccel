// tests/test_backend_cuda_graceful.cpp
// CUDA absence must NEVER fail the build or the test suite.
#include "mini_test.h"
#include "gpu/VulkanBackend.h"
#include "backend/BackendManager.h"

int main() {
    std::cout << "== test_backend_cuda_graceful ==\n";
    agr::VulkanBackend vk;
    agr::BackendManager mgr(vk);
    mgr.discover(); // must not throw or crash even with no NVIDIA hardware

    const agr::ComputeDevice* cd = mgr.getDevice(agr::ComputeBackend::CUDA);
    AGR_CHECK(cd != nullptr); // always present
    if (cd->is_available) {
        std::cout << "  CUDA available: " << cd->name << "\n";
        AGR_CHECK(!cd->name.empty());
    } else {
        std::cout << "  CUDA unavailable (expected on non-NVIDIA): "
                  << cd->unavailable_reason << "\n";
        AGR_CHECK(!cd->unavailable_reason.empty());
    }
    AGR_TEST_MAIN_END();
}
