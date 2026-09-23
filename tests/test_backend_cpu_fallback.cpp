// tests/test_backend_cpu_fallback.cpp
// CPU must always be available as guaranteed fallback.
#include "mini_test.h"
#include "gpu/VulkanBackend.h"
#include "backend/BackendManager.h"
#include "backend/DeviceSelector.h"

int main() {
    std::cout << "== test_backend_cpu_fallback ==\n";
    agr::VulkanBackend vk;
    agr::BackendManager mgr(vk);
    mgr.discover();
    agr::DeviceSelector selector(mgr);

    // Request more memory than any device could have -> fallback
    agr::WorkloadHint hint;
    hint.required_memory  = static_cast<size_t>(1024) * 1024 * 1024 * 1024; // 1 TB
    hint.allow_cpu_fallback = true;
    agr::ComputeDevice selected = selector.selectDevice(hint);

    // Must always return something valid (runtime never crashes)
    AGR_CHECK(selected.is_available);
    std::cout << "  Fallback selected: " << agr::toString(selected.backend)
              << " (" << selected.name << ")\n";

    AGR_TEST_MAIN_END();
}
