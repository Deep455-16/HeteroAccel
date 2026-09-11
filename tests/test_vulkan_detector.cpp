#include "mini_test.h"
#include "hardware/VulkanDetector.h"

int main() {
    std::cout << "== test_vulkan_detector ==\n";
    agr::VulkanInfo vk = agr::VulkanDetector::detect();

    std::cout << "  available=" << (vk.available ? "true" : "false") << "\n";
    if (!vk.available) {
        std::cout << "  reason=" << vk.unavailable_reason << "\n";
        // If we report unavailable, we must say why -- never a silent false.
        AGR_CHECK(!vk.unavailable_reason.empty());
    } else {
        std::cout << "  device_count=" << vk.devices.size() << "\n";
        AGR_CHECK(!vk.devices.empty());
        bool anyCompute = false;
        for (const auto& d : vk.devices) {
            std::cout << "    - " << d.name << " (" << d.device_type << ")"
                      << " compute_queue=" << (d.has_compute_capable_queue ? "yes" : "no") << "\n";
            if (d.has_compute_capable_queue) anyCompute = true;
        }
        AGR_CHECK(anyCompute);
    }

    // This call must complete without throwing/crashing either way --
    // that is the Phase 1 contract this test primarily exists to prove.
    AGR_CHECK(true);

    AGR_TEST_MAIN_END();
}
