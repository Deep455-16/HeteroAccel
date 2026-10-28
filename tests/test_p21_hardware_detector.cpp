// tests/test_p21_hardware_detector.cpp
// Phase 21: Hardware Detector Unit Tests

#include "mini_test.h"
#include "orchestration/HardwareDetector.h"
#include <iostream>

using namespace agr::orchestration;

int main() {
    std::cout << "[TEST] Running test_p21_hardware_detector...\n";

    auto& hw = HardwareDetector::instance();

    // 1. Detect runs without crashing
    hw.detect();
    AGR_CHECK(true);

    // 2. Available system RAM > 0
    size_t avail_ram = hw.available_system_ram();
    std::cout << "[INFO] Available system RAM: " << (avail_ram / (1024 * 1024)) << " MB\n";
    AGR_CHECK(avail_ram > 0);

    // 3. Device list is non-empty (at least CPU device present)
    const auto& devs = hw.devices();
    std::cout << "[INFO] Enumerated " << devs.size() << " device(s)\n";
    AGR_CHECK(!devs.empty());

    bool found_cpu = false;
    for (const auto& d : devs) {
        if (d.api_support & static_cast<uint32_t>(DeviceInfo::APISupport::CPU)) {
            found_cpu = true;
            std::cout << "[INFO] Found CPU device: " << d.name << "\n";
            AGR_CHECK(d.vendor != DeviceInfo::Vendor::Unknown);
        }
    }
    AGR_CHECK(found_cpu);

    // 4. Vulkan should be detected on this Intel Iris Xe system
    bool vulkan_ok = hw.has_vulkan();
    std::cout << "[INFO] has_vulkan(): " << (vulkan_ok ? "true" : "false") << "\n";
    AGR_CHECK(vulkan_ok);

    // 5. best_gpu() returns non-null when Vulkan/GPU is available
    const auto* best_gpu = hw.best_gpu();
    if (vulkan_ok) {
        AGR_CHECK(best_gpu != nullptr);
        if (best_gpu) {
            std::cout << "[INFO] best_gpu: " << best_gpu->name << " (VRAM: "
                      << (best_gpu->vram_bytes / (1024 * 1024)) << " MB)\n";
            AGR_CHECK(best_gpu->vulkan_ok);
        }
    }

    // 6. Idempotency: detect() safe to call again
    size_t count_before = devs.size();
    hw.detect();
    AGR_CHECK(hw.devices().size() == count_before);
    AGR_CHECK(hw.best_gpu() == best_gpu);

    // 7. Print summary without error
    hw.print_summary(std::cout);

    AGR_TEST_MAIN_END();
}
