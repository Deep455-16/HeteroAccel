// tests/test_profiler_persistence.cpp
#include "mini_test.h"
#include "profiler/Profiler.h"
#include "scheduler/PerformanceHistory.h"

using namespace agr;

int main() {
    std::cout << "== test_profiler_persistence ==\n";
    PerformanceHistory history;
    HardwareInfo hw;
    hw.cpu.model_name = "test-cpu";
    Profiler profiler(history, hw);

    ProfileEvent e;
    e.backend = ComputeBackend::VULKAN;
    e.workload_type = "llm-inference";
    e.model_name = "test-model";
    e.execution_duration_ms = 100.0;
    e.success = true;

    profiler.recordEvent(e);
    AGR_CHECK(profiler.getEvents().size() == 1);

    std::string path = "test_profile.json";
    AGR_CHECK(profiler.saveProfile(path));

    profiler.reset();
    AGR_CHECK(profiler.getEvents().empty());

    // predictDurationMs should be negative after reset
    ProfileKey key;
    key.hardware_id = "test-cpu";
    key.backend = ComputeBackend::VULKAN;
    key.workload_type = "llm-inference";
    key.model_name = "test-model";
    AGR_CHECK(history.predictDurationMs(key) < 0.0);

    AGR_CHECK(profiler.loadProfile(path));
    
    // History should be restored
    double pred = history.predictDurationMs(key);
    AGR_CHECK(pred == 100.0);

    // Clean up
    std::remove(path.c_str());

    AGR_TEST_MAIN_END();
}
