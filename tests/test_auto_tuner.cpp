// tests/test_auto_tuner.cpp
#include "mini_test.h"
#include "scheduler/AutoTuner.h"
#include "scheduler/PerformanceHistory.h"

using namespace agr;

int main() {
    std::cout << "== test_auto_tuner ==\n";
    PerformanceHistory history;
    HardwareInfo hw;
    AutoTuner tuner(history, hw);

    ProfileKey key;
    key.hardware_id = "test-hw";
    key.backend = ComputeBackend::VULKAN;
    key.workload_type = "llm-inference";
    key.model_name = "test-model";

    // 1. Initial suggestion (gpu available)
    TuningConfig c1 = tuner.suggestConfiguration(key, true, 8);
    AGR_CHECK(c1.n_gpu_layers == 99);
    AGR_CHECK(c1.n_threads == 4);

    // Record stable results
    tuner.recordResult(key, c1, 100.0, true);
    tuner.recordResult(key, c1, 100.0, true);
    tuner.recordResult(key, c1, 100.0, true);

    // 2. Exploration triggers
    TuningConfig c2 = tuner.suggestConfiguration(key, true, 8);
    AGR_CHECK(c2.n_threads != 4); // changed due to exploration
    
    // 3. Rollback on worse performance
    tuner.recordResult(key, c2, 120.0, true); // Worse!
    TuningConfig c3 = tuner.suggestConfiguration(key, true, 8);
    AGR_CHECK(c3.n_threads == 4); // Rolled back

    // Stabilize again
    tuner.recordResult(key, c3, 100.0, true);
    tuner.recordResult(key, c3, 100.0, true);
    tuner.recordResult(key, c3, 100.0, true);

    // 4. Accept on better performance
    TuningConfig c4 = tuner.suggestConfiguration(key, true, 8);
    tuner.recordResult(key, c4, 80.0, true); // Better!
    TuningConfig c5 = tuner.suggestConfiguration(key, true, 8);
    AGR_CHECK(c5.n_threads == c4.n_threads); // Kept the new config

    // 5. Rollback on failure
    tuner.recordResult(key, c5, 80.0, true);
    tuner.recordResult(key, c5, 80.0, true);
    tuner.recordResult(key, c5, 80.0, true);
    TuningConfig c6 = tuner.suggestConfiguration(key, true, 8);
    tuner.recordResult(key, c6, 0.0, false); // Failed!
    TuningConfig c7 = tuner.suggestConfiguration(key, true, 8);
    AGR_CHECK(c7.n_threads == c5.n_threads); // Rolled back

    AGR_TEST_MAIN_END();
}
