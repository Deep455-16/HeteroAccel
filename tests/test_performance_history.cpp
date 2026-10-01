// tests/test_performance_history.cpp
#include "mini_test.h"
#include "scheduler/PerformanceHistory.h"

using namespace agr;

int main() {
    std::cout << "== test_performance_history ==\n";
    PerformanceHistory history;
    
    ProfileKey key;
    key.hardware_id = "test-hw";
    key.backend = ComputeBackend::VULKAN;
    key.workload_type = "llm-inference";
    key.model_name = "test-model";

    // 1. Cold start / No history
    AGR_CHECK(history.predictDurationMs(key) < 0.0);
    AGR_CHECK(history.getStats(key).confidence() == 0.0);

    // 2. First observation
    ProfileEvent e1;
    e1.execution_duration_ms = 100.0;
    e1.success = true;
    history.recordEvent(key, e1);

    AGR_CHECK(history.predictDurationMs(key) == 100.0);
    AGR_CHECK(history.getStats(key).confidence() == 0.2);
    AGR_CHECK(history.getStats(key).sample_count == 1);

    // 3. Subsequent predictions & EMA
    ProfileEvent e2;
    e2.execution_duration_ms = 150.0;
    e2.success = true;
    history.recordEvent(key, e2);

    double pred = history.predictDurationMs(key);
    AGR_CHECK(pred > 100.0 && pred < 150.0); // EMA moved up
    AGR_CHECK(history.getStats(key).confidence() == 0.4);

    // 4. Failure handling / Reliability
    ProfileEvent e3;
    e3.success = false;
    history.recordEvent(key, e3);
    AGR_CHECK(history.getStats(key).sample_count == 3);
    AGR_CHECK(history.getStats(key).failure_count == 1);
    AGR_CHECK(history.getStats(key).reliability() < 1.0);

    // 5. Regression detection: pump many stable samples to get tight variance
    for(int i=0; i<30; i++) {
        ProfileEvent ev;
        ev.execution_duration_ms = 100.0;
        ev.success = true;
        history.recordEvent(key, ev);
    }
    AGR_CHECK(!history.getStats(key).regression_detected);
    
    // Spike must be far beyond 2 stddev from the tight EMA
    // EMA converges to ~100ms, variance ~0, stddev ~0 → any large spike triggers it
    ProfileEvent spike;
    spike.execution_duration_ms = 900.0;
    spike.success = true;
    history.recordEvent(key, spike);
    AGR_CHECK(history.getStats(key).regression_detected);

    AGR_TEST_MAIN_END();
}
