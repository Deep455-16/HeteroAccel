// src/scheduler/AutoTuner.h
// Phase 8: AutoTuning and Configuration selection
#pragma once

#include "profiler/IProfiler.h"
#include "scheduler/PerformanceHistory.h"
#include "hardware/HardwareDetector.h"
#include <mutex>
#include <unordered_map>

namespace agr {

struct TuningConfig {
    int n_gpu_layers = 99;
    int n_threads = 4;
};

class AutoTuner {
public:
    AutoTuner(const PerformanceHistory& history, const HardwareInfo& hw);

    /// Generate a candidate configuration for execution.
    TuningConfig suggestConfiguration(const ProfileKey& key, bool gpu_available, int max_threads);

    /// Record the result of using a configuration.
    void recordResult(const ProfileKey& key, const TuningConfig& config, double duration_ms, bool success);

private:
    struct ConfigStats {
        double ema_duration_ms = 0.0;
        uint64_t sample_count = 0;
        uint64_t failures = 0;
    };

    struct WorkloadState {
        TuningConfig current_best;
        TuningConfig exploring;
        bool is_exploring = false;
        double current_best_duration = -1.0;
        uint64_t stable_samples = 0;
    };

    const PerformanceHistory& history_;
    const HardwareInfo& hw_;

    std::mutex mutex_;
    std::unordered_map<ProfileKey, WorkloadState> states_;

    // Config parameters
    int min_samples_before_tune_ = 3;
    double improvement_threshold_ = 0.05; // 5%
};

} // namespace agr
