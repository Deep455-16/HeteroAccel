// src/scheduler/PerformanceHistory.h
// Tracks historical execution metrics for adaptive feedback.
#pragma once

#include "backend/ComputeDevice.h"
#include "scheduler/SchedulerTypes.h"
#include <unordered_map>
#include <mutex>

namespace agr {

/// Maintains an Exponential Moving Average (EMA) of backend performance metrics.
class PerformanceHistory {
public:
    PerformanceHistory();

    /// Feed telemetry from a completed task to update estimates.
    void recordCompletion(ComputeBackend backend, const Workload& workload, const TaskResult& result);

    /// Get estimated compute ops per millisecond for a backend.
    double getEstimatedOpsPerMs(ComputeBackend backend) const;

    /// Get estimated transfer bandwidth (bytes per millisecond).
    double getEstimatedBandwidthBytesPerMs(ComputeBackend backend) const;

private:
    struct BackendMetrics {
        double ema_ops_per_ms        = 1000.0;     // Fallback default
        double ema_bandwidth_per_ms  = 1024.0 * 1024.0; // Fallback ~1GB/s
        uint64_t sample_count        = 0;
    };

    mutable std::mutex mutex_;
    std::unordered_map<ComputeBackend, BackendMetrics> metrics_;
    
    // Smoothing factor for EWMA (0 < alpha <= 1). Higher = faster adaptation to new data.
    double alpha_ = 0.2; 
};

} // namespace agr
